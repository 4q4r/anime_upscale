import os
import sys
import subprocess
import argparse
import threading
import queue
import numpy as np
import torch
import torch.nn.functional as F
from spandrel import ModelLoader
from tqdm import tqdm

# Включаем аппаратные оптимизации ядер CUDA
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def get_video_info(input_path):
    """Получает разрешение, FPS и число кадров через ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
        "-of", "default=noprint_wrappers=1:nokey=1", input_path
    ]
    out = subprocess.check_output(cmd).decode("utf-8").strip().splitlines()
    w, h = int(out[0]), int(out[1])
    num, den = map(int, out[2].split("/"))
    fps = num / den
    nb_frames = int(out[3]) if len(out) > 3 and out[3].isdigit() else None
    return w, h, fps, nb_frames


class SingleGPUWorker:
    """Единый воркер для апскейла и RIFE-интерполяции на одной видеокарте."""
    def __init__(self, upscale_model_path: str, rife_dir: str = None, fp16: bool = True):
        self.device = torch.device("cuda:0")
        self.fp16 = fp16
        
        print("Загрузка модели апскейла...")
        loader = ModelLoader(device=self.device)
        self.upscale_model = loader.load_from_file(upscale_model_path).eval().to(self.device)
        if self.fp16:
            self.upscale_model = self.upscale_model.half()
        self.scale = self.upscale_model.scale

        self.rife_model = None
        if rife_dir and os.path.exists(rife_dir):
            print("Загрузка модели RIFE (интерполяция)...")
            sys.path.insert(0, rife_dir)
            from train_log.RIFE_HDv3 import Model
            self.rife_model = Model()
            self.rife_model.load_model(os.path.join(rife_dir, "train_log"), -1)
            self.rife_model.eval()
            self.rife_model.device()
            self.rife_model.net.to(self.device)

    @torch.inference_mode()
    def upscale(self, frame_np: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(frame_np).to(self.device, non_blocking=True)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        dtype = torch.float16 if self.fp16 else torch.float32
        tensor = tensor.to(dtype=dtype).div_(255.0)
        
        out = self.upscale_model(tensor)
        return out.clamp_(0.0, 1.0)

    @torch.inference_mode()
    def interpolate(self, t0: torch.Tensor, t1: torch.Tensor, scene_thresh: float = 0.18) -> np.ndarray:
        diff = torch.mean(torch.abs(t0 - t1)).item()
        if diff > scene_thresh:
            mid = t0
        else:
            _, _, h, w = t0.shape
            ph = ((h - 1) // 64 + 1) * 64
            pw = ((w - 1) // 64 + 1) * 64
            padding = (0, pw - w, 0, ph - h)
            i0_pad = F.pad(t0, padding)
            i1_pad = F.pad(t1, padding)
            mid = self.rife_model.inference(i0_pad, i1_pad, timestep=0.5)[:, :, :h, :w]

        return (
            mid.squeeze(0)
            .permute(1, 2, 0)
            .clamp_(0.0, 1.0)
            .mul_(255.0)
            .round_()
            .to(torch.uint8)
            .cpu()
            .numpy()
        )

    def tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        return (
            tensor.squeeze(0)
            .permute(1, 2, 0)
            .mul_(255.0)
            .round_()
            .to(torch.uint8)
            .cpu()
            .numpy()
        )


def process_video(args):
    w, h, fps, nb_frames = get_video_info(args.input)
    worker = SingleGPUWorker(args.upscale_model, rife_dir=args.rife_dir if args.interpolate else None, fp16=True)
    out_w, out_h = w * worker.scale, h * worker.scale

    target_fps = fps * 2 if args.interpolate else fps
    decoder, encoder = start_ffmpeg_pipes(args.input, args.output, w, h, out_w, out_h, target_fps)
    frame_bytes = w * h * 3

    pbar = tqdm(total=nb_frames, desc="Processing Anime (RTX 5080)", unit="frame")

    if not args.interpolate:
        # Режим только апскейл
        while True:
            raw = decoder.stdout.read(frame_bytes)
            if not raw:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
            up_tensor = worker.upscale(frame)
            out_np = worker.tensor_to_numpy(up_tensor)
            encoder.stdin.write(out_np.tobytes())
            pbar.update(1)
    else:
        # Режим апскейл + интерполяция RIFE
        tensor_queue = queue.Queue(maxsize=6)

        def producer():
            while True:
                raw = decoder.stdout.read(frame_bytes)
                if not raw:
                    tensor_queue.put(None)
                    break
                frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
                tensor_queue.put(worker.upscale(frame))

        t = threading.Thread(target=producer, daemon=True)
        t.start()

        prev_tensor = tensor_queue.get()
        if prev_tensor is not None:
            while True:
                curr_tensor = tensor_queue.get()
                if curr_tensor is None:
                    encoder.stdin.write(worker.tensor_to_numpy(prev_tensor).tobytes())
                    pbar.update(1)
                    break

                first_np = worker.tensor_to_numpy(prev_tensor)
                mid_np = worker.interpolate(prev_tensor, curr_tensor)

                encoder.stdin.write(first_np.tobytes())
                encoder.stdin.write(mid_np.tobytes())
                prev_tensor = curr_tensor
                pbar.update(1)

    close_pipes(decoder, encoder, pbar)


def start_ffmpeg_pipes(input_path, output_path, in_w, in_h, out_w, out_h, out_fps):
    decode_cmd = ["ffmpeg", "-loglevel", "error", "-i", input_path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    encode_cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{out_w}x{out_h}", "-r", str(out_fps), "-i", "-",
        "-i", input_path,
        "-map", "0:v:0", "-map", "1:a?", "-map", "1:s?",
        "-c:v", "hevc_nvenc", "-preset", "p7", "-tune", "hq",
        "-pix_fmt", "p010le", "-rc", "vbr", "-cq", "16", "-b:v", "0",
        "-c:a", "copy", "-c:s", "copy",
        output_path
    ]
    decoder = subprocess.Popen(decode_cmd, stdout=subprocess.PIPE, bufsize=in_w * in_h * 3 * 4)
    encoder = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE, bufsize=out_w * out_h * 3 * 4)
    return decoder, encoder


def close_pipes(decoder, encoder, pbar):
    pbar.close()
    decoder.stdout.close()
    encoder.stdin.close()
    decoder.wait()
    encoder.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-GPU Anime Upscaler & Interpolator")
    parser.add_argument("-i", "--input", required=True, help="Путь к исходному видео")
    parser.add_argument("-o", "--output", required=True, help="Путь к готовому видео")
    parser.add_argument("-m", "--upscale-model", default="models/upscale/model.pth", help="Путь к весам апскейла")
    parser.add_argument("--rife-dir", default="models/rife", help="Папка с Practical-RIFE")
    parser.add_argument("--interpolate", action="store_true", help="Включить интерполяцию кадров (удвоение FPS через RIFE)")
    args = parser.parse_args()

    process_video(args)
