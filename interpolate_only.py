import argparse
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Включаем аппаратные оптимизации CUDA для RTX 5080
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def get_video_info(input_path):
    """Получает параметры исходного видео через ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,nb_frames",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        input_path,
    ]
    out = subprocess.check_output(cmd).decode("utf-8").strip().splitlines()
    w, h = int(out[0]), int(out[1])
    num, den = map(int, out[2].split("/"))
    fps = num / den
    nb_frames = int(out[3]) if len(out) > 3 and out[3].isdigit() else 0
    return w, h, fps, nb_frames


def interpolate_video(args) -> None:
    w, h, orig_fps, nb_frames = get_video_info(args.input)
    target_fps = args.target_fps


    rife_abs_path = os.path.abspath(args.rife_dir)

    # Добавляем подпапку 'model' в системный путь Python, так как код RIFE теперь лежит там
    model_dir = os.path.join(rife_abs_path, "model")
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    if rife_abs_path not in sys.path:
        sys.path.insert(0, rife_abs_path)

    try:
        from RIFE_HDv3 import Model
    except ImportError:
        raise

    model = Model()
    weights_path = os.path.join(rife_abs_path, "train_log")

    if not os.path.exists(weights_path) or not os.listdir(weights_path):
        return

    model.load_model(weights_path, -1)
    model.eval()
    model.device()
    model.net.to(torch.device("cuda:0"))

    # Настройка конвейеров FFmpeg для чтения исходника и записи результата через NVENC
    decode_cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-i",
        args.input,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    encode_cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(target_fps),
        "-i",
        "-",
        "-i",
        args.input,
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-map",
        "1:s?",
        "-c:v",
        "hevc_nvenc",
        "-preset",
        "p7",
        "-tune",
        "hq",
        "-pix_fmt",
        "yuv420p10le",
        "-rc",
        "vbr",
        "-cq",
        "16",
        "-b:v",
        "0",
        "-c:a",
        "copy",
        "-c:s",
        "copy",
        args.output,
    ]

    decoder = subprocess.Popen(decode_cmd, stdout=subprocess.PIPE, bufsize=w * h * 3 * 4)
    encoder = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE, bufsize=w * h * 3 * 4)
    frame_bytes = w * h * 3

    pbar = tqdm(total=nb_frames if nb_frames > 0 else None, desc="Интерполяция RIFE", unit="кадр")

    raw0 = decoder.stdout.read(frame_bytes)
    if not raw0:
        return

    curr_frame_np = np.frombuffer(raw0, dtype=np.uint8).reshape((h, w, 3))
    curr_tensor = (
        torch.from_numpy(curr_frame_np)
        .to("cuda:0", non_blocking=True)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .div_(255.0)
    )

    encoder.stdin.write(curr_frame_np.tobytes())
    pbar.update(1)

    out_frame_idx = 1
    in_frame_idx = 1

    # Основной цикл потоковой интерполяции
    while True:
        raw1 = decoder.stdout.read(frame_bytes)
        if not raw1:
            break

        next_frame_np = np.frombuffer(raw1, dtype=np.uint8).reshape((h, w, 3))
        next_tensor = (
            torch.from_numpy(next_frame_np)
            .to("cuda:0", non_blocking=True)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .div_(255.0)
        )
        in_frame_idx += 1

        target_time_end = in_frame_idx / orig_fps
        target_time_start = (in_frame_idx - 1) / orig_fps

        while (out_frame_idx + 1) / target_fps <= target_time_end:
            t_out = (out_frame_idx + 1) / target_fps
            timestep = (t_out - target_time_start) / (target_time_end - target_time_start)
            timestep = float(np.clip(timestep, 0.0, 1.0))

            if timestep >= 0.99:
                break

            with torch.inference_mode():
                # Защита от резких склеек сцен
                diff = torch.mean(torch.abs(curr_tensor - next_tensor)).item()
                if diff > 0.18:
                    mid = curr_tensor if timestep < 0.5 else next_tensor
                else:
                    ph = ((h - 1) // 64 + 1) * 64
                    pw = ((w - 1) // 64 + 1) * 64
                    padding = (0, pw - w, 0, ph - h)
                    i0_pad = F.pad(curr_tensor, padding)
                    i1_pad = F.pad(next_tensor, padding)
                    mid_inf = model.inference(i0_pad, i1_pad, timestep=timestep)
                    mid = mid_inf[:, :, :h, :w]

                mid_np = (
                    mid.squeeze(0)
                    .permute(1, 2, 0)
                    .clamp_(0.0, 1.0)
                    .mul_(255.0)
                    .round_()
                    .to(torch.uint8)
                    .cpu()
                    .numpy()
                )

            encoder.stdin.write(mid_np.tobytes())
            out_frame_idx += 1

        encoder.stdin.write(next_frame_np.tobytes())
        out_frame_idx += 1
        pbar.update(1)
        curr_tensor = next_tensor

    pbar.close()
    decoder.stdout.close()
    encoder.stdin.close()
    decoder.wait()
    encoder.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Интерполяция кадров RIFE до целевого FPS")
    parser.add_argument("-i", "--input", required=True, help="Путь к исходному видео")
    parser.add_argument("-o", "--output", required=True, help="Путь к выходному файлу")
    parser.add_argument("--target-fps", type=float, default=60.0, help="Целевой FPS")
    parser.add_argument("--rife-dir", default="models/rife", help="Папка репозитория RIFE")
    args = parser.parse_args()

    interpolate_video(args)
