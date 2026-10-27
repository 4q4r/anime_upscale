"""
Per-stage benchmark: run each stage chain on a short sample and report fps.

Usage: uv run python scripts/benchmark.py [-i SAMPLE] [--outdir benchmarks]
Writes a dated markdown entry named after the detected GPU.
"""

import argparse
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from anime_upscale.config import HardwareOverrides, load_config, resolve_run
from anime_upscale.hardware import detect_hardware, format_gpu_summary
from anime_upscale.pipeline import run_pipeline
from anime_upscale.reporting import Reporter, resolve_mode
from anime_upscale.stages import StageContext
from anime_upscale.tools import resolve_tool

CHAINS: dict[str, tuple[str, ...]] = {
    "encode-only": ("encode",),
    "interp-only": ("interpolate", "encode"),
    "upscale-only": ("upscale", "encode"),
    "full-chain": ("descale", "deband", "upscale", "interpolate", "encode"),
}


def _count_frames(path: Path) -> int:
    """Real decoded frame count via ffprobe (ffprobe is a resolved tool, §12.3)."""
    ffprobe = resolve_tool("ffprobe")
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [
            ffprobe,
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(result.stdout.strip() or 0)


def main() -> int:
    """Run every stage chain over the sample; write a dated benchmarks/ entry."""
    parser = argparse.ArgumentParser(description="Per-stage fps benchmark (plan §7 step 15).")
    parser.add_argument("-i", "--input", default="sample.mkv", help="Имя файла в input/.")
    parser.add_argument("--outdir", default="benchmarks")
    parser.add_argument("--descale-height", type=int, default=360)
    args = parser.parse_args()

    root = Path.cwd()
    reporter = Reporter(resolve_mode(no_color=True, no_tty=True))
    config = load_config(root / "config.toml")
    input_path = root / "input" / args.input
    if not input_path.is_file():
        reporter.error(f"Нет файла {input_path}")
        return 1

    ffmpeg_path = resolve_tool("ffmpeg", config.tools.ffmpeg)
    hardware = detect_hardware(config.hardware, cli_num_streams=None, nn_stages_selected=True)
    gpu_label = format_gpu_summary(hardware).replace("×1", "").replace("×", "x").strip(", ")
    gpu_slug = gpu_label.split(",")[0].replace(" ", "").replace("NVIDIA", "").strip("-") or "GPU"
    rows: list[str] = []

    for chain_name, stages in CHAINS.items():
        kwargs = {}
        if "descale" in stages:
            kwargs["descale_height"] = args.descale_height
        run = resolve_run(config, stages=stages, **kwargs)
        nn = bool({"upscale", "interpolate"} & set(stages))
        hw = detect_hardware(
            HardwareOverrides(),
            cli_num_streams=None,
            nn_stages_selected=nn,
        )
        ctx = StageContext(
            run=run,
            hardware=hw,
            project_root=root,
            reporter=reporter,
            models_dir=root / config.models.dir,
            models_manifest=root / config.models.manifest,
        )
        output = root / "output" / f"_bench_{chain_name}.mkv"
        if output.exists():
            output.unlink()
        start = time.monotonic()
        run_pipeline(ctx, input_path, output, ffmpeg_path=ffmpeg_path)
        elapsed = time.monotonic() - start
        out_size = output.stat().st_size / (1024 * 1024)
        frames = _count_frames(output)
        fps_text = f"{frames / elapsed:.2f}" if frames and elapsed > 0 else "n/a"
        rows.append(
            f"| {chain_name} | {' → '.join(stages)} | {elapsed:.1f} | "
            f"{fps_text} ({frames} frames) | {out_size:.1f} |"
        )
        reporter.success(f"{chain_name}: {elapsed:.1f} s ({fps_text} fps)")
        output.unlink(missing_ok=True)

    stamp = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    out_file = Path(args.outdir) / f"{stamp}-{gpu_slug}.md"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    sample_lines = [
        "# Benchmark — anime_upscale",
        "",
        f"- Date: {stamp} (UTC)",
        f"- GPU: {format_gpu_summary(hardware)} (torch detect)",
        f"- CPU cores: {hardware.cpu_logical_cores}",
        f"- Sample: input/{args.input}",
        f"- Encoder: libx265 CRF {config.stages.encode.crf} preset {config.stages.encode.preset}",
        "",
        "Wall-clock per FULL chain pass over the sample (per-stage split is not",
        "measurable without instrumenting VS internals; chains isolate stages by",
        "construction — encode-only vs NN chains).",
        "",
        "| chain | stages | wall, s | fps (decoded frames) | output, MiB |",
        "| :-- | :-- | --: | --: | --: |",
        *rows,
        "",
        "NOTE: single-GPU machine; num_streams per the VRAM budget formula (§9.2).",
        "",
    ]
    out_file.write_text("\n".join(sample_lines), encoding="utf-8")
    reporter.success(f"Результаты: {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
