"""
Per-stage benchmark + hardware sweeps (plan C).

Default behavior (no --sweep): run each stage chain on a short sample and
report fps — unchanged from the original script.

Opt-in sweeps close the calibration loop (plan §3):
  --sweep streams  1..N points on the full chain: honest resident-set VRAM
                   anchor (torch peak reserved + nvidia-smi) per stream count.
  --sweep pools    scalar x265 pool sizes on the encode-only chain: pure CPU
                   A/B (fps, wall, encoder-side child RSS high-water mark).

Methodology per point (perf_analyzer/vs-mlrt canon, verified 2026-11-06):
fixed test clip, warm-up pass discarded, N measured repeats → median,
saturation = first gain < 3%. Result: machine-readable
benchmarks/<gpu-slug>/anchor.json consumed by hardware.py (config >
measured anchors > constants).

Without a real clip a documented synthetic fallback is generated via lavfi
(testsrc2 90 s 1080p) — plumbing validation only, never calibration-grade.
"""

import argparse
import dataclasses
import itertools
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from statistics import median as _median

from anime_upscale.config import Config, RunConfig, load_config, resolve_run
from anime_upscale.hardware import (
    ANCHOR_SCHEMA_VERSION,
    HardwareProfile,
    detect_hardware,
    format_gpu_summary,
    gpu_slug,
    load_anchor,
    query_driver_version,
    resolve_pools_from_anchor,
    resolve_streams_from_anchor,
    validate_anchor,
)
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

_STREAMS_CHAIN_DEFAULT = "full-chain"  # resident-set VRAM anchor needs the real NN set
_POOLS_CHAIN = "encode-only"  # pure CPU A/B — no GPU involvement by design

# Saturation: a point is worthless if the next one adds < 3% throughput
# (below run-to-run noise with 3 repeats; plan §3).
_SATURATION_GAIN = 1.03

_SYNTHETIC_CLIP_SECONDS = 90


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


def plan_stream_points(max_streams: int) -> list[int]:
    """Streams sweep grid: 1..max_streams (plan §3; covers clamp 4 + headroom)."""
    if max_streams < 1:
        msg = "--max-streams должен быть целым ≥ 1"
        raise ValueError(msg)
    return list(range(1, max_streams + 1))


def plan_pool_points(pool_list: str) -> list[int]:
    """Parse --pool-list into a sorted, deduplicated-free integer grid."""
    tokens = [token.strip() for token in pool_list.split(",") if token.strip()]
    if not tokens:
        msg = "--pool-list пуст"
        raise ValueError(msg)
    values: list[int] = []
    for token in tokens:
        if not token.isdigit():
            msg = f"--pool-list: нецелое значение «{token}»"
            raise ValueError(msg)
        values.append(int(token))
    if any(value < 1 for value in values):
        msg = "--pool-list: значения должны быть ≥ 1"
        raise ValueError(msg)
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        msg = f"--pool-list: дубли точек {duplicates}"
        raise ValueError(msg)
    return sorted(values)


def find_saturation(points: list[tuple[int, float]], gain: float = _SATURATION_GAIN) -> int | None:
    """Smallest point s where fps(s+1) < fps(s) × gain; None if still rising."""
    for (point_prev, fps_prev), (_, fps_next) in itertools.pairwise(points):
        if fps_next < fps_prev * gain:
            return point_prev
    return None


def _query_smi_used_mb() -> float | None:
    """Driver-side absolute GPU memory (nvidia-smi), best-effort → None."""
    try:
        out = subprocess.run(
            [  # noqa: S607 - diagnostic-only PATH lookup, fixed args
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    try:
        return float(out.stdout.splitlines()[0].strip())
    except (IndexError, ValueError):
        return None


def _reset_torch_peak() -> None:
    """reset_peak_memory_stats immediately before a measured pass (torch canon)."""
    import torch  # noqa: PLC0415 - lazy: pools sweep must not load the torch stack

    torch.cuda.reset_peak_memory_stats()


def _torch_reserved_peak_mb() -> float | None:
    """Absolute caching-allocator high-water mark after a measured pass (MiB)."""
    import torch  # noqa: PLC0415 - lazy

    return torch.cuda.max_memory_reserved() / (1024 * 1024)


def _children_rss_peak_mb() -> float:
    """
    Encoder-side child RSS high-water (ffmpeg/x265), stdlib getrusage.

    NOTE: RUSAGE_CHILDREN is a cumulative high-water mark across all reaped
    children — later points can over-report earlier peaks. Recorded as a
    secondary diagnostic; the A/B decision variable is fps (plan §2.3).
    """
    import resource  # noqa: PLC0415 - POSIX-only; pools sweep is Linux-target

    rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return rss / 1024 if sys.platform.startswith("linux") else rss / (1024 * 1024)


def _generate_synthetic_clip(input_path: Path, ffmpeg_path: str, reporter: Reporter) -> Path:
    """
    Documented synthetic fallback (lavfi testsrc2, 90 s 1080p).

    Fit for validating sweep mechanics end-to-end; NOT calibration-grade
    (noise cards and test patterns are not representative anime content).
    """
    input_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(  # noqa: S603 - fixed argv, no shell
        [
            ffmpeg_path,
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=duration={_SYNTHETIC_CLIP_SECONDS}:size=1920x1080:rate=24000/1001",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(input_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    reporter.note(
        "Сгенерирован синтетический клип (lavfi testsrc2, 90 с, 1080p). Он годится "
        "для проверки механики свипа, но НЕ для калибровочных якорей: положите "
        "реальный фрагмент аниме 60–120 с в input/ и перезапустите свип."
    )
    return input_path


def _parse_rate(rate_text: str) -> float | None:
    if "/" in rate_text:
        num, _, den = rate_text.partition("/")
        try:
            denominator = float(den)
        except ValueError:
            return None
        return float(num) / denominator if denominator else None
    try:
        return float(rate_text)
    except ValueError:
        return None


def _probe_clip(input_path: Path, ffmpeg_path: str | None = None) -> dict:  # noqa: ARG001 - symmetric with resolved tools
    """Clip description for the anchor (frames, fps fraction, height, seconds)."""
    ffprobe = resolve_tool("ffprobe")
    frames = _count_frames(input_path)
    probe = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate,height",
            "-of",
            "csv=p=0",
            str(input_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    line = probe.stdout.strip().splitlines()[0] if probe.stdout.strip() else ","
    rate_text, _, height_text = line.partition(",")
    rate = _parse_rate(rate_text.strip())
    seconds = frames / rate if rate else 0.0
    return {
        "file": input_path.name,
        "frames_in": frames,
        "src_fps": rate_text.strip() or "unknown",
        "height": int(height_text.strip() or 0),
        "seconds": seconds,
    }


def _run_pass(ctx: StageContext, input_path: Path, output: Path, ffmpeg_path: str) -> float:
    """One pipeline pass over the fixed clip; returns wall seconds."""
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    start = time.monotonic()
    run_pipeline(ctx, input_path, output, ffmpeg_path=ffmpeg_path)
    return time.monotonic() - start


def _make_context(
    run: RunConfig, hardware: HardwareProfile, root: Path, reporter: Reporter, config: Config
) -> StageContext:
    return StageContext(
        run=run,
        hardware=hardware,
        project_root=root,
        reporter=reporter,
        models_dir=root / config.models.dir,
        models_manifest=root / config.models.manifest,
    )


def _measured_repeats(  # noqa: PLR0913 - mirrors the plan §3 per-point protocol
    ctx: StageContext,
    input_path: Path,
    ffmpeg_path: str,
    scratch_dir: Path,
    *,
    warmup: int,
    repeats: int,
    tag: str,
    measure_vram: bool,
    reporter: Reporter,
) -> tuple[list[float], list[float], float | None, float | None, float | None]:
    """
    Warm-up (discarded) + measured passes for one sweep point.

    Returns (fps_repeats, wall_repeats, vram_peak_mb, torch_reserved_mb,
    smi_delta_mb).

    VRAM semantics (plan §3 + perf_analyzer canon): torch_reserved_mb is the
    ABSOLUTE allocator high-water of the pass (reset immediately before);
    smi_delta_mb is the driver-side delta across the whole point
    (before warm-up → after last measured pass); vram_peak_mb = max of the
    absolute torch peak and the absolute smi reading.
    """
    for index in range(warmup):
        _run_pass(ctx, input_path, scratch_dir / f"{tag}_w{index}.mkv", ffmpeg_path)
    smi_before = _query_smi_used_mb() if measure_vram else None
    fps_repeats: list[float] = []
    wall_repeats: list[float] = []
    torch_reserved: float | None = None
    smi_after: float | None = None
    for index in range(repeats):
        output_path = scratch_dir / f"{tag}_r{index}.mkv"
        if measure_vram:
            _reset_torch_peak()
        wall = _run_pass(ctx, input_path, output_path, ffmpeg_path)
        frames = _count_frames(output_path)
        output_path.unlink(missing_ok=True)
        wall_repeats.append(wall)
        fps_repeats.append(frames / wall if wall > 0 and frames else 0.0)
        if measure_vram:
            torch_reserved = _torch_reserved_peak_mb()
            smi_after = _query_smi_used_mb()
    vram_peak: float | None = None
    smi_delta: float | None = None
    if measure_vram:
        absolutes = [value for value in (torch_reserved, smi_after) if value is not None]
        vram_peak = max(absolutes) if absolutes else None
        smi_delta = (
            smi_after - smi_before if smi_after is not None and smi_before is not None else None
        )
        if vram_peak is None:
            reporter.note(
                "Не удалось измерить VRAM (ни torch reserved, ни nvidia-smi) — "
                "vram_peak_mb не будет записан для этой точки."
            )
        if smi_delta is None:
            reporter.note(
                "nvidia-smi недоступен — smi_delta_mb записан как null "
                "(замер VRAM остаётся по torch reserved)."
            )
    return fps_repeats, wall_repeats, vram_peak, torch_reserved, smi_delta


def run_streams_sweep(  # noqa: PLR0913, PLR0917 - driver-context bundle (args/config/root/reporter/hw/tool/clip)
    args: argparse.Namespace,
    config: Config,
    root: Path,
    reporter: Reporter,
    hardware: HardwareProfile,
    ffmpeg_path: str,
    input_path: Path,
) -> tuple[list[dict], int | None, str]:
    """Streams sweep on an NN chain: per-point resident-set VRAM + throughput."""
    points = plan_stream_points(args.max_streams)
    chain = args.chain or _STREAMS_CHAIN_DEFAULT
    if chain == _POOLS_CHAIN:
        msg = "Свип streams требует NN-цепочку (full-chain или interp-only), а не encode-only"
        raise ValueError(msg)
    stages = CHAINS[chain]
    kwargs: dict[str, object] = {}
    if "descale" in stages:
        kwargs["descale_height"] = args.descale_height
    run = resolve_run(config, stages=stages, **kwargs)
    rows: list[dict] = []
    for streams in points:
        profile_point = dataclasses.replace(hardware, num_streams_per_gpu=streams)
        ctx = _make_context(run, profile_point, root, reporter, config)
        fps_repeats, _, vram_peak, torch_reserved, smi_delta = _measured_repeats(
            ctx,
            input_path,
            ffmpeg_path,
            root / "output" / "_bench",
            warmup=args.warmup,
            repeats=args.repeats,
            tag=f"streams_s{streams}",
            measure_vram=True,
            reporter=reporter,
        )
        fps = _median(fps_repeats) if fps_repeats else 0.0
        row: dict = {
            "streams": streams,
            "fps": fps,
            "fps_repeats": fps_repeats,
            "vram_peak_mb": vram_peak,
            "torch_reserved_mb": torch_reserved,
            "smi_delta_mb": smi_delta,
        }
        rows.append(row)
        vram_text = f"{vram_peak:.0f}" if vram_peak is not None else "н/д"
        reporter.success(
            f"streams={streams}: {fps:.2f} fps (медиана {args.repeats}, VRAM peak {vram_text} МБ)"
        )
    saturation = find_saturation([(row["streams"], row["fps"]) for row in rows])
    if saturation is None:
        reporter.note(
            "Сатурация не достигнута в проверенном диапазоне — повторите с большим --max-streams."
        )
    return rows, saturation, chain


def run_pools_sweep(  # noqa: PLR0913, PLR0917 - driver-context bundle (args/config/root/reporter/hw/tool/clip)
    args: argparse.Namespace,
    config: Config,
    root: Path,
    reporter: Reporter,
    hardware: HardwareProfile,
    ffmpeg_path: str,
    input_path: Path,
) -> tuple[list[dict], int | None, str]:
    """Pools sweep on encode-only: scalar x265 pool A/B (fps, wall, RSS)."""
    points = plan_pool_points(args.pool_list)
    stages = CHAINS[_POOLS_CHAIN]
    run = resolve_run(config, stages=stages)
    rows: list[dict] = []
    for pools in points:
        profile_point = dataclasses.replace(hardware, encode_threads=pools)
        ctx = _make_context(run, profile_point, root, reporter, config)
        fps_repeats, wall_repeats, _, _, _ = _measured_repeats(
            ctx,
            input_path,
            ffmpeg_path,
            root / "output" / "_bench",
            warmup=args.warmup,
            repeats=args.repeats,
            tag=f"pools_p{pools}",
            measure_vram=False,
            reporter=reporter,
        )
        fps = _median(fps_repeats) if fps_repeats else 0.0
        wall = _median(wall_repeats) if wall_repeats else 0.0
        rows.append(
            {
                "pools": pools,
                "fps": fps,
                "fps_repeats": fps_repeats,
                "wall_s": wall,
                "rss_peak_mb": _children_rss_peak_mb(),
            }
        )
        reporter.success(
            f"pools={pools}: {fps:.2f} fps (медиана {args.repeats}, wall {wall:.1f} с)"
        )
    saturation = find_saturation([(row["pools"], row["fps"]) for row in rows])
    if saturation is None:
        reporter.note(
            "Сатурация pools не достигнута в проверенном диапазоне — расширите --pool-list."
        )
    return rows, saturation, _POOLS_CHAIN


def _stack_versions() -> tuple[str | None, str | None]:
    try:
        import torch  # noqa: PLC0415 - lazy
    except ImportError:
        return None, None
    return str(torch.__version__), getattr(torch.version, "cuda", None)


def _git_commit(root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607 - fixed argv, PATH lookup
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            cwd=root,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    return out.stdout.strip() or "unknown"


@dataclasses.dataclass(frozen=True)
class SweepOutcome:
    """Result of one sweep: recorded rows, detected saturation, chain used."""

    rows: list[dict]
    saturation: int | None
    chain: str


def build_anchor(  # noqa: PLR0913 - schema mirror; every field is a schema section
    *,
    hardware: HardwareProfile,
    config: Config,
    root: Path,
    clip: dict,
    streams: SweepOutcome | None,
    pools: SweepOutcome | None,
    notes: str,
) -> dict:
    """Machine-readable anchor exactly per plan §3 schema."""
    torch_version, cuda_version = _stack_versions()
    return {
        "schema_version": ANCHOR_SCHEMA_VERSION,
        "date": datetime.now(tz=UTC).strftime("%Y-%m-%d"),
        "gpu_name": hardware.gpus[0].name if hardware.gpus else "unknown",
        "gpu_vram_mb": hardware.gpus[0].vram_mb if hardware.gpus else 0,
        "driver": query_driver_version() or "unknown",
        "cuda": cuda_version,
        "torch": torch_version,
        "pipeline_commit": _git_commit(root),
        "run_config": {
            "preset": config.stages.encode.preset,
            "crf": config.stages.encode.crf,
            "interp_format": hardware.recommended_interp_format,
            "trt": hardware.recommended_trt,
            "chains": {
                "streams": streams.chain if streams else _STREAMS_CHAIN_DEFAULT,
                "pools": pools.chain if pools else _POOLS_CHAIN,
            },
        },
        "clip": clip,
        "streams_sweep": streams.rows if streams else [],
        "pools_sweep": pools.rows if pools else [],
        "saturation_streams": streams.saturation if streams else None,
        "saturation_pools": pools.saturation if pools else None,
        "notes": notes,
    }


def write_anchor(root: Path, anchor: dict) -> Path:
    """One anchor file per GPU, overwritten per run (plan §3)."""
    path = root / "benchmarks" / gpu_slug(anchor["gpu_name"]) / "anchor.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(anchor, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def check_anchor(path: Path, reporter: Reporter | None = None) -> int:
    """Validate an anchor + print what hardware.py would derive (GPU-less)."""
    reporter = reporter or Reporter(resolve_mode(no_color=True, no_tty=True))
    if not path.is_file():
        reporter.error(f"Якорь не найден: {path}")
        return 1
    anchor = load_anchor(path)
    if anchor is None:
        reporter.error(f"Якорь не читается как JSON-объект: {path}")
        return 1
    problems = validate_anchor(anchor)
    if problems:
        reporter.error(f"Якорь НЕВАЛИДЕН: {path}")
        for problem in problems:
            reporter.note(f"  - {problem}")
        return 1
    vram = int(anchor.get("gpu_vram_mb", 0) or 0)
    streams = resolve_streams_from_anchor(anchor, vram)
    cores = os.cpu_count() or 1
    pools = resolve_pools_from_anchor(anchor, cores=cores, active_encoders=1)
    reporter.success(f"Якорь валиден (schema_version {anchor.get('schema_version')}): {path}")
    reporter.note(f"  GPU: {anchor.get('gpu_name')} ({vram} МБ, driver {anchor.get('driver')})")
    reporter.note(f"  streams → hardware.py выберет: {streams}")
    reporter.note(f"  pools   → hardware.py выберет: {pools} (cores={cores}, active_encoders=1)")
    reporter.note(
        f"  сатурация: streams={anchor.get('saturation_streams')}, "
        f"pools={anchor.get('saturation_pools')}"
    )
    return 0


def print_sweep_plan(args: argparse.Namespace, reporter: Reporter | None = None) -> int:
    """--dry-run: print the sweep plan (points × passes, chains) and exit."""
    reporter = reporter or Reporter(resolve_mode(no_color=True, no_tty=True))
    reporter.note("План свипа (без выполнения):")
    total = 0
    if args.sweep is None or "streams" in args.sweep:
        chain = args.chain or _STREAMS_CHAIN_DEFAULT
        points = plan_stream_points(args.max_streams)
        passes = args.warmup + args.repeats
        total += len(points) * passes
        reporter.note(
            f"  streams: точки 1..{args.max_streams} = {points}, цепочка {chain}, "
            f"{passes} проходов на точку (warm-up {args.warmup} + повторы {args.repeats})"
        )
    if args.sweep is None or "pools" in args.sweep:
        points = plan_pool_points(args.pool_list)
        passes = args.warmup + args.repeats
        total += len(points) * passes
        reporter.note(
            f"  pools: точки {points} (chain {_POOLS_CHAIN}), "
            f"{passes} проходов на точку (warm-up {args.warmup} + повторы {args.repeats})"
        )
    reporter.note(f"Всего проходов пайплайна: {total}")
    reporter.note("Якорь будет записан в benchmarks/<gpu-slug>/anchor.json")
    return 0


def _run_sweeps(
    args: argparse.Namespace, config: Config, root: Path, reporter: Reporter, input_path: Path
) -> int:
    ffmpeg_path = resolve_tool("ffmpeg", config.tools.ffmpeg)
    notes: list[str] = []
    if not input_path.is_file():
        input_path = _generate_synthetic_clip(input_path, ffmpeg_path, reporter)
        notes.append(
            "Синтетический клип (lavfi testsrc2): механика свипа проверена, "
            "калибровочные якоря по нему строить нельзя."
        )
    clip = _probe_clip(input_path)
    sweeps = args.sweep or []
    streams_outcome: SweepOutcome | None = None
    pools_outcome: SweepOutcome | None = None
    streams_chain = (
        (args.chain or _STREAMS_CHAIN_DEFAULT) if "streams" in sweeps else _STREAMS_CHAIN_DEFAULT
    )
    pools_chain = _POOLS_CHAIN
    if args.chain is not None and "pools" in sweeps:
        reporter.note(
            "Свип pools всегда выполняется на encode-only (чистый CPU A/B); "
            "--chain применяется только к свипу streams."
        )

    needs_nn = "streams" in sweeps
    hardware = detect_hardware(
        config.hardware,
        cli_num_streams=None,
        nn_stages_selected=needs_nn,
    )

    if "streams" in sweeps:
        streams_rows_list, streams_saturation, streams_chain = run_streams_sweep(
            args, config, root, reporter, hardware, ffmpeg_path, input_path
        )
        streams_outcome = SweepOutcome(streams_rows_list, streams_saturation, streams_chain)
    if "pools" in sweeps:
        pools_rows_list, pools_saturation, pools_chain = run_pools_sweep(
            args, config, root, reporter, hardware, ffmpeg_path, input_path
        )
        pools_outcome = SweepOutcome(pools_rows_list, pools_saturation, pools_chain)

    anchor = build_anchor(
        hardware=hardware,
        config=config,
        root=root,
        clip=clip,
        streams=streams_outcome,
        pools=pools_outcome,
        notes=" ".join(notes),
    )
    problems = validate_anchor(anchor)
    if problems:
        for problem in problems:
            reporter.error(f"Якорь не прошел валидацию: {problem}")
        return 1
    anchor_path = write_anchor(root, anchor)
    reporter.success(f"Якорь записан: {anchor_path}")
    _write_sweep_markdown(args, anchor, reporter)
    return 0


def _write_sweep_markdown(args: argparse.Namespace, anchor: dict, reporter: Reporter) -> None:
    stamp = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    out_file = Path(args.outdir) / f"{stamp}-{gpu_slug(anchor['gpu_name'])}.md"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    gpu_text = f"{anchor['gpu_name']} ({anchor['gpu_vram_mb']} МБ)"
    lines = [
        "# Benchmark sweep — anime_upscale",
        "",
        f"- Date: {stamp} (UTC)",
        f"- GPU: {gpu_text}",
        f"- Clip: {anchor['clip']['file']} ({anchor['clip']['frames_in']} frames)",
        (
            "- Encoder: libx265 CRF "
            f"{anchor['run_config']['crf']} "
            f"preset {anchor['run_config']['preset']}"
        ),
        f"- Anchor: benchmarks/{gpu_slug(anchor['gpu_name'])}/anchor.json",
        "",
    ]
    if anchor["streams_sweep"]:
        streams_header = (
            "| streams | fps (median) | fps repeats | VRAM peak, MB "
            "| torch reserved, MB | smi delta, MB |"
        )
        lines += [
            f"## Sweep: streams (chain {anchor['run_config']['chains']['streams']})",
            "",
            streams_header,
            "| :-- | --: | :-- | --: | --: | --: |",
        ]
        for row in anchor["streams_sweep"]:
            repeats_text = [round(v, 2) for v in row["fps_repeats"]]
            lines.append(
                f"| {row['streams']} | {row['fps']:.2f} | {repeats_text} "
                f"| {row['vram_peak_mb']} | {row['torch_reserved_mb']} | {row['smi_delta_mb']} |"
            )
        lines += ["", f"Saturation: streams={anchor['saturation_streams']}", ""]
    if anchor["pools_sweep"]:
        lines += [
            f"## Sweep: pools (chain {anchor['run_config']['chains']['pools']})",
            "",
            "| pools | fps (median) | fps repeats | wall, s | RSS peak, MB |",
            "| :-- | --: | :-- | --: | --: |",
        ]
        for row in anchor["pools_sweep"]:
            repeats_text = [round(v, 2) for v in row["fps_repeats"]]
            lines.append(
                f"| {row['pools']} | {row['fps']:.2f} | {repeats_text} "
                f"| {row['wall_s']:.1f} | {row['rss_peak_mb']} |"
            )
        lines += ["", f"Saturation: pools={anchor['saturation_pools']}", ""]
    if anchor["notes"]:
        lines += [f"Notes: {anchor['notes']}", ""]
    out_file.write_text("\n".join(lines), encoding="utf-8")
    reporter.success(f"Отчет: {out_file}")


def main() -> int:
    """Run stage chains / hardware sweeps; write benchmarks/ artifacts."""
    parser = argparse.ArgumentParser(
        description="Per-stage fps benchmark + hardware sweeps (plan C)."
    )
    parser.add_argument("-i", "--input", default="sample.mkv", help="Имя файла в input/.")
    parser.add_argument("--outdir", default="benchmarks")
    parser.add_argument("--descale-height", type=int, default=360)
    parser.add_argument(
        "--sweep",
        action="append",
        choices=("streams", "pools"),
        default=None,
        help="Свип нагрузки: streams (1..N, full-chain) и/или pools (encode-only). "
        "Повторяемый флаг.",
    )
    parser.add_argument(
        "--max-streams", type=int, default=6, help="Верхняя точка свипа streams (по умолчанию 6)."
    )
    parser.add_argument(
        "--pool-list", default="8,16,24,32", help="Точки свипа pools через запятую."
    )
    parser.add_argument(
        "--chain",
        default=None,
        help="Цепочка свипа streams (по умолчанию full-chain; pools всегда encode-only).",
    )
    parser.add_argument("--warmup", type=int, default=1, help="Отбрасываемых прогревов на точку.")
    parser.add_argument(
        "--repeats", type=int, default=3, help="Измеряемых повторов на точку (медиана)."
    )
    parser.add_argument("--dry-run", action="store_true", help="Печатает план свипа и выходит.")
    parser.add_argument(
        "--check-anchor",
        default=None,
        metavar="PATH",
        help="Валидация якоря + что выведет hardware.py.",
    )
    args = parser.parse_args()

    if args.check_anchor:
        return check_anchor(Path(args.check_anchor))
    if args.warmup < 0:
        parser.error("--warmup должен быть ≥ 0")
    if args.repeats < 1:
        parser.error("--repeats должен быть ≥ 1")
    if args.dry_run:
        try:
            return print_sweep_plan(args)
        except ValueError as exc:
            parser.error(str(exc))

    root = Path.cwd()
    reporter = Reporter(resolve_mode(no_color=True, no_tty=True))
    config = load_config(root / "config.toml")
    input_path = root / "input" / args.input

    if args.sweep:
        try:
            return _run_sweeps(args, config, root, reporter, input_path)
        except ValueError as exc:
            reporter.error(str(exc))
            return 1

    if not input_path.is_file():
        reporter.error(f"Нет файла {input_path}")
        return 1
    return _run_legacy_chains(args, config, root, reporter, input_path)


def _run_legacy_chains(
    args: argparse.Namespace,
    config: Config,
    root: Path,
    reporter: Reporter,
    input_path: Path,
) -> int:
    """Pre-sweep behavior, unchanged: one pass per fixed chain + dated md."""
    ffmpeg_path = resolve_tool("ffmpeg", config.tools.ffmpeg)
    hardware = detect_hardware(config.hardware, cli_num_streams=None, nn_stages_selected=True)
    gpu_label = format_gpu_summary(hardware).replace("×1", "").replace("×", "x").strip(", ")
    gpu_slug_legacy = (
        gpu_label.split(",")[0].replace(" ", "").replace("NVIDIA", "").strip("-") or "GPU"
    )
    rows: list[str] = []

    for chain_name, stages in CHAINS.items():
        kwargs = {}
        if "descale" in stages:
            kwargs["descale_height"] = args.descale_height
        run = resolve_run(config, stages=stages, **kwargs)
        nn = bool({"upscale", "interpolate"} & set(stages))
        hw = detect_hardware(
            config.hardware,
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
        output_path = root / "output" / f"_bench_{chain_name}.mkv"
        if output_path.exists():
            output_path.unlink()
        start = time.monotonic()
        run_pipeline(ctx, input_path, output_path, ffmpeg_path=ffmpeg_path)
        elapsed = time.monotonic() - start
        out_size = output_path.stat().st_size / (1024 * 1024)
        frames = _count_frames(output_path)
        fps_text = f"{frames / elapsed:.2f}" if frames and elapsed > 0 else "n/a"
        rows.append(
            f"| {chain_name} | {' → '.join(stages)} | {elapsed:.1f} | "
            f"{fps_text} ({frames} frames) | {out_size:.1f} |"
        )
        reporter.success(f"{chain_name}: {elapsed:.1f} s ({fps_text} fps)")
        output_path.unlink(missing_ok=True)

    stamp = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    out_file = Path(args.outdir) / f"{stamp}-{gpu_slug_legacy}.md"
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
        "NOTE: single-GPU machine; num_streams per the VRAM budget formula (§9.2)",
        "unless benchmarks/<gpu-slug>/anchor.json overrides it (measured anchor).",
        "",
    ]
    out_file.write_text("\n".join(sample_lines), encoding="utf-8")
    reporter.success(f"Результаты: {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
