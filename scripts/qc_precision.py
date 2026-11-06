"""
QC precision harness: RGBH (fp16) vs RGBS (fp32) for RIFE interpolation.

Measures whether the fp16 clip format is safe AND profitable on the current
GPU: the same production RIFE 4.25 call (vsrife, identical model/params per
stages/interpolate.py) runs twice over three deterministic synthetic fixtures
— once on an RGBS fp32 clip (baseline trust), once on an RGBH fp16 clip
(candidate, format-driven precision per the vsrife canon: "RGBH performs
inference in FP16 mode while RGBS performs inference in FP32 mode"). Pixels
are compared frame-by-frame straight out of interpolation (no encode step:
the compared pixels are identical to what the encoder would receive).

Verdict canon (thresholds below, Russian reporter-style output):
PASS  = global PSNR >= 35 dB AND worst-frame PSNR >= 28 dB AND fps gain >= 10%
FAIL  = a quality threshold is missed (fps is irrelevant then)
INSUFFICIENT-GAIN = quality is fine but the speed-up does not clear 10%

Frames around scene cuts are the elevated-risk zone: the worst-frame table
carries every frame's distance to the nearest known cut (cut indices come
from fixture generation, not from re-detection). The Blackwell rule in
anime_upscale.hardware is NOT touched by this script — the candidate format
is forced here via --format; flipping the production rule is the owner's
separate decision after reading this verdict (docs/qc-precision.md).

The clip is NEVER split (vsrife has no global frame offset — canon): both
runs interpolate each fixture whole.

Measurement honesty: the fixture's numpy content is regenerated inside the
timed drain (identically on both sides), so ABSOLUTE fps figures are lower
than production decode-fed throughput — only the A/B delta is decision-
grade; never compare the absolutes with benchmark.py anchors.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    import vapoursynth as vs

# --- verdict canon (dispatch thresholds; do not tune from data) --------------
PASS_GLOBAL_PSNR_DB = 35.0
PASS_WORST_PSNR_DB = 28.0
PASS_MIN_FPS_GAIN_PCT = 10.0
# Source-frame distance to the nearest cut inside which a worst frame counts
# as "cut-zone risk" (interpolated frames near cuts are the fp16-sensitive spot).
CUT_RISK_FRAMES = 5

QC_SCHEMA_VERSION = 1
QC_FILENAME = "qc_rgbh.json"
# Display precision per candidate format (vsrife: RGBH = fp16, RGBS = fp32).
_FORMAT_PRECISION = {"rgbh": "fp16", "rgbs": "fp32"}

# --- fixture plan constants --------------------------------------------------
FIXTURE_NAMES = ("hard_cuts", "pan_motion", "mixed")
SOURCE_FPS = (24000, 1001)
HARD_CUT_PERIOD = 24  # source frames between cuts in hard_cuts (~1 s)
MIXED_CUT_PERIOD = 48  # source frames between cuts in mixed (~2 s)
WORST_TABLE_ROWS = 10
PEAK = 1.0  # float RGB value range top (PSNR reference level)
# Per-fixture pan speed in source pixels per frame; the gradient/line/texture
# layers derive their offsets from it (fractional, so lines never alias).
MOTION_PX: dict[str, float] = {"hard_cuts": 2.0, "pan_motion": 16.0, "mixed": 6.0}
# Fractional offsets multiplier: keeps every layer's per-frame shift
# non-integer (an integer line shift would alias a 16 px period onto itself).
_LAYER_SHIFT = 0.35
# Fine-line square wave geometry (px).
_LINE_PERIOD_PX = 16.0
_LINE_DUTY_PX = 8.0
# Fixture length band from the dispatch brief, in seconds.
_SECONDS_MIN = 5
_SECONDS_MAX = 8


@dataclass(frozen=True)
class FixturePlan:
    """Deterministic clip specification; cut indices are known by construction."""

    name: str
    frames_src: int
    cut_indices: tuple[int, ...]
    motion_px: float


@dataclass(frozen=True)
class FrameQuality:
    """Per-frame deviation of the candidate against the fp32 reference."""

    index: int
    mse: float
    psnr_db: float
    mad: float


@dataclass(frozen=True)
class AggregateQuality:
    """Fixture-level quality: global PSNR, mean MAD, worst frame, full series."""

    global_psnr_db: float
    mean_mad: float
    worst: FrameQuality
    per_frame: tuple[FrameQuality, ...]


def seconds_to_frames(seconds: int) -> int:
    """Fixture length in source frames at 24000/1001 fps (6 s -> 144)."""
    return round(seconds * SOURCE_FPS[0] / SOURCE_FPS[1])


def fixture_plan(name: str, frames_src: int = 144) -> FixturePlan:
    """
    Build the deterministic plan for one fixture; unknown names are loud.

    - hard_cuts: a full-plate flip every ~1 s (scene-change zone probe);
    - pan_motion: no cuts, fast continuous pan (flow/fine-line probe);
    - mixed: moderate pan + rare ~2 s cuts (production-like mix).
    """
    if name == "hard_cuts":
        cuts: tuple[int, ...] = tuple(range(HARD_CUT_PERIOD, frames_src, HARD_CUT_PERIOD))
    elif name == "pan_motion":
        cuts = ()
    elif name == "mixed":
        cuts = tuple(range(MIXED_CUT_PERIOD, frames_src, MIXED_CUT_PERIOD))
    else:
        msg = f"Неизвестная фикстура «{name}» — доступны: {', '.join(FIXTURE_NAMES)}"
        raise ValueError(msg)
    return FixturePlan(
        name=name,
        frames_src=frames_src,
        cut_indices=cuts,
        motion_px=MOTION_PX[name],
    )


def _plate_id(plan: FixturePlan, n: int) -> int:
    """Plate index at frame n: flips at every known cut (deterministic)."""
    flips = sum(1 for cut in plan.cut_indices if cut <= n)
    return flips % 2


def frame_content(plan: FixturePlan, width: int, height: int, n: int) -> np.ndarray:
    """
    Frame n content as (3, height, width) float32 in [0, 1] — pure arithmetic.

    No RNG anywhere: identical inputs yield bit-identical frames on any
    platform (the plan itself is the "seed"). Layers: an angled gradient
    plate, a moving fine-line square wave (jaggy-line probe), and a
    mid-frequency deterministic texture tile — all translating with
    fractional per-frame offsets so motion never aliases.
    """
    xs = np.arange(width, dtype=np.float64)
    ys = np.arange(height, dtype=np.float64)
    layer_shift = plan.motion_px * n * _LAYER_SHIFT
    grad_shift = plan.motion_px * GRADIENT_SPEED_PX * layer_shift
    frac_x = ((xs + grad_shift) % width) / width
    lines = np.where((xs + layer_shift) % _LINE_PERIOD_PX < _LINE_DUTY_PX, 0.06, -0.06)
    tex = (
        0.03
        * np.sin(0.55 * ((xs + layer_shift * 0.5) % 64.0))[None, :]
        * np.cos(0.45 * ((ys + layer_shift * 0.25) % 64.0))[:, None]
    )
    if _plate_id(plan, n) == 0:
        base = 0.12 + 0.38 * frac_x[None, :] + 0.10 * (ys / height)[:, None]
        plate = base + lines[None, :] + tex
    else:
        base = 0.80 - 0.38 * frac_x[None, :] - 0.10 * (ys / height)[:, None]
        plate = base - lines[None, :] + tex
    channels = np.stack((plate, plate + 0.04, plate - 0.03))
    return np.clip(channels, 0.0, 1.0).astype(np.float32)


# Layer shift speed of the gradient in px per frame (on top of the layer shift).
GRADIENT_SPEED_PX = 4.0


# --- metrics -----------------------------------------------------------------


def frame_metrics(ref: np.ndarray, cand: np.ndarray) -> FrameQuality:
    """
    MSE / PSNR / MAD of one frame pair; float peak = 1.0 (RGB float range).

    fp16 inputs are cast to float32 BEFORE the math so the metric never
    accumulates in half precision (the candidate's own rounding stays in the
    data, the metric stays trustworthy). Identical frames give PSNR = inf.
    """
    if ref.shape != cand.shape:
        msg = f"Метрика: несовпадение shape {ref.shape} vs {cand.shape}"
        raise ValueError(msg)
    diff = np.asarray(cand, dtype=np.float32).astype(np.float64) - np.asarray(
        ref, dtype=np.float32
    ).astype(np.float64)
    mse = float(np.mean(diff * diff))
    mad = float(np.mean(np.abs(diff)))
    psnr = math.inf if mse == 0.0 else 10.0 * math.log10(PEAK * PEAK / mse)
    return FrameQuality(index=0, mse=mse, psnr_db=psnr, mad=mad)


def aggregate_frame_metrics(pairs: list[tuple[np.ndarray, np.ndarray]]) -> AggregateQuality:
    """Fold per-frame pairs into global PSNR (mean of equal-size frame MSEs)."""
    if not pairs:
        msg = "Метрика: пустой список кадров"
        raise ValueError(msg)
    per_frame: list[FrameQuality] = []
    for index, (ref, cand) in enumerate(pairs):
        quality = frame_metrics(ref, cand)
        per_frame.append(
            FrameQuality(index=index, mse=quality.mse, psnr_db=quality.psnr_db, mad=quality.mad)
        )
    global_mse = sum(m.mse for m in per_frame) / len(per_frame)
    global_psnr = math.inf if global_mse == 0.0 else 10.0 * math.log10(PEAK * PEAK / global_mse)
    worst = max(per_frame, key=lambda m: m.mse)
    return AggregateQuality(
        global_psnr_db=global_psnr,
        mean_mad=sum(m.mad for m in per_frame) / len(per_frame),
        worst=worst,
        per_frame=tuple(per_frame),
    )


# --- worst frames and scene-cut binding --------------------------------------


def distance_to_nearest_cut(
    output_index: int, cut_indices: tuple[int, ...], src_ratio: float
) -> float:
    """Distance from output frame k to the nearest cut, in SOURCE frames."""
    if not cut_indices:
        return math.inf
    source_pos = output_index / src_ratio
    return min(abs(source_pos - cut) for cut in cut_indices)


def is_cut_risk(output_index: int, cut_indices: tuple[int, ...], src_ratio: float) -> bool:
    """Return true when the frame sits within CUT_RISK_FRAMES of a known cut."""
    return distance_to_nearest_cut(output_index, cut_indices, src_ratio) <= CUT_RISK_FRAMES


def worst_frame_table(
    per_frame: tuple[FrameQuality, ...],
    cut_indices: tuple[int, ...],
    src_ratio: float,
    top: int = WORST_TABLE_ROWS,
) -> list[dict]:
    """Worst-first table rows with each frame's cut distance (risk binding)."""
    ordered = sorted(per_frame, key=lambda m: (-m.mse, m.index))[:top]
    return [
        {
            "index": m.index,
            "psnr_db": m.psnr_db,
            "mad": round(m.mad, 6),
            "dist_to_cut_src": round(distance_to_nearest_cut(m.index, cut_indices, src_ratio), 3),
        }
        for m in ordered
    ]


# --- verdict -----------------------------------------------------------------


def decide_verdict(
    global_psnr_db: float, worst_psnr_db: float, fps_gain_pct: float
) -> tuple[str, str]:
    """
    Verdict + Russian reason (reporter canon): quality first, speed second.

    PASS needs ALL of: global >= 35 dB, worst-frame >= 28 dB, gain >= 10%.
    A quality miss is FAIL regardless of speed; quality-safe but slow is
    INSUFFICIENT-GAIN. Boundaries are inclusive.

    Total function: any non-finite input (NaN from broken pixels/metrics,
    inf from zero-deviation comparisons) yields ("ERROR", reason) — never
    PASS. NaN bypasses every `<` comparison in Python (all False), so an
    unguarded NaN path silently lands in PASS; a decision instrument that
    can flip a production precision rule must refuse verdicts on numbers
    it cannot trust.
    """
    values = {
        "глобальный PSNR": global_psnr_db,
        "worst-кадр PSNR": worst_psnr_db,
        "выигрыш fps": fps_gain_pct,
    }
    non_finite = [name for name, value in values.items() if not math.isfinite(value)]
    if non_finite:
        listed = ", ".join(f"{name}={value}" for name, value in values.items())
        reason = (
            "Некорректные метрики (NaN/inf) в: "
            + ", ".join(non_finite)
            + f" — прогон недостоверен ({listed}). Правило не переключать."
        )
        return "ERROR", reason
    if global_psnr_db < PASS_GLOBAL_PSNR_DB or worst_psnr_db < PASS_WORST_PSNR_DB:
        reason = (
            f"Качество ниже порога: глобальный PSNR {global_psnr_db:.2f} дБ "
            f"(порог {PASS_GLOBAL_PSNR_DB} дБ), worst-кадр {worst_psnr_db:.2f} дБ "
            f"(порог {PASS_WORST_PSNR_DB} дБ). Правило RGBH не переключать."
        )
        return "FAIL", reason
    if fps_gain_pct < PASS_MIN_FPS_GAIN_PCT:
        reason = (
            f"Качество в пороге, но выигрыш скорости {fps_gain_pct:.2f}% меньше "
            f"{PASS_MIN_FPS_GAIN_PCT}% — переход на RGBH недостаточно выгоден."
        )
        return "INSUFFICIENT-GAIN", reason
    reason = (
        f"Глобальный PSNR {global_psnr_db:.2f} дБ (>= {PASS_GLOBAL_PSNR_DB}), "
        f"worst-кадр {worst_psnr_db:.2f} дБ (>= {PASS_WORST_PSNR_DB}), "
        f"выигрыш {fps_gain_pct:.2f}% (>= {PASS_MIN_FPS_GAIN_PCT}%). "
        "RGBH fp16 безопасен и быстрее — правило можно флипнуть."
    )
    return "PASS", reason


def _poisoned_min(values: list[float]) -> float:
    """
    min() that poisons deterministically: any non-finite input → NaN.

    Plain ``min`` with NaN is order-dependent (comparisons with NaN are all
    False), which would let a single broken fixture select a healthy value
    on one axis and poison another — the aggregate must be predictable so
    the verdict guard can refuse the whole run.
    """
    if any(not math.isfinite(value) for value in values):
        return math.nan
    return min(values)


def worst_case_aggregate(fixtures: list[dict]) -> tuple[float, float, float]:
    """
    Honest worst case across fixtures: (min global PSNR, min worst, min gain).

    NaN/inf anywhere on an axis poisons that axis to NaN deterministically
    (see _poisoned_min) — decide_verdict then refuses the verdict loudly.
    """
    if not fixtures:
        msg = "Вердикт: нет ни одной фикстуры"
        raise ValueError(msg)
    return (
        _poisoned_min([fixture["global_psnr_db"] for fixture in fixtures]),
        _poisoned_min([fixture["worst_frame"]["psnr_db"] for fixture in fixtures]),
        _poisoned_min([fixture["fps_delta_pct"] for fixture in fixtures]),
    )


# --- production-call parity ---------------------------------------------------


def build_rife_kwargs(  # noqa: PLR0913 - mirrors stages/interpolate.py's rife(...) 1:1
    *,
    clip_width: int,
    clip_height: int,
    interpolate_cfg: object,
    device_index: int,
    trt_allowed: bool,
    cache_dir: Path,
) -> dict:
    """
    Build the exact production rife() kwargs (stages/interpolate.py), format aside.

    Format is NOT part of the kwargs: vsrife selects fp16/fp32 from the clip
    format itself (RGBH = fp16, RGBS = fp32 — upstream canon), so both runs
    share these kwargs and differ only in the clip they receive. The TRT gate
    mirrors production: cfg.trt AND hardware.recommended_trt.
    """
    use_trt = bool(interpolate_cfg.trt) and trt_allowed  # type: ignore[attr-defined]
    expanded = Path(cache_dir).expanduser()
    expanded.mkdir(parents=True, exist_ok=True)  # vsrife requires an existing dir
    return {
        "device_index": device_index,
        "model": interpolate_cfg.model,  # type: ignore[attr-defined]
        "auto_download": True,
        "fps_num": interpolate_cfg.target_fps[0],  # type: ignore[attr-defined]
        "fps_den": interpolate_cfg.target_fps[1],  # type: ignore[attr-defined]
        "sc": interpolate_cfg.scene_detection,  # type: ignore[attr-defined]
        "sc_threshold": (  # type: ignore[attr-defined]
            interpolate_cfg.scene_threshold if interpolate_cfg.scene_detection else None
        ),
        "trt": use_trt,
        "trt_static_shape": True,
        "trt_opt_shape": [clip_height, clip_width],
        "trt_cache_dir": str(expanded),
    }


def ensure_frame_identity(ref_count: int, cand_count: int) -> None:
    """Dropped/duplicated frames disqualify the run loudly (research §6.1)."""
    if ref_count != cand_count:
        msg = (
            "Число кадров разошлось между прогонами: "
            f"референс {ref_count}, кандидат {cand_count} — интерполяция "
            "теряет или дублирует кадры, результат недействителен."
        )
        raise RuntimeError(msg)


# --- report -------------------------------------------------------------------


def build_qc_report(
    *,
    gpu_name: str,
    fixtures: list[dict],
    run_config: dict,
    notes: list[str],
) -> dict:
    """Machine-readable QC record for benchmarks/<gpu-slug>/qc_rgbh.json."""
    global_psnr, worst_psnr, fps_gain = worst_case_aggregate(fixtures)
    verdict, reason = decide_verdict(global_psnr, worst_psnr, fps_gain)
    return {
        "schema_version": QC_SCHEMA_VERSION,
        "date": datetime.now(tz=UTC).strftime("%Y-%m-%d"),
        "gpu_name": gpu_name,
        "run_config": run_config,
        "thresholds": {
            "global_psnr_db": PASS_GLOBAL_PSNR_DB,
            "worst_psnr_db": PASS_WORST_PSNR_DB,
            "fps_gain_pct": PASS_MIN_FPS_GAIN_PCT,
        },
        "fixtures": fixtures,
        "fps_delta_pct": round(fps_gain, 2),
        "verdict": verdict,
        "verdict_reason": reason,
        "notes": notes,
    }


def _null_infinite(value: object) -> object:
    """JSON has no Infinity: map inf/-inf/nan to null (= identical frames)."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _null_infinite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_null_infinite(item) for item in value]
    return value


def serialize_qc_report(report: dict) -> str:
    """Stable JSON text; infinite PSNR becomes null (= identical frames)."""
    return json.dumps(_null_infinite(report), indent=2, ensure_ascii=False) + "\n"


def write_qc_report(out_dir: Path, report: dict) -> Path:
    """One QC file per GPU, overwritten per run (same layout as anchor.json)."""
    from anime_upscale.hardware import gpu_slug  # noqa: PLC0415 - lazy, keeps import light

    path = out_dir / gpu_slug(report["gpu_name"]) / QC_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_qc_report(report), encoding="utf-8")
    return path


# --- CLI ----------------------------------------------------------------------


def parse_qc_args(argv: list[str] | None = None) -> argparse.Namespace:
    """QC-run CLI; the candidate format is forced HERE, never via hardware.py."""
    parser = argparse.ArgumentParser(
        description=(
            "QC-стенд: RGBH (fp16) против RGBS (fp32) для RIFE — качество "
            "(PSNR/кадры) и скорость на детерминированных синтетических фикстурах."
        )
    )
    parser.add_argument(
        "--format",
        choices=("rgbh", "rgbs"),
        default="rgbh",
        help="Формат КАНДИДАТА (референс всегда RGBS fp32). Явный форс вместо правила hardware.py.",
    )
    parser.add_argument("--seconds", type=int, default=6, help="Длина фикстур, 5..8 с.")
    parser.add_argument("--warmup", type=int, default=1, help="Отбрасываемых прогревов на прогон.")
    parser.add_argument("--repeats", type=int, default=3, help="Измеряемых повторов (медиана fps).")
    parser.add_argument(
        "--outdir", default="benchmarks", help="Каталог для qc_rgbh.json (по gpu-slug)."
    )
    parser.add_argument("--dry-run", action="store_true", help="Печатает план QC и выходит.")
    args = parser.parse_args(argv)
    if not _SECONDS_MIN <= args.seconds <= _SECONDS_MAX:
        parser.error(f"--seconds: допустимо {_SECONDS_MIN}..{_SECONDS_MAX} с")
    if args.warmup < 0:
        parser.error("--warmup должен быть >= 0")
    if args.repeats < 1:
        parser.error("--repeats должен быть >= 1")
    return args


# --- VapourSynth seams (lazy: metrics above run without VS/torch/vsrife) ------


def build_fixture_clip(plan: FixturePlan, width: int, height: int) -> vs.VideoNode:
    """Build the fixture as a VS RGBS clip: numpy content written per frame."""
    import vapoursynth as vs  # noqa: PLC0415 - lazy VS seam

    core = vs.core
    base = core.std.BlankClip(
        format=vs.RGBS,
        width=width,
        height=height,
        length=plan.frames_src,
        fpsnum=SOURCE_FPS[0],
        fpsden=SOURCE_FPS[1],
        color=[0.0, 0.0, 0.0],
    )

    def selector(n, f):  # type: ignore[no-untyped-def]  # noqa: ANN001, ANN202 - VS selector
        # VS R81 invokes selectors with keyword args n=/f=; a single-clip
        # ModifyFrame passes the frame itself, multi-clip passes a list.
        frames = f if isinstance(f, list) else [f]
        new_frame = frames[0].copy()
        content = frame_content(plan, width, height, n)
        for plane in range(new_frame.format.num_planes):
            with new_frame[plane] as view:
                np.copyto(np.asarray(view), content[plane])
        return new_frame

    return core.std.ModifyFrame(base, [base], selector)  # ty: ignore[unresolved-attribute]


def to_candidate_format(clip: vs.VideoNode) -> vs.VideoNode:
    """Cast RGBS -> RGBH (same-family: no matrix, mirrors pipeline _to_rgb)."""
    import vapoursynth as vs  # noqa: PLC0415 - lazy VS seam

    return clip.resize.Bilinear(format=vs.RGBH)  # ty: ignore[unresolved-attribute]


def apply_rife(clip: vs.VideoNode, kwargs: dict, rife_fn: Callable | None = None) -> vs.VideoNode:
    """Run the GPU seam: vsrife.rife (production path) or an injected fake in tests."""
    if rife_fn is None:
        from vsrife import rife as rife_fn  # noqa: PLC0415 - lazy GPU stack

    return rife_fn(clip, **kwargs)


def apply_constant_offset(clip: vs.VideoNode, offset: float) -> vs.VideoNode:
    """Add a known constant on every plane (test helper for known-answer metrics)."""
    import vapoursynth as vs  # noqa: PLC0415 - lazy VS seam

    core = vs.core

    def selector(n, f):  # type: ignore[no-untyped-def]  # noqa: ANN001, ANN202, ARG001 - VS selector, n required by VS
        frames = f if isinstance(f, list) else [f]
        new_frame = frames[0].copy()
        for plane in range(new_frame.format.num_planes):
            with new_frame[plane] as view:
                np.asarray(view)[:] += np.float32(offset)
        return new_frame

    return core.std.ModifyFrame(clip, [clip], selector)  # ty: ignore[unresolved-attribute]


def clip_frame_planes(clip: vs.VideoNode, index: int) -> np.ndarray:
    """Read frame planes as a stacked (planes, h, w) array (float16 for RGBH)."""
    frame = clip.get_frame(index)
    return np.stack([np.asarray(frame[plane]) for plane in range(frame.format.num_planes)])


def extract_pair_metrics(ref_clip: vs.VideoNode, cand_clip: vs.VideoNode) -> AggregateQuality:
    """Compare the two graphs lockstep, frame by frame, straight out of RIFE."""
    ref_count, cand_count = len(ref_clip), len(cand_clip)
    ensure_frame_identity(ref_count, cand_count)
    pairs = [
        (clip_frame_planes(ref_clip, i), clip_frame_planes(cand_clip, i)) for i in range(ref_count)
    ]
    return aggregate_frame_metrics(pairs)


def drain_wall_seconds(clip: vs.VideoNode) -> float:
    """Drive the whole graph once (the production work), return wall seconds."""
    start = time.monotonic()
    for index in range(len(clip)):
        clip.get_frame(index)
    return time.monotonic() - start


def _reset_torch_peak() -> None:
    import torch  # noqa: PLC0415 - lazy GPU stack

    torch.cuda.reset_peak_memory_stats()


def _torch_reserved_peak_mb() -> float:
    import torch  # noqa: PLC0415 - lazy GPU stack

    return torch.cuda.max_memory_reserved() / (1024 * 1024)


def measure_speed(
    build_graph: Callable[[], object],
    *,
    warmup: int,
    repeats: int,
    measure_vram: bool,
) -> tuple[float, list[float], float | None]:
    """
    Warm-up (discarded) + N timed drains of fresh graphs -> median fps, VRAM.

    A fresh rife node per pass keeps VS frame caches from leaking between
    passes; the torch peak is reset immediately before each measured pass
    (benchmark.py canon). Returns (median_fps, fps_repeats, vram_peak_mb).

    Honesty note: the fixture content is numpy-regenerated inside every
    timed drain (ModifyFrame recomputes it per frame) — the same overhead
    on both sides, so the A/B fps DELTA is decision-grade while ABSOLUTE
    fps understates production decode-fed throughput and must NOT be
    compared with benchmark.py anchors.
    """
    for _ in range(warmup):
        drain_wall_seconds(build_graph())
    fps_repeats: list[float] = []
    vram_peak: float | None = None
    for _ in range(repeats):
        clip = build_graph()
        frames = len(clip)
        if measure_vram:
            _reset_torch_peak()
        wall = drain_wall_seconds(clip)
        if wall > 0 and frames:
            fps_repeats.append(frames / wall)
        if measure_vram:
            peak = _torch_reserved_peak_mb()
            vram_peak = peak if vram_peak is None else max(vram_peak, peak)
    if not fps_repeats:
        msg = "Скорость: ни один замер не дал положительного времени/кадров"
        raise RuntimeError(msg)
    return median(fps_repeats), fps_repeats, vram_peak


# --- orchestration ------------------------------------------------------------


def qc_fixture(  # noqa: PLR0913, PLR0917 - driver-context bundle (plan/kwargs/run shape/reporter)
    plan: FixturePlan,
    width: int,
    height: int,
    kwargs: dict,
    candidate_format: str,
    warmup: int,
    repeats: int,
    reporter: object,
    rife_fn: Callable | None = None,
) -> dict:
    """
    Run one fixture: speed A/B (RGBS vs RGBH) + frame-exact quality comparison.

    Clip-split is never done (vsrife lacks a global frame offset — canon);
    both runs interpolate the fixture whole. Speed and metrics each build
    fresh graphs, so no VS frame cache leaks across passes or formats.
    """
    fixture_clip = build_fixture_clip(plan, width, height)

    def make_graph(*, candidate: bool) -> vs.VideoNode:
        source = to_candidate_format(fixture_clip) if candidate else fixture_clip
        return apply_rife(source, kwargs, rife_fn=rife_fn)

    fps_ref, repeats_ref, vram_ref = measure_speed(
        lambda: make_graph(candidate=False), warmup=warmup, repeats=repeats, measure_vram=True
    )
    reporter.note(
        f"[{plan.name}] RGBS fp32: {fps_ref:.2f} fps "
        f"(медиана {repeats}, VRAM peak {vram_ref:.0f} МБ)"
    )
    fps_cand, repeats_cand, vram_cand = measure_speed(
        lambda: make_graph(candidate=True), warmup=warmup, repeats=repeats, measure_vram=True
    )
    reporter.note(
        f"[{plan.name}] {candidate_format.upper()} fp16: {fps_cand:.2f} fps "
        f"(медиана {repeats}, VRAM peak {vram_cand:.0f} МБ)"
    )

    ref_graph = make_graph(candidate=False)
    cand_graph = make_graph(candidate=True)
    ensure_frame_identity(len(ref_graph), len(cand_graph))
    frames_out = len(cand_graph)
    quality = extract_pair_metrics(ref_graph, cand_graph)
    target_fps = Fraction(kwargs["fps_num"], kwargs["fps_den"])
    src_ratio = float(target_fps / Fraction(SOURCE_FPS[0], SOURCE_FPS[1]))
    table = worst_frame_table(quality.per_frame, plan.cut_indices, src_ratio)
    fps_delta = (fps_cand - fps_ref) / fps_ref * 100.0

    record = {
        "name": plan.name,
        "frames_src": plan.frames_src,
        "frames_out": frames_out,
        "cut_indices": list(plan.cut_indices),
        "global_psnr_db": quality.global_psnr_db,
        "mean_mad": round(quality.mean_mad, 6),
        "worst_frame": {
            "index": quality.worst.index,
            "psnr_db": quality.worst.psnr_db,
            "mad": round(quality.worst.mad, 6),
            "dist_to_cut_src": round(
                distance_to_nearest_cut(quality.worst.index, plan.cut_indices, src_ratio), 3
            ),
        },
        "worst_table": table,
        "fps_ref": round(fps_ref, 2),
        "fps_ref_repeats": [round(value, 2) for value in repeats_ref],
        "fps_cand": round(fps_cand, 2),
        "fps_cand_repeats": [round(value, 2) for value in repeats_cand],
        "fps_delta_pct": round(fps_delta, 2),
        "vram_peak_mb_ref": round(vram_ref, 1) if vram_ref is not None else None,
        "vram_peak_mb_cand": round(vram_cand, 1) if vram_cand is not None else None,
    }

    reporter.success(
        f"[{plan.name}] глобальный PSNR {quality.global_psnr_db:.2f} дБ, worst-кадр "
        f"#{quality.worst.index} {quality.worst.psnr_db:.2f} дБ, дельта fps {fps_delta:+.2f}%"
    )
    if is_cut_risk(quality.worst.index, plan.cut_indices, src_ratio):
        reporter.warn(
            f"[{plan.name}] worst-кадр #{quality.worst.index} лежит в зоне кат-а "
            f"(<= {CUT_RISK_FRAMES} исходных кадров) — зона повышенного риска fp16."
        )
    for row in table:
        reporter.note(
            f"[{plan.name}]   кадр #{row['index']}: {row['psnr_db']:.2f} дБ, "
            f"MAD {row['mad']:.6f}, до кат-а {row['dist_to_cut_src']:.2f} исх. кадров"
        )
    return record


def run_qc(args: argparse.Namespace, root: Path, reporter: object) -> int:
    """Full QC: hardware gate -> 3 fixtures -> verdict -> JSON for history."""
    from anime_upscale.config import load_config  # noqa: PLC0415 - lazy project stack
    from anime_upscale.hardware import detect_hardware  # noqa: PLC0415

    config = load_config(root / "config.toml")
    interpolate_cfg = config.stages.interpolate
    hardware = detect_hardware(config.hardware, cli_num_streams=None, nn_stages_selected=True)
    device_index = hardware.gpus[0].index
    trt_allowed = hardware.recommended_trt
    if interpolate_cfg.trt and not trt_allowed:
        reporter.note(
            "TensorRT отключен правилом архитектуры GPU (Blackwell) — QC гоняет "
            "native PyTorch path, как прод."
        )

    kwargs = build_rife_kwargs(
        clip_width=1920,
        clip_height=1080,
        interpolate_cfg=interpolate_cfg,
        device_index=device_index,
        trt_allowed=trt_allowed,
        cache_dir=Path(interpolate_cfg.trt_cache_dir),
    )
    frames_src = seconds_to_frames(args.seconds)
    fixtures: list[dict] = []
    for name in FIXTURE_NAMES:
        plan = fixture_plan(name, frames_src=frames_src)
        fixtures.append(
            qc_fixture(
                plan,
                width=1920,
                height=1080,
                kwargs=kwargs,
                candidate_format=args.format,
                warmup=args.warmup,
                repeats=args.repeats,
                reporter=reporter,
            )
        )

    global_psnr, worst_psnr, fps_gain = worst_case_aggregate(fixtures)
    verdict, reason = decide_verdict(global_psnr, worst_psnr, fps_gain)
    report = build_qc_report(
        gpu_name=hardware.gpus[0].name,
        fixtures=fixtures,
        run_config={
            "candidate_format": args.format,
            "reference_format": "rgbs",
            "model": interpolate_cfg.model,
            "target_fps": f"{interpolate_cfg.target_fps[0]}/{interpolate_cfg.target_fps[1]}",
            "scene_detection": interpolate_cfg.scene_detection,
            "scene_threshold": interpolate_cfg.scene_threshold,
            "trt": kwargs["trt"],
            "device_index": device_index,
            "seconds": args.seconds,
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        notes=[],
    )
    out_path = write_qc_report(Path(args.outdir), report)

    reporter.note("")
    reporter.note("Итог по фикстурам (худший случай):")
    reporter.note(
        f"  глобальный PSNR {global_psnr:.2f} дБ, worst-кадр {worst_psnr:.2f} дБ, "
        f"выигрыш {fps_gain:+.2f}%"
    )
    if verdict == "PASS":
        reporter.success(f"ВЕРДИКТ: PASS — {reason}")
        reporter.note(
            "Следующий шаг (решение владельца): перевернуть правило "
            "hardware.py recommended_interp_format для Blackwell на 'rgbh'."
        )
    elif verdict == "INSUFFICIENT-GAIN":
        reporter.warn(f"ВЕРДИКТ: INSUFFICIENT-GAIN — {reason}")
    elif verdict == "ERROR":
        reporter.error(f"ВЕРДИКТ: ERROR — {reason}")
        reporter.success(f"Результат записан: {out_path}")
        return 1
    else:
        reporter.warn(f"ВЕРДИКТ: FAIL — {reason}")
    reporter.success(f"Результат записан: {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """QC harness entry: dry-run prints the plan; a real run needs a CUDA GPU."""
    args = parse_qc_args(argv)
    root = Path.cwd()
    from anime_upscale.hardware import HardwareError  # noqa: PLC0415 - light module
    from anime_upscale.reporting import Reporter, resolve_mode  # noqa: PLC0415 - lazy project stack

    reporter = Reporter(resolve_mode(no_color=True, no_tty=True))
    if args.dry_run:
        frames_src = seconds_to_frames(args.seconds)
        precision = _FORMAT_PRECISION[args.format]
        reporter.note("План QC (без выполнения):")
        reporter.note(
            f"  кандидат: {args.format.upper()} {precision} vs референс RGBS fp32, "
            f"фикстуры {len(FIXTURE_NAMES)} × {frames_src} исходных кадров "
            f"({args.seconds} с, {SOURCE_FPS[0]}/{SOURCE_FPS[1]} fps)"
        )
        for name in FIXTURE_NAMES:
            plan = fixture_plan(name, frames_src=frames_src)
            reporter.note(
                f"  [{name}] кат-ы: {list(plan.cut_indices) or 'нет'}, "
                f"движение {plan.motion_px} px/кадр"
            )
        reporter.note(
            f"  пороги: global {PASS_GLOBAL_PSNR_DB} дБ, worst {PASS_WORST_PSNR_DB} дБ, "
            f"fps-выигрыш {PASS_MIN_FPS_GAIN_PCT}%"
        )
        reporter.note(f"  результат: {args.outdir}/<gpu-slug>/{QC_FILENAME}")
        return 0
    try:
        return run_qc(args, root, reporter)
    except (HardwareError, ImportError, RuntimeError) as exc:
        # Operational failures (no GPU, missing VS/vsrife/torch, frame-count
        # mismatch) get a loud Russian line and exit 1; anything else is a bug
        # and keeps its traceback (fail loudly, never fake success).
        reporter.error(f"QC-прогон прерван: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
