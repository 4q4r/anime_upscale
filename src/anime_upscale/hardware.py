"""
Hardware detection → frozen HardwareProfile (plan §9).

Detection is a capability probe, never a config source: the single source of
truth stays config.toml. Override precedence: CLI > [hardware] > measured
anchors (benchmarks/<gpu-slug>/anchor.json, strictly invalidated by
gpu_name + driver + run_config) > detected (VRAM formula constants) >
built-in default. No anchor file, no wiring, or any mismatch → byte-identical
formula behavior.
"""

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from anime_upscale.config import HardwareOverrides
from anime_upscale.reporting import Reporter

# VRAM budget constants (plan §9.2, recalibrated per the plan-b-streams §2.5
# verdict). What the number controls: the VapourSynth output-loop prefetch
# depth (frames in flight per GPU, consumed at pipeline.py) — NOT GPU
# inference concurrency. vsrife 5.7.0 exposes no stream knob: the TRT path
# builds via torch_tensorrt and executes one frame per call, the PyTorch path
# serializes forwards behind per-phase locks, so the legacy "streams" reading
# was a misnomer. These are the FALLBACK-tier values, frozen so the stated
# anchors hold byte-identically: 16 GB → 2, 24 GB → 3, T4 15 GB → 1,
# GTX 1650 4 GB → 1. A schema-valid measured anchor
# (benchmarks/<gpu-slug>/anchor.json) outranks them; explicit CLI/[hardware]
# overrides outrank both.
_BUDGET_FRACTION = 0.8
# Legacy pre-2020 CUDA context figure (discuss.pytorch.org #79976: "used to
# take 750MB in previous versions"). Modern canon is ~300–500 MB per process
# (same thread; pytorch/serve #2128) — the measured branch below uses it.
_CONTEXT_RESERVE_MB = 750
_UPSCALE_RESERVE_MB = 2500
_SLACK_MB = 1024
# Frozen legacy placeholder, renamed to state the mechanism it actually
# budgets: VS prefetch frame buffering per slot, not hypothetical GPU
# inference streams (none exist — see the verdict note above). Value frozen
# for byte-identical fallback outputs.
_PREFETCH_SLOT_BUDGET_MB = 4096
_STREAM_CLAMP = (1, 4)

# Measured-branch constants (applied only when a schema-valid anchor matches
# gpu_name + driver + run_config): modern CUDA-context canon upper bound
# (discuss.pytorch #79976 / pytorch/serve #2128) and the conservative
# canonical per-prefetch-slot budget (frame buffers + staging) used when the
# anchor's own sweep cannot yield a per-slot slope. The anchor's measured
# deltas always outrank the canonical slot figure.
_CONTEXT_RESERVE_MB_MEASURED = 500
_PREFETCH_SLOT_MB_MEASURED = 128
# Upper sane prefetch depth when the anchor names no saturation_streams.
_PREFETCH_DEPTH_CAP = 16
# A slope needs at least two swept points (schema forbids duplicates; points
# are strictly increasing, so first/last give the widest, most robust delta).
_SLOPE_MIN_POINTS = 2

# Measured anchors: streams are worth their VRAM only if a point adds ≥5%
# over the single-stream baseline (plan §3, resolution rule).
_ANCHOR_MIN_GAIN_FRACTION = 1.05

ANCHOR_SCHEMA_VERSION = 1
ANCHOR_FILENAME = Path("benchmarks")  # anchor lives at benchmarks/<gpu-slug>/anchor.json
_CRF_MAX = 51  # x265 CRF range top (validator bound; 16.5 is legal, bool is not)

_SOURCE_FORMULA = "formula"
_SOURCE_MEASURED = "measured"
_SOURCE_CONFIG = "config"

_ANCHOR_TOP_KEYS = {
    "schema_version",
    "date",
    "gpu_name",
    "gpu_vram_mb",
    "driver",
    "cuda",
    "torch",
    "pipeline_commit",
    "run_config",
    "clip",
    "streams_sweep",
    "pools_sweep",
    "saturation_streams",
    "saturation_pools",
    "notes",
}
_ANCHOR_RUN_CONFIG_KEYS = {"preset", "crf", "interp_format", "trt", "chains"}
_ANCHOR_CLIP_KEYS = {"file", "frames_in", "src_fps", "height", "seconds"}
_ANCHOR_STREAM_POINT_KEYS = {
    "streams",
    "fps",
    "fps_repeats",
    "vram_peak_mb",
    "torch_reserved_mb",
    "smi_delta_mb",
}
_ANCHOR_POOL_POINT_KEYS = {"pools", "fps", "fps_repeats", "wall_s", "rss_peak_mb"}


class HardwareError(Exception):
    """Loud hardware-prerequisite failure (Russian message)."""


def gpu_slug(name: str) -> str:
    """Directory slug for a GPU name, identical to the benchmark's derivation."""
    return name.replace(" ", "").replace("NVIDIA", "").strip("-") or "GPU"


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _positive_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return value > 0


def _checked_int(value: object, minimum: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _validate_stream_extras(point: dict, i: int, label: str, problems: list[str]) -> bool:
    """streams-point-specific fields; True when the point stays valid."""
    if not _positive_number(point["vram_peak_mb"]):
        problems.append(f"{label}[{i}]: vram_peak_mb обязательно и должно быть числом > 0")
        return False
    for key in ("torch_reserved_mb", "smi_delta_mb"):
        # smi_delta может быть 0 или отрицательным (память уже резидентна,
        # джиттер драйвера) — диагностическое поле, допускаем ≥ 0 или null.
        value = point[key]
        if value is not None and not (_is_number(value) and value >= 0):
            problems.append(f"{label}[{i}]: {key} должно быть числом ≥ 0 или null")
            return False
    return True


def _validate_pool_extras(point: dict, i: int, label: str, problems: list[str]) -> bool:
    """pools-point-specific fields; True when the point stays valid."""
    if not _positive_number(point["wall_s"]):
        problems.append(f"{label}[{i}]: wall_s должно быть числом > 0")
        return False
    rss = point["rss_peak_mb"]
    if rss is not None and not (_is_number(rss) and rss >= 0):
        problems.append(f"{label}[{i}]: rss_peak_mb должно быть числом ≥ 0 или null")
        return False
    return True


def _point_shape_value(  # noqa: PLR0913, PLR0917, PLR0911 - guard-chain validator
    point: object,
    i: int,
    label: str,
    point_keys: set[str],
    seen_values: list[int],
    problems: list[str],
) -> int | None:
    """
    Check common per-point shape; return the point value or None when invalid.

    Early returns are intentional (PLR0911): each check names exactly one
    violation.
    """
    if not isinstance(point, dict):
        problems.append(f"{label}[{i}]: точка должна быть JSON-объектом")
        return None
    missing = point_keys - set(point)
    extra = set(point) - point_keys
    if missing:
        problems.append(f"{label}[{i}]: отсутствуют ключи {sorted(missing)}")
        return None
    if extra:
        problems.append(f"{label}[{i}]: лишние ключи {sorted(extra)}")
        return None
    value = point[label.removesuffix("_sweep")]
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        problems.append(f"{label}[{i}]: значение точки должно быть целым ≥ 1")
        return None
    if seen_values and value <= seen_values[-1]:
        problems.append(f"{label}[{i}]: точки должны быть строго возрастающими без дублей")
        return None
    if not _positive_number(point.get("fps")):
        problems.append(f"{label}[{i}]: fps должно быть числом > 0")
        return None
    repeats = point.get("fps_repeats")
    repeats_ok = (
        isinstance(repeats, list) and bool(repeats) and all(_positive_number(r) for r in repeats)
    )
    if not repeats_ok:
        problems.append(f"{label}[{i}]: fps_repeats должен быть непустым массивом чисел > 0")
        return None
    return value


def _validate_sweep_points(
    points: object,
    point_keys: set[str],
    label: str,
    problems: list[str],
) -> list[dict]:
    """Shape-check one sweep array; returns the points when structurally sane."""
    if not isinstance(points, list):
        problems.append(f"{label}: ожидается массив точек, получено {type(points).__name__}")
        return []
    check_extras = _validate_stream_extras if label == "streams_sweep" else _validate_pool_extras
    seen_values: list[int] = []
    sane: list[dict] = []
    for i, point in enumerate(points):
        value = _point_shape_value(point, i, label, point_keys, seen_values, problems)
        if value is None or not isinstance(point, dict):
            continue
        if not check_extras(point, i, label, problems):
            continue
        seen_values.append(value)
        sane.append(point)
    return sane


def _validate_run_config(run_config: object, problems: list[str]) -> None:
    if not isinstance(run_config, dict):
        problems.append("run_config должен быть JSON-объектом")
        return
    missing_rc = _ANCHOR_RUN_CONFIG_KEYS - set(run_config)
    extra_rc = set(run_config) - _ANCHOR_RUN_CONFIG_KEYS
    if missing_rc:
        problems.append(f"run_config: отсутствуют ключи {sorted(missing_rc)}")
    if extra_rc:
        problems.append(f"run_config: лишние ключи {sorted(extra_rc)}")
    if not isinstance(run_config.get("preset"), str) or not run_config.get("preset"):
        problems.append("run_config.preset должен быть непустой строкой")
    crf = run_config.get("crf")
    if not isinstance(crf, (int, float)) or isinstance(crf, bool):
        problems.append("run_config.crf должен быть числом")
    elif crf < 0 or crf > _CRF_MAX:
        problems.append(f"run_config.crf должен быть в 0..{_CRF_MAX}")
    if run_config.get("interp_format") not in ("rgbs", "rgbh"):
        problems.append("run_config.interp_format должен быть 'rgbs' или 'rgbh'")
    if not isinstance(run_config.get("trt"), bool):
        problems.append("run_config.trt должен быть булевым")
    chains = run_config.get("chains")
    if not isinstance(chains, dict) or not all(
        isinstance(chains.get(k), str) and chains.get(k) for k in ("streams", "pools")
    ):
        problems.append("run_config.chains должен задавать непустые строки streams/pools")


def _validate_clip(clip: object, problems: list[str]) -> None:
    if not isinstance(clip, dict):
        problems.append("clip должен быть JSON-объектом")
        return
    missing_clip = _ANCHOR_CLIP_KEYS - set(clip)
    extra_clip = set(clip) - _ANCHOR_CLIP_KEYS
    if missing_clip:
        problems.append(f"clip: отсутствуют ключи {sorted(missing_clip)}")
    if extra_clip:
        problems.append(f"clip: лишние ключи {sorted(extra_clip)}")
    if not isinstance(clip.get("file"), str) or not clip.get("file"):
        problems.append("clip.file должен быть непустой строкой")
    if not isinstance(clip.get("src_fps"), str) or not clip.get("src_fps"):
        problems.append("clip.src_fps должен быть непустой строкой")
    problems.extend(
        f"clip.{key} должен быть целым ≥ 1"
        for key in ("frames_in", "height")
        if not _checked_int(clip.get(key), 1)
    )
    if not _is_number(clip.get("seconds")) or clip.get("seconds", -1) <= 0:
        problems.append("clip.seconds должен быть числом > 0")


def _validate_saturation(
    anchor: dict,
    streams_points: list[dict],
    pools_points: list[dict],
    problems: list[str],
) -> None:
    """
    Saturation must point inside the recorded sweep.

    null is LEGAL with a non-empty sweep: fps may still rise at the last
    swept point (plan §3 — never assume saturation).
    """
    sat_streams = anchor.get("saturation_streams")
    if sat_streams is not None:
        swept = {int(p["streams"]) for p in streams_points if isinstance(p.get("streams"), int)}
        if sat_streams not in swept:
            problems.append(
                f"saturation_streams={sat_streams} указывает вне проверенных точек {sorted(swept)}"
            )
    sat_pools = anchor.get("saturation_pools")
    if sat_pools is not None:
        swept_pools = {int(p["pools"]) for p in pools_points if isinstance(p.get("pools"), int)}
        outside = sat_pools not in swept_pools
        if outside:
            problems.append(
                f"saturation_pools={sat_pools} указывает вне "
                f"проверенных точек {sorted(swept_pools)}"
            )


def validate_anchor(anchor: object) -> list[str]:
    """
    Manual anchor-schema check (jsonschema-free).

    Returns a list of Russian problem descriptions; an empty list means the
    anchor is schema-valid.
    """
    problems: list[str] = []
    if not isinstance(anchor, dict):
        return ["Якорь должен быть JSON-объектом"]
    missing = _ANCHOR_TOP_KEYS - set(anchor)
    extra = set(anchor) - _ANCHOR_TOP_KEYS
    if missing:
        problems.append(f"Отсутствуют ключи: {sorted(missing)}")
    if extra:
        problems.append(f"Лишние ключи: {sorted(extra)}")
    if anchor.get("schema_version") != ANCHOR_SCHEMA_VERSION:
        problems.append(f"schema_version должен быть {ANCHOR_SCHEMA_VERSION}")
    problems.extend(
        f"{key} должен быть непустой строкой"
        for key in ("gpu_name", "driver")
        if not isinstance(anchor.get(key), str) or not anchor.get(key)
    )
    if not _checked_int(anchor.get("gpu_vram_mb"), 1):
        problems.append("gpu_vram_mb должен быть целым > 0")

    _validate_run_config(anchor.get("run_config"), problems)
    _validate_clip(anchor.get("clip"), problems)
    streams_points = _validate_sweep_points(
        anchor.get("streams_sweep"), _ANCHOR_STREAM_POINT_KEYS, "streams_sweep", problems
    )
    pools_points = _validate_sweep_points(
        anchor.get("pools_sweep"), _ANCHOR_POOL_POINT_KEYS, "pools_sweep", problems
    )
    _validate_saturation(anchor, streams_points, pools_points, problems)

    if not isinstance(anchor.get("notes"), str):
        problems.append("notes должен быть строкой")
    return problems


def load_anchor(path: Path, reporter: Reporter | None = None) -> dict | None:
    """
    Read benchmarks/<gpu-slug>/anchor.json.

    Missing file → None silently (byte-identical fallback). Present but
    unreadable → loud note + None.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        if reporter is not None:
            reporter.note(f"Якорь {path} не читается ({exc}) — откат на формулу VRAM.")
        return None
    return data if isinstance(data, dict) else None


def _resolve_streams_rule(points: list[dict], vram_mb: int) -> int | None:
    """
    Largest swept point within the usable budget that adds ≥5% throughput.

    Baseline = the 1-stream point (plan §3, resolution rule).
    """
    budget = _BUDGET_FRACTION * vram_mb - _SLACK_MB
    gate = 0.0
    chosen: int | None = None
    for i, point in enumerate(points):
        fps = point.get("fps")
        peak = point.get("vram_peak_mb")
        streams = point.get("streams")
        if not isinstance(fps, (int, float)) or isinstance(fps, bool) or fps <= 0:
            return chosen
        if not isinstance(peak, (int, float)) or isinstance(peak, bool) or peak <= 0:
            return chosen
        if not isinstance(streams, int) or isinstance(streams, bool) or streams < 1:
            return chosen
        if i == 0:
            gate = float(fps) * _ANCHOR_MIN_GAIN_FRACTION
        if float(peak) <= budget and float(fps) >= gate:
            chosen = int(streams)
    return chosen


def resolve_streams_from_anchor(anchor: dict, vram_mb: int) -> int | None:
    """
    Measured num_streams for a GPU with vram_mb from a schema-valid anchor.

    Retained for scripts/benchmark.py (anchor-template display); live
    detection resolves prefetch depth via resolve_prefetch_from_anchor.
    """
    points = anchor.get("streams_sweep")
    if not isinstance(points, list):
        return None
    return _resolve_streams_rule([p for p in points if isinstance(p, dict)], vram_mb)


def resolve_prefetch_from_anchor(
    anchor: dict,
    vram_mb: int,
    reporter: Reporter | None = None,
) -> int:
    """
    Measured prefetch depth from a schema-valid anchor (plan-b-streams §4).

    The number budgets VS prefetch slots, not GPU inference streams (vsrife
    5.7.0 exposes no concurrency knob — verified from installed source).
    Per-slot VRAM cost = the vram_peak_mb slope across the streams sweep
    ((peak_last − peak_first) / (streams_last − streams_first)); a single
    point, an empty sweep, or a non-positive slope (allocator jitter) falls
    back to the canonical _PREFETCH_SLOT_MB_MEASURED figure — no value is
    ever invented beyond that documented constant, and the substitution is
    reported loudly (same RU-note doctrine as the other anchor degradations).
    The clamp top is saturation_streams (capped at _PREFETCH_DEPTH_CAP);
    without one the cap applies directly. Budget: modern measured context
    canon (500 MB) in place of the legacy 750 MB fallback figure.
    """
    points = [p for p in anchor.get("streams_sweep", []) if isinstance(p, dict)]
    slot_mb = _prefetch_slot_from_sweep(points)
    if slot_mb is None:
        slot_mb = _PREFETCH_SLOT_MB_MEASURED
        if reporter is not None:
            reporter.note(
                f"Якорь не содержит полезных точек свипа (пусто/одна точка/нулевая дельта) — "
                f"стоимость слота префетча взята из канона "
                f"({_PREFETCH_SLOT_MB_MEASURED} МБ), а не из замера."
            )
    saturation = anchor.get("saturation_streams")
    max_depth = (
        min(saturation, _PREFETCH_DEPTH_CAP)
        if isinstance(saturation, int) and not isinstance(saturation, bool) and saturation >= 1
        else _PREFETCH_DEPTH_CAP
    )
    budget = (
        _BUDGET_FRACTION * vram_mb - _CONTEXT_RESERVE_MB_MEASURED - _UPSCALE_RESERVE_MB - _SLACK_MB
    )
    raw = int(budget // slot_mb)
    return max(1, min(max_depth, raw))


def _prefetch_slot_from_sweep(points: list[dict]) -> float | None:
    """Per-slot MB from the sweep's peak slope; None when un-derivable."""
    if len(points) >= _SLOPE_MIN_POINTS:
        first, last = points[0], points[-1]
        streams_first, streams_last = first.get("streams"), last.get("streams")
        peak_first, peak_last = first.get("vram_peak_mb"), last.get("vram_peak_mb")
        numeric = (
            isinstance(streams_first, int)
            and not isinstance(streams_first, bool)
            and isinstance(streams_last, int)
            and not isinstance(streams_last, bool)
            and isinstance(peak_first, (int, float))
            and not isinstance(peak_first, bool)
            and isinstance(peak_last, (int, float))
            and not isinstance(peak_last, bool)
        )
        if numeric and streams_last > streams_first:
            slope = (float(peak_last) - float(peak_first)) / (streams_last - streams_first)
            if slope > 0:
                return slope
    return None


def resolve_pools_from_anchor(anchor: dict, cores: int, active_encoders: int) -> int | None:
    """Measured x265 pool cap: min(saturation_pools, cores // encoders)."""
    saturation = anchor.get("saturation_pools")
    if not isinstance(saturation, int) or isinstance(saturation, bool) or saturation < 1:
        return None
    return min(saturation, max(1, cores // max(1, active_encoders)))


def _anchor_mismatch_reason(
    anchor: dict,
    gpu: "GPUInfo",
    driver: str | None,
    run_shape: "AnchorRunShape",
) -> str | None:
    """
    Strict invalidation of an anchor against the would-be run (plan §3).

    Any key mismatch → the anchor is ignored wholesale (fallback to
    constants); no partial degradation. Returns the first mismatch reason,
    or None when the anchor fully matches. driver=None → nvidia-smi
    unavailable: the driver check is SKIPPED (caller notes it loudly), the
    anchor still applies.
    """
    reasons: list[str] = []
    if anchor.get("gpu_name") != gpu.name:
        reasons.append(
            f"gpu_name не совпал: якорь для «{anchor.get('gpu_name')}», обнаружено «{gpu.name}»"
        )
    if driver is not None and str(anchor.get("driver")) != str(driver):
        reasons.append(f"driver не совпал: якорь для {anchor.get('driver')}, в системе {driver}")
    run_config = anchor.get("run_config")
    if not isinstance(run_config, dict):
        reasons.append("run_config отсутствует в якоре")
    else:
        if bool(run_config.get("trt")) != bool(run_shape.trt):
            reasons.append(
                f"run_config.trt не совпал: якорь {run_config.get('trt')}, "
                f"прогон бы выбрал {run_shape.trt}"
            )
        if run_config.get("interp_format") != run_shape.interp_format:
            reasons.append(
                f"run_config.interp_format не совпал: якорь «{run_config.get('interp_format')}», "
                f"прогон бы выбрал «{run_shape.interp_format}»"
            )
        if run_shape.preset is None:
            reasons.append("preset кодировщика не передан — строгая проверка run_config невозможна")
        elif run_config.get("preset") != run_shape.preset:
            reasons.append(
                f"run_config.preset не совпал: якорь «{run_config.get('preset')}», "
                f"в конфиге «{run_shape.preset}»"
            )
    return reasons[0] if reasons else None


@dataclass(frozen=True)
class AnchorRunShape:
    """The would-be run's anchor-relevant configuration (strict-invalidation keys)."""

    interp_format: str
    trt: bool
    preset: str | None


@dataclass(frozen=True)
class GPUInfo:
    """One detected CUDA device."""

    index: int
    name: str
    vram_mb: int
    compute_capability: tuple[int, int] = (0, 0)


@dataclass(frozen=True)
class HardwareProfile:
    """Frozen per-run capability snapshot consumed read-only by builders."""

    gpus: tuple[GPUInfo, ...]
    gpu_count: int
    cpu_logical_cores: int
    # VS prefetch depth (frames in flight per GPU), consumed at pipeline.py.
    # NOT GPU inference concurrency: vsrife 5.7.0 exposes no stream knob
    # (TRT path = one frame per engine call; PyTorch path = lock-serialized).
    num_streams_per_gpu: int
    encode_threads: int
    precision: str
    streams_source: str = _SOURCE_FORMULA
    encode_threads_source: str = _SOURCE_FORMULA

    @property
    def is_blackwell(self) -> bool:
        """
        True if compute capability >= (12, 0).

        The cc check is authoritative: name substrings ("RTX 50") false-positive
        on e.g. Quadro RTX 5000/6000 (Turing, cc 7.5) and would silently
        disable TRT on real hardware.
        """
        return any(gpu.compute_capability >= (12, 0) for gpu in self.gpus)

    @property
    def recommended_interp_format(self) -> str:
        """'rgbs' for Blackwell / fp16 overflow risk devices; 'rgbh' for others."""
        return "rgbs" if self.is_blackwell else "rgbh"

    @property
    def recommended_trt(self) -> bool:
        """False for Blackwell due to TRT 11 + RIFE compiler bug; True for older tested GPUs."""
        return not self.is_blackwell


def streams_for_vram(vram_mb: int) -> int:
    """
    Fallback prefetch depth (plan §9.2): clamp(floor((0.8·VRAM − reserves) / slot), 1, 4).

    The result feeds the VapourSynth output-loop prefetch only — it never
    reaches GPU inference concurrency, because vsrife 5.7.0 provides no
    stream knob (TRT: one frame per engine call; PyTorch: forwards
    serialized by locks). Constants are frozen; see the block comment above.
    """
    budget = _BUDGET_FRACTION * vram_mb - _CONTEXT_RESERVE_MB - _UPSCALE_RESERVE_MB - _SLACK_MB
    raw = int(budget // _PREFETCH_SLOT_BUDGET_MB)
    return max(_STREAM_CLAMP[0], min(_STREAM_CLAMP[1], raw))


_CGROUP_V2_CPU_MAX = Path("/sys/fs/cgroup/cpu.max")

# cpu.max is exactly "<quota> <period>"; anything else is malformed (loud fallback).
_CGROUP_CPU_MAX_FIELDS = 2
# x265 canon auto-table: ≥4 cores per encoder → ≥2 frame threads; below that
# preset medium starves (plan D1) — the loud-advise threshold.
_MIN_THREADS_PER_ENCODER = 4


def _cgroup_v2_cpu_quota(reporter: Reporter | None = None) -> int | None:
    """
    CFS CPU quota of this process's own cgroup v2, in whole CPUs (ceiling).

    ``/sys/fs/cgroup/cpu.max`` holds ``"<quota> <period>"`` in microseconds
    (``"200000 100000"`` = 2.0 CPUs) or ``"max <period>"`` when no quota
    applies. The ceiling matches coreutils ``nproc`` (gnulib round-up, the
    behavior thread of CPython gh-149452). Returns None when no quota
    applies or the file is absent (Windows/WSL2/bare metal). A malformed or
    non-positive quota is reported loudly through the reporter and treated
    as unlimited — detection never crashes on a bad cgroup file.

    Documented limitation: only the process's OWN cgroup v2 file is read.
    cgroup v1 hosts and quotas nested in parent cgroups are NOT resolved —
    container runtimes (Docker/Kaggle/Colab) place the limit in this file.
    """
    try:
        raw = _CGROUP_V2_CPU_MAX.read_text(encoding="utf-8").strip()
    except (OSError, ValueError):
        # ValueError covers UnicodeDecodeError: undecodable bytes = malformed file.
        return None
    parts = raw.split()
    if not parts or parts[0] == "max":
        return None
    if len(parts) == _CGROUP_CPU_MAX_FIELDS and all(part.isdigit() for part in parts):
        quota_us, period_us = int(parts[0]), int(parts[1])
        if quota_us > 0 and period_us > 0:
            return -(-quota_us // period_us)
    if reporter is not None:
        reporter.note(
            f"Не удалось разобрать {_CGROUP_V2_CPU_MAX} («{raw}») — "
            "CPU-квота контейнера не учтена, бюджет считается по числу ядер."
        )
    return None


def usable_cpu_count(reporter: Reporter | None = None) -> int:
    """
    CPUs actually usable by this process: affinity/quota-aware, at least 1.

    Basis ladder (first available API wins):
    1. ``os.process_cpu_count()`` (Python 3.13+; honors PYTHON_CPU_COUNT /
       ``-X cpu_count`` — the operator override) — reads the scheduler
       affinity mask;
    2. ``len(os.sched_getaffinity(0))`` (POSIX affinity; the documented
       usable-CPU recipe pre-3.13);
    3. ``os.cpu_count()`` (system logical CPUs; the only count API on
       Windows, which has no stdlib affinity getter).

    The basis is clamped by the own-cgroup v2 CFS quota: ``min(basis,
    ceil(quota))``. CPython gh-149452 (2026-05-06, still open) is the canon
    proof that the quota reader is mandatory: ``os.process_cpu_count()``
    reads affinity only, NOT cpu.max — under ``docker --cpus=2`` it still
    reports the host's logical cores (verbatim probe in the issue: a
    2-CPU container on a 20-core host prints 20), so the quota leg is what
    makes container encode budgets honest. No cgroup file (Windows/WSL2
    guest) → the basis stands as is; a quota clamp is reported loudly.
    """
    if hasattr(os, "process_cpu_count"):
        base = os.process_cpu_count()
    elif hasattr(os, "sched_getaffinity"):
        base = len(os.sched_getaffinity(0))
    else:
        base = os.cpu_count() or 1
    quota = _cgroup_v2_cpu_quota(reporter)
    if quota is not None and quota < base:
        if reporter is not None:
            reporter.note(
                f"Cgroup v2 CPU-квота {quota} ядра меньше доступных {base} — "
                "процессорный бюджет кодирования = квоте."
            )
        base = quota
    return max(1, base)


def _probe_torch_gpus() -> list[GPUInfo]:
    """Torch CUDA probe. torch is a hard dependency (vsrife/TensorRT) — no new deps."""
    try:
        import torch  # noqa: PLC0415 - lazy: encode-only runs must not load the torch stack
    except ImportError as exc:  # pragma: no cover - torch ships with the lockfile
        msg = f"PyTorch недоступен: {exc}"
        raise HardwareError(msg) from exc
    if not torch.cuda.is_available():
        return []
    return [
        GPUInfo(
            index=index,
            name=torch.cuda.get_device_properties(index).name,
            vram_mb=round(torch.cuda.get_device_properties(index).total_memory / (1024 * 1024)),
            compute_capability=torch.cuda.get_device_capability(index),
        )
        for index in range(torch.cuda.device_count())
    ]


def _nvidia_smi_cross_check(detected: list[GPUInfo]) -> None:
    """Diagnostics only: torch sees 0 GPUs but the driver sees N → name the mismatch."""
    try:
        # Diagnostic-only PATH lookup with fixed args (never a config path).
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return
    driver_gpus = [line.strip() for line in out.stdout.splitlines() if line.strip()]
    if not detected and driver_gpus:
        raise HardwareError(
            "torch не видит CUDA-устройства, но nvidia-smi показывает: "
            + ", ".join(driver_gpus)
            + ". Проверьте драйвер и соответствие версий CUDA/torch."
        )


def query_driver_version() -> str | None:
    """
    Best-effort driver version via nvidia-smi; None when unavailable.

    The caller then skips the anchor driver check with a loud note, never
    silently.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = [line.strip() for line in out.stdout.splitlines() if line.strip()]
    return lines[0] if lines else None


def _consult_anchor(  # noqa: PLR0913 - mirrors the precedence contract inputs
    *,
    anchor_path: Path,
    detected: list[GPUInfo],
    vram: int,
    cores: int,
    active_encoders: int,
    encode_preset: str | None,
    reporter: Reporter | None,
) -> tuple[int | None, int | None]:
    """
    Validate + match the anchor; return (measured_streams, measured_pools).

    (None, None) = the anchor is ignored (any reason is reported loudly).
    Measured values that could not be derived (e.g. no qualifying streams
    point) come back as None too — the caller keeps the formula value for
    that lever only. ``cores`` is the USABLE CPU budget (usable_cpu_count),
    not the system size — the measured division must share the formula's
    basis.
    """
    anchor = load_anchor(anchor_path, reporter)
    if anchor is None:
        return None, None
    problems = validate_anchor(anchor)
    if problems:
        if reporter is not None:
            for problem in problems:
                reporter.note(f"Якорь {anchor_path} проигнорирован: {problem}")
        return None, None
    blackwell = any(gpu.compute_capability >= (12, 0) for gpu in detected)
    driver = query_driver_version()
    if driver is None and reporter is not None:
        reporter.note(
            "Не удалось определить версию драйвера (nvidia-smi) — "
            "проверка якоря по драйверу пропущена."
        )
    run_shape = AnchorRunShape(
        interp_format="rgbs" if blackwell else "rgbh",
        trt=not blackwell,
        preset=encode_preset,
    )
    reason = _anchor_mismatch_reason(anchor, detected[0], driver, run_shape)
    if reason is not None:
        if reporter is not None:
            reporter.note(f"Якорь {anchor_path} проигнорирован: {reason} — откат на формулу.")
        return None, None
    measured_streams = resolve_prefetch_from_anchor(anchor, vram, reporter)
    measured_pools = resolve_pools_from_anchor(anchor, cores, active_encoders)
    return measured_streams, measured_pools


def detect_hardware(  # noqa: PLR0913 - optional anchor tiers mirror the precedence contract
    overrides: HardwareOverrides,
    cli_num_streams: int | None,
    *,
    nn_stages_selected: bool,
    active_encoders: int = 1,
    project_root: Path | None = None,
    encode_preset: str | None = None,
    reporter: Reporter | None = None,
) -> HardwareProfile:
    """
    Build the frozen profile once per run; builders consume it read-only.

    Anchor tiers: CLI/[hardware] override > measured anchor (only when
    project_root is wired AND a schema-valid anchor matches gpu_name, driver
    and the would-be run_config) > VRAM formula constants. No wiring, no
    anchor file, or any mismatch → byte-identical formula behavior.
    """
    detected = _probe_torch_gpus()
    if overrides.gpus is not None:
        by_index = {gpu.index: gpu for gpu in detected}
        missing = [index for index in overrides.gpus if index not in by_index]
        if missing:
            names = ", ".join(str(gpu.index) for gpu in detected) or "<нет>"
            msg = f"Заданные в [hardware] gpus отсутствуют: {missing}. Обнаружены GPU: {names}."
            raise HardwareError(msg)
        detected = [by_index[index] for index in overrides.gpus]
    if nn_stages_selected:
        _nvidia_smi_cross_check(detected)
        if not detected:
            msg = (
                "Выбраны стадии нейросетевого инференса (upscale/interpolate), но CUDA-устройств "
                "не обнаружено. Проверьте драйвер, либо выберите подмножество без NN-стадий, "
                "например: --stages descale,deband,encode."
            )
            raise HardwareError(msg)
    vram = detected[0].vram_mb if detected else 0
    detected_streams = streams_for_vram(vram) if detected else 0
    has_cli_override = cli_num_streams is not None
    has_config_override = overrides.num_streams is not None
    num_streams = (
        cli_num_streams
        if has_cli_override
        else (overrides.num_streams if has_config_override else detected_streams)
    )
    streams_source = (
        _SOURCE_CONFIG if (has_cli_override or has_config_override) else _SOURCE_FORMULA
    )
    usable = usable_cpu_count(reporter)
    # Informational only (banner/diagnosis): system size, NOT the encode budget.
    cores = os.cpu_count() or 1
    has_pools_override = overrides.encode_threads is not None
    encode_threads = (
        overrides.encode_threads
        if has_pools_override
        else max(1, usable // max(1, active_encoders))
    )
    if (
        not has_pools_override
        and encode_threads < _MIN_THREADS_PER_ENCODER
        and reporter is not None
    ):
        reporter.note(
            f"На энкодер приходится {encode_threads} поток(а) из {usable} доступных CPU "
            f"({active_encoders} параллельных энкодеров) — x265 на preset medium голодает "
            "ниже 4 потоков на энкодер; уменьшите число параллельных энкодеров "
            "или задайте [hardware] encode_threads явно."
        )
    encode_threads_source = _SOURCE_CONFIG if has_pools_override else _SOURCE_FORMULA

    # Measured-anchor tier: consulted only when wired (project_root) and the
    # machine actually has the GPU that would run NN stages. The anchor file's
    # existence gates ALL anchor work, including the nvidia-smi driver query —
    # no extra subprocess on anchor-less machines.
    if (
        project_root is not None
        and detected
        and nn_stages_selected
        and not (has_cli_override or has_config_override)
    ):
        anchor_path = (
            Path(project_root) / ANCHOR_FILENAME / gpu_slug(detected[0].name) / "anchor.json"
        )
        if anchor_path.is_file():
            measured_streams, measured_pools = _consult_anchor(
                anchor_path=anchor_path,
                detected=detected,
                vram=vram,
                cores=usable,
                active_encoders=active_encoders,
                encode_preset=encode_preset,
                reporter=reporter,
            )
            if measured_streams is not None:
                num_streams = measured_streams
                streams_source = _SOURCE_MEASURED
            if measured_pools is not None:
                encode_threads = measured_pools
                encode_threads_source = _SOURCE_MEASURED

    return HardwareProfile(
        gpus=tuple(detected),
        gpu_count=len(detected),
        cpu_logical_cores=cores,
        num_streams_per_gpu=num_streams,
        encode_threads=encode_threads,
        precision="fp16",
        streams_source=streams_source,
        encode_threads_source=encode_threads_source,
    )


def format_gpu_summary(profile: HardwareProfile) -> str:
    """«NVIDIA GeForce RTX 5080 ×1, 16 ГБ VRAM» / «… ×2 (16+16 ГБ)» (plan §9.6)."""
    if not profile.gpus:
        return "GPU отсутствуют (CPU-режим)"
    names = {gpu.name for gpu in profile.gpus}
    name = " / ".join(sorted(names))
    vrams = "+".join(str(round(gpu.vram_mb / 1024)) for gpu in profile.gpus)
    total_gb = round(sum(gpu.vram_mb for gpu in profile.gpus) / 1024)
    if len(profile.gpus) == 1:
        return f"{name} ×1, {total_gb} ГБ VRAM"
    return f"{name} ×{len(profile.gpus)} ({vrams} ГБ)"


def print_banner(reporter: Reporter, profile: HardwareProfile, stages: tuple[str, ...]) -> None:
    """
    Print the §9.6 run banner (the only place detection results are shown by default).

    Honesty rule (plan §3): the banner always names the regime that produced
    each lever — «замер» (measured anchor), «формула» (VRAM constants),
    «конфиг» (explicit override).
    """
    if profile.gpus:
        source_note = {
            _SOURCE_MEASURED: f"(замер: benchmarks/{gpu_slug(profile.gpus[0].name)}/anchor.json)",
            _SOURCE_CONFIG: "(конфиг)",
            _SOURCE_FORMULA: "(формула VRAM)",
        }[profile.streams_source]
        streams = f"{profile.num_streams_per_gpu} поток(а)/GPU {source_note}"
        encode_source = {
            _SOURCE_MEASURED: "(замер)",
            _SOURCE_CONFIG: "(конфиг)",
            _SOURCE_FORMULA: "(формула)",
        }[profile.encode_threads_source]
        encode = f"{profile.encode_threads} {encode_source}"
    else:
        streams = "без GPU"
        encode = str(profile.encode_threads)
    reporter.hardware_banner(
        gpus=format_gpu_summary(profile),
        streams=streams,
        stages=" → ".join(stages),
        encode=encode,
    )
    if profile.is_blackwell and profile.gpus:
        target_gpu = profile.gpus[0]
        major, minor = target_gpu.compute_capability
        reporter.note(
            f"Детект GPU: {target_gpu.name} (Blackwell sm_{major}{minor}) → "
            "автоматический выбор: FP32 (RGBS) без TensorRT "
            "для предотвращения глитчей/сеток в RIFE."
        )
