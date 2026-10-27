"""
Hardware detection → frozen HardwareProfile (plan §9).

Detection is a capability probe, never a config source: the single source of
truth stays config.toml. Override precedence: CLI > [hardware] > detected >
built-in default.
"""

import os
import subprocess
from dataclasses import dataclass

from anime_upscale.config import HardwareOverrides
from anime_upscale.reporting import Reporter

# VRAM budget constants (plan §9.2). Calibrated so the plan's stated anchors
# hold exactly: 16 GB → 2 streams/GPU, 24 GB → 3 streams/GPU. The first
# benchmarks/ entry refines these (single tunable block, by design).
_BUDGET_FRACTION = 0.8
_CONTEXT_RESERVE_MB = 750
_UPSCALE_RESERVE_MB = 2500
_SLACK_MB = 1024
_RIFE_STREAM_BUDGET_MB = 4096
_STREAM_CLAMP = (1, 4)


class HardwareError(Exception):
    """Loud hardware-prerequisite failure (Russian message)."""


@dataclass(frozen=True)
class GPUInfo:
    """One detected CUDA device."""

    index: int
    name: str
    vram_mb: int


@dataclass(frozen=True)
class HardwareProfile:
    """Frozen per-run capability snapshot consumed read-only by builders."""

    gpus: tuple[GPUInfo, ...]
    gpu_count: int
    cpu_logical_cores: int
    num_streams_per_gpu: int
    encode_threads: int
    precision: str


def streams_for_vram(vram_mb: int) -> int:
    """VRAM budget math (plan §9.2): clamp(floor((0.8·VRAM − reserves) / stream), 1, 4)."""
    budget = _BUDGET_FRACTION * vram_mb - _CONTEXT_RESERVE_MB - _UPSCALE_RESERVE_MB - _SLACK_MB
    raw = int(budget // _RIFE_STREAM_BUDGET_MB)
    return max(_STREAM_CLAMP[0], min(_STREAM_CLAMP[1], raw))


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


def detect_hardware(
    overrides: HardwareOverrides,
    cli_num_streams: int | None,
    *,
    nn_stages_selected: bool,
    active_encoders: int = 1,
) -> HardwareProfile:
    """Build the frozen profile once per run; builders consume it read-only."""
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
    num_streams = (
        cli_num_streams
        if cli_num_streams is not None
        else (overrides.num_streams if overrides.num_streams is not None else detected_streams)
    )
    cores = os.cpu_count() or 1
    encode_threads = (
        overrides.encode_threads
        if overrides.encode_threads is not None
        else max(1, cores // max(1, active_encoders))
    )
    return HardwareProfile(
        gpus=tuple(detected),
        gpu_count=len(detected),
        cpu_logical_cores=cores,
        num_streams_per_gpu=num_streams,
        encode_threads=encode_threads,
        precision="fp16",
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
    """Print the §9.6 run banner (the only place detection results are shown by default)."""
    streams = f"{profile.num_streams_per_gpu} поток(а)/GPU" if profile.gpus else "без GPU"
    reporter.hardware_banner(
        gpus=format_gpu_summary(profile),
        streams=streams,
        stages=" → ".join(stages),
        encode=str(profile.encode_threads),
    )
