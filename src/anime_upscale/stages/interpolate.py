"""
Interpolation stage: misc.SCDetect + vsrife.rife (plan §5).

vsrife 5.7.0, model 4.25: fps_num/fps_den give the ABSOLUTE target fps in one
pass (23.976 → 59.94 needs ×2.5 — impossible by stacking ×2, natively supported).
Weights are wrapper-managed (auto-download into vsrife/models); TRT engines
cache under the configured trt_cache_dir.

`num_streams_per_gpu` feeds VS prefetch depth, not GPU inference concurrency:
vsrife 5.7.0 exposes no stream knob (TRT path executes one frame per engine
call; the PyTorch path serializes forwards behind per-phase locks — one
forward in flight). See the verdict notes in anime_upscale.hardware.
"""

from pathlib import Path

import vapoursynth as vs

from anime_upscale.stages import StageContext, register


@register("interpolate")
def apply(ctx: StageContext, clip: vs.VideoNode) -> vs.VideoNode:
    """Run vsrife RIFE 4.25 to the configured rational target fps."""
    if not ctx.hardware.gpus:
        msg = (
            "Стадия interpolate требует CUDA-устройства: vsrife/TensorRT работают только на GPU. "
            "Исключите стадию из прогона или проверьте драйвер."
        )
        raise RuntimeError(msg)
    from vsrife import rife

    cfg = ctx.run.interpolate
    # Scene detection runs INSIDE vsrife (sc=True + sc_threshold): it converts to GRAY8
    # on a side-branch and copies _SceneChangeNext/_SceneChangePrev back — the canonical
    # misc.SCDetect semantics without double detection.
    shape = [clip.height, clip.width]
    device_index = ctx.hardware.gpus[0].index if ctx.hardware.gpus else 0
    cache_dir = Path(cfg.trt_cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)  # vsrife requires an existing dir

    use_trt = cfg.trt
    if use_trt and not ctx.hardware.recommended_trt:
        ctx.reporter.warn(
            "Обнаружена несовместимость TensorRT с текущей архитектурой GPU (Blackwell). "
            "TensorRT отключен, используется безопасный режим PyTorch CUDA."
        )
        use_trt = False

    return rife(
        clip,
        device_index=device_index,
        model=cfg.model,
        auto_download=True,
        fps_num=cfg.target_fps[0],
        fps_den=cfg.target_fps[1],
        sc=cfg.scene_detection,
        sc_threshold=cfg.scene_threshold if cfg.scene_detection else None,
        trt=use_trt,
        trt_static_shape=True,
        trt_opt_shape=shape,
        trt_cache_dir=str(cache_dir),
    )
