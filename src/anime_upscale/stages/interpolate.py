"""
Interpolation stage: misc.SCDetect + vsrife.rife (plan §5).

vsrife 5.7.0, model 4.25: fps_num/fps_den give the ABSOLUTE target fps in one
pass (23.976 → 59.94 needs ×2.5 — impossible by stacking ×2, natively supported).
Weights are wrapper-managed (auto-download into vsrife/models); TRT engines
cache under the configured trt_cache_dir.
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
    return rife(
        clip,
        device_index=device_index,
        model=cfg.model,
        auto_download=True,
        fps_num=cfg.target_fps[0],
        fps_den=cfg.target_fps[1],
        sc=cfg.scene_detection,
        sc_threshold=cfg.scene_threshold if cfg.scene_detection else None,
        trt=cfg.trt,
        trt_static_shape=True,
        trt_opt_shape=shape,
        trt_cache_dir=str(Path(cfg.trt_cache_dir).expanduser()),
    )
