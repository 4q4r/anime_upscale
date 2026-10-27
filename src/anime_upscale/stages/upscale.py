"""
Upscale stage: spandrel ModelLoader (fp16) inside std.ModifyFrame (plan §1/§5).

The sha256-pinned .pth is loaded verbatim — no ONNX conversion. The model runs
per-frame inside the same VS graph, keeping prefetch overlap with decode and
the interpolate stage.
"""

import numpy as np
import torch
import vapoursynth as vs

from anime_upscale.models.manifest import resolve_model_path
from anime_upscale.stages import StageContext, register


@register("upscale")
def apply(ctx: StageContext, clip: vs.VideoNode) -> vs.VideoNode:
    """Run the pinned spandrel model per-frame inside std.ModifyFrame."""
    if clip.format.id != vs.RGBH:  # pragma: no cover - pipeline guarantees RGBH here
        msg = f"Стадия upscale ожидает RGBH на входе, получен формат {clip.format.name}."
        raise RuntimeError(msg)
    model_path = resolve_model_path(
        manifest_path=ctx.models_manifest,
        models_dir=ctx.models_dir,
        name=ctx.run.upscale.model,
        verify=True,
    )
    device = torch.device(f"cuda:{ctx.hardware.gpus[0].index}" if ctx.hardware.gpus else "cpu")
    if not ctx.hardware.gpus:  # pragma: no cover - zero-GPU gate runs before this
        msg = "Стадия upscale требует CUDA-устройства."
        raise RuntimeError(msg)
    from spandrel import ImageModelDescriptor, ModelLoader

    loaded = ModelLoader(device=device).load_from_file(str(model_path)).eval().to(device)
    if not isinstance(loaded, ImageModelDescriptor):
        msg_0 = (
            f"Модель «{ctx.run.upscale.model}» не является простой image-моделью spandrel "
            "(mask/видео-дескрипторы в v1 не поддерживаются)."
        )
        raise TypeError(msg_0)  # wrong spandrel descriptor type, not a runtime state error
    model = loaded
    if ctx.run.upscale.fp16:
        model.model.half()
    scale = model.scale

    core = vs.core
    out_width, out_height = clip.width * scale, clip.height * scale
    canvas = core.std.BlankClip(clip, width=out_width, height=out_height)  # ty: ignore[unresolved-attribute]

    def selector(_n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
        src, dst_template = f[1], f[0]
        planes = [np.asarray(src.get_read_array(i)) for i in range(3)]  # ty: ignore[unresolved-attribute]
        tensor = torch.from_numpy(np.stack(planes)).to(device=device, non_blocking=True)
        with torch.inference_mode():
            result = model(tensor).clamp_(0.0, 1.0)
        out = dst_template.copy()
        for plane in range(3):
            np.copyto(
                np.asarray(out.get_write_array(plane)),  # ty: ignore[unresolved-attribute]
                result[plane].cpu().numpy(),
            )
        for key in src.props:
            out.props[key] = src.props[key]
        return out

    # std.ModifyFrame + frame arrays are real VS R81 APIs (exercised by tests);
    # the shipped vapoursynth.pyi does not declare runtime-registered members.
    return core.std.ModifyFrame(canvas, [canvas, clip], selector)  # ty: ignore[unresolved-attribute]
