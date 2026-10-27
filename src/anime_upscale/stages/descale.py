"""
Descale stage: core.descale per-kernel deconvolution to a native-res height (plan §5).

NOTE: the plan references ``core.descale.Decompile`` — the original
Irrational-Encoding-Wizardry API. The maintained plugin (vapoursynth-descale 12,
JET) exposes per-kernel functions instead; this stage maps the config kernel
names onto them. Width follows the source aspect ratio, rounded to even.
"""

import vapoursynth as vs

from anime_upscale.stages import StageContext, register

_KERNELS = {
    "bicubic": "Debicubic",
    "bilinear": "Debilinear",
}


@register("descale")
def apply(ctx: StageContext, clip: vs.VideoNode) -> vs.VideoNode:
    """Deconvolve (descale) to the configured native-res target height."""
    core = vs.core
    if not hasattr(core, "descale"):
        msg = (
            "Плагин descale не найден (нужен пакет vapoursynth-descale). "
            "Установите зависимости: uv sync."
        )
        raise RuntimeError(msg)
    cfg = ctx.run.descale
    kernel_name = _KERNELS.get(cfg.kernel)
    if kernel_name is None:
        supported = ", ".join(sorted(_KERNELS))
        msg = f"Неизвестное ядро descale «{cfg.kernel}». Поддерживаются: {supported}."
        raise RuntimeError(msg)
    height = ctx.run.descale_height
    if height is None:  # pragma: no cover - resolve_run rejects this earlier
        msg = "Стадии descale не задана целевая высота."
        raise RuntimeError(msg)
    # Width follows the source aspect; even-rounding keeps chroma alignment.
    width = (round(clip.width * height / clip.height) + 1) // 2 * 2
    dekernel = getattr(core.descale, kernel_name)
    # The plugin requires float32 input; 444 float is the standard descale workspace.
    work = clip.resize.Point(format=vs.YUV444PS)  # ty: ignore[unresolved-attribute]
    return dekernel(work, width, height, src_left=cfg.src_left, src_top=cfg.src_top)
