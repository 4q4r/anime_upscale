"""
Deband stage: vsdeband.f3k_deband (JET's f3kdb wrapper over the vszip plugin).

Runs on YUV at source resolution, before upscale — community convention
(deband before interpolation, research §1). NOTE: the plan's engine reference
("HolyWu's vs-deband") has moved to Jaded-Encoding-Thaumaturgy; the current
PyPI/git distribution exposes the f3kdb semantics as ``f3k_deband`` (backed by
the maintained vszip/neo-f3kdb plugin, an vsdeband dependency). The PyPI 1.3.3
vsdeband distribution itself is an empty stub, so the dependency resolves from
the upstream git repository.
"""

import vapoursynth as vs

from anime_upscale.stages import StageContext, register


@register("deband")
def apply(ctx: StageContext, clip: vs.VideoNode) -> vs.VideoNode:
    """Apply f3kdb deband with the configured radius/threshold/grain."""
    import vsdeband

    cfg = ctx.run.deband
    return vsdeband.f3k_deband(
        clip,
        radius=cfg.radius,
        thr=list(cfg.threshold),
        grain=[float(g) for g in cfg.grain],
    )
