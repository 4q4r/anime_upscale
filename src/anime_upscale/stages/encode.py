"""
stages/encode.py — libx265 video arg-builder (owner decision: CPU-only encoding).

Flag sheet: research/05-encode-flags.md; config contract: architecture plan §4.
keyint/min-keyint follow the «10 s / 1 s» rule computed from the ACTUAL output fps.
"""

from math import ceil

from anime_upscale.config import EncodeConfig


def compute_keyint(output_fps: tuple[int, int]) -> tuple[int, int]:
    """
    (keyint, min-keyint) for «10 s / 1 s» of content at the output fps.

    keyint rounds up (24000/1001 → 240, 60000/1001 → 600); min-keyint rounds to
    nearest (23.976 → 24, 59.94 → 60) — matching the plan's stated anchors.
    """
    num, den = output_fps
    keyint = ceil(num * 10 / den)
    min_keyint = (2 * num + den) // (2 * den)
    return (keyint, min_keyint)


def _x265_param_str(params: tuple[tuple[str, str], ...]) -> str:
    return ":".join(f"{key}={value}" for key, value in params)


def build_video_args(
    cfg: EncodeConfig,
    output_fps: tuple[int, int],
    threads: int | None = None,
) -> list[str]:
    """Ffmpeg video args for the encode stage (everything between -i pipe and output)."""
    args: list[str] = [
        "-c:v",
        "libx265",
        "-preset",
        cfg.preset,
        "-crf",
        str(cfg.crf),
        "-pix_fmt",
        cfg.pix_fmt,
        "-profile:v",
        cfg.profile,
        "-colorspace",
        cfg.colorspace,
        "-color_primaries",
        cfg.color_primaries,
        "-color_trc",
        cfg.color_trc,
        "-colorrange",
        cfg.color_range,
    ]
    if threads is not None:
        args += ["-threads", str(threads)]

    keyint, min_keyint = compute_keyint(output_fps)
    overrides = dict(cfg.x265_params)
    # Manual override in [stages.encode.x265_params] wins — computed values not duplicated.
    sheet = list(overrides.items()) + [
        (k, v)
        for k, v in (("keyint", str(keyint)), ("min-keyint", str(min_keyint)))
        if k not in overrides
    ]
    args += ["-x265-params", _x265_param_str(tuple(sorted(sheet)))]
    return args
