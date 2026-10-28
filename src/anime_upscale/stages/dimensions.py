"""
Mod-2 dimension guard (x265 yuv420 requires even width/height).

Arbitrary sources can carry odd dimensions; libx265 refuses odd sizes with the
misleading «maybe incorrect parameters such as bit_rate, rate, width or height».
The guard fixes the clip BEFORE the encode stage: crop when the total deficit
is within the 2px budget (minimal, visually lossless), scale otherwise.

Format note: a 1px crop is only expressible on non-subsampled formats (RGB/444/
GRAY). On subsampled YUV (420/422) VapourSynth rounds right/bottom crops UP to
the chroma granularity, which would leave odd dimensions — the scale branch is
the only minimal fix there.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from anime_upscale.reporting import Reporter

if TYPE_CHECKING:
    import vapoursynth as vs

AlignAction = Literal["none", "crop", "scale"]

_DEFAULT_MAX_CROP_PX = 2


@dataclass(frozen=True)
class EvenAlign:
    """Resolved mod-2 alignment plan for one clip."""

    action: AlignAction
    width: int
    height: int
    crop_right: int = 0
    crop_bottom: int = 0

    def __post_init__(self) -> None:
        """Reject action values outside the closed {none, crop, scale} set."""
        if self.action not in ("none", "crop", "scale"):
            msg = f"Недопустимое действие выравнивания: {self.action!r}"
            raise ValueError(msg)


def decide_even_align(
    width: int, height: int, max_crop_px: int = _DEFAULT_MAX_CROP_PX
) -> EvenAlign:
    """
    Choose the minimal mod-2 alignment for a clip.

    crop: shave the odd remainder off the right/bottom edges when the total
    (≤1px per axis, ≤2px by default) stays within budget; scale: floor to even
    dimensions when it does not (never upscales).
    """
    crop_right, crop_bottom = width % 2, height % 2
    if crop_right == 0 and crop_bottom == 0:
        return EvenAlign(action="none", width=width, height=height)
    if crop_right + crop_bottom <= max_crop_px:
        return EvenAlign(
            action="crop",
            width=width - crop_right,
            height=height - crop_bottom,
            crop_right=crop_right,
            crop_bottom=crop_bottom,
        )
    return EvenAlign(
        action="scale",
        width=width - crop_right,
        height=height - crop_bottom,
    )


def apply_even_align(clip: "vs.VideoNode", reporter: Reporter) -> "vs.VideoNode":
    """
    Apply the mod-2 guard before the final YUV420P10 conversion (pipeline call).

    On subsampled formats even the in-budget deficit must go through resize:
    a 1px crop there is not expressible (chroma granularity, see module doc).
    """
    plan = decide_even_align(clip.width, clip.height)
    if plan.action == "none":
        return clip

    subsampled = clip.format.color_family.name == "YUV" and (
        clip.format.subsampling_w > 0 or clip.format.subsampling_h > 0
    )
    reason = ""
    if plan.action == "scale":
        reason = " (обрезка превысила бы бюджет 2px)."
    elif subsampled:
        reason = " (обрезка 1px невозможна на субсэмплированном формате — гранулярность хромы)."

    if plan.action == "crop" and not subsampled:
        aligned = clip.std.Crop(  # ty: ignore[unresolved-attribute]
            right=plan.crop_right, bottom=plan.crop_bottom
        )
        reporter.note(
            f"Заметка: нечётные размеры {clip.width}x{clip.height} несовместимы с "
            f"yuv420 (x265): обрезка {plan.crop_right + plan.crop_bottom}px → "
            f"{plan.width}x{plan.height}."
        )
        return aligned

    aligned = clip.resize.Bicubic(  # ty: ignore[unresolved-attribute]
        width=plan.width, height=plan.height
    )
    reporter.note(
        f"Заметка: нечётные размеры {clip.width}x{clip.height} несовместимы с yuv420 "
        f"(x265): масштабирование → {plan.width}x{plan.height}{reason}"
    )
    return aligned
