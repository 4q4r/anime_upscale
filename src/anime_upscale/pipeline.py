"""
Graph assembly: validate subset → build VS nodes in canonical order → y4m → ffmpeg (plan §1).

Format flow: RGBH (fp16) around the upscale/interpolate segment, one conversion
to YUV420P10 (BT.709) right before the pipe. Interp-only runs convert to RGBH
once before RIFE (no RGB round-trips). Scene detection runs inside vsrife
(sc=True + sc_threshold) on a GRAY8 side-branch — requires the misc plugin.
"""

from collections.abc import Callable
from fractions import Fraction
from pathlib import Path

import vapoursynth as vs

from anime_upscale.ffmpeg import build_mux_args, finish_encoder, spawn_encoder
from anime_upscale.stages import (  # noqa: F401 (registry import)  # noqa: F401
    STAGE_BUILDERS,
    StageContext,
    deband,
    descale,
    encode,
    get_builder,
    interpolate,
    upscale,
)
from anime_upscale.stages.encode import build_video_args


class PipelineError(Exception):
    """Loud pipeline failure (Russian message)."""


def _to_rgbh(clip: vs.VideoNode) -> vs.VideoNode:
    return clip.resize.Bilinear(format=vs.RGBH, matrix_in=vs.MATRIX_BT709)  # ty: ignore[unresolved-attribute]


def build_graph(ctx: StageContext, clip: vs.VideoNode) -> vs.VideoNode:
    """Assemble the selected stage subset over the decoded clip (canonical order)."""
    for name in ctx.run.stages:
        if name == "encode":
            continue  # encode is the subprocess sink, not a VS node
        if name == "upscale" or (name == "interpolate" and "upscale" not in ctx.run.stages):
            clip = _to_rgbh(clip)
        clip = get_builder(name)(ctx, clip)
    return clip.resize.Bicubic(format=vs.YUV420P10, matrix=vs.MATRIX_BT709)  # ty: ignore[unresolved-attribute]


def check_target_fps(source_fps: Fraction, target_fps: tuple[int, int]) -> None:
    """§9.5 sanity gate: fail fast with an actionable message, not a Torch traceback."""
    target = Fraction(target_fps[0], target_fps[1])
    if source_fps <= 0:
        msg = f"Не удалось определить fps источника (получено {source_fps})."
        raise PipelineError(msg)
    if target <= source_fps:
        msg = (
            f"Целевой fps ({target_fps[0]}/{target_fps[1]} ≈ {float(target):.3f}) не выше "
            f"исходного ({source_fps} ≈ {float(source_fps):.3f}). "
            "Интерполяция повышает частоту кадров; уменьшите --target-fps "
            "или исключите стадию interpolate."
        )
        raise PipelineError(msg)


def run_pipeline(
    ctx: StageContext,
    input_path: Path,
    output_path: Path,
    ffmpeg_path: str,
    progress_update: Callable[[int, int], None] | None = None,
) -> tuple[int, int]:
    """Run the full chain for ONE input file; returns the output fps as (num, den)."""
    core = vs.core
    try:
        clip = core.bs.VideoSource(str(input_path))  # ty: ignore[unresolved-attribute]
    except Exception as exc:
        msg = f"Не удалось декодировать {input_path}: {exc}"
        raise PipelineError(msg) from exc

    source_fps = Fraction(clip.fps)
    if "interpolate" in ctx.run.stages:
        check_target_fps(source_fps, ctx.run.interpolate.target_fps)

    graph = build_graph(ctx, clip)
    output_fps = Fraction(graph.fps)
    video_args = build_video_args(
        ctx.run.encode,
        (output_fps.numerator, output_fps.denominator),
        threads=ctx.hardware.encode_threads,
    )
    args = build_mux_args(input_path, output_path, video_args, ffmpeg=ffmpeg_path)

    encoder = spawn_encoder(args)
    if encoder.stdin is None:
        msg = "ffmpeg не открыл stdin для y4m-потока (неожиданный сбой Popen)."
        raise PipelineError(msg)
    try:
        try:
            graph.output(
                encoder.stdin,  # ty: ignore[invalid-argument-type] - PIPE is IO[bytes]
                y4m=True,
                prefetch=max(1, ctx.hardware.num_streams_per_gpu) if ctx.hardware.gpus else 0,
                progress_update=progress_update,
            )
        except OSError as exc:
            # A dead ffmpeg surfaces as EPIPE on the VS side; re-raise with the tail.
            finish_encoder(encoder)
            msg = f"Поток в ffmpeg оборвался во время кодирования: {exc}"
            raise PipelineError(msg) from exc
    finally:
        if encoder.stdin is not None and not encoder.stdin.closed:
            encoder.stdin.close()
    finish_encoder(encoder)
    return (output_fps.numerator, output_fps.denominator)
