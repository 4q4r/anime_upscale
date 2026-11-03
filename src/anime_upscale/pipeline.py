"""
Graph assembly: validate subset → build VS nodes in canonical order → rawvideo → ffmpeg (plan §1).

Format flow: RGBH (fp16) around the upscale/interpolate segment, one conversion
to YUV420P10 (BT.709) right before the pipe. Interp-only runs convert to RGBH
once before RIFE (no RGB round-trips). Scene detection runs inside vsrife
(sc=True + sc_threshold) on a GRAY8 side-branch — requires the misc plugin.
"""

import contextlib
import subprocess
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path

import vapoursynth as vs

from anime_upscale import ffmpeg_repair
from anime_upscale.ffmpeg import build_mux_args, finish_encoder, spawn_encoder
from anime_upscale.ffmpeg_repair import verdict_wants_reinstall
from anime_upscale.stages import (  # noqa: F401 - descale/deband/interpolate/upscale register via import
    STAGE_BUILDERS,
    StageContext,
    deband,
    descale,
    encode,
    get_builder,
    interpolate,
    upscale,
)
from anime_upscale.stages.dimensions import apply_even_align
from anime_upscale.stages.encode import build_video_args
from anime_upscale.tools import ToolError, resolve_tool


class PipelineError(Exception):
    """Loud pipeline failure (Russian message)."""


def _to_rgb(
    clip: vs.VideoNode,
    target_format: int = vs.RGBH,
    matrix: int = vs.MATRIX_BT709,
) -> vs.VideoNode:
    return clip.resize.Bilinear(format=target_format, matrix_in=matrix)  # ty: ignore[unresolved-attribute]


def build_graph(
    ctx: StageContext,
    clip: vs.VideoNode,
    matrix: int = vs.MATRIX_BT709,
) -> vs.VideoNode:
    """
    Assemble the selected stage subset over the decoded clip (canonical order).

    ``matrix`` is the source color-space matrix (detected dynamically from the
    container, fallback BT.709) and drives the YUV→RGB conversion so 10-bit
    sources keep correct chroma placement instead of being forced to BT.709.
    """
    for name in ctx.run.stages:
        if name == "encode":
            continue  # encode is the subprocess sink, not a VS node
        if name == "upscale":
            clip = _to_rgb(clip, target_format=vs.RGBH, matrix=matrix)
        elif name == "interpolate" and "upscale" not in ctx.run.stages:
            interp_format = vs.RGBS if ctx.hardware.recommended_interp_format == "rgbs" else vs.RGBH
            clip = _to_rgb(clip, target_format=interp_format, matrix=matrix)
        clip = get_builder(name)(ctx, clip)
    # Mod-2 guard BEFORE the final convert: on non-subsampled formats the 1px
    # crop is exact; on subsampled YUV the chroma granularity forces the scale
    # branch (handled inside). x265 yuv420 refuses odd sizes with a misleading
    # parameter error.
    clip = apply_even_align(clip, ctx.reporter)
    if clip.format.color_family == vs.RGB:
        # RGB → YUV needs the matrix; YUV → YUV depth conversion must NOT pass one
        # (zimg has no same-family path with a matrix — live-verified in benchmark).
        return clip.resize.Bicubic(format=vs.YUV420P10, matrix=vs.MATRIX_BT709)  # ty: ignore[unresolved-attribute]
    return clip.resize.Bicubic(format=vs.YUV420P10)  # ty: ignore[unresolved-attribute]


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


def _detect_source_matrix(input_path: str | Path) -> int:
    """
    Ffprobe the source color_space; map to a vs.MATRIX_* constant, fallback BT.709.

    Ffprobe reports e.g. bt709/smpte170m/bt2020nc/unknown. Unknown or
    unprobeable sources degrade gracefully to BT.709 — never silently wrong.
    """
    try:
        ffprobe = resolve_tool("ffprobe")
    except ToolError:
        return vs.MATRIX_BT709
    matrix_map: dict[str, int] = {
        "bt709": vs.MATRIX_BT709,
        "smpte170m": vs.MATRIX_ST170_M,
        "smpte240m": vs.MATRIX_ST240_M,
        "bt470bg": vs.MATRIX_BT470_BG,
        "bt470m": vs.MATRIX_BT470_BG,
        "bt2020nc": vs.MATRIX_BT2020_NCL,
        "bt2020cl": vs.MATRIX_BT2020_CL,
        "chromaticity-derived-ncl": vs.MATRIX_CHROMATICITY_DERIVED_NCL,
        "chromaticity-derived-cl": vs.MATRIX_CHROMATICITY_DERIVED_CL,
        "ictcp": vs.MATRIX_ICTCP,
    }
    try:
        completed = subprocess.run(  # noqa: S603 - ffprobe path from resolve_tool; input_path is a local file
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=color_space",
                "-of",
                "csv=p=0",
                str(input_path),
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=15,
        )
    except (subprocess.TimeoutExpired, OSError):
        return vs.MATRIX_BT709
    raw = completed.stdout.strip().lower()
    return matrix_map.get(raw, vs.MATRIX_BT709)


def run_pipeline(  # noqa: PLR0913 - driver signature + repair-retry controls
    ctx: StageContext,
    input_path: Path,
    output_path: Path,
    ffmpeg_path: str,
    progress_update: Callable[[int, int], None] | None = None,
    *,
    auto_repair: bool = True,
    _repair_attempted: bool = False,
) -> tuple[int, int]:
    """
    Run the full chain for ONE input file; returns the output fps as (num, den).

    Zero-touch repair: a diagnosis verdict meaning «this build cannot encode»
    (absent libx265 / no 10-bit / dead 8-bit) triggers the winget reinstall
    ladder once and retries the entry with the new binary. «Причина в другом»
    and second failures stay loud terminal errors (continue-and-summarize
    untouched: the batch loop decides what a dead entry means).
    """
    try:
        return _run_pipeline_once(ctx, input_path, output_path, ffmpeg_path, progress_update)
    except (RuntimeError, PipelineError) as exc:
        # PipelineError is the carrier a REAL dead build arrives in: the
        # first-frame encoder-open failure breaks the y4m write (EPIPE/EINVAL)
        # and the OSError handler converts the verdict-carrying RuntimeError
        # into PipelineError (not a RuntimeError subclass — live-verified).
        # RuntimeError covers the post-stream finish_encoder failures.
        report = str(exc)
        if _repair_attempted or not verdict_wants_reinstall(report):
            raise
        repaired = ffmpeg_repair.repair_ffmpeg_build(
            report, ctx.reporter, override_pinned=not auto_repair
        )
        return _run_pipeline_once(ctx, input_path, output_path, str(repaired), progress_update)


def _run_pipeline_once(
    ctx: StageContext,
    input_path: Path,
    output_path: Path,
    ffmpeg_path: str,
    progress_update: Callable[[int, int], None] | None = None,
) -> tuple[int, int]:
    """Run the pipeline body (decode → graph → encode); called directly or retried."""
    core = vs.core
    try:
        clip = core.bs.VideoSource(str(input_path))  # ty: ignore[unresolved-attribute]
    except Exception as exc:
        msg = f"Не удалось декодировать {input_path}: {exc}"
        raise PipelineError(msg) from exc

    source_fps = Fraction(clip.fps)
    if "interpolate" in ctx.run.stages:
        check_target_fps(source_fps, ctx.run.interpolate.target_fps)

    source_matrix = _detect_source_matrix(input_path)
    graph = build_graph(ctx, clip, matrix=source_matrix)
    output_fps = Fraction(graph.fps)
    video_args = build_video_args(
        ctx.run.encode,
        (output_fps.numerator, output_fps.denominator),
        threads=ctx.hardware.encode_threads,
    )
    args = build_mux_args(
        input_path,
        output_path,
        video_args,
        ffmpeg=ffmpeg_path,
        width=graph.width,
        height=graph.height,
        fps_num=output_fps.numerator,
        fps_den=output_fps.denominator,
    )

    encoder = spawn_encoder(args)
    if encoder.stdin is None:
        msg = "ffmpeg не открыл stdin для rawvideo-потока (неожиданный сбой Popen)."
        raise PipelineError(msg)
    try:
        try:
            graph.output(
                encoder.stdin,
                prefetch=max(1, ctx.hardware.num_streams_per_gpu) if ctx.hardware.gpus else 0,
                progress_update=progress_update,
            )
        except OSError as exc:
            # A dead ffmpeg surfaces as EPIPE (Linux) / EINVAL (Windows) on the
            # VS-side write. finish_encoder raises with ffmpeg's ACTUAL exit
            # code + stderr tail (and the libx265 capability diagnosis when the
            # encoder failed to open) — that is the real cause; the bare errno
            # only says the pipe broke (live E05/E06/E10/E11 class).
            try:
                finish_encoder(encoder)
            except RuntimeError as enc_exc:
                raise PipelineError(str(enc_exc)) from exc
            msg = f"Поток в ffmpeg оборвался во время кодирования: {exc}"
            raise PipelineError(msg) from exc
        except BaseException:
            # M2: ANY other escape (VS-internal failure, KeyboardInterrupt) must reap
            # the child — ffmpeg never outlives this call — while the ORIGINAL
            # exception stays the loud failure. If ffmpeg died too, its tail is
            # reported without masking the cause.
            try:
                finish_encoder(encoder)
            except Exception as teardown_exc:  # noqa: BLE001 - teardown must not mask the cause
                ctx.reporter.warn(f"ffmpeg не завершился чисто после сбоя графа: {teardown_exc}")
            raise
    finally:
        if encoder.stdin is not None and not encoder.stdin.closed:
            # Already-broken pipe (dead ffmpeg): the failure above is the real
            # outcome — never let a close-time EINVAL mask it.
            with contextlib.suppress(OSError):
                encoder.stdin.close()
    finish_encoder(encoder)
    return (output_fps.numerator, output_fps.denominator)
