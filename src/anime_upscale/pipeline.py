"""
Graph assembly: validate subset → build VS nodes in canonical order → rawvideo → ffmpeg (plan §1).

Format flow: RGBH (fp16) around the upscale/interpolate segment, one conversion
to YUV420P10 (BT.709) right before the pipe. Interp-only runs convert to RGBH
once before RIFE (no RGB round-trips). Scene detection runs inside vsrife
(sc=True + sc_threshold) on a GRAY8 side-branch — requires the misc plugin.

Long-run GPU protection (P2 wave, GPU-degradation incident design §4):
- per-episode hygiene: ``gc.collect()`` + ``torch.cuda.empty_cache()`` after
  EVERY episode (success or failure) — vsrife per-episode engines/streams and
  the spandrel model die by refcount alone today, and reference cycles plus
  allocator fragmentation (segments from separate cudaMalloc never merge) can
  keep a dead episode resident while the next one loads;
- ``DegradationWatchdog``: per-episode fps windows vs the same episode's early
  baseline + a cross-episode ``memory_reserved`` monotonic-growth trend; first
  degradation warns loudly, a repeated one (no healthy episode in between)
  raises a loud restart-advice RuntimeError;
- allocator cap: ``[hardware] memory_fraction`` (default 0.9) applied once per
  process via ``torch.cuda.set_per_process_memory_fraction`` so the caching
  allocator raises CUDA OOM INSIDE the process before the Windows driver can
  silently spill allocations into system RAM (the incident: ~20 GB on a
  16-GB card via WDDM sysmem fallback). The cap bounds the ALLOCATOR only —
  the CUDA context (~300–500 MB) and non-torch allocations sit outside it,
  which is exactly why the default leaves a 10% headroom instead of 1.0.
"""

import contextlib
import gc
import statistics
import subprocess
import time
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import Protocol

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


class _WarnReporter(Protocol):
    """The only reporter surface hygiene/watchdog need (duck-typed in tests)."""

    def warn(self, text: str) -> None:
        """Emit a loud warning line."""
        ...


# ── Long-run GPU protection: hygiene, watchdog, allocator cap (P2 wave) ─────────────────


# fps windows: a window is this many frames; the baseline is the median of the
# first windows of the SAME episode (stage set), so a mid-episode collapse (the
# incident's signature: the spill onset is crossed mid-episode) is caught by
# later windows against the healthy early baseline.
_FPS_WINDOW_FRAMES = 2000
_FPS_BASELINE_WINDOWS = 3
# Window fps below (1 - this) of the baseline = degradation event.
_FPS_DROP_FRACTION = 0.40
# Cross-episode reserved-VRAM trend: strictly-increasing suffix of at least
# this many episodes with total growth above the fraction = event. A flat pair
# (plateau) breaks the suffix — allocator churn that stabilised is healthy.
_MEMORY_TREND_MIN_SAMPLES = 3
_MEMORY_GROWTH_FRACTION = 0.10
_MEMORY_TREND_BUFFER = 12


def _cuda_reserved_bytes() -> int | None:
    """Allocator-managed pool size (``torch.cuda.memory_reserved``), None without CUDA."""
    try:
        import torch  # noqa: PLC0415 - lazy: encode-only runs must not load the torch stack
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return int(torch.cuda.memory_reserved())


def _episode_hygiene(reporter: _WarnReporter) -> None:
    """
    Per-episode VRAM hygiene: ``gc.collect()`` + ``torch.cuda.empty_cache()``.

    Runs at the episode boundary ONLY — never inside the VS progress callback
    (a mid-graph device sync against in-flight frames is unsafe). Guarded:
    a hygiene failure must never reclassify the episode — warn and continue.
    """
    try:
        gc.collect()
        import torch  # noqa: PLC0415 - lazy, same rationale as the probe

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001 - hygiene must never kill the batch
        reporter.warn(f"Очистка памяти на границе серий не удалась (продолжаем): {exc}")


class DegradationWatchdog:
    """
    Runtime degradation sentinel: per-episode fps windows + VRAM growth trend.

    One instance per PROCESS (module singleton): episode state resets on
    ``begin_episode``, escalation state persists across episodes — exactly the
    long-run semantics both the sequential driver and every fan-out worker
    need (each spawn child owns a fresh instance).

    Escalation policy (owner decision): the FIRST degradation event (fps drop
    or monotonic reserved growth) warns loudly and marks one-time mitigation —
    the unconditional boundary hygiene that follows IS that cleanup. A
    REPEATED event with no fully-healthy episode in between means the cleanup
    did not restore health → loud RuntimeError with restart advice. A healthy
    episode after a mitigation resets the flag: an isolated transient must not
    arm a permanent tripwire. Per-episode warn dedup: at most one degradation
    verdict per episode.
    """

    def __init__(self, reporter: _WarnReporter) -> None:
        """Bind the reporter; episode and escalation state start clean."""
        self._reporter = reporter
        self._clock_fn: Callable[[], float] = time.monotonic
        # per-episode state (reset by begin_episode)
        self._windows: list[float] = []
        self._window_anchor_frames: int | None = None
        self._window_anchor_time: float = 0.0
        self._window_frames: int = _FPS_WINDOW_FRAMES
        self._baseline: float | None = None
        self._episode_warned: bool = False
        # Set by _degradation, consumed (and cleared) by finish_episode: the
        # boundary of the very episode that degraded is NOT a recovery.
        self._event_pending_boundary: bool = False
        # cross-episode state
        self._mitigated: bool = False
        self._reserved_samples: list[int] = []

    def begin_episode(self) -> None:
        """Reset the per-episode fps-window state (escalation state persists)."""
        self._windows = []
        self._window_anchor_frames = None
        self._window_anchor_time = 0.0
        self._baseline = None
        self._episode_warned = False

    def wrap_progress(
        self, progress_update: Callable[[int, int], None] | None
    ) -> Callable[[int, int], None] | None:
        """Return a callback that feeds fps windows AND the original consumer."""
        if progress_update is None:
            return None

        def watched(done: int, total: int) -> None:
            self._on_frame(done, total)
            progress_update(done, total)

        return watched

    def _on_frame(self, done: int, total: int) -> None:  # noqa: ARG002 - VS callback shape
        if self._window_anchor_frames is None or (done < self._window_anchor_frames):
            # Lazy anchor, or the frame counter restarted (retry/segment) — re-anchor.
            self._window_anchor_frames = done
            self._window_anchor_time = self._clock_fn()
            return
        produced = done - self._window_anchor_frames
        if produced < self._window_frames:
            return
        now = self._clock_fn()
        elapsed = now - self._window_anchor_time
        if elapsed > 0:
            self._evaluate_window(produced / elapsed)
        self._window_anchor_frames = done
        self._window_anchor_time = now

    def _evaluate_window(self, fps: float) -> None:
        self._windows.append(fps)
        baseline = self._baseline
        if baseline is None:
            if len(self._windows) < _FPS_BASELINE_WINDOWS:
                return
            baseline = statistics.median(self._windows[:_FPS_BASELINE_WINDOWS])
            self._baseline = baseline
            return  # the window completing the baseline is never evaluated itself
        if fps >= baseline * (1 - _FPS_DROP_FRACTION):
            return
        if self._episode_warned:
            return  # one degradation verdict per episode — no per-window spam
        self._degradation(
            f"скорость обработки упала с {baseline:.1f} до {fps:.1f} кадр/с (>40%)",
            "Типичный признак переполнения VRAM (на Windows — уход аллокаций CUDA "
            "в shared GPU memory / sysmem fallback).",
        )

    def _memory_trend_detail(self, reserved: int) -> str | None:
        """Append a sample; return the RU trend detail when growth is monotonic."""
        self._reserved_samples.append(reserved)
        if len(self._reserved_samples) > _MEMORY_TREND_BUFFER:
            del self._reserved_samples[: len(self._reserved_samples) - _MEMORY_TREND_BUFFER]
        suffix: list[int] = [self._reserved_samples[-1]]
        for value in reversed(self._reserved_samples[:-1]):
            if value < suffix[0]:
                suffix.insert(0, value)
            else:
                break  # plateau/dip: the strictly-increasing run ends here
        if len(suffix) < _MEMORY_TREND_MIN_SAMPLES:
            return None
        first, last = suffix[0], suffix[-1]
        growth = (last - first) / first
        if growth <= _MEMORY_GROWTH_FRACTION:
            return None
        return (
            f"зарезервированная VRAM растёт {len(suffix)} серии подряд без плато "
            f"({first / (1024**3):.2f} → {last / (1024**3):.2f} ГБ, +{growth * 100:.0f}%)"
        )

    def _degradation(self, detail: str, advice: str) -> None:
        """Escalation core: first event = loud warn + one-time mitigation; repeat = fail."""
        self._episode_warned = True
        self._event_pending_boundary = True
        if self._mitigated:
            msg = (
                f"Деградация GPU сохраняется после очистки ({detail}). Перезапустите "
                "пайплайн: прогресс серий сохранён, незавершённые серии будут обработаны "
                "при следующем запуске (см. docs/ проекта)."
            )
            raise RuntimeError(msg)
        self._mitigated = True
        self._reporter.warn(
            f"ВНИМАНИЕ: {detail}. {advice} Одноразовая очистка памяти "
            "(gc + empty_cache) будет выполнена на границе этой серии; "
            "продолжаем и наблюдаем."
        )

    def finish_episode(self, *, allow_escalation: bool) -> None:
        """
        Episode boundary: sample the VRAM trend, run hygiene, escalate if due.

        ``allow_escalation=False`` (the pipeline body already failed) records
        the trend and warns but NEVER raises — a watchdog verdict must not
        mask the episode's own loud failure.
        """
        try:
            reserved = _cuda_reserved_bytes()
        except Exception as exc:  # noqa: BLE001 - telemetry must not kill the boundary
            self._reporter.warn(f"Метрика memory_reserved недоступна (продолжаем): {exc}")
            reserved = None
        trend_detail = self._memory_trend_detail(reserved) if reserved is not None else None
        if trend_detail is not None:
            advice = "Похоже на утечку или фрагментацию аллокатора PyTorch."
            if self._mitigated and not allow_escalation:
                self._event_pending_boundary = True
                self._reporter.warn(
                    f"ВНИМАНИЕ: деградация GPU сохраняется после очистки ({trend_detail}) — "
                    "следующие серии будут падать громко и быстро; перезапустите пайплайн "
                    "(прогресс серий сохранён). Эскалация подавлена: серия уже завершилась "
                    "ошибкой."
                )
            else:
                self._degradation(trend_detail, advice)
        if not self._event_pending_boundary and self._mitigated:
            self._mitigated = False  # fully healthy episode = recovery observed
        self._event_pending_boundary = False
        _episode_hygiene(self._reporter)


_WATCHDOG: DegradationWatchdog | None = None


def _episode_watchdog(reporter: _WarnReporter) -> DegradationWatchdog:
    """Per-process watchdog singleton (state persists across episodes)."""
    global _WATCHDOG  # noqa: PLW0603 - module singleton: cross-episode escalation state
    if _WATCHDOG is None:
        _WATCHDOG = DegradationWatchdog(reporter)
    return _WATCHDOG


_CAP_APPLIED = False


def _apply_memory_cap(ctx: StageContext) -> None:
    """
    Apply ``[hardware] memory_fraction`` via ``torch.cuda.set_per_process_memory_fraction``.

    Once per process, at the first pipeline call — after detect_hardware,
    before any inference. Goal (the incident): the caching allocator raises a
    CUDA OOM INSIDE the process earlier than the Windows driver can silently
    spill allocations into system RAM (WDDM sysmem fallback — the owner's
    20-GB-on-16-GB run). Why the default is 0.9 and not 1.0: the cap bounds
    the ALLOCATOR only — the CUDA context (~300–500 MB) and non-torch
    allocations (VS/TRT driver surfaces) sit outside it, so a 10% headroom
    absorbs context + fragmentation while still stopping chronic growth.
    CPU path / no CUDA = no-op. Application failure warns loudly and
    continues without the cap: it is defense-in-depth (the watchdog remains),
    never a reason to kill the batch. ``float()`` is mandatory — torch
    rejects int fractions with TypeError.
    """
    global _CAP_APPLIED  # noqa: PLW0603 - module singleton: cap applies once per process
    if _CAP_APPLIED:
        return
    _CAP_APPLIED = True  # one attempt per process, success or not
    try:
        gpus = ctx.hardware.gpus
        if not gpus:
            return  # CPU path / no CUDA: no-op by contract
        fraction = float(ctx.run.memory_fraction)
        device_index = gpus[0].index
        import torch  # noqa: PLC0415 - lazy, same rationale as the probe

        if not torch.cuda.is_available():
            return
        torch.cuda.set_per_process_memory_fraction(fraction, device_index)
        ctx.reporter.note(
            f"Лимит аллокатора PyTorch: {fraction:.2f}×VRAM (device {device_index}) — "
            "CUDA OOM внутри процесса раньше спилла в системную RAM "
            "([hardware] memory_fraction)."
        )
    except Exception as exc:  # noqa: BLE001 - see docstring: warn, never kill the batch
        ctx.reporter.warn(f"Лимит памяти PyTorch не применён (продолжаем без капа): {exc}")


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

    Long-run protection (P2): the allocator cap applies once per process;
    the per-episode watchdog brackets the body — ``begin_episode`` before,
    ``finish_episode`` after (both the success and the failure path), so the
    VRAM hygiene runs unconditionally and the degradation escalation never
    masks the pipeline's own loud failure.
    """
    _apply_memory_cap(ctx)
    watchdog = _episode_watchdog(ctx.reporter)
    watchdog.begin_episode()
    try:
        result = _run_pipeline_with_repair(
            ctx,
            input_path,
            output_path,
            ffmpeg_path,
            watchdog.wrap_progress(progress_update),
            auto_repair=auto_repair,
            _repair_attempted=_repair_attempted,
        )
    except BaseException:
        watchdog.finish_episode(allow_escalation=False)
        raise
    watchdog.finish_episode(allow_escalation=True)
    return result


def _run_pipeline_with_repair(  # noqa: PLR0913 - driver signature + repair-retry controls
    ctx: StageContext,
    input_path: Path,
    output_path: Path,
    ffmpeg_path: str,
    progress_update: Callable[[int, int], None] | None = None,
    *,
    auto_repair: bool = True,
    _repair_attempted: bool = False,
) -> tuple[int, int]:
    """Repair-wrapped pipeline body (the pre-P2 ``run_pipeline`` contract)."""
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
