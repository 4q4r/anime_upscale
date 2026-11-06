"""ffmpeg encode/mux subprocess: argv builder + rawvideo stdin wiring (plan §1, §12.3)."""

import json
import os
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from anime_upscale.reporting import Reporter, resolve_mode
from anime_upscale.tools import ToolError, resolve_tool

_FFMPEG = "ffmpeg"
_FFPROBE = "ffprobe"


def resolve_ffmpeg(override: str | None = None) -> str:
    """PATH-first with [tools] ffmpeg override (plan §12.3)."""
    return resolve_tool(_FFMPEG, override)


# Matroska-portable subtitle codecs (F8; probe7b on ffmpeg 9.0.2: an MP4 with
# mov_text + ``-c:s copy`` into matroska dies at mux — "Subtitle codec mov_text
# (94213) is not supported", rc=218, undiagnosed). mov_text/tx3g are MP4-native
# and transcode to srt (live probe: mov_text→srt muxes clean into subrip);
# pgssub/dvdsub/dvbsub are Matroska-native and stream-copy as-is. Anything
# outside this closed set drops with a warning.
_SUBTITLE_TRANSCODE_TO_SRT = {"mov_text", "tx3g"}
_SUBTITLE_COPY_AS_IS = {"pgssub", "dvdsub", "dvbsub"}


def subtitle_verdict(
    codecs: list[str],
) -> tuple[Literal["copy", "transcode", "drop"], str | None]:
    """
    Pre-mux verdict for the source's subtitle codecs: (args mode, warning).

    copy = keep ``-c:s copy``; transcode = ``-c:s srt``; drop = ``-sn``. A
    mixed text+bitmap set also drops: one ``-c:s`` serves every subtitle
    stream and bitmap→text conversion cannot succeed (ffmpeg docs).
    """
    names = set(codecs)
    if not names:
        return "copy", None
    unknown = sorted(names - _SUBTITLE_TRANSCODE_TO_SRT - _SUBTITLE_COPY_AS_IS)
    if unknown:
        return "drop", f"Субтитры типа {', '.join(unknown)} не переносятся в MKV — отброшены."
    has_text = bool(names & _SUBTITLE_TRANSCODE_TO_SRT)
    has_bitmap = bool(names & _SUBTITLE_COPY_AS_IS)
    if has_text and has_bitmap:
        # One -c:s serves every subtitle stream, and bitmap→text conversion
        # cannot succeed (ffmpeg docs) — drop rather than die at mux.
        mixed = ", ".join(sorted(names))
        return "drop", f"Субтитры типа {mixed} не переносятся в MKV единым кодеком — отброшены."
    if has_text:
        return "transcode", None
    return "copy", None


def _resolve_ffprobe(ffmpeg: str) -> str | None:
    """
    Locate ffprobe next to the resolved ffmpeg, then on PATH.

    ffprobe ships side-by-side with ffmpeg in every packaging (winget/Gyan
    zip, distro packages), so the sibling rule also covers [tools] ffmpeg
    overrides. None = binary absent: the subtitle preflight never blocks
    (plan F8 — no ffprobe means today's copy behavior).
    """
    suffix = ".exe" if os.name == "nt" else ""
    sibling = Path(ffmpeg).with_name(_FFPROBE + suffix)
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return str(sibling)
    try:
        return resolve_tool(_FFPROBE, None)
    except ToolError:
        return None


def _ffprobe_subtitle_codecs(
    ffprobe: str,
    source: str | Path,
    runner: Any,  # noqa: ANN401 - subprocess.run-shaped callable
) -> list[str] | None:
    """
    Subtitle codec names in the source (read-only ffprobe JSON query).

    None on any failure (spawn error, timeout, nonzero exit, bad JSON) — the
    preflight then keeps today's copy behavior instead of blocking the mux.
    """
    args = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "s",
        "-show_entries",
        "stream=codec_name",
        "-of",
        "json",
        str(source),
    ]
    try:
        completed = runner(args, capture_output=True, timeout=_PROBE_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    try:
        payload = json.loads(_decode(completed.stdout))
    except ValueError:
        return None
    streams = payload.get("streams") if isinstance(payload, dict) else None
    if not isinstance(streams, list):
        return None
    return [
        str(entry["codec_name"])
        for entry in streams
        if isinstance(entry, dict) and isinstance(entry.get("codec_name"), str)
    ]


def _warn(text: str) -> None:
    """
    User-facing warning through the reporting layer (Russian, ⚠, TTY-aware).

    The mux path passes no reporter (pipeline builds argv without one), so
    the preflight resolves its own output mode. Explicit CLI flags
    (--no-color/--no-tty) are not visible here; TTY detection and NO_COLOR are.
    """
    Reporter(mode=resolve_mode(stream=sys.stdout)).warn(text)


def _subtitle_preflight(source: str | Path, ffmpeg: str, runner: Any) -> list[str]:  # noqa: ANN401
    """
    Subtitle argv segment from a pre-mux ffprobe verdict (plan F8).

    copy = today's behavior; transcode = ``-c:s srt`` for MP4 text subs;
    drop = ``-sn``. Every failure path degrades to copy — the preflight
    never blocks or alters a mux it cannot classify.
    """
    ffprobe = _resolve_ffprobe(ffmpeg)
    if ffprobe is None:
        return ["-c:s", "copy"]
    codecs = _ffprobe_subtitle_codecs(ffprobe, source, runner)
    if not codecs:
        return ["-c:s", "copy"]
    mode, warning = subtitle_verdict(codecs)
    if warning is not None:
        _warn(warning)
    if mode == "transcode":
        return ["-c:s", "srt"]
    if mode == "drop":
        return ["-sn"]
    return ["-c:s", "copy"]


def build_mux_args(  # noqa: PLR0913 - driver signature: source, output, video_args, ffmpeg, geometry/fps, probe runner
    source: str | Path,
    output: str | Path,
    video_args: list[str],
    ffmpeg: str | None = None,
    *,
    width: int | None = None,
    height: int | None = None,
    fps_num: int | None = None,
    fps_den: int | None = None,
    subtitle_probe_runner: Any | None = None,  # noqa: ANN401 - subprocess.run-shaped callable
) -> list[str]:
    """
    Full ffmpeg argv: rawvideo on stdin + source streams → MKV.

    rawvideo replaces yuv4mpegpipe because VS writes C420p10 in the y4m header
    which ffmpeg misreads as 8-bit → violet-square artifacts on 10-bit sources.
    width/height/fps_num/fps_den come from the assembled VS graph (known after
    build_graph) and are required for rawvideo demuxer framing.

    Audio stream muxing problem & investigation:
    --------------------------------------------
    Problem:
        When encoding rawvideo from stdin and simultaneously copying audio tracks,
        subtitles, and font attachments from the source container into a single MKV,
        the resulting audio tracks ended up empty (0 bytes).

    Root Cause (Investigation):
        Anime release containers typically contain sparse streams: ASS subtitle
        events and attachment fonts (OTF/TTF). FFmpeg's default stream interleaving
        algorithm (`max_interleave_delta=10000000`, i.e., 10 seconds) buffers packets
        and waits for packets from all streams to maintain interleaved presentation.
        Because rawvideo from stdin was fed using the legacy `-r` option instead of
        `-framerate`, timestamps on the raw demuxer could desync or lack granular
        framing markers. Simultaneously, without `-thread_queue_size`, the demuxer's
        internal packet queues for both stdin and file streams starved. When the
        rawvideo stdin reached EOF and closed, the demuxer flushed the remaining
        state while dropping un-interleaved audio packets, writing 0 bytes of audio data.

    Solution & Rationale:
        To resolve this cleanly in a single FFmpeg process without disk overhead or
        intermediate temporary files:
        1. `-framerate {fps_num}/{fps_den}`: Sets exact timestamps (PTS) directly on
           the rawvideo demuxer (unlike `-r` which drops/duplicates frames).
        2. `-thread_queue_size 4096`: Placed before both inputs (`-i -` and `-i {source}`)
           to prevent queue starvation and buffer packet bursts during slow encoding.
        3. `-max_interleave_delta 0`: Forces FFmpeg to flush interleaved packets
           immediately upon readiness rather than buffering up to 10 seconds waiting
           for sparse subtitle/font packets.
        4. `-max_muxing_queue_size 4096`: Extends muxer packet buffer capacity to safely
           handle slower libx265 processing intervals without buffer overruns.

    Subtitle preflight (F8): before the mux, ffprobe reads the source's
    subtitle codecs and the argv picks a Matroska-portable verdict —
    mov_text/tx3g → ``-c:s srt``, pgssub/dvdsub/dvbsub → copy as-is, anything
    else → ``-sn`` with a Russian warning. Every probe failure (no ffprobe,
    no subtitles, unreadable JSON) keeps ``-c:s copy`` so the preflight can
    never block a mux that works today. Tests inject a subprocess.run-shaped
    callable via subtitle_probe_runner.
    """
    if width is None or height is None or fps_num is None or fps_den is None:
        msg = "build_mux_args требует width, height, fps_num, fps_den для rawvideo-входа."
        raise ValueError(msg)
    run = subtitle_probe_runner if subtitle_probe_runner is not None else subprocess.run
    subtitle_args = _subtitle_preflight(source, ffmpeg or _FFMPEG, run)
    return [
        ffmpeg or _FFMPEG,
        "-y",
        "-loglevel",
        "error",
        "-stats",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "yuv420p10le",
        "-s",
        f"{width}x{height}",
        "-framerate",
        f"{fps_num}/{fps_den}",
        "-thread_queue_size",
        "4096",
        "-i",
        "-",
        "-thread_queue_size",
        "4096",
        "-i",
        str(source),
        *video_args,
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-map",
        "1:s?",
        "-map",
        "1:t?",
        "-map_metadata",
        "1",
        "-map_chapters",
        "1",
        "-c:a",
        "copy",
        *subtitle_args,
        "-c:t",
        "copy",
        "-max_interleave_delta",
        "0",
        "-max_muxing_queue_size",
        "4096",
        "-f",
        "matroska",
        str(output),
    ]


@dataclass
class Encoder:
    """A running ffmpeg process with a drained stderr ring (last lines kept)."""

    process: subprocess.Popen[bytes]
    tail: list[bytes] = field(default_factory=list)
    drain_thread: threading.Thread | None = None

    @property
    def stdin(self) -> Any:  # noqa: ANN401 - Popen.stdin is IO[bytes] | None
        """The rawvideo write end (clip.output target; y4m is forbidden — see spawn_encoder)."""
        return self.process.stdin


_TAIL_LINES = 50

# Encoder-open failure markers in ffmpeg stderr (ffmpeg emits these untranslated).
# The second class fires when the build carries no libx265 at all.
_ENCODER_OPEN_MARKERS = ("Cannot open libx265 encoder", "Unknown encoder 'libx265'")

# x265's own parameter/build errors (live 2026-11-02). They identify a parameter
# problem rather than a dead build, so they must lead the report and never
# trigger a reinstall.
_X265_ERROR_MARKERS = ("x265 [error]", "x265 [fatal]")

# Probe pixel formats: 10-bit is the pipeline default (Main10), 8-bit is the
# minimal-capability reference point for the diagnosis matrix.
_PROBE_FMT_10BIT = "yuv420p10le"
_PROBE_FMT_8BIT = "yuv420p"

_FULL_BUILD_HINT = (
    "Установите полную сборку ffmpeg:\n"
    "  • Windows: https://www.gyan.dev/ffmpeg/builds/ — пакет «release full»\n"
    "  • Windows (альтернатива): https://github.com/BtbN/FFmpeg-Builds/releases"
    " — сборка win64-gpl\n"
    "  • Linux: ffmpeg из официальных репозиториев дистрибутива"
)

_PROBE_TIMEOUT_S = 60


def encoder_open_failure(tail_text: str) -> bool:
    """Whether an ffmpeg stderr tail indicates the libx265 encoder failed to open."""
    return any(marker in tail_text for marker in _ENCODER_OPEN_MARKERS)


def is_x265_error_line(line: str) -> bool:
    """Whether one stderr line is an x265 parameter/fatal error."""
    return any(marker in line for marker in _X265_ERROR_MARKERS)


def x265_error_lines(text: str) -> list[str]:
    """Every x265 [error]/[fatal] line in the given stderr text, in order."""
    return [line for line in text.splitlines() if is_x265_error_line(line)]


def _order_tail_lines(text: str) -> str:
    """x265 errors first, then the last 10 ordinary lines (never cut the cause)."""
    lines = text.strip().splitlines()
    errors = [line for line in lines if is_x265_error_line(line)]
    rest = [line for line in lines if not is_x265_error_line(line)][-10:]
    return "\n".join(errors + rest)


def _x265_params_from_args(args: object) -> str | None:
    """Extract the production ``-x265-params`` value from the failed argv."""
    if not isinstance(args, (list, tuple)):
        return None
    try:
        index = args.index("-x265-params")
    except ValueError:
        return None
    if index + 1 >= len(args):
        return None
    return str(args[index + 1])


def _decode(value: bytes | str | None) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value or ""


def _probe_args(
    ffmpeg: str,
    pix_fmt: str,
    output: Path,
    x265_params: str | None = None,
) -> list[str]:
    args = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=black:s=320x240:d=1",
        "-pix_fmt",
        pix_fmt,
        "-c:v",
        "libx265",
    ]
    if x265_params is not None:
        # ONE shared source: the probe runs with the EXACT production params
        # (extracted from the failed argv), so it can never pass while the
        # real run dies on those params (live 2026-11-02 misdiagnosis).
        args += ["-x265-params", x265_params]
    args += ["-frames:v", "3", str(output)]
    return args


def _run_probe(
    ffmpeg: str,
    pix_fmt: str,
    runner: Any,  # noqa: ANN401 - subprocess.run-shaped callable
    x265_params: str | None = None,
) -> tuple[bool, str]:
    """Minimal encode probe; (ok, stderr-tail) — failures keep the actual error."""
    with tempfile.TemporaryDirectory(prefix="anime-upscale-probe-") as tmp:
        output = Path(tmp) / f"probe_{pix_fmt}.mkv"
        try:
            completed = runner(
                _probe_args(ffmpeg, pix_fmt, output, x265_params),
                capture_output=True,
                timeout=_PROBE_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            return False, f"проба {pix_fmt} превысила таймаут {_PROBE_TIMEOUT_S} с."
        return completed.returncode == 0, _decode(completed.stderr).strip()


def _list_x265_encoders(ffmpeg: str, runner: Any) -> list[str]:  # noqa: ANN401
    """List ``-encoders`` lines mentioning x265 (empty list = encoder absent)."""
    args = [ffmpeg, "-hide_banner", "-encoders"]
    completed = runner(args, capture_output=True, timeout=30)
    return [line for line in _decode(completed.stdout).splitlines() if "x265" in line]


def probe_10bit(
    ffmpeg: str,
    runner: Any = None,  # noqa: ANN401 - subprocess.run-shaped callable
    x265_params: str | None = None,
) -> tuple[bool, str]:
    """Public 10-bit capability probe (the repair ladder's re-verification step)."""
    run = runner if runner is not None else subprocess.run
    return _run_probe(ffmpeg, _PROBE_FMT_10BIT, run, x265_params)


def diagnose_encoder(
    ffmpeg: str,
    tail_text: str,
    runner: Any = None,  # noqa: ANN401 - subprocess.run-shaped callable
    x265_params: str | None = None,
) -> str | None:
    """
    Capability diagnosis for a failed libx265 open; loud Russian verdict.

    Probes replay the production ``-x265-params`` (extracted from the failed
    argv). An x265 [error]/[fatal] from that probe is a PARAMETER problem: the
    verdict quotes it and never triggers a winget reinstall. Otherwise the
    build matrix applies: encoder absent → full build; 8-bit works but 10-bit
    does not → incomplete build; neither works → broken build; the probe passes
    → the failure is not reproducible with these params. None when the failure
    is not an encoder-open failure.
    """
    if not encoder_open_failure(tail_text):
        return None
    run = runner if runner is not None else subprocess.run
    encoders = _list_x265_encoders(ffmpeg, run)
    if not encoders:
        return (
            f"\nДИАГНОСТИКА КОДЕКА: libx265 отсутствует в этой сборке ffmpeg.\n{_FULL_BUILD_HINT}"
        )

    ok10, err10 = _run_probe(ffmpeg, _PROBE_FMT_10BIT, run, x265_params)
    param_errors = x265_error_lines(err10)
    if not ok10 and param_errors and x265_params is not None:
        joined = "\n".join(param_errors)
        return (
            "\nДИАГНОСТИКА КОДЕКА: пробное кодирование с боевыми параметрами x265 упало — "
            "причина в параметрах, а не в сборке ffmpeg.\n"
            f"{joined}\n"
            "Исправьте [stages.encode.x265_params] в config.toml "
            "(frame-threads 0..16; бюджет CPU — pools=<N>)."
        )
    if ok10:
        return (
            "\nДИАГНОСТИКА КОДЕКА: проба с боевыми параметрами прошла — сборка и параметры "
            "рабочие.\n"
            "Пробное 10-битное кодирование (320x240, 3 кадра в MKV) с теми же "
            "-x265-params завершилось успешно; сбой не воспроизводится. "
            "Смотрите полный хвост stderr выше."
        )

    ok8, err8 = _run_probe(ffmpeg, _PROBE_FMT_8BIT, run, x265_params)
    if ok8:
        probe_error = err10.splitlines()[-1] if err10 else "(пустой stderr)"
        return (
            "\nДИАГНОСТИКА КОДЕКА: libx265 есть, но 10-бит (Main10) не поддерживается — "
            "нужна полная сборка.\n"
            "8-битное кодирование работает, 10-битное — нет; конвейер кодирует в yuv420p10le.\n"
            f"Ошибка 10-битной пробы: {probe_error}\n"
            f"{_FULL_BUILD_HINT}"
        )
    probe_error = err8.splitlines()[-1] if err8 else "(пустой stderr)"
    return (
        "\nДИАГНОСТИКА КОДЕКА: libx265 есть, но не работает даже 8-битное кодирование — "
        "сборка ffmpeg неисправна.\n"
        f"Ошибка 8-битной пробы: {probe_error}\n"
        f"{_FULL_BUILD_HINT}"
    )


def spawn_encoder(args: list[str]) -> Encoder:
    """
    Start ffmpeg reading rawvideo (yuv420p10le) on stdin for the caller's graph.

    The caller feeds ``graph.output(encoder.stdin, ...)`` RAW frames with NO
    y4m flag anywhere (the pipeline mux path). NEVER enable y4m here (no
    ``clip.output(encoder.stdin, y4m=True)``): VapourSynth's y4m writer emits
    a C420p10 header that ffmpeg's y4m demuxer misreads as 8-bit, and the
    demuxer's timestamp handling breaks PTS — the violet-square kaleidoscope
    + broken picture/audio bug fixed in PR #16. The rawvideo argv instead
    carries width/height/framerate explicitly (build_mux_args).

    A daemon thread drains stderr continuously: without it, ``-stats`` progress
    fills the 64 KiB pipe buffer, ffmpeg blocks on its own stderr write and the
    VS-side write dies with EPIPE (observed live in the parity gate).
    """
    process = subprocess.Popen(  # noqa: S603 - fixed argv list built by this module, no shell
        args,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    encoder = Encoder(process=process)
    stderr_ref = process.stderr

    def _drain() -> None:
        if stderr_ref is None:
            return
        for line in iter(stderr_ref.readline, b""):
            encoder.tail.append(line)
            if len(encoder.tail) > _TAIL_LINES:
                del encoder.tail[: len(encoder.tail) - _TAIL_LINES]
        stderr_ref.close()

    drain = threading.Thread(target=_drain, daemon=True)
    drain.start()
    encoder.drain_thread = drain
    return encoder


def finish_encoder(encoder: Encoder) -> None:
    """
    Close stdin, wait, and fail loudly with the ffmpeg stderr tail on error.

    A libx265 open failure additionally gets the capability diagnosis (probe
    matrix with a concrete full-build recommendation) appended to the error.
    """
    process = encoder.process
    try:
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
    except OSError:
        # ffmpeg already died: the write end breaking (EPIPE on Linux, EINVAL
        # on Windows) is expected here — the exit code + tail below carry the
        # real cause, so a close error must never mask them (live E05-class).
        pass
    process.wait()
    if encoder.drain_thread is not None:
        # wait() returned → stderr hit EOF → the drain loop has finished or is
        # a moment away; joining guarantees the tail is complete.
        encoder.drain_thread.join(timeout=5)
    if process.returncode != 0:
        raw_tail = b"".join(encoder.tail).decode("utf-8", "replace").strip()
        tail_text = _order_tail_lines(raw_tail)
        message = f"ffmpeg завершился с ошибкой (код {process.returncode}).\n{tail_text}"
        raw_args = process.args
        ffmpeg = str(raw_args[0]) if raw_args else _FFMPEG  # ty: ignore[not-subscriptable]
        x265_params = _x265_params_from_args(raw_args)
        report = diagnose_encoder(ffmpeg, tail_text, x265_params=x265_params)
        if report is not None:
            message += report
        raise RuntimeError(message)
