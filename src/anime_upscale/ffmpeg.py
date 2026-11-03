"""ffmpeg encode/mux subprocess: argv builder + y4m stdin wiring (plan §1, §12.3)."""

import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from anime_upscale.tools import resolve_tool

_FFMPEG = "ffmpeg"


def resolve_ffmpeg(override: str | None = None) -> str:
    """PATH-first with [tools] ffmpeg override (plan §12.3)."""
    return resolve_tool(_FFMPEG, override)


def build_mux_args(  # noqa: PLR0913 - driver signature: source, output, video_args, ffmpeg, geometry/fps
    source: str | Path,
    output: str | Path,
    video_args: list[str],
    ffmpeg: str | None = None,
    *,
    width: int | None = None,
    height: int | None = None,
    fps_num: int | None = None,
    fps_den: int | None = None,
) -> list[str]:
    """
    Full ffmpeg argv: rawvideo on stdin + source streams → MKV.

    rawvideo replaces yuv4mpegpipe because VS writes C420p10 in the y4m header
    which ffmpeg misreads as 8-bit → violet-square artifacts on 10-bit sources.
    width/height/fps_num/fps_den come from the assembled VS graph (known after
    build_graph) and are required for rawvideo demuxer framing.
    """
    if width is None or height is None or fps_num is None or fps_den is None:
        msg = "build_mux_args требует width, height, fps_num, fps_den для rawvideo-входа."
        raise ValueError(msg)
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
        "-r",
        f"{fps_num}/{fps_den}",
        "-i",
        "-",
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
        "-c:s",
        "copy",
        "-c:t",
        "copy",
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
        """The y4m write end (clip.output target)."""
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
    Start ffmpeg reading y4m on stdin; the caller feeds clip.output(enc.stdin, y4m=True).

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
