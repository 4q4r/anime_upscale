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


def build_mux_args(
    source: str | Path,
    output: str | Path,
    video_args: list[str],
    ffmpeg: str | None = None,
) -> list[str]:
    """Full ffmpeg argv: yuv4mpegpipe on stdin + source streams → MKV."""
    return [
        ffmpeg or _FFMPEG,
        "-y",
        "-loglevel",
        "error",
        "-stats",
        "-f",
        "yuv4mpegpipe",
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


def _decode(value: bytes | str | None) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value or ""


def _probe_args(ffmpeg: str, pix_fmt: str, output: Path) -> list[str]:
    return [
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
        "-frames:v",
        "3",
        str(output),
    ]


def _run_probe(
    ffmpeg: str,
    pix_fmt: str,
    runner: Any,  # noqa: ANN401 - subprocess.run-shaped callable
) -> tuple[bool, str]:
    """Minimal encode probe; (ok, stderr-tail) — failures keep the actual error."""
    with tempfile.TemporaryDirectory(prefix="anime-upscale-probe-") as tmp:
        output = Path(tmp) / f"probe_{pix_fmt}.mkv"
        try:
            completed = runner(
                _probe_args(ffmpeg, pix_fmt, output),
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


def diagnose_encoder(
    ffmpeg: str,
    tail_text: str,
    runner: Any = None,  # noqa: ANN401 - subprocess.run-shaped callable
) -> str | None:
    """
    Capability diagnosis for a failed libx265 open; loud Russian verdict.

    Matrix: encoder absent → full-build recommendation; 8-bit works but 10-bit
    does not → incomplete build; both work → the cause is elsewhere (tail);
    neither works → broken build. None when the failure is not an encoder-open
    failure.
    """
    if not encoder_open_failure(tail_text):
        return None
    run = runner if runner is not None else subprocess.run
    encoders = _list_x265_encoders(ffmpeg, run)
    if not encoders:
        return (
            f"\nДИАГНОСТИКА КОДЕКА: libx265 отсутствует в этой сборке ffmpeg.\n{_FULL_BUILD_HINT}"
        )

    ok10, err10 = _run_probe(ffmpeg, _PROBE_FMT_10BIT, run)
    if ok10:
        return (
            "\nДИАГНОСТИКА КОДЕКА: 10-бит работает — причина в другом.\n"
            "libx265 присутствует, пробное 10-битное кодирование (320x240, 3 кадра в MKV) "
            "прошло успешно. Ищите причину в хвосте stderr выше "
            "(например, несовместимые параметры потока)."
        )

    ok8, err8 = _run_probe(ffmpeg, _PROBE_FMT_8BIT, run)
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
        tail = b"".join(encoder.tail).decode("utf-8", "replace").strip().splitlines()[-10:]
        tail_text = "\n".join(tail)
        message = f"ffmpeg завершился с ошибкой (код {process.returncode}).\n{tail_text}"
        raw_args = process.args
        ffmpeg = str(raw_args[0]) if raw_args else _FFMPEG  # ty: ignore[not-subscriptable]
        report = diagnose_encoder(ffmpeg, tail_text)
        if report is not None:
            message += report
        raise RuntimeError(message)
