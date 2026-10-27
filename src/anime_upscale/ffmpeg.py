"""ffmpeg encode/mux subprocess: argv builder + y4m stdin wiring (plan §1, §12.3)."""

import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

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

    @property
    def stdin(self) -> BinaryIO | None:
        """The y4m write end (clip.output target)."""
        return self.process.stdin


_TAIL_LINES = 50


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

    threading.Thread(target=_drain, daemon=True).start()
    return encoder


def finish_encoder(encoder: Encoder) -> None:
    """Close stdin, wait, and fail loudly with the ffmpeg stderr tail on error."""
    process = encoder.process
    if process.stdin is not None and not process.stdin.closed:
        process.stdin.close()
    process.wait()
    if process.returncode != 0:
        tail = b"".join(encoder.tail).decode("utf-8", "replace").strip().splitlines()[-10:]
        raise RuntimeError(
            f"ffmpeg завершился с ошибкой (код {process.returncode}).\n" + "\n".join(tail)
        )
