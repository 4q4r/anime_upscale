"""ffmpeg encode/mux subprocess: argv builder + y4m stdin wiring (plan §1, §12.3)."""

import subprocess
from pathlib import Path

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
    """Full ffmpeg argv: y4m on stdin (video) + source (audio/subs/attachments) → MKV."""
    return [
        ffmpeg or _FFMPEG,
        "-y",
        "-loglevel",
        "error",
        "-stats",
        "-f",
        "y4m",
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


def spawn_encoder(args: list[str]) -> subprocess.Popen[bytes]:
    """Start ffmpeg reading y4m on stdin; the caller feeds clip.output(proc.stdin, y4m=True)."""
    return subprocess.Popen(  # noqa: S603 - fixed argv list built by this module, no shell
        args,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def finish_encoder(process: subprocess.Popen[bytes]) -> None:
    """Close stdin, wait, and fail loudly with the ffmpeg stderr tail on error."""
    if process.stdin is not None and not process.stdin.closed:
        process.stdin.close()
    _, stderr = process.communicate()
    if process.returncode != 0:
        tail = (stderr or b"").decode("utf-8", "replace").strip().splitlines()[-10:]
        raise RuntimeError(
            f"ffmpeg завершился с ошибкой (код {process.returncode}).\n" + "\n".join(tail)
        )
