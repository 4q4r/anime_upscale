"""
External tools discovery (plan §12.3): PATH first, [tools] config override second.

One resolution helper for ffmpeg/ffprobe; a missing tool is a loud Russian
error naming the tool and the override key. nvidia-smi is diagnostic-only and
never blocks (hardware.py treats its absence as "no cross-check data").
"""

import os
import shutil
from pathlib import Path


class ToolError(Exception):
    """Loud external-tool resolution failure (Russian message)."""


def resolve_tool(name: str, override: str | None = None) -> str:
    """
    Resolve an external tool: [tools] override if set, else PATH lookup.

    `name` is the logical tool name ("ffmpeg", "ffprobe"); on Windows the
    lookup also matches the .exe spelling via shutil.which semantics.
    """
    if override:
        candidate = Path(override).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
        # On Windows a configured absolute path may omit the extension.
        if os.name == "nt" and candidate.suffix == "":
            with_ext = candidate.with_suffix(".exe")
            if with_ext.is_file():
                return str(with_ext)
        msg = (
            f'Инструмент «{name}» задан в конфиге ([tools] {name} = "{override}"), '
            "но файл не найден или не исполняемый."
        )
        raise ToolError(msg)
    found = shutil.which(name)
    if found:
        return found
    msg = (
        f"Инструмент «{name}» не найден в PATH. Установите его и добавьте в PATH, "
        f'или укажите путь в config.toml: [tools]\n{name} = "C:\\\\путь\\\\{name}.exe"'
    )
    raise ToolError(msg)
