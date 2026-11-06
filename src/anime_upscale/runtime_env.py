"""
Runtime platform detection → frozen RuntimeProfile (plan-kaggle-colab §2.1/F1).

Detection is NOT configuration: the app keeps reading settings exclusively from
config.toml. This module only *identifies the platform* (read-only os.environ
probes and path probes) and resolves root/state defaults — the same class of
sensing hardware.py already does for GPUs. No env writes, no config keys, ever.

Owner decisions fixed for the notebook wave (2026-11-06):
- encode preset stays "medium" on every platform (no [profiles.notebook]);
- the upload.gofile.state_file config key is NOT added (wave N2 concern,
  owner question still open);
- torch / torch-tensorrt versions are NOT this module's concern — the
  notebook bootstrap cells resolve the driver-dependent index swap.

Precedence: kaggle > colab > local.
- kaggle: env KAGGLE_KERNEL_RUN_TYPE present OR /kaggle/working exists;
- colab: importlib.util.find_spec("google.colab") is not None OR COLAB_GPU present;
- else local.

Zero torch/vapoursynth imports at module level (CLI-lazy discipline):
this module is stdlib-only.

Test seams (F10): _path_exists() and importlib.util.find_spec() are called at
call time so contract tests can monkeypatch them; env signals go through
os.environ.get() reads only.
"""

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

RuntimeKind = Literal["local", "kaggle", "colab"]

KAGGLE_WORKING_DIR: Final = "/kaggle/working"
COLAB_CONTENT_DIR: Final = "/content"

_SESSION_NOTES: Final[dict[str, str]] = {
    "kaggle": (
        "Kaggle: /kaggle/working сохраняется между сессиями (лимит ~20 ГБ), "
        "сессия ≤12 ч — для длинных батчей используйте Save & Run All."
    ),
    "colab": (
        "Colab: хранилище эфемерно — результаты и config переносите на "
        "Google Drive; сессия может быть завершена без предупреждения."
    ),
    "local": (
        "Локальная среда: основной целевой рантайм (Windows-сервер RTX 5080), "
        "хранилище персистентно."
    ),
}


@dataclass(frozen=True)
class RuntimeProfile:
    """
    Frozen per-run platform snapshot consumed read-only (plan §2.1).

    persistent_root is the platform-provided surface that survives the session
    end: /kaggle/working on Kaggle; None on Colab (everything is ephemeral —
    persistence is an explicit Drive mount) and locally (the project directory
    itself is persistent, there is no special runtime surface).
    """

    kind: RuntimeKind
    persistent_root: str | None
    session_note: str


def _path_exists(path: str) -> bool:
    """Read-only path probe; module seam monkeypatched by contract tests."""
    return Path(path).exists()


def _has_colab_spec() -> bool:
    """
    Probe for the google.colab package.

    find_spec imports the `google` parent for a dotted name: on machines where
    that parent is absent entirely it raises ModuleNotFoundError — that is
    "Colab absent", not a failure, and must fall through to local.
    """
    try:
        return importlib.util.find_spec("google.colab") is not None
    except (ImportError, ValueError):
        return False


def detect_runtime() -> RuntimeProfile:
    """Identify the platform with kaggle > colab > local precedence (plan F1)."""
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None or _path_exists(KAGGLE_WORKING_DIR):
        kind: RuntimeKind = "kaggle"
    elif os.environ.get("COLAB_GPU") is not None or _has_colab_spec():
        kind = "colab"
    else:
        kind = "local"
    return RuntimeProfile(
        kind=kind,
        persistent_root=KAGGLE_WORKING_DIR if kind == "kaggle" else None,
        session_note=_SESSION_NOTES[kind],
    )


def default_project_root() -> Path:
    """Platform default root: kaggle/colab clone dir or the current directory."""
    kind = detect_runtime().kind
    if kind == "kaggle":
        return Path(KAGGLE_WORKING_DIR) / "anime_upscale"
    if kind == "colab":
        return Path(COLAB_CONTENT_DIR) / "anime_upscale"
    return Path.cwd()
