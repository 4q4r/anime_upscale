"""RuntimeProfile contract tests (plan-kaggle-colab F10).

Covers detection precedence kaggle > colab > local, default project roots per
platform, and non-detection = local. Emulation only — no Kaggle/Colab account
or GPU needed: env vars via monkeypatch, platform paths via the _path_exists()
seam, and a fake google.colab module spec via monkeypatched
importlib.util.find_spec (prescribed mechanism).
"""

import dataclasses
import importlib.machinery
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from anime_upscale.runtime_env import (
    KAGGLE_WORKING_DIR,
    RuntimeProfile,
    default_project_root,
    detect_runtime,
)

_REAL_FIND_SPEC = importlib.util.find_spec
_COLAB_SPEC = importlib.machinery.ModuleSpec("google.colab", None)


def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)
    monkeypatch.delenv("COLAB_GPU", raising=False)


def _patch_paths(monkeypatch: pytest.MonkeyPatch, existing: tuple[str, ...] = ()) -> None:
    monkeypatch.setattr("anime_upscale.runtime_env._path_exists", lambda path: path in existing)


def _patch_colab_spec(monkeypatch: pytest.MonkeyPatch, spec: object | None) -> None:
    """Fake google.colab presence through monkeypatched importlib.util.find_spec."""

    def fake_find_spec(name: str, *args: object, **kwargs: object) -> object:
        if name == "google.colab":
            return spec
        return _REAL_FIND_SPEC(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)


def _patch_colab_spec_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Emulate a machine without the `google` parent package at all."""

    def raising_find_spec(name: str, *args: object, **kwargs: object) -> object:
        if name == "google.colab":
            msg = "No module named 'google'"
            raise ModuleNotFoundError(msg, name=name)
        return _REAL_FIND_SPEC(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(importlib.util, "find_spec", raising_find_spec)


def _local_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip every detection signal: hermetic local-baseline emulation."""
    _clear_env(monkeypatch)
    _patch_paths(monkeypatch)
    _patch_colab_spec(monkeypatch, None)


class TestKaggleDetection:
    def test_env_var_set_detects_kaggle(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _local_environment(monkeypatch)
        monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
        assert detect_runtime().kind == "kaggle"

    def test_env_var_present_even_empty_detects_kaggle(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _local_environment(monkeypatch)
        monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "")
        assert detect_runtime().kind == "kaggle"

    def test_working_dir_probe_detects_kaggle(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clear_env(monkeypatch)
        _patch_paths(monkeypatch, (KAGGLE_WORKING_DIR,))
        _patch_colab_spec(monkeypatch, None)
        assert detect_runtime().kind == "kaggle"

    def test_kaggle_precedes_colab(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Both platform signals present at once → kaggle wins (spec precedence)."""
        _patch_paths(monkeypatch)
        _patch_colab_spec(monkeypatch, _COLAB_SPEC)
        monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Batch")
        monkeypatch.setenv("COLAB_GPU", "1")
        assert detect_runtime().kind == "kaggle"


class TestColabDetection:
    def test_fake_google_colab_module_spec_detects_colab(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _clear_env(monkeypatch)
        _patch_paths(monkeypatch)
        _patch_colab_spec(monkeypatch, _COLAB_SPEC)
        assert detect_runtime().kind == "colab"

    def test_colab_gpu_env_detects_colab(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _local_environment(monkeypatch)
        monkeypatch.setenv("COLAB_GPU", "1")
        assert detect_runtime().kind == "colab"

    def test_colab_gpu_env_present_even_empty_detects_colab(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _local_environment(monkeypatch)
        monkeypatch.setenv("COLAB_GPU", "")
        assert detect_runtime().kind == "colab"

    def test_missing_google_parent_falls_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """find_spec raising ModuleNotFoundError (no `google` parent) = not Colab,
        not a crash — detection falls through to local."""
        _clear_env(monkeypatch)
        _patch_paths(monkeypatch)
        _patch_colab_spec_error(monkeypatch)
        assert detect_runtime().kind == "local"


class TestLocalDetection:
    def test_no_signals_detects_local(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _local_environment(monkeypatch)
        assert detect_runtime().kind == "local"

    def test_local_profile_has_no_persistent_root(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _local_environment(monkeypatch)
        profile = detect_runtime()
        assert profile.persistent_root is None
        assert profile.session_note  # non-empty Russian note


class TestDefaultProjectRoot:
    def test_kaggle_root(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _local_environment(monkeypatch)
        monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
        assert default_project_root() == Path("/kaggle/working/anime_upscale")

    def test_colab_root(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clear_env(monkeypatch)
        _patch_paths(monkeypatch)
        _patch_colab_spec(monkeypatch, _COLAB_SPEC)
        assert default_project_root() == Path("/content/anime_upscale")

    def test_local_root_is_cwd(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _local_environment(monkeypatch)
        assert default_project_root() == Path.cwd()


class TestRuntimeProfileContract:
    def test_profile_is_frozen(self) -> None:
        profile = RuntimeProfile(kind="local", persistent_root=None, session_note="примечание")
        with pytest.raises(dataclasses.FrozenInstanceError):
            profile.kind = "kaggle"  # type: ignore[misc]

    def test_persistent_root_per_kind(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert detect_runtime().persistent_root in {None, "/kaggle/working"}
        # local / colab: no platform-provided persistence surface.
        _local_environment(monkeypatch)
        assert detect_runtime().persistent_root is None
        _patch_paths(monkeypatch)
        _patch_colab_spec(monkeypatch, _COLAB_SPEC)
        assert detect_runtime().persistent_root is None
        # kaggle: the only persistent surface is /kaggle/working.
        monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
        assert detect_runtime().persistent_root == "/kaggle/working"

    def test_session_note_nonempty_per_kind(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _local_environment(monkeypatch)
        local_note = detect_runtime().session_note
        assert isinstance(local_note, str) and local_note

        _patch_paths(monkeypatch)
        _patch_colab_spec(monkeypatch, _COLAB_SPEC)
        colab_note = detect_runtime().session_note
        assert colab_note and colab_note != local_note

        monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
        kaggle_note = detect_runtime().session_note
        assert kaggle_note and kaggle_note not in {local_note, colab_note}


class TestReadOnlyContract:
    def test_detection_never_writes_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Read-only discipline: detection must not mutate os.environ (plan §2.1)."""
        monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
        monkeypatch.setenv("COLAB_GPU", "1")
        before = dict(os.environ)
        detect_runtime()
        assert dict(os.environ) == before


class TestLazyImports:
    def test_runtime_env_import_does_not_load_torch_or_vs(self) -> None:
        """CLI-lazy discipline: the stdlib-only detection module must never pull
        the torch/vapoursynth stacks at import time."""
        code = (
            "import sys; import anime_upscale.runtime_env; "
            "assert 'torch' not in sys.modules, 'torch loaded'; "
            "assert 'vapoursynth' not in sys.modules, 'vapoursynth loaded'"
        )
        result = subprocess.run(  # noqa: S603 - fixed argv: venv python + inline assert code
            [sys.executable, "-c", code], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr
