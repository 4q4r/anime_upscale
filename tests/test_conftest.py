"""conftest policy tests (M8): markers, CUDA deselection, VS-less collection."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parent.parent
CONFTEST = REPO / "tests" / "conftest.py"
GPU_TEST_ID = "tests/test_stages.py::TestGpuStages::test_interpolate_cpu_backed_graph"


def _load_conftest() -> ModuleType:
    spec = importlib.util.spec_from_file_location("tests_conftest_under_test", CONFTEST)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeItem:
    def __init__(self, nodeid: str, keywords: set[str]) -> None:
        self.nodeid = nodeid
        self.keywords = keywords


class TestDeselectHelper:
    def test_gpu_items_removed_only_without_cuda(self) -> None:
        conftest = _load_conftest()
        items = [
            _FakeItem("a", {"gpu"}),
            _FakeItem("b", set()),
            _FakeItem("c", {"vapoursynth"}),
        ]
        deselected = conftest.deselect_gpu_items(items, cuda_available=False)
        assert [item.nodeid for item in deselected] == ["a"]
        assert [item.nodeid for item in items] == ["b", "c"]

    def test_with_cuda_nothing_is_deselected(self) -> None:
        conftest = _load_conftest()
        items = [_FakeItem("a", {"gpu"})]
        assert conftest.deselect_gpu_items(items, cuda_available=True) == []
        assert len(items) == 1


class _HookRecorder:
    def __init__(self) -> None:
        self.deselected: list[object] = []

    def pytest_deselected(self, items: list[object]) -> None:
        self.deselected.extend(items)


class _FakeConfig:
    def __init__(self) -> None:
        self.hook = _HookRecorder()


class TestDeselectedHook:
    def test_modifyitems_reports_dropped_items_via_hook(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """PR1-review Minor: deselection must be VISIBLE — the removed items go
        through config.hook.pytest_deselected (item objects, not bare ids), so
        pytest's summary shows the drop instead of a silently smaller count."""
        conftest = _load_conftest()
        monkeypatch.setattr(conftest, "_cuda_available", lambda: False)
        config = _FakeConfig()
        items = [_FakeItem("gpu1", {"gpu"}), _FakeItem("plain", set())]
        conftest.pytest_collection_modifyitems(config, items)  # type: ignore[arg-type]
        assert [item.nodeid for item in config.hook.deselected] == ["gpu1"]
        assert [item.nodeid for item in items] == ["plain"]

    def test_modifyitems_hook_silent_when_cuda_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        conftest = _load_conftest()
        monkeypatch.setattr(conftest, "_cuda_available", lambda: True)
        config = _FakeConfig()
        items = [_FakeItem("gpu1", {"gpu"})]
        conftest.pytest_collection_modifyitems(config, items)  # type: ignore[arg-type]
        assert config.hook.deselected == []
        assert len(items) == 1

    def test_deselection_visible_in_run_output(self) -> None:
        """PR1-review Minor: with CUDA hidden the summary must carry the drop
        (pytest's own ``N deselected`` tail) plus the reason line."""
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ""
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_stages.py", "--no-header"],
            capture_output=True,
            text=True,
            cwd=REPO,
            env=env,
            check=False,
        )
        combined = result.stdout + result.stderr
        assert result.returncode == 0, combined
        assert "1 deselected" in combined
        assert "gpu tests (no CUDA device)" in combined


class TestVsCollectionGate:
    def test_vs_modules_ignored_only_without_vapoursynth(self) -> None:
        conftest = _load_conftest()
        assert conftest.vs_ignored_modules(vapoursynth_available=True) == []
        ignored = conftest.vs_ignored_modules(vapoursynth_available=False)
        # test_batch/test_batch_fanout/test_benchmark join the matrix
        # (PR1-review Minor): they patch/import the pipeline module at RUNTIME,
        # so VS-less machines would otherwise hit ModuleNotFoundError mid-test,
        # not at collection.
        assert {
            "test_pipeline.py",
            "test_stages.py",
            "test_batch.py",
            "test_batch_fanout.py",
            "test_benchmark.py",
        } == set(ignored)

    def test_conftest_computes_ignores_for_this_machine(self) -> None:
        conftest = _load_conftest()
        has_vs = importlib.util.find_spec("vapoursynth") is not None
        assert conftest.collect_ignore == conftest.vs_ignored_modules(vapoursynth_available=has_vs)


class TestMarkerRegistration:
    def test_markers_visible_to_strict_mode(self) -> None:
        """M8: gpu/vapoursynth must be registered (single source: conftest) so
        --strict-markers never fails on them."""
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--markers", "-q"],
            capture_output=True,
            text=True,
            cwd=REPO,
            check=False,
        )
        combined = result.stdout + result.stderr
        assert "gpu: requires a CUDA device" in combined
        assert "vapoursynth: requires importable VapourSynth" in combined


class TestGpuDeselectionLive:
    def test_gpu_test_deselected_when_cuda_hidden(self) -> None:
        """M8: with no CUDA visible, the gpu-marked test must be DESELECTED
        (absent from the run), not executed and not merely skipped."""
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ""  # torch.cuda.is_available() → False
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_stages.py", "--collect-only"],
            capture_output=True,
            text=True,
            cwd=REPO,
            env=env,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert GPU_TEST_ID not in result.stdout

    def test_gpu_test_collected_when_cuda_present(self) -> None:
        try:
            import torch
        except ImportError:  # pragma: no cover - torch ships with the lockfile
            pytest.skip("torch unavailable")
        if not torch.cuda.is_available():
            pytest.skip("no CUDA on this machine — deselection branch is the contract")
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_stages.py", "--collect-only"],
            capture_output=True,
            text=True,
            cwd=REPO,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert GPU_TEST_ID in result.stdout
