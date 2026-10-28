"""
Test-session policy (M8): markers, CUDA deselection, VS-less collection.

- gpu/vapoursynth markers are registered HERE (single source of truth with
  --strict-markers; pyproject declares only addopts/testpaths).
- gpu-marked tests are DESELECTED (removed from the run, not merely skipped)
  when torch sees no CUDA device — a machine without CUDA gets a green run
  that simply does not include them.
- Test modules that import vapoursynth at module level are collection-ignored
  on machines without VapourSynth, keeping collection green everywhere.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

# Modules whose import chain needs vapoursynth: direct module-level imports
# (test_pipeline, test_stages) AND runtime pipeline-module consumers
# (test_batch/test_benchmark patch anime_upscale.pipeline — VS-less machines
# would hit ModuleNotFoundError mid-test, not at collection).
VAPOURSYNTH_TEST_MODULES = (
    "test_pipeline.py",
    "test_stages.py",
    "test_batch.py",
    "test_benchmark.py",
)


def _cuda_available() -> bool:
    try:
        import torch
    except Exception:  # noqa: BLE001 - the probe must never break collection
        return False
    return bool(torch.cuda.is_available())


def _vapoursynth_available() -> bool:
    return importlib.util.find_spec("vapoursynth") is not None


def deselect_gpu_items(items: list[pytest.Item], *, cuda_available: bool) -> list[pytest.Item]:
    """DESELECT gpu-marked items when CUDA is absent; returns the removed items."""
    if cuda_available:
        return []
    deselected: list[pytest.Item] = []
    for item in list(items):
        if "gpu" in item.keywords:
            items.remove(item)
            deselected.append(item)
    return deselected


def vs_ignored_modules(*, vapoursynth_available: bool) -> list[str]:
    """Test modules to collection-ignore on this machine (VS-less ⇒ all VS modules)."""
    return [] if vapoursynth_available else list(VAPOURSYNTH_TEST_MODULES)


collect_ignore = vs_ignored_modules(vapoursynth_available=_vapoursynth_available())

# Per-session count of gpu deselections, for the terminal-summary reason line
# (fresh subprocess per pytest run — no cross-run state).
_gpu_deselected: list[int] = []


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "gpu: requires a CUDA device (deselected automatically without one)",
    )
    config.addinivalue_line(
        "markers",
        "vapoursynth: requires importable VapourSynth (collection-ignored otherwise)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    deselected = deselect_gpu_items(items, cuda_available=_cuda_available())
    if deselected:
        # PR1-review Minor: report the drop so a smaller "collected" count can
        # never hide it silently; pytest's summary then shows "N deselected".
        _gpu_deselected.append(len(deselected))
        config.hook.pytest_deselected(items=deselected)


def pytest_terminal_summary(
    terminalreporter: object, exitstatus: int, config: pytest.Config
) -> None:
    """Name WHY items were dropped (the hook alone shows only the count)."""
    total = sum(_gpu_deselected)
    if total:
        terminalreporter.write_line(f"Deselected {total}: gpu tests (no CUDA device)")
