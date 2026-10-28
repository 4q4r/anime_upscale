"""Benchmark script tests: [hardware] overrides must govern the measured runs (M3)."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from anime_upscale.config import RunConfig
from anime_upscale.hardware import GPUInfo, HardwareProfile

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "benchmark.py"


def _load_benchmark() -> ModuleType:
    spec = importlib.util.spec_from_file_location("benchmark_script_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fake_hw() -> HardwareProfile:
    return HardwareProfile(
        gpus=(GPUInfo(index=0, name="FAKE", vram_mb=8192),),
        gpu_count=1,
        cpu_logical_cores=8,
        num_streams_per_gpu=1,
        encode_threads=8,
        precision="fp16",
    )


def _fake_run(stages: tuple[str, ...]) -> RunConfig:
    from anime_upscale.config import (
        DebandConfig,
        DescaleConfig,
        EncodeConfig,
        InterpolateConfig,
        UpscaleConfig,
    )

    return RunConfig(
        stages=stages,
        descale=DescaleConfig(),
        deband=DebandConfig(),
        upscale=UpscaleConfig(),
        interpolate=InterpolateConfig(trt=False),
        encode=EncodeConfig(),
        descale_height=None,
        upload=False,
    )


def _project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_text: str) -> Path:
    (tmp_path / "input").mkdir()
    (tmp_path / "output").mkdir()
    (tmp_path / "input" / "sample.mkv").write_bytes(b"x")
    (tmp_path / "config.toml").write_text(config_text, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path


CONFIG = (
    "[pipeline]\n"
    'order = ["descale", "deband", "upscale", "interpolate", "encode"]\n'
    "[hardware]\n"
    "num_streams = 3\n"
    "encode_threads = 5\n"
    "[stages.encode]\n"
    "crf = 16\n"
    "[profiles.enc]\n"
    'stages = ["encode"]\n'
)


def test_measured_runs_use_config_hardware_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M3: every per-chain detect_hardware call receives the parsed [hardware]
    overrides — a user's num_streams/encode_threads must govern the benchmark."""
    _project(tmp_path, monkeypatch, CONFIG)
    bench = _load_benchmark()
    seen: list[object] = []

    def _spy(overrides: object, cli_num_streams: object, *, nn_stages_selected: bool) -> object:
        seen.append(overrides)
        return _fake_hw()

    monkeypatch.setattr(sys, "argv", ["benchmark.py"])
    monkeypatch.setattr(bench, "detect_hardware", _spy)
    monkeypatch.setattr(bench, "resolve_run", lambda config, stages, **kw: _fake_run(stages))
    monkeypatch.setattr(
        bench,
        "run_pipeline",
        lambda ctx, src, out, ffmpeg_path, progress_update=None: out.write_bytes(b"r"),
    )
    monkeypatch.setattr(bench, "_count_frames", lambda path: 10)
    monkeypatch.setattr(bench, "resolve_tool", lambda name, override=None: "/usr/bin/ffmpeg")

    exit_code = bench.main()
    assert exit_code == 0
    assert seen, "detect_hardware was never called"
    config_overrides = bench.load_config(Path("config.toml")).hardware
    for overrides in seen:
        # Value (not identity) equality: pre-fix code passed HardwareOverrides()
        # defaults — num_streams/encode_threads would be None here.
        assert overrides == config_overrides, (
            "measured run built its profile from defaults instead of [hardware]"
        )
        assert overrides.num_streams == 3
        assert overrides.encode_threads == 5
