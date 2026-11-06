"""Benchmark sweep tests: planner, saturation, frozen-profile replacement,
dry-run, anchor writing, check-anchor, synthetic-clip fallback (plan §3/§4)."""

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from anime_upscale.config import RunConfig
from anime_upscale.hardware import GPUInfo, HardwareProfile, validate_anchor

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "benchmark.py"


def _load_benchmark() -> ModuleType:
    spec = importlib.util.spec_from_file_location("benchmark_sweeps_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fake_hw(num_streams: int = 1, encode_threads: int = 8) -> HardwareProfile:
    return HardwareProfile(
        gpus=(GPUInfo(index=0, name="FAKE GPU X", vram_mb=16384, compute_capability=(12, 0)),),
        gpu_count=1,
        cpu_logical_cores=32,
        num_streams_per_gpu=num_streams,
        encode_threads=encode_threads,
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


def _project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "input").mkdir()
    (tmp_path / "output").mkdir()
    (tmp_path / "input" / "sample.mkv").write_bytes(b"clip")
    (tmp_path / "config.toml").write_text(
        "[pipeline]\n"
        'order = ["descale", "deband", "upscale", "interpolate", "encode"]\n'
        "[stages.encode]\n"
        "crf = 16\n"
        'preset = "medium"\n'
        "[profiles.enc]\n"
        'stages = ["encode"]\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _mock_passes(bench: ModuleType, monkeypatch: pytest.MonkeyPatch, seen: list[int]) -> None:
    """Deterministic clock: wall = 10 s × point index (point = every 4th pass)."""
    monkeypatch.setattr(bench, "detect_hardware", lambda *a, **k: _fake_hw())
    state = {"t": 0.0, "calls": 0, "wall": 10.0}

    def _clock() -> float:
        if state["calls"] % 2 == 0:
            state["wall"] = 10.0 * (len(seen) // 4 + 1)
            state["calls"] += 1
            return state["t"]
        state["calls"] += 1
        state["t"] += state["wall"]
        return state["t"]

    monkeypatch.setattr(bench.time, "monotonic", _clock)
    monkeypatch.setattr(bench, "_count_frames", lambda path: 30)
    monkeypatch.setattr(bench, "_query_smi_used_mb", lambda: 600.0)
    monkeypatch.setattr(bench, "_reset_torch_peak", lambda: None)
    monkeypatch.setattr(
        bench, "_torch_reserved_peak_mb", lambda: 1000.0 * (max(seen) if seen else 1)
    )
    monkeypatch.setattr(bench, "_children_rss_peak_mb", lambda: 100.0)
    monkeypatch.setattr(
        bench,
        "_probe_clip",
        lambda path, ffmpeg_path=None: {
            "file": path.name,
            "frames_in": 1440,
            "src_fps": "24000/1001",
            "height": 1080,
            "seconds": 60.0,
        },
    )

    def _fake_pipeline(
        ctx: object, src: object, out: Path, ffmpeg_path: object, progress_update: object = None
    ) -> None:
        hw = getattr(ctx, "hardware")  # noqa: B009 - duck-typed in test
        seen.append(hw.num_streams_per_gpu)
        Path(out).write_bytes(b"r")

    monkeypatch.setattr(bench, "run_pipeline", _fake_pipeline)


class TestPlanner:
    def test_stream_points_cover_1_to_max(self) -> None:
        bench = _load_benchmark()
        assert bench.plan_stream_points(6) == [1, 2, 3, 4, 5, 6]
        assert bench.plan_stream_points(1) == [1]

    def test_stream_points_reject_non_positive(self) -> None:
        bench = _load_benchmark()
        with pytest.raises(ValueError, match="max-streams"):
            bench.plan_stream_points(0)

    def test_pool_points_parse_and_sort(self) -> None:
        bench = _load_benchmark()
        assert bench.plan_pool_points("8,16,24,32") == [8, 16, 24, 32]
        assert bench.plan_pool_points("32, 8") == [8, 32]

    def test_pool_points_reject_garbage(self) -> None:
        bench = _load_benchmark()
        with pytest.raises(ValueError, match="abc"):
            bench.plan_pool_points("8,abc")
        with pytest.raises(ValueError, match="дубли"):
            bench.plan_pool_points("16,16")


class TestSaturation:
    def test_first_sub_threshold_gain_is_saturation(self) -> None:
        bench = _load_benchmark()
        points = [(1, 10.0), (2, 20.0), (3, 30.0), (4, 30.5)]  # 30.5/30 < 1.03
        assert bench.find_saturation(points) == 3

    def test_still_rising_at_edge_is_none(self) -> None:
        bench = _load_benchmark()
        assert bench.find_saturation([(1, 10.0), (2, 20.0), (3, 30.0), (4, 40.0)]) is None

    def test_noisy_plateau_is_stable(self) -> None:
        bench = _load_benchmark()
        points = [(1, 10.0), (2, 19.7), (3, 20.1), (4, 20.3)]  # ±2% noise band
        assert bench.find_saturation(points) == 2

    def test_single_point_is_none(self) -> None:
        bench = _load_benchmark()
        assert bench.find_saturation([(1, 10.0)]) is None


class TestFrozenReplacement:
    def test_replace_leaves_original_untouched(self) -> None:
        hw = _fake_hw(num_streams=1)
        replaced = dataclasses.replace(hw, num_streams_per_gpu=3)
        assert hw.num_streams_per_gpu == 1
        assert replaced is not hw
        assert replaced.num_streams_per_gpu == 3
        with pytest.raises(dataclasses.FrozenInstanceError):
            hw.num_streams_per_gpu = 5  # type: ignore[misc]


class TestDryRun:
    def test_dry_run_prints_plan_and_never_runs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _project(tmp_path, monkeypatch)
        bench = _load_benchmark()
        monkeypatch.setattr(
            sys, "argv", ["benchmark.py", "--sweep", "streams", "--max-streams", "3", "--dry-run"]
        )

        def _boom(*a: object, **k: object) -> None:
            raise AssertionError("dry-run must not execute the pipeline")

        monkeypatch.setattr(bench, "run_pipeline", _boom)
        assert bench.main() == 0
        out = capsys.readouterr().out
        assert "1" in out and "3" in out
        assert "8" in out or "поток" in out or "stream" in out.lower()

    def test_dry_run_pools_plans_pool_list(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _project(tmp_path, monkeypatch)
        bench = _load_benchmark()
        monkeypatch.setattr(sys, "argv", ["benchmark.py", "--sweep", "pools", "--dry-run"])

        def _boom(*a: object, **k: object) -> None:
            raise AssertionError("dry-run must not execute the pipeline")

        monkeypatch.setattr(bench, "run_pipeline", _boom)
        assert bench.main() == 0
        out = capsys.readouterr().out
        for pool in ("8", "16", "24", "32"):
            assert pool in out


class TestStreamsSweepEndToEnd:
    def test_sweep_replaces_profile_per_point_and_writes_anchor(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _project(tmp_path, monkeypatch)
        bench = _load_benchmark()
        seen: list[int] = []
        _mock_passes(bench, monkeypatch, seen)
        monkeypatch.setattr(
            sys, "argv", ["benchmark.py", "--sweep", "streams", "--max-streams", "2"]
        )
        assert bench.main() == 0
        # warm-up + 3 repeats per point, but the warm-up is a run too (the mock
        # sees every pass): 2 points × 4 passes = 8 pipeline calls.
        assert len(seen) == 8
        assert set(seen) == {1, 2}  # replaced profiles only — never the base value 0/other

        anchor_path = root / "benchmarks" / "FAKEGPUX" / "anchor.json"
        assert anchor_path.is_file()
        anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
        assert validate_anchor(anchor) == []
        assert [p["streams"] for p in anchor["streams_sweep"]] == [1, 2]
        assert all(len(p["fps_repeats"]) == 3 for p in anchor["streams_sweep"])
        assert anchor["run_config"]["preset"] == "medium"
        assert anchor["run_config"]["crf"] == 16
        assert anchor["run_config"]["trt"] is False  # Blackwell fake → recommended_trt False
        assert anchor["run_config"]["interp_format"] == "rgbs"
        # Deterministic clock: point1 fps median = 30/10 = 3.0; point2 = 30/20 = 1.5
        assert anchor["streams_sweep"][0]["fps"] == pytest.approx(3.0)
        assert anchor["streams_sweep"][1]["fps"] == pytest.approx(1.5)
        assert anchor["saturation_streams"] == 1  # 1.5 < 3.0 × 1.03
        assert anchor["gpu_name"] == "FAKE GPU X"
        assert anchor["gpu_vram_mb"] == 16384

    def test_missing_clip_falls_back_to_documented_synthetic(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _project(tmp_path, monkeypatch)
        (root / "input" / "sample.mkv").unlink()
        bench = _load_benchmark()
        seen: list[int] = []
        _mock_passes(bench, monkeypatch, seen)
        generated: list[Path] = []

        def _fake_generate(input_path: Path, ffmpeg_path: object, reporter: object) -> Path:
            path = root / "input" / "sample.mkv"
            path.write_bytes(b"synthetic")
            generated.append(path)
            return path

        monkeypatch.setattr(bench, "_generate_synthetic_clip", _fake_generate)
        monkeypatch.setattr(
            sys, "argv", ["benchmark.py", "--sweep", "streams", "--max-streams", "1"]
        )
        assert bench.main() == 0
        assert generated == [root / "input" / "sample.mkv"]
        anchor = json.loads(
            (root / "benchmarks" / "FAKEGPUX" / "anchor.json").read_text(encoding="utf-8")
        )
        assert "синтетический" in anchor["notes"].lower()


class TestPoolsSweepEndToEnd:
    def test_pools_sweep_replaces_encode_threads_and_writes_anchor(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _project(tmp_path, monkeypatch)
        bench = _load_benchmark()
        seen: list[int] = []
        _mock_passes(bench, monkeypatch, seen)
        pools_seen: list[int] = []

        def _spy_pipeline(
            ctx: object, src: object, out: Path, ffmpeg_path: object, progress_update: object = None
        ) -> None:
            hw = getattr(ctx, "hardware")  # noqa: B009
            pools_seen.append(hw.encode_threads)
            Path(out).write_bytes(b"r")

        monkeypatch.setattr(bench, "run_pipeline", _spy_pipeline)
        monkeypatch.setattr(
            sys, "argv", ["benchmark.py", "--sweep", "pools", "--pool-list", "8,16"]
        )
        assert bench.main() == 0
        assert set(pools_seen) == {8, 16}
        assert len(pools_seen) == 8  # 2 points × (1 warm-up + 3 repeats)

        anchor = json.loads(
            (root / "benchmarks" / "FAKEGPUX" / "anchor.json").read_text(encoding="utf-8")
        )
        assert validate_anchor(anchor) == []
        assert [p["pools"] for p in anchor["pools_sweep"]] == [8, 16]
        assert anchor["saturation_pools"] in (8, 16, None)  # deterministic-clock outcome recorded
        assert anchor["streams_sweep"] == []  # pools-only run: streams side untouched
        assert anchor["saturation_streams"] is None


class TestCheckAnchor:
    def test_check_anchor_valid_prints_resolution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = _project(tmp_path, monkeypatch)
        bench = _load_benchmark()
        seen: list[int] = []
        _mock_passes(bench, monkeypatch, seen)
        monkeypatch.setattr(
            sys, "argv", ["benchmark.py", "--sweep", "streams", "--max-streams", "2"]
        )
        bench.main()
        anchor_path = root / "benchmarks" / "FAKEGPUX" / "anchor.json"

        monkeypatch.setattr(sys, "argv", ["benchmark.py", "--check-anchor", str(anchor_path)])
        assert bench.main() == 0
        out = capsys.readouterr().out
        assert "streams" in out
        assert "pools" in out

    def test_check_anchor_invalid_is_loud_nonzero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _project(tmp_path, monkeypatch)
        bench = _load_benchmark()
        bad = tmp_path / "bad.json"
        bad.write_text('{"schema_version": 2}', encoding="utf-8")
        monkeypatch.setattr(sys, "argv", ["benchmark.py", "--check-anchor", str(bad)])
        assert bench.main() == 1
        out = capsys.readouterr().out
        assert "schema_version" in out
