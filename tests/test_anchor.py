"""Measured-anchor contract tests: schema validation, strict matching,
resolution priority (config > measured > formula), banner honesty (plan §3/§4)."""

import json
from pathlib import Path

import pytest

from anime_upscale.config import HardwareOverrides
from anime_upscale.hardware import (
    GPUInfo,
    HardwareProfile,
    detect_hardware,
    gpu_slug,
    resolve_pools_from_anchor,
    resolve_streams_from_anchor,
    validate_anchor,
)

GPU_NAME = "NVIDIA GeForce RTX 5080"
SLUG = "GeForceRTX5080"
DRIVER = "580.88"


def _anchor() -> dict:
    """Minimal valid anchor exactly per plan §3 schema (all keys, real shapes)."""
    return {
        "schema_version": 1,
        "date": "2026-11-06",
        "gpu_name": GPU_NAME,
        "gpu_vram_mb": 16384,
        "driver": DRIVER,
        "cuda": "12.9",
        "torch": "2.14.1+cu126",
        "pipeline_commit": "bfd9b5f",
        "run_config": {
            "preset": "medium",
            "crf": 16,
            "interp_format": "rgbs",
            "trt": False,
            "chains": {"streams": "full-chain", "pools": "encode-only"},
        },
        "clip": {
            "file": "sample.mkv",
            "frames_in": 1440,
            "src_fps": "24000/1001",
            "height": 1080,
            "seconds": 60.0,
        },
        "streams_sweep": [
            {
                "streams": s,
                "fps": fps,
                "fps_repeats": [fps - 0.2, fps + 0.2, fps],
                "vram_peak_mb": 2100.0 + 400.0 * s,
                "torch_reserved_mb": 1450.0 + 300.0 * s,
                "smi_delta_mb": 2100.0 + 400.0 * s,
            }
            for s, fps in ((1, 8.4), (2, 14.2), (3, 17.0), (4, 17.2))
        ],
        "pools_sweep": [
            {
                "pools": p,
                "fps": fps,
                "fps_repeats": [fps - 0.2, fps, fps + 0.2],
                "wall_s": 60.0,
                "rss_peak_mb": 1000.0 + 50 * p,
            }
            for p, fps in ((8, 22.0), (16, 38.5), (24, 41.0), (32, 41.1))
        ],
        "saturation_streams": 3,
        "saturation_pools": 24,
        "notes": "",
    }


def _write_anchor(root: Path, anchor: dict, slug: str = SLUG) -> Path:
    path = root / "benchmarks" / slug / "anchor.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(anchor), encoding="utf-8")
    return path


def _gpu(name: str = GPU_NAME, cc: tuple[int, int] = (12, 0)) -> GPUInfo:
    return GPUInfo(index=0, name=name, vram_mb=16384, compute_capability=cc)


def _detect(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **kwargs: object) -> HardwareProfile:
    monkeypatch.setattr("anime_upscale.hardware._probe_torch_gpus", lambda: [_gpu()])
    monkeypatch.setattr("anime_upscale.hardware._nvidia_smi_cross_check", lambda _d: None)
    if "_driver" not in kwargs:
        kwargs["_driver"] = DRIVER  # type: ignore[assignment]
    kwargs.setdefault("project_root", tmp_path)
    kwargs.setdefault("nn_stages_selected", True)
    kwargs.setdefault("encode_preset", "medium")  # full run_config comparability
    driver: str | None = kwargs.pop("_driver", None)  # type: ignore[assignment]
    monkeypatch.setattr("anime_upscale.hardware.query_driver_version", lambda: driver)
    return detect_hardware(
        overrides=HardwareOverrides(),
        cli_num_streams=None,
        **kwargs,  # type: ignore[arg-type]
    )


class TestValidateAnchor:
    def test_valid_minimal_anchor_has_no_problems(self) -> None:
        assert validate_anchor(_anchor()) == []

    def test_missing_key_is_named(self) -> None:
        anchor = _anchor()
        del anchor["gpu_name"]
        problems = validate_anchor(anchor)
        assert problems and "gpu_name" in problems[0]

    def test_extra_key_is_rejected(self) -> None:
        anchor = _anchor()
        anchor["junk"] = 1
        problems = validate_anchor(anchor)
        assert problems and "junk" in problems[0]

    def test_wrong_schema_version_rejected(self) -> None:
        anchor = _anchor()
        anchor["schema_version"] = 2
        problems = validate_anchor(anchor)
        assert problems and "schema_version" in problems[0]

    def test_duplicate_stream_points_rejected(self) -> None:
        anchor = _anchor()
        anchor["streams_sweep"][2] = dict(anchor["streams_sweep"][1])
        problems = validate_anchor(anchor)
        assert any("3" in p or "поток" in p or "streams" in p.lower() for p in problems)

    def test_non_monotonic_pools_rejected(self) -> None:
        anchor = _anchor()
        anchor["pools_sweep"] = list(reversed(anchor["pools_sweep"]))
        problems = validate_anchor(anchor)
        assert any("pools" in p.lower() for p in problems)

    def test_empty_fps_repeats_rejected(self) -> None:
        anchor = _anchor()
        anchor["streams_sweep"][0]["fps_repeats"] = []
        problems = validate_anchor(anchor)
        assert problems and any("fps_repeats" in p for p in problems)

    def test_non_positive_fps_rejected(self) -> None:
        anchor = _anchor()
        anchor["streams_sweep"][1]["fps"] = 0
        problems = validate_anchor(anchor)
        assert problems and any("fps" in p for p in problems)

    def test_missing_vram_peak_rejected(self) -> None:
        anchor = _anchor()
        anchor["streams_sweep"][2]["vram_peak_mb"] = None
        problems = validate_anchor(anchor)
        assert any("vram_peak_mb" in p for p in problems)

    def test_saturation_outside_sweep_rejected(self) -> None:
        anchor = _anchor()
        anchor["saturation_streams"] = 9
        problems = validate_anchor(anchor)
        assert any("saturation_streams" in p for p in problems)

    def test_null_saturation_with_nonempty_sweep_is_valid(self) -> None:
        """Saturation not reached within the swept range is a legal outcome."""
        anchor = _anchor()
        anchor["saturation_streams"] = None
        assert validate_anchor(anchor) == []

    def test_empty_gpu_name_rejected(self) -> None:
        anchor = _anchor()
        anchor["gpu_name"] = ""
        problems = validate_anchor(anchor)
        assert problems and any("gpu_name" in p for p in problems)

    def test_empty_driver_rejected(self) -> None:
        anchor = _anchor()
        anchor["driver"] = ""
        problems = validate_anchor(anchor)
        assert problems and any("driver" in p for p in problems)

    def test_single_sweep_only_is_valid(self) -> None:
        """Owner may run --sweep streams alone: pools side may be empty."""
        anchor = _anchor()
        anchor["pools_sweep"] = []
        anchor["saturation_pools"] = None
        assert validate_anchor(anchor) == []

    def test_wrong_typed_sweep_rejected(self) -> None:
        anchor = _anchor()
        anchor["streams_sweep"] = "nope"
        problems = validate_anchor(anchor)
        assert problems and any("streams_sweep" in p for p in problems)


class TestResolveStreams:
    def test_largest_point_within_budget_and_gain(self) -> None:
        # budget = 0.8*16384 - 1024 = 12083.2 MB; every point fits, fps(1)*1.05=8.82.
        assert resolve_streams_from_anchor(_anchor(), 16384) == 4

    def test_point_over_budget_excluded(self) -> None:
        anchor = _anchor()
        anchor["streams_sweep"][3]["vram_peak_mb"] = 99999.0
        assert resolve_streams_from_anchor(anchor, 16384) == 3

    def test_point_without_gain_excluded(self) -> None:
        anchor = _anchor()
        anchor["streams_sweep"][3]["fps"] = 8.5  # < 8.4*1.05
        assert resolve_streams_from_anchor(anchor, 16384) == 3

    def test_no_qualifying_point_returns_none(self) -> None:
        anchor = _anchor()
        anchor["streams_sweep"] = [dict(anchor["streams_sweep"][0])]
        anchor["streams_sweep"][0]["fps"] = 0.5  # fps(1)*1.05 = 0.525 > 0.5
        assert resolve_streams_from_anchor(anchor, 16384) is None

    def test_empty_streams_sweep_returns_none(self) -> None:
        anchor = _anchor()
        anchor["streams_sweep"] = []
        anchor["saturation_streams"] = None
        assert resolve_streams_from_anchor(anchor, 16384) is None


class TestResolvePools:
    def test_saturation_caps_cores_division(self) -> None:
        assert resolve_pools_from_anchor(_anchor(), cores=32, active_encoders=1) == 24

    def test_fewer_cores_than_saturation(self) -> None:
        assert resolve_pools_from_anchor(_anchor(), cores=8, active_encoders=1) == 8

    def test_multi_encoder_divides_then_caps(self) -> None:
        assert resolve_pools_from_anchor(_anchor(), cores=64, active_encoders=2) == 24

    def test_null_saturation_returns_none(self) -> None:
        anchor = _anchor()
        anchor["saturation_pools"] = None
        assert resolve_pools_from_anchor(anchor, cores=32, active_encoders=1) is None


class TestDetectIntegration:
    def test_no_anchor_file_formula_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        profile = _detect(tmp_path, monkeypatch)
        assert profile.num_streams_per_gpu == 2  # frozen VRAM formula for 16 GB
        assert profile.streams_source == "formula"
        assert profile.encode_threads_source == "formula"

    def test_valid_anchor_gives_measured_streams(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_anchor(tmp_path, _anchor())
        profile = _detect(tmp_path, monkeypatch)
        assert profile.num_streams_per_gpu == 4
        assert profile.streams_source == "measured"

    def test_valid_anchor_gives_measured_pools(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_anchor(tmp_path, _anchor())
        monkeypatch.setattr("anime_upscale.hardware.usable_cpu_count", lambda reporter=None: 32)
        profile = _detect(tmp_path, monkeypatch, encode_preset="medium")
        assert profile.encode_threads == 24
        assert profile.encode_threads_source == "measured"

    def test_gpu_name_mismatch_falls_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        anchor = _anchor()
        anchor["gpu_name"] = "NVIDIA GeForce RTX 5090"
        _write_anchor(tmp_path, anchor)
        profile = _detect(tmp_path, monkeypatch)
        assert profile.num_streams_per_gpu == 2
        assert profile.streams_source == "formula"

    def test_driver_mismatch_falls_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        anchor = _anchor()
        anchor["driver"] = "555.99"
        _write_anchor(tmp_path, anchor)
        profile = _detect(tmp_path, monkeypatch)
        assert profile.streams_source == "formula"
        assert profile.num_streams_per_gpu == 2

    def test_driver_unavailable_skips_driver_check_with_note(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_anchor(tmp_path, _anchor())
        lines: list[str] = []

        class Intercept:
            def note(self, text: str) -> None:
                lines.append(text)

        profile = _detect(tmp_path, monkeypatch, _driver=None, reporter=Intercept())  # type: ignore[arg-type]
        assert profile.streams_source == "measured"
        assert any("драйвер" in line.lower() for line in lines)

    def test_trt_flag_mismatch_falls_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        anchor = _anchor()
        anchor["run_config"]["trt"] = True  # would-be run recommends trt=False on Blackwell
        _write_anchor(tmp_path, anchor)
        profile = _detect(tmp_path, monkeypatch)
        assert profile.streams_source == "formula"

    def test_interp_format_mismatch_falls_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        anchor = _anchor()
        anchor["run_config"]["interp_format"] = "rgbh"  # Blackwell run recommends rgbs
        _write_anchor(tmp_path, anchor)
        profile = _detect(tmp_path, monkeypatch)
        assert profile.streams_source == "formula"

    def test_preset_mismatch_falls_back_wholesale(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Strict invalidation: any run_config mismatch (incl. preset) ignores
        the WHOLE anchor — no partial degradation (dispatch canon)."""
        anchor = _anchor()
        anchor["run_config"]["preset"] = "slow"
        _write_anchor(tmp_path, anchor)
        monkeypatch.setattr("anime_upscale.hardware.usable_cpu_count", lambda reporter=None: 32)
        profile = _detect(tmp_path, monkeypatch, encode_preset="medium")
        assert profile.num_streams_per_gpu == 2
        assert profile.streams_source == "formula"
        assert profile.encode_threads_source == "formula"
        assert profile.encode_threads == 32  # cores // encoders, no anchor

    def test_no_preset_argument_ignores_anchor_wholesale(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_anchor(tmp_path, _anchor())
        monkeypatch.setattr("anime_upscale.hardware.usable_cpu_count", lambda reporter=None: 32)
        profile = _detect(tmp_path, monkeypatch, encode_preset=None)
        assert profile.streams_source == "formula"
        assert profile.encode_threads_source == "formula"

    def test_config_override_beats_anchor(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_anchor(tmp_path, _anchor())
        monkeypatch.setattr("anime_upscale.hardware._probe_torch_gpus", lambda: [_gpu()])
        monkeypatch.setattr("anime_upscale.hardware._nvidia_smi_cross_check", lambda _d: None)
        monkeypatch.setattr("anime_upscale.hardware.query_driver_version", lambda: DRIVER)
        profile = detect_hardware(
            overrides=HardwareOverrides(num_streams=1, encode_threads=4),
            cli_num_streams=None,
            nn_stages_selected=True,
            project_root=tmp_path,
            encode_preset="medium",
        )
        assert profile.num_streams_per_gpu == 1
        assert profile.streams_source == "config"
        assert profile.encode_threads == 4
        assert profile.encode_threads_source == "config"

    def test_cli_override_beats_anchor(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_anchor(tmp_path, _anchor())
        monkeypatch.setattr("anime_upscale.hardware._probe_torch_gpus", lambda: [_gpu()])
        monkeypatch.setattr("anime_upscale.hardware._nvidia_smi_cross_check", lambda _d: None)
        monkeypatch.setattr("anime_upscale.hardware.query_driver_version", lambda: DRIVER)
        profile = detect_hardware(
            overrides=HardwareOverrides(num_streams=1),
            cli_num_streams=5,
            nn_stages_selected=True,
            project_root=tmp_path,
        )
        assert profile.num_streams_per_gpu == 5
        assert profile.streams_source == "config"

    def test_no_nn_stages_no_anchor_read(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        anchor = _anchor()
        anchor["gpu_name"] = "MISMATCH"  # would invalidate, but it must not even be read
        _write_anchor(tmp_path, anchor)
        profile = _detect(tmp_path, monkeypatch, nn_stages_selected=False)
        assert profile.gpu_count == 1
        assert profile.streams_source == "formula"
        assert profile.num_streams_per_gpu == 2

    def test_corrupt_json_falls_back_loudly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "benchmarks" / SLUG / "anchor.json"
        path.parent.mkdir(parents=True)
        path.write_text("{not json", encoding="utf-8")
        lines: list[str] = []

        class Intercept:
            def note(self, text: str) -> None:
                lines.append(text)

        profile = _detect(tmp_path, monkeypatch, reporter=Intercept())  # type: ignore[arg-type]
        assert profile.streams_source == "formula"
        assert lines, "corrupt anchor must produce a loud note"

    def test_invalid_schema_falls_back_with_named_reason(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        anchor = _anchor()
        del anchor["schema_version"]
        _write_anchor(tmp_path, anchor)
        lines: list[str] = []

        class Intercept:
            def note(self, text: str) -> None:
                lines.append(text)

        profile = _detect(tmp_path, monkeypatch, reporter=Intercept())  # type: ignore[arg-type]
        assert profile.streams_source == "formula"
        assert any("schema_version" in line for line in lines)


class TestBannerSource:
    def test_banner_names_measured_source(self) -> None:
        from anime_upscale.hardware import print_banner
        from anime_upscale.reporting import Reporter, resolve_mode

        lines: list[str] = []

        class InterceptReporter(Reporter):
            def hardware_banner(self, gpus: str, streams: str, stages: str, encode: str) -> None:
                lines.append(f"{streams} | {encode}")
                super().hardware_banner(gpus, streams, stages, encode)

        reporter = InterceptReporter(resolve_mode(no_color=True, no_tty=True))
        profile = HardwareProfile(
            gpus=(_gpu(),),
            gpu_count=1,
            cpu_logical_cores=32,
            num_streams_per_gpu=4,
            encode_threads=24,
            precision="fp16",
            streams_source="measured",
            encode_threads_source="measured",
        )
        print_banner(reporter, profile, stages=("interpolate", "encode"))
        assert any("(замер" in line and "anchor.json" in line for line in lines)
        assert any("24" in line.split("|")[1] and "замер" in line.split("|")[1] for line in lines)

    def test_banner_names_formula_and_config_sources(self) -> None:
        from anime_upscale.hardware import print_banner
        from anime_upscale.reporting import Reporter, resolve_mode

        lines: list[str] = []

        class InterceptReporter(Reporter):
            def hardware_banner(self, gpus: str, streams: str, stages: str, encode: str) -> None:
                lines.append(f"{streams} | {encode}")

        reporter = InterceptReporter(resolve_mode(no_color=True, no_tty=True))
        base = {
            "gpus": (_gpu(),),
            "gpu_count": 1,
            "cpu_logical_cores": 32,
            "precision": "fp16",
        }
        print_banner(
            reporter,
            HardwareProfile(num_streams_per_gpu=2, encode_threads=32, **base),
            stages=("encode",),
        )
        print_banner(
            reporter,
            HardwareProfile(
                num_streams_per_gpu=1,
                encode_threads=8,
                **base,
                streams_source="config",
                encode_threads_source="config",
            ),
            stages=("encode",),
        )
        assert any("формула" in line for line in lines)
        assert any("конфиг" in line for line in lines)


class TestSlug:
    def test_slug_matches_benchmark_derivation(self) -> None:
        assert gpu_slug(GPU_NAME) == SLUG

    def test_slug_fallback(self) -> None:
        assert gpu_slug("") == "GPU"
