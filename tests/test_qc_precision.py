"""
QC precision harness tests: fixtures, metrics, verdicts, cut-binding, prod-call parity.

CPU-safe by construction: scripts/qc_precision.py keeps every VS/torch/vsrife
import lazy (inside functions), so this module imports and runs everywhere.
GPU seams are exercised with injected fakes / importorskip-guarded VS probes
(VS is CPU-only); the real RIFE run stays owner acceptance on the 5080.
"""

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "qc_precision.py"


def _load_qc() -> ModuleType:
    spec = importlib.util.spec_from_file_location("qc_precision_script_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def qc() -> ModuleType:
    return _load_qc()


# ---------------------------------------------------------------------------
# Verdict thresholds (constants at the top of the script, per dispatch canon)
# ---------------------------------------------------------------------------


class TestThresholdConstants:
    def test_threshold_values_are_the_dispatch_canon(self, qc: ModuleType) -> None:
        assert qc.PASS_GLOBAL_PSNR_DB == 35.0
        assert qc.PASS_WORST_PSNR_DB == 28.0
        assert qc.PASS_MIN_FPS_GAIN_PCT == 10.0

    def test_cut_risk_window_defined(self, qc: ModuleType) -> None:
        assert qc.CUT_RISK_FRAMES >= 1


# ---------------------------------------------------------------------------
# Fixture plans: deterministic clip specifications with known cut indices
# ---------------------------------------------------------------------------


class TestFixturePlans:
    def test_three_named_fixtures_exist(self, qc: ModuleType) -> None:
        for name in ("hard_cuts", "pan_motion", "mixed"):
            plan = qc.fixture_plan(name)
            assert plan.name == name

    def test_hard_cuts_plan_cut_indices(self, qc: ModuleType) -> None:
        plan = qc.fixture_plan("hard_cuts", frames_src=144)
        # ~1 s at 24000/1001 fps -> a cut every 24 source frames
        assert plan.cut_indices == (24, 48, 72, 96, 120)
        assert plan.frames_src == 144

    def test_pan_motion_plan_has_no_cuts(self, qc: ModuleType) -> None:
        plan = qc.fixture_plan("pan_motion", frames_src=144)
        assert plan.cut_indices == ()

    def test_mixed_plan_rare_cuts(self, qc: ModuleType) -> None:
        plan = qc.fixture_plan("mixed", frames_src=144)
        # ~2 s apart: fewer, rarer cuts than hard_cuts
        assert plan.cut_indices == (48, 96)

    def test_plan_is_deterministic(self, qc: ModuleType) -> None:
        assert qc.fixture_plan("hard_cuts", frames_src=144) == qc.fixture_plan(
            "hard_cuts", frames_src=144
        )
        assert qc.fixture_plan("mixed", frames_src=144) == qc.fixture_plan("mixed", frames_src=144)

    def test_unknown_fixture_name_is_loud(self, qc: ModuleType) -> None:
        with pytest.raises(ValueError, match=r"hard_cuts|pan_motion|mixed"):
            qc.fixture_plan("no_such_fixture")

    def test_seconds_to_frames_conversion(self, qc: ModuleType) -> None:
        # 5..8 s band from the dispatch; 24000/1001 fps source.
        assert qc.seconds_to_frames(6) == 144
        assert qc.seconds_to_frames(5) == 120
        assert qc.seconds_to_frames(8) == 192


# ---------------------------------------------------------------------------
# Frame content: pure-arithmetic, seeded-by-plan (no RNG) -> bit-identical
# ---------------------------------------------------------------------------


class TestFrameContent:
    def test_same_inputs_identical_frames(self, qc: ModuleType) -> None:
        plan = qc.fixture_plan("mixed", frames_src=144)
        a = qc.frame_content(plan, 64, 32, 7)
        b = qc.frame_content(plan, 64, 32, 7)
        assert a.dtype == np.float32
        assert np.array_equal(a, b)

    def test_shape_and_range(self, qc: ModuleType) -> None:
        plan = qc.fixture_plan("pan_motion", frames_src=144)
        frame = qc.frame_content(plan, 64, 32, 3)
        assert frame.shape == (3, 32, 64)
        assert float(frame.min()) >= 0.0
        assert float(frame.max()) <= 1.0

    def test_cut_flips_the_plate(self, qc: ModuleType) -> None:
        plan = qc.fixture_plan("hard_cuts", frames_src=144)
        before = qc.frame_content(plan, 64, 32, 23)
        after = qc.frame_content(plan, 64, 32, 24)
        # a full-plate flip: mean abs diff must be large enough to trip
        # vsrife's internal sc_threshold=0.15 scene detection
        assert float(np.abs(after - before).mean()) > 0.15

    def test_pan_motion_moves_every_frame(self, qc: ModuleType) -> None:
        plan = qc.fixture_plan("pan_motion", frames_src=144)
        f0 = qc.frame_content(plan, 64, 32, 10)
        f1 = qc.frame_content(plan, 64, 32, 11)
        mad = float(np.abs(f1 - f0).mean())
        assert mad > 0.0  # motion exists
        assert mad < 0.5  # ...but it is a translate, not a cut

    def test_plans_differ_in_content(self, qc: ModuleType) -> None:
        hard = qc.fixture_plan("hard_cuts", frames_src=144)
        pan = qc.fixture_plan("pan_motion", frames_src=144)
        assert not np.array_equal(
            qc.frame_content(hard, 64, 32, 5), qc.frame_content(pan, 64, 32, 5)
        )


# ---------------------------------------------------------------------------
# Metrics: PSNR / MAD with known answers (float peak = 1.0 RGB)
# ---------------------------------------------------------------------------


class TestFrameMetrics:
    def test_identical_frames_infinite_psnr(self, qc: ModuleType) -> None:
        frame = np.full((3, 8, 8), 0.37, dtype=np.float32)
        m = qc.frame_metrics(frame, frame.copy())
        assert m.mse == 0.0
        assert m.mad == 0.0
        assert m.psnr_db == math.inf

    def test_known_constant_offset(self, qc: ModuleType) -> None:
        ref = np.zeros((3, 4, 4), dtype=np.float32)
        cand = np.full((3, 4, 4), 0.25, dtype=np.float32)
        m = qc.frame_metrics(ref, cand)
        assert m.mse == pytest.approx(0.0625)
        assert m.mad == pytest.approx(0.25)
        assert m.psnr_db == pytest.approx(10 * math.log10(1.0 / 0.0625), rel=1e-6)

    def test_fp16_candidate_cast_before_math(self, qc: ModuleType) -> None:
        ref = np.zeros((3, 4, 4), dtype=np.float32)
        cand16 = np.full((3, 4, 4), np.float16(0.25))
        m = qc.frame_metrics(ref, cand16)
        assert m.mse == pytest.approx(0.0625, rel=1e-6)
        assert m.psnr_db == pytest.approx(12.0412, abs=1e-3)

    def test_shape_mismatch_is_loud(self, qc: ModuleType) -> None:
        with pytest.raises(ValueError, match="shape"):
            qc.frame_metrics(np.zeros((3, 4, 4), dtype=np.float32), np.zeros((3, 8, 8)))


class TestAggregateMetrics:
    def test_global_psnr_from_equal_frames(self, qc: ModuleType) -> None:
        # two frames, each exactly the 0.25 constant offset -> global == per-frame
        ref = np.zeros((3, 4, 4), dtype=np.float32)
        cand = np.full((3, 4, 4), 0.25, dtype=np.float32)
        frames = [(ref, cand), (ref, cand.copy())]
        agg = qc.aggregate_frame_metrics(frames)
        assert agg.global_psnr_db == pytest.approx(10 * math.log10(1.0 / 0.0625), rel=1e-6)
        assert agg.mean_mad == pytest.approx(0.25)
        assert agg.worst.psnr_db == pytest.approx(10 * math.log10(1.0 / 0.0625), rel=1e-6)

    def test_worst_frame_identified_by_max_mse(self, qc: ModuleType) -> None:
        ref = np.zeros((3, 4, 4), dtype=np.float32)
        small = np.full((3, 4, 4), 0.05, dtype=np.float32)
        large = np.full((3, 4, 4), 0.4, dtype=np.float32)
        frames = [(ref, small), (ref, large), (ref, small)]
        agg = qc.aggregate_frame_metrics(frames)
        assert agg.worst.index == 1
        assert agg.worst.mad == pytest.approx(0.4)

    def test_global_psnr_infinite_when_all_identical(self, qc: ModuleType) -> None:
        frame = np.zeros((3, 4, 4), dtype=np.float32)
        agg = qc.aggregate_frame_metrics([(frame, frame.copy())])
        assert agg.global_psnr_db == math.inf


# ---------------------------------------------------------------------------
# Worst-frame <-> scene-cut binding
# ---------------------------------------------------------------------------


class TestCutBinding:
    def test_distance_to_nearest_cut(self, qc: ModuleType) -> None:
        cuts = (24, 48, 72, 96, 120)
        ratio = 2.5  # 59.94 from 23.976
        assert qc.distance_to_nearest_cut(60, cuts, ratio) == pytest.approx(0.0)
        assert qc.distance_to_nearest_cut(61, cuts, ratio) == pytest.approx(61 / 2.5 - 24)
        assert qc.distance_to_nearest_cut(58, cuts, ratio) == pytest.approx(24 - 58 / 2.5)
        assert qc.distance_to_nearest_cut(0, (), ratio) == math.inf

    def test_cut_risk_flag_window(self, qc: ModuleType) -> None:
        cuts = (24, 48, 72, 96, 120)
        ratio = 2.5
        assert qc.is_cut_risk(60, cuts, ratio)  # exactly at the cut
        assert qc.is_cut_risk(62, cuts, ratio)  # 0.8 src frames away
        assert not qc.is_cut_risk(40, cuts, ratio)  # far from any cut

    def test_worst_table_rows_carry_distance(self, qc: ModuleType) -> None:
        ref = np.zeros((3, 4, 4), dtype=np.float32)
        frames = [
            (ref, np.full((3, 4, 4), 0.1, dtype=np.float32)),
            (ref, np.full((3, 4, 4), 0.4, dtype=np.float32)),
            (ref, np.full((3, 4, 4), 0.2, dtype=np.float32)),
        ]
        agg = qc.aggregate_frame_metrics(frames)
        rows = qc.worst_frame_table(agg.per_frame, cut_indices=(1,), src_ratio=2.5, top=2)
        assert len(rows) == 2
        assert rows[0]["index"] == 1
        assert rows[0]["dist_to_cut_src"] == pytest.approx(0.6)  # |1/2.5 - 1|
        assert rows[0]["psnr_db"] < rows[1]["psnr_db"]  # sorted worst-first


# ---------------------------------------------------------------------------
# Verdict logic (threshold cases from the dispatch brief)
# ---------------------------------------------------------------------------


class TestVerdict:
    def test_pass(self, qc: ModuleType) -> None:
        verdict, _ = qc.decide_verdict(38.0, 30.0, 12.0)
        assert verdict == "PASS"

    def test_pass_boundaries_are_inclusive(self, qc: ModuleType) -> None:
        verdict, _ = qc.decide_verdict(35.0, 28.0, 10.0)
        assert verdict == "PASS"

    def test_fail_when_global_psnr_low(self, qc: ModuleType) -> None:
        verdict, reason = qc.decide_verdict(30.0, 30.0, 50.0)
        assert verdict == "FAIL"
        assert "35" in reason

    def test_fail_when_worst_frame_low(self, qc: ModuleType) -> None:
        verdict, reason = qc.decide_verdict(40.0, 20.0, 50.0)
        assert verdict == "FAIL"
        assert "28" in reason

    def test_insufficient_gain(self, qc: ModuleType) -> None:
        verdict, reason = qc.decide_verdict(38.0, 30.0, 5.0)
        assert verdict == "INSUFFICIENT-GAIN"
        assert "10" in reason

    def test_insufficient_gain_when_slower(self, qc: ModuleType) -> None:
        verdict, _ = qc.decide_verdict(38.0, 30.0, -3.0)
        assert verdict == "INSUFFICIENT-GAIN"

    def test_quality_failure_outranks_gain(self, qc: ModuleType) -> None:
        verdict, _ = qc.decide_verdict(20.0, 20.0, 1.0)
        assert verdict == "FAIL"

    def test_infinite_psnr_passes(self, qc: ModuleType) -> None:
        verdict, _ = qc.decide_verdict(math.inf, math.inf, 25.0)
        assert verdict == "PASS"


class TestWorstCaseAggregate:
    def test_mins_across_fixtures(self, qc: ModuleType) -> None:
        fixtures = [
            {"global_psnr_db": 40.0, "worst_frame": {"psnr_db": 30.0}, "fps_delta_pct": 20.0},
            {"global_psnr_db": 36.0, "worst_frame": {"psnr_db": 29.0}, "fps_delta_pct": 12.0},
        ]
        g, w, gain = qc.worst_case_aggregate(fixtures)
        assert (g, w, gain) == (36.0, 29.0, 12.0)

    def test_empty_fixtures_is_loud(self, qc: ModuleType) -> None:
        with pytest.raises(ValueError, match="фикстур"):
            qc.worst_case_aggregate([])


# ---------------------------------------------------------------------------
# Production-call parity: the RIFE kwargs must mirror stages/interpolate.py
# ---------------------------------------------------------------------------


class TestProdCallParity:
    def test_kwargs_mirror_interpolate_stage(self, qc: ModuleType, tmp_path: Path) -> None:
        pytest.importorskip("anime_upscale.config")
        from anime_upscale.config import InterpolateConfig

        cfg = InterpolateConfig()
        kwargs = qc.build_rife_kwargs(
            clip_width=1920,
            clip_height=1080,
            interpolate_cfg=cfg,
            device_index=0,
            trt_allowed=False,  # Blackwell gate result (rule not touched)
            cache_dir=tmp_path / "vsrife",
        )
        assert kwargs == {
            "device_index": 0,
            "model": cfg.model,
            "auto_download": True,
            "fps_num": cfg.target_fps[0],
            "fps_den": cfg.target_fps[1],
            "sc": cfg.scene_detection,
            "sc_threshold": cfg.scene_threshold,
            "trt": False,
            "trt_static_shape": True,
            "trt_opt_shape": [1080, 1920],
            "trt_cache_dir": str(tmp_path / "vsrife"),
        }

    def test_trt_allowed_when_hardware_says_so(self, qc: ModuleType, tmp_path: Path) -> None:
        from anime_upscale.config import InterpolateConfig

        cfg = InterpolateConfig()
        kwargs = qc.build_rife_kwargs(
            clip_width=64,
            clip_height=32,
            interpolate_cfg=cfg,
            device_index=1,
            trt_allowed=True,
            cache_dir=tmp_path / "vsrife",
        )
        assert kwargs["trt"] is True
        assert kwargs["device_index"] == 1

    def test_scene_detection_off_nulls_threshold(self, qc: ModuleType, tmp_path: Path) -> None:
        from anime_upscale.config import InterpolateConfig

        cfg = InterpolateConfig(scene_detection=False)
        kwargs = qc.build_rife_kwargs(
            clip_width=64,
            clip_height=32,
            interpolate_cfg=cfg,
            device_index=0,
            trt_allowed=False,
            cache_dir=tmp_path / "vsrife",
        )
        assert kwargs["sc"] is False
        assert kwargs["sc_threshold"] is None

    def test_trt_gate_mirrors_production(self, qc: ModuleType, tmp_path: Path) -> None:
        """The production trt gate: cfg.trt AND hardware.recommended_trt."""
        from anime_upscale.config import InterpolateConfig

        cfg_trt_off = InterpolateConfig(trt=False)
        kwargs = qc.build_rife_kwargs(
            clip_width=64,
            clip_height=32,
            interpolate_cfg=cfg_trt_off,
            device_index=0,
            trt_allowed=True,
            cache_dir=tmp_path / "vsrife",
        )
        assert kwargs["trt"] is False


# ---------------------------------------------------------------------------
# Frame-count identity (no dropped/duplicated frames)
# ---------------------------------------------------------------------------


class TestFrameIdentity:
    def test_equal_counts_pass(self, qc: ModuleType) -> None:
        qc.ensure_frame_identity(360, 360)

    def test_mismatch_is_loud(self, qc: ModuleType) -> None:
        with pytest.raises(RuntimeError, match="кадр"):
            qc.ensure_frame_identity(360, 359)


# ---------------------------------------------------------------------------
# Report schema + JSON serialization (inf -> null)
# ---------------------------------------------------------------------------


class TestReportSchema:
    def _fixture_record(self) -> dict:
        return {
            "name": "hard_cuts",
            "frames_src": 144,
            "frames_out": 360,
            "cut_indices": [24, 48, 72, 96, 120],
            "global_psnr_db": 38.25,
            "mean_mad": 0.000123,
            "worst_frame": {
                "index": 61,
                "psnr_db": 29.4,
                "mad": 0.0042,
                "dist_to_cut_src": 0.4,
            },
            "worst_table": [],
            "fps_ref": 41.2,
            "fps_ref_repeats": [41.0, 41.2, 41.4],
            "fps_cand": 47.8,
            "fps_cand_repeats": [47.6, 47.8, 48.0],
            "fps_delta_pct": 16.02,
            "vram_peak_mb_ref": 512.0,
            "vram_peak_mb_cand": 480.0,
        }

    def test_report_top_level_keys(self, qc: ModuleType) -> None:
        report = qc.build_qc_report(
            gpu_name="NVIDIA GeForce RTX 5080",
            fixtures=[self._fixture_record()],
            run_config={"candidate_format": "rgbh"},
            notes=[],
        )
        for key in (
            "schema_version",
            "date",
            "gpu_name",
            "run_config",
            "fixtures",
            "fps_delta_pct",
            "verdict",
            "verdict_reason",
            "thresholds",
        ):
            assert key in report
        assert report["schema_version"] == qc.QC_SCHEMA_VERSION == 1
        assert report["verdict"] == "PASS"
        assert report["fps_delta_pct"] == 16.02

    def test_json_serialization_maps_inf_to_null(self, qc: ModuleType) -> None:
        fixture = self._fixture_record()
        fixture["global_psnr_db"] = math.inf
        report = qc.build_qc_report(gpu_name="GPU", fixtures=[fixture], run_config={}, notes=[])
        payload = json.loads(qc.serialize_qc_report(report))
        assert payload["fixtures"][0]["global_psnr_db"] is None
        assert payload["fixtures"][0]["worst_frame"]["psnr_db"] == pytest.approx(29.4)


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


class TestCli:
    def test_defaults(self, qc: ModuleType) -> None:
        args = qc.parse_qc_args([])
        assert args.format == "rgbh"  # candidate is FORCED here, not via hardware.py
        assert args.seconds == 6
        assert args.warmup == 1
        assert args.repeats == 3

    def test_candidate_format_is_explicit(self, qc: ModuleType) -> None:
        assert qc.parse_qc_args(["--format", "rgbs"]).format == "rgbs"

    def test_invalid_format_rejected(self, qc: ModuleType) -> None:
        with pytest.raises(SystemExit):
            qc.parse_qc_args(["--format", "yuv"])

    def test_seconds_band_enforced(self, qc: ModuleType) -> None:
        with pytest.raises(SystemExit):
            qc.parse_qc_args(["--seconds", "30"])
        with pytest.raises(SystemExit):
            qc.parse_qc_args(["--seconds", "2"])


# ---------------------------------------------------------------------------
# VS seams (CPU-only VapourSynth; skipped where VS is absent)
# ---------------------------------------------------------------------------


class TestVapourSynthSeams:
    def test_fixture_clip_matches_numpy_content(self, qc: ModuleType) -> None:
        vapoursynth = pytest.importorskip("vapoursynth")
        del vapoursynth
        plan = qc.fixture_plan("pan_motion", frames_src=4)
        clip = qc.build_fixture_clip(plan, width=64, height=32)
        assert clip.format.name == "RGBS"
        assert len(clip) == 4
        assert clip.fps.numerator == 24000 and clip.fps.denominator == 1001
        expected = qc.frame_content(plan, 64, 32, 2)
        got = qc.clip_frame_planes(clip, 2)
        assert got.dtype == np.float32
        assert np.allclose(got, expected, atol=1e-6)

    def test_candidate_format_is_rgbh(self, qc: ModuleType) -> None:
        pytest.importorskip("vapoursynth")
        plan = qc.fixture_plan("pan_motion", frames_src=2)
        clip = qc.build_fixture_clip(plan, width=32, height=16)
        cand = qc.to_candidate_format(clip)
        assert cand.format.name == "RGBH"
        assert len(cand) == len(clip)

    def test_rife_seam_injected_fake(self, qc: ModuleType) -> None:
        """The RIFE call is a seam: a fake proves the kwargs/format wiring
        without GPU weights (real run = owner acceptance)."""
        pytest.importorskip("vapoursynth")
        plan = qc.fixture_plan("pan_motion", frames_src=2)
        clip = qc.build_fixture_clip(plan, width=32, height=16)

        calls: list[tuple[object, dict]] = []

        def fake_rife(node, **kwargs):
            calls.append((node, kwargs))
            return node  # identity "interpolation"

        cand = qc.apply_rife(clip, kwargs={"model": "4.25", "trt": False}, rife_fn=fake_rife)
        assert len(calls) == 1
        node, kwargs = calls[0]
        assert node.format.id == clip.format.id
        assert kwargs["model"] == "4.25"
        assert cand is not None

    def test_pair_extraction_metrics(self, qc: ModuleType) -> None:
        pytest.importorskip("vapoursynth")
        plan = qc.fixture_plan("pan_motion", frames_src=3)
        ref = qc.build_fixture_clip(plan, width=32, height=16)
        # candidate = ref + a known constant offset on every plane
        cand = qc.apply_constant_offset(ref, offset=0.25)
        agg = qc.extract_pair_metrics(ref, cand)
        assert agg.global_psnr_db == pytest.approx(10 * math.log10(1.0 / 0.0625), rel=1e-4)
        assert agg.mean_mad == pytest.approx(0.25, abs=1e-6)
        assert agg.worst.index in (0, 1, 2)

    def test_pair_extraction_counts_mismatch(self, qc: ModuleType) -> None:
        pytest.importorskip("vapoursynth")
        plan_a = qc.fixture_plan("pan_motion", frames_src=3)
        plan_b = qc.fixture_plan("pan_motion", frames_src=2)
        ref = qc.build_fixture_clip(plan_a, width=32, height=16)
        cand = qc.build_fixture_clip(plan_b, width=32, height=16)
        with pytest.raises(RuntimeError, match="кадр"):
            qc.extract_pair_metrics(ref, cand)
