"""HardwareProfile tests: frozen-ness, override precedence, VRAM budget, zero-GPU gate."""

import json
import os
from pathlib import Path

import pytest

from anime_upscale.config import HardwareOverrides
from anime_upscale.hardware import (
    GPUInfo,
    HardwareProfile,
    _cgroup_v2_cpu_quota,
    detect_hardware,
    resolve_prefetch_from_anchor,
    streams_for_vram,
    usable_cpu_count,
)
from anime_upscale.reporting import Reporter, resolve_mode


def _gpu(vram_mb: int = 16384, compute_capability: tuple[int, int] = (12, 0)) -> GPUInfo:
    return GPUInfo(
        index=0,
        name="NVIDIA GeForce RTX 5080",
        vram_mb=vram_mb,
        compute_capability=compute_capability,
    )


class _NoteCapture(Reporter):
    """Intercept reporter notes for loud-output assertions."""

    def __init__(self) -> None:
        super().__init__(resolve_mode(no_color=True, no_tty=True))
        self.notes: list[str] = []

    def note(self, text: str) -> None:
        self.notes.append(text)
        super().note(text)


class TestVramBudget:
    def test_16gb_yields_2_streams_per_plan(self) -> None:
        assert streams_for_vram(16384) == 2

    def test_24gb_yields_3_streams_per_plan(self) -> None:
        assert streams_for_vram(24564) == 3

    def test_t4_15gb_yields_1_stream(self) -> None:
        """Kaggle/Colab T4 (15360 MB): floor((12288 − 4274)/4096) = 1 — frozen fallback pin."""
        assert streams_for_vram(15360) == 1

    def test_4gb_yields_1_stream(self) -> None:
        assert streams_for_vram(4096) == 1


def _sweep_anchor(streams_peaks: list[tuple[int, float]], saturation: int | None) -> dict:
    """Minimal anchor-shaped dict for resolve_prefetch_from_anchor unit tests."""
    return {
        "streams_sweep": [
            {"streams": streams, "vram_peak_mb": peak} for streams, peak in streams_peaks
        ],
        "saturation_streams": saturation,
    }


class TestPrefetchAnchorResolution:
    """Measured prefetch depth: per-slot VRAM from sweep deltas, saturation clamp."""

    def test_slot_from_deltas_and_saturation_clamp(self) -> None:
        # Slope: (2228 − 2100) / (2 − 1) = 128 MB/slot; budget = 9083.2 → 70 slots → clamp 4.
        anchor = _sweep_anchor([(1, 2100.0), (2, 2228.0)], saturation=4)
        assert resolve_prefetch_from_anchor(anchor, 16384) == 4

    def test_no_saturation_caps_at_16(self) -> None:
        anchor = _sweep_anchor([(1, 2100.0), (2, 2228.0)], saturation=None)
        assert resolve_prefetch_from_anchor(anchor, 16384) == 16

    def test_saturation_above_cap_is_capped(self) -> None:
        # Slope over 1..32 = (6068 − 2100) / 31 = 128; saturation 32 still caps at 16.
        anchor = _sweep_anchor([(1, 2100.0), (32, 6068.0)], saturation=32)
        assert resolve_prefetch_from_anchor(anchor, 16384) == 16

    def test_single_point_uses_canonical_slot(self) -> None:
        # No deltas derivable → canonical 128 MB/slot; depth 4 ≠ formula's 2 at 16 GB.
        anchor = _sweep_anchor([(1, 2100.0)], saturation=4)
        assert resolve_prefetch_from_anchor(anchor, 16384) == 4

    def test_flat_deltas_fall_back_to_canonical_slot(self) -> None:
        anchor = _sweep_anchor([(1, 2100.0), (2, 2100.0)], saturation=None)
        assert resolve_prefetch_from_anchor(anchor, 16384) == 16

    def test_depth_floors_never_rounds(self) -> None:
        # VRAM 5632 → budget 481.6 MB → floor(481.6 / 128) = 3 (rounding would give 4).
        anchor = _sweep_anchor([(1, 2100.0), (8, 3008.0)], saturation=8)
        assert resolve_prefetch_from_anchor(anchor, 5632) == 3

    def test_depth_never_below_one(self) -> None:
        # GTX 1650 class: negative measured budget still clamps to 1.
        anchor = _sweep_anchor([(1, 2100.0)], saturation=4)
        assert resolve_prefetch_from_anchor(anchor, 4096) == 1

    def test_empty_sweep_uses_canonical_slot(self) -> None:
        anchor = _sweep_anchor([], saturation=4)
        assert resolve_prefetch_from_anchor(anchor, 16384) == 4


class TestDetect:
    def test_profile_is_frozen(self) -> None:
        profile = HardwareProfile(
            gpus=(_gpu(),),
            gpu_count=1,
            cpu_logical_cores=16,
            num_streams_per_gpu=2,
            encode_threads=16,
            precision="fp16",
        )
        import dataclasses

        with pytest.raises(dataclasses.FrozenInstanceError):
            profile.gpu_count = 2  # type: ignore[misc]

    def test_zero_gpu_with_nn_stage_is_loud_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("anime_upscale.hardware._probe_torch_gpus", list)
        monkeypatch.setattr("anime_upscale.hardware._nvidia_smi_cross_check", lambda _d: None)
        with pytest.raises(Exception, match="CUDA"):
            detect_hardware(
                overrides=HardwareOverrides(),
                cli_num_streams=None,
                nn_stages_selected=True,
            )

    def test_zero_gpu_encode_only_is_fine(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("anime_upscale.hardware._probe_torch_gpus", list)
        monkeypatch.setattr("anime_upscale.hardware._nvidia_smi_cross_check", lambda _d: None)
        profile = detect_hardware(
            overrides=HardwareOverrides(),
            cli_num_streams=None,
            nn_stages_selected=False,
        )
        assert profile.gpu_count == 0

    def test_override_precedence_config_then_cli(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("anime_upscale.hardware._probe_torch_gpus", lambda: [_gpu()])
        detected = detect_hardware(
            overrides=HardwareOverrides(),
            cli_num_streams=None,
            nn_stages_selected=True,
        )
        from_config = detect_hardware(
            overrides=HardwareOverrides(num_streams=3),
            cli_num_streams=None,
            nn_stages_selected=True,
        )
        from_cli = detect_hardware(
            overrides=HardwareOverrides(num_streams=3),
            cli_num_streams=4,
            nn_stages_selected=True,
        )
        assert from_config.num_streams_per_gpu == 3
        assert from_cli.num_streams_per_gpu == 4
        assert detected.num_streams_per_gpu == streams_for_vram(16384)

    def test_encode_threads_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("anime_upscale.hardware._probe_torch_gpus", lambda: [_gpu()])
        profile = detect_hardware(
            overrides=HardwareOverrides(encode_threads=4),
            cli_num_streams=None,
            nn_stages_selected=True,
            active_encoders=2,
        )
        assert profile.encode_threads == 4

    def test_gpus_subset_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        two = [GPUInfo(index=0, name="A", vram_mb=24564), GPUInfo(index=1, name="B", vram_mb=24564)]
        monkeypatch.setattr("anime_upscale.hardware._probe_torch_gpus", lambda: two)
        profile = detect_hardware(
            overrides=HardwareOverrides(gpus=(1,)),
            cli_num_streams=None,
            nn_stages_selected=True,
        )
        assert profile.gpu_count == 1
        assert profile.gpus[0].index == 1


class TestBlackwellDetection:
    def test_blackwell_by_compute_capability(self) -> None:
        gpu = GPUInfo(index=0, name="Custom GPU", vram_mb=16384, compute_capability=(12, 0))
        profile = HardwareProfile(
            gpus=(gpu,),
            gpu_count=1,
            cpu_logical_cores=8,
            num_streams_per_gpu=2,
            encode_threads=8,
            precision="fp16",
        )
        assert profile.is_blackwell is True
        assert profile.recommended_interp_format == "rgbs"
        assert profile.recommended_trt is False

    def test_blackwell_by_name_alone_is_not_enough(self) -> None:
        """W7: the name substring heuristic is gone — cc ≥ (12,0) is the only
        signal. A literal "RTX 5080" name with a stale/zero cc is NOT Blackwell."""
        gpu = GPUInfo(
            index=0,
            name="NVIDIA GeForce RTX 5080",
            vram_mb=16384,
            compute_capability=(0, 0),
        )
        profile = HardwareProfile(
            gpus=(gpu,),
            gpu_count=1,
            cpu_logical_cores=8,
            num_streams_per_gpu=2,
            encode_threads=8,
            precision="fp16",
        )
        assert profile.is_blackwell is False
        assert profile.recommended_interp_format == "rgbh"
        assert profile.recommended_trt is True

    def test_quadro_rtx_5000_turing_not_blackwell(self) -> None:
        """W7 false-positive class: "Quadro RTX 5000" contains the old "RTX 50"
        substring but is Turing (cc 7.5) — it must keep TRT enabled and RGBH."""
        gpu = GPUInfo(
            index=0,
            name="NVIDIA Quadro RTX 5000",
            vram_mb=16384,
            compute_capability=(7, 5),
        )
        profile = HardwareProfile(
            gpus=(gpu,),
            gpu_count=1,
            cpu_logical_cores=8,
            num_streams_per_gpu=2,
            encode_threads=8,
            precision="fp16",
        )
        assert profile.is_blackwell is False
        assert profile.recommended_interp_format == "rgbh"
        assert profile.recommended_trt is True

    def test_blackwell_real_5080_by_cc(self) -> None:
        """The real RTX 5080 reports cc (12, 0) via torch — detected Blackwell."""
        gpu = GPUInfo(
            index=0,
            name="NVIDIA GeForce RTX 5080",
            vram_mb=16384,
            compute_capability=(12, 0),
        )
        profile = HardwareProfile(
            gpus=(gpu,),
            gpu_count=1,
            cpu_logical_cores=8,
            num_streams_per_gpu=2,
            encode_threads=8,
            precision="fp16",
        )
        assert profile.is_blackwell is True
        assert profile.recommended_interp_format == "rgbs"
        assert profile.recommended_trt is False

    def test_older_gpu_defaults(self) -> None:
        gpu = GPUInfo(
            index=0,
            name="NVIDIA GeForce RTX 4090",
            vram_mb=24576,
            compute_capability=(8, 9),
        )
        profile = HardwareProfile(
            gpus=(gpu,),
            gpu_count=1,
            cpu_logical_cores=16,
            num_streams_per_gpu=3,
            encode_threads=16,
            precision="fp16",
        )
        assert profile.is_blackwell is False
        assert profile.recommended_interp_format == "rgbh"
        assert profile.recommended_trt is True

    def test_banner_blackwell_highlight(self) -> None:
        from anime_upscale.hardware import print_banner
        from anime_upscale.reporting import Reporter, resolve_mode

        lines: list[str] = []

        class InterceptReporter(Reporter):
            def note(self, text: str) -> None:
                lines.append(text)
                super().note(text)

        reporter = InterceptReporter(resolve_mode(no_color=True, no_tty=True))
        gpu = GPUInfo(
            index=0,
            name="NVIDIA GeForce RTX 5080",
            vram_mb=16384,
            compute_capability=(12, 0),
        )
        profile = HardwareProfile(
            gpus=(gpu,),
            gpu_count=1,
            cpu_logical_cores=8,
            num_streams_per_gpu=2,
            encode_threads=8,
            precision="fp16",
        )
        print_banner(reporter, profile, stages=("interpolate", "encode"))
        expected_msg = (
            "Детект GPU: NVIDIA GeForce RTX 5080 (Blackwell sm_120) → "
            "автоматический выбор: FP32 (RGBS) без TensorRT "
            "для предотвращения глитчей/сеток в RIFE."
        )
        assert any(expected_msg in line for line in lines)


class TestUsableCpuCountLadder:
    """CPU-basis ladder: process_cpu_count → affinity → cpu_count (§3 canon)."""

    def test_process_cpu_count_is_preferred(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(os, "process_cpu_count", lambda: 6, raising=False)
        monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(99)), raising=False)
        monkeypatch.setattr(os, "cpu_count", lambda: 99)
        monkeypatch.setattr("anime_upscale.hardware._CGROUP_V2_CPU_MAX", Path("/nonexistent"))
        assert usable_cpu_count() == 6

    def test_affinity_fallback_without_process_cpu_count(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Python 3.12 floor: no process_cpu_count → affinity mask length."""
        monkeypatch.delattr(os, "process_cpu_count", raising=False)
        monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(8)))
        monkeypatch.setattr(os, "cpu_count", lambda: 99)
        monkeypatch.setattr("anime_upscale.hardware._CGROUP_V2_CPU_MAX", Path("/nonexistent"))
        assert usable_cpu_count() == 8

    def test_cpu_count_fallback_without_affinity(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Windows pre-3.13 shape: neither affinity API exists → system count."""
        monkeypatch.delattr(os, "process_cpu_count", raising=False)
        monkeypatch.delattr(os, "sched_getaffinity", raising=False)
        monkeypatch.setattr(os, "cpu_count", lambda: 16)
        monkeypatch.setattr("anime_upscale.hardware._CGROUP_V2_CPU_MAX", Path("/nonexistent"))
        assert usable_cpu_count() == 16

    def test_cpu_count_none_means_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delattr(os, "process_cpu_count", raising=False)
        monkeypatch.delattr(os, "sched_getaffinity", raising=False)
        monkeypatch.setattr(os, "cpu_count", lambda: None)
        monkeypatch.setattr("anime_upscale.hardware._CGROUP_V2_CPU_MAX", Path("/nonexistent"))
        assert usable_cpu_count() == 1


class TestCgroupV2Quota:
    """cgroup v2 cpu.max reader: min-semantics, ceil, absence, loud malformed fallback."""

    def _quota_file(self, tmp_path: Path, content: str) -> Path:
        quota_file = tmp_path / "cpu.max"
        quota_file.write_text(content, encoding="utf-8")
        return quota_file

    def test_quota_clamps_basis_min_semantics(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Canon: basis 32 + quota 4 → 4 (docker --cpus=4 on a 32T host)."""
        monkeypatch.setattr(os, "process_cpu_count", lambda: 32, raising=False)
        monkeypatch.setattr(
            "anime_upscale.hardware._CGROUP_V2_CPU_MAX",
            self._quota_file(tmp_path, "400000 100000\n"),
        )
        assert usable_cpu_count() == 4

    def test_quota_is_ceiled_nproc_parity(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """2.5 CPUs ("250000 100000") → 3 (nproc/gnulib round-up parity, gh-149452)."""
        monkeypatch.setattr(os, "process_cpu_count", lambda: 32, raising=False)
        monkeypatch.setattr(
            "anime_upscale.hardware._CGROUP_V2_CPU_MAX",
            self._quota_file(tmp_path, "250000 100000\n"),
        )
        assert usable_cpu_count() == 3

    def test_max_means_unlimited(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr(os, "process_cpu_count", lambda: 8, raising=False)
        monkeypatch.setattr(
            "anime_upscale.hardware._CGROUP_V2_CPU_MAX",
            self._quota_file(tmp_path, "max 100000\n"),
        )
        assert usable_cpu_count() == 8

    def test_missing_file_is_silent_no_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Windows/WSL2/bare metal: no cgroup file → basis as is, no noise."""
        monkeypatch.setattr(os, "process_cpu_count", lambda: 8, raising=False)
        capture = _NoteCapture()
        monkeypatch.setattr("anime_upscale.hardware._CGROUP_V2_CPU_MAX", Path("/nonexistent"))
        assert usable_cpu_count(capture) == 8
        assert capture.notes == []

    def test_quota_zero_falls_back_loudly(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(os, "process_cpu_count", lambda: 8, raising=False)
        capture = _NoteCapture()
        monkeypatch.setattr(
            "anime_upscale.hardware._CGROUP_V2_CPU_MAX",
            self._quota_file(tmp_path, "0 100000\n"),
        )
        assert usable_cpu_count(capture) == 8
        assert len(capture.notes) == 1
        assert "cpu.max" in capture.notes[0]

    def test_broken_format_falls_back_loudly(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(os, "process_cpu_count", lambda: 8, raising=False)
        capture = _NoteCapture()
        monkeypatch.setattr(
            "anime_upscale.hardware._CGROUP_V2_CPU_MAX",
            self._quota_file(tmp_path, "garbage\n"),
        )
        assert usable_cpu_count(capture) == 8
        assert len(capture.notes) == 1

    def test_single_field_falls_back_loudly(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(os, "process_cpu_count", lambda: 8, raising=False)
        capture = _NoteCapture()
        monkeypatch.setattr(
            "anime_upscale.hardware._CGROUP_V2_CPU_MAX",
            self._quota_file(tmp_path, "500000\n"),
        )
        assert usable_cpu_count(capture) == 8
        assert len(capture.notes) == 1

    def test_reader_never_raises(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        quota_file = tmp_path / "cpu.max"
        quota_file.write_bytes(b"\xff\xfe\x00binary")
        monkeypatch.setattr("anime_upscale.hardware._CGROUP_V2_CPU_MAX", quota_file)
        assert _cgroup_v2_cpu_quota() is None

    def test_result_never_below_one(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Sub-1-CPU quota still clamps to a usable budget of 1."""
        monkeypatch.setattr(os, "process_cpu_count", lambda: 32, raising=False)
        monkeypatch.setattr(
            "anime_upscale.hardware._CGROUP_V2_CPU_MAX",
            self._quota_file(tmp_path, "50000 100000\n"),
        )
        assert usable_cpu_count() == 1

    def test_clamp_is_reported(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr(os, "process_cpu_count", lambda: 32, raising=False)
        capture = _NoteCapture()
        monkeypatch.setattr(
            "anime_upscale.hardware._CGROUP_V2_CPU_MAX",
            self._quota_file(tmp_path, "200000 100000\n"),
        )
        assert usable_cpu_count(capture) == 2
        assert any("квота" in note for note in capture.notes)


class TestEncodeBudgetDivision:
    """encode_threads = usable // active_encoders + the <4-per-lane warn."""

    def _detect(self, monkeypatch: pytest.MonkeyPatch, usable: int, **kwargs: object):
        monkeypatch.setattr("anime_upscale.hardware._probe_torch_gpus", lambda: [_gpu()])
        monkeypatch.setattr("anime_upscale.hardware._nvidia_smi_cross_check", lambda _d: None)
        monkeypatch.setattr("anime_upscale.hardware.usable_cpu_count", lambda reporter=None: usable)
        return detect_hardware(
            overrides=HardwareOverrides(),
            cli_num_streams=None,
            nn_stages_selected=True,
            **kwargs,  # type: ignore[arg-type]
        )

    def test_division_active_encoders_2(self, monkeypatch: pytest.MonkeyPatch) -> None:
        profile = self._detect(monkeypatch, usable=8, active_encoders=2)
        assert profile.encode_threads == 4

    def test_division_active_encoders_3(self, monkeypatch: pytest.MonkeyPatch) -> None:
        profile = self._detect(monkeypatch, usable=10, active_encoders=3)
        assert profile.encode_threads == 3

    def test_division_active_encoders_1_uses_full_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        profile = self._detect(monkeypatch, usable=8, active_encoders=1)
        assert profile.encode_threads == 8

    def test_cpu_logical_cores_stays_informational(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """System size (banner/diagnosis) vs usable budget: host 32, quota 4."""
        monkeypatch.setattr(os, "cpu_count", lambda: 32)
        profile = self._detect(monkeypatch, usable=4, active_encoders=1)
        assert profile.cpu_logical_cores == 32
        assert profile.encode_threads == 4

    def test_warn_below_4_per_encoder(self, monkeypatch: pytest.MonkeyPatch) -> None:
        capture = _NoteCapture()
        monkeypatch.setattr("anime_upscale.hardware._probe_torch_gpus", lambda: [_gpu()])
        monkeypatch.setattr("anime_upscale.hardware._nvidia_smi_cross_check", lambda _d: None)
        monkeypatch.setattr("anime_upscale.hardware.usable_cpu_count", lambda reporter=None: 6)
        detect_hardware(
            overrides=HardwareOverrides(),
            cli_num_streams=None,
            nn_stages_selected=True,
            active_encoders=2,
            reporter=capture,
        )
        warn = [n for n in capture.notes if "энкодер" in n]
        assert len(warn) == 1
        assert "4" in warn[0]

    def test_no_warn_at_4_or_more(self, monkeypatch: pytest.MonkeyPatch) -> None:
        capture = _NoteCapture()
        monkeypatch.setattr("anime_upscale.hardware._probe_torch_gpus", lambda: [_gpu()])
        monkeypatch.setattr("anime_upscale.hardware._nvidia_smi_cross_check", lambda _d: None)
        monkeypatch.setattr("anime_upscale.hardware.usable_cpu_count", lambda reporter=None: 8)
        detect_hardware(
            overrides=HardwareOverrides(),
            cli_num_streams=None,
            nn_stages_selected=True,
            active_encoders=2,
            reporter=capture,
        )
        assert not [n for n in capture.notes if "энкодер" in n]

    def test_no_warn_on_explicit_config_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Explicit [hardware] encode_threads is the owner's call — never warned."""
        capture = _NoteCapture()
        monkeypatch.setattr("anime_upscale.hardware._probe_torch_gpus", lambda: [_gpu()])
        monkeypatch.setattr("anime_upscale.hardware._nvidia_smi_cross_check", lambda _d: None)
        monkeypatch.setattr("anime_upscale.hardware.usable_cpu_count", lambda reporter=None: 16)
        detect_hardware(
            overrides=HardwareOverrides(encode_threads=2),
            cli_num_streams=None,
            nn_stages_selected=True,
            active_encoders=1,
            reporter=capture,
        )
        assert not [n for n in capture.notes if "энкодер" in n]

    def test_measured_anchor_divides_usable_budget(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Anchor tier: saturation_pools=32 on usable 8 with 2 encoders → 4 (measured)."""
        anchor = {
            "schema_version": 1,
            "date": "2026-11-06",
            "gpu_name": "NVIDIA GeForce RTX 5080",
            "gpu_vram_mb": 16384,
            "driver": "580.88",
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
                    "fps": 8.4,
                    "fps_repeats": [8.2, 8.4, 8.6],
                    "vram_peak_mb": 2100.0,
                    "torch_reserved_mb": 1450.0,
                    "smi_delta_mb": 2100.0,
                }
                for s in (1, 2)
            ],
            "pools_sweep": [
                {
                    "pools": p,
                    "fps": 22.0,
                    "fps_repeats": [21.8, 22.0, 22.2],
                    "wall_s": 60.0,
                    "rss_peak_mb": 1000.0,
                }
                for p in (4, 8)
            ],
            "saturation_streams": 2,
            "saturation_pools": 8,
            "notes": "",
        }
        path = tmp_path / "benchmarks" / "GeForceRTX5080" / "anchor.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(anchor), encoding="utf-8")
        monkeypatch.setattr("anime_upscale.hardware._probe_torch_gpus", lambda: [_gpu()])
        monkeypatch.setattr("anime_upscale.hardware._nvidia_smi_cross_check", lambda _d: None)
        monkeypatch.setattr("anime_upscale.hardware.query_driver_version", lambda: "580.88")
        monkeypatch.setattr("anime_upscale.hardware.usable_cpu_count", lambda reporter=None: 32)
        profile = detect_hardware(
            overrides=HardwareOverrides(),
            cli_num_streams=None,
            nn_stages_selected=True,
            active_encoders=2,
            project_root=tmp_path,
            encode_preset="medium",
        )
        # formula branch would be 32//2 = 16; the measured anchor saturates at 8.
        assert profile.encode_threads_source == "measured"
        assert profile.encode_threads == min(8, 32 // 2)


def _full_valid_anchor(
    streams_peaks: list[tuple[int, float]], saturation_streams: int | None
) -> dict:
    """Schema-valid anchor for detect_hardware tests; streams points carry peaks."""
    return {
        "schema_version": 1,
        "date": "2026-11-06",
        "gpu_name": "NVIDIA GeForce RTX 5080",
        "gpu_vram_mb": 16384,
        "driver": "580.88",
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
                "streams": streams,
                "fps": 8.4,
                "fps_repeats": [8.2, 8.4, 8.6],
                "vram_peak_mb": peak,
                "torch_reserved_mb": 1450.0,
                "smi_delta_mb": peak,
            }
            for streams, peak in streams_peaks
        ],
        "pools_sweep": [
            {
                "pools": p,
                "fps": 22.0,
                "fps_repeats": [21.8, 22.0, 22.2],
                "wall_s": 60.0,
                "rss_peak_mb": 1000.0,
            }
            for p in (4, 8)
        ],
        "saturation_streams": saturation_streams,
        "saturation_pools": 8,
        "notes": "",
    }


class TestMeasuredPrefetchBranch:
    """Anchor-tier streams: depth from sweep deltas; CLI outranks the anchor."""

    def _detect_with_anchor(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        cli_num_streams: int | None,
    ):
        streams_peaks = [(1, 2100.0), (4, 2484.0)]  # slope = 128 MB/slot
        anchor = _full_valid_anchor(streams_peaks, saturation_streams=4)
        path = tmp_path / "benchmarks" / "GeForceRTX5080" / "anchor.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(anchor), encoding="utf-8")
        monkeypatch.setattr("anime_upscale.hardware._probe_torch_gpus", lambda: [_gpu()])
        monkeypatch.setattr("anime_upscale.hardware._nvidia_smi_cross_check", lambda _d: None)
        monkeypatch.setattr("anime_upscale.hardware.query_driver_version", lambda: "580.88")
        monkeypatch.setattr("anime_upscale.hardware.usable_cpu_count", lambda reporter=None: 32)
        return detect_hardware(
            overrides=HardwareOverrides(),
            cli_num_streams=cli_num_streams,
            nn_stages_selected=True,
            active_encoders=2,
            project_root=tmp_path,
            encode_preset="medium",
        )

    def test_measured_anchor_sets_prefetch_depth(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        profile = self._detect_with_anchor(monkeypatch, tmp_path, cli_num_streams=None)
        # Formula would give 2; the measured sweep (128 MB/slot, saturation 4) gives 4.
        assert profile.num_streams_per_gpu == 4
        assert profile.streams_source == "measured"

    def test_cli_override_beats_measured_anchor(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        profile = self._detect_with_anchor(monkeypatch, tmp_path, cli_num_streams=2)
        assert profile.num_streams_per_gpu == 2
        assert profile.streams_source == "config"
