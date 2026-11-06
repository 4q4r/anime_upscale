"""HardwareProfile tests: frozen-ness, override precedence, VRAM budget, zero-GPU gate."""

import pytest

from anime_upscale.config import HardwareOverrides
from anime_upscale.hardware import (
    GPUInfo,
    HardwareProfile,
    detect_hardware,
    streams_for_vram,
)


def _gpu(vram_mb: int = 16384, compute_capability: tuple[int, int] = (12, 0)) -> GPUInfo:
    return GPUInfo(
        index=0,
        name="NVIDIA GeForce RTX 5080",
        vram_mb=vram_mb,
        compute_capability=compute_capability,
    )


class TestVramBudget:
    def test_16gb_yields_2_streams_per_plan(self) -> None:
        assert streams_for_vram(16384) == 2

    def test_24gb_yields_3_streams_per_plan(self) -> None:
        assert streams_for_vram(24564) == 3

    def test_4gb_yields_1_stream(self) -> None:
        assert streams_for_vram(4096) == 1


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
