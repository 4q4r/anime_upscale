"""HardwareProfile tests: frozen-ness, override precedence, VRAM budget, zero-GPU gate."""

import pytest

from anime_upscale.config import HardwareOverrides
from anime_upscale.hardware import (
    GPUInfo,
    HardwareProfile,
    detect_hardware,
    streams_for_vram,
)


def _gpu(vram_mb: int = 16384) -> GPUInfo:
    return GPUInfo(index=0, name="NVIDIA GeForce RTX 5080", vram_mb=vram_mb)


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
