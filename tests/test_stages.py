"""Stage builder tests on 5-frame BlankClips (plan §2).

GPU-dependent builders are marked `gpu` and skipped without CUDA; pure-VS
builders are marked `vapoursynth` and skip gracefully without VapourSynth.
"""

import pytest
import vapoursynth as vs

from anime_upscale.config import RunConfig
from anime_upscale.hardware import GPUInfo, HardwareProfile
from anime_upscale.reporting import Reporter, resolve_mode
from anime_upscale.stages import StageContext

pytestmark = pytest.mark.vapoursynth


def _skip_without_vs() -> None:
    pytest.importorskip("vapoursynth")


def _blank(format_id: int = vs.YUV420P8, width: int = 64, height: int = 64) -> vs.VideoNode:
    core = vs.core
    return core.std.BlankClip(
        format=format_id, width=width, height=height, length=5, color=[100, 50, 50]
    )


def _ctx(run: RunConfig, with_gpu: bool = True) -> StageContext:
    gpus = (GPUInfo(index=0, name="TEST", vram_mb=4096),) if with_gpu else ()
    hw = HardwareProfile(
        gpus=gpus,
        gpu_count=len(gpus),
        cpu_logical_cores=4,
        num_streams_per_gpu=1 if gpus else 0,
        encode_threads=4,
        precision="fp16" if gpus else "none",
    )
    return StageContext(
        run=run,
        hardware=hw,
        project_root=None,
        reporter=Reporter(resolve_mode(no_color=True, no_tty=True)),
    )  # type: ignore[arg-type]


def _run_config(stages: tuple[str, ...], **over: object) -> RunConfig:
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
        descale_height=over.get("descale_height", 32),
        upload=False,
    )


class TestPureVsStages:
    def test_descale(self) -> None:
        _skip_without_vs()
        from anime_upscale.stages.descale import apply as descale_apply

        clip = _blank()
        out = descale_apply(_ctx(_run_config(("descale", "encode"))), clip)
        assert out.width == 32 and out.height == 32 and len(out) == 5

    def test_deband(self) -> None:
        _skip_without_vs()
        from anime_upscale.stages.deband import apply as deband_apply

        out = deband_apply(_ctx(_run_config(("deband", "encode"))), _blank())
        assert out.width == 64 and out.fps == _blank().fps


class TestGpuStages:
    @pytest.mark.gpu
    def test_interpolate_cpu_backed_graph(self) -> None:
        """vsrife requires RGB input; the pipeline feeds RGBH. trt=False keeps this
        torch-CPU; the model download is network-bound — run only when cached."""
        _skip_without_vs()
        pytest.importorskip("torch")
        pytest.importorskip("vsrife")
        from anime_upscale.stages.interpolate import apply as interp_apply

        rgbh = _blank(vs.RGBH, 64, 64)
        out = interp_apply(_ctx(_run_config(("interpolate", "encode"))), rgbh)
        assert out.fps.numerator == 60000 and out.fps.denominator == 1001

    def test_upscale_requires_model(self) -> None:
        """Missing pinned artifact = loud manifest error BEFORE any GPU work
        (offline-safe forever: the model dir is empty by construction)."""
        _skip_without_vs()
        from pathlib import Path

        from anime_upscale.stages.upscale import apply as upscale_apply

        repo = Path(__file__).resolve().parent.parent
        rgbh = _blank(vs.RGBH, 64, 64)
        ctx = _ctx(_run_config(("upscale", "encode")))
        ctx_empty = StageContext(
            run=ctx.run,
            hardware=ctx.hardware,
            project_root=ctx.project_root,
            reporter=ctx.reporter,
            models_dir=Path("/nonexistent-models"),
            models_manifest=repo / "models" / "manifest.toml",
        )
        with pytest.raises(Exception, match=r"models fetch|не найдена|Модель"):
            upscale_apply(ctx_empty, rgbh)

    def test_interpolate_requires_gpu(self) -> None:
        """The stage refuses a GPU-less context loudly, before any vsrife work."""
        _skip_without_vs()
        from anime_upscale.stages.interpolate import apply as interp_apply

        with pytest.raises(RuntimeError, match="CUDA"):
            interp_apply(_ctx(_run_config(("interpolate", "encode")), with_gpu=False), _blank())
