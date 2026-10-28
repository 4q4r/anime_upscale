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


class TestUpscaleFramePull:
    def test_modifyframe_selector_contract_with_real_model(self) -> None:
        """Regression (parity gate): VS R81 calls selectors with KEYWORD args n=/f=,
        VideoFrame exposes plane views via frame[plane] (no get_read_array), and
        spandrel expects a batched [B,C,H,W] tensor. Pulls 3 REAL frames through
        the fetched model — runs only with CUDA + a fetched+verified model."""
        _skip_without_vs()
        pytest.importorskip("torch")
        pytest.importorskip("spandrel")
        import torch

        if not torch.cuda.is_available():
            pytest.skip("no CUDA device")
        from pathlib import Path

        from anime_upscale.stages.upscale import apply as upscale_apply

        repo = Path(__file__).resolve().parent.parent
        ctx = _ctx(_run_config(("upscale", "encode")))
        manifest = repo / "models" / "manifest.toml"
        if not manifest.is_file():
            pytest.skip("no manifest")
        rgbh = _blank(vs.RGBH, 64, 64)
        try:
            out = upscale_apply(ctx, rgbh)
            frame = out.get_frame(2)  # force the selector to execute
            assert frame.width == 128 and frame.height == 128
        except Exception as exc:  # model not fetched → loud manifest error is fine offline
            if "models fetch" in str(exc) or "не найдена" in str(exc):
                pytest.skip(f"model not fetched: {exc}")
            raise


class _NotesReporter:
    """Duck-typed reporter collecting note() lines (guard decision assertions)."""

    def __init__(self) -> None:
        self.notes: list[str] = []

    def note(self, text: str) -> None:
        self.notes.append(text)


class TestEvenAlignApply:
    """VS application of the mod-2 guard (pipeline calls it before the final convert)."""

    def _apply(self, clip: "vs.VideoNode") -> tuple["vs.VideoNode", _NotesReporter]:
        from anime_upscale.stages.dimensions import apply_even_align

        reporter = _NotesReporter()
        return apply_even_align(clip, reporter), reporter  # ty: ignore[arg-type]

    def test_rgb_odd_clip_crops_one_pixel(self) -> None:
        core = vs.core
        clip = core.std.BlankClip(format=vs.RGBH, width=321, height=241, length=2)
        out, reporter = self._apply(clip)
        assert out.width == 320 and out.height == 241 - 1
        assert reporter.notes and "320x240" in reporter.notes[0]
        assert "обрезка" in reporter.notes[0]

    def test_subsampled_format_routes_crop_plan_to_scale(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # An odd-dimension YUV420 clip cannot even be constructed in VS (formats
        # enforce evenness), so the routing for "crop plan over a subsampled
        # format" is exercised with a synthetic plan on a real even clip: the
        # routing must resize (1px crop is not expressible on 420), never crop.
        import anime_upscale.stages.dimensions as dims

        monkeypatch.setattr(
            dims,
            "decide_even_align",
            lambda w, h: dims.EvenAlign(
                action="crop", width=w - 2, height=h - 2, crop_right=1, crop_bottom=1
            ),
        )
        core = vs.core
        clip = core.std.BlankClip(format=vs.YUV420P10, width=320, height=240, length=2)
        out, reporter = self._apply(clip)
        assert out.width == 318 and out.height == 238
        assert reporter.notes and "масштабирование" in reporter.notes[0]

    def test_even_clip_untouched_no_note(self) -> None:
        core = vs.core
        clip = core.std.BlankClip(format=vs.RGBH, width=320, height=240, length=2)
        out, reporter = self._apply(clip)
        assert out is clip
        assert reporter.notes == []
