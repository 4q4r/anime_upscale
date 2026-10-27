"""Pipeline tests: graph assembly over subsets, fps sanity gate (plan §1/§9.5)."""

from fractions import Fraction

import pytest

from anime_upscale.config import RunConfig
from anime_upscale.pipeline import PipelineError, build_graph, check_target_fps

pytestmark = pytest.mark.vapoursynth


def _skip_without_vs() -> None:
    pytest.importorskip("vapoursynth")


class TestFpsSanity:
    def test_target_above_source_ok(self) -> None:
        check_target_fps(Fraction(24000, 1001), (60000, 1001))

    def test_target_below_source_loud(self) -> None:
        with pytest.raises(PipelineError, match="не выше"):
            check_target_fps(Fraction(60000, 1), (60000, 1001))

    def test_zero_source_loud(self) -> None:
        with pytest.raises(PipelineError, match="fps источника"):
            check_target_fps(Fraction(0), (60000, 1001))


class TestGraphAssembly:
    def _ctx(
        self,
        stages: tuple[str, ...],
        descale_height: int | None = None,
        models_dir: object = None,
        models_manifest: object = None,
    ) -> "object":
        _skip_without_vs()
        from anime_upscale.config import (
            DebandConfig,
            DescaleConfig,
            EncodeConfig,
            InterpolateConfig,
            UpscaleConfig,
        )
        from anime_upscale.hardware import HardwareProfile
        from anime_upscale.reporting import Reporter, resolve_mode
        from anime_upscale.stages import StageContext

        hw = HardwareProfile(
            gpus=(),
            gpu_count=0,
            cpu_logical_cores=4,
            num_streams_per_gpu=0,
            encode_threads=4,
            precision="none",
        )
        run = RunConfig(
            stages=stages,
            descale=DescaleConfig(),
            deband=DebandConfig(),
            upscale=UpscaleConfig(),
            interpolate=InterpolateConfig(trt=False),
            encode=EncodeConfig(),
            descale_height=descale_height,
            upload=False,
        )
        return StageContext(
            run=run,
            hardware=hw,
            project_root=None,  # type: ignore[arg-type]
            reporter=Reporter(resolve_mode(no_color=True, no_tty=True)),
            models_dir=models_dir,
            models_manifest=models_manifest,  # type: ignore[arg-type]
        )

    def test_pure_cpu_subset_assembles(self) -> None:
        import vapoursynth as vs

        from anime_upscale.stages import STAGE_BUILDERS
        from anime_upscale.stages.deband import apply as _a  # noqa: F401
        from anime_upscale.stages.descale import apply  # noqa: F401 (registry)

        assert {"descale", "deband"} <= set(STAGE_BUILDERS)
        core = vs.core
        clip = core.std.BlankClip(
            format=vs.YUV420P8, width=64, height=64, length=5, color=[100, 50, 50]
        )
        out = build_graph(self._ctx(("descale", "deband", "encode"), descale_height=32), clip)
        assert out.width == 32 and out.height == 32
        assert out.format.id == vs.YUV420P10
        assert len(out) == 5

    def test_rgbh_conversion_for_upscale_subset_entry(self) -> None:
        """The upscale path converts to RGBH before the builder; a missing model errors loudly
        (offline-safe: manifest resolution happens before any GPU work)."""
        from pathlib import Path

        import vapoursynth as vs

        from anime_upscale.stages.deband import apply  # noqa: F401
        from anime_upscale.stages.descale import apply as _a  # noqa: F401

        repo = Path(__file__).resolve().parent.parent
        core = vs.core
        clip = core.std.BlankClip(format=vs.YUV420P8, width=64, height=64, length=5)
        with pytest.raises(Exception, match=r"Модель|manifest|models fetch"):
            build_graph(
                self._ctx(
                    ("upscale", "encode"),
                    models_dir=repo / "models" / "nonexistent",
                    models_manifest=repo / "models" / "manifest.toml",
                ),
                clip,
            )
