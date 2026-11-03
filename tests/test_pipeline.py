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
        is_blackwell: bool = False,
    ) -> "object":
        _skip_without_vs()
        from anime_upscale.config import (
            DebandConfig,
            DescaleConfig,
            EncodeConfig,
            InterpolateConfig,
            UpscaleConfig,
        )
        from anime_upscale.hardware import GPUInfo, HardwareProfile
        from anime_upscale.reporting import Reporter, resolve_mode
        from anime_upscale.stages import StageContext

        gpus = (
            (GPUInfo(index=0, name="RTX 5080", vram_mb=16384, compute_capability=(12, 0)),)
            if is_blackwell
            else ()
        )
        hw = HardwareProfile(
            gpus=gpus,
            gpu_count=len(gpus),
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
            models_dir=models_dir,  # type: ignore[arg-type]
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

    def test_interp_format_selection_rgbs_for_blackwell(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When interpolate stage is entered without upscale on Blackwell, RGBS is selected."""
        import vapoursynth as vs

        from anime_upscale.pipeline import _to_rgb
        from anime_upscale.stages import STAGE_BUILDERS

        converted_formats: list[int] = []

        def spy_to_rgb(
            clip: vs.VideoNode,
            target_format: int = vs.RGBH,
            matrix: int = vs.MATRIX_BT709,
        ) -> vs.VideoNode:
            converted_formats.append(target_format)
            return _to_rgb(clip, target_format=target_format, matrix=matrix)

        monkeypatch.setattr("anime_upscale.pipeline._to_rgb", spy_to_rgb)

        # Mock the interpolate builder so we don't need real RIFE/GPU
        monkeypatch.setitem(
            STAGE_BUILDERS,
            "interpolate",
            lambda ctx, clip: clip,
        )

        core = vs.core
        clip = core.std.BlankClip(format=vs.YUV420P8, width=64, height=64, length=5)

        # Blackwell context
        ctx_bw = self._ctx(("interpolate", "encode"), is_blackwell=True)
        out_bw = build_graph(ctx_bw, clip)
        assert converted_formats[-1] == vs.RGBS
        assert out_bw.format.id == vs.YUV420P10

        # Non-Blackwell context
        ctx_non_bw = self._ctx(("interpolate", "encode"), is_blackwell=False)
        out_non_bw = build_graph(ctx_non_bw, clip)
        assert converted_formats[-1] == vs.RGBH
        assert out_non_bw.format.id == vs.YUV420P10


class _RecordingReporter:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warn(self, text: str) -> None:
        self.warnings.append(text)


def _teardown_ctx(reporter: object | None = None) -> "object":
    """Minimal StageContext for run_pipeline teardown tests (encode-only run)."""
    from anime_upscale.config import EncodeConfig, RunConfig
    from anime_upscale.hardware import HardwareProfile
    from anime_upscale.reporting import Reporter, resolve_mode
    from anime_upscale.stages import StageContext

    return StageContext(
        run=RunConfig(
            stages=("encode",),
            descale=object(),  # type: ignore[arg-type]
            deband=object(),  # type: ignore[arg-type]
            upscale=object(),  # type: ignore[arg-type]
            interpolate=object(),  # type: ignore[arg-type]
            encode=EncodeConfig(),
            descale_height=None,
            upload=False,
        ),
        hardware=HardwareProfile(
            gpus=(),
            gpu_count=0,
            cpu_logical_cores=4,
            num_streams_per_gpu=0,
            encode_threads=2,
            precision="none",
        ),
        project_root=None,  # type: ignore[arg-type]
        reporter=reporter or Reporter(resolve_mode(no_color=True, no_tty=True)),
    )


class TestEncoderTeardown:
    """M2: no exception path may orphan the ffmpeg child (wait() + drain join)."""

    @staticmethod
    def _wire(monkeypatch: pytest.MonkeyPatch, output_exc: Exception) -> list:
        import anime_upscale.pipeline as pl

        waits: list[bool] = []

        class _FakeStdin:
            closed = True

        class _FakeProcess:
            stdin = _FakeStdin()
            returncode = 0
            tail_shared = True

            def wait(self) -> int:
                waits.append(True)
                return self.returncode

        class _FakeEncoder:
            process = _FakeProcess()
            tail: list[bytes] = []
            stdin = _FakeStdin()

        class _FakeGraph:
            fps = Fraction(60000, 1001)
            width = 640
            height = 360

            def output(self, stream: object, **_kw: object) -> None:
                raise output_exc

        class _FakeClip:
            fps = Fraction(24000, 1001)

        class _FakeBs:
            @staticmethod
            def VideoSource(path: str) -> _FakeClip:
                return _FakeClip()

        monkeypatch.setattr(pl.vs, "core", type("_FakeCore", (), {"bs": _FakeBs})())
        monkeypatch.setattr(pl, "build_graph", lambda ctx, clip, **_kw: _FakeGraph())
        monkeypatch.setattr(pl, "spawn_encoder", lambda args: _FakeEncoder())
        return waits

    def test_non_oserror_reaps_encoder_and_reraises_original(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """M2: a VS-internal RuntimeError must not leave ffmpeg running, and the
        ORIGINAL exception — not a teardown error — is the loud failure."""
        from anime_upscale.pipeline import run_pipeline

        boom = RuntimeError("vs internal boom")
        waits = self._wire(monkeypatch, boom)
        with pytest.raises(RuntimeError, match="vs internal boom"):
            run_pipeline(
                _teardown_ctx(),  # type: ignore[arg-type]
                input_path="fake.mkv",  # type: ignore[arg-type]
                output_path="out.mkv",  # type: ignore[arg-type]
                ffmpeg_path="/usr/bin/ffmpeg",
            )
        assert waits == [True]  # the child was reaped on the failure path

    def test_dead_encoder_during_vs_failure_reports_tail_but_keeps_original(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """M2: if ffmpeg ALSO died, its tail is reported as a warning while the
        original VS exception still wins (no masking)."""
        import anime_upscale.pipeline as pl
        from anime_upscale.pipeline import run_pipeline

        reporter = _RecordingReporter()
        waits = self._wire(monkeypatch, RuntimeError("vs internal boom"))

        # Make finish_encoder's loud branch fire on the teardown path.
        def _finish_fail(encoder: object) -> None:
            encoder.process.wait()  # type: ignore[attr-defined]
            raise RuntimeError("ffmpeg завершился с ошибкой (код 1). stderr-tail")

        monkeypatch.setattr(pl, "finish_encoder", _finish_fail)
        with pytest.raises(RuntimeError, match="vs internal boom"):
            run_pipeline(
                _teardown_ctx(reporter),  # type: ignore[arg-type]
                input_path="fake.mkv",  # type: ignore[arg-type]
                output_path="out.mkv",  # type: ignore[arg-type]
                ffmpeg_path="/usr/bin/ffmpeg",
            )
        assert waits == [True]
        assert any("ffmpeg" in w for w in reporter.warnings)

    def test_keyboard_interrupt_reaps_encoder(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """M2: even a BaseException interruption reaps the child."""
        from anime_upscale.pipeline import run_pipeline

        waits = self._wire(monkeypatch, KeyboardInterrupt())
        with pytest.raises(KeyboardInterrupt):
            run_pipeline(
                _teardown_ctx(),  # type: ignore[arg-type]
                input_path="fake.mkv",  # type: ignore[arg-type]
                output_path="out.mkv",  # type: ignore[arg-type]
                ffmpeg_path="/usr/bin/ffmpeg",
            )
        assert waits == [True]


class TestInterpolateStageTRTFallback:
    def test_trt_disabled_on_blackwell_with_warning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys
        import types

        from anime_upscale.config import (
            DebandConfig,
            DescaleConfig,
            EncodeConfig,
            InterpolateConfig,
            RunConfig,
            UpscaleConfig,
        )
        from anime_upscale.hardware import GPUInfo, HardwareProfile
        from anime_upscale.reporting import Reporter, resolve_mode
        from anime_upscale.stages import StageContext
        from anime_upscale.stages.interpolate import apply as interp_apply

        captured_kwargs: dict[str, object] = {}

        fake_vsrife = types.ModuleType("vsrife")

        def fake_rife(clip: object, **kwargs: object) -> object:
            captured_kwargs.update(kwargs)
            return clip

        fake_vsrife.rife = fake_rife  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "vsrife", fake_vsrife)

        warnings: list[str] = []

        class InterceptReporter(Reporter):
            def warn(self, text: str) -> None:
                warnings.append(text)
                super().warn(text)

        gpu = GPUInfo(index=0, name="RTX 5080", vram_mb=16384, compute_capability=(12, 0))
        hw = HardwareProfile(
            gpus=(gpu,),
            gpu_count=1,
            cpu_logical_cores=8,
            num_streams_per_gpu=2,
            encode_threads=8,
            precision="fp16",
        )
        run = RunConfig(
            stages=("interpolate",),
            descale=DescaleConfig(),
            deband=DebandConfig(),
            upscale=UpscaleConfig(),
            interpolate=InterpolateConfig(trt=True),
            encode=EncodeConfig(),
            descale_height=None,
            upload=False,
        )
        reporter = InterceptReporter(resolve_mode(no_color=True, no_tty=True))
        ctx = StageContext(
            run=run,
            hardware=hw,
            project_root=None,  # type: ignore[arg-type]
            reporter=reporter,
            models_dir=None,  # type: ignore[arg-type]
            models_manifest=None,  # type: ignore[arg-type]
        )

        class FakeClip:
            width = 1920
            height = 1080

        interp_apply(ctx, FakeClip())  # type: ignore[arg-type]

        assert captured_kwargs["trt"] is False
        expected_warning = "TensorRT отключен, используется безопасный режим PyTorch CUDA"
        assert any(expected_warning in w for w in warnings)
