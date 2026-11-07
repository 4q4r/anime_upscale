"""GPU degradation protection tests (P2 wave): watchdog triggers, escalation,
per-episode VRAM hygiene, allocator memory cap.

Everything here is CPU-safe: the torch.cuda surface is faked via
``sys.modules`` (no CUDA device needed), the watchdog clock is injected, and
the reserved-VRAM sampler is monkeypatched wherever the trend is not the
behavior under test. The module imports ``anime_upscale.pipeline`` (which
imports vapoursynth), so it is collection-ignored on VS-less machines via
conftest.
"""

import gc
import sys
import types
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import anime_upscale.pipeline as pl
from anime_upscale.config import EncodeConfig, RunConfig
from anime_upscale.hardware import GPUInfo, HardwareProfile

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = pytest.mark.vapoursynth


class _RecordingReporter:
    def __init__(self) -> None:
        self.warnings: list[str] = []
        self.notes: list[str] = []

    def warn(self, text: str) -> None:
        self.warnings.append(text)

    def note(self, text: str) -> None:
        self.notes.append(text)


def _watchdog(reporter: _RecordingReporter, *, window_frames: int = 10) -> pl.DegradationWatchdog:
    wd = pl.DegradationWatchdog(reporter)  # type: ignore[arg-type]
    wd._window_frames = window_frames
    wd._fake_now = 0.0
    wd._clock_fn: Callable[[], float] = lambda: wd._fake_now  # type: ignore[assignment]
    return wd


def _feed_window(wd: "pl.DegradationWatchdog", frames: int, fps: float) -> None:
    """Anchor lazily, advance the fake clock, complete exactly one window at ``fps``."""
    wd._on_frame(0, 100)
    wd._fake_now += frames / fps  # type: ignore[attr-defined]
    wd._on_frame(frames, 100)


def _no_cuda_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pl, "_cuda_reserved_bytes", lambda: None)


class TestFpsDropDetection:
    def test_first_drop_warns_and_continues(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _no_cuda_telemetry(monkeypatch)
        reporter = _RecordingReporter()
        wd = _watchdog(reporter)
        wd.begin_episode()
        for _ in range(3):
            _feed_window(wd, 10, 100.0)
        _feed_window(wd, 10, 50.0)  # >40% below the 100 fps baseline
        assert len(reporter.warnings) == 1
        assert "кадр/с" in reporter.warnings[0]
        assert wd._mitigated is True

    def test_small_drop_does_not_trigger(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _no_cuda_telemetry(monkeypatch)
        reporter = _RecordingReporter()
        wd = _watchdog(reporter)
        wd.begin_episode()
        for _ in range(3):
            _feed_window(wd, 10, 100.0)
        _feed_window(wd, 10, 70.0)  # exactly -30%: inside the threshold
        assert reporter.warnings == []
        assert wd._mitigated is False

    def test_baseline_windows_never_trigger(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Only the first 3 windows exist: all of them ARE the baseline."""
        _no_cuda_telemetry(monkeypatch)
        reporter = _RecordingReporter()
        wd = _watchdog(reporter)
        wd.begin_episode()
        _feed_window(wd, 10, 100.0)
        _feed_window(wd, 10, 100.0)
        _feed_window(wd, 10, 10.0)  # slow, but still building the baseline
        assert reporter.warnings == []

    def test_repeated_drop_raises_loud_restart_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _no_cuda_telemetry(monkeypatch)
        reporter = _RecordingReporter()
        wd = _watchdog(reporter)
        wd.begin_episode()
        for _ in range(3):
            _feed_window(wd, 10, 100.0)
        _feed_window(wd, 10, 50.0)  # first event: warn + mitigate
        wd.begin_episode()
        wd.finish_episode(allow_escalation=True)  # boundary hygiene after the warn
        for _ in range(3):
            _feed_window(wd, 10, 100.0)
        with pytest.raises(RuntimeError, match=r"(?i)перезапуст"):
            _feed_window(wd, 10, 50.0)  # degradation persists after cleanup

    def test_restart_error_names_saved_progress(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _no_cuda_telemetry(monkeypatch)
        reporter = _RecordingReporter()
        wd = _watchdog(reporter)
        wd.begin_episode()
        for _ in range(3):
            _feed_window(wd, 10, 100.0)
        _feed_window(wd, 10, 50.0)
        wd.begin_episode()
        wd.finish_episode(allow_escalation=True)
        for _ in range(3):
            _feed_window(wd, 10, 100.0)
        with pytest.raises(RuntimeError, match="сохранён"):
            _feed_window(wd, 10, 50.0)

    def test_healthy_episode_resets_mitigation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An isolated transient recovers; a later fresh drop warns again, not fails."""
        _no_cuda_telemetry(monkeypatch)
        reporter = _RecordingReporter()
        wd = _watchdog(reporter)
        wd.begin_episode()
        for _ in range(3):
            _feed_window(wd, 10, 100.0)
        _feed_window(wd, 10, 50.0)  # event 1
        wd.begin_episode()
        wd.finish_episode(allow_escalation=True)
        # the boundary of the degraded episode itself is NOT a recovery
        assert wd._mitigated is True
        wd.begin_episode()
        wd.finish_episode(allow_escalation=True)  # a fully healthy episode → recovery
        assert wd._mitigated is False
        wd.begin_episode()
        for _ in range(3):
            _feed_window(wd, 10, 100.0)
        _feed_window(wd, 10, 50.0)  # a NEW first event: warn again, no raise

    def test_warn_dedup_per_episode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _no_cuda_telemetry(monkeypatch)
        reporter = _RecordingReporter()
        wd = _watchdog(reporter)
        wd.begin_episode()
        for _ in range(3):
            _feed_window(wd, 10, 100.0)
        _feed_window(wd, 10, 50.0)
        _feed_window(wd, 10, 40.0)  # still degraded in the SAME episode
        assert len(reporter.warnings) == 1


class TestMemoryTrendDetection:
    def _finish_with_reserved(
        self,
        wd: "pl.DegradationWatchdog",
        monkeypatch: pytest.MonkeyPatch,
        reserved: int,
    ) -> None:
        monkeypatch.setattr(pl, "_cuda_reserved_bytes", lambda: reserved)
        wd.finish_episode(allow_escalation=True)

    def test_monotonic_growth_triggers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        reporter = _RecordingReporter()
        wd = _watchdog(reporter)
        wd.begin_episode()
        self._finish_with_reserved(wd, monkeypatch, 1000)
        wd.begin_episode()
        self._finish_with_reserved(wd, monkeypatch, 1100)
        wd.begin_episode()
        self._finish_with_reserved(wd, monkeypatch, 1250)  # +25% over 3 episodes
        assert any("VRAM" in w for w in reporter.warnings)
        assert wd._mitigated is True

    def test_small_growth_does_not_trigger(self, monkeypatch: pytest.MonkeyPatch) -> None:
        reporter = _RecordingReporter()
        wd = _watchdog(reporter)
        wd.begin_episode()
        self._finish_with_reserved(wd, monkeypatch, 1000)
        wd.begin_episode()
        self._finish_with_reserved(wd, monkeypatch, 1010)
        wd.begin_episode()
        self._finish_with_reserved(wd, monkeypatch, 1020)  # +2% total: noise
        assert reporter.warnings == []

    def test_plateau_breaks_the_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A flat pair means the growth stalled — the strictly-increasing suffix wins."""
        reporter = _RecordingReporter()
        wd = _watchdog(reporter)
        wd.begin_episode()
        self._finish_with_reserved(wd, monkeypatch, 1000)
        wd.begin_episode()
        self._finish_with_reserved(wd, monkeypatch, 1500)
        wd.begin_episode()
        self._finish_with_reserved(wd, monkeypatch, 1500)  # plateau
        wd.begin_episode()
        self._finish_with_reserved(wd, monkeypatch, 1501)
        assert reporter.warnings == []

    def test_fewer_than_three_samples_never_triggers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        reporter = _RecordingReporter()
        wd = _watchdog(reporter)
        wd.begin_episode()
        self._finish_with_reserved(wd, monkeypatch, 1000)
        wd.begin_episode()
        self._finish_with_reserved(wd, monkeypatch, 2000)  # +100%, but 2 samples
        assert reporter.warnings == []

    def test_repeated_trend_event_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        reporter = _RecordingReporter()
        wd = _watchdog(reporter)
        wd.begin_episode()
        for value in (1000, 1100, 1250):  # first event fires at 1250: warn + mitigate
            self._finish_with_reserved(wd, monkeypatch, value)
            wd.begin_episode()
        assert len(reporter.warnings) == 1
        assert wd._mitigated is True
        with pytest.raises(RuntimeError, match=r"(?i)перезапуст"):
            # chronic growth continues straight through the mitigation boundary
            self._finish_with_reserved(wd, monkeypatch, 1300)

    def test_failure_path_never_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The episode already failed loudly — the watchdog must not mask it."""
        reporter = _RecordingReporter()
        wd = _watchdog(reporter)
        wd.begin_episode()
        for value in (1000, 1100, 1250):  # first event → warn + mitigate
            self._finish_with_reserved(wd, monkeypatch, value)
            wd.begin_episode()
        assert wd._mitigated is True
        monkeypatch.setattr(pl, "_cuda_reserved_bytes", lambda: 1350)
        wd.finish_episode(allow_escalation=False)  # repeated trend, but failure path
        assert any("сохраняется" in w for w in reporter.warnings)  # loud, informationally
        # and crucially: no exception escaped this call

    def test_no_cuda_telemetry_is_a_no_op(self, monkeypatch: pytest.MonkeyPatch) -> None:
        reporter = _RecordingReporter()
        wd = _watchdog(reporter)
        _no_cuda_telemetry(monkeypatch)
        for _ in range(5):
            wd.begin_episode()
            wd.finish_episode(allow_escalation=True)
        assert reporter.warnings == []


class TestEpisodeHygiene:
    def _fake_torch(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, list[object]]:
        calls: dict[str, list[object]] = {"empty_cache": [], "fraction": []}

        fake_cuda = types.SimpleNamespace(
            is_available=lambda: True,
            empty_cache=lambda: calls["empty_cache"].append(True),
            set_per_process_memory_fraction=lambda fraction, device: calls["fraction"].append(
                (fraction, device)
            ),
            memory_reserved=lambda: 42,
        )
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = fake_cuda  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        self.calls = calls  # type: ignore[attr-defined]
        return calls

    def test_hygiene_calls_gc_and_empty_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        reporter = _RecordingReporter()
        calls = self._fake_torch(monkeypatch)
        collected: list[int] = []
        monkeypatch.setattr(gc, "collect", lambda: collected.append(1))
        pl._episode_hygiene(reporter)  # type: ignore[arg-type]
        assert collected == [1]
        assert calls["empty_cache"] == [True]
        assert reporter.warnings == []

    def test_hygiene_survives_empty_cache_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        reporter = _RecordingReporter()
        fake_cuda = types.SimpleNamespace(
            is_available=lambda: True,
            empty_cache=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = fake_cuda  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setattr(gc, "collect", lambda: None)
        pl._episode_hygiene(reporter)  # type: ignore[arg-type]
        assert any("очистка" in w.lower() for w in reporter.warnings)

    def test_hygiene_without_cuda_skips_empty_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        reporter = _RecordingReporter()
        fake_cuda = types.SimpleNamespace(is_available=lambda: False)
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = fake_cuda  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setattr(gc, "collect", lambda: None)
        pl._episode_hygiene(reporter)  # type: ignore[arg-type]
        assert reporter.warnings == []


def _cpu_ctx(reporter: object, *, gpus: tuple[GPUInfo, ...] = ()) -> object:
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
            gpus=gpus,
            gpu_count=len(gpus),
            cpu_logical_cores=4,
            num_streams_per_gpu=0,
            encode_threads=2,
            precision="none",
        ),
        project_root=None,  # type: ignore[arg-type]
        reporter=reporter,  # type: ignore[arg-type]
    )


def _wire_pipeline(monkeypatch: pytest.MonkeyPatch, output_exc: Exception | None) -> list[object]:
    """Fake the VS/encoder surface of run_pipeline; return graph.output kwargs seen."""

    class _FakeStdin:
        closed = True

    class _FakeProcess:
        stdin = _FakeStdin()
        returncode = 0

        def wait(self) -> int:
            return self.returncode

    class _FakeEncoder:
        process = _FakeProcess()
        tail: list[bytes] = []
        stdin = _FakeStdin()
        drain_thread = None

    seen_kwargs: list[dict[str, object]] = []

    class _FakeGraph:
        fps = Fraction(60000, 1001)
        width = 640
        height = 360

        def output(self, stream: object, **kw: object) -> None:
            seen_kwargs.append(dict(kw))
            if output_exc is not None:
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
    return seen_kwargs  # type: ignore[return-value]


class TestRunPipelineProtectionIntegration:
    def test_success_runs_hygiene(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _wire_pipeline(monkeypatch, output_exc=None)
        reporter = _RecordingReporter()
        harness = TestEpisodeHygiene()
        harness._fake_torch(monkeypatch)
        collected: list[int] = []
        monkeypatch.setattr(gc, "collect", lambda: collected.append(1))
        _no_cuda_telemetry(monkeypatch)
        monkeypatch.setattr(pl, "_WATCHDOG", None)
        monkeypatch.setattr(pl, "_CAP_APPLIED", True)  # cap covered by its own tests
        pl.run_pipeline(
            _cpu_ctx(reporter),  # type: ignore[arg-type]
            input_path=Path("fake.mkv"),
            output_path=Path("out.mkv"),
            ffmpeg_path="/usr/bin/ffmpeg",
        )
        assert collected == [1]

    def test_failure_still_runs_hygiene_and_keeps_original_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _wire_pipeline(monkeypatch, output_exc=RuntimeError("vs boom"))
        reporter = _RecordingReporter()
        harness = TestEpisodeHygiene()
        harness._fake_torch(monkeypatch)
        collected: list[int] = []
        monkeypatch.setattr(gc, "collect", lambda: collected.append(1))
        _no_cuda_telemetry(monkeypatch)
        monkeypatch.setattr(pl, "_WATCHDOG", None)
        monkeypatch.setattr(pl, "_CAP_APPLIED", True)
        with pytest.raises(RuntimeError, match="vs boom"):
            pl.run_pipeline(
                _cpu_ctx(reporter),  # type: ignore[arg-type]
                input_path=Path("fake.mkv"),
                output_path=Path("out.mkv"),
                ffmpeg_path="/usr/bin/ffmpeg",
            )
        assert collected == [1]
        assert reporter.warnings == []  # no degradation noise on a clean-watchdog failure

    def test_progress_update_reaches_graph_through_watchdog(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen_kwargs = _wire_pipeline(monkeypatch, output_exc=None)
        reporter = _RecordingReporter()
        harness = TestEpisodeHygiene()
        harness._fake_torch(monkeypatch)
        monkeypatch.setattr(gc, "collect", lambda: None)
        _no_cuda_telemetry(monkeypatch)
        monkeypatch.setattr(pl, "_WATCHDOG", None)
        monkeypatch.setattr(pl, "_CAP_APPLIED", True)
        forwarded: list[tuple[int, int]] = []

        def original(done: int, total: int) -> None:
            forwarded.append((done, total))

        pl.run_pipeline(
            _cpu_ctx(reporter),  # type: ignore[arg-type]
            input_path=Path("fake.mkv"),
            output_path=Path("out.mkv"),
            ffmpeg_path="/usr/bin/ffmpeg",
            progress_update=original,
        )
        # The graph received A progress callback, but not the bare original.
        assert seen_kwargs and seen_kwargs[0]["progress_update"] is not None
        assert seen_kwargs[0]["progress_update"] is not original
        # The wrapper forwards frame events to the original callback.
        seen_kwargs[0]["progress_update"](7, 100)  # type: ignore[operator]
        assert forwarded == [(7, 100)]

    def test_cap_applied_on_gpu_path_with_device_index(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _wire_pipeline(monkeypatch, output_exc=None)
        reporter = _RecordingReporter()
        harness = TestEpisodeHygiene()
        calls = harness._fake_torch(monkeypatch)
        monkeypatch.setattr(gc, "collect", lambda: None)
        _no_cuda_telemetry(monkeypatch)
        gpu = GPUInfo(index=3, name="RTX 5080", vram_mb=16384, compute_capability=(12, 0))
        monkeypatch.setattr(pl, "_WATCHDOG", None)
        monkeypatch.setattr(pl, "_CAP_APPLIED", False)
        pl.run_pipeline(
            _cpu_ctx(reporter, gpus=(gpu,)),  # type: ignore[arg-type]
            input_path=Path("fake.mkv"),
            output_path=Path("out.mkv"),
            ffmpeg_path="/usr/bin/ffmpeg",
        )
        assert calls["fraction"] == [(0.9, 3)]

    def test_cap_skipped_on_cpu_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _wire_pipeline(monkeypatch, output_exc=None)
        reporter = _RecordingReporter()
        harness = TestEpisodeHygiene()
        calls = harness._fake_torch(monkeypatch)
        monkeypatch.setattr(gc, "collect", lambda: None)
        _no_cuda_telemetry(monkeypatch)
        monkeypatch.setattr(pl, "_WATCHDOG", None)
        monkeypatch.setattr(pl, "_CAP_APPLIED", False)
        pl.run_pipeline(
            _cpu_ctx(reporter),  # type: ignore[arg-type]
            input_path=Path("fake.mkv"),
            output_path=Path("out.mkv"),
            ffmpeg_path="/usr/bin/ffmpeg",
        )
        assert calls["fraction"] == []

    def test_cap_applied_once_per_process(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _wire_pipeline(monkeypatch, output_exc=None)
        reporter = _RecordingReporter()
        harness = TestEpisodeHygiene()
        calls = harness._fake_torch(monkeypatch)
        monkeypatch.setattr(gc, "collect", lambda: None)
        _no_cuda_telemetry(monkeypatch)
        gpu = GPUInfo(index=0, name="RTX 5080", vram_mb=16384, compute_capability=(12, 0))
        monkeypatch.setattr(pl, "_WATCHDOG", None)
        monkeypatch.setattr(pl, "_CAP_APPLIED", False)
        ctx = _cpu_ctx(reporter, gpus=(gpu,))
        for _ in range(2):
            pl.run_pipeline(
                ctx,  # type: ignore[arg-type]
                input_path=Path("fake.mkv"),
                output_path=Path("out.mkv"),
                ffmpeg_path="/usr/bin/ffmpeg",
            )
        assert len(calls["fraction"]) == 1

    def test_cap_failure_warns_but_never_kills_the_episode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _wire_pipeline(monkeypatch, output_exc=None)
        reporter = _RecordingReporter()
        fake_cuda = types.SimpleNamespace(
            is_available=lambda: True,
            set_per_process_memory_fraction=lambda fraction, device: (_ for _ in ()).throw(
                RuntimeError("driver refused")
            ),
        )
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = fake_cuda  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setattr(gc, "collect", lambda: None)
        _no_cuda_telemetry(monkeypatch)
        gpu = GPUInfo(index=0, name="RTX 5080", vram_mb=16384, compute_capability=(12, 0))
        monkeypatch.setattr(pl, "_WATCHDOG", None)
        monkeypatch.setattr(pl, "_CAP_APPLIED", False)
        pl.run_pipeline(
            _cpu_ctx(reporter, gpus=(gpu,)),  # type: ignore[arg-type]
            input_path=Path("fake.mkv"),
            output_path=Path("out.mkv"),
            ffmpeg_path="/usr/bin/ffmpeg",
        )
        assert any("лимит" in w.lower() for w in reporter.warnings)


class TestWatchdogSingleton:
    def test_singleton_persists_across_episodes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        reporter = _RecordingReporter()
        monkeypatch.setattr(pl, "_WATCHDOG", None)
        wd1 = pl._episode_watchdog(reporter)  # type: ignore[arg-type]
        wd2 = pl._episode_watchdog(reporter)  # type: ignore[arg-type]
        assert wd1 is wd2
