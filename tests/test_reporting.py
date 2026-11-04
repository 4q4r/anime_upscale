"""ProgressThrottle tests (F3) + resolve_mode policy matrix (M10, §10.4)."""

import pytest

from anime_upscale.reporting import (
    OutputMode,
    PipelineProgressBar,
    ProgressThrottle,
    Reporter,
    format_eta,
    resolve_mode,
)


class _FakeStream:
    def __init__(self, tty: bool) -> None:
        self._tty = tty
        self.written: list[str] = []

    def isatty(self) -> bool:
        return self._tty

    def write(self, s: str) -> int:
        self.written.append(s)
        return len(s)

    def flush(self) -> None:
        pass


class TestResolveModeMatrix:
    """M10: the --no-color / --no-tty / NO_COLOR policy matrix (§10.4)."""

    def test_real_tty_color_and_tty_on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NO_COLOR", raising=False)
        assert resolve_mode(stream=_FakeStream(True)) == OutputMode(color=True, tty=True)

    def test_no_color_flag_keeps_tty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NO_COLOR", raising=False)
        mode = resolve_mode(no_color=True, stream=_FakeStream(True))
        assert mode == OutputMode(color=False, tty=True)

    def test_no_tty_flag_disables_both(self) -> None:
        mode = resolve_mode(no_tty=True, stream=_FakeStream(True))
        assert mode == OutputMode(color=False, tty=False)

    def test_no_color_env_disables_color_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NO_COLOR", "1")
        mode = resolve_mode(stream=_FakeStream(True))
        assert mode == OutputMode(color=False, tty=True)

    def test_plain_pipe_disables_both(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NO_COLOR", raising=False)
        mode = resolve_mode(stream=_FakeStream(False))
        assert mode == OutputMode(color=False, tty=False)

    def test_both_flags_disable_both(self) -> None:
        mode = resolve_mode(no_color=True, no_tty=True, stream=_FakeStream(True))
        assert mode == OutputMode(color=False, tty=False)

    def test_cli_flags_win_over_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NO_COLOR", "1")
        assert resolve_mode(no_tty=True, stream=_FakeStream(True)).tty is False


class _CollectingReporter(Reporter):
    def __init__(self) -> None:
        super().__init__(resolve_mode(no_color=True, no_tty=True))
        self.lines: list[str] = []

    def line(self, text: str) -> None:
        self.lines.append(text)


def _throttle() -> tuple[ProgressThrottle, _CollectingReporter]:
    reporter = _CollectingReporter()
    return ProgressThrottle(reporter, label="Обработка"), reporter


class TestProgressThrottle:
    def test_full_run_emits_at_most_11_lines(self) -> None:
        """Synthetic per-frame sequence for 720 frames → at most 0..100% in 10% steps + final."""
        throttle, reporter = _throttle()
        total = 720
        for done in range(total + 1):
            throttle.update(done, total)
        assert len(reporter.lines) <= 11
        assert reporter.lines[-1].endswith("(кадр 720/720)")

    def test_interval_bounded_regardless_of_call_count(self) -> None:
        throttle, reporter = _throttle()
        for done in range(0, 100_000, 7):  # dense irregular callbacks
            throttle.update(done, 100_000)
        throttle.update(100_000, 100_000)
        assert len(reporter.lines) <= 12

    def test_line_format_matches_policy(self) -> None:
        throttle, reporter = _throttle()
        throttle.update(0, 10)
        throttle.update(5, 10)
        throttle.update(10, 10)
        assert reporter.lines[0] == "Обработка: 0% (кадр 0/10)"
        assert any("Обработка: 100% (кадр 10/10)" in line for line in reporter.lines)
        assert all("(кадр " in line for line in reporter.lines)

    def test_label_prefix_for_batch_identity(self) -> None:
        reporter = _CollectingReporter()
        throttle = ProgressThrottle(reporter, label="[ep02.mkv] Обработка")
        throttle.update(10, 100)
        assert reporter.lines[0].startswith("[ep02.mkv] Обработка: 10%")

    def test_unknown_total_stays_silent(self) -> None:
        throttle, reporter = _throttle()
        for done in range(0, 500, 50):
            throttle.update(done, 0)
        assert reporter.lines == []

    def test_callback_adapter_signature(self) -> None:
        throttle, reporter = _throttle()
        cb = throttle.callback()
        cb(50, 100)
        cb(51, 100)  # within the same window → suppressed
        assert len(reporter.lines) == 1


class TestFormatEta:
    """Format ETA requirements: >0h -> 'Hч Mмин Sс', 0h -> 'Mмин Sс', 0h0m -> 'Sс'."""

    def test_seconds_only(self) -> None:
        assert format_eta(35) == "35с"
        assert format_eta(0) == "0с"
        assert format_eta(59.4) == "59с"
        assert format_eta(30) == "30с"

    def test_minutes_and_seconds(self) -> None:
        # 8m 42s = 8 * 60 + 42 = 522s
        assert format_eta(522) == "8мин 42с"
        # 5m 12s = 5 * 60 + 12 = 312s
        assert format_eta(312) == "5мин 12с"

    def test_hours_minutes_and_seconds(self) -> None:
        # 1h 24m 15s = 3600 + 1440 + 15 = 5055s
        assert format_eta(5055) == "1ч 24мин 15с"
        # 2h 10m 5s = 7200 + 600 + 5 = 7805s
        assert format_eta(7805) == "2ч 10мин 5с"

    def test_invalid_or_none(self) -> None:
        assert format_eta(None) == "?"
        assert format_eta(-5) == "?"


class _DummyTtyStream(_FakeStream):
    def __init__(self) -> None:
        super().__init__(tty=True)


class TestPipelineProgressBar:
    def test_disabled_on_non_tty(self) -> None:
        stream = _FakeStream(tty=False)
        pbar = PipelineProgressBar(total=100, stream=stream)
        assert pbar.disabled is True
        cb = pbar.callback()
        cb(10, 100)
        pbar.close()
        assert pbar.closed is True

    def test_enabled_on_tty(self) -> None:
        stream = _DummyTtyStream()
        mode = OutputMode(color=True, tty=True)
        with PipelineProgressBar(total=100, stream=stream, mode=mode) as pbar:
            assert pbar.disabled is False
            cb = pbar.callback()
            cb(50, 100)
            assert pbar._current_n == 50
        assert pbar.closed is True

    def test_rate_formatting_in_format_dict(self) -> None:
        stream = _DummyTtyStream()
        mode = OutputMode(color=True, tty=True)
        with PipelineProgressBar(total=100, stream=stream, mode=mode) as pbar:
            assert pbar._pbar is not None
            # Force rate in format_dict
            d = pbar._pbar.format_dict
            assert "кадр/с" in str(d.get("custom_rate"))

    def test_context_manager_closes_on_error(self) -> None:
        stream = _DummyTtyStream()
        mode = OutputMode(color=True, tty=True)
        pbar_ref = None
        try:
            with PipelineProgressBar(total=100, stream=stream, mode=mode) as pbar:
                pbar_ref = pbar
                msg = "boom"
                raise RuntimeError(msg)
        except RuntimeError:
            pass
        assert pbar_ref is not None
        assert pbar_ref.closed is True

    def test_update_changes_total_if_adjusted(self) -> None:
        stream = _DummyTtyStream()
        mode = OutputMode(color=True, tty=True)
        with PipelineProgressBar(total=100, stream=stream, mode=mode) as pbar:
            pbar.update(20, 200)
            assert pbar._total == 200
            assert pbar._current_n == 20


class TestUploadProgressBar:
    def test_disabled_on_non_tty(self) -> None:
        stream = _FakeStream(tty=False)
        reporter = Reporter(resolve_mode(no_color=True, no_tty=True), stream=stream)
        with reporter.upload_progress_bar(total=1024 * 1024, filename="ep01.mkv") as pbar:
            assert pbar.disabled is True
            cb = pbar.callback()
            cb(512 * 1024, 1024 * 1024)
        assert pbar.closed is True
        # Check fallback emitted progress line
        assert any("[ep01.mkv] Загрузка:" in line for line in stream.written)

    def test_enabled_on_tty(self) -> None:
        stream = _DummyTtyStream()
        mode = OutputMode(color=True, tty=True)
        reporter = Reporter(mode, stream=stream)
        with reporter.upload_progress_bar(total=10 * 1024 * 1024, filename="ep02.mkv") as pbar:
            assert pbar.disabled is False
            assert pbar._pbar is not None
            assert "[ep02.mkv] Загрузка" in pbar._pbar.desc
            cb = pbar.callback()
            cb(5 * 1024 * 1024, 10 * 1024 * 1024)
            assert pbar._current_n == 5 * 1024 * 1024
        assert pbar.closed is True

    def test_rate_formatting_in_format_dict(self) -> None:
        stream = _DummyTtyStream()
        mode = OutputMode(color=True, tty=True)
        reporter = Reporter(mode, stream=stream)
        with reporter.upload_progress_bar(total=1024 * 1024, filename="ep03.mkv") as pbar:
            assert pbar._pbar is not None
            d = pbar._pbar.format_dict
            assert "МБ/с" in str(d.get("custom_rate"))
