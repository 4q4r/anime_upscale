"""ProgressThrottle tests (F3) + resolve_mode policy matrix (M10, §10.4)."""

import pytest

from anime_upscale.reporting import OutputMode, ProgressThrottle, Reporter, resolve_mode


class _FakeStream:
    def __init__(self, tty: bool) -> None:
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


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
