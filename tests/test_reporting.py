"""ProgressThrottle tests (F3) + resolve_mode policy matrix (M10, §10.4)."""

import pytest
import tqdm.std

from anime_upscale.reporting import (
    OutputMode,
    PipelineProgressBar,
    ProgressThrottle,
    Reporter,
    _eta_seconds,
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
        assert format_eta(None) == "—"
        assert format_eta(-5) == "—"


class _FakeClock:
    """Deterministic tqdm clock; rebind tqdm.std.time BEFORE bar construction."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def advance(self, dt: float) -> None:
        self.now += dt

    def __call__(self) -> float:
        return self.now


def _tty_pipeline_bar(
    monkeypatch: pytest.MonkeyPatch, clock: _FakeClock, *, total: int
) -> PipelineProgressBar:
    monkeypatch.setattr(tqdm.std, "time", clock)
    return PipelineProgressBar(
        total=total,
        stream=_DummyTtyStream(),
        mode=OutputMode(color=True, tty=True),
    )


class TestEtaFromFormatDict:
    """Regression: tqdm 4.70.1 format_dict has NO `remaining` key (upstream computes
    it only inside format_meter's render dict), so ETA must be derived from
    rate/total/n (elapsed-based fallback) — «?» is never a valid ETA."""

    def test_eta_from_rate_at_episode_scale(self, monkeypatch: pytest.MonkeyPatch) -> None:
        clock = _FakeClock()
        with _tty_pipeline_bar(monkeypatch, clock, total=85211) as pbar:
            clock.advance(10.0)
            pbar.update(17664, 85211)  # 21% — the owner's live screenshot state
            d = pbar._pbar.format_dict
            assert "remaining" not in d  # the upstream gap this fix routes around
            assert d["rate"] == pytest.approx(1766.4)  # 17664 frames / 10 s
            assert d["custom_eta"] == "38с"  # (85211 - 17664) / 1766.4 = 38.24 -> 38с
            assert d["custom_rate"] == "1766.4 кадр/с"
            rendered = str(pbar._pbar)
            assert "[17664/85211] [38с, 1766.4 кадр/с]" in rendered
            assert "?" not in rendered

    def test_eta_dash_and_rate_dash_at_zero_progress(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with _tty_pipeline_bar(monkeypatch, _FakeClock(), total=7) as pbar:
            d = pbar._pbar.format_dict
            assert d["custom_eta"] == "—"
            assert d["custom_rate"] == "—"
            assert "?" not in str(pbar._pbar)

    def test_eta_dash_when_rate_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        clock = _FakeClock()
        with _tty_pipeline_bar(monkeypatch, clock, total=1000) as pbar:
            pbar.update(100, 1000)  # dt == 0 -> no EMA sample -> rate stays None
            d = pbar._pbar.format_dict
            assert d["rate"] is None
            assert d["custom_eta"] == "—"
            assert d["custom_rate"] == "—"

    def test_upload_eta_from_rate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        clock = _FakeClock()
        monkeypatch.setattr(tqdm.std, "time", clock)
        reporter = Reporter(OutputMode(color=True, tty=True), stream=_DummyTtyStream())
        with reporter.upload_progress_bar(total=10 * 1024 * 1024, filename="ep02.mkv") as pbar:
            clock.advance(2.0)
            pbar.update(5 * 1024 * 1024, 10 * 1024 * 1024)
            d = pbar._pbar.format_dict
            assert d["custom_eta"] == "2с"  # (10 - 5) MiB / (5 MiB / 2 s) = 2 s
            assert d["custom_rate"] == "2.5 МБ/с"
            assert "?" not in str(pbar._pbar)

    def test_elapsed_fallback_when_rate_not_yet_sampled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = _FakeClock()
        with _tty_pipeline_bar(monkeypatch, clock, total=85211) as pbar:
            clock.advance(0.05)  # below mininterval -> no EMA sample -> rate stays None
            pbar.update(17664, 85211)
            d = pbar._pbar.format_dict
            assert d["rate"] is None
            # (85211 - 17664) frames * 0.05 s / 17664 frames ~= 0.19 s -> rounds to 0.
            assert d["custom_eta"] == "0с"

    def test_helper_rate_and_elapsed_paths(self) -> None:
        # Rate path at full episode scale (85211 frames must not break the arithmetic).
        d_rate = {"total": 85211, "n": 17664, "rate": 1766.4}
        assert _eta_seconds(d_rate) == pytest.approx((85211 - 17664) / 1766.4)
        # Elapsed fallback at hour scale: 84211 frames * 100 s / 1000 frames = 8421.1 s.
        d_elapsed = {"total": 85211, "n": 1000, "rate": None, "elapsed": 100.0}
        assert _eta_seconds(d_elapsed) == pytest.approx(8421.1)
        assert format_eta(_eta_seconds(d_elapsed)) == "2ч 20мин 21с"

    def test_helper_unknown_inputs_yield_none(self) -> None:
        assert _eta_seconds({}) is None
        assert _eta_seconds({"total": None, "n": 0, "rate": None, "elapsed": 0.0}) is None
        assert _eta_seconds({"total": 10, "n": 0, "rate": None, "elapsed": 5.0}) is None
        assert _eta_seconds({"total": 10, "n": 5, "rate": 0, "elapsed": 0.0}) is None
        assert _eta_seconds({"total": 10, "n": 10, "rate": 5.0, "elapsed": 2.0}) is None


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

    def test_rate_formatting_in_format_dict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        clock = _FakeClock()
        with _tty_pipeline_bar(monkeypatch, clock, total=100) as pbar:
            assert pbar._pbar is not None
            # At zero progress there is no honest rate: dash, not «?».
            assert pbar._pbar.format_dict.get("custom_rate") == "—"
            clock.advance(10.0)
            pbar.update(50, 100)
            assert pbar._pbar.format_dict.get("custom_rate") == "5.0 кадр/с"

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

    def test_rate_formatting_in_format_dict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        clock = _FakeClock()
        monkeypatch.setattr(tqdm.std, "time", clock)
        stream = _DummyTtyStream()
        mode = OutputMode(color=True, tty=True)
        reporter = Reporter(mode, stream=stream)
        with reporter.upload_progress_bar(total=1024 * 1024, filename="ep03.mkv") as pbar:
            assert pbar._pbar is not None
            # At zero progress there is no honest rate: dash, not «?».
            assert pbar._pbar.format_dict.get("custom_rate") == "—"
            clock.advance(2.0)
            pbar.update(512 * 1024, 1024 * 1024)
            d = pbar._pbar.format_dict
            assert d.get("custom_rate") == "0.2 МБ/с"


class TestCurrentRateProperty:
    """P1 RateWatch feed: the bar exposes tqdm's OWN rate (the same number the
    bar renders) — the detector must never re-measure progress itself."""

    def test_none_before_any_honest_rate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        clock = _FakeClock()
        with _tty_pipeline_bar(monkeypatch, clock, total=100) as pbar:
            assert pbar.current_rate is None
            pbar.update(10, 100)  # dt == 0 → no EMA sample yet
            assert pbar.current_rate is None

    def test_rate_matches_format_dict_number(self, monkeypatch: pytest.MonkeyPatch) -> None:
        clock = _FakeClock()
        with _tty_pipeline_bar(monkeypatch, clock, total=100) as pbar:
            clock.advance(10.0)
            pbar.update(50, 100)
            assert pbar.current_rate == pytest.approx(5.0)
            assert pbar.current_rate == pbar._pbar.format_dict["rate"]

    def test_none_when_bar_disabled_non_tty(self) -> None:
        pbar = PipelineProgressBar(total=100, stream=_FakeStream(tty=False))
        assert pbar.current_rate is None
        pbar.update(50, 100)
        assert pbar.current_rate is None


class TestDownloadProgressBar:
    """Бар скачивания = та же ETA-механика #41, что на барах обработки/выгрузки:
    проценты, размер, rate (МБ/с) и русский ETA через _eta_seconds — никогда «?»."""

    def test_factory_desc_and_enabled_on_tty(self) -> None:
        stream = _DummyTtyStream()
        mode = OutputMode(color=True, tty=True)
        reporter = Reporter(mode, stream=stream)
        with reporter.download_progress_bar(total=10 * 1024 * 1024, filename="ep03.mkv") as pbar:
            assert pbar.disabled is False
            assert pbar._pbar is not None
            assert "[ep03.mkv] Скачивание" in pbar._pbar.desc
            cb = pbar.callback()
            cb(5 * 1024 * 1024, 10 * 1024 * 1024)
            assert pbar._current_n == 5 * 1024 * 1024
        assert pbar.closed is True

    def test_eta_and_rate_format_dict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        clock = _FakeClock()
        monkeypatch.setattr(tqdm.std, "time", clock)
        stream = _DummyTtyStream()
        mode = OutputMode(color=True, tty=True)
        reporter = Reporter(mode, stream=stream)
        with reporter.download_progress_bar(total=1024 * 1024, filename="ep.mkv") as pbar:
            clock.advance(2.0)
            pbar.update(512 * 1024, 1024 * 1024)
            d = pbar._pbar.format_dict  # ty: ignore[index]
            assert d.get("custom_rate") == "0.2 МБ/с"
            eta = d.get("custom_eta")
            assert eta is not None and "?" not in eta
            # (1MiB - 0.5MiB) / (0.5MiB/2s) = 2 s.
            assert eta == "2с"

    def test_non_tty_fallback_throttled_lines(self) -> None:
        stream = _FakeStream(tty=False)
        reporter = Reporter(resolve_mode(no_color=True, no_tty=True), stream=stream)
        with reporter.download_progress_bar(total=100 * 1024 * 1024, filename="ep.mkv") as pbar:
            cb = pbar.callback()
            for step in range(1, 11):
                cb(step * 10 * 1024 * 1024, 100 * 1024 * 1024)
        lines = [line for line in stream.written if "Скачивание" in line]
        assert lines, "не-TTY фолбэк обязан печатать долю"
        assert all("?" not in line for line in lines)
        assert any("100%" in line for line in lines)

    def test_update_backward_reset_restarts_progress(self) -> None:
        """Retry after a failed attempt rewinds the callback to small offsets —
        the bar must accept the reset (tqdm counter included) instead of
        freezing until the new attempt passes the old maximum."""
        stream = _DummyTtyStream()
        mode = OutputMode(color=True, tty=True)
        reporter = Reporter(mode, stream=stream)
        with reporter.download_progress_bar(total=100, filename="ep.mkv") as pbar:
            pbar.update(90, 100)
            assert pbar._pbar.n == 90  # ty: ignore[index]
            pbar.update(5, 100)
            assert pbar._current_n == 5
            assert pbar._pbar.n == 5  # ty: ignore[index]

    def test_upload_bar_reset_safe_too(self) -> None:
        stream = _DummyTtyStream()
        mode = OutputMode(color=True, tty=True)
        reporter = Reporter(mode, stream=stream)
        with reporter.upload_progress_bar(total=100, filename="ep.mkv") as pbar:
            pbar.update(90, 100)
            pbar.update(5, 100)
            assert pbar._current_n == 5
            assert pbar._pbar.n == 5  # ty: ignore[index]
