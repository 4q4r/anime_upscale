"""ProgressThrottle tests (F3): ≤10%-interval lines, per §10.4 extended to the pipeline."""

from anime_upscale.reporting import ProgressThrottle, Reporter, resolve_mode


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
