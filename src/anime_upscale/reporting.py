"""
TTY/color reporting: Russian user-facing output, --no-color / --no-tty policy.

Policy (plan §10.4): animated output only on a real TTY and only when color is
allowed; otherwise plain machine-readable lines with zero ANSI.
"""

import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TextIO

RESET = "\x1b[0m"
BOLD = "\x1b[1m"
CYAN = "\x1b[36m"
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
RED = "\x1b[31m"
MAGENTA = "\x1b[35m"


@dataclass(frozen=True)
class OutputMode:
    """Resolved output policy for one run."""

    color: bool
    tty: bool


def resolve_mode(
    *,
    no_color: bool = False,
    no_tty: bool = False,
    stream: TextIO | None = None,
) -> OutputMode:
    """CLI > NO_COLOR env (platform convention, not config) > TTY detection."""
    stream = stream if stream is not None else sys.stdout
    tty = hasattr(stream, "isatty") and stream.isatty() and not no_tty
    color = tty and not no_color and not os.environ.get("NO_COLOR")
    return OutputMode(color=color, tty=tty)


def enable_windows_console_vt() -> None:
    """
    Enable VT processing on Windows consoles before first colored output (plan §12.4).

    stdlib-only: the ``os.system("")`` trick makes the legacy console host
    honor ANSI escapes; on POSIX this is a no-op. No colorama dependency.
    """
    if os.name == "nt":
        os.system("")  # noqa: S605, S607 - documented VT-enable trick, no shell injection surface


@dataclass
class Reporter:
    """All user-facing output goes through here (Russian, colored on TTY)."""

    mode: OutputMode
    stream: TextIO = field(default_factory=lambda: sys.stdout)

    def _paint(self, text: str, *codes: str) -> str:
        if not self.mode.color or not codes:
            return text
        return "".join(codes) + text + RESET

    def _emit(self, text: str) -> None:
        print(text, file=self.stream, flush=True)

    def line(self, text: str) -> None:
        """Plain line (color applied by the caller)."""
        self._emit(text)

    def banner_line(self, text: str) -> None:
        """Bold-cyan banner line."""
        self._emit(self._paint(text, BOLD, CYAN))

    def file_banner(self, index: int, total: int, name: str) -> None:
        """«◈ Файл N/M: name» batch identity line."""
        self.banner_line(f"◈ Файл {index}/{total}: {name}")

    def warn(self, text: str) -> None:
        """Yellow warning with ⚠ prefix."""
        self._emit(self._paint(f"⚠ {text}", BOLD, YELLOW))

    def error(self, text: str) -> None:
        """Red error with ✗ prefix."""
        self._emit(self._paint(f"✗ Ошибка: {text}", BOLD, RED))

    def success(self, text: str) -> None:
        """Green success line with ✓ prefix."""
        self._emit(self._paint(f"✓ {text}", BOLD, GREEN))

    def note(self, text: str) -> None:
        """Magenta informational note with ⏭ prefix."""
        self._emit(self._paint(f"⏭ {text}", MAGENTA))

    def link(self, url: str) -> None:
        """Print a download page link (upload post-step)."""
        self._emit(self._paint(f"🔗 Ссылка: {url}", BOLD, CYAN))

    def progress_bytes(self, label: str, done: int, total: int) -> None:
        """Plain percentage line, ≤10% granularity (§10.4)."""
        """Plain percentage line, ≤10% granularity (TTY bars are handled by tqdm upstream)."""
        if total <= 0:
            self._emit(f"{label}: {done} байт")
            return
        percent = int(done * 100 / total)
        mebibytes_done = done / (1024 * 1024)
        mebibytes_total = total / (1024 * 1024)
        self._emit(f"{label}: {percent}% ({mebibytes_done:.0f}/{mebibytes_total:.0f} МиБ)")

    def hardware_banner(self, gpus: str, streams: str, stages: str, encode: str) -> None:
        """Print the three §9.6 banner lines."""
        self.banner_line(f"◈ Оборудование: {gpus}")
        self.banner_line(f"◈ Стадии: {stages}")
        self.banner_line(f"◈ Инференс: {streams} · Кодирование: {encode} потоков")

    def batch_summary(self, results: list[tuple[str, bool, str]]) -> None:
        """One line per entry + total («Итог: 4/5 успешно, 1 с ошибкой»)."""
        ok_count = sum(1 for _, ok, _ in results if ok)
        for name, ok, detail in results:
            if ok:
                self.success(name)
            else:
                self.error(f"{name}: {detail}")
        failed = len(results) - ok_count
        self.banner_line(f"Итог: {ok_count}/{len(results)} успешно, {failed} с ошибкой")


class ProgressThrottle:
    """
    ≤10%-interval pipeline progress lines (§10.4 policy, extended to the pipeline).

    The VS ``output(progress_update=...)`` callback fires per frame (~86k calls
    per episode) — emitting every call would flood the console. This throttle
    emits at most one line per 10% step plus the final frame; unknown totals
    stay silent (nothing honest to report).
    """

    def __init__(self, reporter: Reporter, label: str = "Обработка") -> None:
        """Bind a reporter and a line label (batch entries prefix the file name)."""
        self._reporter = reporter
        self._label = label
        self._last_percent = -10  # so the first in-window step (0%) is not skipped

    def update(self, done: int, total: int) -> bool:
        """Feed one progress callback; returns True when a line was emitted."""
        if total <= 0 or done < 0:
            return False
        percent = min(100, int(done * 100 / total))
        at_end = done >= total
        if not at_end and percent < self._last_percent + 10:
            return False
        self._last_percent = percent
        self._reporter.line(f"{self._label}: {percent}% (кадр {done}/{total})")
        return True

    def callback(self) -> Callable[[int, int], None]:
        """Adapter matching the VS output(progress_update=…) signature."""

        def _update(done: int, total: int) -> None:
            self.update(done, total)

        return _update
