"""
TTY/color reporting: Russian user-facing output, --no-color / --no-tty policy.

Policy (plan §10.4): animated output only on a real TTY and only when color is
allowed; otherwise plain machine-readable lines with zero ANSI.
"""

import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Self, TextIO

from tqdm import tqdm

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

    def progress_bar(
        self,
        *,
        total: int | None = None,
        label: str = "Обработка",
    ) -> "PipelineProgressBar":
        """Create a pipeline progress bar respecting this reporter's mode."""
        return PipelineProgressBar(
            total=total,
            label=label,
            stream=self.stream,
            mode=self.mode,
        )

    def upload_progress_bar(
        self,
        *,
        total: int,
        filename: str,
    ) -> "UploadProgressBar":
        """Create a network upload progress bar respecting this reporter's mode."""
        return UploadProgressBar(
            total=total,
            filename=filename,
            stream=self.stream,
            mode=self.mode,
        )


def format_eta(seconds: float | None) -> str:
    """
    Format remaining time (ETA) without leading zero units.

    - > 0 hours: `Чч Ммин Сс` (e.g. `1ч 24мин 15с`)
    - 0 hours and > 0 minutes: `Ммин Сс` (e.g. `8мин 42с`)
    - 0 hours and 0 minutes: `Сс` (e.g. `35с`)
    - None or negative: `—` (unknown ETA is never rendered as «?»)
    """
    if seconds is None or seconds < 0:
        return "—"
    total_seconds = round(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    if hours > 0:
        return f"{hours}ч {minutes}мин {secs}с"
    if minutes > 0:
        return f"{minutes}мин {secs}с"
    return f"{secs}с"


def _eta_seconds(d: Mapping[str, object]) -> float | None:
    """
    Derive remaining seconds from a tqdm ``format_dict`` snapshot.

    tqdm (verified against 4.70.1, std.py ``format_dict`` property) exposes only
    ``n/total/elapsed/rate/...`` — the ``remaining`` key exists solely inside
    ``format_meter``'s render-time dict, so it must be recomputed here:

    - primary: ``remaining = (total - n) / rate`` when the EMA rate is available;
    - fallback: ``remaining = (total - n) * elapsed / n`` when no rate sample yet;
    - ``None`` when nothing honest is derivable (no progress, done, unknown total).
    """
    total = d.get("total")
    n = d.get("n")
    if not isinstance(total, (int, float)) or not isinstance(n, (int, float)):
        return None
    if total <= 0 or n <= 0 or n >= total:
        return None

    rate = d.get("rate")
    if isinstance(rate, (int, float)) and rate > 0:
        return (total - n) / rate

    elapsed = d.get("elapsed")
    if isinstance(elapsed, (int, float)) and elapsed > 0:
        return (total - n) * elapsed / n
    return None


class _CustomTqdm(tqdm):
    """tqdm subclass with customized format_dict for Russian time/rate formatting."""

    @property
    def format_dict(self) -> dict[str, object]:
        d = super().format_dict
        rate = d.get("rate")
        if rate:
            d["custom_rate"] = f"{rate:.1f} кадр/с"
        else:
            d["custom_rate"] = "—"

        d["custom_eta"] = format_eta(_eta_seconds(d))
        return d


class _UploadCustomTqdm(tqdm):
    """tqdm subclass with customized format_dict for network upload rate and Russian ETA."""

    @property
    def format_dict(self) -> dict[str, object]:
        d = super().format_dict
        rate = d.get("rate")
        if rate:
            # rate is in bytes per second because unit="B", unit_scale=True, unit_divisor=1024
            mbytes = rate / (1024 * 1024)
            d["custom_rate"] = f"{mbytes:.1f} МБ/с"
        else:
            d["custom_rate"] = "—"

        d["custom_eta"] = format_eta(_eta_seconds(d))
        return d


class UploadProgressBar:
    """
    Smooth, live TQDM progress bar for network uploads (plan §10).

    Shows:
    - Prefix: `[{filename}] Загрузка`
    - Percentage: every 1% update
    - Transferred bytes out of total with unit_divisor=1024 (e.g. 512.0M/10.0G or MiB/GiB)
    - Real-time speed in `МБ/с`
    - Compact Russian ETA formatted via `format_eta` (`35с`, `8мин 42с`, `1ч 24мин 15с`)
    - Context manager protocol (`with reporter.upload_progress_bar(...) as pbar:`)
    - Safe non-interactive / non-TTY fallback (line-buffered progress_bytes).
    """

    def __init__(
        self,
        *,
        total: int,
        filename: str,
        stream: TextIO | None = None,
        mode: OutputMode | None = None,
        mininterval: float = 0.1,
    ) -> None:
        """Initialize upload progress bar with filename, total bytes, and output mode."""
        self._stream = stream if stream is not None else sys.stdout
        self._mode = mode if mode is not None else resolve_mode(stream=self._stream)
        self._filename = filename
        self._total = total
        self._current_n = 0
        self._closed = False

        is_tty = self._mode.tty and hasattr(self._stream, "isatty") and self._stream.isatty()
        self._disable = not is_tty

        desc = f"[{filename}] Загрузка"
        bar_format = (
            "{desc}: {percentage:3.0f}%|{bar}| [{n_fmt}/{total_fmt}] [{custom_eta}, {custom_rate}]"
        )

        self._pbar: tqdm | None = None
        self._fallback_reporter: Reporter | None = None
        if not self._disable:
            self._pbar = _UploadCustomTqdm(
                total=total,
                desc=desc,
                file=self._stream,
                bar_format=bar_format,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                dynamic_ncols=True,
                mininterval=mininterval,
                leave=True,
            )
        else:
            self._fallback_reporter = Reporter(mode=self._mode, stream=self._stream)

    @property
    def disabled(self) -> bool:
        """Whether dynamic display is disabled (e.g. non-TTY)."""
        return self._disable

    @property
    def closed(self) -> bool:
        """Whether the progress bar has been closed."""
        return self._closed

    def update(self, done: int, total: int | None = None) -> None:
        """Update progress with current transferred bytes and optional total."""
        if self._closed:
            return
        if total is not None and total > 0 and self._total != total:
            self._total = total
            if self._pbar is not None:
                self._pbar.total = total

        delta = done - self._current_n
        self._current_n = done

        if self._pbar is not None and delta > 0:
            self._pbar.update(delta)
        elif self._fallback_reporter is not None:
            self._fallback_reporter.progress_bytes(
                f"[{self._filename}] Загрузка", done, self._total
            )

    def close(self) -> None:
        """Close and cleanly finalize the progress bar."""
        if self._closed:
            return
        self._closed = True
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None

    def __enter__(self) -> Self:
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit context manager, ensuring progress bar is closed."""
        self.close()

    def callback(self) -> Callable[[int, int], None]:
        """Adapter matching the progress(sent, total) signature."""

        def _update(done: int, total: int) -> None:
            self.update(done, total)

        return _update


class PipelineProgressBar:
    """
    Smooth, live TQDM progress bar for VapourSynth processing.

    Accepts (done, total) updates from `clip.output(progress_update=...)`.
    Shows:
    - Percentage and smooth progress bar.
    - Rate in frames/sec (`124.5 кадр/с`).
    - Exact frame count: `[кадр/всего]`.
    - Compact Russian ETA without leading zeros (`1ч 24мин 15с`, `8мин 42с`, `35с`).
    - Clean shutdown/close on finish and error.
    - Safe non-interactive / no-tty fallback (disabled or line-buffered, no cursor thrashing).
    """

    def __init__(
        self,
        *,
        total: int | None = None,
        label: str = "Обработка",
        stream: TextIO | None = None,
        mode: OutputMode | None = None,
        mininterval: float = 0.1,
    ) -> None:
        """Initialize pipeline progress bar with optional total frame count and label."""
        self._stream = stream if stream is not None else sys.stdout
        self._mode = mode if mode is not None else resolve_mode(stream=self._stream)
        self._label = label
        self._total = total
        self._current_n = 0
        self._closed = False

        # If non-TTY, disable dynamic bar to prevent carriage return spam
        is_tty = self._mode.tty and hasattr(self._stream, "isatty") and self._stream.isatty()
        self._disable = not is_tty

        bar_format = (
            "{desc}: {percentage:3.0f}%|{bar}| [{n_fmt}/{total_fmt}] [{custom_eta}, {custom_rate}]"
        )

        self._pbar: tqdm | None = None
        self._fallback_throttle: ProgressThrottle | None = None
        if not self._disable:
            self._pbar = _CustomTqdm(
                total=total,
                desc=label,
                file=self._stream,
                bar_format=bar_format,
                unit="кадр",
                unit_scale=False,
                dynamic_ncols=True,
                mininterval=mininterval,
                leave=True,
            )
        else:
            # Fallback throttle for non-TTY / captured streams (M10, §10.4)
            reporter = Reporter(mode=self._mode, stream=self._stream)
            self._fallback_throttle = ProgressThrottle(reporter, label=label)

    @property
    def disabled(self) -> bool:
        """Whether dynamic display is disabled (e.g. non-TTY)."""
        return self._disable

    @property
    def closed(self) -> bool:
        """Whether the progress bar has been closed."""
        return self._closed

    def update(self, done: int, total: int) -> None:
        """Update progress from VapourSynth progress_update(done, total)."""
        if self._closed:
            return
        if total > 0 and self._total != total:
            self._total = total
            if self._pbar is not None:
                self._pbar.total = total

        delta = done - self._current_n
        self._current_n = done

        if self._pbar is not None and delta > 0:
            self._pbar.update(delta)
        elif self._fallback_throttle is not None:
            self._fallback_throttle.update(done, total)

    def close(self) -> None:
        """Close and cleanly finalize the progress bar."""
        if self._closed:
            return
        self._closed = True
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None

    def __enter__(self) -> Self:
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit context manager, ensuring progress bar is closed."""
        self.close()

    def callback(self) -> Callable[[int, int], None]:
        """Adapter matching the VS output(progress_update=…) signature."""

        def _update(done: int, total: int) -> None:
            self.update(done, total)

        return _update


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
