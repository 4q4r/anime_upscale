"""
OS power-management guard for batch runs (anti-throttling wave P1).

Why: Windows puts long console batches to sleep via the system idle timer
(CPU/GPU activity does NOT reset it — learn.microsoft.com «system sleep
criteria», fetched 2026-11-07) and classifies background processes through
EcoQoS/Low-QoS heuristics unless the process opts out (SetProcessInformation,
the documented «HighQoS» idiom ControlMask=EXECUTION_SPEED, StateMask=0).
This module provides four stdlib-only instruments:

- :func:`keep_system_awake` — SetThreadExecutionState(ES_CONTINUOUS |
  ES_SYSTEM_REQUIRED) held for the whole batch, reset in ``finally`` (Ctrl+C
  discipline; ES_DISPLAY_REQUIRED deliberately NOT taken — the display may
  sleep, only the system idle timer is the enemy);
- :func:`disable_power_throttling` — one-per-process EcoQoS opt-out;
- :func:`run_startup_doctor` — read-only warn-only checks (active power plan
  GUID, battery presence on Windows; legacy cpufreq governors on Linux);
  never changes a single system setting;
- :class:`RateWatch` — sliding-window fps-drop detector fed by the progress
  bar's own rate (the same number tqdm renders — no duplicate measurement).

Runtime: stdlib only (ctypes/subprocess/sysfs reads) — importable before any
VapourSynth/torch stack loads. Windows API surfaces are module seams so the
whole test matrix runs on Linux with fakes.

Russian user-facing strings follow the repo canon (workflow memory
2026-09-20: «юзеру надо понимать что происходит»), loud but warn-only —
the pipeline never blocks on power findings (owner decision, wave P1).
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from anime_upscale.runtime_env import detect_runtime

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from anime_upscale.reporting import Reporter

# --- Windows API constants (verified against learn.microsoft.com 2026-11-07
# and the mingw-w64 processthreadsapi.h mirror; see the module docstring) ----

ES_CONTINUOUS: Final = 0x80000000
ES_SYSTEM_REQUIRED: Final = 0x00000001

# SetProcessInformation(ProcessPowerThrottling, PROCESS_POWER_THROTTLING_STATE):
# the class value is the zero-based position of ProcessPowerThrottling in the
# PROCESS_INFORMATION_CLASS enum (official declaration order).
_PROCESS_POWER_THROTTLING_CLASS: Final = 4
_PROCESS_POWER_THROTTLING_CURRENT_VERSION: Final = 1
_PROCESS_POWER_THROTTLING_EXECUTION_SPEED: Final = 0x1

# Built-in power plan GUIDs (powercfg /aliases; locale-independent — names are
# localized, GUIDs are not, so comparison is by GUID only).
GUID_BALANCED: Final = "381b4222-f694-41f0-9685-ff5bb260df2e"
GUID_HIGH_PERFORMANCE: Final = "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"
GUID_POWER_SAVER: Final = "a1841308-3541-4fab-bc81-f71556f20b4a"
GUID_ULTIMATE: Final = "e9a42b02-d5df-448d-aa00-03f14749eb61"

# Linux cpufreq: warn only for legacy drivers + powersave. intel_pstate /
# amd-pstate ship powersave as their DEFAULT governor over HWP/CPPC hardware
# (kernel doc admin-guide/pm/cpufreq.html) — warning there would false-alarm
# nearly every modern Linux box (design anti-goal).
LEGACY_CPUFREQ_DRIVERS: Final = frozenset({"acpi-cpufreq", "intel_cpufreq"})
_CPUFREQ_GOVERNOR_PATH: Final = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
_CPUFREQ_DRIVER_PATH: Final = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_driver"

_GUID_RE: Final = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
_POWERCFG_TIMEOUT: Final = 10  # seconds; any doctor subprocess must stay bounded

# --- RateWatch thresholds (module constants with rationale — deliberately
# NOT config keys: no invented scope, revisit after live runs; plan §P2) ------
#
# WINDOW=5s: long enough to smooth per-frame jitter, short enough to catch a
# throttle/sleep within seconds. WARMUP=2: engine build and cache warm-up make
# early windows dishonest. BASELINE=4 (windows 3-6): median over four windows
# follows the repo benchmark canon («warm-up + медиана», scripts/benchmark.py).
# DROP=0.6 with 2 consecutive windows: one dropped window may be a phase hiccup
# or a foreground app stealing the GPU; two in a row at <60% is a real sag.
RATE_WINDOW_SECONDS: Final = 5.0
RATE_WARMUP_WINDOWS: Final = 2
RATE_BASELINE_WINDOWS: Final = 4
RATE_DROP_RATIO: Final = 0.6
RATE_LOW_STREAK: Final = 2

RATE_DROP_WARN: Final = (
    "Скорость упала до {pct}% от начальной ({base:.1f} → {current:.1f} кадр/с). "
    "Похоже на троттлинг, сон системы или переход ОС в экономный режим. "
    "Проверьте план питания (powercfg /getactivescheme на Windows; "
    "cpupower frequency-info на Linux) и частоты GPU "
    "(nvidia-smi --query-gpu=clocks.sm,utilization.gpu,pstate --format=csv). "
    "Прогон продолжается; подробности — docs/performance.md."
)

_SYSTEMD_INHIBIT_NOTE: Final = (
    "На Linux сон хоста изнутри процесса не удерживается — для гарантии "
    'оберните запуск: systemd-inhibit --who=anime_upscale --why="батч" '
    "anime-pipeline ... (см. docs/performance.md)."
)

_ECOQOS_NOTE: Final = (
    "EcoQoS opt-out недоступен ({reason}) — на батарее ОС может снижать частоту фоновых процессов."
)

# Once-per-process guard for the EcoQoS opt-out (ffmpeg_repair `_repair_state`
# pattern: module-level mutable state, mutated in place, reset by tests).
_ECOQOS_CACHE: Final[dict[str, bool]] = {"attempted": False, "ok": False}


def _reset_process_guards() -> None:
    """Clear once-per-process caches (test support only)."""
    _ECOQOS_CACHE["attempted"] = False
    _ECOQOS_CACHE["ok"] = False


# --- Windows seams (every external surface; Linux CI fakes them) -------------


def _is_windows() -> bool:
    return os.name == "nt"


def _set_thread_execution_state(flags: int) -> int:
    """Seam: kernel32.SetThreadExecutionState; OSError on the NULL return."""
    result = ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_ulong(flags))  # ty: ignore[unresolved-attribute]
    if result == 0:
        msg = "SetThreadExecutionState вернула 0"
        raise OSError(msg)
    return int(result)


def _get_ac_line_status() -> int | None:
    """
    Seam: GetSystemPowerStatus → ACLineStatus (0=battery, 1=AC, 255=unknown).

    None = the probe itself failed; the doctor must then say «не проверили»,
    never «всё хорошо» (no silent success — design anti-goal).
    """

    class _SystemPowerStatus(ctypes.Structure):
        _fields_ = [
            ("ACLineStatus", ctypes.c_ubyte),
            ("BatteryFlag", ctypes.c_ubyte),
            ("BatteryLifePercent", ctypes.c_ubyte),
            ("SystemStatusFlag", ctypes.c_ubyte),
            ("BatteryLifeTime", ctypes.c_ulong),
            ("BatteryFullLifeTime", ctypes.c_ulong),
        ]

    status = _SystemPowerStatus()
    ok = ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status))  # ty: ignore[unresolved-attribute]
    if not ok:
        return None
    return int(status.ACLineStatus)


def _set_process_information_optout() -> tuple[bool, str]:
    """
    Apply the documented EcoQoS opt-out.

    ControlMask=EXECUTION_SPEED, StateMask=0 — Microsoft's own «HighQoS»
    example on the SetProcessInformation page. Returns (ok, reason); never
    raises.
    """

    class _PowerThrottlingState(ctypes.Structure):
        _fields_ = [
            ("Version", ctypes.c_ulong),
            ("ControlMask", ctypes.c_ulong),
            ("StateMask", ctypes.c_ulong),
        ]

    state = _PowerThrottlingState(
        _PROCESS_POWER_THROTTLING_CURRENT_VERSION,
        _PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
        0,
    )
    try:
        kernel32 = ctypes.windll.kernel32  # ty: ignore[unresolved-attribute]
        ok = kernel32.SetProcessInformation(
            kernel32.GetCurrentProcess(),
            _PROCESS_POWER_THROTTLING_CLASS,
            ctypes.byref(state),
            ctypes.sizeof(state),
        )
    except OSError as exc:  # defensive: old/odd systems, missing entry point
        return False, str(exc)
    if not ok:
        # No GetLastError code here: without use_last_error=True the value is
        # unreliable — reporting a fabricated code would be dishonest.
        return False, "SetProcessInformation вернула 0"
    return True, ""


def _run_powercfg(args: list[str]) -> tuple[int, str]:
    """
    Seam: ``powercfg <args>`` → (returncode, stdout), bounded timeout.

    (-1, reason) covers spawn failures/timeouts so callers never re-raise.
    """
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no user input
            ["powercfg", *args],  # noqa: S607 - documented system tool, PATH lookup
            capture_output=True,
            text=True,
            timeout=_POWERCFG_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return -1, str(exc)
    return proc.returncode, proc.stdout


def _read_sysfs(path: str) -> str | None:
    """
    Seam: one small sysfs read; None = missing/unreadable (silent skip).

    ValueError covers decode surprises (UnicodeDecodeError ⊂ ValueError).
    """
    try:
        return Path(path).read_text(encoding="ascii").strip()
    except (OSError, ValueError):
        return None


# --- keep-awake --------------------------------------------------------------


@dataclass(frozen=True)
class KeepAwakeStatus:
    """Outcome of the keep-awake attempt (honest, never a fake success)."""

    active: bool
    detail: str


@contextlib.contextmanager
def keep_system_awake(reporter: Reporter) -> Generator[KeepAwakeStatus, None, None]:
    """
    Hold the system in the working state for the enclosed batch work.

    Windows: SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) on
    entry, ES_CONTINUOUS alone in ``finally`` — the documented best practice
    «call ... only for active operations, and always clear with
    SetThreadExecutionState(ES_CONTINUOUS)» (system-sleep-criteria page).
    The reset runs on every exit path, including KeyboardInterrupt unwinding.

    Linux: no in-process keep-awake API — one note pointing at the
    systemd-inhibit wrapper (docs/performance.md).

    Cloud runtimes (kaggle/colab): silent no-op — platform sessions have no
    host sleep manager to fight.

    Failure = one warn + the batch continues (soft, but never silent).
    """
    if detect_runtime().kind != "local":
        yield KeepAwakeStatus(active=False, detail="cloud")
        return

    if not _is_windows():
        reporter.note(_SYSTEMD_INHIBIT_NOTE)
        yield KeepAwakeStatus(active=False, detail="linux")
        return

    active = False
    try:
        _set_thread_execution_state(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        active = True
    except OSError as exc:
        reporter.warn(
            f"Keep-awake не удалось включить ({exc}) — система может уснуть посреди батча."
        )
    try:
        yield KeepAwakeStatus(active, "ES_CONTINUOUS|ES_SYSTEM_REQUIRED" if active else "")
    finally:
        if active:
            try:
                _set_thread_execution_state(ES_CONTINUOUS)
            except OSError as exc:
                reporter.warn(
                    f"Keep-awake не удалось снять ({exc}) — при выходе из процесса "
                    "запрос сбросится сам."
                )


# --- EcoQoS opt-out ----------------------------------------------------------


def disable_power_throttling(reporter: Reporter) -> bool:
    """
    Opt this process out of EcoQoS heuristics — once per process.

    Main + each fan-out child calls it; repeats are cached no-ops.

    True = the opt-out is applied (or was applied earlier in this process).
    False = not applicable (non-Windows) or the API refused — one quiet note
    per process, never a warning storm, never an exception.
    """
    if _ECOQOS_CACHE["attempted"]:
        return _ECOQOS_CACHE["ok"]
    _ECOQOS_CACHE["attempted"] = True
    if not _is_windows():
        _ECOQOS_CACHE["ok"] = False
        return False
    ok, reason = _set_process_information_optout()
    _ECOQOS_CACHE["ok"] = ok
    if not ok:
        reporter.note(_ECOQOS_NOTE.format(reason=reason))
    return ok


# --- startup doctor (read-only, warn-only) -----------------------------------


def run_startup_doctor(reporter: Reporter) -> list[str]:
    """
    Run the read-only power checks and print the findings.

    Every check is best-effort: a failed probe becomes a note («не смогли
    проверить»), never a fake all-clear and never a fatal error. Returns the
    list of emitted warnings (the ⚠ texts; notes are informational and are
    not counted). Cloud runtimes skip the whole doctor with one note.
    """
    runtime = detect_runtime()
    if runtime.kind != "local":
        reporter.note(f"Облачная сессия ({runtime.kind}) — power-чеки пропущены.")
        return []
    warnings: list[str] = []
    if _is_windows():
        warnings.extend(_doctor_windows(reporter))
    elif sys.platform == "linux":
        warnings.extend(_doctor_linux(reporter))
    return warnings


def _active_scheme_guid() -> str | None:
    """Read-only: GUID of the active power plan, lowercase; None on any failure."""
    returncode, out = _run_powercfg(["/getactivescheme"])
    if returncode != 0:
        return None
    match = _GUID_RE.search(out)
    return match.group(0).lower() if match else None


def _doctor_windows(reporter: Reporter) -> list[str]:
    warnings: list[str] = []

    # 1. Battery: EcoQoS-class throttling is battery-driven, and Modern
    #    Standby truncates power requests 5 minutes after the sleep timeout
    #    on DC power — both documented MS behaviors (plan §1.2/§1.3).
    ac_line = _get_ac_line_status()
    if ac_line is None:
        reporter.note(
            "Статус питания не удалось определить (GetSystemPowerStatus) — "
            "проверка батареи пропущена."
        )
    elif ac_line == 0:
        warn = (
            "Питание от батареи: Windows может снижать частоту фоновых процессов "
            "(EcoQoS/Low-QoS) и обрывать keep-awake через 5 минут после сна-таймаута "
            "(Modern Standby). Подключите питание для гарантированной скорости."
        )
        reporter.warn(warn)
        warnings.append(warn)

    # 2. Active plan, compared by GUID only (localized names are never parsed).
    scheme = _active_scheme_guid()
    if scheme is None:
        reporter.note(
            "Активный план питания не удалось определить (powercfg /getactivescheme) — "
            "проверка плана пропущена."
        )
    elif scheme == GUID_POWER_SAVER:
        warn = (
            "Активен план «Экономия энергии» — Windows занижает частоты CPU. "
            f"Фикс: powercfg /setactive {GUID_HIGH_PERFORMANCE} (Высокая производительность)."
        )
        reporter.warn(warn)
        warnings.append(warn)
    elif scheme == GUID_BALANCED:
        # Not scolding: Balanced is Microsoft's recommended default on Server
        # (power-performance-tuning guide); one informational note with the
        # exact command for owners who want maximum reproducibility.
        reporter.note(
            "Активен сбалансированный план (рекомендация Microsoft для Windows Server). "
            f"Для гарантии производительности: powercfg /setactive {GUID_HIGH_PERFORMANCE}."
        )
    # GUID_HIGH_PERFORMANCE / GUID_ULTIMATE / custom GUIDs: silence.

    return warnings


def _doctor_linux(reporter: Reporter) -> list[str]:
    driver = _read_sysfs(_CPUFREQ_DRIVER_PATH)
    governor = _read_sysfs(_CPUFREQ_GOVERNOR_PATH)
    if driver is None or governor is None:
        return []  # guest/VM/ARM without cpufreq: silent skip (nothing to say)
    if governor == "powersave" and driver in LEGACY_CPUFREQ_DRIVERS:
        warn = (
            f"Governor powersave на legacy-драйвере {driver}: частоты CPU могут "
            "занижаться. Фикс (root): cpupower frequency-set -g performance."
        )
        reporter.warn(warn)
        return [warn]
    return []  # modern pstate drivers + powersave = default norm: silence


# --- RateWatch: runtime fps-sag detector -------------------------------------


class RateWatch:
    """
    Sliding-window fps-drop detector fed by the progress bar's own rate.

    Contract: ``update(fps)`` receives the SAME number the tqdm bar renders
    (``PipelineProgressBar.current_rate``) — the detector never re-measures.
    Windows of ``RATE_WINDOW_SECONDS`` close lazily on the next update; the
    first ``RATE_WARMUP_WINDOWS`` are discarded (engine warm-up), the baseline
    is the median of the next ``RATE_BASELINE_WINDOWS`` windows, and two
    consecutive windows below ``RATE_DROP_RATIO`` of the baseline produce
    exactly one loud Russian warning per stage. ``reset()`` starts a new
    stage (new baseline, warn guard cleared).

    Honest boundary (plan §P2): the detector does not diagnose WHY the rate
    sagged (throttling vs thermal vs foreground competition) — the warning
    says «похоже на», never claims a cause. Non-positive/None fps samples are
    skipped: no honest number, no window, no crash.
    """

    def __init__(
        self,
        reporter: Reporter,
        label: str = "",
        *,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        """Bind a reporter, an optional warn-label prefix, and the clock."""
        self._reporter = reporter
        self._label = label
        self._now = now
        self._reset_state()

    def _reset_state(self) -> None:
        self._window_start: float | None = None
        self._samples: list[float] = []
        self._completed: list[float] = []
        self._baseline: float | None = None
        self._low_streak = 0
        self._warned = False

    def reset(self) -> None:
        """Start a new stage: fresh baseline, warn guard cleared."""
        self._reset_state()

    def update(self, fps: float | None) -> None:
        """Feed one rate sample (None/non-positive = nothing honest, ignored)."""
        if fps is None or fps <= 0:
            return
        stamp = self._now()
        if self._window_start is None:
            self._window_start = stamp
        self._samples.append(fps)
        if stamp - self._window_start < RATE_WINDOW_SECONDS:
            return
        self._close_window(stamp)

    def _close_window(self, stamp: float) -> None:
        self._window_start = stamp
        if not self._samples:
            return  # an empty window (no samples) is skipped, not counted
        window_rate = statistics.fmean(self._samples)
        self._samples.clear()
        self._completed.append(window_rate)
        self._evaluate()

    def _evaluate(self) -> None:
        if len(self._completed) <= RATE_WARMUP_WINDOWS:
            return  # warm-up windows: observed but discarded
        if self._baseline is None:
            if len(self._completed) >= RATE_WARMUP_WINDOWS + RATE_BASELINE_WINDOWS:
                self._baseline = statistics.median(
                    self._completed[
                        RATE_WARMUP_WINDOWS : RATE_WARMUP_WINDOWS + RATE_BASELINE_WINDOWS
                    ]
                )
            return
        window_rate = self._completed[-1]
        if window_rate < RATE_DROP_RATIO * self._baseline:
            self._low_streak += 1
        else:
            self._low_streak = 0
        if self._low_streak >= RATE_LOW_STREAK and not self._warned:
            self._warned = True
            percent = round(window_rate / self._baseline * 100)
            message = RATE_DROP_WARN.format(pct=percent, base=self._baseline, current=window_rate)
            self._reporter.warn(f"{self._label} {message}" if self._label else message)
