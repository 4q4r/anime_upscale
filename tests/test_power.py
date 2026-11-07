"""P1 anti-throttling wave: keep-awake, EcoQoS opt-out, startup doctor, RateWatch.

Env-emulation matrix (plan-power-throttling.md §3.4 DoD): every Windows surface
runs through a module seam (``_set_thread_execution_state``,
``_set_process_information_optout``, ``_run_powercfg``, ``_get_ac_line_status``,
``_read_sysfs``) so the whole suite is green on Linux without Windows. RateWatch
uses an injected monotonic clock — no real sleeping anywhere.
"""

import sys
from pathlib import Path

import pytest

from anime_upscale import power
from anime_upscale.reporting import Reporter, resolve_mode
from anime_upscale.runtime_env import RuntimeProfile

ES_CONTINUOUS = power.ES_CONTINUOUS
ES_SYSTEM_REQUIRED = power.ES_SYSTEM_REQUIRED
SET_FLAGS = ES_CONTINUOUS | ES_SYSTEM_REQUIRED
RESET_FLAGS = ES_CONTINUOUS

GUID_BALANCED = "381b4222-f694-41f0-9685-ff5bb260df2e"
GUID_HIGH = "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"
GUID_SAVER = "a1841308-3541-4fab-bc81-f71556f20b4a"
GUID_ULTIMATE = "e9a42b02-d5df-448d-aa00-03f14749eb61"

POWERCFG_BALANCED = f"Power Scheme GUID: {GUID_BALANCED.upper()}  (Balanced) *"
POWERCFG_HIGH = f"Power Scheme GUID: {GUID_HIGH.upper()}  (High performance)"
POWERCFG_SAVER_RU = f"GUID схемы питания: {GUID_SAVER.upper()}  (Экономия энергии) *"
POWERCFG_ULTIMATE = f"Power Scheme GUID: {GUID_ULTIMATE.upper()}  (Ultimate Performance)"
POWERCFG_GARBAGE = "внезапно сломанный вывод без GUID"

WINDOW = power.RATE_WINDOW_SECONDS


class _CapReporter(Reporter):
    """Reporter double collecting warn/note streams separately."""

    def __init__(self) -> None:
        super().__init__(resolve_mode(no_color=True, no_tty=True))
        self.warns: list[str] = []
        self.notes: list[str] = []

    def warn(self, text: str) -> None:
        self.warns.append(text)

    def note(self, text: str) -> None:
        self.notes.append(text)


@pytest.fixture(autouse=True)
def _fresh_process_guards() -> None:
    """Once-per-process caches must not leak between tests."""
    power._reset_process_guards()
    yield
    power._reset_process_guards()


class _FakeStateSetter:
    """Recording SetThreadExecutionState double with a fail-list."""

    def __init__(self, fail_on: frozenset[int] = frozenset()) -> None:
        self.calls: list[int] = []
        self.fail_on = fail_on

    def __call__(self, flags: int) -> int:
        self.calls.append(flags)
        if flags in self.fail_on:
            msg = "fake SetThreadExecutionState failure"
            raise OSError(msg)
        return flags


class _Clock:
    """Deterministic monotonic clock for RateWatch windows."""

    def __init__(self, start: float = 0.0) -> None:
        self.value = start

    def advance(self, dt: float) -> None:
        self.value += dt

    def __call__(self) -> float:
        return self.value


def _feed_window(watch: power.RateWatch, clock: _Clock, fps: float) -> None:
    """Feed one logical window: a sample, the window boundary, the next sample."""
    watch.update(fps)
    clock.advance(WINDOW)
    watch.update(fps)


def _patch_windows(monkeypatch: pytest.MonkeyPatch, setter: _FakeStateSetter) -> None:
    monkeypatch.setattr(power, "_is_windows", lambda: True)
    monkeypatch.setattr(power, "_set_thread_execution_state", setter)


# --- keep_system_awake -------------------------------------------------------


class TestKeepSystemAwake:
    def test_enter_sets_and_exit_resets(self, monkeypatch: pytest.MonkeyPatch) -> None:
        setter = _FakeStateSetter()
        _patch_windows(monkeypatch, setter)
        reporter = _CapReporter()
        with power.keep_system_awake(reporter) as status:
            assert status.active is True
            assert setter.calls == [SET_FLAGS]
        assert setter.calls == [SET_FLAGS, RESET_FLAGS]
        assert reporter.warns == []

    def test_body_exception_still_resets(self, monkeypatch: pytest.MonkeyPatch) -> None:
        setter = _FakeStateSetter()
        _patch_windows(monkeypatch, setter)

        def boom() -> None:
            msg = "boom"
            raise RuntimeError(msg)

        with pytest.raises(RuntimeError, match="boom"), power.keep_system_awake(_CapReporter()):
            boom()
        assert setter.calls == [SET_FLAGS, RESET_FLAGS]

    def test_keyboardinterrupt_resets_then_propagates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        setter = _FakeStateSetter()
        _patch_windows(monkeypatch, setter)
        with pytest.raises(KeyboardInterrupt), power.keep_system_awake(_CapReporter()):
            raise KeyboardInterrupt
        assert setter.calls == [SET_FLAGS, RESET_FLAGS]

    def test_entry_failure_warns_once_body_still_runs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        setter = _FakeStateSetter(fail_on=frozenset({SET_FLAGS}))
        _patch_windows(monkeypatch, setter)
        reporter = _CapReporter()
        with power.keep_system_awake(reporter) as status:
            assert status.active is False
            assert reporter.warns != []  # loud, not silent
            assert len(reporter.warns) == 1
        assert setter.calls == [SET_FLAGS]  # reset must not be attempted

    def test_linux_is_noop_with_systemd_inhibit_note(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(power, "_is_windows", lambda: False)
        setter = _FakeStateSetter()
        monkeypatch.setattr(power, "_set_thread_execution_state", setter)
        reporter = _CapReporter()
        with power.keep_system_awake(reporter) as status:
            assert status.active is False
        assert setter.calls == []
        assert len(reporter.notes) == 1
        assert "systemd-inhibit" in reporter.notes[0]

    def test_cloud_runtime_is_silent_noop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_runtime() -> RuntimeProfile:
            return RuntimeProfile(kind="kaggle", persistent_root="/kaggle/working", session_note="")

        monkeypatch.setattr(power, "detect_runtime", fake_runtime)
        setter = _FakeStateSetter()
        monkeypatch.setattr(power, "_set_thread_execution_state", setter)
        reporter = _CapReporter()
        with power.keep_system_awake(reporter) as status:
            assert status.active is False
        assert setter.calls == []
        assert reporter.notes == [] and reporter.warns == []


# --- disable_power_throttling ------------------------------------------------


class TestDisablePowerThrottling:
    def test_not_windows_returns_false_without_calls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(power, "_is_windows", lambda: False)

        def boom() -> tuple[bool, str]:
            raise AssertionError("EcoQoS seam called on non-Windows")

        monkeypatch.setattr(power, "_set_process_information_optout", boom)
        reporter = _CapReporter()
        assert power.disable_power_throttling(reporter) is False
        assert reporter.notes == [] and reporter.warns == []

    def test_ok_true_and_second_call_is_cached_noop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(power, "_is_windows", lambda: True)
        calls: list[int] = []

        def seam() -> tuple[bool, str]:
            calls.append(1)
            return True, ""

        monkeypatch.setattr(power, "_set_process_information_optout", seam)
        reporter = _CapReporter()
        assert power.disable_power_throttling(reporter) is True
        assert power.disable_power_throttling(reporter) is True
        assert calls == [1]  # exactly one attempt per process
        assert reporter.notes == [] and reporter.warns == []

    def test_failure_false_one_note_then_silent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(power, "_is_windows", lambda: True)

        def seam() -> tuple[bool, str]:
            return False, "SetProcessInformation вернула 0"

        monkeypatch.setattr(power, "_set_process_information_optout", seam)
        reporter = _CapReporter()
        assert power.disable_power_throttling(reporter) is False
        assert power.disable_power_throttling(reporter) is False
        assert len(reporter.notes) == 1
        assert "EcoQoS" in reporter.notes[0]
        assert "батарее" in reporter.notes[0]


# --- run_startup_doctor: Windows ---------------------------------------------


def _patch_doctor_windows(
    monkeypatch: pytest.MonkeyPatch,
    *,
    powercfg: tuple[int, str],
    ac_line_status: int | None = 1,
) -> None:
    monkeypatch.setattr(power, "_is_windows", lambda: True)
    monkeypatch.setattr(power, "_run_powercfg", lambda _args: powercfg)
    monkeypatch.setattr(power, "_get_ac_line_status", lambda: ac_line_status)


class TestDoctorWindows:
    def test_balanced_gets_note_with_fix_command(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_doctor_windows(monkeypatch, powercfg=(0, POWERCFG_BALANCED))
        reporter = _CapReporter()
        warns = power.run_startup_doctor(reporter)
        assert warns == []  # Balanced is a note, never a warn
        assert len(reporter.notes) == 1
        assert "сбалансированный" in reporter.notes[0]
        assert f"powercfg /setactive {GUID_HIGH}" in reporter.notes[0]

    def test_high_performance_is_silent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_doctor_windows(monkeypatch, powercfg=(0, POWERCFG_HIGH))
        reporter = _CapReporter()
        assert power.run_startup_doctor(reporter) == []
        assert reporter.warns == [] and reporter.notes == []

    def test_ultimate_is_silent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_doctor_windows(monkeypatch, powercfg=(0, POWERCFG_ULTIMATE))
        reporter = _CapReporter()
        assert power.run_startup_doctor(reporter) == []
        assert reporter.warns == [] and reporter.notes == []

    def test_power_saver_warns_with_fix_command(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Uppercase + localized label: comparison must be GUID-only, never names.
        _patch_doctor_windows(monkeypatch, powercfg=(0, POWERCFG_SAVER_RU))
        reporter = _CapReporter()
        warns = power.run_startup_doctor(reporter)
        assert len(warns) == 1
        assert reporter.warns == warns
        assert f"powercfg /setactive {GUID_HIGH}" in warns[0]

    def test_malformed_powercfg_output_is_a_note(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_doctor_windows(monkeypatch, powercfg=(0, POWERCFG_GARBAGE))
        reporter = _CapReporter()
        assert power.run_startup_doctor(reporter) == []
        assert len(reporter.notes) == 1
        assert "powercfg /getactivescheme" in reporter.notes[0]

    def test_powercfg_failure_is_a_note(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_doctor_windows(monkeypatch, powercfg=(1, ""))
        reporter = _CapReporter()
        assert power.run_startup_doctor(reporter) == []
        assert len(reporter.notes) == 1

    def test_battery_warns_about_throttling_and_expense(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_doctor_windows(monkeypatch, powercfg=(0, POWERCFG_HIGH), ac_line_status=0)
        reporter = _CapReporter()
        warns = power.run_startup_doctor(reporter)
        assert len(warns) == 1
        assert "батаре" in warns[0]
        assert "Питание" in warns[0] or "питание" in warns[0]

    def test_ac_power_is_silent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_doctor_windows(monkeypatch, powercfg=(0, POWERCFG_HIGH), ac_line_status=1)
        reporter = _CapReporter()
        assert power.run_startup_doctor(reporter) == []
        assert reporter.warns == [] and reporter.notes == []

    def test_power_status_probe_failure_is_a_note(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_doctor_windows(monkeypatch, powercfg=(0, POWERCFG_HIGH), ac_line_status=None)
        reporter = _CapReporter()
        assert power.run_startup_doctor(reporter) == []
        assert len(reporter.notes) == 1

    def test_cloud_runtime_skips_all_probes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            power,
            "detect_runtime",
            lambda: RuntimeProfile(kind="colab", persistent_root=None, session_note=""),
        )

        def boom(_args: list[str]) -> tuple[int, str]:
            raise AssertionError("powercfg called on a cloud runtime")

        monkeypatch.setattr(power, "_run_powercfg", boom)
        reporter = _CapReporter()
        assert power.run_startup_doctor(reporter) == []
        assert len(reporter.notes) == 1


# --- run_startup_doctor: Linux -----------------------------------------------


def _patch_doctor_linux(monkeypatch: pytest.MonkeyPatch, files: dict[str, str]) -> None:
    monkeypatch.setattr(power, "_is_windows", lambda: False)
    monkeypatch.setattr(power.sys, "platform", "linux")  # pin intent off-Linux too
    monkeypatch.setattr(power, "_read_sysfs", files.get)


GOV = power._CPUFREQ_GOVERNOR_PATH
DRV = power._CPUFREQ_DRIVER_PATH


class TestDoctorLinux:
    def test_legacy_driver_powersave_warns_with_cpupower_fix(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_doctor_linux(monkeypatch, {GOV: "powersave", DRV: "acpi-cpufreq"})
        reporter = _CapReporter()
        warns = power.run_startup_doctor(reporter)
        assert len(warns) == 1
        assert "cpupower frequency-set -g performance" in warns[0]
        assert "acpi-cpufreq" in warns[0]

    def test_intel_cpufreq_passive_powersave_warns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_doctor_linux(monkeypatch, {GOV: "powersave", DRV: "intel_cpufreq"})
        reporter = _CapReporter()
        assert len(power.run_startup_doctor(reporter)) == 1

    def test_modern_pstate_powersave_is_silent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for driver in ("intel_pstate", "amd-pstate"):
            _patch_doctor_linux(monkeypatch, {GOV: "powersave", DRV: driver})
            reporter = _CapReporter()
            assert power.run_startup_doctor(reporter) == []
            assert reporter.warns == [] and reporter.notes == []

    def test_missing_sysfs_files_are_silent_skip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_doctor_linux(monkeypatch, {})
        reporter = _CapReporter()
        assert power.run_startup_doctor(reporter) == []
        assert reporter.warns == [] and reporter.notes == []

    def test_performance_governor_is_silent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_doctor_linux(monkeypatch, {GOV: "performance", DRV: "acpi-cpufreq"})
        reporter = _CapReporter()
        assert power.run_startup_doctor(reporter) == []
        assert reporter.warns == []


# --- RateWatch ---------------------------------------------------------------


class TestRateWatch:
    def test_constant_rate_never_warns(self) -> None:
        clock = _Clock()
        reporter = _CapReporter()
        watch = power.RateWatch(reporter, label="[ep.mkv]", now=clock)
        for _ in range(12):
            _feed_window(watch, clock, 100.0)
        assert reporter.warns == []

    def test_baseline_is_median_of_windows_3_to_6(self) -> None:
        clock = _Clock()
        reporter = _CapReporter()
        watch = power.RateWatch(reporter, now=clock)
        for fps in (90.0, 110.0):  # warm-up windows 1-2: discarded
            _feed_window(watch, clock, fps)
        for fps in (100.0, 90.0, 110.0, 100.0):  # windows 3-6: baseline pool
            _feed_window(watch, clock, fps)
        assert watch._baseline == pytest.approx(100.0)

    def test_two_consecutive_low_windows_warn_exactly_once(self) -> None:
        clock = _Clock()
        reporter = _CapReporter()
        watch = power.RateWatch(reporter, label="[ep.mkv]", now=clock)
        for _ in range(6):
            _feed_window(watch, clock, 100.0)
        _feed_window(watch, clock, 50.0)  # window 7: low streak = 1, silent
        assert reporter.warns == []
        _feed_window(watch, clock, 50.0)  # window 8: second low → ONE warn
        assert len(reporter.warns) == 1
        assert "[ep.mkv]" in reporter.warns[0]
        assert "50%" in reporter.warns[0]
        assert "100.0" in reporter.warns[0]
        _feed_window(watch, clock, 50.0)
        _feed_window(watch, clock, 50.0)
        assert len(reporter.warns) == 1  # once-per-stage guard

    def test_single_low_window_then_recovery_warns_nothing(self) -> None:
        clock = _Clock()
        reporter = _CapReporter()
        watch = power.RateWatch(reporter, now=clock)
        for _ in range(6):
            _feed_window(watch, clock, 100.0)
        _feed_window(watch, clock, 50.0)
        _feed_window(watch, clock, 100.0)  # recovery resets the streak
        _feed_window(watch, clock, 100.0)
        assert reporter.warns == []

    def test_reset_lets_a_new_stage_warn_again(self) -> None:
        clock = _Clock()
        reporter = _CapReporter()
        watch = power.RateWatch(reporter, now=clock)
        for _ in range(6):
            _feed_window(watch, clock, 100.0)
        _feed_window(watch, clock, 50.0)
        _feed_window(watch, clock, 50.0)
        assert len(reporter.warns) == 1
        watch.reset()
        clock2 = _Clock(start=clock.value)
        watch2_state = power.RateWatch(reporter, now=clock2)
        for _ in range(6):
            _feed_window(watch2_state, clock2, 80.0)
        _feed_window(watch2_state, clock2, 40.0)
        _feed_window(watch2_state, clock2, 40.0)
        assert len(reporter.warns) == 2

    def test_none_zero_negative_fps_never_crash_or_roll(self) -> None:
        clock = _Clock()
        reporter = _CapReporter()
        watch = power.RateWatch(reporter, now=clock)
        watch.update(None)
        watch.update(0.0)
        watch.update(-1.0)
        clock.advance(WINDOW * 3)
        watch.update(None)
        assert watch._completed == []
        assert watch._samples == []
        assert reporter.warns == []

    def test_time_gaps_do_not_fabricate_windows(self) -> None:
        """A silence (stall) absorbs into the pending window: the next sample
        closes exactly ONE window, no phantom empty entries appear."""
        clock = _Clock()
        reporter = _CapReporter()
        watch = power.RateWatch(reporter, now=clock)
        for _ in range(6):
            _feed_window(watch, clock, 100.0)
        assert len(watch._completed) == 6
        clock.advance(WINDOW * 6)  # long silence — no samples at all
        watch.update(50.0)
        assert len(watch._completed) == 7
        assert watch._completed[-1] == pytest.approx(50.0)
        assert reporter.warns == []  # one low window is a streak of 1: silent

    def test_thresholds_are_module_constants_not_config(self) -> None:
        """DoD: thresholds live as documented module constants (no config keys)."""
        assert power.RATE_WINDOW_SECONDS == 5.0
        assert power.RATE_WARMUP_WINDOWS == 2
        assert power.RATE_BASELINE_WINDOWS == 4
        assert power.RATE_DROP_RATIO == 0.6
        assert power.RATE_LOW_STREAK == 2


# --- module hygiene ----------------------------------------------------------


class TestModuleHygiene:
    def test_power_module_never_imports_gpu_stacks(self) -> None:
        """P1 DoD: power.py stays importable before any VS/torch stack
        (fresh-subprocess probe: the pytest session itself probes torch)."""
        import subprocess

        code = (
            "import sys; import anime_upscale.power; "
            "assert 'torch' not in sys.modules, 'torch loaded'; "
            "assert 'vapoursynth' not in sys.modules, 'vapoursynth loaded'"
        )
        result = subprocess.run(  # noqa: S603 - fixed argv: venv python + inline assert code
            [sys.executable, "-c", code], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr
        source = Path(power.__file__).read_text(encoding="utf-8")
        assert "import torch" not in source
        assert "import vapoursynth" not in source
        assert "from anime_upscale.pipeline" not in source
        assert "from anime_upscale.stages" not in source
