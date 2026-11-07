<div align="center">

# ⚡ Guaranteed performance

**The pipeline defends its own throughput so you don't have to tune the OS: the system cannot sleep mid-batch, Windows power-throttling heuristics are excluded for the pipeline's processes, a read-only doctor names any hostile power setting together with the exact fix command, and an fps-drop detector calls throttling out in the act. Tier 1 is built in — nothing for you to do. Tier 2 lists optional system-wide one-liners for radical cases.**

![Windows](https://img.shields.io/badge/Windows-keep--awake_%2B_EcoQoS_opt--out-0078D4?style=flat-square&logo=windows&logoColor=white)
![Linux](https://img.shields.io/badge/Linux-systemd--inhibit-FCC624?style=flat-square&logo=linux&logoColor=white)
![NVIDIA](https://img.shields.io/badge/nvidia--smi_-lgc-Volta%2B-76B900?style=flat-square&logo=nvidia&logoColor=white)
![Doctor](https://img.shields.io/badge/power--doctor-read__only-green?style=flat-square&logo=powershell&logoColor=white)

</div>

**Status:** documentation for the performance-protection surface (design canon: the power-throttling plan, wave P4). Every external fact below was fetched **2026-11-07** at main `6807196` and cites its source page with its date; every quotation was re-verified verbatim against its live source page on 2026-11-07; nothing is asserted from memory. Tier 1 measures are part of the pipeline itself — they run automatically on every batch, and Windows-side behavior is **owner acceptance** (validated live on the Windows Server RTX 5080 deployment), not CI. The on-screen doctor/detector texts are Russian (the pipeline's TTY canon); this page explains the same machinery in English.

## 📑 Table of Contents

- [✅ Built-in guarantees (Tier 1)](#-built-in-guarantees-tier-1)
- [🧰 Optional system-wide measures (Tier 2)](#-optional-system-wide-measures-tier-2)
- [💤 How the OS can slow a busy batch down](#-how-the-os-can-slow-a-busy-batch-down)
- [📝 Honest notes](#-honest-notes)
- [🚫 What the pipeline deliberately does not do](#-what-the-pipeline-deliberately-does-not-do)
- [📚 Sources](#-sources)

---

## ✅ Built-in guarantees (Tier 1)

Nothing to configure and nothing to click: every batch run wraps itself in the guarantees below. Two opt-out flags exist for completeness — `--no-keep-awake` and `--no-power-check` — in the same CLI family as `--no-color`/`--no-tty`. You should never need them.

> **Windows live = owner acceptance.** The Windows-specific behavior (the keep-awake request visible in `powercfg /requests`, the EcoQoS opt-out, the doctor checks) is validated live on the Windows Server RTX 5080 deployment. Linux CI covers the logic through test seams — not the OS itself.

### Keep-awake for the whole batch

For the duration of a batch the pipeline calls `SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)` and clears it (`ES_CONTINUOUS`) in a `finally` block when the batch ends — including on Ctrl+C.

- Why this is needed at all: Windows decides "time to sleep" from **idle timers**, and *"Activities that are not automatically detected include disk or CPU activity and video display"* — a batch burning 100% CPU and a full GPU does **not** reset the system idle timer ([SetThreadExecutionState, Remarks](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setthreadexecutionstate), access 2026-11-07). The [system sleep criteria](https://learn.microsoft.com/en-us/windows/win32/power/system-sleep-criteria) page states the mechanics from the other side: *"if the system idle timer is greater than the system time-out value and no applications have requested the system by calling SetThreadExecutionState with ES_SYSTEM_REQUIRED, the system enters sleep"* (access 2026-11-07). Only an explicit execution-state request resets that timer: `ES_SYSTEM_REQUIRED` *"Forces the system to be in the working state by resetting the system idle timer"*.
- The current [system sleep criteria](https://learn.microsoft.com/en-us/windows/win32/power/system-sleep-criteria) guidance matches this design exactly: *"Avoid holding `ES_SYSTEM_REQUIRED | ES_CONTINUOUS` indefinitely"*, and *"Clear with `SetThreadExecutionState(ES_CONTINUOUS)` once the active operation completes"* (access 2026-11-07) — a batch is an active operation, and the pipeline clears in a `finally` block.
- The request deliberately does **not** include `ES_DISPLAY_REQUIRED`: the display may blank while you are away — only the system idle timer is the enemy.
- Limits the OS enforces (honesty): a keep-awake request *"cannot be used to prevent the user from putting the computer to sleep"* — you pressing Sleep, closing the lid, or the power button always wins. Multi-GPU fan-out workers hold their own request: execution state is per-thread/per-process, so every spawned worker calls in for itself.

### EcoQoS opt-out

At process start (main process and every fan-out worker) the pipeline takes control of the power-throttling feature via `SetProcessInformation` with `ProcessPowerThrottling`, `ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED`, `StateMask = 0` — the process manages its own execution-speed policy instead of inheriting one ([SetProcessInformation](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-setprocessinformation), access 2026-11-07). What the feature gates, in the page's own words: *"When a process opts into enabling `PROCESS_POWER_THROTTLING_EXECUTION_SPEED`, the process will be classified as EcoQoS"* — enabling the flag's on-state is the EcoQoS opt-in; taking control with the state off is the opt-out this pipeline makes.

- Without this call, Windows does not promise neutrality: *"If an application does not explicitly enable `PROCESS_POWER_THROTTLING_EXECUTION_SPEED`, the system will use its own heuristics to automatically infer a Quality of Service level"* ([SetProcessInformation](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-setprocessinformation), access 2026-11-07) — a windowless, minimized console batch can get classified into a throttled QoS class.
- The API failing (pre-1709 Windows, odd configurations) never blocks a run: one quiet note is printed, and the batch continues. A failed safety measure is announced, never silently skipped.

### Startup power doctor (read-only)

Before the queue starts, the doctor prints what it sees — and warns with **one exact fix command per finding**. It never changes a single setting: no power plan switches, no registry writes, no governor flips. Everything is read-only, every subprocess is time-boxed, and a failed check is skipped with a note — never fatal, never silent success.

**Windows checks** (`os.name == "nt"`, local runtime):

| Check | Warns when | Fix command the doctor prints |
| :-- | :-- | :-- |
| AC / battery | running on battery: EcoQoS/Low-QoS downclocking applies, and on Modern Standby laptops keep-awake requests are cut 5 minutes after the sleep timeout | plug the machine in |
| Sleep timeout of the active plan (`powercfg /query SCHEME_CURRENT SUB_SLEEP STANDBYIDLE`) | AC sleep timeout > 0 | `powercfg /change standby-timeout-ac 0` (keep-awake already covers you; this removes the dependency on the request being honored) |
| Active plan (`powercfg /getactivescheme`) | a Power Saver-class plan | `powercfg /setactive SCHEME_MIN` + a link to [Tier 2](#-optional-system-wide-measures-tier-2) |
| Active plan on **client** Windows | Balanced (info-level note, not a warning) | `powercfg /setactive SCHEME_MIN`, "only if you want maximum reproducibility" |
| Per-app GPU preference (`HKCU\...\UserGpuPreferences`) | `GpuPreference=1` ("Power saving") was manually assigned to the pipeline's executable | Settings → System → Display → Graphics → set *High performance* |
| Available sleep states (`powercfg /availablesleepstates`) | never warns — informational only | — |
| Keep-awake self-check | never warns — one note: verify the request is registered via `powercfg /requests` | `powercfg /requests` |

Plans are identified **by GUID, never by localized name** (a Russian "Экономия энергии" and an English "Power saver" must behave identically). Balanced on a **Windows Server** SKU produces silence — that is Microsoft's own recommendation (see [Honest notes](#-honest-notes)).

**Linux checks** (`sys.platform == "linux"`, local runtime):

| Check | Warns when | Fix command the doctor prints |
| :-- | :-- | :-- |
| cpufreq driver + governor (`/sys/devices/system/cpu/cpu0/cpufreq/scaling_driver`, `scaling_governor`) | a **legacy** driver (`acpi-cpufreq`-class) combined with `powersave` | `cpupower frequency-set -g performance` (root) |
| systemd-inhibit availability | the host runs systemd and the pipeline is not wrapped in an inhibitor | `systemd-inhibit --who=anime_upscale --why="batch" anime-pipeline …` (see [Tier 2](#-optional-system-wide-measures-tier-2)) |
| cgroup v2 CPU quota | a quota is active | nothing to fix — one note that the banner's `encode=N` reflects the container quota, not a defect (the budget math is already cgroup-aware) |
| Docker / container | logind absent inside the container | wrap on the **host**: `systemd-inhibit … docker compose run …` |

On `intel_pstate`/`amd-pstate` drivers, `powersave` is the everyday setting and does not mean slow (see [Tier 2](#-optional-system-wide-measures-tier-2)) — the doctor stays silent there to avoid false alarms on 99% of modern systems. Missing sysfs files (guest/ARM) are skipped quietly.

Kaggle/Colab runtimes skip the doctor entirely with a one-line note (cloud sessions have no host sleep to fight); keep-awake is a no-op there.

### RateWatch: the fps-drop detector

Throughput is measured from the progress stream the pipeline already produces (frames done per 5 s window). After a warm-up, the baseline is the **median** of the first windows; if the rate falls below **60% of baseline for two consecutive windows**, the run prints one loud warning per episode and keeps going:

> ⚠ Speed dropped to X% of the initial rate (was A fps → now B). Looks like throttling or the system entering an economy mode. Check: (1) did the system sleep, (2) power plan (`powercfg /getactivescheme`), (3) `nvidia-smi` — did GPU clocks drop. The run continues.

*English rendering of the on-screen Russian warning.*

Honest boundary: the detector signals *that* the rate fell, not *why* — throttling, thermals, background competition, and phase-dependent graph speed all look the same from outside, so the warning says "looks like", never a fake diagnosis. Baselines reset between episodes (different resolutions legitimately run at different speeds). Multi-GPU fan-out children each run their own detector on their own progress.

### Memory hygiene (landing in the memory-watchdog wave)

The same protection program includes a memory watchdog, per-episode VRAM hygiene, and an allocator cap. These land in the **memory-watchdog wave** and are documented here so the guarantees read as one picture; they are memory-bound, not power-bound, mechanisms.

---

## 🧰 Optional system-wide measures (Tier 2)

Everything below changes the **whole machine**, not just this pipeline — every other workload on the box lives with your choice. Tier 1 is designed to make these unnecessary; reach for them only if you want maximum aggressiveness (or you enjoy measuring).

### Windows

**High performance power plan** — processors are then *"always locked at the highest performance state (including "turbo" frequencies)"* and *"all cores are unparked"* ([Windows Server power performance tuning](https://learn.microsoft.com/en-us/windows-server/administration/performance-tuning/hardware/power/power-performance-tuning), access 2026-11-07):

```powershell
# System-global: affects the whole machine, not just the pipeline.
powercfg /setactive SCHEME_MIN          # alias of 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c
powercfg /aliases                        # the alias→GUID map on your machine

# Revert (Balanced — recommended default, see Honest notes):
powercfg /setactive SCHEME_BALANCED     # 381b4222-f694-41f0-9685-ff5bb260df2e
```

On **Windows Server**, Balanced is the Microsoft-recommended default — switching is a deliberate choice, not a fix (see [Honest notes](#-honest-notes)). The doctor does not scold you either way; it only flags Power Saver-class plans.

**Win32PrioritySeparation (registry — careful)** — the scheduler quantum/foreground-boost knob at `HKLM\SYSTEM\CurrentControlSet\Control\PriorityControl\Win32PrioritySeparation`:

```powershell
reg query "HKLM\SYSTEM\CurrentControlSet\Control\PriorityControl" /v Win32PrioritySeparation
```

We deliberately do **not** give a recommended value: the bit encoding of this value is **not documented in current official Microsoft documentation** (checked on learn.microsoft.com, 2026-11-07; the classical treatment lives in *Windows Internals*). The pipeline does not need it — a batch is one busy foreground-owned process, not a starved background app. If someone hands you a "magic value", you are changing scheduler behavior for the entire machine on community folklore; do it at your own risk and note the original value first.

**Lock GPU clocks: `nvidia-smi -lgc`** — pins SM clocks so the driver cannot downclock between the pipeline's CPU and GPU phases:

```powershell
# Elevated prompt. Volta and newer (the RTX 5080 qualifies).
nvidia-smi -lgc 1500,1500     # <min>,<max> in MHz; pick your card's gaming clock
nvidia-smi -rgc               # RESET when done — the lock does not expire by itself
```

Facts from the [official nvidia-smi documentation](https://docs.nvidia.com/deploy/nvidia-smi/) (access 2026-11-07): the pair *"defines closest desired locked GPU clock speed in MHz"*, the section ends *"Supported on Volta+. Requires root"* (on Windows: administrator), and `-rgc, --reset-gpu-clocks` *"Resets the GPU clocks to the default value"* — the lock does not expire by itself. Because the locked pair pins the clock range rather than hinting at it, the driver cannot downclock below it at idle while the lock stands — that is what protects the pipeline's GPU phases, and also why idle **power draw and heat** stay elevated: reset it after the batch. One honest caveat, quoted with the documentation's own typo: *"it is possible for clocks to report a lower freqency [sic] than the lowest frequency that can be set by SW due to HW optimizations in certain scenarios"* — it is a clamp, not a physical guarantee.

### Linux

**`systemd-inhibit` wrapper** — takes a shutdown/sleep/idle inhibitor lock for exactly the lifetime of the command (the default is `--what=idle:sleep:shutdown`, `--mode=block`, which *"prohibits any of the requested operations without time limit"*); the wrapper's exit status is the pipeline's exit status ([systemd-inhibit(1)](https://www.freedesktop.org/software/systemd/man/latest/systemd-inhibit.html), access 2026-11-07):

```bash
systemd-inhibit --who=anime_upscale --why="batch processing" \
  anime-pipeline --profile full -i episode_01.mkv -i episode_02.mkv

# In Docker the sleep decision belongs to the HOST — wrap on the host, not in the container:
systemd-inhibit --who=anime_upscale --why="batch processing" \
  docker compose run --rm anime-pipeline --profile interp-only --upload
```

Inside a container `logind` is absent, so this is a host-level measure by definition — the pipeline says so instead of pretending it can help from inside.

**cpufreq performance governor** — relevant only on legacy drivers:

```bash
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_driver      # intel_pstate / amd-pstate / acpi-cpufreq …
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor

sudo cpupower frequency-set -g performance    # only meaningful for legacy (acpi-cpufreq-class) drivers
```

On modern `intel_pstate`/`amd-pstate` drivers, a `powersave` reading does not mean slow hardware management: in active mode the `powersave` P-state algorithm of `intel_pstate` *"is not a counterpart of the generic `powersave` governor (roughly, it corresponds to the `schedutil` and `ondemand` governors)"* ([kernel intel_pstate documentation](https://docs.kernel.org/admin-guide/pm/intel_pstate.html), access 2026-11-07). Switching the governor there buys nothing; the doctor stays silent precisely to avoid that false alarm.

### Summary

| Measure | Scope | Revert |
| :-- | :-- | :-- |
| High performance plan | whole machine, until switched back | `powercfg /setactive SCHEME_BALANCED` |
| Win32PrioritySeparation | whole machine scheduler, persists across reboots | restore the value you noted before the edit |
| `nvidia-smi -lgc` | whole machine GPU, persists until reset | `nvidia-smi -rgc` |
| `systemd-inhibit` | exactly the wrapped command | nothing to revert |
| cpufreq governor | whole machine, persists across reboots | `cpupower frequency-set -g <previous>` |

---

## 💤 How the OS can slow a busy batch down

The mechanisms below are why Tier 1 exists. Each fact cites its source page and access date; where a threshold is **not** publicly documented, we say so instead of guessing.

### The system idle timer ignores CPU activity

The power plan's system idle timer counts *input and display* signals: *"Activities that are not automatically detected include disk or CPU activity and video display"* ([SetThreadExecutionState, Remarks](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setthreadexecutionstate), access 2026-11-07), and the [system sleep criteria](https://learn.microsoft.com/en-us/windows/win32/power/system-sleep-criteria) page supplies the mechanics: *"if the system idle timer is greater than the system time-out value and no applications have requested the system by calling SetThreadExecutionState with ES_SYSTEM_REQUIRED, the system enters sleep"* (access 2026-11-07). A console batch that left the mouse untouched is indistinguishable from an empty room — the timer expires and the system sleeps, mid-render. No busy-loop can fix this; only an execution-state request can ([SetThreadExecutionState](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setthreadexecutionstate), access 2026-11-07). This is the mechanism behind "I walked away from the server, and progress went sideways".

### QoS classification and EcoQoS

Windows classifies every process into a Quality-of-Service class ([Quality of Service](https://learn.microsoft.com/en-us/windows/win32/procthread/quality-of-service), access 2026-11-07). Without an explicit opt-out, the class comes from *"its own heuristics"* — and one documented heuristic is aimed exactly at unattended batches: *"By default, Windows may lower the QoS policy of a foreground application to Medium QoS after a period of user inactivity where no input is detected"*, and Microsoft's own testing note adds: *"Automated tests lacking user input may trigger this feature"*. On battery, the lower classes mean the most power-efficient cores and frequencies. On AC power this user-inactivity feature does not fire — which is why the doctor checks AC/battery, and why the EcoQoS opt-out (one API call) removes the whole class of doubt. The feature does have an official off-switch (`HKLM\SYSTEM\CurrentControlSet\Control\Power\PowerThrottling\DisableUserPresenceQos = 1`), but it is a global registry edit that only ever matters on battery — we do not recommend touching it.

### GPU P-states: downclocking between phases

NVIDIA drivers drop GPU clocks when nothing is running (dynamic power management), and raise them under load. The exact idle-downclock thresholds and delays are **not documented publicly** — the driver's internal policy, subject to change between versions; we mark this as a model, not a fact. The pipeline alternates CPU phases (decode, VapourSynth filters, and the encode — libx265 runs on the CPU by design) with GPU phases (upscale, RIFE), so GPU utilization legitimately breathes 0↔100% as phases change. When a slowdown is *not* legitimate, the driver may have parked clocks during a long CPU phase and been slow to ramp — `nvidia-smi -lgc` ([Tier 2](#-optional-system-wide-measures-tier-2)) is the documented clamp, and RateWatch is the honest measurement of what throughput actually did.

### Laptops: Modern Standby cuts keep-awake short

On Modern Standby systems **on battery (DC) power**, system and execution power requests are *"terminated 5 minutes after the system sleep timeout has expired"* ([PowerSetRequest](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-powersetrequest), page updated 2025-07-01, access 2026-11-07) — and user-initiated sleep (power button, lid, Sleep menu) terminates them immediately. On a wall-powered desktop — and on Windows Server, which uses the Traditional Sleep model — no such 5-minute cut applies. This is why the doctor warns rather than lies: on a battery laptop, plug in for a guaranteed run.

---

## 📝 Honest notes

- **Balanced on Windows Server is a valid choice.** Microsoft's own tuning guidance names *"Balanced (recommended)"* as the default that *"targets good energy efficiency with minimal performance impact"*, adds that *"during the steady state, throughput does not change with varying utilizations until the system is overloaded"* ([processor power management tuning](https://learn.microsoft.com/en-us/windows-server/administration/performance-tuning/hardware/power/processor-power-management-tuning), access 2026-11-07), and the Server troubleshooting guide recommends to *"leave the power plan set to Balanced and evaluate the proper settings within the Balanced power plan"* ([power performance tuning](https://learn.microsoft.com/en-us/windows-server/administration/performance-tuning/hardware/power/power-performance-tuning), [troubleshooting](https://learn.microsoft.com/en-us/troubleshoot/windows-server/performance/slow-performance-when-using-power-plan), access 2026-11-07). The doctor therefore stays **silent** on Server + Balanced — it informs, it never scolds. On client Windows, Balanced earns only an info-level note with the optional switch.
- **Task Manager does not show Compute engines by default — CPU% spikes are usually pipeline phases.** Task Manager samples GPU engines (3D, Copy, Video Encode/Decode, Compute…), and *"by default, the Task Manager will pick 4 engines to be displayed"*; as a Microsoft engineer put it: *"Since the Compute engines are not displayed by default, they do not contribute to the reported utilization"* ([DirectX developer blog, "GPUs in the Task Manager", 2017-07-21](https://devblogs.microsoft.com/directx/gpus-in-the-task-manager/); [tensorflow-directml#134, jstoecker, 2021-01-04](https://github.com/microsoft/tensorflow-directml/issues/134)). A CUDA batch can read 0–1% there while running at full tilt. Combine that with the phase alternation above (CPU-heavy x265 encode → GPU idles → clocks drop → utilization reads 0%), and "spikes 0–100%" is mostly **structure plus a measurement artifact**, not necessarily misbehavior. True telemetry = the pipeline's own frames/s (the progress bar) and `nvidia-smi` — not Task Manager's aggregate column.
- **The doctor informs, it never mutates.** Every check is read-only; every warning carries exactly one fix command for *you* to run (or not). The pipeline never switches plans, writes the registry, locks clocks, or flips governors on its own.

---

## 🚫 What the pipeline deliberately does not do

- **Timer-resolution games (`timeBeginPeriod`).** The hot path is compute-bound (VS output loop, GPU kernels, x265 thread pools, the ffmpeg pipe) — not `Sleep()`-precision loops; timer granularity affects wait timeouts, not batch throughput ([timeBeginPeriod](https://learn.microsoft.com/en-us/windows/win32/api/timeapi/nf-timeapi-timebeginperiod), access 2026-11-07). Documented here so nobody "optimizes" it in later.
- **Process priority manipulation, `Win32PrioritySeparation` tweaks.** One busy batch process has no priority race to win; the registry value's encoding is not covered by official Microsoft documentation (see [Tier 2](#-optional-system-wide-measures-tier-2)), so we neither set it nor recommend values.
- **Game Mode.** A console batch is not a game: Game Mode grants resource priority to the foreground-focused app ([Game Mode portal](https://learn.microsoft.com/en-us/previous-versions/windows/desktop/gamemode/game-mode-portal), access 2026-11-07). The only real-world interplay: if *you* play a game while the batch runs in the background, the game may legitimately take priority — plan heavy batches for when you are not gaming.
- **Automatic system changes of any kind.** No power-plan switches from code, no GPU clock locks from code, no config keys for OS settings. The pipeline protects itself within its own process; everything system-wide remains an explicit, documented, human decision.

---

## 📚 Sources

All pages fetched 2026-11-07; each entry states what it establishes for this document.

| Source | Date | Establishes |
| :-- | :-- | :-- |
| [System sleep criteria](https://learn.microsoft.com/en-us/windows/win32/power/system-sleep-criteria) | access 2026-11-07 | Sleep-entry mechanics (idle timer exceeds timeout + no `ES_SYSTEM_REQUIRED` request = sleep); current `SetThreadExecutionState` usage guidance |
| [SetThreadExecutionState](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setthreadexecutionstate) | access 2026-11-07 | `ES_CONTINUOUS`/`ES_SYSTEM_REQUIRED` semantics; *"Activities that are not automatically detected include disk or CPU activity and video display"* (Remarks); user sleep always wins |
| [SetProcessInformation](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-setprocessinformation) | access 2026-11-07 | power-throttling opt-out/opt-in (`ProcessPowerThrottling`, control/state masks); default = OS heuristics |
| [Quality of Service](https://learn.microsoft.com/en-us/windows/win32/procthread/quality-of-service) | access 2026-11-07 | QoS class table (1709 → Windows 11); user-inactivity downgrade on battery; benchmark warning |
| [PowerSetRequest](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-powersetrequest) | updated 2025-07-01 | Modern Standby + DC: requests terminated 5 min after sleep timeout; user-initiated sleep wins |
| [Modern Standby](https://learn.microsoft.com/en-us/windows-hardware/design/device-experiences/modern-standby) | access 2026-11-07 | S0-idle model; idle-out enters Modern Standby; client-SKU feature (Server uses Traditional Sleep) |
| [timeBeginPeriod](https://learn.microsoft.com/en-us/windows/win32/api/timeapi/nf-timeapi-timebeginperiod) | access 2026-11-07 | timer resolution per-process since Windows 10 2004; Win11 occlusion rule; wait-accuracy scope |
| [Windows Server power performance tuning](https://learn.microsoft.com/en-us/windows-server/administration/performance-tuning/hardware/power/power-performance-tuning) | access 2026-11-07 | Balanced (recommended) semantics; High performance = highest state, all cores unparked; steady-state throughput |
| [Slow performance when using a power plan (troubleshooting)](https://learn.microsoft.com/en-us/troubleshoot/windows-server/performance/slow-performance-when-using-power-plan) | access 2026-11-07 | "leave the plan set to Balanced and tune within it" guidance |
| [Processor power management options (PPM)](https://learn.microsoft.com/en-us/windows-hardware/customize/power-settings/configure-processor-power-management-options) | access 2026-11-07 | PPM parts: core parking engine + performance-state engine |
| [Processor power management tuning](https://learn.microsoft.com/en-us/windows-server/administration/performance-tuning/hardware/power/processor-power-management-tuning) | access 2026-11-07 | *"During the steady state, throughput does not change with varying utilizations until the system is overloaded"* |
| [powercfg command-line options](https://learn.microsoft.com/en-us/windows-hardware/design/device-experiences/powercfg-command-line-options) | access 2026-11-07 | `/getactivescheme`, `/query`, `/requests`, `/availablesleepstates`, `/aliases`; Balanced GUID in official examples |
| [nvidia-smi documentation](https://docs.nvidia.com/deploy/nvidia-smi/) | access 2026-11-07 | `-lgc` *"defines closest desired locked GPU clock speed in MHz"*, *"Supported on Volta+. Requires root"*; `-rgc` reset; clocks-may-report-lower caveat (quoted verbatim, incl. the page's own typo) |
| [systemd-inhibit(1)](https://www.freedesktop.org/software/systemd/man/latest/systemd-inhibit.html) | access 2026-11-07 | inhibitor locks; default `--what=idle:sleep:shutdown`, `--mode=block`; exit-status passthrough |
| [logind.conf(5)](https://www.freedesktop.org/software/systemd/man/latest/logind.conf.html) | access 2026-11-07 | `IdleAction`/`IdleActionSec` — the idle automation the `idle` inhibitor blocks |
| [kernel cpufreq documentation](https://docs.kernel.org/admin-guide/pm/cpufreq.html) | © 2017 Intel, access 2026-11-07 | CPUFreq governors; sysfs `scaling_governor`/`scaling_driver` ABI |
| [kernel intel_pstate documentation](https://docs.kernel.org/admin-guide/pm/intel_pstate.html) | access 2026-11-07 | active-mode `powersave` *"is not a counterpart of the generic `powersave` governor"* — not "slow" |
| [GPUs in the Task Manager (DirectX dev blog)](https://devblogs.microsoft.com/directx/gpus-in-the-task-manager/) | 2017-07-21 | Task Manager samples WDDM engines; default 4-engine selection |
| [tensorflow-directml#134 (Microsoft, jstoecker)](https://github.com/microsoft/tensorflow-directml/issues/134) | comment 2021-01-04 | *"Compute engines are not displayed by default"* — CUDA batches under-reported in Task Manager |
| [cgroup v2 kernel documentation](https://docs.kernel.org/admin-guide/cgroup-v2.html) | access 2026-11-07 | `cpu.max` quota semantics (the pipeline's CPU budget is already cgroup-aware) |

Design canon for this page: the power-throttling plan (wave P4), summarized architecturally in [architecture-plan.md](architecture-plan.md) §14.
