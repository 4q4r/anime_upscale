"""CLI tests: parsing, flags, profile-not-found, input/ single-file rule (plan §3)."""

from pathlib import Path

import pytest

from anime_upscale.cli import EntryError, _default_inputs, build_parser
from anime_upscale.config import ConfigError, resolve_run
from anime_upscale.tools import ToolError, resolve_tool


class TestParser:
    def test_repeatable_inputs(self) -> None:
        args = build_parser().parse_args(
            [
                "--profile",
                "full",
                "-i",
                "a.mkv",
                "-i",
                "https://x/b.mkv",
                "-i",
                "https://gofile.io/d/zz",
            ]
        )
        assert args.inputs == ["a.mkv", "https://x/b.mkv", "https://gofile.io/d/zz"]

    def test_flags(self) -> None:
        args = build_parser().parse_args(
            ["--profile", "full", "--no-color", "--no-tty", "--upload"]
        )
        assert args.no_color and args.no_tty and args.upload

    def test_models_subcommand(self) -> None:
        args = build_parser().parse_args(["models", "verify"])
        assert args.command == "models"
        assert args.action == "verify"

    def test_stages_and_overrides(self) -> None:
        args = build_parser().parse_args(
            ["--stages", "descale,deband,encode", "--target-fps", "60", "--descale-height", "720"]
        )
        assert args.stages == "descale,deband,encode"
        assert args.target_fps == "60"
        assert args.descale_height == 720

    def test_num_streams_flag_parses_and_defaults_to_none(self) -> None:
        args = build_parser().parse_args(["--profile", "full", "--num-streams", "3"])
        assert args.num_streams == 3
        assert build_parser().parse_args(["--profile", "full"]).num_streams is None


class TestNumStreamsPassthrough:
    """W6: --num-streams → cli_num_streams → detect_hardware, приоритет флага."""

    def test_flag_reaches_detect_hardware(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from anime_upscale import cli
        from anime_upscale.cli import _resolve_batch_inputs
        from anime_upscale.config import load_config
        from anime_upscale.hardware import HardwareProfile

        repo = Path(__file__).resolve().parent.parent
        config = load_config(repo / "config.toml")
        args = build_parser().parse_args(
            ["--stages", "encode", "-i", "a.mkv", "--num-streams", "2"]
        )
        seen: dict[str, object] = {}
        fake = HardwareProfile(
            gpus=(),
            gpu_count=0,
            cpu_logical_cores=1,
            num_streams_per_gpu=5,
            encode_threads=1,
            precision="fp16",
        )

        def _spy(overrides: object, cli_num_streams: int | None, **_kw: object):
            seen["cli_num_streams"] = cli_num_streams
            seen["nn_stages_selected"] = _kw.get("nn_stages_selected")
            return fake

        monkeypatch.setattr(cli, "detect_hardware", _spy)
        _, hardware, _ffmpeg, entries = _resolve_batch_inputs(args, config, Path.cwd())
        assert seen["cli_num_streams"] == 2
        assert seen["nn_stages_selected"] is False  # encode-only: no NN stage
        assert entries == ["a.mkv"]
        assert hardware.num_streams_per_gpu == 5  # the detected profile flows through


class TestProfileResolutionViaCli:
    def test_profile_not_found_lists_available(self) -> None:
        from anime_upscale.config import load_config

        repo = Path(__file__).resolve().parent.parent
        config = load_config(repo / "config.toml")
        with pytest.raises(ConfigError, match="upscale-only"):
            resolve_run(config, profile="nope")

    def test_no_selection_lists_profiles(self) -> None:
        from anime_upscale.config import load_config

        repo = Path(__file__).resolve().parent.parent
        config = load_config(repo / "config.toml")
        with pytest.raises(ConfigError, match="interp-only"):
            resolve_run(config, profile=None)


class TestDefaultInputs:
    def test_exactly_one_file_selected(self, tmp_path: Path) -> None:
        (tmp_path / "only.mkv").write_bytes(b"x")
        (tmp_path / "notes.txt").write_text("ignored: not media")
        assert _default_inputs(tmp_path) == ["only.mkv"]

    def test_zero_files_is_loud(self, tmp_path: Path) -> None:
        with pytest.raises(Exception, match="пуст"):
            _default_inputs(tmp_path)

    def test_multiple_files_is_loud_with_listing(self, tmp_path: Path) -> None:
        (tmp_path / "a.mkv").write_bytes(b"x")
        (tmp_path / "b.mp4").write_bytes(b"x")
        with pytest.raises(EntryError, match=r"a\.mkv") as excinfo:
            _default_inputs(tmp_path)
        assert "b.mp4" in str(excinfo.value)


def _mute():
    from anime_upscale.reporting import Reporter

    return Reporter.__new__(Reporter)  # never emits in these tests


class TestLazyImports:
    def test_cli_import_does_not_load_vs_or_torch(self) -> None:
        """M7: importing the CLI (the `models verify` entry) must not load the
        VS/torch stacks — pipeline/stages are pulled only by the batch path."""
        import subprocess
        import sys

        code = (
            "import sys; import anime_upscale.cli; "
            "assert 'vapoursynth' not in sys.modules, 'vapoursynth loaded'; "
            "assert 'torch' not in sys.modules, 'torch loaded'; "
            "assert 'anime_upscale.pipeline' not in sys.modules, 'pipeline loaded'"
        )
        result = subprocess.run(  # noqa: S603 - fixed argv: venv python + inline assert code
            [sys.executable, "-c", code], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr


class TestTools:
    def test_override_missing_file_is_loud_naming_key(self, tmp_path: Path) -> None:
        with pytest.raises(ToolError, match=r"\[tools\] ffmpeg"):
            resolve_tool("ffmpeg", str(tmp_path / "missing.exe"))

    def test_override_used_when_valid(self, tmp_path: Path) -> None:
        fake = tmp_path / "my_ffmpeg"
        fake.write_text("#!/bin/sh\n")
        fake.chmod(0o755)
        assert resolve_tool("ffmpeg", str(fake)) == str(fake)

    def test_path_lookup(self) -> None:
        assert resolve_tool("ffmpeg")  # ffmpeg present on this machine
        with pytest.raises(ToolError, match="PATH"):
            resolve_tool("definitely-not-a-tool-42")


# --- P1 power wave: batch wiring (doctor → EcoQoS opt-out → keep-awake) ------
#
# VS-free by construction: encode-only profile + patched cli._process_entry —
# no vapoursynth import happens, so these tests are safe in this module on
# VS-less machines (unlike the pipeline-seam tests in test_batch*).


def _enc_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Minimal encode-only project (same shape as test_batch_fanout._tmp_project)."""
    (tmp_path / "input").mkdir()
    (tmp_path / "output").mkdir()
    for name in ("a.mkv", "b.mkv"):
        (tmp_path / "input" / name).write_bytes(name.encode())
    (tmp_path / "config.toml").write_text(
        "[pipeline]\n"
        'order = ["descale", "deband", "upscale", "interpolate", "encode"]\n'
        "[stages.encode]\n"
        "crf = 16\n"
        "[profiles.enc]\n"
        'stages = ["encode"]\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)


class _PowerHarness:
    """Unified event timeline for the power entry points + the spawn seam."""

    def __init__(self) -> None:
        self.events: list[str] = []

    def record(self, label: str) -> None:
        self.events.append(label)

    def count(self, label: str) -> int:
        return self.events.count(label)


def _install_power_spies(monkeypatch: pytest.MonkeyPatch, harness: _PowerHarness) -> None:
    """Patch cli's power entry points with recording doubles (ordered events)."""
    import contextlib

    from anime_upscale import cli
    from anime_upscale.power import KeepAwakeStatus

    monkeypatch.setattr(cli, "run_startup_doctor", lambda _reporter: harness.record("doctor"))

    def spy_optout(_reporter: object) -> bool:
        harness.record("optout")
        return True

    monkeypatch.setattr(cli, "disable_power_throttling", spy_optout)

    def spy_keep(_reporter: object) -> object:
        harness.record("keep-enter")

        @contextlib.contextmanager
        def _fake_ctx():
            try:
                yield KeepAwakeStatus(active=True, detail="fake")
            finally:
                harness.record("keep-exit")

        return _fake_ctx()

    monkeypatch.setattr(cli, "keep_system_awake", spy_keep)


class TestPowerWiring:
    def _patch_batch_harness(
        self,
        monkeypatch: pytest.MonkeyPatch,
        harness: _PowerHarness,
        *,
        gpu_count: int,
    ) -> None:
        """Fake hardware/tool/pipeline seams; encode-only keeps VS out."""
        from anime_upscale import cli
        from anime_upscale.hardware import GPUInfo, HardwareProfile

        gpus = tuple(GPUInfo(index=i, name=f"FAKE{i}", vram_mb=16384) for i in range(gpu_count))
        fake = HardwareProfile(
            gpus=gpus,
            gpu_count=gpu_count,
            cpu_logical_cores=8,
            num_streams_per_gpu=2,
            encode_threads=8,
            precision="fp16",
        )

        def fake_detect(_overrides: object, **_kw: object) -> HardwareProfile:
            return fake

        monkeypatch.setattr(cli, "detect_hardware", fake_detect)
        monkeypatch.setattr(cli, "resolve_tool", lambda name, override=None: "/usr/bin/ffmpeg")
        monkeypatch.setattr(cli, "_process_entry", lambda *_args, **_kw: (None, None))
        _install_power_spies(monkeypatch, harness)

    def test_sequential_batch_runs_doctor_optout_and_keeps_awake(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import multiprocessing

        from anime_upscale import cli

        def boom(method: str) -> object:
            raise AssertionError(f"spawn requested on single-GPU run ({method!r})")

        monkeypatch.setattr(multiprocessing, "get_context", boom)
        _enc_project(tmp_path, monkeypatch)
        harness = _PowerHarness()
        self._patch_batch_harness(monkeypatch, harness, gpu_count=1)
        exit_code = cli.main(["--profile", "enc", "-i", "a.mkv", "--config", "config.toml"])
        assert exit_code == 0
        assert harness.events == ["doctor", "optout", "keep-enter", "keep-exit"]

    def test_power_guard_survives_ctrl_c_with_130(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from anime_upscale import cli

        _enc_project(tmp_path, monkeypatch)
        harness = _PowerHarness()
        self._patch_batch_harness(monkeypatch, harness, gpu_count=1)

        def interrupt(*_args: object, **_kw: object) -> tuple[None, None]:
            raise KeyboardInterrupt

        monkeypatch.setattr(cli, "_process_entry", interrupt)
        exit_code = cli.main(["--profile", "enc", "-i", "a.mkv", "--config", "config.toml"])
        assert exit_code == 130
        assert harness.events == ["doctor", "optout", "keep-enter", "keep-exit"]

    def test_fanout_parent_and_children_each_keep_awake(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        fake_spawn_ctx: object,
    ) -> None:
        from anime_upscale import cli

        _enc_project(tmp_path, monkeypatch)
        harness = _PowerHarness()
        self._patch_batch_harness(monkeypatch, harness, gpu_count=2)
        monkeypatch.setattr(cli, "_process_entry", lambda *_args, **_kw: (None, None))
        # Record child spawns onto the shared timeline (the fixture's Process
        # objects are wrapped, not replaced).
        real_process_factory = fake_spawn_ctx.Process  # type: ignore[attr-defined]

        def recording_process(**kwargs: object) -> object:
            proc = real_process_factory(**kwargs)
            real_start = proc.start  # type: ignore[attr-defined]

            def spy_start() -> None:
                harness.record("spawn")
                real_start()

            proc.start = spy_start  # type: ignore[attr-defined]
            return proc

        monkeypatch.setattr(
            fake_spawn_ctx,
            "Process",
            recording_process,  # type: ignore[attr-defined]
        )
        exit_code = cli.main(
            ["--profile", "enc", "-i", "a.mkv", "-i", "b.mkv", "--config", "config.toml"]
        )
        assert exit_code == 0
        # Parent: doctor + opt-out once. Children (spawned processes) re-run the
        # opt-out and keep-awake themselves (execution state is per-process).
        assert harness.count("doctor") == 1
        assert harness.count("optout") == 3
        assert harness.count("keep-enter") == 3
        assert harness.count("keep-exit") == 3
        # Ordering pin: the opt-out happens BEFORE any child spawns.
        assert harness.events.index("spawn") > harness.events.index("optout")
        # No leaked worker threads.
        for proc in fake_spawn_ctx.processes:  # type: ignore[attr-defined]
            if proc.thread is not None:
                proc.thread.join(timeout=10)
                assert not proc.thread.is_alive()


@pytest.fixture
def fake_spawn_ctx(
    monkeypatch: pytest.MonkeyPatch,
) -> object:
    """In-process spawn double running the REAL _fanout_worker in threads."""
    import multiprocessing
    import os
    import queue as queue_mod
    import threading

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", ""))

    class _FakeProcess:
        def __init__(self, *, target: object, args: tuple, daemon: bool = False) -> None:
            self.target = target
            self.args = args
            self.daemon = daemon
            self.thread: threading.Thread | None = None

        def start(self) -> None:
            self.thread = threading.Thread(target=self.target, args=self.args, daemon=True)
            self.thread.start()

        def join(self) -> None:
            if self.thread is not None:
                self.thread.join()

    class _FakeSpawnContext:
        def __init__(self) -> None:
            self.processes: list[_FakeProcess] = []

        def Queue(self) -> queue_mod.Queue:
            return queue_mod.Queue()

        def Process(self, *, target: object, args: tuple, daemon: bool = False) -> _FakeProcess:
            proc = _FakeProcess(target=target, args=args, daemon=daemon)
            self.processes.append(proc)
            return proc

    ctx = _FakeSpawnContext()

    def fake_get_context(method: str) -> _FakeSpawnContext:
        return ctx

    monkeypatch.setattr(multiprocessing, "get_context", fake_get_context)
    return ctx
