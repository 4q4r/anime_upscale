"""Auto-reinstall ladder tests: verdict trigger matrix, winget uninstall/install,
locator (Links-first + Packages walk), override interaction, retry-once flow."""

import os
from pathlib import Path
from typing import Any

import pytest

from anime_upscale.ffmpeg import diagnose_encoder
from anime_upscale.ffmpeg_repair import (
    list_winget_ffmpeg_ids,
    locate_winget_ffmpeg,
    repair_ffmpeg_build,
    uninstall_winget_ffmpeg_packages,
    verdict_wants_reinstall,
)

_ABSENT_VERDICT = (
    "ffmpeg завершился с ошибкой (код 1).\nUnknown encoder 'libx265'\n"
    "\nДИАГНОСТИКА КОДЕКА: libx265 отсутствует в этой сборке ffmpeg.\n"
)
_10BIT_VERDICT = (
    "ffmpeg завершился с ошибкой (код 1).\nCannot open libx265 encoder.\n"
    "\nДИАГНОСТИКА КОДЕКА: libx265 есть, но 10-бит (Main10) не поддерживается — "
    "нужна полная сборка.\n"
)
_8BIT_VERDICT = (
    "ffmpeg завершился с ошибкой (код 1).\nCannot open libx265 encoder.\n"
    "\nДИАГНОСТИКА КОДЕКА: libx265 есть, но не работает даже 8-битное кодирование — "
    "сборка ffmpeg неисправна.\n"
)
_OTHER_VERDICT = (
    "ffmpeg завершился с ошибкой (код 1).\nCannot open libx265 encoder.\n"
    "\nДИАГНОСТИКА КОДЕКА: 10-бит работает — причина в другом.\n"
)


class _NotesReporter:
    """Duck-typed reporter collecting all emitted lines."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def line(self, text: str) -> None:
        self.lines.append(text)

    def note(self, text: str) -> None:
        self.lines.append(text)

    def warn(self, text: str) -> None:
        self.lines.append(text)

    def success(self, text: str) -> None:
        self.lines.append(text)

    def error(self, text: str) -> None:
        self.lines.append(text)


class _WingetRunner:
    """Scripted subprocess.run: winget list/uninstall/install + 10-bit probes."""

    def __init__(
        self,
        *,
        list_output: str = "",
        list_rc: int = 0,
        install_rc: int = 0,
        install_stderr: bytes = b"",
        probe_ok: bool = True,
        encoders_stdout: bytes = b"",
    ) -> None:
        self.list_output = list_output
        self.list_rc = list_rc
        self.install_rc = install_rc
        self.install_stderr = install_stderr
        self.probe_ok = probe_ok
        self.encoders_stdout = encoders_stdout
        self.calls: list[list[str]] = []

    def __call__(self, args: list[str], **_kwargs: Any) -> Any:
        import subprocess

        self.calls.append(args)
        if "-encoders" in args:
            return subprocess.CompletedProcess(
                args=args, returncode=0, stdout=self.encoders_stdout, stderr=b""
            )
        if "list" in args:
            return subprocess.CompletedProcess(
                args=args, returncode=self.list_rc, stdout=self.list_output.encode(), stderr=b""
            )
        if "uninstall" in args or "install" in args:
            return subprocess.CompletedProcess(
                args=args,
                returncode=self.install_rc,
                stdout=b"winget progress",
                stderr=self.install_stderr,
            )
        if "-pix_fmt" in args:  # 10-bit probe against the newly installed binary
            return subprocess.CompletedProcess(
                args=args, returncode=0 if self.probe_ok else 1, stdout=b"", stderr=b"probe stderr"
            )
        return subprocess.CompletedProcess(args=args, returncode=0, stdout=b"", stderr=b"")


_WINGET_LIST_WITH_FFMPEG = (
    "\r \r Name            Id                Version   Source\r\n"
    "-------------------------------------------------------\r\n"
    "Gyan.FFmpeg       Gyan.FFmpeg       9.0.2     winget\r\n"
    "Broken Builds     Other.FFmpeg.old  1.0       winget\r\n"
    "Shim only         Microsoft.FFmpegOverloads   1.0  winget\r\n"
)


class TestVerdictWantsReinstall:
    def test_absent_encoder_triggers(self) -> None:
        assert verdict_wants_reinstall(_ABSENT_VERDICT) is True

    def test_ten_bit_unsupported_triggers(self) -> None:
        assert verdict_wants_reinstall(_10BIT_VERDICT) is True

    def test_eight_bit_dead_triggers(self) -> None:
        assert verdict_wants_reinstall(_8BIT_VERDICT) is True

    def test_other_cause_does_not_trigger(self) -> None:
        assert verdict_wants_reinstall(_OTHER_VERDICT) is False

    def test_unrelated_failure_does_not_trigger(self) -> None:
        assert (
            verdict_wants_reinstall("ffmpeg завершился с ошибкой (код 1).\nNo such file") is False
        )

    def test_verdicts_come_from_the_real_diagnosis(self) -> None:
        """The markers the ladder greps are the ones diagnose_encoder emits."""
        absent = diagnose_encoder("ffmpeg", "Unknown encoder 'libx265'", runner=_WingetRunner())
        other = diagnose_encoder(
            "ffmpeg",
            "Cannot open libx265 encoder.",
            runner=_WingetRunner(
                encoders_stdout=b" V....D libx265              libx265 H.265 / HEVC\n",
                probe_ok=True,
            ),
        )
        assert verdict_wants_reinstall(absent) is True
        assert verdict_wants_reinstall(other) is False


class TestListWingetFfmpegIds:
    def test_parses_dotted_ids(self) -> None:
        runner = _WingetRunner(list_output=_WINGET_LIST_WITH_FFMPEG)
        ids = list_winget_ffmpeg_ids(runner)
        assert "Gyan.FFmpeg" in ids
        assert "Other.FFmpeg.old" in ids

    def test_excludes_microsoft_alias_shims(self) -> None:
        runner = _WingetRunner(list_output=_WINGET_LIST_WITH_FFMPEG)
        assert "Microsoft.FFmpegOverloads" not in list_winget_ffmpeg_ids(runner)

    def test_no_match_or_nonzero_list_is_empty(self) -> None:
        assert list_winget_ffmpeg_ids(_WingetRunner(list_output="")) == []
        assert list_winget_ffmpeg_ids(_WingetRunner(list_output="", list_rc=1)) == []


class TestUninstallWingetFfmpeg:
    def test_uninstalls_each_found_id_unattended(self) -> None:
        runner = _WingetRunner(list_output=_WINGET_LIST_WITH_FFMPEG)
        reporter = _NotesReporter()
        uninstall_winget_ffmpeg_packages(runner, reporter)
        uninstalls = [c for c in runner.calls if "uninstall" in c]
        assert len(uninstalls) == 2
        for call in uninstalls:
            assert "--id" in call
            assert "-e" in call and "--silent" in call and "--disable-interactivity" in call

    def test_no_winget_known_package_warns_manual(self) -> None:
        runner = _WingetRunner(list_output="")
        reporter = _NotesReporter()
        uninstall_winget_ffmpeg_packages(runner, reporter)
        assert [c for c in runner.calls if "uninstall" in c] == []
        assert any("установлен вручную" in line for line in reporter.lines)


class TestLocateWingetFfmpeg:
    def _fake_localappdata(self, monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
        monkeypatch.setenv("LOCALAPPDATA", str(root))

    def test_links_shim_wins(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        links = tmp_path / "Microsoft" / "WinGet" / "Links"
        packages = tmp_path / "Microsoft" / "WinGet" / "Packages"
        links.mkdir(parents=True)
        packages.mkdir(parents=True)
        (links / "ffmpeg.exe").write_bytes(b"shim")
        deep = packages / "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe" / "bin"
        deep.mkdir(parents=True)
        (deep / "ffmpeg.exe").write_bytes(b"real")
        self._fake_localappdata(monkeypatch, tmp_path)
        assert locate_winget_ffmpeg() == links / "ffmpeg.exe"

    def test_packages_walk_finds_deep_binary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        deep = (
            tmp_path
            / "Microsoft"
            / "WinGet"
            / "Packages"
            / "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
            / "ffmpeg-9.0.2-full_build"
            / "bin"
        )
        deep.mkdir(parents=True)
        (deep / "ffmpeg.exe").write_bytes(b"real")
        self._fake_localappdata(monkeypatch, tmp_path)
        assert locate_winget_ffmpeg() == deep / "ffmpeg.exe"

    def test_nothing_found_is_none(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._fake_localappdata(monkeypatch, tmp_path)
        assert locate_winget_ffmpeg() is None


class TestRepairFfmpegBuild:
    def test_pinned_override_blocks_the_ladder(self) -> None:
        runner = _WingetRunner()
        reporter = _NotesReporter()
        with pytest.raises(RuntimeError, match=r"\[tools\] ffmpeg"):
            repair_ffmpeg_build(_ABSENT_VERDICT, reporter, runner=runner, override_pinned=True)
        assert runner.calls == []  # nothing was touched

    def test_other_cause_verdict_never_reinstalls(self) -> None:
        runner = _WingetRunner(list_output=_WINGET_LIST_WITH_FFMPEG)
        reporter = _NotesReporter()
        assert repair_ffmpeg_build(_OTHER_VERDICT, reporter, runner=runner) is None
        assert runner.calls == []

    def test_missing_winget_is_loud_with_doc_link(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import anime_upscale.ffmpeg_repair as repair_mod

        monkeypatch.setattr(repair_mod, "_winget_bin", lambda: None)
        with pytest.raises(RuntimeError, match=r"learn\.microsoft\.com"):
            repair_ffmpeg_build(_ABSENT_VERDICT, _NotesReporter(), runner=_WingetRunner())

    def test_happy_path_ladder(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import anime_upscale.ffmpeg_repair as repair_mod

        monkeypatch.setattr(repair_mod, "_winget_bin", lambda: "winget")
        links = tmp_path / "Microsoft" / "WinGet" / "Links"
        links.mkdir(parents=True)
        shim = links / "ffmpeg.exe"
        shim.write_bytes(b"shim")
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
        monkeypatch.setenv("PATH", "/orig-only")
        runner = _WingetRunner(list_output=_WINGET_LIST_WITH_FFMPEG, probe_ok=True)
        reporter = _NotesReporter()
        result = repair_ffmpeg_build(_10BIT_VERDICT, reporter, runner=runner)
        assert result == shim
        # The SAME 10-bit probe was re-run against the NEW binary.
        probes = [c for c in runner.calls if "-pix_fmt" in c and "yuv420p10le" in c]
        assert len(probes) == 1 and probes[0][0] == str(shim)
        # Session PATH got the Links dir prepended.
        assert os.environ["PATH"].startswith(str(links))
        # The five contract progress lines.
        joined = "\n".join(reporter.lines)
        assert "[ЭТАП] ffmpeg не умеет 10-битный x265 — переустанавливаю через winget…" in joined
        assert "[ЭТАП] Удаляю текущий ffmpeg…" in joined
        assert "[ЭТАП] Устанавливаю Gyan.FFmpeg…" in joined
        assert "[ЭТАП] Проверяю 10-битное кодирование…" in joined
        assert any(line.startswith("✓ ffmpeg готов:") for line in reporter.lines)
        # Install used the exact unattended command shape.
        installs = [c for c in runner.calls if "install" in c]
        assert installs == [
            [
                "winget",
                "install",
                "--id",
                "Gyan.FFmpeg",
                "-e",
                "--silent",
                "--accept-source-agreements",
                "--accept-package-agreements",
                "--disable-interactivity",
            ]
        ]

    def test_probe_still_failing_is_terminal_with_winget_tail(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import anime_upscale.ffmpeg_repair as repair_mod

        monkeypatch.setattr(repair_mod, "_winget_bin", lambda: "winget")
        links = tmp_path / "Microsoft" / "WinGet" / "Links"
        links.mkdir(parents=True)
        (links / "ffmpeg.exe").write_bytes(b"shim")
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
        runner = _WingetRunner(list_output="", probe_ok=False)
        with pytest.raises(RuntimeError) as exc_info:
            repair_ffmpeg_build(_ABSENT_VERDICT, _NotesReporter(), runner=runner)
        message = str(exc_info.value)
        assert "gyan.dev" in message and "BtbN" in message

    def test_locate_failure_after_install_is_terminal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import anime_upscale.ffmpeg_repair as repair_mod

        monkeypatch.setattr(repair_mod, "_winget_bin", lambda: "winget")
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))  # nothing installed anywhere
        runner = _WingetRunner(
            list_output="", install_rc=1, install_stderr=b"0x8a15002b: installer failed"
        )
        with pytest.raises(RuntimeError) as exc_info:
            repair_ffmpeg_build(_ABSENT_VERDICT, _NotesReporter(), runner=runner)
        message = str(exc_info.value)
        assert "0x8a15002b" in message
        assert "gyan.dev" in message


class TestPipelineRetryOnce:
    """The retry wiring: repair verdict → one automatic re-run with the new binary."""

    def _ctx(self) -> Any:
        from anime_upscale.reporting import Reporter, resolve_mode

        return type(
            "Ctx",
            (),
            {"reporter": Reporter(resolve_mode(no_color=True, no_tty=True))},
        )()

    def test_repair_verdict_retries_once_with_new_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import anime_upscale.ffmpeg_repair as repair_mod
        import anime_upscale.pipeline as pipeline_mod

        new_binary = tmp_path / "new" / "ffmpeg.exe"
        calls: list[str] = []

        def fake_once(ctx, input_path, output_path, ffmpeg_path, progress_update=None):
            calls.append(ffmpeg_path)
            if len(calls) == 1:
                raise RuntimeError(_10BIT_VERDICT)
            return (60, 1)

        def fake_repair(report, reporter, runner=None, override_pinned=False):
            return new_binary

        monkeypatch.setattr(pipeline_mod, "_run_pipeline_once", fake_once)
        monkeypatch.setattr(repair_mod, "repair_ffmpeg_build", fake_repair)
        monkeypatch.setattr(pipeline_mod, "verdict_wants_reinstall", lambda _r: "ДИАГНОСТИКА" in _r)
        assert pipeline_mod.run_pipeline(self._ctx(), Path("a.mkv"), Path("o.mkv"), "ffmpeg") == (
            60,
            1,
        )
        assert calls == ["ffmpeg", str(new_binary)]

    def test_other_cause_failure_does_not_repair(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import anime_upscale.ffmpeg_repair as repair_mod
        import anime_upscale.pipeline as pipeline_mod

        def fake_once(ctx, input_path, output_path, ffmpeg_path, progress_update=None):
            raise RuntimeError(_OTHER_VERDICT)

        def no_repair(*a: Any, **k: Any) -> Path:
            msg = "repair must not be called"
            raise AssertionError(msg)

        monkeypatch.setattr(pipeline_mod, "_run_pipeline_once", fake_once)
        monkeypatch.setattr(repair_mod, "repair_ffmpeg_build", no_repair)
        monkeypatch.setattr(pipeline_mod, "verdict_wants_reinstall", lambda _r: False)
        with pytest.raises(RuntimeError, match="причина в другом"):
            pipeline_mod.run_pipeline(self._ctx(), Path("a.mkv"), Path("o.mkv"), "ffmpeg")

    def test_second_failure_after_repair_is_terminal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import anime_upscale.ffmpeg_repair as repair_mod
        import anime_upscale.pipeline as pipeline_mod

        def fake_once(ctx, input_path, output_path, ffmpeg_path, progress_update=None):
            raise RuntimeError(_ABSENT_VERDICT)

        monkeypatch.setattr(pipeline_mod, "_run_pipeline_once", fake_once)
        monkeypatch.setattr(
            repair_mod,
            "repair_ffmpeg_build",
            lambda *a, **k: tmp_path / "ffmpeg.exe",
        )
        monkeypatch.setattr(pipeline_mod, "verdict_wants_reinstall", lambda _r: True)
        with pytest.raises(RuntimeError, match="ДИАГНОСТИКА"):
            pipeline_mod.run_pipeline(self._ctx(), Path("a.mkv"), Path("o.mkv"), "ffmpeg")
