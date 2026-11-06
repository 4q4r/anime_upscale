"""ffmpeg mux/subprocess arg tests (plan §1/§5, research/05 §5)."""

import shutil
import subprocess
from typing import Any

import pytest

import anime_upscale.ffmpeg as ffmpeg_mod
from anime_upscale.ffmpeg import (
    Encoder,
    build_mux_args,
    diagnose_encoder,
    encoder_open_failure,
    finish_encoder,
)
from anime_upscale.ffmpeg_repair import verdict_wants_reinstall

HAVE_FFMPEG = shutil.which("ffmpeg") is not None
FFMPEG_BIN = shutil.which("ffmpeg") or "ffmpeg"


class TestMuxArgs:
    def test_two_input_shape(self) -> None:
        args = build_mux_args(
            source="/proj/input/src.mkv",
            output="/proj/output/dst.mkv",
            video_args=["-c:v", "libx265", "-crf", "16"],
            width=1920,
            height=1080,
            fps_num=60000,
            fps_den=1001,
        )
        assert args[:2] == ["ffmpeg", "-y"]
        assert "-f" in args and args[args.index("-f") + 1] == "rawvideo"
        assert args[args.index("-pix_fmt") + 1] == "yuv420p10le"
        assert args[args.index("-s") + 1] == "1920x1080"
        assert "-r" not in args
        assert args[args.index("-framerate") + 1] == "60000/1001"
        assert args.count("-thread_queue_size") == 2
        stdin_idx = args.index("-")
        assert args[stdin_idx - 1] == "-i"
        assert args[stdin_idx - 3 : stdin_idx - 1] == ["-thread_queue_size", "4096"]
        src_index = args.index("/proj/input/src.mkv")
        assert args[src_index - 1] == "-i"
        assert args[src_index - 3 : src_index - 1] == ["-thread_queue_size", "4096"]
        assert "-map" in args and args[args.index("-map") + 1] == "0:v:0"
        maps = [args[i + 1] for i, a in enumerate(args) if a == "-map"]
        assert maps == ["0:v:0", "1:a?", "1:s?", "1:t?"]
        assert args[args.index("-map_metadata") + 1] == "1"
        assert args[args.index("-map_chapters") + 1] == "1"
        assert args[args.index("-c:a") + 1] == "copy"
        assert args[args.index("-c:s") + 1] == "copy"
        assert args[args.index("-c:t") + 1] == "copy"
        assert args[args.index("-max_interleave_delta") + 1] == "0"
        assert args[args.index("-max_muxing_queue_size") + 1] == "4096"
        assert args[-1] == "/proj/output/dst.mkv"

    def test_video_args_land_between_inputs_and_output(self) -> None:
        video = ["-c:v", "libx265", "-x265-params", "keyint=600"]
        args = build_mux_args(
            source="s.mkv",
            output="d.mkv",
            video_args=video,
            width=1280,
            height=720,
            fps_num=24000,
            fps_den=1001,
        )
        vpos = args.index("-c:v")
        srcpos = args.index("s.mkv")
        outpos = args.index("d.mkv")
        assert srcpos < vpos < outpos

    def test_stats_flag_present_for_progress(self) -> None:
        args = build_mux_args(
            source="s.mkv",
            output="d.mkv",
            video_args=[],
            width=1280,
            height=720,
            fps_num=24000,
            fps_den=1001,
        )
        assert "-stats" in args


class _ScriptedFfprobe:
    """Duck-typed subprocess.run returning one canned ffprobe JSON payload."""

    def __init__(self, stdout: bytes, returncode: int = 0) -> None:
        self.stdout = stdout
        self.returncode = returncode
        self.calls: list[list[str]] = []

    def __call__(self, args: list[str], **_kwargs: Any) -> Any:
        self.calls.append(list(args))
        return subprocess.CompletedProcess(
            args=args, returncode=self.returncode, stdout=self.stdout, stderr=b""
        )


_MOV_TEXT_JSON = b'{"streams": [{"codec_name": "mov_text"}]}'
_TX3G_JSON = b'{"streams": [{"codec_name": "tx3g"}]}'
_PGSSUB_JSON = b'{"streams": [{"codec_name": "pgssub"}]}'
_XSUB_JSON = b'{"streams": [{"codec_name": "xsub"}]}'
_NO_SUBS_JSON = b'{"streams": []}'


@pytest.fixture
def ffprobe_pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin ffprobe resolution so the arg-builder tests never need the binary."""
    monkeypatch.setattr(ffmpeg_mod, "_resolve_ffprobe", lambda _ffmpeg: "/usr/bin/ffprobe")


class TestSubtitlePreflight:
    """F8: MP4/TS mov_text dies at mux (rc=218, undiagnosed) — the preflight
    must pick a Matroska-portable verdict BEFORE the mux starts."""

    def _mux(self, **kwargs: Any) -> list[str]:
        return build_mux_args(
            source="src.mp4",
            output="out.mkv",
            video_args=["-c:v", "libx265", "-crf", "16"],
            width=1280,
            height=720,
            fps_num=24000,
            fps_den=1001,
            **kwargs,
        )

    def test_mov_text_transcodes_to_srt(self, ffprobe_pinned: None) -> None:
        runner = _ScriptedFfprobe(_MOV_TEXT_JSON)
        args = self._mux(subtitle_probe_runner=runner)
        assert args[args.index("-c:s") + 1] == "srt"

    def test_tx3g_transcodes_to_srt(self, ffprobe_pinned: None) -> None:
        args = self._mux(subtitle_probe_runner=_ScriptedFfprobe(_TX3G_JSON))
        assert args[args.index("-c:s") + 1] == "srt"

    def test_probe_uses_ffprobe_json_contract(self, ffprobe_pinned: None) -> None:
        """The probe argv must be exactly the documented read-only ffprobe call."""
        runner = _ScriptedFfprobe(_MOV_TEXT_JSON)
        self._mux(subtitle_probe_runner=runner)
        (call,) = runner.calls
        assert call[0] == "/usr/bin/ffprobe"
        assert call[1:5] == ["-v", "error", "-select_streams", "s"]
        assert call[call.index("-show_entries") + 1] == "stream=codec_name"
        assert call[call.index("-of") + 1] == "json"

    def test_pgssub_keeps_stream_copy(self, ffprobe_pinned: None) -> None:
        args = self._mux(subtitle_probe_runner=_ScriptedFfprobe(_PGSSUB_JSON))
        assert args[args.index("-c:s") + 1] == "copy"

    def test_dvdsub_and_dvbsub_keep_stream_copy(self, ffprobe_pinned: None) -> None:
        payload = b'{"streams": [{"codec_name": "dvdsub"}, {"codec_name": "dvbsub"}]}'
        args = self._mux(subtitle_probe_runner=_ScriptedFfprobe(payload))
        assert args[args.index("-c:s") + 1] == "copy"

    def test_xsub_drops_subtitles_with_warning(
        self, ffprobe_pinned: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        warnings: list[str] = []
        monkeypatch.setattr(ffmpeg_mod, "_warn", warnings.append)
        args = self._mux(subtitle_probe_runner=_ScriptedFfprobe(_XSUB_JSON))
        assert "-sn" in args
        assert "-c:s" not in args
        assert warnings == ["Субтитры типа xsub не переносятся в MKV — отброшены."]

    def test_mixed_text_and_bitmap_drops_subtitles(
        self, ffprobe_pinned: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A single -c:s serves every subtitle stream; bitmap→text conversion
        cannot succeed (ffmpeg docs), so a mov_text+pgssub mix must drop."""
        payload = b'{"streams": [{"codec_name": "mov_text"}, {"codec_name": "pgssub"}]}'
        warnings: list[str] = []
        monkeypatch.setattr(ffmpeg_mod, "_warn", warnings.append)
        args = self._mux(subtitle_probe_runner=_ScriptedFfprobe(payload))
        assert "-sn" in args
        assert warnings and "отброшены" in warnings[0]

    def test_no_subtitles_leaves_args_unchanged(self, ffprobe_pinned: None) -> None:
        args = self._mux(subtitle_probe_runner=_ScriptedFfprobe(_NO_SUBS_JSON))
        assert args[args.index("-c:s") + 1] == "copy"
        assert "-sn" not in args

    def test_missing_ffprobe_leaves_args_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No ffprobe anywhere = today's behavior, the preflight never blocks."""
        monkeypatch.setattr(ffmpeg_mod, "_resolve_ffprobe", lambda _ffmpeg: None)
        runner = _ScriptedFfprobe(_MOV_TEXT_JSON)
        args = self._mux(subtitle_probe_runner=runner)
        assert runner.calls == []
        assert args[args.index("-c:s") + 1] == "copy"

    def test_failing_ffprobe_degrades_to_copy(self, ffprobe_pinned: None) -> None:
        runner = _ScriptedFfprobe(b"", returncode=1)
        args = self._mux(subtitle_probe_runner=runner)
        assert args[args.index("-c:s") + 1] == "copy"

    def test_garbage_probe_json_degrades_to_copy(self, ffprobe_pinned: None) -> None:
        runner = _ScriptedFfprobe(b"<not json>")
        args = self._mux(subtitle_probe_runner=runner)
        assert args[args.index("-c:s") + 1] == "copy"

    def test_probe_oserror_degrades_to_copy(self, ffprobe_pinned: None) -> None:
        """A dead binary (FileNotFoundError on spawn) must not change the argv."""

        def _boom(_args: list[str], **_kw: Any) -> Any:
            msg = "ffprobe vanished"
            raise FileNotFoundError(msg)

        args = self._mux(subtitle_probe_runner=_boom)
        assert args[args.index("-c:s") + 1] == "copy"


class TestEncoderOpenFailure:
    def test_libx265_open_marker(self) -> None:
        assert encoder_open_failure("Cannot open libx265 encoder.\nError opening output files")

    def test_unknown_encoder_marker(self) -> None:
        assert encoder_open_failure("Unknown encoder 'libx265'")

    def test_other_failures_not_classified(self) -> None:
        assert not encoder_open_failure("No such file or directory")
        assert not encoder_open_failure("")


class _FakeProcess:
    """Duck-typed Popen for finish_encoder (stdin/wait/returncode/args only)."""

    def __init__(self, returncode: int, stderr_tail: bytes, *, close_raises: bool = False) -> None:
        self.args = ["ffmpeg"]
        self.returncode = -999
        self._returncode = returncode
        self._stderr_tail = stderr_tail
        self.closed = False
        self.close_raises = close_raises

        class _Stdin:
            def __init__(self, outer: "_FakeProcess") -> None:
                self._outer = outer
                self.closed = False

            def close(self) -> None:
                if self._outer.close_raises:
                    self.closed = True
                    msg = "OSError: [Errno 22] Invalid argument"
                    raise OSError(msg)
                self.closed = True
                self._outer.closed = True

        self.stdin = _Stdin(self)

        class _Stderr:
            def __init__(self, outer: "_FakeProcess") -> None:
                self._outer = outer

            def readline(self) -> bytes:
                tail, self._outer._stderr_tail = self._outer._stderr_tail, b""
                return tail

            def close(self) -> None:
                pass

        self.stderr = _Stderr(self)

    def wait(self) -> int:
        self.returncode = self._returncode
        return self.returncode


class TestFinishEncoder:
    def _encoder(self, returncode: int, tail: list[bytes], **proc_kwargs: Any) -> Encoder:
        process = _FakeProcess(returncode, b"", **proc_kwargs)
        encoder = Encoder(process=process)  # ty: ignore[arg-type]
        encoder.tail.extend(tail)  # the drain thread normally fills this ring
        return encoder

    def test_success_returns_silently(self) -> None:
        finish_encoder(self._encoder(0, []))

    def test_failure_reports_stderr_tail(self) -> None:
        encoder = self._encoder(1, [b"boom\n", b"line2\n"])
        with pytest.raises(RuntimeError, match="код 1") as exc_info:
            finish_encoder(encoder)
        assert "boom" in str(exc_info.value)

    def test_broken_stdin_close_does_not_mask_the_tail(self) -> None:
        """E05/E06/E10/E11 live class: the EINVAL on close must never replace
        ffmpeg's own exit code + stderr tail (Windows raises EINVAL where Linux
        raises EPIPE when the peer is already dead)."""
        encoder = self._encoder(1, [b"Cannot open libx265 encoder.\n"], close_raises=True)
        with pytest.raises(RuntimeError, match="код 1") as exc_info:
            finish_encoder(encoder)
        assert "Cannot open libx265 encoder" in str(exc_info.value)

    def test_x265_error_lines_surface_first_from_full_ring(self) -> None:
        """Live bug: the real x265 [error] sat above the last-10 line cut, so the
        message showed only generic ffmpeg noise. Every x265 error/fatal line in
        the 50-line ring must lead the report."""
        encoder = self._encoder(1, [b"Cannot open libx265 encoder.\n"])
        encoder.tail.extend(
            [
                (
                    b"[libx265 @ 0x1] x265 [error]: frameNumThreads (--frame-threads) must be "
                    b"[0 .. X265_MAX_FRAME_THREADS)\n"
                ),
                *[f"filler-{index}\n".encode() for index in range(20)],
            ]
        )
        with pytest.raises(RuntimeError) as exc_info:
            finish_encoder(encoder)
        message = str(exc_info.value)
        assert "x265 [error]" in message
        error_pos = message.index("x265 [error]")
        assert error_pos < message.index("filler-19")


def _probe_result(returncode: int, stderr: bytes) -> Any:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=b"", stderr=stderr)


class _FakeRunner:
    """Scripted subprocess.run stand-in: maps pix_fmt probes to canned results."""

    def __init__(self, encoders_lines: bytes, probes: dict[str, tuple[int, bytes]]) -> None:
        self.encoders_lines = encoders_lines
        self.probes = probes
        self.calls: list[list[str]] = []

    def __call__(self, args: list[str], **_kwargs: Any) -> Any:
        self.calls.append(list(args))
        if "-encoders" in args:
            return subprocess.CompletedProcess(
                args=args, returncode=0, stdout=self.encoders_lines, stderr=b""
            )
        pix_fmt = args[args.index("-pix_fmt") + 1]
        returncode, stderr = self.probes.get(pix_fmt, (1, f"{pix_fmt} probe not scripted".encode()))
        return _probe_result(returncode, stderr)

    def probe_calls(self) -> list[list[str]]:
        return [call for call in self.calls if "-pix_fmt" in call]


_FIX_URL = "gyan.dev"


class TestDiagnoseEncoder:
    def test_absent_encoder_verdict(self) -> None:
        runner = _FakeRunner(b" V....D libx264 ... \n", {})
        report = diagnose_encoder("ffmpeg", "Unknown encoder 'libx265'", runner=runner)
        assert report is not None
        assert "libx265 отсутствует в этой сборке ffmpeg" in report
        assert _FIX_URL in report

    def test_ten_bit_unsupported_verdict(self) -> None:
        runner = _FakeRunner(
            b" V....D libx265              libx265 H.265 / HEVC\n",
            {"yuv420p10le": (1, b"10-bit probe died"), "yuv420p": (0, b"")},
        )
        report = diagnose_encoder("ffmpeg", "Cannot open libx265 encoder.", runner=runner)
        assert report is not None
        assert "10-бит (Main10) не поддерживается" in report
        assert _FIX_URL in report
        assert "10-bit probe died" in report

    def test_ten_bit_ok_means_no_reproducible_cause(self) -> None:
        runner = _FakeRunner(
            b" V....D libx265              libx265 H.265 / HEVC\n",
            {"yuv420p10le": (0, b"")},
        )
        report = diagnose_encoder("ffmpeg", "Cannot open libx265 encoder.", runner=runner)
        assert report is not None
        assert "проба с боевыми параметрами" in report
        assert "причина в другом" not in report
        assert _FIX_URL not in report
        assert verdict_wants_reinstall(report) is False

    def test_probe_runs_with_production_x265_params(self) -> None:
        runner = _FakeRunner(
            b" V....D libx265              libx265 H.265 / HEVC\n",
            {"yuv420p10le": (0, b"")},
        )
        diagnose_encoder(
            "ffmpeg",
            "Cannot open libx265 encoder.",
            runner=runner,
            x265_params="pools=32:keyint=600",
        )
        probes = runner.probe_calls()
        assert probes, "diagnosis must actually probe"
        for probe in probes:
            assert probe[probe.index("-x265-params") + 1] == "pools=32:keyint=600"

    def test_x265_param_failure_is_params_verdict_without_reinstall(self) -> None:
        """Live lie: the probe used default x265 params, passed, and the verdict
        said «причина в другом» while production params were the actual cause."""
        x265_error = (
            b"x265 [error]: frameNumThreads (--frame-threads) must be [0 .. X265_MAX_FRAME_THREADS)"
        )
        runner = _FakeRunner(
            b" V....D libx265              libx265 H.265 / HEVC\n",
            {"yuv420p10le": (1, x265_error), "yuv420p": (1, x265_error)},
        )
        report = diagnose_encoder(
            "ffmpeg",
            "Cannot open libx265 encoder.",
            runner=runner,
            x265_params="pools=32",
        )
        assert report is not None
        assert "frameNumThreads" in report
        assert "параметр" in report
        assert "причина в другом" not in report
        assert verdict_wants_reinstall(report) is False

    def test_even_eight_bit_dead_verdict(self) -> None:
        runner = _FakeRunner(
            b" V....D libx265              libx265 H.265 / HEVC\n",
            {
                "yuv420p10le": (1, b"no"),
                "yuv420p": (1, b"8-bit probe died too"),
            },
        )
        report = diagnose_encoder("ffmpeg", "Cannot open libx265 encoder.", runner=runner)
        assert report is not None
        assert "даже 8-битное кодирование" in report
        assert "8-bit probe died too" in report

    def test_unrelated_failure_not_diagnosed(self) -> None:
        runner = _FakeRunner(b"", {})
        assert diagnose_encoder("ffmpeg", "No such file or directory", runner=runner) is None


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not on PATH")
class TestDiagnoseEncoderLive:
    """Live probe smoke against the machine's real ffmpeg build."""

    def test_probe_pipeline_runs_and_classifies(self) -> None:
        encoders = subprocess.run(  # noqa: S603 - fixed argv, resolved binary
            [FFMPEG_BIN, "-hide_banner", "-encoders"], capture_output=True, check=False
        )
        assert encoders.returncode == 0
        report = diagnose_encoder(
            FFMPEG_BIN, "Cannot open libx265 encoder. Error while opening encoder"
        )
        assert report is not None
        # On any sane ffmpeg build one of the three documented verdicts fires.
        assert any(
            phrase in report
            for phrase in (
                "libx265 отсутствует",
                "10-бит (Main10) не поддерживается",
                "проба с боевыми параметрами",
                "даже 8-битное кодирование",
            )
        )


class TestFinishEncoderWithDiagnosis:
    def test_encoder_open_failure_appends_the_report(self) -> None:
        if not HAVE_FFMPEG:
            pytest.skip("ffmpeg not on PATH")
        process = _FakeProcess(1, b"")
        encoder = Encoder(process=process)  # ty: ignore[arg-type]
        encoder.tail.extend([b"Cannot open libx265 encoder.\n"])
        with pytest.raises(RuntimeError, match="ДИАГНОСТИКА") as exc_info:
            finish_encoder(encoder)
        assert "libx265" in str(exc_info.value)


@pytest.mark.skipif(
    not HAVE_FFMPEG or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not on PATH",
)
class TestSubtitlePreflightLive:
    """Live verdict against the machine's real ffmpeg 9 build (probe7b parity).

    probe7b evidence: an MP4 with mov_text + ``-c:s copy`` into matroska dies
    with "Subtitle codec mov_text (94213) is not supported", rc=218."""

    def test_real_mov_text_source_transcodes_to_srt(self, tmp_path: Any) -> None:
        clip = tmp_path / "in.mp4"
        subs = tmp_path / "sub.srt"
        subs.write_text("1\n00:00:00,000 --> 00:00:01,000\nТестовая реплика.\n", encoding="utf-8")
        make = subprocess.run(  # noqa: S603 - fixed argv, resolved binary
            [
                FFMPEG_BIN,
                "-y",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=1:size=320x240:rate=10",
                "-i",
                str(subs),
                "-map",
                "0:v",
                "-map",
                "1:s",
                "-c:s",
                "mov_text",
                str(clip),
            ],
            capture_output=True,
            check=False,
        )
        assert make.returncode == 0, make.stderr.decode("utf-8", "replace")
        args = build_mux_args(
            source=clip,
            output=tmp_path / "out.mkv",
            video_args=["-c:v", "copy"],
            width=320,
            height=240,
            fps_num=10,
            fps_den=1,
        )
        assert args[args.index("-c:s") + 1] == "srt"
