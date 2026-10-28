"""ffmpeg mux/subprocess arg tests (plan §1/§5, research/05 §5)."""

import shutil
import subprocess
from typing import Any

import pytest

from anime_upscale.ffmpeg import (
    Encoder,
    build_mux_args,
    diagnose_encoder,
    encoder_open_failure,
    finish_encoder,
)

HAVE_FFMPEG = shutil.which("ffmpeg") is not None
FFMPEG_BIN = shutil.which("ffmpeg") or "ffmpeg"


class TestMuxArgs:
    def test_two_input_shape(self) -> None:
        args = build_mux_args(
            source="/proj/input/src.mkv",
            output="/proj/output/dst.mkv",
            video_args=["-c:v", "libx265", "-crf", "16"],
        )
        assert args[:2] == ["ffmpeg", "-y"]
        assert "-f" in args and args[args.index("-f") + 1] == "yuv4mpegpipe"
        assert args[args.index("-i") + 1] == "-"
        src_index = args.index("/proj/input/src.mkv")
        assert args[src_index - 1] == "-i"
        assert "-map" in args and args[args.index("-map") + 1] == "0:v:0"
        maps = [args[i + 1] for i, a in enumerate(args) if a == "-map"]
        assert maps == ["0:v:0", "1:a?", "1:s?", "1:t?"]
        assert args[args.index("-map_metadata") + 1] == "1"
        assert args[args.index("-map_chapters") + 1] == "1"
        assert args[args.index("-c:a") + 1] == "copy"
        assert args[args.index("-c:s") + 1] == "copy"
        assert args[args.index("-c:t") + 1] == "copy"
        assert args[-1] == "/proj/output/dst.mkv"

    def test_video_args_land_between_inputs_and_output(self) -> None:
        video = ["-c:v", "libx265", "-x265-params", "keyint=600"]
        args = build_mux_args(source="s.mkv", output="d.mkv", video_args=video)
        vpos = args.index("-c:v")
        srcpos = args.index("s.mkv")
        outpos = args.index("d.mkv")
        assert srcpos < vpos < outpos

    def test_stats_flag_present_for_progress(self) -> None:
        args = build_mux_args(source="s.mkv", output="d.mkv", video_args=[])
        assert "-stats" in args


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


def _probe_result(returncode: int, stderr: bytes) -> Any:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=b"", stderr=stderr)


class _FakeRunner:
    """Scripted subprocess.run stand-in: maps pix_fmt probes to canned results."""

    def __init__(self, encoders_lines: bytes, probes: dict[str, tuple[int, bytes]]) -> None:
        self.encoders_lines = encoders_lines
        self.probes = probes

    def __call__(self, args: list[str], **_kwargs: Any) -> Any:
        if "-encoders" in args:
            return subprocess.CompletedProcess(
                args=args, returncode=0, stdout=self.encoders_lines, stderr=b""
            )
        pix_fmt = args[args.index("-pix_fmt") + 1]
        returncode, stderr = self.probes.get(pix_fmt, (1, f"{pix_fmt} probe not scripted".encode()))
        return _probe_result(returncode, stderr)


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

    def test_ten_bit_ok_means_other_cause(self) -> None:
        runner = _FakeRunner(
            b" V....D libx265              libx265 H.265 / HEVC\n",
            {"yuv420p10le": (0, b"")},
        )
        report = diagnose_encoder("ffmpeg", "Cannot open libx265 encoder.", runner=runner)
        assert report is not None
        assert "10-бит работает — причина в другом" in report
        assert _FIX_URL not in report

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
                "10-бит работает — причина в другом",
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
