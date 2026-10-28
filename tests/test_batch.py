"""Batch driver tests: continue-and-summarize policy (controller ruling, §11.5)."""

from pathlib import Path

import pytest

from anime_upscale import cli
from anime_upscale.hardware import GPUInfo, HardwareProfile
from anime_upscale.reporting import Reporter, resolve_mode

pytestmark = pytest.mark.vapoursynth  # patches the pipeline module seam (VS/torch stack)


def _fake_hw() -> HardwareProfile:
    return HardwareProfile(
        gpus=(GPUInfo(index=0, name="FAKE", vram_mb=16384),),
        gpu_count=1,
        cpu_logical_cores=8,
        num_streams_per_gpu=2,
        encode_threads=8,
        precision="fp16",
    )


def _tmp_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Minimal project: config.toml + input files; cwd switched; encode-only config."""
    (tmp_path / "input").mkdir()
    (tmp_path / "output").mkdir()
    (tmp_path / "input" / "a.mkv").write_bytes(b"a")
    (tmp_path / "input" / "b.mkv").write_bytes(b"b")
    config = (
        "[pipeline]\n"
        'order = ["descale", "deband", "upscale", "interpolate", "encode"]\n'
        "[stages.encode]\n"
        "crf = 16\n"
        "[profiles.enc]\n"
        'stages = ["encode"]\n'
        "[profiles.upl]\n"
        'stages = ["encode"]\n'
        "upload = true\n"
    )
    (tmp_path / "config.toml").write_text(config, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_batch_continues_after_failure_and_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _tmp_project(tmp_path, monkeypatch)
    calls: list[str] = []

    def fake_run_pipeline(ctx, source, output, ffmpeg_path, progress_update=None):
        from anime_upscale.pipeline import PipelineError

        calls.append(source.name)
        if source.name == "b.mkv":
            raise PipelineError("батч-фейк: декодер умер")
        return (60, 1)

    # M7: cli resolves run_pipeline as anime_upscale.pipeline.<attr> at call
    # time — the production seam is patched here, not a cli-level alias.
    monkeypatch.setattr("anime_upscale.pipeline.run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(cli, "detect_hardware", lambda *a, **kw: _fake_hw())
    monkeypatch.setattr(cli, "resolve_tool", lambda name, override=None: "/usr/bin/ffmpeg")
    exit_code = cli.main(
        ["--profile", "enc", "-i", "a.mkv", "-i", "b.mkv", "--config", "config.toml"]
    )
    out = capsys.readouterr().out
    # Both entries attempted (continue-and-summarize ruling).
    assert calls == ["a.mkv", "b.mkv"]
    # Per-file banners reflect the actual queue.
    assert "Файл 1/2: a.mkv" in out
    assert "Файл 2/2: b.mkv" in out
    # Summary names the failed entry and totals.
    assert "✗" in out and "батч-фейк" in out
    assert "Итог: 1/2 успешно, 1 с ошибкой" in out
    assert exit_code == 1


def test_batch_all_ok_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _tmp_project(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "anime_upscale.pipeline.run_pipeline",
        lambda ctx, s, o, ffmpeg_path, progress_update=None: (60, 1),
    )
    monkeypatch.setattr(cli, "detect_hardware", lambda *a, **kw: _fake_hw())
    monkeypatch.setattr(cli, "resolve_tool", lambda name, override=None: "/usr/bin/ffmpeg")
    exit_code = cli.main(
        ["--profile", "enc", "-i", "a.mkv", "-i", "b.mkv", "--config", "config.toml"]
    )
    out = capsys.readouterr().out
    assert "Итог: 2/2 успешно, 0 с ошибкой" in out
    assert exit_code == 0


def test_batch_summary_output_shapes() -> None:
    reporter = Reporter(resolve_mode(no_color=True, no_tty=True))
    reporter.batch_summary([("ok.mkv", True, ""), ("bad.mkv", False, "причина")])
    # captured implicitly; smoke for the summary math path


def test_preflight_network_failure_is_loud_exit_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """M1: a pre-flight network failure (DoH/reset) must print the loud Russian
    error and exit 1 — never surface as a raw traceback."""
    import requests

    _tmp_project(tmp_path, monkeypatch)

    def boom(entries: list[str], client_factory: object, reporter: Reporter) -> list:
        raise requests.ConnectionError("Connection reset by peer")

    monkeypatch.setattr(cli, "build_queue", boom)
    monkeypatch.setattr(cli, "detect_hardware", lambda *a, **kw: _fake_hw())
    monkeypatch.setattr(cli, "resolve_tool", lambda name, override=None: "/usr/bin/ffmpeg")
    exit_code = cli.main(["--profile", "enc", "-i", "a.mkv", "--config", "config.toml"])
    out = capsys.readouterr().out
    assert exit_code == 1
    assert "✗ Ошибка" in out
    assert "Connection reset by peer" in out


def test_upload_failure_gets_dedicated_exit_3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """M10/§10: pipeline ok + upload failed = the dedicated exit code 3, the
    summary marks the entry as upload-failed, and the pipeline result stays ok."""
    from anime_upscale.upload import UploadError

    _tmp_project(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "anime_upscale.pipeline.run_pipeline",
        lambda ctx, s, o, ffmpeg_path, progress_update=None: (60, 1),
    )

    def fake_upload(path: object, client: object, reporter: object, progress: object = None) -> str:
        raise UploadError("gofile upload вернул «error-serverError»")

    monkeypatch.setattr(cli, "upload_output", fake_upload)
    monkeypatch.setattr(cli, "detect_hardware", lambda *a, **kw: _fake_hw())
    monkeypatch.setattr(cli, "resolve_tool", lambda name, override=None: "/usr/bin/ffmpeg")
    exit_code = cli.main(["--profile", "upl", "-i", "a.mkv", "--config", "config.toml"])
    out = capsys.readouterr().out
    assert exit_code == cli.EXIT_UPLOAD == 3
    assert "загрузка не удалась" in out
    assert "Итог: 0/1 успешно, 1 с ошибкой" in out


def test_pipeline_failure_outranks_upload_failure_on_mixed_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """M10/§11.5: a batch with both a pipeline failure and an upload failure
    exits 1 (pipeline rank first), not 3."""
    from anime_upscale.pipeline import PipelineError
    from anime_upscale.upload import UploadError

    _tmp_project(tmp_path, monkeypatch)

    def fake_run(ctx, source, output, ffmpeg_path, progress_update=None):
        if source.name == "b.mkv":
            raise PipelineError("батч-фейк: декодер умер")
        return (60, 1)

    def fake_upload(path: object, client: object, reporter: object, progress: object = None) -> str:
        raise UploadError("upload dead")

    monkeypatch.setattr("anime_upscale.pipeline.run_pipeline", fake_run)
    monkeypatch.setattr(cli, "upload_output", fake_upload)
    monkeypatch.setattr(cli, "detect_hardware", lambda *a, **kw: _fake_hw())
    monkeypatch.setattr(cli, "resolve_tool", lambda name, override=None: "/usr/bin/ffmpeg")
    exit_code = cli.main(
        ["--profile", "upl", "-i", "a.mkv", "-i", "b.mkv", "--config", "config.toml"]
    )
    out = capsys.readouterr().out
    assert exit_code == 1
    assert "upload dead" in out and "батч-фейк" in out


def test_vsless_batch_run_is_loud_russian_not_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """PR1-review Minor: the lazy pipeline import sits INSIDE the guarded
    region — a VS-less machine gets the loud Russian error through the batch
    summary (exit 1), never a raw ModuleNotFoundError traceback."""
    import sys

    import anime_upscale

    _tmp_project(tmp_path, monkeypatch)
    # VS-less simulation seam: None in sys.modules makes `import vapoursynth`
    # raise ImportError. Evicting the cached pipeline module AND the package
    # attribute (earlier imports cache it there — `from pkg import mod` hits
    # the attribute first) forces the lazy re-import to actually hit it.
    monkeypatch.setitem(sys.modules, "vapoursynth", None)
    monkeypatch.delitem(sys.modules, "anime_upscale.pipeline", raising=False)
    monkeypatch.delattr(anime_upscale, "pipeline", raising=False)
    monkeypatch.setattr(cli, "detect_hardware", lambda *a, **kw: _fake_hw())
    monkeypatch.setattr(cli, "resolve_tool", lambda name, override=None: "/usr/bin/ffmpeg")
    exit_code = cli.main(["--profile", "enc", "-i", "a.mkv", "--config", "config.toml"])
    out = capsys.readouterr().out
    assert exit_code == 1
    assert "VapourSynth не найден" in out


def test_upload_time_network_error_gets_dedicated_exit_3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """PR1-review nit pin: a connection death DURING upload (pick_server /
    upload_file session calls) is wrapped by upload_output into UploadError →
    the dedicated exit 3 — NOT mislabeled as a pipeline failure (exit 1).
    (Verified already-correct on main via live probe; this test pins it.)"""
    import requests

    class DeadSession:
        def get(self, *a: object, **kw: object) -> object:
            raise requests.ConnectionError("Connection reset by peer")

        def post(self, *a: object, **kw: object) -> object:
            raise requests.ConnectionError("Connection reset by peer")

        def pin(self, *a: object, **kw: object) -> None:
            return None

    class DeadClient:
        def __init__(self, *a: object, **kw: object) -> None:
            self.session = DeadSession()

    _tmp_project(tmp_path, monkeypatch)
    # The real upload runs: the (faked) pipeline's output must exist so
    # upload_file reaches pick_server, where the session dies.
    (tmp_path / "output" / "a.mkv").write_bytes(b"x")
    monkeypatch.setattr(
        "anime_upscale.pipeline.run_pipeline",
        lambda ctx, s, o, ffmpeg_path, progress_update=None: (60, 1),
    )
    monkeypatch.setattr(cli, "GofileClient", DeadClient)
    monkeypatch.setattr(cli, "detect_hardware", lambda *a, **kw: _fake_hw())
    monkeypatch.setattr(cli, "resolve_tool", lambda name, override=None: "/usr/bin/ffmpeg")
    exit_code = cli.main(["--profile", "upl", "-i", "a.mkv", "--config", "config.toml"])
    out = capsys.readouterr().out
    assert exit_code == cli.EXIT_UPLOAD == 3
    assert "Connection reset by peer" in out
    assert "загрузка не удалась" in out
