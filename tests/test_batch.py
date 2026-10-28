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
