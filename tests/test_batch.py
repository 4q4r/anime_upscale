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

    def boom(
        entries: list[str], client_factory: object, reporter: Reporter, strict_tls: bool = False
    ) -> list:
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


# --- Live-bug regression (2026-10-29, Windows server batch run) -------------
# Owner's live log showed «Загрузка» lines appearing where «Обработка» lines
# were expected and E01/E02 seemingly never processed. Diagnosis: the driver
# was (and stays) strictly sequential — the log is the signature of per-entry
# failures being SILENT until the end-of-batch summary, so a dead entry is
# indistinguishable from a running one and the next banner+download "steals"
# the console. These tests pin (a) CLI-order sequential execution, (b) the
# absence of any download-ahead, and (c) loud per-entry outcome lines.


def _url_batch_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Project fixture + three direct-URL entries resolved without network."""
    _tmp_project(tmp_path, monkeypatch)
    entries = [f"https://mirror.example/video/ep{n}.mkv" for n in (1, 2, 3)]

    def fake_remote_name(url: str, *, strict_tls: bool = False) -> str:
        return f"ep{url.rsplit('ep', 1)[1][0]}.mkv"

    monkeypatch.setattr("anime_upscale.fetch_input._remote_name", fake_remote_name)
    return entries


def _patch_batch_harness(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
    *,
    fail_pipeline_for: set[str] | None = None,
    upload_mode: str = "off",
) -> None:
    """Wire fakes: fetch emits two «Загрузка» lines, pipeline emits one
    «Обработка» frame callback; both append to ``events`` for order proofs."""

    def fake_fetch(
        item: object, input_dir: Path, client_factory: object, reporter: Reporter
    ) -> Path:
        name = item.name  # ty: ignore[unresolved-attribute]
        events.append(f"fetch:{name}")
        for percent in (10, 100):
            reporter.progress_bytes("Загрузка", percent, 100)
        return input_dir / name

    def fake_run_pipeline(ctx, source, output, ffmpeg_path, progress_update=None):
        events.append(f"pipeline:{source.name}")
        if fail_pipeline_for and source.name in fail_pipeline_for:
            from anime_upscale.pipeline import PipelineError

            raise PipelineError("фейк: декодер умер на первом кадре")
        if progress_update is not None:
            progress_update(0, 100)
        return (60, 1)

    monkeypatch.setattr(cli, "fetch_queue_item", fake_fetch)
    monkeypatch.setattr("anime_upscale.pipeline.run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(cli, "detect_hardware", lambda *a, **kw: _fake_hw())
    monkeypatch.setattr(cli, "resolve_tool", lambda name, override=None: "/usr/bin/ffmpeg")


def test_batch_url_entries_process_in_cli_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI order is execution order (§11.5): three URL entries run the pipeline
    strictly in 1,2,3 order — never reordered (the live-log «3,1,2» reading)."""
    entries = _url_batch_project(tmp_path, monkeypatch)
    events: list[str] = []
    _patch_batch_harness(monkeypatch, events)
    exit_code = cli.main(
        [
            "--profile",
            "enc",
            "-i",
            entries[0],
            "-i",
            entries[1],
            "-i",
            entries[2],
            "--config",
            "config.toml",
        ]
    )
    out = capsys.readouterr().out
    pipeline_order = [e for e in events if e.startswith("pipeline:")]
    assert pipeline_order == [
        "pipeline:ep1.mkv",
        "pipeline:ep2.mkv",
        "pipeline:ep3.mkv",
    ]
    assert "Файл 1/3: ep1.mkv" in out
    assert "Файл 2/3: ep2.mkv" in out
    assert "Файл 3/3: ep3.mkv" in out
    assert exit_code == 0


def test_batch_fetch_precedes_each_pipeline_no_download_ahead(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Single-threaded execution proof: entry N+1's download starts only after
    entry N's pipeline finished — no prefetch queue, no overlapping download."""
    entries = _url_batch_project(tmp_path, monkeypatch)
    events: list[str] = []
    _patch_batch_harness(monkeypatch, events)
    cli.main(
        [
            "--profile",
            "enc",
            "-i",
            entries[0],
            "-i",
            entries[1],
            "-i",
            entries[2],
            "--config",
            "config.toml",
        ]
    )
    positions = {event: index for index, event in enumerate(events)}
    for n in (1, 2):
        assert positions[f"pipeline:ep{n}.mkv"] < positions[f"fetch:ep{n + 1}.mkv"], (
            f"ep{n + 1} download started before ep{n} pipeline finished (download-ahead)"
        )


def test_entry_failure_is_loud_before_next_entry_banner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """THE live-bug regression: a per-entry pipeline failure must print its ✗
    error line AT THE POINT of failure — before the next file's banner. On the
    pre-fix driver the error was deferred to the end summary, so the console
    jumped from «Обработка: 0%» straight to «Файл N+1/M» + «Загрузка» lines:
    the owner-visible «интерполяция дерётся со скачиванием за вывод»."""
    entries = _url_batch_project(tmp_path, monkeypatch)
    events: list[str] = []
    _patch_batch_harness(monkeypatch, events, fail_pipeline_for={"ep1.mkv"})
    exit_code = cli.main(
        [
            "--profile",
            "enc",
            "-i",
            entries[0],
            "-i",
            entries[1],
            "-i",
            entries[2],
            "--config",
            "config.toml",
        ]
    )
    out_lines = capsys.readouterr().out.splitlines()

    def line_index(fragment: str) -> int:
        matches = [i for i, line in enumerate(out_lines) if fragment in line]
        assert matches, f"no output line contains {fragment!r}"
        return matches[0]

    fail_idx = line_index("ep1.mkv: фейк: декодер умер")
    banner2_idx = line_index("Файл 2/3")
    assert fail_idx < banner2_idx, "entry-1 failure printed after entry 2's banner (deferred)"
    # The continue-and-summarize ruling stays: the batch went on and the
    # end summary still names every entry (exit 1 for the pipeline failure).
    summary_idx = line_index("Итог: 2/3 успешно, 1 с ошибкой")
    assert summary_idx > line_index("Файл 3/3")
    assert exit_code == 1


def test_entry_success_terminates_phase_before_next_entry_banner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Output discipline: a successful entry prints its ✓ completion line when
    it finishes — the phase terminator that keeps one progress stream per file
    and makes «next banner» unambiguous in unattended runs."""
    entries = _url_batch_project(tmp_path, monkeypatch)
    events: list[str] = []
    _patch_batch_harness(monkeypatch, events)
    cli.main(
        [
            "--profile",
            "enc",
            "-i",
            entries[0],
            "-i",
            entries[1],
            "-i",
            entries[2],
            "--config",
            "config.toml",
        ]
    )
    out_lines = capsys.readouterr().out.splitlines()

    def line_index(fragment: str) -> int:
        matches = [i for i, line in enumerate(out_lines) if fragment in line]
        assert matches, f"no output line contains {fragment!r}"
        return matches[0]

    done_idx = line_index("ep1.mkv — готово")
    assert done_idx < line_index("Файл 2/3"), "no completion line before entry 2's banner"


def test_progress_streams_never_interleave_across_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Exactly one progress stream at a time: the console reads as strict
    per-file blocks — banner → «Загрузка» lines (download) → «Обработка» line
    (interpolation) → next banner. No download line may appear inside another
    entry's processing block (the live-log complaint)."""
    entries = _url_batch_project(tmp_path, monkeypatch)
    events: list[str] = []
    _patch_batch_harness(monkeypatch, events)
    cli.main(
        [
            "--profile",
            "enc",
            "-i",
            entries[0],
            "-i",
            entries[1],
            "-i",
            entries[2],
            "--config",
            "config.toml",
        ]
    )
    kinds: list[str] = []
    for line in capsys.readouterr().out.splitlines():
        if line.startswith("◈ Файл "):
            kinds.append("banner")
        elif line.startswith("Загрузка:"):
            kinds.append("download")
        elif "Обработка:" in line:
            kinds.append("process")
    assert kinds == ["banner", "download", "download", "process"] * 3


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
            self.strict_tls = False

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
