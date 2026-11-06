"""Multi-GPU fan-out dispatch tests (plan §9.3 as-built L4; load-dist canon §1/§4).

Env-emulation only: multiprocessing is faked at the ``multiprocessing.get_context``
seam — fake ``Process`` objects run the REAL ``_fanout_worker`` in in-process
threads, so no real spawn (and no real second GPU) is ever needed. The pinned
child view (CUDA_VISIBLE_DEVICES → one visible card) is emulated by the detect
fake, exactly like every other hardware-emulated test in this suite.

Pins:
- ≥2 GPUs → dispatch mode (spawn context, one worker per GPU, dynamic queue);
- 1 GPU / 0 GPU → the sequential driver path, byte-for-byte (no multiprocessing).
"""

import multiprocessing
import os
import queue as queue_mod
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from anime_upscale import cli
from anime_upscale.hardware import GPUInfo, HardwareProfile
from anime_upscale.reporting import Reporter

pytestmark = pytest.mark.vapoursynth  # patches the pipeline module seam (VS stack)


def _fake_hw(count: int) -> HardwareProfile:
    """Fake profile with ``count`` CUDA devices (index = position)."""
    gpus = tuple(GPUInfo(index=i, name=f"FAKE{i}", vram_mb=16384) for i in range(count))
    return HardwareProfile(
        gpus=gpus,
        gpu_count=count,
        cpu_logical_cores=8,
        num_streams_per_gpu=2,
        encode_threads=8,
        precision="fp16",
    )


def _tmp_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    names: tuple[str, ...] = ("a.mkv", "b.mkv"),
    config_text: str | None = None,
) -> Path:
    """Minimal project: config.toml + input files; cwd switched; encode-only."""
    (tmp_path / "input").mkdir()
    (tmp_path / "output").mkdir()
    for name in names:
        (tmp_path / "input" / name).write_bytes(name.encode())
    if config_text is None:
        config_text = (
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
    (tmp_path / "config.toml").write_text(config_text, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path


class _FakeProcess:
    """Recording Process double: start() runs the REAL target in a thread."""

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
    """``multiprocessing.get_context("spawn")`` double with stdlib queues."""

    def __init__(self) -> None:
        self.requested_methods: list[str] = []
        self.processes: list[_FakeProcess] = []

    def Queue(self) -> queue_mod.Queue:
        return queue_mod.Queue()

    def Process(self, *, target: object, args: tuple, daemon: bool = False) -> _FakeProcess:
        proc = _FakeProcess(target=target, args=args, daemon=daemon)
        self.processes.append(proc)
        return proc


@pytest.fixture
def fake_spawn(monkeypatch: pytest.MonkeyPatch) -> _FakeSpawnContext:
    """Patch the spawn seam; the dispatcher's real worker bodies run as threads."""
    ctx = _FakeSpawnContext()

    def fake_get_context(method: str) -> _FakeSpawnContext:
        ctx.requested_methods.append(method)
        return ctx

    monkeypatch.setattr(multiprocessing, "get_context", fake_get_context)
    return ctx


def _patch_harness(
    monkeypatch: pytest.MonkeyPatch,
    gpu_count: int,
    *,
    detect_calls: list[int] | None = None,
    run_pipeline: Callable[..., object] | None = None,
    upload: Callable[..., object] | None = None,
) -> None:
    """Wire fakes shared by parent (pre-flight) and worker threads."""
    # The fake-thread workers execute the REAL _pin_visible_gpu, which mutates
    # os.environ in-process. setenv (even to the current value) ALWAYS registers
    # a restore — delenv would no-op on an absent variable and let the pinned
    # value leak into later tests (conftest's live subprocess probes CUDA).
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", ""))

    def fake_detect(*_args: object, **_kw: object) -> HardwareProfile:
        if detect_calls is not None:
            detect_calls.append(1)
        return _fake_hw(gpu_count)

    def default_pipeline(
        ctx: object, source: Path, output: Path, ffmpeg_path: str, **_kw: object
    ) -> tuple[int, int]:
        return (60, 1)

    monkeypatch.setattr(cli, "detect_hardware", fake_detect)
    monkeypatch.setattr("anime_upscale.pipeline.run_pipeline", run_pipeline or default_pipeline)
    monkeypatch.setattr(cli, "resolve_tool", lambda name, override=None: "/usr/bin/ffmpeg")
    if upload is not None:
        monkeypatch.setattr(cli, "upload_output", upload)


# --- Zero-regression contract: 1 GPU / 0 GPU keep the sequential driver -----


def test_single_gpu_never_spawns_sequential_path_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """PIN (zero-regression contract): a single-GPU batch must never touch the
    spawn seam — the strict sequential driver keeps its exact v1 semantics."""

    def boom(method: str) -> object:
        raise AssertionError(f"spawn context requested ({method!r}) on a single-GPU run")

    monkeypatch.setattr(multiprocessing, "get_context", boom)
    _tmp_project(tmp_path, monkeypatch)
    _patch_harness(monkeypatch, gpu_count=1)
    exit_code = cli.main(
        ["--profile", "enc", "-i", "a.mkv", "-i", "b.mkv", "--config", "config.toml"]
    )
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Файл 1/2: a.mkv" in out
    assert "Файл 2/2: b.mkv" in out
    assert "Итог: 2/2 успешно, 0 с ошибкой" in out


def test_zero_gpu_never_spawns_sequential_path_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """PIN: an empty torch GPU list (CPU-only box) keeps the sequential path."""

    def boom(method: str) -> object:
        raise AssertionError(f"spawn context requested ({method!r}) on a zero-GPU run")

    monkeypatch.setattr(multiprocessing, "get_context", boom)
    _tmp_project(tmp_path, monkeypatch)
    _patch_harness(monkeypatch, gpu_count=0)
    exit_code = cli.main(
        ["--profile", "enc", "-i", "a.mkv", "-i", "b.mkv", "--config", "config.toml"]
    )
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Итог: 2/2 успешно, 0 с ошибкой" in out


# --- Dispatch mode: ≥2 GPUs -------------------------------------------------


def test_two_gpus_dispatch_spawns_two_pinned_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fake_spawn: _FakeSpawnContext,
) -> None:
    """Two GPUs → spawn context requested, one worker per GPU index, each child
    re-detects hardware itself (the pinned cuda:0 view), workers drain cleanly."""
    detect_calls: list[int] = []
    _tmp_project(tmp_path, monkeypatch)
    _patch_harness(monkeypatch, gpu_count=2, detect_calls=detect_calls)
    exit_code = cli.main(
        ["--profile", "enc", "-i", "a.mkv", "-i", "b.mkv", "--config", "config.toml"]
    )
    out = capsys.readouterr().out
    assert exit_code == 0
    assert fake_spawn.requested_methods == ["spawn"]
    # One worker per GPU: the CUDA_VISIBLE_DEVICES pin source is the GPU index.
    assert sorted(proc.args[0] for proc in fake_spawn.processes) == [0, 1]
    # Parent detection + per-child re-detection (children call it themselves).
    assert len(detect_calls) == 3
    # Children print the per-entry banners (inherited stdout).
    assert "Файл 1/2: a.mkv" in out
    assert "Файл 2/2: b.mkv" in out
    assert "Итог: 2/2 успешно, 0 с ошибкой" in out
    # Every worker exited after the sentinel drain (no leaked threads).
    for proc in fake_spawn.processes:
        assert proc.thread is not None
        proc.thread.join(timeout=10)
        assert not proc.thread.is_alive()


def test_dynamic_queue_fast_worker_takes_more(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fake_spawn: _FakeSpawnContext,
) -> None:
    """Canon §1 (multigpuscale pattern): workers pull from the shared queue when
    free — a fast worker must end up with more entries than the busy one (not
    round-robin)."""
    names = ("slow1.mkv", "f2.mkv", "f3.mkv", "f4.mkv")
    _tmp_project(tmp_path, monkeypatch, names=names)
    per_thread: dict[int, list[str]] = {}

    def fake_run(
        ctx: object, source: Path, output: Path, ffmpeg_path: str, **_kw: object
    ) -> tuple[int, int]:
        entries = per_thread.setdefault(threading.get_ident(), [])
        entries.append(source.name)
        if source.name == "slow1.mkv":
            time.sleep(0.3)  # the one long entry; the other worker drains the rest
        return (60, 1)

    _patch_harness(monkeypatch, gpu_count=2, run_pipeline=fake_run)
    exit_code = cli.main(
        [
            "--profile",
            "enc",
            "-i",
            "slow1.mkv",
            "-i",
            "f2.mkv",
            "-i",
            "f3.mkv",
            "-i",
            "f4.mkv",
            "--config",
            "config.toml",
        ]
    )
    assert exit_code == 0
    # Dispatch actually engaged: two distinct worker threads did the pipeline.
    assert len(per_thread) == 2, f"expected 2 workers, got {len(per_thread)}"
    counts = sorted((len(v) for v in per_thread.values()), reverse=True)
    assert sum(counts) == 4
    assert counts[0] >= 3, f"fast worker did not pull more entries: {counts}"


def test_fanout_summary_keeps_queue_order_and_exit_codes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fake_spawn: _FakeSpawnContext,
) -> None:
    """Continue-and-summarize in dispatch mode: a failing entry is reported at
    its point of failure, and the END summary lists entries in QUEUE order
    (a before b) even when b's result reached the parent first — the summary
    semantics stay byte-identical to the sequential driver (exit 1)."""

    def fake_run(
        ctx: object, source: Path, output: Path, ffmpeg_path: str, **_kw: object
    ) -> tuple[int, int]:
        from anime_upscale.pipeline import PipelineError

        if source.name == "b.mkv":
            raise PipelineError("батч-фейк: декодер умер")
        time.sleep(0.3)  # a.mkv finishes LAST → b's result arrives first
        return (60, 1)

    _tmp_project(tmp_path, monkeypatch)
    _patch_harness(monkeypatch, gpu_count=2, run_pipeline=fake_run)
    exit_code = cli.main(
        ["--profile", "enc", "-i", "a.mkv", "-i", "b.mkv", "--config", "config.toml"]
    )
    out = capsys.readouterr().out
    assert exit_code == 1
    lines = out.splitlines()
    # The end summary (tail block) is queue-ordered: ✓ a.mkv → ✗ b.mkv → Итог.
    summary_success = [i for i, line in enumerate(lines) if line == "✓ a.mkv"]
    assert len(summary_success) == 1, "summary success line for a.mkv missing"
    summary_error = [i for i, line in enumerate(lines) if "✗ Ошибка: b.mkv: батч-фейк" in line]
    summary_total = [
        i for i, line in enumerate(lines) if line.startswith("Итог: 1/2 успешно, 1 с ошибкой")
    ]
    assert summary_error and summary_total
    assert summary_total[0] > summary_error[-1] > summary_success[0]


def test_fanout_album_mode_chains_folder_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fake_spawn: _FakeSpawnContext,
) -> None:
    """Album chaining survives fan-out: the head entry alone decides the album
    folder (hold-back), the rest are released with its parent folder id — the
    same [None, "album-parent-uuid"] contract as the sequential driver."""
    _tmp_project(
        tmp_path,
        monkeypatch,
        config_text=(
            "[pipeline]\n"
            'order = ["descale", "deband", "upscale", "interpolate", "encode"]\n'
            "[stages.encode]\n"
            "crf = 16\n"
            "[upload.gofile]\n"
            'mode = "album"\n'
            "[profiles.upl]\n"
            'stages = ["encode"]\n'
            "upload = true\n"
        ),
    )
    folder_ids_received: list[str | None] = []

    def fake_upload(
        path: Path,
        client: object,
        reporter: Reporter,
        progress: object = None,
        folder_id: str | None = None,
    ) -> tuple[str, str | None]:
        folder_ids_received.append(folder_id)
        return "https://gofile.io/d/ALBUM123", "album-parent-uuid"

    _patch_harness(monkeypatch, gpu_count=2, upload=fake_upload)
    exit_code = cli.main(
        ["--profile", "upl", "-i", "a.mkv", "-i", "b.mkv", "--config", "config.toml"]
    )
    out = capsys.readouterr().out
    assert exit_code == 0
    assert folder_ids_received == [None, "album-parent-uuid"]
    assert "https://gofile.io/d/ALBUM123" in out


def test_fanout_delete_after_upload_wiring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fake_spawn: _FakeSpawnContext,
) -> None:
    """delete_after_upload travels through the worker spec: the child deletes
    the uploaded file and prints the note — identical to the sequential path."""
    _tmp_project(
        tmp_path,
        monkeypatch,
        names=("a.mkv",),
        config_text=(
            "[pipeline]\n"
            'order = ["descale", "deband", "upscale", "interpolate", "encode"]\n'
            "[stages.encode]\n"
            "crf = 16\n"
            "[upload.gofile]\n"
            "delete_after_upload = true\n"
            'mode = "separate"\n'
            "[profiles.upl]\n"
            'stages = ["encode"]\n'
            "upload = true\n"
        ),
    )

    def fake_run(
        ctx: object, source: Path, output: Path, ffmpeg_path: str, **_kw: object
    ) -> tuple[int, int]:
        output.write_bytes(b"encoded content")
        return (60, 1)

    def fake_upload(
        path: Path,
        client: object,
        reporter: Reporter,
        progress: object = None,
        folder_id: str | None = None,
    ) -> tuple[str, str | None]:
        return "https://gofile.io/d/OK", "uuid"

    _patch_harness(monkeypatch, gpu_count=2, run_pipeline=fake_run, upload=fake_upload)
    exit_code = cli.main(["--profile", "upl", "-i", "a.mkv", "--config", "config.toml"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert not (tmp_path / "output" / "a.mkv").exists()
    assert "Файл a.mkv удалён из output/ после выгрузки" in out


def test_fanout_logs_cpu_budget_note(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fake_spawn: _FakeSpawnContext,
) -> None:
    """L2 contract intact: active_encoders stays per-child = 1; the PARENT logs
    the informational encoder-lane budget (usable_cpu_count // gpu_count)."""
    _tmp_project(tmp_path, monkeypatch)
    _patch_harness(monkeypatch, gpu_count=2)
    monkeypatch.setattr(cli, "usable_cpu_count", lambda reporter=None: 32)
    exit_code = cli.main(
        ["--profile", "enc", "-i", "a.mkv", "-i", "b.mkv", "--config", "config.toml"]
    )
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Мульти-GPU fan-out" in out
    assert "32 ядер(а) / 2 воркера" in out
    assert "16 поток(ов) на энкодер-полосу" in out


def test_pin_visible_gpu_sets_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The child-side belt-and-braces pin: CUDA_VISIBLE_DEVICES=<gpu_index> is
    set at process start (before any torch import) — driver-level pinning."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "would-keep-original")
    cli._pin_visible_gpu(3)
    assert str(os.environ["CUDA_VISIBLE_DEVICES"]) == "3"
