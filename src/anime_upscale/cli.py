"""
CLI root command + models subcommand (plan §3; batch §11.5-§11.6; ruling: continue-and-summarize).

Exit codes: 0 = every entry ok; 1 = pipeline/pre-flight failure; 3 = pipeline
ok but a per-file upload failed (plan §10 dedicated code). Usage errors exit
2 — argparse's own SystemExit, not a module constant.

VS/torch-touching imports (pipeline, stages) are lazy: `models fetch/verify`
and usage errors must never load the GPU stacks.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from anime_upscale import ffmpeg_repair
from anime_upscale.config import (
    Config,
    ConfigError,
    RunConfig,
    load_config,
    parse_fps_argument,
    resolve_run,
)
from anime_upscale.fetch_input import EntryError, QueueItem, build_queue, fetch_queue_item
from anime_upscale.gofile import GofileClient, GofileError
from anime_upscale.hardware import (
    HardwareError,
    HardwareProfile,
    detect_hardware,
    print_banner,
    usable_cpu_count,
)
from anime_upscale.models.fetch import FetchError, ensure_vsrife_seed, verify_all
from anime_upscale.models.manifest import ManifestError
from anime_upscale.reporting import (
    Reporter,
    enable_windows_console_vt,
    resolve_mode,
)
from anime_upscale.tools import ToolError, resolve_tool
from anime_upscale.upload import UploadError, upload_output

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from anime_upscale.stages import StageContext

MEDIA_EXTENSIONS = frozenset(
    {".mkv", ".mp4", ".avi", ".webm", ".mov", ".m4v", ".ts", ".m2ts", ".wmv", ".flv"}
)

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_UPLOAD = 3

# Multi-GPU dispatch threshold (plan §9.3 as-built): at this many detected
# CUDA cards the batch driver switches to spawned per-GPU workers; below it
# the sequential v1 driver runs untouched (zero-regression contract).
_FANOUT_MIN_GPUS = 2


def _make_client_factory(config: Config) -> Callable[[], GofileClient]:
    """Build the gofile client factory from the resolved config (wiring helper)."""

    def client_factory() -> GofileClient:
        return GofileClient(
            account_token=config.upload.gofile.token,
            user_agent=config.fetch.gofile.user_agent,
            language=config.fetch.gofile.language,
            wt_salt=config.fetch.gofile.wt_salt,
            strict_tls=config.network.strict_tls,
        )

    return client_factory


def build_parser() -> argparse.ArgumentParser:
    """Build the root argument parser (flags per plan §3 + models subcommand)."""
    parser = argparse.ArgumentParser(
        prog="anime-pipeline",
        description="Аниме-пайплайн: descale → deband → upscale → interpolate → encode.",
    )
    parser.add_argument("--profile", help="Профиль из config.toml (например interp-only).")
    parser.add_argument(
        "--stages",
        help="Подмножество стадий через запятую (порядок соблюдается автоматически).",
    )
    parser.add_argument(
        "-i",
        dest="inputs",
        action="append",
        metavar="ВХОД",
        help="Вход: имя из input/, ссылка https:// или gofile.io/d/<id> (повторяемо — батч).",
    )
    parser.add_argument("-o", dest="output", metavar="ИМЯ", help="Имя результата в output/ (.mkv).")
    parser.add_argument(
        "--config", default="config.toml", help="Путь к config.toml (по умолчанию ./config.toml)."
    )
    parser.add_argument("--target-fps", help="Целевой fps: 60, 23.976 или 60000/1001.")
    parser.add_argument("--descale-height", type=int, help="Целевая высота стадии descale.")
    parser.add_argument(
        "--num-streams",
        type=int,
        metavar="N",
        help=(
            "Глубина префетча VapourSynth (кадров в полёте на GPU). НЕ управляет "
            "параллелизмом GPU-инференса: vsrife его не предоставляет. "
            "Приоритет: флаг > [hardware] > замер > формула."
        ),
    )
    parser.add_argument(
        "--upload", action="store_true", help="Загрузить результат на gofile.io после прогона."
    )
    parser.add_argument("--no-color", action="store_true", help="Отключить цветной вывод.")
    parser.add_argument("--no-tty", action="store_true", help="Простой вывод без анимации и ANSI.")
    sub = parser.add_subparsers(dest="command")
    models = sub.add_parser("models", help="Управление моделями из models/manifest.toml.")
    models.add_argument(
        "action", choices=["fetch", "verify"], help="fetch — скачать; verify — проверить sha256."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: parse, dispatch (models | batch), map errors to exit codes."""
    enable_windows_console_vt()
    parser = build_parser()
    args = parser.parse_args(argv)
    reporter = Reporter(resolve_mode(no_color=args.no_color, no_tty=args.no_tty))
    try:
        if args.command == "models":
            return _run_models(args, reporter)
        return _run_batch(args, reporter)
    except (
        ConfigError,
        EntryError,
        GofileError,
        ToolError,
        HardwareError,
        ManifestError,
        FetchError,
        UploadError,
    ) as exc:
        reporter.error(str(exc))
        return EXIT_FAILURE
    except RuntimeError as exc:  # stage-level loud failures
        reporter.error(str(exc))
        return EXIT_FAILURE
    except OSError as exc:
        # Safety net for pre-flight I/O that escaped specific wrapping
        # (requests.RequestException is an OSError subclass): the loud-error
        # contract forbids raw tracebacks on any documented failure path.
        reporter.error(str(exc))
        return EXIT_FAILURE
    except KeyboardInterrupt:
        reporter.warn("Прервано пользователем.")
        return 130


def _run_models(args: argparse.Namespace, reporter: Reporter) -> int:
    from anime_upscale.models.fetch import fetch_all

    root = Path.cwd()
    config = load_config(root / args.config)
    manifest_path = root / config.models.manifest
    models_dir = root / config.models.dir
    if args.action == "fetch":
        fetch_all(manifest_path, models_dir, reporter)
        return EXIT_OK
    verified = verify_all(manifest_path, models_dir)
    for name, path in verified.items():
        reporter.success(f"{name}: {path}")
    reporter.line(f"Проверено моделей: {len(verified)}.")
    return EXIT_OK


def _default_inputs(input_dir: Path) -> list[str]:
    """Apply §3: -i omitted → input/ must contain exactly one media file."""
    if not input_dir.is_dir():
        msg = f"Каталог {input_dir} не существует."
        raise EntryError(msg)
    files = sorted(
        p.name for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in MEDIA_EXTENSIONS
    )
    if len(files) == 1:
        return [files[0]]
    if not files:
        msg = (
            "Каталог input/ пуст. Положите файл в input/ "
            "или укажите -i (имя / URL / ссылка gofile)."
        )
        raise EntryError(msg)
    listing = ", ".join(files)
    msg = (
        f"В input/ несколько файлов ({listing}) — "
        "укажите -i явно (батч по каталогу не входит в v1)."
    )
    raise EntryError(msg)


def _update_album_tracking(
    upload_info: tuple[str, str | None] | None,
    current_folder_id: str | None,
    current_album_link: str | None,
    mode: str,
) -> tuple[str | None, str | None]:
    """Track first parentFolder and downloadPage link for batch album mode."""
    if upload_info is None or mode != "album":
        return current_folder_id, current_album_link
    link, parent_folder = upload_info
    folder_id = current_folder_id if current_folder_id is not None else parent_folder
    album_link = current_album_link if current_album_link is not None else link
    return folder_id, album_link


def _resolve_batch_inputs(
    args: argparse.Namespace,
    config: Config,
    root: Path,
) -> tuple[RunConfig, HardwareProfile, str, list[str]]:
    """Resolve stages, run config, hardware detection and input list."""
    stages: tuple[str, ...] | None = None
    if args.stages:
        stages = tuple(s.strip() for s in args.stages.split(",") if s.strip())
    target_fps = parse_fps_argument(args.target_fps) if args.target_fps else None
    run = resolve_run(
        config,
        profile=args.profile,
        stages=stages,
        target_fps=target_fps,
        descale_height=args.descale_height,
        upload=True if args.upload else None,
    )
    nn_selected = bool({"upscale", "interpolate"} & set(run.stages))
    hardware = detect_hardware(
        config.hardware, cli_num_streams=args.num_streams, nn_stages_selected=nn_selected
    )
    ffmpeg_path = resolve_tool("ffmpeg", config.tools.ffmpeg)
    entries = args.inputs or _default_inputs(root / "input")
    return run, hardware, ffmpeg_path, entries


def _run_batch(args: argparse.Namespace, reporter: Reporter) -> int:
    root = Path.cwd()
    config = load_config(root / args.config)
    run, hardware, ffmpeg_path, entries = _resolve_batch_inputs(args, config, root)
    print_banner(reporter, hardware, run.stages)

    input_dir = root / "input"
    output_dir = root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-flight (§11.5): queue resolution incl. gofile folder expansion (metadata only).
    client_factory = _make_client_factory(config)
    queue = build_queue(entries, client_factory, reporter, strict_tls=config.network.strict_tls)
    if "upscale" in run.stages:
        verify_all(root / config.models.manifest, root / config.models.dir)
    if "interpolate" in run.stages:
        # E01/E02 recurrence guard: vsrife's run-time auto-download from github
        # is unreliable under TLS interception — the seed must be pre-placed by
        # `models fetch` (verified sha256) before any batch work starts.
        ensure_vsrife_seed(root / config.models.manifest)
    output_override = _resolve_output_override(args.output, len(queue))

    # Multi-GPU dispatch (plan §9.3 as-built): two or more visible CUDA cards
    # engage fan-out — one spawned worker per GPU over a shared dynamic queue.
    # One card or a CPU-only box keeps the exact sequential v1 driver below
    # (zero-regression contract, pinned by tests/test_batch_fanout.py).
    if hardware.gpu_count >= _FANOUT_MIN_GPUS:
        return _run_batch_fanout(
            queue=queue,
            hardware=hardware,
            config=config,
            run=run,
            ffmpeg_path=ffmpeg_path,
            output_override=output_override,
            root=root,
            reporter=reporter,
            num_streams=args.num_streams,
            no_color=args.no_color,
            no_tty=args.no_tty,
        )

    ctx = _stage_context(config, run, hardware, root, reporter)
    return _run_batch_sequential(
        ctx=ctx,
        queue=queue,
        config=config,
        run=run,
        ffmpeg_path=ffmpeg_path,
        output_override=output_override,
        input_dir=input_dir,
        output_dir=output_dir,
        client_factory=client_factory,
        reporter=reporter,
    )


def _run_batch_sequential(  # noqa: PLR0913 - driver wiring bundle
    *,
    ctx: StageContext,
    queue: list[QueueItem],
    config: Config,
    run: RunConfig,
    ffmpeg_path: str,
    output_override: str | None,
    input_dir: Path,
    output_dir: Path,
    client_factory: Callable[[], GofileClient],
    reporter: Reporter,
) -> int:
    """Run the strict sequential v1 driver: unchanged loop, continue-and-summarize."""
    results: list[tuple[str, bool, str]] = []
    failures = {"pipeline": False, "upload": False}
    batch_folder_id: str | None = None
    batch_album_link: str | None = None

    for index, item in enumerate(queue, start=1):
        # M1: an earlier entry's successful repair replaces the stale resolved
        # path, so later entries start directly on the repaired binary instead
        # of re-dying on the old build (one ladder per process).
        repaired = ffmpeg_repair.repaired_ffmpeg()
        if repaired is not None:
            ffmpeg_path = str(repaired)
        reporter.file_banner(index, len(queue), item.name)
        folder_for_file = batch_folder_id if config.upload.gofile.mode == "album" else None
        outcome, upload_info = _process_entry(
            ctx,
            item,
            input_dir,
            output_dir,
            client_factory,
            ffmpeg_path,
            output_override=output_override,
            upload_enabled=run.upload,
            auto_repair=not bool(config.tools.ffmpeg),
            folder_id=folder_for_file,
            delete_after_upload=config.upload.gofile.delete_after_upload,
        )
        if outcome is None:
            batch_folder_id, batch_album_link = _update_album_tracking(
                upload_info,
                batch_folder_id,
                batch_album_link,
                config.upload.gofile.mode,
            )
            # Per-entry phase terminator (live-bug fix 2026-10-29): the ✓ line
            # closes the file's download→processing→upload block so the next
            # banner can never read as if it stole the console mid-file.
            reporter.success(f"{item.name} — готово")
            results.append((item.name, True, ""))
            continue
        kind, detail = outcome
        failures[kind] = True
        # Live-bug fix 2026-10-29: report the failure AT THE POINT of failure,
        # not only in the end summary — a deferred-only ✗ made a dead entry
        # indistinguishable from a running one (the next banner + «Загрузка»
        # lines looked like the downloader fighting the interpolation). The
        # continue-and-summarize ruling is untouched: the batch goes on and
        # the summary still names every entry.
        reporter.error(f"{item.name}: {detail}")
        results.append((item.name, False, detail))
    reporter.batch_summary(results)
    if config.upload.gofile.mode == "album" and run.upload and batch_album_link is not None:
        reporter.link(batch_album_link)
    if failures["pipeline"]:
        return EXIT_FAILURE
    if failures["upload"]:
        return EXIT_UPLOAD
    return EXIT_OK


# --- Multi-GPU fan-out (plan §9.3 as-built; load-dist canon §1/§2/§4) --------
#
# Canon: episode-level distribution only — a clip is NEVER split across GPUs
# (vs-mlrt #145: interleaved frames disrupt RIFE temporal consistency).
# Process-per-GPU because torch/vsrife device selection is process-global
# state (VS prefetch would race per-branch pinning) and CUDA+fork is unsafe
# ("poison fork"); spawn is also the only portable start method on Windows
# (the owner's server) — Python 3.14 additionally switched the Linux default
# away from fork, so the context is always requested explicitly.
#
# TRT engine sharing (canon §2): every child reads the same
# [stages.interpolate] trt_cache_dir (default ~/.cache/vsrife); vsrife keys
# engines by GPU model, so a same-SKU farm (e.g. Kaggle 2×T4) shares ONE
# engine — the first child to need it builds it, the rest load it cached.
# Known limitation (documented, deliberately no invented write-guard): on the
# FIRST parallel run over brand-new cards two children can race the engine
# build and build it twice — safe, costs only time.
#
# Progress: children inherit the parent's stdout, so per-entry lines from
# concurrent workers can interleave on a TTY (accepted for Kaggle/Compose
# runs); the parent prints command-level progress (N/M entries completed) and
# the end-of-batch summary stays byte-identical to the sequential driver.
# No merge barriers, no per-child progress plumbing (simplicity over
# cleverness).


@dataclass(frozen=True)
class _FanoutSpec:
    """Spawn-safe worker wiring bundle (plain picklable data only)."""

    config: Config
    run: RunConfig
    root: Path
    ffmpeg_path: str
    output_override: str | None
    upload_enabled: bool
    auto_repair: bool
    delete_after_upload: bool
    total: int
    num_streams: int | None
    no_color: bool
    no_tty: bool


@dataclass(frozen=True)
class _FanoutTask:
    """One unit of fan-out work: 1-based queue ordinal, entry, album folder."""

    ordinal: int
    item: QueueItem
    folder_id: str | None


@dataclass(frozen=True)
class _FanoutResult:
    """One processed entry, reported back for the parent's batch summary."""

    ordinal: int
    name: str
    # None = success; otherwise "pipeline" | "upload" (same ranking as the
    # sequential driver's failure classification for exit codes).
    failure_kind: str | None
    detail: str
    upload_info: tuple[str, str | None] | None


def _pin_visible_gpu(gpu_index: int) -> None:
    """
    Belt-and-braces child-side GPU pin (driver-level, not config delivery).

    Called at the very start of the worker, BEFORE any torch import: the CUDA
    runtime reads CUDA_VISIBLE_DEVICES at init time, so the spawned child sees
    exactly one card and its own detect_hardware reports it at cuda:0.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)


def _fanout_worker(
    gpu_index: int,
    task_queue: multiprocessing.Queue[_FanoutTask | None],
    result_queue: multiprocessing.Queue[_FanoutResult],
    spec: _FanoutSpec,
) -> None:
    """
    One episode-processing worker per GPU (spawn child entry point).

    Module-top-level by multiprocessing requirement (picklable target). Runs
    the FULL ``_process_entry`` chain (fetch → pipeline → upload) for entries
    pulled from the shared dynamic queue: whoever finishes first takes the
    next one — fast cards pull more, no round-robin. Exactly one
    ``_FanoutResult`` is pushed per entry (every Python-level escape is
    classified as a pipeline failure so the parent's collector can never hang
    on a raised worker); a hard OS-level kill of the worker process remains
    the same hazard class as any multiprocessing consumer.

    Hardware: the child re-runs detect_hardware itself — with the pin from
    ``_pin_visible_gpu`` the probe sees exactly one card at index 0, so
    ``interpolate``/``upscale`` keep their unchanged ``gpus[0]`` device
    resolution (cuda:0 == the pinned physical card).
    """
    _pin_visible_gpu(gpu_index)
    reporter = Reporter(resolve_mode(no_color=spec.no_color, no_tty=spec.no_tty))
    input_dir = spec.root / "input"
    output_dir = spec.root / "output"
    nn_stages_selected = bool({"upscale", "interpolate"} & set(spec.run.stages))
    hardware = detect_hardware(
        spec.config.hardware,
        cli_num_streams=spec.num_streams,
        nn_stages_selected=nn_stages_selected,
    )
    ctx = _stage_context(spec.config, spec.run, hardware, spec.root, reporter)

    def client_factory() -> GofileClient:
        return GofileClient(
            account_token=spec.config.upload.gofile.token,
            user_agent=spec.config.fetch.gofile.user_agent,
            language=spec.config.fetch.gofile.language,
            wt_salt=spec.config.fetch.gofile.wt_salt,
            strict_tls=spec.config.network.strict_tls,
        )

    ffmpeg_path = spec.ffmpeg_path
    while True:
        task = task_queue.get()
        if task is None:  # sentinel: the queue is fully drained
            return
        # M1 (sequential-driver parity): an earlier entry's successful repair
        # replaces the stale resolved path for later entries in this worker.
        repaired = ffmpeg_repair.repaired_ffmpeg()
        if repaired is not None:
            ffmpeg_path = str(repaired)
        reporter.file_banner(task.ordinal, spec.total, task.item.name)
        try:
            outcome, upload_info = _process_entry(
                ctx,
                task.item,
                input_dir,
                output_dir,
                client_factory,
                ffmpeg_path,
                output_override=spec.output_override,
                upload_enabled=spec.upload_enabled,
                auto_repair=spec.auto_repair,
                folder_id=task.folder_id,
                delete_after_upload=spec.delete_after_upload,
            )
        except Exception as exc:  # noqa: BLE001 - see the docstring hang guard
            outcome, upload_info = ("pipeline", str(exc)), None
        if outcome is None:
            reporter.success(f"{task.item.name} — готово")
            result_queue.put(_FanoutResult(task.ordinal, task.item.name, None, "", upload_info))
        else:
            kind, detail = outcome
            reporter.error(f"{task.item.name}: {detail}")
            result_queue.put(_FanoutResult(task.ordinal, task.item.name, kind, detail, None))


def _run_batch_fanout(  # noqa: PLR0913 - driver wiring bundle
    *,
    queue: list[QueueItem],
    hardware: HardwareProfile,
    config: Config,
    run: RunConfig,
    ffmpeg_path: str,
    output_override: str | None,
    root: Path,
    reporter: Reporter,
    num_streams: int | None,
    no_color: bool,
    no_tty: bool,
) -> int:
    """
    Dispatch mode: one spawned worker per GPU over a shared dynamic queue.

    Canon (load-dist §1/§4): episode-level tasks, process-per-GPU, explicit
    spawn context (Windows-mandatory; CUDA fork-unsafe). Children pin their
    card via CUDA_VISIBLE_DEVICES and re-detect hardware (one visible device
    at cuda:0), so stage device resolution is untouched. The feed/collect
    policy lives in ``_drive_fanout_queue``; the summary tail below is the
    sequential driver's, byte-for-byte (results re-sorted by queue ordinal,
    pipeline failure outranks upload failure). KeyboardInterrupt in the
    parent (Ctrl-C) returns 130; the daemonized workers are torn down with it.
    """
    gpu_indexes = [gpu.index for gpu in hardware.gpus]
    usable = usable_cpu_count(reporter)
    lanes = max(1, usable // len(gpu_indexes))
    # L2 contract untouched: active_encoders stays per-child = 1; this note is
    # informational only (the encoder-lane budget a fan-out run divides into).
    reporter.note(
        f"Мульти-GPU fan-out: {len(gpu_indexes)} процесса-воркера (по одному на GPU), "
        "эпизоды из общей динамической очереди. Бюджет CPU: "
        f"{usable} ядер(а) / {len(gpu_indexes)} воркера ≈ "
        f"{lanes} поток(ов) на энкодер-полосу "
        "(информационно — каждый воркер кодирует одну серию)."
    )

    mp_ctx = multiprocessing.get_context("spawn")
    task_queue: multiprocessing.Queue[_FanoutTask | None] = mp_ctx.Queue()
    result_queue: multiprocessing.Queue[_FanoutResult] = mp_ctx.Queue()
    spec = _FanoutSpec(
        config=config,
        run=run,
        root=root,
        ffmpeg_path=ffmpeg_path,
        output_override=output_override,
        upload_enabled=run.upload,
        auto_repair=not bool(config.tools.ffmpeg),
        delete_after_upload=config.upload.gofile.delete_after_upload,
        total=len(queue),
        num_streams=num_streams,
        no_color=no_color,
        no_tty=no_tty,
    )
    workers = [
        mp_ctx.Process(
            target=_fanout_worker,
            args=(gpu_index, task_queue, result_queue, spec),
            daemon=True,
        )
        for gpu_index in gpu_indexes
    ]
    for worker in workers:
        worker.start()

    results, failures, _batch_folder_id, batch_album_link = _drive_fanout_queue(
        queue=queue,
        task_queue=task_queue,
        result_queue=result_queue,
        workers=workers,
        upload_mode=config.upload.gofile.mode,
        reporter=reporter,
        album_hold=config.upload.gofile.mode == "album" and run.upload,
    )

    for worker in workers:
        worker.join()

    results.sort(key=lambda entry: entry.ordinal)
    reporter.batch_summary(
        [(entry.name, entry.failure_kind is None, entry.detail) for entry in results]
    )
    if config.upload.gofile.mode == "album" and run.upload and batch_album_link is not None:
        reporter.link(batch_album_link)
    if failures["pipeline"]:
        return EXIT_FAILURE
    if failures["upload"]:
        return EXIT_UPLOAD
    return EXIT_OK


def _drive_fanout_queue(  # noqa: PLR0913 - driver wiring bundle
    *,
    queue: list[QueueItem],
    task_queue: multiprocessing.Queue[_FanoutTask | None],
    result_queue: multiprocessing.Queue[_FanoutResult],
    workers: Sequence[multiprocessing.process.BaseProcess],
    upload_mode: str,
    reporter: Reporter,
    album_hold: bool,
) -> tuple[list[_FanoutResult], dict[str, bool], str | None, str | None]:
    """
    Feed the task queue and collect one result per entry (the parent's loop).

    Feeding: without album chaining the whole queue is pre-loaded up front
    (depth = N ≥ 2×GPU for real batches — the canon idle-prevention rule);
    workers pull dynamically, so a fast card takes more entries. In album
    mode with upload enabled the head entry is fed alone and the rest are
    released only after its upload result sets the album folder (or leaves
    it unset — the exact chain the sequential driver produces), so every
    entry still lands in ONE album. Sentinels (one per worker) follow the
    last task; workers exit after the drain.

    Returns (results in arrival order, failure flags, album folder, album link).
    """
    batch_folder_id: str | None = None
    batch_album_link: str | None = None
    results: list[_FanoutResult] = []
    failures = {"pipeline": False, "upload": False}
    released = not album_hold

    def feed_sentinels() -> None:
        """One exit sentinel per worker, ordered AFTER every task (FIFO drain)."""
        for _ in workers:
            task_queue.put(None)

    if released:
        for ordinal, item in enumerate(queue, start=1):
            task_queue.put(_FanoutTask(ordinal, item, None))
        feed_sentinels()
    else:
        # Album mode + upload: the head entry alone decides the album folder;
        # entries 2..N (plus sentinels) are released after its result arrives.
        task_queue.put(_FanoutTask(1, queue[0], None))

    collected = 0
    while collected < len(queue):
        result = result_queue.get()
        collected += 1
        if result.failure_kind is None:
            batch_folder_id, batch_album_link = _update_album_tracking(
                result.upload_info,
                batch_folder_id,
                batch_album_link,
                upload_mode,
            )
        else:
            failures[result.failure_kind] = True
        results.append(result)
        reporter.note(f"Fan-out: завершено {collected}/{len(queue)}")
        if not released:
            released = True
            # The head entry decided the album state (folder set, or its
            # upload/pipeline failed → later entries chain None, exactly like
            # the sequential driver's first-wins tracking).
            for ordinal, item in enumerate(queue[1:], start=2):
                task_queue.put(_FanoutTask(ordinal, item, batch_folder_id))
            feed_sentinels()
    return results, failures, batch_folder_id, batch_album_link


def _resolve_output_override(output: str | None, queue_len: int) -> str | None:
    """Validate -o: bare name only, .mkv forced; ambiguous with a batch = loud error."""
    if output is None:
        return None
    if "/" in output or "\\" in output or ".." in output:
        msg = f"Некорректное имя вывода: «{output}» — допустимы только имена файлов из output/."
        raise EntryError(msg)
    if queue_len > 1:
        msg = (
            f"-o задан для батча из {queue_len} файлов — имя вывода неоднозначно. "
            "Либо один -i/-o, либо omit -o (stem каждого файла)."
        )
        raise EntryError(msg)
    return output if output.endswith(".mkv") else f"{output}.mkv"


def _delete_uploaded_output(path: Path, name: str, reporter: Reporter) -> None:
    """
    Delete an uploaded output file; cleanup must never reclassify the upload.

    Windows AV/indexer locks make unlink fail transiently right after upload:
    retry once after 0.5 s (handles are usually released by then), then warn
    and leave the file in output/. The entry stays successful either way.
    """
    try:
        path.unlink()
    except OSError:
        time.sleep(0.5)
        try:
            path.unlink()
        except OSError as exc:
            reporter.warn(f"Файл {name} не удалён: {exc} — остался в output/")
            return
    reporter.note(f"Файл {name} удалён из output/ после выгрузки")


def _process_entry(  # noqa: PLR0913, PLR0917 - driver wiring bundle
    ctx: StageContext,
    item: QueueItem,
    input_dir: Path,
    output_dir: Path,
    client_factory: Callable[[], GofileClient],
    ffmpeg_path: str,
    *,
    output_override: str | None = None,
    upload_enabled: bool,
    auto_repair: bool,
    folder_id: str | None = None,
    delete_after_upload: bool = False,
) -> tuple[tuple[str, str] | None, tuple[str, str | None] | None]:
    """
    Run one batch entry (fetch → pipeline → optional upload).

    Returns (outcome, upload_info):
    - outcome: None on success, or ("pipeline" | "upload", detail) on failure
    - upload_info: (link, parent_folder_id) if uploaded, else None
    """
    reporter = ctx.reporter
    upload_info: tuple[str, str | None] | None = None
    try:
        # Lazy VS/torch stack (M7): resolved at call time as a MODULE so tests
        # can patch anime_upscale.pipeline.run_pipeline and the attribute
        # lookup at the call site sees the patch. The import sits INSIDE the
        # guarded region (PR1-review Minor): on a VS-less machine the
        # ImportError surfaces as the loud Russian error below, not a raw
        # ModuleNotFoundError traceback.
        from anime_upscale import pipeline as pipeline_mod

        source_path = fetch_queue_item(item, input_dir, client_factory, reporter)
        output_name = output_override or f"{Path(item.name).stem}.mkv"
        output_path = output_dir / output_name
        if output_path.exists():
            reporter.warn(f"Файл существует, будет перезаписан: output/{output_name}")

        # Smooth live TQDM progress bar with Russian ETA and fps rate.
        with reporter.progress_bar(label=f"[{item.name}] Обработка") as pbar:
            pipeline_mod.run_pipeline(
                ctx,
                source_path,
                output_path,
                ffmpeg_path=ffmpeg_path,
                progress_update=pbar.callback(),
                auto_repair=auto_repair,
            )
        if upload_enabled:
            total_size = output_path.stat().st_size if output_path.exists() else 0
            with reporter.upload_progress_bar(total=total_size, filename=item.name) as upbar:
                upload_info = upload_output(
                    output_path,
                    client_factory(),
                    reporter,
                    progress=upbar.callback(),
                    folder_id=folder_id,
                )
            if delete_after_upload:
                # F2 (adversarial review): the unlink sits in its own guard so
                # an OSError can never reach the pipeline-error handler below
                # and flip a successful upload into a pipeline failure.
                _delete_uploaded_output(output_path, output_name, reporter)
    except ImportError as exc:
        # Must precede the pipeline_mod tuple below: with the import itself
        # failed, pipeline_mod is unbound and evaluating that tuple would
        # raise NameError over the real cause.
        msg = f"VapourSynth не найден — установите зависимости (`uv sync`) ({exc})."
        return ("pipeline", msg), None
    except UploadError as exc:
        return ("upload", f"пайплайн ok, загрузка не удалась: {exc}"), None
    except (pipeline_mod.PipelineError, EntryError, GofileError, RuntimeError, OSError) as exc:
        return ("pipeline", str(exc)), None
    return None, upload_info


def _stage_context(
    config: Config,
    run: RunConfig,
    hardware: HardwareProfile,
    root: Path,
    reporter: Reporter,
) -> StageContext:
    """Assemble the read-only stage context (plan §2 registry contract)."""
    from anime_upscale.stages import StageContext

    return StageContext(
        run=run,
        hardware=hardware,
        project_root=root,
        reporter=reporter,
        models_dir=root / config.models.dir,
        models_manifest=root / config.models.manifest,
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
