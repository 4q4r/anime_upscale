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
import sys
from pathlib import Path
from typing import TYPE_CHECKING

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
from anime_upscale.hardware import HardwareError, HardwareProfile, detect_hardware, print_banner
from anime_upscale.models.fetch import FetchError, verify_all
from anime_upscale.models.manifest import ManifestError
from anime_upscale.reporting import (
    ProgressThrottle,
    Reporter,
    enable_windows_console_vt,
    resolve_mode,
)
from anime_upscale.tools import ToolError, resolve_tool
from anime_upscale.upload import UploadError, upload_output

if TYPE_CHECKING:
    from collections.abc import Callable

    from anime_upscale.stages import StageContext

MEDIA_EXTENSIONS = frozenset(
    {".mkv", ".mp4", ".avi", ".webm", ".mov", ".m4v", ".ts", ".m2ts", ".wmv", ".flv"}
)

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_UPLOAD = 3


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


def _run_batch(args: argparse.Namespace, reporter: Reporter) -> int:
    root = Path.cwd()
    config = load_config(root / args.config)
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
        config.hardware, cli_num_streams=None, nn_stages_selected=nn_selected
    )
    print_banner(reporter, hardware, run.stages)

    ffmpeg_path = resolve_tool("ffmpeg", config.tools.ffmpeg)
    input_dir = root / "input"
    output_dir = root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = args.inputs or _default_inputs(input_dir)

    # Pre-flight (§11.5): queue resolution incl. gofile folder expansion (metadata only).
    def client_factory() -> GofileClient:
        return GofileClient(
            account_token=config.upload.gofile.token,
            user_agent=config.fetch.gofile.user_agent,
            language=config.fetch.gofile.language,
            wt_salt=config.fetch.gofile.wt_salt,
        )

    queue = build_queue(entries, client_factory, reporter)
    if "upscale" in run.stages:
        verify_all(root / config.models.manifest, root / config.models.dir)
    output_override = _resolve_output_override(args.output, len(queue))

    ctx = _stage_context(config, run, hardware, root, reporter)
    results: list[tuple[str, bool, str]] = []
    failures = {"pipeline": False, "upload": False}
    for index, item in enumerate(queue, start=1):
        reporter.file_banner(index, len(queue), item.name)
        outcome = _process_entry(
            ctx,
            item,
            input_dir,
            output_dir,
            client_factory,
            ffmpeg_path,
            output_override=output_override,
            upload_enabled=run.upload,
        )
        if outcome is None:
            results.append((item.name, True, ""))
            continue
        kind, detail = outcome
        failures[kind] = True
        results.append((item.name, False, detail))
    reporter.batch_summary(results)
    if failures["pipeline"]:
        return EXIT_FAILURE
    if failures["upload"]:
        return EXIT_UPLOAD
    return EXIT_OK


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
) -> tuple[str, str] | None:
    """
    Run one batch entry (fetch → pipeline → optional upload).

    Returns None on success, or ("pipeline" | "upload", detail) on failure —
    the continue-and-summarize ruling keeps the batch alive either way.
    """
    # Lazy VS/torch stack (M7): resolved at call time as a MODULE so tests can
    # patch anime_upscale.pipeline.run_pipeline and the attribute lookup at the
    # call site sees the patch.
    from anime_upscale import pipeline as pipeline_mod

    reporter = ctx.reporter
    try:
        source_path = fetch_queue_item(item, input_dir, client_factory, reporter)
        output_name = output_override or f"{Path(item.name).stem}.mkv"
        output_path = output_dir / output_name
        if output_path.exists():
            reporter.warn(f"Файл существует, будет перезаписан: output/{output_name}")

        # ≤10%-interval progress (§10.4 extended to the pipeline): the VS callback
        # fires per frame — unthrottled printing would flood the console.
        progress = ProgressThrottle(reporter, label=f"[{item.name}] Обработка").callback()

        pipeline_mod.run_pipeline(
            ctx, source_path, output_path, ffmpeg_path=ffmpeg_path, progress_update=progress
        )
        if upload_enabled:
            upload_output(
                output_path,
                client_factory(),
                reporter,
                progress=lambda done, total: reporter.progress_bytes("Загрузка", done, total),
            )
    except UploadError as exc:
        return ("upload", f"пайплайн ok, загрузка не удалась: {exc}")
    except (pipeline_mod.PipelineError, EntryError, GofileError, RuntimeError, OSError) as exc:
        return ("pipeline", str(exc))
    return None


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
