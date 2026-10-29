"""
Automatic ffmpeg repair via winget (zero-touch, Windows).

Triggered ONLY by the diagnosis verdicts that mean the current build cannot
encode the pipeline's format at all (libx265 absent / 10-bit unsupported /
even 8-bit dead). The verdict «10-бит работает — причина в другом» is a real
parameter problem — it stays a loud terminal failure with NO reinstall.

Ladder: best-effort `winget uninstall` of winget-known ffmpeg packages →
`winget install --id Gyan.FFmpeg` (official winget-pkgs manifest: zip/portable,
no elevation, per-user) → locate the new ffmpeg.exe (WinGet Links shim first,
then a recursive walk of WinGet Packages — the build_misc_plugin.bat WinLibs
g++ pattern) → re-run the SAME 10-bit probe → prepend the binary's dir to the
session PATH. The caller retries the failed encode exactly once.
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from anime_upscale.ffmpeg import probe_10bit
from anime_upscale.reporting import Reporter

# Official winget-pkgs manifest (installer.1.12.0, version 9.0.2):
# https://github.com/microsoft/winget-pkgs/blob/master/manifests/g/Gyan/FFmpeg/9.0.2/Gyan.FFmpeg.installer.yaml
# InstallerType: zip / NestedInstallerType: portable / PortableCommandAlias: ffmpeg
# (no ElevationRequirement key → no elevation; per-user %LOCALAPPDATA% layout).
_WINGET_ID = "Gyan.FFmpeg"
_WINGET_INSTALL_ARGS = [
    "winget",
    "install",
    "--id",
    _WINGET_ID,
    "-e",
    "--silent",
    "--accept-source-agreements",
    "--accept-package-agreements",
    "--disable-interactivity",
]
_WINGET_TIMEOUT_S = 900

# Diagnosis verdict substrings that warrant the reinstall ladder.
_REINSTALL_MARKERS = (
    "libx265 отсутствует в этой сборке ffmpeg",
    "10-бит (Main10) не поддерживается",
    "не работает даже 8-битное кодирование",
)

_MANUAL_LINKS = (
    "  • https://www.gyan.dev/ffmpeg/builds/ — пакет «release full»\n"
    "  • https://github.com/BtbN/FFmpeg-Builds/releases — сборка win64-gpl"
)

_WINGET_DOC = "https://learn.microsoft.com/windows/package-manager/winget/"

_LIST_TIMEOUT_S = 120


def verdict_wants_reinstall(report: str) -> bool:
    """Whether a diagnosis report describes a build that cannot encode at all."""
    return any(marker in report for marker in _REINSTALL_MARKERS)


def _winget_bin() -> str | None:
    """Resolve the winget executable via PATH; None when absent."""
    return shutil.which("winget")


def _decode(value: bytes | str | None) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value or ""


def list_winget_ffmpeg_ids(runner: Any) -> list[str]:  # noqa: ANN401 - subprocess.run-shaped
    """
    Return the IDs of winget-known ffmpeg packages (unattended-safe listing).

    `winget list ffmpeg`: a row qualifies when a token contains «ffmpeg», is
    dotted (real package IDs carry a publisher dot) and is not a Microsoft
    alias shim (e.g. Microsoft.FFmpegOverloads — an execution-alias provider,
    not a build). A non-zero listing exit (no matches) yields an empty list.
    """
    try:
        completed = runner(
            ["winget", "list", "ffmpeg"], capture_output=True, timeout=_LIST_TIMEOUT_S
        )
    except subprocess.TimeoutExpired:
        return []
    ids: list[str] = []
    for raw in _decode(completed.stdout).splitlines():
        line = raw.replace("\r", " ").strip()
        if not line or set(line) <= {"-"}:
            continue
        for token in line.split():
            lowered = token.lower()
            if (
                "ffmpeg" in lowered
                and "." in token
                and not lowered.startswith("microsoft.")
                and token not in ids
            ):
                ids.append(token)
    return ids


def uninstall_winget_ffmpeg_packages(runner: Any, reporter: Reporter) -> list[str]:  # noqa: ANN401
    """
    Best-effort uninstall of every winget-known ffmpeg package.

    Only winget-managed installs can be removed; a manual/unknown install gets
    the manual-removal warning (the new install will shadow it on PATH).
    """
    ids = list_winget_ffmpeg_ids(runner)
    if not ids:
        reporter.warn(
            "Текущий ffmpeg установлен вручную (winget о нём не знает) — "
            "будет перекрыт по PATH, рекомендуется удалить вручную."
        )
        return []
    for package_id in ids:
        try:
            completed = runner(
                [
                    "winget",
                    "uninstall",
                    "--id",
                    package_id,
                    "-e",
                    "--silent",
                    "--disable-interactivity",
                ],
                capture_output=True,
                timeout=_WINGET_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            reporter.warn(f"Не удалось удалить {package_id}: winget превысил таймаут — продолжаю.")
            continue
        if completed.returncode != 0:
            reporter.warn(
                f"Не удалось удалить {package_id} (код {completed.returncode}) — продолжаю."
            )
    return ids


def _winget_roots() -> tuple[Path, Path] | None:
    r"""(Links, Packages) under %LOCALAPPDATA%\Microsoft\WinGet, or None."""
    local = os.environ.get("LOCALAPPDATA")
    if not local:
        return None
    base = Path(local) / "Microsoft" / "WinGet"
    return base / "Links", base / "Packages"


def locate_winget_ffmpeg() -> Path | None:
    """
    Locate the winget-installed ffmpeg.exe.

    The per-user Links shim first, then a recursive walk of the Packages tree
    (the portable zip layout buries the binary a few levels deep) — the same
    proven pattern as the .bat WinLibs locator.
    """
    roots = _winget_roots()
    if roots is None:
        return None
    links, packages = roots
    shim = links / "ffmpeg.exe"
    if shim.is_file():
        return shim
    if packages.is_dir():
        for candidate in sorted(packages.rglob("ffmpeg.exe")):
            return candidate
    return None


def repair_ffmpeg_build(
    report: str,
    reporter: Reporter,
    runner: Any = None,  # noqa: ANN401 - subprocess.run-shaped
    *,
    override_pinned: bool = False,
) -> Path | None:
    """
    Run the reinstall ladder for a build that cannot encode.

    Returns the new ffmpeg path; None when the verdict does not warrant a
    reinstall. Terminal RuntimeError (no silent fallbacks): winget missing,
    install failed or the located binary still fails the 10-bit probe — always
    with the winget output tail and manual full-build links.
    """
    if not verdict_wants_reinstall(report):
        return None
    if override_pinned:
        msg = (
            "ffmpeg задан явно в конфиге ([tools] ffmpeg) — автоматическая "
            "переустановка не может исправить фиксированный путь. "
            "Очистите параметр [tools] ffmpeg в config.toml и перезапустите "
            "(установленная полная сборка будет найдена в PATH)."
        )
        raise RuntimeError(msg)
    if _winget_bin() is None:
        msg = (
            "winget не найден — автоматическая переустановка ffmpeg невозможна. "
            f"Установите winget: {_WINGET_DOC} "
            "либо установите полную сборку ffmpeg вручную:\n" + _MANUAL_LINKS
        )
        raise RuntimeError(msg)

    run = runner if runner is not None else subprocess.run
    reporter.line("[ЭТАП] ffmpeg не умеет 10-битный x265 — переустанавливаю через winget…")
    reporter.line("[ЭТАП] Удаляю текущий ffmpeg…")
    uninstall_winget_ffmpeg_packages(run, reporter)
    reporter.line("[ЭТАП] Устанавливаю Gyan.FFmpeg…")
    try:
        installed = run(_WINGET_INSTALL_ARGS, capture_output=True, timeout=_WINGET_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        msg = (
            f"winget install {_WINGET_ID} превысил таймаут "
            f"{_WINGET_TIMEOUT_S} с — установка не завершилась.\n"
            "Установите полную сборку ffmpeg вручную:\n" + _MANUAL_LINKS
        )
        raise RuntimeError(msg) from exc

    located = locate_winget_ffmpeg()
    if located is None:
        tail = _decode(installed.stderr).strip().splitlines()[-5:]
        tail_text = "\n".join(tail) if tail else "(пустой вывод winget)"
        msg = (
            f"Установка {_WINGET_ID} завершена (код {installed.returncode}), "
            "но ffmpeg.exe не найден ни в WinGet Links, ни в WinGet Packages.\n"
            f"Хвост вывода winget:\n{tail_text}\n"
            "Установите полную сборку ffmpeg вручную:\n" + _MANUAL_LINKS
        )
        raise RuntimeError(msg)

    reporter.line("[ЭТАП] Проверяю 10-битное кодирование…")
    ok, err = probe_10bit(str(located), run)
    if not ok:
        probe_error = err.splitlines()[-1] if err else "(пустой stderr)"
        msg = (
            f"Установленный {_WINGET_ID} не прошёл 10-битную пробу: {probe_error}\n"
            "Установите полную сборку ffmpeg вручную:\n" + _MANUAL_LINKS
        )
        raise RuntimeError(msg)

    links_dir = located.parent
    os.environ["PATH"] = str(links_dir) + os.pathsep + os.environ.get("PATH", "")
    reporter.success(f"✓ ffmpeg готов: {located}")
    return located
