"""
Model fetch/verify: download + sha256-verify manifest artifacts (plan §6/§10 step 11).

AnimeJaNai release zips: download to temp, extract the pinned member, verify
sha256 BEFORE writing the final file (temp + atomic rename), loud mismatch.
"""

import zipfile
from pathlib import Path

import requests

from anime_upscale.models.manifest import (
    ManifestError,
    ModelEntry,
    expected_model_path,
    load_manifest,
    sha256_of,
)
from anime_upscale.reporting import Reporter


class FetchError(Exception):
    """Loud fetch failure (Russian message)."""


_HTTP_OK = 200


def _download(url: str, destination: Path) -> None:
    try:
        with requests.get(url, stream=True, timeout=(10, 300), allow_redirects=True) as response:
            if response.status_code != _HTTP_OK:
                msg = f"Не удалось скачать {url}: HTTP {response.status_code}."
                raise FetchError(msg)
            with destination.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    handle.write(chunk)
    except requests.RequestException as exc:
        msg = f"Ошибка сети при скачивании {url}: {exc}"
        raise FetchError(msg) from exc


def _reconcile_existing(target: Path, entry: ModelEntry, reporter: Reporter | None) -> bool:
    """Report whether the target is present+verified; mismatched files are re-fetched."""
    if not target.is_file():
        return False
    if sha256_of(target) == entry.sha256:
        if reporter is not None:
            reporter.note(f"Модель «{entry.name}» уже на месте и верифицирована: {target}")
        return True
    if reporter is not None:
        reporter.warn(f"Файл {target} существует, но sha256 не совпадает — перекачивается.")
    target.unlink()
    return False


def _extract_and_verify(archive_path: Path, entry: ModelEntry, target: Path) -> None:
    """Extract the pinned member, verify sha256 BEFORE the atomic rename."""
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if entry.member not in names:
            msg = f"В архиве нет члена «{entry.member}». Содержимое: {', '.join(names[:20])}"
            raise FetchError(msg)
        member_tmp = target.parent / f".{entry.name}.part"
        member_tmp.write_bytes(archive.read(entry.member))
    actual = sha256_of(member_tmp)
    if actual != entry.sha256:
        member_tmp.unlink(missing_ok=True)
        msg = (
            f"Контрольная сумма модели «{entry.name}» не совпадает:\n"
            f"  ожидалось: {entry.sha256}\n  фактически: {actual}"
        )
        raise FetchError(msg)
    member_tmp.rename(target)  # atomic: verified bytes only reach the final path


def fetch_model(
    entry: ModelEntry,
    manifest_path: Path,  # noqa: ARG001 - kept for CLI-call-site symmetry with verify_all
    models_dir: Path,
    reporter: Reporter | None = None,
) -> Path:
    """Download + verify one manifest entry; returns the final verified path."""
    if not entry.url or not entry.sha256:
        msg = (
            f"Модель «{entry.name}» не имеет url/sha256 в манифесте "
            "(wrapper-managed модели не скачиваются этим методом)."
        )
        raise FetchError(msg)
    target = expected_model_path(models_dir, entry)
    if _reconcile_existing(target, entry, reporter):
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    archive_tmp = target.parent / f".{entry.name}.zip.part"
    try:
        if reporter is not None:
            reporter.line(f"Скачивание: {entry.url}")
        _download(entry.url, archive_tmp)
        _extract_and_verify(archive_tmp, entry, target)
    except zipfile.BadZipFile as exc:
        msg = f"Повреждённый архив с моделью «{entry.name}»: {exc}"
        raise FetchError(msg) from exc
    finally:
        archive_tmp.unlink(missing_ok=True)
    if reporter is not None:
        reporter.success(f"Модель «{entry.name}» загружена и верифицирована: {target}")
    return target


def fetch_all(
    manifest_path: Path,
    models_dir: Path,
    reporter: Reporter | None = None,
) -> list[Path]:
    """Fetch every pinned entry that is missing or mismatched."""
    entries = load_manifest(manifest_path)
    fetched: list[Path] = []
    for entry in entries.values():
        if not entry.url or not entry.sha256:
            if reporter is not None and entry.source_note:
                reporter.note(f"«{entry.name}»: {entry.source_note}")
            continue
        fetched.append(fetch_model(entry, manifest_path, models_dir, reporter))
    return fetched


def verify_all(manifest_path: Path, models_dir: Path) -> dict[str, Path]:
    """Offline audit: recompute checksums for every pinned entry; loud on failure."""
    entries = load_manifest(manifest_path)
    verified: dict[str, Path] = {}
    for name, entry in entries.items():
        if not entry.sha256:
            continue
        verified[name] = expected_model_path(models_dir, entry)
        actual = sha256_of(verified[name]) if verified[name].is_file() else None
        if actual is None:
            msg = (
                f"Модель «{name}» не найдена: {verified[name]}. "
                "Выполните: anime-pipeline models fetch"
            )
            raise ManifestError(msg)
        if actual != entry.sha256:
            msg = f"Контрольная сумма модели «{name}» не совпадает: {actual} ≠ {entry.sha256}"
            raise ManifestError(msg)
    return verified
