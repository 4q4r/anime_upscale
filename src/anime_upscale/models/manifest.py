"""models/manifest.toml parsing + sha256 verification (plan §6)."""

import hashlib
import tomllib
from dataclasses import dataclass
from pathlib import Path


class ManifestError(Exception):
    """Loud manifest/model verification failure (Russian message)."""


@dataclass(frozen=True)
class ModelEntry:
    """One models/manifest.toml entry (pinned artifacts carry url/member/sha256)."""

    name: str
    url: str = ""
    member: str = ""
    sha256: str = ""
    license: str = ""
    source_note: str = ""


def load_manifest(manifest_path: Path) -> dict[str, ModelEntry]:
    """Parse models/manifest.toml; every table becomes a frozen ModelEntry."""
    if not manifest_path.is_file():
        msg = f"Манифест моделей не найден: {manifest_path}"
        raise ManifestError(msg)
    raw = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    models_raw = raw.get("models", {})
    if not isinstance(models_raw, dict) or not models_raw:
        msg = f"[models] в {manifest_path} пуст или отсутствует."
        raise ManifestError(msg)
    entries: dict[str, ModelEntry] = {}
    for name, table in models_raw.items():
        entries[name] = ModelEntry(
            name=name,
            url=str(table.get("url", "")),
            member=str(table.get("member", "")),
            sha256=str(table.get("sha256", "")).lower(),
            license=str(table.get("license", "")),
            source_note=str(table.get("source", "")),
        )
    return entries


def expected_model_path(models_dir: Path, entry: ModelEntry) -> Path:
    """models/upscale/<member-or-name>.pth layout (plan §2/§6)."""
    """models/upscale/<member-or-name>.pth layout (plan §2/§6)."""
    filename = entry.member or f"{entry.name}.pth"
    return models_dir / "upscale" / filename


def sha256_of(path: Path) -> str:
    """Streaming sha256 (1 MiB chunks) — models are small but stay bounded."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model(models_dir: Path, entry: ModelEntry) -> Path:
    """Recompute the checksum; loud Russian error on mismatch or absence."""
    if not entry.sha256:
        msg = f"Модель «{entry.name}» не закреплена sha256 в манифесте — проверка невозможна."
        raise ManifestError(msg)
    path = expected_model_path(models_dir, entry)
    if not path.is_file():
        msg = f"Модель «{entry.name}» не найдена: {path}. Выполните: anime-pipeline models fetch"
        raise ManifestError(msg)
    actual = sha256_of(path)
    if actual != entry.sha256:
        msg = (
            f"Контрольная сумма модели «{entry.name}» не совпадает:\n"
            f"  ожидалось: {entry.sha256}\n  фактически: {actual}\n"
            f"Файл: {path}. Удалите его и выполните `anime-pipeline models fetch`."
        )
        raise ManifestError(msg)
    return path


def resolve_model_path(
    manifest_path: Path,
    models_dir: Path,
    name: str,
    *,
    verify: bool = True,
) -> Path:
    """Resolve + verify a pinned model by manifest name (used by the upscale stage)."""
    entries = load_manifest(manifest_path)
    if name not in entries:
        msg = (
            f"Модель «{name}» отсутствует в {manifest_path}. "
            f"Записаны: {', '.join(sorted(entries))}."
        )
        raise ManifestError(msg)
    entry = entries[name]
    if verify:
        return verify_model(models_dir, entry)
    return expected_model_path(models_dir, entry)
