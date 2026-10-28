"""models/manifest.toml parsing + sha256 verification (plan §6)."""

import hashlib
import importlib.util
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
    vsrife_seed: bool = False


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
            vsrife_seed=bool(table.get("vsrife_seed", False)),
        )
    return entries


def _vsrife_package_dir() -> Path:
    """
    Directory of the installed vsrife package (no import: find_spec only resolves).

    Platform-aware by construction — site-packages differs between Linux
    (.venv/lib/pythonX.Y/site-packages) and Windows (.venv/Lib/site-packages),
    and find_spec follows the ACTUAL environment.
    """
    spec = importlib.util.find_spec("vsrife")
    if spec is None or not spec.origin:
        msg = "Пакет vsrife не найден в окружении — кэш моделей RIFE недоступен."
        raise ManifestError(msg)
    return Path(spec.origin).resolve().parent


def _seed_filename(entry: ModelEntry) -> str:
    """Return the cache filename vsrife itself derives from the URL (basename)."""
    if not entry.url:
        msg = f"Модель-предзаполнение «{entry.name}» не имеет url в манифесте."
        raise ManifestError(msg)
    return entry.url.rsplit("/", 1)[-1]


def vsrife_seed_path(entry: ModelEntry) -> Path:
    """Exact file vsrife expects: <package>/models/<url-basename>."""
    return _vsrife_package_dir() / "models" / _seed_filename(entry)


def expected_model_path(models_dir: Path, entry: ModelEntry) -> Path:
    """
    Flat models/upscale/<member-basename> layout (plan §2/§6).

    The manifest's `member` carries the full path INSIDE the release zip
    (folder prefix); the local artifact layout stays flat. vsrife-seed entries
    resolve into the vsrife package cache instead.
    """
    if entry.vsrife_seed:
        return vsrife_seed_path(entry)
    filename = Path(entry.member).name if entry.member else f"{entry.name}.pth"
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
