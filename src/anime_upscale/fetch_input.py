"""
Input-entry resolution (plan §11.1; addendum 1: gofile folder expansion).

Entry grammar: bare file name | https:// direct URL | gofile.io/d/<id> share
link. Folder links expand IN PLACE (recursive children walk, deterministic
order: folders first, then files, each alphabetically; video whitelist filter
with printed skips). Bare names are never renamed; download collisions
disambiguate as <stem>-2<ext> with a printed note (plan §11.6).
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests
from requests.structures import CaseInsensitiveDict

from anime_upscale.gofile import GOFILE_SHARE_RE, GofileClient, PinnedSession
from anime_upscale.reporting import Reporter


class EntryError(Exception):
    """Loud input-entry failure (Russian message)."""


@dataclass(frozen=True)
class BareName:
    """A bare file name resolved inside input/ (must already exist)."""

    value: str


@dataclass(frozen=True)
class DirectUrl:
    """A direct http(s) download URL."""

    url: str


@dataclass(frozen=True)
class GofileLink:
    """A gofile.io/d/<id> share link (single file or folder → addendum 1)."""

    content_id: str


def classify_entry(entry: str) -> BareName | DirectUrl | GofileLink:
    """§11.1 entry grammar; other gofile URL shapes are loud errors."""
    text = entry.strip()
    if text.startswith(("http://", "https://")):
        host = (urlparse(text).hostname or "").removeprefix("www.")
        if host == "gofile.io":
            match = GOFILE_SHARE_RE.search(text)
            if not match:
                msg = (
                    f"Непонятная форма ссылки gofile: «{entry}». Поддерживаются только "
                    "ссылки вида gofile.io/d/<id>."
                )
                raise EntryError(msg)
            return GofileLink(content_id=match.group(1))
        return DirectUrl(url=text)
    if "/" in text or "\\" in text or ".." in text:
        msg = (
            f"Некорректное имя входного файла: «{entry}» — допустимы только простые имена "
            "файлов из каталога input/ (без разделителей путей)."
        )
        raise EntryError(msg)
    return BareName(value=text)


def derive_filename(response_headers: CaseInsensitiveDict[str], url: str) -> str:
    """Content-Disposition (filename* → filename) → last URL path segment → loud error."""
    disposition = response_headers.get("Content-Disposition", "")
    if disposition:
        star = _rfc5987_filename(disposition)
        if star:
            return star
        plain = _plain_filename(disposition)
        if plain:
            return plain
    path = urlparse(url).path.rstrip("/")
    segment = unquote(path.rsplit("/", 1)[-1]) if path else ""
    if segment:
        return segment
    msg = f"Невозможно определить имя файла для {url}: нет Content-Disposition и пустой путь."
    raise EntryError(msg)


def _rfc5987_filename(disposition: str) -> str:
    for raw_part in disposition.split(";"):
        part = raw_part.strip()
        if part.lower().startswith("filename*="):
            value = part.split("=", 1)[1].strip().strip('"')
            if "''" in value:  # charset'lang'percent-encoded
                _, _, encoded = value.partition("''")
                return unquote(encoded)
            return unquote(value)
    return ""


def _plain_filename(disposition: str) -> str:
    for raw_part in disposition.split(";"):
        part = raw_part.strip()
        if part.lower().startswith("filename="):
            return part.split("=", 1)[1].strip().strip('"')
    return ""


def disambiguate(name: str, taken: set[str], reporter: Reporter) -> str:
    """Resolve §11.6 collisions: <stem>-2<ext>, <stem>-3<ext>, … + loud note."""
    if name not in taken:
        return name
    stem, dot, ext = name.rpartition(".")
    counter = 2
    while True:
        candidate = f"{stem}-{counter}{dot}{ext}" if dot and stem else f"{name}-{counter}"
        if candidate not in taken:
            reporter.warn(
                f"Конфликт имён: «{name}» уже занят в очереди — "
                f"файл будет сохранён как «{candidate}»."
            )
            return candidate
        counter += 1


@dataclass
class QueueItem:
    """One fully-resolved batch entry (addendum 1: folders expand in place)."""

    source: str  # original CLI entry text
    origin: str  # human-readable origin for the summary (e.g. "gofile folder → child")
    name: str  # file name inside input/
    kind: str  # "local" | "url" | "gofile"
    url: str = ""  # download URL for url/gofile kinds


def build_queue(
    entries: list[str],
    client_factory: Callable[[], GofileClient],
    reporter: Reporter,
    name_prober: Callable[[str], str] | None = None,
) -> list[QueueItem]:
    """
    Resolve all entries into a sequential queue (pre-flight metadata only).

    No file downloads here — those happen per-entry at each file's turn.
    `name_prober` is a test seam (default: live header probe).
    """
    prober = name_prober or _remote_name
    queue: list[QueueItem] = []
    taken: set[str] = set()
    seen_entries: set[str] = set()
    for entry in entries:
        text = entry.strip()
        if text in seen_entries:
            reporter.warn(f"Дублирующийся вход «{text}» — выполняется один раз.")
            continue
        seen_entries.add(text)
        classified = classify_entry(text)
        match classified:
            case BareName(value=name):
                if name in taken:
                    msg = (
                        f"Имя «{name}» уже стоит в очереди — локальные имена не переименовываются."
                    )
                    raise EntryError(msg)
                taken.add(name)
                queue.append(
                    QueueItem(source=text, origin="локальный файл", name=name, kind="local")
                )
            case DirectUrl(url=url):
                name = prober(url)
                name = disambiguate(name, taken, reporter)
                taken.add(name)
                queue.append(
                    QueueItem(source=text, origin="прямая ссылка", name=name, kind="url", url=url)
                )
            case GofileLink(content_id=content_id):
                client = client_factory()
                children = client.expand_entries(content_id, reporter)
                for child in children:
                    name = disambiguate(child.name, taken, reporter)
                    taken.add(name)
                    queue.append(
                        QueueItem(
                            source=text,
                            origin=f"gofile папка {content_id} → файл",
                            name=name,
                            kind="gofile",
                            url=child.link,
                        )
                    )
    return queue


_HTTP_OK = 200
_PERCENT = 100


def _remote_name(url: str) -> str:
    """Probe download headers for the filename; fall back to the URL path segment."""
    if urlparse(url).hostname == "gofile.io":
        msg = f"Ссылка gofile вне обработчика: {url}"
        raise EntryError(msg)
    session = PinnedSession()
    session.pin(urlparse(url).hostname or "")
    try:
        response = session.head(url, timeout=30, allow_redirects=True)
    except requests.RequestException as exc:
        msg = f"Недоступна прямая ссылка {url}: {exc}"
        raise EntryError(msg) from exc
    if response.status_code == _HTTP_OK:
        try:
            return derive_filename(response.headers, response.url or url)
        except EntryError:
            pass  # fall through to the URL-segment fallback below
    path = urlparse(url).path.rstrip("/")
    segment = unquote(path.rsplit("/", 1)[-1]) if path else ""
    if segment:
        return segment
    msg = f"Невозможно определить имя файла для {url}."
    raise EntryError(msg)


def ensure_local(name: str, input_dir: Path) -> Path:
    """Apply the bare-name rule: input/<name> must exist (loud error otherwise)."""
    path = input_dir / name
    if not path.is_file():
        msg = f"Файл не найден: {path}. Положите его в input/ или укажите URL/ссылку gofile."
        raise EntryError(msg)
    return path


def fetch_queue_item(
    item: QueueItem,
    input_dir: Path,
    client_factory: Callable[[], GofileClient],
    reporter: Reporter,
) -> Path:
    """Per-entry download step (idempotent skip; .part + atomic rename inside client)."""
    target = input_dir / item.name
    if target.is_file():
        reporter.note(f"Уже в input/: {item.name} — загрузка пропущена")
        return target
    if item.kind == "local":
        return ensure_local(item.name, input_dir)
    client = client_factory()
    input_dir.mkdir(parents=True, exist_ok=True)
    last_note = [0]

    def progress(done: int, total: int) -> None:
        if total <= 0:
            return
        percent = int(done * _PERCENT / total)
        if percent >= last_note[0] + 10 or percent == _PERCENT:  # ≤10% intervals (§10.4)
            last_note[0] = percent
            reporter.progress_bytes("Загрузка", done, total)

    client.download_file(item.url, target, progress=progress)
    return target
