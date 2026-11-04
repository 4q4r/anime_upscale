"""
Input-entry resolution (plan §11.1; addendum 1: gofile folder expansion).

Entry grammar: bare file name | https:// direct URL | gofile.io/d/<id> share
link. Folder links expand IN PLACE (recursive children walk, deterministic
order: folders first, then files, each alphabetically; video whitelist filter
with printed skips). Bare names are never renamed; download collisions
disambiguate as <stem>-2<ext> with a printed note (plan §11.6).
"""

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import requests
from requests.structures import CaseInsensitiveDict

from anime_upscale import gofile
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
    """
    §11.1 entry grammar; other gofile URL shapes are loud errors.

    Plain http:// entries are rejected: ingestion is https-only — the DoH
    pinning stance (§10.1) cannot protect an unencrypted channel from
    content substitution.
    """
    text = entry.strip()
    if text.startswith("http://"):
        msg = (
            f"Отказано: прямые ссылки только по HTTPS («{entry}»). "
            "HTTP-загрузка обходит DoH-пиннинг и защиту от подмены содержимого (§10.1)."
        )
        raise EntryError(msg)
    if text.startswith(("https://",)):
        parsed = urlparse(text)
        host = (parsed.hostname or "").removeprefix("www.")
        is_share_host = host == "gofile.io"
        if is_share_host or host.endswith(".gofile.io"):
            # gofile-family hosts are NOT generic direct URLs: §11.1 accepts ONLY
            # the canonical share link gofile.io/d/<id>. Every other shape —
            # store/api subdomains (even with a /d/ path), /upload, /contents,
            # bare ids — is a loud spec error, never a silent DirectUrl.
            match = GOFILE_SHARE_RE.match(parsed.path) if is_share_host else None
            if not match:
                msg = (
                    f"Непонятная форма ссылки gofile: «{entry}». Поддерживаются только "
                    "ссылки вида gofile.io/d/<id>; прочие формы (поддомены вроде "
                    "store.gofile.io, /upload, /contents, голый id) входными "
                    "ссылками не являются."
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


def sanitize_remote_filename(name: str, source: str) -> str:
    """
    One gate for EVERY remote-derived file name (addendum-grade security).

    Remote names must be flat filenames: path separators, `..` shapes, absolute
    forms and dotfile tricks are rejected with a loud Russian error naming the
    offending name and where it came from. Benign unicode/spaces pass through.
    """
    if not name:
        msg = f"Пустое имя файла из удалённого источника ({source})."
        raise EntryError(msg)
    if "/" in name or "\\" in name:
        msg = (
            f"Отказано в имени файла «{name}» из {source}: "
            "имена из удалённых источников должны быть простыми именами файлов "
            "(без разделителей путей)."
        )
        raise EntryError(msg)
    if ":" in name:
        # Windows drive-letter forms ("C:evil.mkv", "C:\path") escape input/ on
        # Windows; a colon is never valid in a flat cross-platform filename.
        msg = (
            f"Отказано в имени файла «{name}» из {source}: "
            "символ «:» недопустим в имени файла (формы дисков Windows)."
        )
        raise EntryError(msg)
    if name in {".", ".."} or name.startswith(".."):
        msg = (
            f"Отказано в имени файла «{name}» из {source}: "
            "имена вида «..» допускают выход за пределы input/."
        )
        raise EntryError(msg)
    if name.startswith("."):
        msg = (
            f"Отказано в имени файла «{name}» из {source}: "
            "скрытые (начинающиеся с точки) имена от удалённых серверов не принимаются."
        )
        raise EntryError(msg)
    return name


def _cd_candidate(headers: CaseInsensitiveDict[str]) -> tuple[str, str] | None:
    """(raw name, source label) when a Content-Disposition name is present."""
    disposition = headers.get("Content-Disposition", "")
    if disposition:
        star = _rfc5987_filename(disposition)
        if star:
            return star, "заголовок Content-Disposition (filename*)"
        plain = _plain_filename(disposition)
        if plain:
            return plain, "заголовок Content-Disposition (filename)"
    return None


def _url_segment_candidate(url: str) -> tuple[str, str] | None:
    """(raw name, source label) from the last URL path segment, if derivable."""
    path = urlparse(url).path.rstrip("/")
    segment = unquote(path.rsplit("/", 1)[-1]) if path else ""
    if segment:
        return segment, "последний сегмент пути URL"
    return None


def derive_filename(response_headers: CaseInsensitiveDict[str], url: str) -> str:
    """
    Content-Disposition (filename* → filename) → last URL path segment → loud error.

    Fix-round-2 contract: fall through from the header to the URL segment ONLY on
    ABSENCE of a header name; a sanitizer rejection of a present candidate is
    terminal (never swallowed into the next candidate). Every candidate is
    sanitized individually.
    """
    candidate = _cd_candidate(response_headers) or _url_segment_candidate(url)
    if candidate is None:
        msg = f"Невозможно определить имя файла для {url}: нет Content-Disposition и пустой путь."
        raise EntryError(msg)
    name, source = candidate
    return sanitize_remote_filename(name, source)


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
    expected_size: int | None = None
    expected_md5: str | None = None


def build_queue(
    entries: list[str],
    client_factory: Callable[[], GofileClient],
    reporter: Reporter,
    name_prober: Callable[[str], str] | None = None,
    *,
    strict_tls: bool = False,
) -> list[QueueItem]:
    """
    Resolve all entries into a sequential queue (pre-flight metadata only).

    No file downloads here — those happen per-entry at each file's turn.
    `name_prober` is a test seam (default: live header probe, threaded
    ``strict_tls`` for interception-adaptive TLS).
    """
    prober = name_prober or (lambda url: _remote_name(url, strict_tls=strict_tls))
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
                    safe_name = sanitize_remote_filename(
                        child.name, f"gofile дочерний элемент (папка {content_id})"
                    )
                    name = disambiguate(safe_name, taken, reporter)
                    taken.add(name)
                    queue.append(
                        QueueItem(
                            source=text,
                            origin=f"gofile папка {content_id} → файл",
                            name=name,
                            kind="gofile",
                            url=child.link,
                            expected_size=child.size,
                            expected_md5=child.md5,
                        )
                    )
    return queue


_HTTP_OK = 200
_PERCENT = 100


def _remote_name(url: str, *, strict_tls: bool = False) -> str:
    """Probe download headers for the filename; fall back to the URL path segment."""
    if urlparse(url).hostname == "gofile.io":
        msg = f"Ссылка gofile вне обработчика: {url}"
        raise EntryError(msg)
    session = PinnedSession()
    session.pin(urlparse(url).hostname or "")
    try:
        response = gofile.adaptive_tls_request(
            session.head,
            url=url,
            strict_tls=strict_tls,
            timeout=30,
            allow_redirects=True,
        )
    except requests.RequestException as exc:
        msg = f"Недоступна прямая ссылка {url}: {exc}"
        raise EntryError(msg) from exc
    if response.status_code == _HTTP_OK:
        # A sanitizer rejection inside derive_filename is TERMINAL — it must
        # never fall through to the raw URL segment (fix-round-2, item 1).
        return derive_filename(response.headers, response.url or url)
    # Headers absent (non-200 probe): the URL segment is the only candidate —
    # sanitized through the same gate.
    candidate = _url_segment_candidate(url)
    if candidate is None:
        msg = f"Невозможно определить имя файла для {url}."
        raise EntryError(msg)
    name, source = candidate
    return sanitize_remote_filename(name, source)


def ensure_local(name: str, input_dir: Path) -> Path:
    """Apply the bare-name rule: input/<name> must exist (loud error otherwise)."""
    path = input_dir / name
    if not path.is_file():
        msg = f"Файл не найден: {path}. Положите его в input/ или укажите URL/ссылку gofile."
        raise EntryError(msg)
    return path


def compute_file_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute hex-encoded MD5 digest of a local file."""
    md5 = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest().lower()


class FetchCache:
    """Persistent cache in input/.fetch_cache.json for downloaded remote inputs."""

    def __init__(self, cache_file: Path) -> None:
        """Initialize and load cache from file."""
        self.cache_file = cache_file
        self._entries: dict[str, dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        """Load cache entries from JSON file on disk."""
        if not self.cache_file.is_file():
            self._entries = {}
            return
        try:
            with self.cache_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    self._entries = {k: v for k, v in data.items() if isinstance(v, dict)}
                else:
                    self._entries = {}
        except (json.JSONDecodeError, OSError):
            self._entries = {}

    def save(self) -> None:
        """Save cache entries to JSON file on disk atomically."""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            tmp_file = self.cache_file.with_name(f"{self.cache_file.name}.tmp")
            with tmp_file.open("w", encoding="utf-8") as f:
                json.dump(self._entries, f, indent=2, ensure_ascii=False)
            tmp_file.replace(self.cache_file)
        except OSError:
            pass

    def get(self, filename: str) -> dict[str, Any] | None:
        """Retrieve cached metadata for filename."""
        return self._entries.get(filename)

    def record(  # noqa: PLR0913, PLR0917
        self,
        filename: str,
        source: str,
        size: int,
        md5: str | None,
        local_size: int,
        mtime: float,
    ) -> None:
        """Record or update entry in cache."""
        self._entries[filename] = {
            "source": source,
            "size": size,
            "md5": md5,
            "local_size": local_size,
            "mtime": mtime,
        }
        self.save()


def _verify_existing_remote(
    item: QueueItem,
    target: Path,
    cache: FetchCache,
    reporter: Reporter,
) -> bool:
    """Verify local file matches expected size and MD5, returning True if valid."""
    local_stat = target.stat()
    local_size = local_stat.st_size

    # 1. Size verification if expected_size is known
    if item.expected_size is not None and local_size != item.expected_size:
        reporter.warn(
            f"Файл {item.name} повреждён или недокачан "
            f"({local_size} байт вместо {item.expected_size}) — перекачиваю..."
        )
        target.unlink(missing_ok=True)
        return False

    # 2. MD5 verification if expected_md5 is known
    if item.expected_md5:
        cached = cache.get(item.name)
        expected_clean = item.expected_md5.lower().strip()
        if (
            cached
            and cached.get("mtime") == local_stat.st_mtime
            and cached.get("local_size") == local_size
            and cached.get("md5")
        ):
            local_md5 = str(cached["md5"]).lower().strip()
        else:
            local_md5 = compute_file_md5(target)
            cache.record(
                filename=item.name,
                source=item.source,
                size=item.expected_size if item.expected_size is not None else local_size,
                md5=local_md5,
                local_size=local_size,
                mtime=local_stat.st_mtime,
            )

        if local_md5 != expected_clean:
            reporter.warn(
                f"Файл {item.name} повреждён: хэш MD5 не совпадает "
                f"({local_md5} вместо {expected_clean}) — перекачиваю..."
            )
            target.unlink(missing_ok=True)
            return False

    reporter.note(f"Уже в input/ и актуален: {item.name} — загрузка пропущена")
    return True


def fetch_queue_item(
    item: QueueItem,
    input_dir: Path,
    client_factory: Callable[[], GofileClient],
    reporter: Reporter,
) -> Path:
    """Per-entry download step with smart cache, size/MD5 integrity and delta downloads."""
    target = input_dir / item.name
    cache = FetchCache(input_dir / ".fetch_cache.json")

    if target.is_file():
        if item.kind == "local":
            reporter.note(f"Уже в input/: {item.name} — загрузка пропущена")
            return target

        if _verify_existing_remote(item, target, cache, reporter):
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

    # After download, record to cache if target exists
    if target.is_file():
        stat = target.stat()
        local_md5 = compute_file_md5(target) if item.expected_md5 else None
        cache.record(
            filename=item.name,
            source=item.source,
            size=item.expected_size if item.expected_size is not None else stat.st_size,
            md5=local_md5 or item.expected_md5,
            local_size=stat.st_size,
            mtime=stat.st_mtime,
        )

    return target
