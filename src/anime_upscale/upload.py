"""
Optional gofile.io upload post-step (plan §10; mechanics delegated to gofile.py).

Trigger: --upload > profile upload=true > off. Runs only after a completed
pipeline; failure is loud, the output file is untouched, one attempt in v1.
"""

import http.client
import os
import random
import time
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

import requests
import urllib3.exceptions

from anime_upscale.gofile import GofileClient, GofileError, adaptive_tls_request

if TYPE_CHECKING:
    from anime_upscale.reporting import Reporter

_HTTP_OK = 200

# Transient upload failures (live-batch wave): the same classes as the
# download ladder plus server-side blips (502/503/504) and rate-limit 429
# (Retry-After honored when present). urllib3 surfaces mid-stream breaks to
# requests as ConnectionError/ChunkedEncodingError; raw IncompleteRead and
# ProtocolError are kept in the net in case a layer skips the wrapping.
_TRANSIENT_UPLOAD_STATUSES = frozenset({429, 502, 503, 504})
_TRANSIENT_EXCEPTIONS: tuple[type[Exception], ...] = (
    requests.exceptions.ConnectionError,
    requests.exceptions.ChunkedEncodingError,
    http.client.IncompleteRead,
    urllib3.exceptions.ProtocolError,
)

# F11: the file part Content-Type follows the filename suffix (gofile accepts
# arbitrary video payloads; unknown suffixes degrade to a generic binary type).
_FILE_CONTENT_TYPES = {
    ".mkv": "video/x-matroska",
    ".mp4": "video/mp4",
    ".ts": "video/mp2t",
}


def _content_type_for(path: Path) -> str:
    """Content-Type of the file part, derived from the filename suffix."""
    return _FILE_CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")


class UploadError(Exception):
    """Loud upload failure (Russian message; dedicated exit code upstream)."""


def pick_server(client: GofileClient) -> str:
    """GET https://api.gofile.io/servers (live-verified 2026-10-27) → first store host."""
    client.session.pin("api.gofile.io")
    response = adaptive_tls_request(
        client.session.get,
        url="https://api.gofile.io/servers",
        strict_tls=client.strict_tls,
        timeout=30,
    )
    if response.status_code != _HTTP_OK:
        msg = f"gofile /servers: HTTP {response.status_code}."
        raise UploadError(msg)
    payload = response.json()
    servers = (payload.get("data") or {}).get("servers", [])  # "data": null tolerated
    if payload.get("status") != "ok" or not servers:
        msg = f"gofile /servers вернул «{payload.get('status')}» без серверов."
        raise UploadError(msg)
    return str(servers[0]["name"])


class AdaptiveStreamingMultipart:
    """
    Streaming multipart/form-data generator with adaptive chunk sizing (plan §10).

    Strict field ordering matching Gofile's official uploadService.js:
    1. token (if provided)
    2. folderId (if provided)
    3. file (name, Content-Type: video/x-matroska, streamed from disk)

    Adaptive chunk scaling:
    - Starts at 2 MiB (2 * 1024 * 1024).
    - Measures transfer time of each chunk.
    - If transfer throughput is high (chunk transfer completes quickly < 0.5s),
      scales next chunk up (2 -> 4 -> 8 -> 16 -> up to 32 MiB).
    - If transfer throughput slows down (> 1.5s), scales chunk window smoothly down
      towards 1-2 MiB.
    - __len__ returns exact total body length (Content-Length) so requests/urllib3
      streams it directly over the socket without buffering in RAM.

    TOCTOU hardening (adversarial review F1): the file is opened once in __init__
    and the declared size comes from os.fstat() on that same handle, so the
    length and the streamed bytes are one coherent pair even if the path is
    renamed over or stat() races. __iter__ never reads beyond the declared size
    (a grown file streams its declared prefix and still ends with a valid
    closing boundary), and a shrunk file raises a loud RuntimeError instead of
    letting urllib3 silently truncate the body to the stale declared length.
    The streamer is single-use: requests consumes the body exactly once.
    """

    MIN_CHUNK_SIZE = 1 * 1024 * 1024  # 1 MiB
    INIT_CHUNK_SIZE = 2 * 1024 * 1024  # 2 MiB
    MAX_CHUNK_SIZE = 32 * 1024 * 1024  # 32 MiB
    FAST_ELAPSED_THRESHOLD = 0.5  # seconds
    SLOW_ELAPSED_THRESHOLD = 1.5  # seconds

    def __init__(  # noqa: PLR0913
        self,
        file_path: Path,
        *,
        token: str | None = None,
        folder_id: str | None = None,
        boundary: str | None = None,
        progress: Callable[[int, int], None] | None = None,
        initial_chunk_size: int = INIT_CHUNK_SIZE,
    ) -> None:
        """Initialize adaptive multipart stream, pinning the file handle and its size."""
        self.file_path = file_path
        # F1: one open descriptor feeds both the declared length and the stream,
        # so a concurrent mutation between "stat" and "read" cannot desync them.
        self._file_handle: BinaryIO | None = file_path.open("rb")
        self.file_size = os.fstat(self._file_handle.fileno()).st_size
        self.token = token
        self.folder_id = folder_id
        self.boundary = boundary or f"----WebKitFormBoundary{uuid.uuid4().hex}"
        self.progress = progress
        self.chunk_size = max(self.MIN_CHUNK_SIZE, min(initial_chunk_size, self.MAX_CHUNK_SIZE))

        # Build headers for fields according to strict Gofile specification
        # 1. token
        # 2. folderId
        # 3. file
        boundary_bytes = self.boundary.encode("ascii")
        self._pre_file_bytes = bytearray()
        if self.token:
            self._pre_file_bytes.extend(
                b"--" + boundary_bytes + b"\r\n"
                b'Content-Disposition: form-data; name="token"\r\n\r\n'
                + self.token.encode("utf-8")
                + b"\r\n"
            )
        if self.folder_id:
            self._pre_file_bytes.extend(
                b"--" + boundary_bytes + b"\r\n"
                b'Content-Disposition: form-data; name="folderId"\r\n\r\n'
                + self.folder_id.encode("utf-8")
                + b"\r\n"
            )

        # File part header
        filename = self.file_path.name
        file_content_type = _content_type_for(file_path)
        self._file_header = (
            b"--" + boundary_bytes + b"\r\n"
            b'Content-Disposition: form-data; name="file"; filename="'
            + filename.encode("utf-8")
            + b'"\r\n'
            + f"Content-Type: {file_content_type}\r\n\r\n".encode("ascii")
        )

        # Closing boundary after file data
        self._closing_bytes = b"\r\n--" + boundary_bytes + b"--\r\n"

        self._total_len = (
            len(self._pre_file_bytes)
            + len(self._file_header)
            + self.file_size
            + len(self._closing_bytes)
        )

    def __len__(self) -> int:
        """Exact content length in bytes for the entire multipart payload."""
        return self._total_len

    def close(self) -> None:
        """Release the pinned file handle; idempotent, safe after full iteration."""
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None

    def __iter__(self) -> Iterator[bytes]:
        """Stream headers, at most the declared file size, then the closing boundary."""
        if self._file_handle is None:
            msg = "Multipart-поток уже использован или закрыт."
            raise RuntimeError(msg)
        handle = self._file_handle

        # Yield metadata headers before file
        if self._pre_file_bytes:
            yield bytes(self._pre_file_bytes)
        yield bytes(self._file_header)

        # Stream the declared-size prefix of the file with an adaptive window.
        # F1: reads are clamped to the declared size, so a file that grew after
        # __init__ still yields a complete body ending with a valid boundary.
        sent = 0
        total = self.file_size
        current_chunk_size = self.chunk_size

        try:
            while sent < total:
                start_time = time.monotonic()
                chunk = handle.read(min(current_chunk_size, total - sent))
                if not chunk:
                    break
                sent += len(chunk)
                if self.progress is not None:
                    self.progress(sent, total)
                yield chunk
                elapsed = time.monotonic() - start_time

                # Adaptive tuning based on throughput
                if elapsed > 0:
                    if (
                        elapsed < self.FAST_ELAPSED_THRESHOLD
                        and current_chunk_size < self.MAX_CHUNK_SIZE
                    ):
                        # Fast transfer: double chunk window up to MAX_CHUNK_SIZE
                        current_chunk_size = min(current_chunk_size * 2, self.MAX_CHUNK_SIZE)
                    elif (
                        elapsed > self.SLOW_ELAPSED_THRESHOLD
                        and current_chunk_size > self.MIN_CHUNK_SIZE
                    ):
                        # Slow transfer: halve chunk window down to MIN_CHUNK_SIZE
                        current_chunk_size = max(current_chunk_size // 2, self.MIN_CHUNK_SIZE)
        finally:
            self.close()

        # F1 parity check: the file shrank below the declared size, so requests
        # would send a truncated body (urllib3 cuts at the stale Content-Length)
        # and gofile would store a broken file under a "success". Fail loudly
        # before the closing boundary instead.
        if sent < total:
            msg = (
                f"Файл изменился во время загрузки: заявлено {total} байт, "
                f"прочитано {sent}. Отправка прервана, "
                f"чтобы не отправлять усечённое тело."
            )
            raise RuntimeError(msg)

        yield bytes(self._closing_bytes)


def _upload_once(  # noqa: PLR0913, PLR0917 - single-attempt wiring bundle
    client: GofileClient,
    host: str,
    path: Path,
    token: str,
    folder_id: str | None,
    progress: Callable[[int, int], None] | None,
    chunk_size: int,
) -> requests.Response:
    """
    One upload POST with a freshly built single-use streamer.

    The pinned file handle never outlives the attempt (cli deletes the output
    right after, and an open handle blocks unlink on Windows); a retry builds
    a new streamer — fresh handle, generator and boundary, headers rebuilt to
    match.
    """
    streamer = AdaptiveStreamingMultipart(
        path,
        token=token,
        folder_id=folder_id,
        progress=progress,
        initial_chunk_size=chunk_size,
    )
    try:
        headers = {
            # F11: no manual Content-Length — requests derives it from len(data)
            # and would override a manual header anyway.
            "Content-Type": f"multipart/form-data; boundary={streamer.boundary}",
        }
        return adaptive_tls_request(
            client.session.post,
            url=f"https://{host}/uploadfile",
            strict_tls=client.strict_tls,
            data=streamer,
            headers=headers,
            cookies={"accountToken": token} if token else {},
            timeout=(10, 3600),
        )
    finally:
        streamer.close()


def _finish_upload_response(response: requests.Response) -> tuple[str, str]:
    """
    Validate the upload response contract (loud on every violation).

    Returns (download_page, parent_folder_id); parentFolder is always present
    per the official gofile contract — its absence raises loudly (F12), since
    a missing folder would silently degrade album mode into orphan folders.
    """
    if response.status_code != _HTTP_OK:
        msg = f"gofile upload: HTTP {response.status_code}."
        raise UploadError(msg)
    payload = response.json()
    if payload.get("status") != "ok":
        msg = (
            f"gofile upload вернул «{payload.get('status')}» "
            f"({str(payload.get('data'))[:200]}). Файл не изменён."
        )
        raise UploadError(msg)
    data = payload.get("data") or {}  # "data": null tolerated
    download_page = data.get("downloadPage")
    if not download_page:
        msg = f"gofile upload не вернул downloadPage: {str(payload)[:200]}"
        raise UploadError(msg)
    parent_folder = data.get("parentFolder")
    if not parent_folder:
        msg = "Gofile не вернул parentFolder — альбом невозможен."
        raise UploadError(msg)
    return str(download_page), str(parent_folder)


def upload_file(  # noqa: PLR0913
    path: Path,
    client: GofileClient,
    progress: Callable[[int, int], None] | None = None,
    folder_id: str | None = None,
    chunk_size: int = AdaptiveStreamingMultipart.INIT_CHUNK_SIZE,
    *,
    max_attempts: int = 3,
    backoff_base: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    jitter: Callable[[], float] = random.random,
    on_retry: Callable[[int, int, str], None] | None = None,
) -> tuple[str, str]:
    """
    Upload one file.

    Returns tuple of (download_page, parent_folder_id).
    Token cookie when configured (§10.3).

    Transient failures (HTTP 502/503/504, 429 with Retry-After honored when
    present, ConnectionError, ChunkedEncodingError, IncompleteRead) are retried
    up to ``max_attempts`` with exponential backoff. The streaming body is
    single-use: every attempt RECREATES the AdaptiveStreamingMultipart from the
    file (fresh handle + generator + boundary), never re-iterates a consumed
    streamer. Token resolution (and the stale-token recovery in fetch_contents)
    is untouched — the ladder wraps only the POST. A permanent refusal (any
    other HTTP status or business status) fails immediately.
    """
    if not path.is_file():
        msg = f"Файл для загрузки не найден: {path}"
        raise UploadError(msg)
    server = pick_server(client)
    host = f"{server}.gofile.io"
    client.session.pin(host)

    token = client.ensure_token()
    for attempt in range(1, max_attempts + 1):
        try:
            response = _upload_once(client, host, path, token, folder_id, progress, chunk_size)
        except _TRANSIENT_EXCEPTIONS as exc:
            if attempt == max_attempts:
                msg = f"Загрузка на gofile не удалась после {max_attempts} попыток: {exc}"
                raise UploadError(msg) from exc
            delay = backoff_base * 2 ** (attempt - 1) + jitter()
            if on_retry is not None:
                on_retry(attempt, max_attempts, str(exc))
            sleep(delay)
            continue
        if response.status_code not in _TRANSIENT_UPLOAD_STATUSES:
            return _finish_upload_response(response)
        if attempt == max_attempts:
            msg = (
                f"Загрузка на gofile не удалась после {max_attempts} попыток: "
                f"HTTP {response.status_code}."
            )
            raise UploadError(msg)
        delay = _429_retry_after(response) or backoff_base * 2 ** (attempt - 1) + jitter()
        if on_retry is not None:
            on_retry(attempt, max_attempts, f"HTTP {response.status_code}")
        sleep(delay)
    msg = f"Загрузка на gofile не удалась: исчерпаны попытки ({max_attempts})."
    raise UploadError(msg)  # pragma: no cover - loop returns or raises above


def _429_retry_after(response: object) -> float | None:
    """Honor Retry-After on a 429 when the header parses to a usable delay."""
    headers = getattr(response, "headers", None)
    getter = getattr(headers, "get", None)
    if getter is None:
        return None
    raw = getter("Retry-After")
    if raw is None:
        return None
    try:
        delay = float(raw)
    except (TypeError, ValueError):
        return None
    return delay if delay >= 0 else None


def upload_output(  # noqa: PLR0913
    path: Path,
    client: GofileClient,
    reporter: "Reporter",
    progress: Callable[[int, int], None] | None = None,
    folder_id: str | None = None,
    *,
    max_attempts: int = 3,
    backoff_base: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    jitter: Callable[[], float] = random.random,
) -> tuple[str, str]:
    """Post-step entry: loud failure keeps the output untouched; prints the link."""
    try:
        link, parent_folder = upload_file(
            path,
            client,
            progress=progress,
            folder_id=folder_id,
            max_attempts=max_attempts,
            backoff_base=backoff_base,
            sleep=sleep,
            jitter=jitter,
            on_retry=lambda attempt, total, reason: reporter.warn(
                f"Попытка {attempt}/{total} загрузки не удалась ({reason}) — повторяю..."
            ),
        )
    except (GofileError, requests.RequestException) as exc:
        msg = f"Загрузка {path.name} не удалась: {exc}"
        raise UploadError(msg) from exc
    reporter.link(link)
    return link, parent_folder
