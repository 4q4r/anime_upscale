"""
Optional gofile.io upload post-step (plan §10; mechanics delegated to gofile.py).

Trigger: --upload > profile upload=true > off. Runs only after a completed
pipeline; failure is loud, the output file is untouched, one attempt in v1.
"""

import time
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import requests

from anime_upscale.gofile import GofileClient, GofileError, adaptive_tls_request

if TYPE_CHECKING:
    from anime_upscale.reporting import Reporter

_HTTP_OK = 200


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
        """Initialize adaptive multipart stream with metadata fields and file boundary."""
        self.file_path = file_path
        self.file_size = file_path.stat().st_size
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
        self._file_header = (
            b"--" + boundary_bytes + b"\r\n"
            b'Content-Disposition: form-data; name="file"; filename="'
            + filename.encode("utf-8")
            + b'"\r\n'
            b"Content-Type: video/x-matroska\r\n\r\n"
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

    def __iter__(self) -> Iterator[bytes]:
        """Stream multipart headers, dynamic file chunks, and closing boundary."""
        # Yield metadata headers before file
        if self._pre_file_bytes:
            yield bytes(self._pre_file_bytes)
        yield bytes(self._file_header)

        # Stream file with adaptive window
        sent = 0
        total = self.file_size
        current_chunk_size = self.chunk_size

        with self.file_path.open("rb") as handle:
            while True:
                start_time = time.monotonic()
                chunk = handle.read(current_chunk_size)
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

        yield bytes(self._closing_bytes)


def upload_file(
    path: Path,
    client: GofileClient,
    progress: Callable[[int, int], None] | None = None,
    folder_id: str | None = None,
    chunk_size: int = AdaptiveStreamingMultipart.INIT_CHUNK_SIZE,
) -> str:
    """Upload one file; returns data.downloadPage. Token cookie when configured (§10.3)."""
    if not path.is_file():
        msg = f"Файл для загрузки не найден: {path}"
        raise UploadError(msg)
    server = pick_server(client)
    host = f"{server}.gofile.io"
    client.session.pin(host)

    token = client.ensure_token()
    streamer = AdaptiveStreamingMultipart(
        path,
        token=token,
        folder_id=folder_id,
        progress=progress,
        initial_chunk_size=chunk_size,
    )
    headers = {
        "Content-Type": f"multipart/form-data; boundary={streamer.boundary}",
        "Content-Length": str(len(streamer)),
    }
    response = adaptive_tls_request(
        client.session.post,
        url=f"https://{host}/uploadfile",
        strict_tls=client.strict_tls,
        data=streamer,
        headers=headers,
        cookies={"accountToken": token} if token else {},
        timeout=(10, 3600),
    )
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
    download_page = (payload.get("data") or {}).get("downloadPage")  # "data": null tolerated
    if not download_page:
        msg = f"gofile upload не вернул downloadPage: {str(payload)[:200]}"
        raise UploadError(msg)
    return str(download_page)


def upload_output(
    path: Path,
    client: GofileClient,
    reporter: "Reporter",
    progress: Callable[[int, int], None] | None = None,
) -> str:
    """Post-step entry: loud failure keeps the output untouched; prints the link."""
    try:
        link = upload_file(path, client, progress=progress)
    except (GofileError, requests.RequestException) as exc:
        msg = f"Загрузка {path.name} не удалась: {exc}"
        raise UploadError(msg) from exc
    reporter.link(link)
    return link
