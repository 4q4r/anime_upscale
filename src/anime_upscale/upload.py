"""
Optional gofile.io upload post-step (plan §10; mechanics delegated to gofile.py).

Trigger: --upload > profile upload=true > off. Runs only after a completed
pipeline; failure is loud, the output file is untouched, one attempt in v1.
"""

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import requests

from anime_upscale.gofile import GofileClient, GofileError

if TYPE_CHECKING:
    from anime_upscale.reporting import Reporter

_HTTP_OK = 200


class UploadError(Exception):
    """Loud upload failure (Russian message; dedicated exit code upstream)."""


def pick_server(client: GofileClient) -> str:
    """GET https://api.gofile.io/servers (live-verified 2026-10-27) → first store host."""
    client.session.pin("api.gofile.io")
    response = client.session.get("https://api.gofile.io/servers", timeout=30)
    if response.status_code != _HTTP_OK:
        msg = f"gofile /servers: HTTP {response.status_code}."
        raise UploadError(msg)
    payload = response.json()
    servers = (payload.get("data") or {}).get("servers", [])  # "data": null tolerated
    if payload.get("status") != "ok" or not servers:
        msg = f"gofile /servers вернул «{payload.get('status')}» без серверов."
        raise UploadError(msg)
    return str(servers[0]["name"])


class _ChunkReader:
    """File-like adapter over a chunk generator (requests requires .read)."""

    def __init__(self, generator: Iterator[bytes]) -> None:
        self._generator = generator
        self._buffer = b""

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            drained = self._buffer + b"".join(self._generator)
            self._buffer = b""
            return drained
        while len(self._buffer) < size:
            try:
                self._buffer += next(self._generator)
            except StopIteration:
                break
        chunk, self._buffer = self._buffer[:size], self._buffer[size:]
        return chunk


def upload_file(
    path: Path,
    client: GofileClient,
    progress: Callable[[int, int], None] | None = None,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Upload one file; returns data.downloadPage. Token cookie when configured (§10.3)."""
    if not path.is_file():
        msg = f"Файл для загрузки не найден: {path}"
        raise UploadError(msg)
    server = pick_server(client)
    host = f"{server}.gofile.io"
    client.session.pin(host)
    total = path.stat().st_size
    sent = 0

    def file_stream() -> Iterator[bytes]:
        nonlocal sent
        with path.open("rb") as handle:
            while chunk := handle.read(chunk_size):
                sent += len(chunk)
                if progress is not None:
                    progress(sent, total)
                yield chunk

    token = client.ensure_token()
    # requests requires a file-like (reads via .read()); the reader keeps the
    # chunked-multipart contract and the progress callback.
    files_dict = {"file": (path.name, _ChunkReader(file_stream()))}
    response = client.session.post(
        f"https://{host}/uploadfile",
        files=files_dict,
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
