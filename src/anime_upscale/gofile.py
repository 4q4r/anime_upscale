"""
Shared gofile.io API client (plan §10.1/§10.2 + §11.2–§11.4; addendum 1: folder expansion).

Every remote host is resolved STRICTLY via Google DoH and pinned for the
session (connect-by-IP, SNI/Host/TLS-verify by hostname) — no system-resolver
fallback, ever (plan §10.1). The wt (X-Website-Token) scheme is ported
VERIFIED-FROM-SOURCE from the owner's gofile-dl tool (run.py; formula, 4-hour
window, contents headers/params, retry rules).
"""

import hashlib
import os
import re
import time
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
import urllib3.connection
import urllib3.util.connection
from requests.adapters import HTTPAdapter

from anime_upscale.reporting import Reporter

# ── wt-scheme constants (gofile-dl run.py:98-118 — VERIFIED-FROM-SOURCE) ────────────────
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
DEFAULT_LANGUAGE = "en-US"
DEFAULT_WT_SALT = "12af056dacea0b"
WT_WINDOW_SECONDS = 14400  # 4-hour rotating window used by GoFile (run.py:107)

# Query parameters GoFile's web client sends on every /contents request (run.py:112-118).
CONTENTS_QUERY_PARAMS: dict[str, Any] = {
    "contentFilter": "",
    "page": 1,
    "pageSize": 1000,
    "sortField": "createTime",
    "sortDirection": -1,
}

GOFILE_SHARE_RE = re.compile(r"gofile\.io/d/([^/?#\s]+)", re.IGNORECASE)

# Video whitelist (spec addendum 1): only these extensions are queued from folders.
VIDEO_EXTENSIONS = frozenset({".mkv", ".mp4", ".avi", ".webm", ".mov", ".m4v", ".ts", ".m2ts"})

DOH_ENDPOINT = "https://dns.google/resolve"
_HTTP_OK = 200
_MAX_FOLDER_DEPTH = 10


class GofileError(Exception):
    """Loud gofile/DoH failure (Russian message)."""


# ── Strict Google DoH resolve + pin (plan §10.1) ─────────────────────────────────────────


def resolve_doh(hostname: str, timeout: float = 15.0) -> list[str]:
    """Resolve via Google DoH JSON API; loud Russian errors, no fallback."""
    try:
        response = requests.get(
            DOH_ENDPOINT,
            params={"name": hostname, "type": "A"},
            timeout=timeout,
            headers={"accept": "application/dns-json"},
        )
    except requests.RequestException as exc:
        msg = f"DoH-запрос к {DOH_ENDPOINT} для «{hostname}» не удался: {exc}"
        raise GofileError(msg) from exc
    if response.status_code != _HTTP_OK:
        msg = (
            f"DoH вернул HTTP {response.status_code} для «{hostname}» — "
            "системный резолвер не используется."
        )
        raise GofileError(msg)
    payload = response.json()
    if payload.get("Status") != 0:
        msg = f"DoH вернул Status={payload.get('Status')} для «{hostname}» — хост не разрешён."
        raise GofileError(msg)
    answers = [a for a in payload.get("Answer", []) if a.get("type") == 1]
    if not answers:
        msg = f"DoH не вернул A-записей для «{hostname}»."
        raise GofileError(msg)
    return [str(a["data"]) for a in answers]


class _PinnedConnectionBase(urllib3.connection.HTTPSConnection):
    """Dial the pinned IP; SNI, Host header and TLS verification stay on the hostname."""

    pins: dict[str, str]

    def _new_conn(self):  # noqa: ANN202 - matches urllib3 signature
        pinned_ip = self.pins.get(self.host)
        if pinned_ip is None:
            msg = f"Хост «{self.host}» не закреплён через DoH — соединение запрещено политикой."
            raise GofileError(msg)
        return urllib3.util.connection.create_connection(
            (pinned_ip, self.port),
            self.timeout,
            source_address=self.source_address,
            socket_options=self.socket_options,
        )


def _pinned_connection_class(pins: dict[str, str]) -> type:
    """Build a connection subclass bound to one session's pin map."""
    return type(
        "PinnedHTTPSConnection",
        (_PinnedConnectionBase,),
        {"pins": pins},
    )


class _PinnedPoolManager(urllib3.PoolManager):
    """
    PoolManager that swaps the connection class AFTER pool construction.

    Passing connection_class through connection_pool_kw would leak into
    urllib3's PoolKey (TypeError) — _new_pool is the correct seam.
    """

    pins: dict[str, str]

    def _new_pool(
        self,
        scheme: str,
        host: str,
        port: int,
        request_context: Any = None,  # noqa: ANN401 - urllib3 opaque context
    ) -> urllib3.HTTPConnectionPool:
        pool = super()._new_pool(scheme, host, port, request_context)
        pool.ConnectionCls = _pinned_connection_class(self.pins)
        return pool


class PinnedSession(requests.Session):
    """Session with per-host DoH pins; proxy env is ignored (incompatible with pinning)."""

    def __init__(self) -> None:
        """Build an unproxied session: pinning is incompatible with HTTP proxies."""
        super().__init__()
        self.trust_env = False  # a proxy would resolve hostnames itself — bypasses pinning
        self._pins: dict[str, str] = {}
        self._adapter = _PinnedAdapter(self._pins)
        self.mount("https://", self._adapter)

    def pin(self, hostname: str) -> str:
        """Resolve hostname via DoH and pin the first A record for this session."""
        if hostname not in self._pins:
            self._pins[hostname] = resolve_doh(hostname)[0]
        return self._pins[hostname]

    def pin_all(self, hostnames: list[str]) -> None:
        """Pin every hostname up-front (pre-flight for multi-host flows)."""
        for hostname in hostnames:
            self.pin(hostname)


class _PinnedAdapter(HTTPAdapter):
    def __init__(self, pins: dict[str, str], **kwargs: Any) -> None:  # noqa: ANN401
        """Store the pin map; remaining kwargs go to requests' HTTPAdapter."""
        self._pins = pins
        super().__init__(**kwargs)

    def init_poolmanager(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        """Install the pinned pool manager (dial-by-IP, hostname TLS)."""
        manager = _PinnedPoolManager(*args, **kwargs)
        manager.pins = self._pins
        self.poolmanager = manager


# ── wt (X-Website-Token) — ported verbatim from gofile-dl run.py:121-144 ────────────────


def generate_website_token(
    account_token: str,
    window_offset: int = 0,
    user_agent: str = DEFAULT_USER_AGENT,
    language: str = DEFAULT_LANGUAGE,
    salt: str = DEFAULT_WT_SALT,
) -> str:
    """
    Compute gofile's X-Website-Token (gofile-dl run.py:88, 121-144, verbatim).

    sha256("{userAgent}::{language}::{accountToken}::{window}::{salt}") → 64-hex.
    """
    window = int(time.time() // WT_WINDOW_SECONDS) + window_offset
    raw = f"{user_agent}::{language}::{account_token}::{window}::{salt}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def content_headers(
    token: str,
    user_agent: str = DEFAULT_USER_AGENT,
    language: str = DEFAULT_LANGUAGE,
    salt: str = DEFAULT_WT_SALT,
    window_offset: int = 0,
) -> dict[str, str]:
    """Headers GoFile expects on /contents (run.py:456-466)."""
    return {
        "Authorization": f"Bearer {token}",
        "X-Website-Token": generate_website_token(
            token, window_offset=window_offset, user_agent=user_agent, language=language, salt=salt
        ),
        "X-BL": language,
        "User-Agent": user_agent,
        "Accept": "*/*",
        "Origin": "https://gofile.io",
        "Referer": "https://gofile.io/",
    }


# ── Guest-token lifecycle (plan §11.2; §12.1 platform-aware state path) ──────────────────


def _is_windows() -> bool:
    """Platform seam for tests (pathlib dispatches on the real os.name)."""
    return os.name == "nt"


def state_file_path() -> Path:
    r"""
    Platform-aware state location — one resolution function (plan §12.1).

    Linux:   $XDG_STATE_HOME/anime_upscale/gofile_guest.toml, default ~/.local/state/…
    Windows: %LOCALAPPDATA%\\anime_upscale\\state\\gofile_guest.toml.
    """
    if _is_windows():
        local_app_data = os.environ.get("LOCALAPPDATA")
        base = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
        return base / "anime_upscale" / "state" / "gofile_guest.toml"
    state_home = os.environ.get("XDG_STATE_HOME")
    base = Path(state_home) if state_home else Path.home() / ".local" / "state"
    return base / "anime_upscale" / "gofile_guest.toml"


def load_saved_guest(path: Path | None = None) -> str:
    """Read the saved guest token; corrupt file = loud error naming the path + remedy."""
    path = path or state_file_path()
    if not path.is_file():
        return ""
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        token = raw["token"]
    except (tomllib.TOMLDecodeError, KeyError, OSError) as exc:
        msg = (
            f"Файл состояния гостевого токена повреждён: {path} ({exc}). "
            "Удалите его и повторите запуск — будет создан новый гостевой токен."
        )
        raise GofileError(msg) from exc
    if not isinstance(token, str) or not token:
        msg = (
            f"Файл состояния гостевого токена повреждён: {path} "
            "(token отсутствует или пуст). Удалите его и повторите запуск."
        )
        raise GofileError(msg)
    return token


def save_guest(token: str, path: Path | None = None) -> Path:
    """Persist the guest token once (creation is the rate-limited operation)."""
    path = path or state_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    path.write_text(f'token = "{token}"\ncreated = "{created}"\n', encoding="utf-8")
    return path


@dataclass(frozen=True)
class ResolvedChild:
    """One video file expanded from a (possibly nested) folder link."""

    name: str
    link: str


class GofileClient:
    """gofile API client consumed by fetch_input.py (downloads) and upload.py (uploads)."""

    def __init__(
        self,
        account_token: str = "",
        user_agent: str = DEFAULT_USER_AGENT,
        language: str = DEFAULT_LANGUAGE,
        wt_salt: str = DEFAULT_WT_SALT,
        state_path: Path | None = None,
    ) -> None:
        """Wire the client to config-derived constants and a pinned session."""
        self.user_agent = user_agent
        self.language = language
        self.wt_salt = wt_salt
        self.state_path = state_path
        self.session = PinnedSession()
        self._account_token = account_token
        self.token = account_token

    # -- token lifecycle ---------------------------------------------------
    def ensure_token(self) -> str:
        """Resolve the token: account > saved guest > create once + save (plan §11.2)."""
        if self.token:
            return self.token
        saved = load_saved_guest(self.state_path)
        if saved:
            self.token = saved
            return self.token
        self.session.pin("api.gofile.io")
        response = self.session.post(
            "https://api.gofile.io/accounts",
            headers={"User-Agent": self.user_agent, "Origin": "https://gofile.io"},
            timeout=30,
        )
        payload = response.json() if response.status_code == _HTTP_OK else {}
        if payload.get("status") != "ok" or not payload.get("data", {}).get("token"):
            msg = (
                f"Не удалось создать гостевой токен gofile: HTTP {response.status_code}, "
                f"ответ: {payload.get('status', response.text[:200])}"
            )
            raise GofileError(msg)
        self.token = payload["data"]["token"]
        if not self._account_token:  # account tokens never overwrite the guest state file
            save_guest(self.token, self.state_path)
        return self.token

    # -- contents (with the ported minimal retries) ------------------------
    def fetch_contents(self, content_id: str) -> dict[str, Any]:
        """GET /contents/{id} with wt headers; ported retry rules (run.py:595-667)."""
        self.ensure_token()
        self.session.pin("api.gofile.io")
        url = f"https://api.gofile.io/contents/{content_id}"
        max_attempts = 4
        for attempt in range(max_attempts):
            window_offset = -1 if attempt >= 1 else 0  # current window, then −1 once
            response = self.session.get(
                url,
                headers=content_headers(
                    self.token,
                    user_agent=self.user_agent,
                    language=self.language,
                    salt=self.wt_salt,
                    window_offset=window_offset,
                ),
                params=CONTENTS_QUERY_PARAMS,
                timeout=60,
            )
            if response.status_code != _HTTP_OK:
                msg = f"gofile /contents/{content_id}: HTTP {response.status_code}."
                raise GofileError(msg)
            payload = response.json()
            status = payload.get("status")
            if status == "ok":
                return payload
            if status == "error-rateLimit":
                wait = 3 * (attempt + 1)  # linear backoff 3 s × attempt (run.py:631-639)
                if attempt < max_attempts - 1:
                    time.sleep(wait)
                    continue
                msg = (
                    "gofile ограничил частоту запросов (error-rateLimit после всех попыток). "
                    "Подождите несколько минут и повторите."
                )
                raise GofileError(msg)
            if status == "error-notPremium":
                if attempt == 0:
                    continue  # retry once with window − 1 (clock boundary, run.py:641-647)
                msg = (
                    "gofile отклонил X-Website-Token (error-notPremium). Вероятно, "
                    "ротировалась соль wt.obf.js — обновите [fetch.gofile] wt_salt в config.toml."
                )
                raise GofileError(msg)
            if status == "error-notFound":
                msg = f"gofile: содержимое {content_id} не найдено (удалено?)."
                raise GofileError(msg)
            msg = f"gofile вернул ошибку «{status}» для /contents/{content_id}."
            raise GofileError(msg)
        msg = f"gofile /contents/{content_id}: исчерпаны попытки."
        raise GofileError(msg)

    # -- folder expansion (spec addendum 1) --------------------------------
    def expand_entries(
        self,
        content_id: str,
        reporter: Reporter | None = None,
    ) -> list[ResolvedChild]:
        """
        Walk data.children recursively (spec addendum 1).

        Deterministic order: folders first, then files, each alphabetically by
        name; only VIDEO_EXTENSIONS children are kept (skips printed loudly,
        never silent); empty-after-filter = loud error.
        """
        payload = self.fetch_contents(content_id)
        data = payload.get("data", {})
        if data.get("passwordStatus", "passwordOk") != "passwordOk":
            msg = (
                f"gofile: содержимое {content_id} защищено паролем "
                f"(passwordStatus={data.get('passwordStatus')}) — поддержка паролей не входит в v1."
            )
            raise GofileError(msg)
        children = data.get("children")
        found: list[ResolvedChild] = []
        if children is None:
            if data.get("type") == "file":
                return [ResolvedChild(name=data.get("name", ""), link=data.get("link", ""))]
            msg = f"gofile: у содержимого {content_id} отсутствуют данные children."
            raise GofileError(msg)
        self._walk(children, found, reporter, depth=0)
        if not found:
            msg = (
                f"В папке gofile {content_id} нет видеофайлов после фильтра "
                f"({', '.join(sorted(VIDEO_EXTENSIONS))}) — нечего ставить в очередь."
            )
            raise GofileError(msg)
        return found

    def _walk(
        self,
        children: list[dict[str, Any]],
        found: list[ResolvedChild],
        reporter: Reporter | None,
        depth: int,
    ) -> None:
        if depth > _MAX_FOLDER_DEPTH:
            msg = "Слишком глубокая вложенность папки gofile (лимит 10 уровней)."
            raise GofileError(msg)
        folders = sorted(
            (c for c in children if c.get("type") == "folder"), key=lambda c: str(c.get("name", ""))
        )
        files = sorted(
            (c for c in children if c.get("type") != "folder"), key=lambda c: str(c.get("name", ""))
        )
        for folder in folders:
            sub = self.fetch_contents(str(folder.get("id", "")))
            sub_children = sub.get("data", {}).get("children")
            if not isinstance(sub_children, list):
                msg = f"gofile: у подпапки «{folder.get('name')}» повреждены данные children."
                raise GofileError(msg)
            self._walk(sub_children, found, reporter, depth + 1)
        for child in files:
            name = str(child.get("name", ""))
            suffix = Path(name).suffix.lower()
            if suffix not in VIDEO_EXTENSIONS:
                if reporter is not None:
                    reporter.note(f"Пропуск (не видео): {name}")
                continue
            link = child.get("link")
            if not link:
                msg = f"gofile: у файла «{name}» отсутствует ссылка link (повреждённые данные)."
                raise GofileError(msg)
            found.append(ResolvedChild(name=name, link=str(link)))

    # -- file download ------------------------------------------------------
    def download_file(
        self,
        link: str,
        destination: Path,
        progress: Callable[[int, int], None] | None = None,
        chunk_size: int = 1024 * 1024,
    ) -> None:
        """GET <link> with accountToken cookie; .part + atomic rename (run.py:945-1000)."""
        self.ensure_token()
        host = urlparse(link).hostname or ""
        if not host:
            msg = f"Некорректная ссылка для скачивания: {link}"
            raise GofileError(msg)
        self.session.pin(host)
        temp = destination.with_name(destination.name + ".part")
        try:
            with self.session.get(
                link,
                headers={
                    "Cookie": f"accountToken={self.token}",
                    "User-Agent": self.user_agent,
                    "Referer": "https://gofile.io/",
                },
                stream=True,
                timeout=(10, 300),
            ) as response:
                if response.status_code != _HTTP_OK:
                    msg = f"Скачивание {link}: HTTP {response.status_code}."
                    raise GofileError(msg)
                total = int(response.headers.get("Content-Length", 0))
                done = 0
                with temp.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        handle.write(chunk)
                        done += len(chunk)
                        if progress is not None:
                            progress(done, total)
            temp.rename(destination)  # atomic: only complete files land in input/
        except (requests.RequestException, OSError) as exc:
            temp.unlink(missing_ok=True)
            msg = f"Ошибка скачивания {link}: {exc}"
            raise GofileError(msg) from exc
