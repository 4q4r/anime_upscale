"""
Shared gofile.io API client (plan §10.1/§10.2 + §11.2–§11.4; addendum 1: folder expansion).

Every remote host is resolved STRICTLY via DoH and pinned for the session
(connect-by-IP, SNI/Host/TLS-verify by hostname) — no system-resolver
fallback, ever (plan §10.1). DoH itself uses a resolver ladder of verified
JSON-API endpoints (dns.google domain → Google by bare IP → Cloudflare by
bare IP): networks that TLS-intercept the dns.google DOMAIN cannot touch the
bare-IP endpoints (their certificates carry IP SANs, verified natively).
The wt (X-Website-Token) scheme is ported VERIFIED-FROM-SOURCE from the
owner's gofile-dl tool (run.py; formula, 4-hour window, contents
headers/params, retry rules).
"""

import hashlib
import http.client
import os
import random
import re
import sys
import time
import tomllib
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
import urllib3.connection
import urllib3.exceptions
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

# Share path INSIDE a gofile URL — matched against urlparse(...).path only,
# never the whole string (a /d/<id> shape in a query string must not classify).
GOFILE_SHARE_RE = re.compile(r"^/d/([^/?#]+?)/?$", re.IGNORECASE)

# Video whitelist (spec addendum 1): only these extensions are queued from folders.
VIDEO_EXTENSIONS = frozenset({".mkv", ".mp4", ".avi", ".webm", ".mov", ".m4v", ".ts", ".m2ts"})

DOH_ENDPOINTS: tuple[str, ...] = (
    "https://dns.google/resolve",  # canonical Google DoH JSON API (domain form)
    "https://8.8.8.8/resolve",  # Google DoH by bare IP — cert carries an IP SAN
    "https://1.1.1.1/dns-query",  # Cloudflare DoH by bare IP — same cert property
)
_HTTP_OK = 200
# Range/206 resume negotiation statuses (download ladder).
_HTTP_PARTIAL_CONTENT = 206
_HTTP_RANGE_NOT_SATISFIABLE = 416
_MAX_FOLDER_DEPTH = 10
# Pagination runaway guard (official /contents contract): a page budget well
# beyond any real album (50 × pageSize 1000 = 50k entries); exceeding it means
# the server never stops claiming hasNextPage — treated as an API malfunction.
_MAX_CONTENTS_PAGES = 50


class GofileError(Exception):
    """Loud gofile/DoH failure (Russian message)."""


class GofileTransientError(GofileError):
    """Transient gofile HTTP failure (502/503/504, 416-restart) — retryable."""


# ── Interception-adaptive TLS (live Windows-server feedback, 2026-10-28) ────────────────
#
# The owner's server network TLS-intercepts gofile-family hosts (api.gofile.io and
# store hosts answer with self-signed chains — the same selective DPI that hits
# dns.google). DoH stays STRICTLY verified (the anti-spoofing anchor: it proves the
# DESTINATION); only the last-mile certificate check of gofile/fetch requests is
# adaptive. On a certificate-verification failure specifically, the request is
# retried ONCE with verify=False, preceded by a loud one-time Russian banner. The
# owner accepts provider visibility of his gofile traffic; the kill-switch is
# [network] strict_tls = true.

INTERCEPT_WARNING = (
    "⚠ Перехват TLS на сети (self-signed цепочка) — соединение продолжено "
    "без проверки сертификата. Назначение подтверждено DoH-пиннингом; "
    "провайдер видит трафик. Отключить: [network] strict_tls = true"
)

_intercept_warning_shown = False


def _warn_intercepted_tls() -> None:
    """Print the interception banner once per process (module-level dedup flag)."""
    global _intercept_warning_shown  # noqa: PLW0603 - one-shot banner dedup is process-global by design
    if _intercept_warning_shown:
        return
    _intercept_warning_shown = True
    sys.stderr.write(INTERCEPT_WARNING + "\n")
    sys.stderr.flush()


def is_certificate_verify_failure(exc: requests.exceptions.SSLError) -> bool:
    """Report whether this SSL error is specifically a certificate-verification failure."""
    text = str(exc)
    return "CERTIFICATE_VERIFY_FAILED" in text or "self-signed" in text


def _strict_tls_diagnosis(exc: requests.exceptions.SSLError) -> str:
    return (
        f"{exc}. Диагноз: сеть перехватывает TLS-трафик (self-signed цепочка сертификата). "
        "Проверка сертификата не отключена: работает [network] strict_tls = true. "
        "Разрешить продолжение соединения на перехватывающей сети: "
        "[network] strict_tls = false (по умолчанию) в config.toml."
    )


def adaptive_tls_request(
    sender: Callable[..., requests.Response],
    *,
    strict_tls: bool = False,
    **kwargs: Any,  # noqa: ANN401 - mirrors the requests call surface
) -> requests.Response:
    """
    ONE shared seam for every gofile-family/fetch HTTP request.

    Calls ``sender(**kwargs)`` (a bound requests session verb); on a certificate-
    verification SSL failure specifically: strict mode re-raises the same SSL
    error enriched with the interception diagnosis and the config flag name
    (no retry), otherwise prints the one-time banner and retries ONCE with
    ``verify=False``. Non-SSL errors and non-verification SSL errors pass
    through untouched, on both paths.
    """
    try:
        return sender(**kwargs)
    except requests.exceptions.SSLError as exc:
        if not is_certificate_verify_failure(exc):
            raise
        if strict_tls:
            raise requests.exceptions.SSLError(_strict_tls_diagnosis(exc)) from exc
        _warn_intercepted_tls()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", urllib3.exceptions.InsecureRequestWarning)
            return sender(**{**kwargs, "verify": False})


# ── Strict DoH resolve ladder + pin (plan §10.1) ─────────────────────────────────────────


def resolve_doh(hostname: str, timeout: float = 15.0) -> list[str]:
    """
    Resolve via the DoH ladder (first-success-wins); loud Russian errors, no fallback.

    Success per endpoint = HTTP 200 + Status==0 + non-empty A records. The ladder
    advances ONLY on transport errors (connection/SSL/timeout — any
    requests.RequestException, unparseable bodies included); a received answer is
    final: non-200, Status!=0 and empty-A behave exactly as the single-endpoint
    implementation. Timeout, headers and proxy handling are identical for every
    endpoint. Bare-IP endpoints never touch the system resolver: requests
    verifies their certificates natively via the IP SANs.

    The ladder runs on a dedicated Session with ``trust_env = False`` (the same
    policy as PinnedSession): an environment proxy would resolve hostnames by
    itself and would route the DoH first hop away from the pinned endpoints.
    """
    failures: list[str] = []
    with requests.Session() as session:
        session.trust_env = False  # a proxy would resolve hostnames itself — bypasses the ladder
        for endpoint in DOH_ENDPOINTS:
            try:
                response = session.get(
                    endpoint,
                    params={"name": hostname, "type": "A"},
                    timeout=timeout,
                    headers={"accept": "application/dns-json"},
                )
                payload = response.json()
            except requests.RequestException as exc:
                failures.append(f"{endpoint} — {exc}")
                continue
            if response.status_code != _HTTP_OK:
                msg = (
                    f"DoH вернул HTTP {response.status_code} для «{hostname}» — "
                    "системный резолвер не используется."
                )
                raise GofileError(msg)
            if payload.get("Status") != 0:
                msg = (
                    f"DoH вернул Status={payload.get('Status')} для «{hostname}» — "
                    "хост не разрешён."
                )
                raise GofileError(msg)
            answers = [a for a in payload.get("Answer", []) if a.get("type") == 1]
            if not answers:
                msg = f"DoH не вернул A-записей для «{hostname}»."
                raise GofileError(msg)
            return [str(a["data"]) for a in answers]
    details = "\n".join(f"  • {failure}" for failure in failures)
    msg = (
        f"DoH-разрешение «{hostname}» не удалось ни на одном из {len(DOH_ENDPOINTS)} "
        f"резолверов:\n{details}\n"
        "Ошибки сертификатов (self-signed) на всех резолверах означают перехват "
        "TLS-трафика вашей сетью (DPI/MITM). Системный резолвер не используется — "
        "защита от подмены (§10.1)."
    )
    raise GofileError(msg)


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
        # Runtime-valid: urllib3 pools accept a BaseHTTPSConnection subclass.
        pool.ConnectionCls = _pinned_connection_class(self.pins)  # ty: ignore[invalid-assignment]
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
    """
    Persist the guest token once (creation is the rate-limited operation).

    The token is a credential: the temp file is CREATED at 0600 via os.open
    (owner-only from the first byte — no world-readable window before a
    post-write chmod), then published with os.replace (atomic — a crash
    mid-write never leaves a corrupt state file behind).
    """
    path = path or state_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    temp = path.with_name(path.name + ".tmp")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(f'token = "{token}"\ncreated = "{created}"\n')
    temp.chmod(0o600)  # umask-proof re-assert (Windows: read-only attribute only)
    temp.replace(path)  # atomic: only complete, owner-only files reach the state path
    return path


def _server_date_offset(response: requests.Response) -> float | None:
    """
    Measured server-clock offset in seconds from the HTTP ``Date`` header.

    None when the header is absent or unparseable — never invent an offset.
    """
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    raw = headers.get("Date")
    if not raw:
        return None
    try:
        parsed = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if parsed is None:  # pragma: no cover - stdlib always returns a datetime here
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return (parsed - datetime.now(UTC)).total_seconds()


@dataclass(frozen=True)
class ResolvedChild:
    """One video file expanded from a (possibly nested) folder link."""

    name: str
    link: str
    size: int | None = None
    md5: str | None = None


@dataclass
class _FetchState:
    """Mutable retry bookkeeping for one fetch_contents call."""

    window_offset: int = 0
    refreshed: bool = False
    not_premium_stage: int = 0


@dataclass
class _ListingState:
    """Pagination accumulator for one /contents listing (official contract)."""

    page: int = 1
    base_payload: dict[str, Any] | None = None
    accumulated: list[dict[str, Any]] = field(default_factory=list)


def _children_values(children: Any) -> list[dict[str, Any]]:  # noqa: ANN401 - live payload
    """Children entries of one /contents page: dict (live shape) or list (legacy)."""
    if isinstance(children, dict):
        return [child for child in children.values() if isinstance(child, dict)]
    if isinstance(children, list):
        return [child for child in children if isinstance(child, dict)]
    return []


class GofileClient:
    """gofile API client consumed by fetch_input.py (downloads) and upload.py (uploads)."""

    def __init__(  # noqa: PLR0913 - config-derived constructor bundle, mirrors config sections 1:1
        self,
        account_token: str = "",
        user_agent: str = DEFAULT_USER_AGENT,
        language: str = DEFAULT_LANGUAGE,
        wt_salt: str = DEFAULT_WT_SALT,
        state_path: Path | None = None,
        *,
        strict_tls: bool = False,
    ) -> None:
        """Wire the client to config-derived constants and a pinned session."""
        self.user_agent = user_agent
        self.language = language
        self.wt_salt = wt_salt
        self.state_path = state_path
        self.strict_tls = strict_tls
        self.session = PinnedSession()
        self._account_token = account_token
        self.token = account_token

    # -- token lifecycle ---------------------------------------------------
    def _create_guest_token(self) -> str:
        """Create one guest account (the rate-limited operation) and return its token."""
        self.session.pin("api.gofile.io")
        try:
            response = adaptive_tls_request(
                self.session.post,
                url="https://api.gofile.io/accounts",
                strict_tls=self.strict_tls,
                headers={"User-Agent": self.user_agent, "Origin": "https://gofile.io"},
                timeout=30,
            )
        except requests.RequestException as exc:
            # §11.3: a pre-flight connection failure is a loud Russian error, not a traceback.
            msg = f"Сетевая ошибка при создании гостевого токена gofile: {exc}"
            raise GofileError(msg) from exc
        payload = response.json() if response.status_code == _HTTP_OK else {}
        data = payload.get("data") or {}  # live API may send explicit `"data": null`
        if payload.get("status") != "ok" or not data.get("token"):
            msg = (
                f"Не удалось создать гостевой токен gofile: HTTP {response.status_code}, "
                f"ответ: {payload.get('status', response.text[:200])}"
            )
            raise GofileError(msg)
        return str(data["token"])

    def ensure_token(self) -> str:
        """Resolve the token: account > saved guest > create once + save (plan §11.2)."""
        if self.token:
            return self.token
        saved = load_saved_guest(self.state_path)
        if saved:
            self.token = saved
            return self.token
        self.token = self._create_guest_token()
        save_guest(self.token, self.state_path)
        return self.token

    def refresh_guest(self) -> str:
        """
        Force a NEW guest token and persist it (stale-token recovery, 2026-11-02).

        Account tokens are never replaced by a guest: this raises instead.
        """
        if self._account_token:
            msg = (
                "Задан токен аккаунта ([upload.gofile] token) — гостевой токен не создаётся "
                "и не подставляется вместо него. Проверьте токен аккаунта в config.toml."
            )
            raise GofileError(msg)
        self.token = self._create_guest_token()
        save_guest(self.token, self.state_path)
        return self.token

    # -- contents (with the ported minimal retries) ------------------------
    @staticmethod
    def _parse_contents_response(
        response: requests.Response,
        content_id: str,
    ) -> dict[str, Any]:
        """Parse JSON BEFORE any HTTP-status branch (live 401 carried a useful status)."""
        try:
            payload = response.json()
        except ValueError as exc:
            body = (getattr(response, "text", "") or "")[:200]
            msg = (
                f"gofile /contents/{content_id}: HTTP {response.status_code}, "
                f"ответ не JSON: {body!r}"
            )
            raise GofileError(msg) from exc
        if not isinstance(payload, dict):
            body = (getattr(response, "text", "") or "")[:200]
            msg = (
                f"gofile /contents/{content_id}: HTTP {response.status_code}, "
                f"ожидался JSON-объект, получено: {body!r}"
            )
            raise GofileError(msg)
        return payload

    def _not_premium_error(self, response: requests.Response, *, refreshed: bool) -> GofileError:
        """Loud terminal notPremium verdict with the MEASURED clock offset when available."""
        actions = "после окна −1"
        if refreshed:
            actions += " и обновления гостевого токена"
        offset = _server_date_offset(response)
        if offset is None:
            clock = "Сдвиг часов измерить не удалось: заголовок Date отсутствует."
        else:
            clock = f"Измеренный сдвиг часов сервера по заголовку Date: {offset:+.0f} с."
        msg = (
            f"gofile отклонил X-Website-Token (error-notPremium) {actions}. "
            "Вероятная причина: ротация соли wt.obf.js — обновите [fetch.gofile] wt_salt "
            f"в config.toml; либо расхождение часов с сервером. {clock}"
        )
        return GofileError(msg)

    def _recover_token_error(self, status: str, state: _FetchState) -> None:
        """Handle error-wrongToken/error-token: ONE guest refresh, never for account."""
        if self._account_token:
            msg = (
                f"gofile отклонил токен аккаунта (status={status}). "
                "Гостевой токен не подставляется — проверьте [upload.gofile] token "
                "в config.toml."
            )
            raise GofileError(msg)
        if state.refreshed:
            path = self.state_path or state_file_path()
            msg = (
                f"gofile отклонил гостевой токен (status={status}) даже после "
                f"принудительного обновления. Файл состояния: {path}. "
                "Удалите его и повторите запуск."
            )
            raise GofileError(msg)
        self.refresh_guest()  # exactly once per fetch
        state.refreshed = True
        state.window_offset = 0

    def _step_not_premium(self, response: requests.Response, state: _FetchState) -> None:
        """Owner-verified notPremium order: window −1 → fresh guest token → loud."""
        if state.not_premium_stage == 0:
            state.not_premium_stage = 1
            state.window_offset = -1
            return
        can_refresh = not state.refreshed and not self._account_token
        if state.not_premium_stage == 1 and can_refresh:
            state.not_premium_stage = 2
            self.refresh_guest()
            state.refreshed = True
            state.window_offset = 0
            return
        raise self._not_premium_error(response, refreshed=state.refreshed)

    def fetch_contents(self, content_id: str) -> dict[str, Any]:
        """
        GET /contents/{id}: JSON first, then status (live 401 recovery, 2026-11-02).

        Official listing contract (gofile.io/api, fetched 2026-11-06): folder
        listings paginate via ``page``/``pageSize`` and every page carries a
        ``metadata`` object next to ``data`` with ``hasNextPage``/``totalCount``.
        Children are accumulated across pages until ``hasNextPage`` is falsy;
        the collected total is then cross-checked against ``totalCount``
        (mismatch = loud error — insurance against a silent API change). A
        response without ``metadata`` is treated as the legacy single-page
        shape and returned untouched. Retry/recovery state (rate limit
        backoff, guest refresh, notPremium ladder) applies per page and is
        bounded by the same budgets as the single-page implementation.
        """
        self.ensure_token()
        self.session.pin("api.gofile.io")
        url = f"https://api.gofile.io/contents/{content_id}"
        max_attempts = 4
        state = _FetchState()
        listing = _ListingState()
        attempts = 0
        while True:
            attempts += 1
            try:
                response = adaptive_tls_request(
                    self.session.get,
                    url=url,
                    strict_tls=self.strict_tls,
                    headers=content_headers(
                        self.token,
                        user_agent=self.user_agent,
                        language=self.language,
                        salt=self.wt_salt,
                        window_offset=state.window_offset,
                    ),
                    params={**CONTENTS_QUERY_PARAMS, "page": listing.page},
                    timeout=60,
                )
            except requests.RequestException as exc:
                # §11.3: connection reset etc. = loud error quoting the diagnosis.
                msg = (
                    f"Сетевая ошибка при запросе gofile /contents/{content_id} "
                    f"(страница {listing.page}, попытка {attempts}/{max_attempts}): {exc}"
                )
                raise GofileError(msg) from exc
            payload = self._parse_contents_response(response, content_id)
            status = payload.get("status")
            if status == "ok":
                final = self._advance_listing(payload, content_id, listing)
                if final is not None:
                    return final
                continue
            if status == "error-rateLimit":
                if attempts < max_attempts:
                    time.sleep(3 * attempts)  # linear backoff 3 s × attempt (run.py:631-639)
                    continue
                msg = (
                    "gofile ограничил частоту запросов (error-rateLimit после всех попыток). "
                    "Подождите несколько минут и повторите."
                )
                raise GofileError(msg)
            if status in {"error-wrongToken", "error-token"}:
                self._recover_token_error(status, state)
                continue
            if status == "error-notPremium":
                self._step_not_premium(response, state)
                continue
            if status == "error-notFound":
                msg = f"gofile: содержимое {content_id} не найдено (удалено или недоступно)."
                raise GofileError(msg)
            msg = (
                f"gofile вернул неизвестный статус «{status}» для /contents/{content_id} "
                f"(HTTP {response.status_code})."
            )
            raise GofileError(msg)

    def _advance_listing(
        self,
        payload: dict[str, Any],
        content_id: str,
        listing: _ListingState,
    ) -> dict[str, Any] | None:
        """
        Accumulate one ``ok`` page into the listing state.

        Returns the final merged payload (single-page shape or fully collected
        listing) or None when the server says there is a next page. Loud on
        pagination-contract violations: metadata vanishing mid-stream,
        totalCount/collected mismatch, runaway hasNextPage.
        """
        data = payload.get("data") or {}  # live API may send explicit `"data": null`
        children = data.get("children")
        if listing.base_payload is None and not isinstance(children, (dict, list)):
            # A file (children absent) or a broken payload: not a listing,
            # pagination does not apply — return the single page as is.
            return payload
        listing.accumulated.extend(_children_values(children))
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            if listing.base_payload is None:
                # Legacy single-page shape (no pagination object): return
                # untouched — byte-for-byte the pre-pagination behavior.
                return payload
            msg = (
                f"gofile /contents/{content_id}: страница {listing.page} ответила без metadata "
                "внутри пагинированного листинга — листинг неполный, продолжение запрещено."
            )
            raise GofileError(msg)
        if listing.base_payload is None:
            listing.base_payload = payload
        if not metadata.get("hasNextPage"):
            self._check_listing_total(content_id, metadata, len(listing.accumulated))
            (listing.base_payload.get("data") or {})["children"] = listing.accumulated
            return listing.base_payload
        listing.page += 1
        if listing.page > _MAX_CONTENTS_PAGES:
            msg = (
                f"gofile /contents/{content_id}: сервер не остановил пагинацию "
                f"за {_MAX_CONTENTS_PAGES} страниц (hasNextPage всегда true) — "
                "остановлено как защита от зацикливания API."
            )
            raise GofileError(msg)
        return None

    @staticmethod
    def _check_listing_total(
        content_id: str,
        metadata: dict[str, Any],
        collected: int,
    ) -> None:
        """Cross-check metadata.totalCount against the accumulated children."""
        total = metadata.get("totalCount")
        if isinstance(total, int) and not isinstance(total, bool) and total != collected:
            msg = (
                f"gofile /contents/{content_id}: metadata.totalCount={total}, "
                f"собрано {collected} детей — листинг неполный "
                "(расхождение с контрактом API). Продолжение запрещено."
            )
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
        data = payload.get("data") or {}  # live API may send explicit `"data": null`
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
                raw_size = data.get("size")
                size = int(raw_size) if raw_size is not None and str(raw_size).isdigit() else None
                md5 = str(data.get("md5")) if data.get("md5") is not None else None
                return [
                    ResolvedChild(
                        name=data.get("name", ""),
                        link=data.get("link", ""),
                        size=size,
                        md5=md5,
                    )
                ]
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

    @staticmethod
    def _children_list(children: Any) -> list[dict[str, Any]]:  # noqa: ANN401 - live payload
        """
        Normalize gofile's children payload for the walk.

        LIVE API (verified 2026-10-28): a dict keyed by child id; older
        documented shapes used a plain list — both normalize here.
        """
        if isinstance(children, dict):
            return [child for child in children.values() if isinstance(child, dict)]
        if isinstance(children, list):
            return [child for child in children if isinstance(child, dict)]
        msg = "gofile: повреждённые данные children (ожидается dict или list)."
        raise GofileError(msg)

    def _walk(
        self,
        children: Any,  # noqa: ANN401 - live API payload
        found: list[ResolvedChild],
        reporter: Reporter | None,
        depth: int,
    ) -> None:
        if depth > _MAX_FOLDER_DEPTH:
            msg = "Слишком глубокая вложенность папки gofile (лимит 10 уровней)."
            raise GofileError(msg)
        normalized = self._children_list(children)
        folders = sorted(
            (c for c in normalized if c.get("type") == "folder"),
            key=lambda c: str(c.get("name", "")),
        )
        files = sorted(
            (c for c in normalized if c.get("type") != "folder"),
            key=lambda c: str(c.get("name", "")),
        )
        for folder in folders:
            sub = self.fetch_contents(str(folder.get("id", "")))
            sub_children = (sub.get("data") or {}).get("children")  # "data": null tolerated
            if not sub_children:
                # null / {} / [] — an empty subfolder contributes nothing (skip note),
                # it is not a corruption abort (same policy as the root normalizer).
                if reporter is not None:
                    reporter.note(f"Пропуск (пустая папка): {folder.get('name')}")
                continue
            # Same normalizer as the root: live API shapes are dict-keyed or list.
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
                msg = f"gofile: у файла «{name}» нет ссылки link (повреждённые данные)."
                raise GofileError(msg)
            raw_size = child.get("size")
            size = int(raw_size) if raw_size is not None and str(raw_size).isdigit() else None
            raw_md5 = child.get("md5")
            md5 = str(raw_md5) if raw_md5 is not None else None
            found.append(ResolvedChild(name=name, link=str(link), size=size, md5=md5))

    # -- file download ------------------------------------------------------
    # Transient-failure classes for the retry ladder (live-batch wave):
    # HTTP 502/503/504 from the store, connection breaks mid-stream (urllib3
    # surfaces them as requests' ConnectionError / ChunkedEncodingError, the
    # owner's screenshot shows "Connection broken: IncompleteRead(...)"), and
    # a raw IncompleteRead / ProtocolError in case a layer skips the wrapping.
    TRANSIENT_HTTP_STATUSES = frozenset({502, 503, 504})
    _TRANSIENT_EXCEPTIONS: tuple[type[Exception], ...] = (
        requests.exceptions.ConnectionError,
        requests.exceptions.ChunkedEncodingError,
        http.client.IncompleteRead,
        urllib3.exceptions.ProtocolError,
    )

    def download_file(  # noqa: PLR0913 - retry-wiring bundle, mirrors upload_file
        self,
        link: str,
        destination: Path,
        progress: Callable[[int, int], None] | None = None,
        chunk_size: int = 1024 * 1024,
        *,
        max_attempts: int = 3,
        backoff_base: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[], float] = random.random,
        on_retry: Callable[[int, int, str], None] | None = None,
    ) -> None:
        """
        GET <link> with accountToken cookie; .part + atomic rename (run.py:945-1000).

        Transient failures (HTTP 502/503/504, ConnectionError,
        ChunkedEncodingError, IncompleteRead) are retried up to ``max_attempts``
        (default 3) with exponential backoff ``backoff_base * 2**(attempt-1) +
        jitter`` (1/2/4 s + jitter at defaults); every retry prints nothing by
        itself — the caller wires ``on_retry(attempt, total, reason)`` for the
        Russian warn line. Range/206 resume: store-*.gofile.io honors Range
        (verified against the owner's gofile-dl run.py:967-977, which resumes
        with ``Range: bytes={size}-``, and the wider gofile downloader
        ecosystem), so a mid-stream failure resumes from the .part byte
        offset: a 206 answer appends, a 200 answer (Range ignored) restarts
        from scratch, 416 drops the partial and retries. The .part file is
        deleted after the final attempt and on permanent (non-transient)
        failures — a broken file never lands under the final name.
        """
        self.ensure_token()
        host = urlparse(link).hostname or ""
        if not host:
            msg = f"Некорректная ссылка для скачивания: {link}"
            raise GofileError(msg)
        self.session.pin(host)
        temp = destination.with_name(destination.name + ".part")
        for attempt in range(1, max_attempts + 1):
            offset = temp.stat().st_size if temp.is_file() else 0
            try:
                self._download_attempt(link, temp, destination, progress, chunk_size, offset)
            except (GofileTransientError, *self._TRANSIENT_EXCEPTIONS) as exc:
                if attempt == max_attempts:
                    temp.unlink(missing_ok=True)
                    msg = (
                        f"Ошибка скачивания {link}: не удалось после {max_attempts} попыток: {exc}"
                    )
                    raise GofileError(msg) from exc
                delay = backoff_base * 2 ** (attempt - 1) + jitter()
                if on_retry is not None:
                    on_retry(attempt, max_attempts, str(exc))
                sleep(delay)
            except GofileError:
                # Permanent refusal (403/404/…): no retry makes it transient.
                temp.unlink(missing_ok=True)
                raise
            except (requests.RequestException, OSError) as exc:
                # Non-transient network residue (SSL, timeouts, socket I/O):
                # the loud-error contract wraps them in Russian, temp is junk.
                temp.unlink(missing_ok=True)
                msg = f"Ошибка скачивания {link}: {exc}"
                raise GofileError(msg) from exc
            else:
                return  # complete attempt: .part renamed onto the destination
        msg = f"Ошибка скачивания {link}: исчерпаны попытки ({max_attempts})."
        raise GofileError(msg)  # pragma: no cover - loop returns or raises above

    def _download_attempt(  # noqa: PLR0913, PLR0917 - one-attempt bundle, mirrors download_file
        self,
        link: str,
        temp: Path,
        destination: Path,
        progress: Callable[[int, int], None] | None,
        chunk_size: int,
        offset: int,
    ) -> None:
        """One download attempt; resumes from ``offset`` when the server honors it."""
        headers = {
            "Cookie": f"accountToken={self.token}",
            "User-Agent": self.user_agent,
            "Referer": "https://gofile.io/",
        }
        if offset > 0:
            headers["Range"] = f"bytes={offset}-"
        with adaptive_tls_request(
            self.session.get,
            url=link,
            strict_tls=self.strict_tls,
            headers=headers,
            stream=True,
            timeout=(10, 300),
        ) as response:
            if response.status_code in self.TRANSIENT_HTTP_STATUSES:
                msg = f"Скачивание {link}: HTTP {response.status_code}."
                raise GofileTransientError(msg)
            if response.status_code == _HTTP_RANGE_NOT_SATISFIABLE:
                # partial ≥ full size: drop the stale partial and retry fresh.
                temp.unlink(missing_ok=True)
                msg = f"Скачивание {link}: HTTP 416 (диапазон не удовлетворим) — запуск заново."
                raise GofileTransientError(msg)
            if response.status_code not in {_HTTP_OK, _HTTP_PARTIAL_CONTENT}:
                msg = f"Скачивание {link}: HTTP {response.status_code}."
                raise GofileError(msg)
            resume = response.status_code == _HTTP_PARTIAL_CONTENT
            content_length = int(response.headers.get("Content-Length", 0))
            total = offset + content_length if resume else content_length
            done = offset if resume else 0
            # 200 after a resume request = Range ignored → truncate, start over.
            with temp.open("ab" if resume else "wb") as handle:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    handle.write(chunk)
                    done += len(chunk)
                    if progress is not None:
                        progress(done, total)
        temp.rename(destination)  # atomic: only complete files land in input/
