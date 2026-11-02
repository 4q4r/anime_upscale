"""gofile client tests: wt vectors (source), token lifecycle, folder expansion."""

import hashlib
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from pathlib import Path
from typing import Self

import pytest
import requests

from anime_upscale import gofile
from anime_upscale.gofile import (
    CONTENTS_QUERY_PARAMS,
    GofileClient,
    GofileError,
    content_headers,
    generate_website_token,
    load_saved_guest,
    resolve_doh,
    save_guest,
    state_file_path,
)


class TestWtFormula:
    def test_matches_source_formula_verbatim(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(gofile.time, "time", lambda: 0.0)
        token = generate_website_token("tok", user_agent="UA", language="en", salt="s")
        expected = hashlib.sha256(b"UA::en::tok::0::s").hexdigest()
        assert token == expected
        assert len(token) == 64

    def test_window_offset_minus_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(gofile.time, "time", lambda: 14399.0)  # window 0 boundary
        assert generate_website_token("t", user_agent="U", language="L", salt="S") == (
            hashlib.sha256(b"U::L::t::0::S").hexdigest()
        )
        assert generate_website_token(
            "t", window_offset=-1, user_agent="U", language="L", salt="S"
        ) == (hashlib.sha256(b"U::L::t::-1::S").hexdigest())

    def test_content_headers_shape(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(gofile.time, "time", lambda: 0.0)
        headers = content_headers("tok", user_agent="UA", language="en-US", salt="12af056dacea0b")
        assert headers["Authorization"] == "Bearer tok"
        assert (
            headers["X-Website-Token"]
            == hashlib.sha256(b"UA::en-US::tok::0::12af056dacea0b").hexdigest()
        )
        assert headers["X-BL"] == "en-US"
        assert headers["Accept"] == "*/*"
        assert headers["Origin"] == "https://gofile.io"
        assert headers["Referer"] == "https://gofile.io/"
        assert headers["User-Agent"] == "UA"

    def test_contents_query_params_verbatim(self) -> None:
        assert CONTENTS_QUERY_PARAMS == {
            "contentFilter": "",
            "page": 1,
            "pageSize": 1000,
            "sortField": "createTime",
            "sortDirection": -1,
        }


class TestStateFile:
    def test_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "guest.toml"
        save_guest("abc123", path)
        assert load_saved_guest(path) == "abc123"
        raw = path.read_text(encoding="utf-8")
        assert 'token = "abc123"' in raw
        assert "created = " in raw

    def test_corrupt_file_is_loud_error(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.toml"
        path.write_text("token = [broken", encoding="utf-8")
        with pytest.raises(GofileError, match=r"broken\.toml"):
            load_saved_guest(path)

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert load_saved_guest(tmp_path / "absent.toml") == ""

    def test_save_is_atomic_no_tmp_leftover(self, tmp_path: Path) -> None:
        """M4: write via temp + os.replace — no partial file at the state path."""
        path = tmp_path / "guest.toml"
        save_guest("abc123", path)
        assert load_saved_guest(path) == "abc123"
        assert not (tmp_path / "guest.toml.tmp").exists()

    def test_save_sets_owner_only_permissions(self, tmp_path: Path) -> None:
        """M4: the token is a credential — the file must be 0600 regardless of umask."""
        import os

        path = tmp_path / "guest.toml"
        old_umask = os.umask(0o022)  # prove chmod, not umask luck
        try:
            save_guest("abc123", path)
        finally:
            os.umask(old_umask)
        assert path.stat().st_mode & 0o777 == 0o600

    def test_temp_created_owner_only_no_world_readable_window(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """PR1-review Minor: write_text created the .tmp world-readable (0644)
        and only chmod'ed AFTER the content hit disk — the token sat exposed
        for the write duration. The temp must be CREATED at 0o600 (os.open
        mode), closing the permission window entirely."""
        import os

        real_open = os.open
        creations: list[tuple[str, int, int]] = []

        def spy_open(path: object, flags: int, mode: int = 0o777, **kw: object) -> int:
            if str(path).endswith(".tmp") and flags & os.O_CREAT:
                creations.append((str(path), flags, mode))
            return real_open(path, flags, mode)  # type: ignore[arg-type]

        monkeypatch.setattr(os, "open", spy_open)
        path = tmp_path / "guest.toml"
        save_guest("abc123", path)
        # Round-trip stays intact through the new write path.
        assert load_saved_guest(path) == "abc123"
        assert creations, "temp file must be created via os.open(O_CREAT), not write_text"
        assert creations[0][2] == 0o600, creations

    def test_platform_state_paths(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr(gofile, "_is_windows", lambda: False)
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
        assert state_file_path() == tmp_path / "state" / "anime_upscale" / "gofile_guest.toml"
        monkeypatch.setattr(gofile, "_is_windows", lambda: True)
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "AppData" / "Local"))
        assert state_file_path() == (
            tmp_path / "AppData" / "Local" / "anime_upscale" / "state" / "gofile_guest.toml"
        )


class _FakeResponse:
    def __init__(
        self,
        payload: dict,
        status_code: int = 200,
        *,
        headers: dict[str, str] | None = None,
        json_raises: bool = False,
    ) -> None:
        self._payload = payload
        self.status_code = status_code
        # Preserve a subclass's class-level headers (the download fake sets them
        # that way); only an explicit argument overrides them.
        if headers is not None:
            self.headers = headers
        elif not hasattr(type(self), "headers"):
            self.headers = {}
        self._json_raises = json_raises
        self.text = payload if isinstance(payload, str) else str(payload)

    def json(self) -> dict:
        if self._json_raises:
            msg = "Expecting value: line 1 column 1 (char 0)"
            raise ValueError(msg)
        return self._payload


class _FakeSession:
    """Records calls; returns queued payloads per (method, url-substring)."""

    def __init__(self) -> None:
        self.pins: list[str] = []
        self.responses: dict[str, list[_FakeResponse]] = {}
        self.calls: list[tuple[str, str]] = []

    def pin(self, hostname: str) -> str:
        self.pins.append(hostname)
        return "203.0.113.7"

    def _queue(self, key: str, payload: "_FakeResponse | dict") -> None:
        response = payload if isinstance(payload, _FakeResponse) else _FakeResponse(payload)
        self.responses.setdefault(key, []).append(response)

    def get(self, url: str, **_kw: object) -> _FakeResponse:
        self.calls.append(("GET", url))
        for key, bucket in self.responses.items():
            if key in url and bucket:
                return bucket.pop(0)
        raise AssertionError(f"Unexpected GET {url}")

    def post(self, url: str, **_kw: object) -> _FakeResponse:
        self.calls.append(("POST", url))
        for key, bucket in self.responses.items():
            if key in url and bucket:
                return bucket.pop(0)
        raise AssertionError(f"Unexpected POST {url}")


def _client(fake: _FakeSession, **kw: object) -> GofileClient:
    client = GofileClient(**kw)  # type: ignore[arg-type]
    client.session = fake  # type: ignore[assignment]
    return client


class _FakeReporter:
    def __init__(self) -> None:
        self.notes: list[str] = []

    def note(self, text: str) -> None:
        self.notes.append(text)


class TestTokenLifecycle:
    def test_account_token_wins_and_state_not_touched(self, tmp_path: Path) -> None:
        fake = _FakeSession()
        client = _client(fake, account_token="premium", state_path=tmp_path / "g.toml")
        assert client.ensure_token() == "premium"
        assert not (tmp_path / "g.toml").exists()
        assert fake.calls == []  # no guest creation call

    def test_saved_guest_reused(self, tmp_path: Path) -> None:
        state = save_guest("saved-guest", tmp_path / "g.toml")
        fake = _FakeSession()
        client = _client(fake, state_path=state)
        assert client.ensure_token() == "saved-guest"
        assert fake.calls == []

    def test_guest_created_and_saved(self, tmp_path: Path) -> None:
        fake = _FakeSession()
        fake._queue(
            "api.gofile.io/accounts",
            _FakeResponse({"status": "ok", "data": {"token": "new-guest"}}),
        )
        state = tmp_path / "sub" / "g.toml"
        client = _client(fake, state_path=state)
        assert client.ensure_token() == "new-guest"
        assert load_saved_guest(state) == "new-guest"

    def test_guest_creation_failure_is_loud(self, tmp_path: Path) -> None:
        fake = _FakeSession()
        fake._queue("accounts", _FakeResponse({"status": "error-serverError"}, status_code=500))
        client = _client(fake, state_path=tmp_path / "g.toml")
        with pytest.raises(GofileError, match="гостевой токен"):
            client.ensure_token()

    def test_guest_creation_data_null_is_loud_not_attribute_error(self, tmp_path: Path) -> None:
        """M1: `"data": null` (explicit key, null value) must hit the loud branch."""
        fake = _FakeSession()
        fake._queue("accounts", _FakeResponse({"status": "ok", "data": None}))
        client = _client(fake, state_path=tmp_path / "g.toml")
        with pytest.raises(GofileError, match="гостевой токен"):
            client.ensure_token()

    def test_guest_creation_connection_error_is_gofile_error(self, tmp_path: Path) -> None:
        """M1: pre-flight network failure = loud GofileError, never a raw traceback."""
        fake = _FakeSession()

        def boom(url: str, **_kw: object) -> object:
            raise requests.ConnectionError("Connection reset by peer")

        fake.post = boom  # type: ignore[method-assign]
        client = _client(fake, state_path=tmp_path / "g.toml")
        with pytest.raises(GofileError, match="Connection reset by peer"):
            client.ensure_token()


class TestFetchContents:
    def test_ok(self, tmp_path: Path) -> None:
        save_guest("g", tmp_path / "g.toml")
        fake = _FakeSession()
        fake._queue("contents", _FakeResponse({"status": "ok", "data": {"type": "file"}}))
        client = _client(fake, state_path=tmp_path / "g.toml")
        assert client.fetch_contents("abc")["data"]["type"] == "file"

    def test_not_found_terminal(self, tmp_path: Path) -> None:
        save_guest("g", tmp_path / "g.toml")
        fake = _FakeSession()
        fake._queue("contents", _FakeResponse({"status": "error-notFound"}))
        client = _client(fake, state_path=tmp_path / "g.toml")
        with pytest.raises(GofileError, match="не найдено"):
            client.fetch_contents("gone")

    def test_rate_limit_retries_then_loud(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        save_guest("g", tmp_path / "g.toml")
        monkeypatch.setattr(gofile.time, "sleep", lambda _s: None)
        fake = _FakeSession()
        for _ in range(4):
            fake._queue("contents", _FakeResponse({"status": "error-rateLimit"}))
        client = _client(fake, state_path=tmp_path / "g.toml")
        with pytest.raises(GofileError, match="error-rateLimit"):
            client.fetch_contents("abc")
        assert len(fake.calls) == 4

    def test_not_premium_retries_previous_window_then_names_salt_recovery(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        save_guest("g", tmp_path / "g.toml")
        monkeypatch.setattr(gofile.time, "sleep", lambda _s: None)
        fake = _FakeSession()
        fake._queue("contents", _FakeResponse({"status": "error-notPremium"}))
        fake._queue("contents", _FakeResponse({"status": "error-notPremium"}))
        fake._queue("accounts", _FakeResponse({"status": "ok", "data": {"token": "fresh"}}))
        fake._queue("contents", _FakeResponse({"status": "error-notPremium"}))
        client = _client(fake, state_path=tmp_path / "g.toml")
        with pytest.raises(GofileError, match="wt_salt"):
            client.fetch_contents("abc")
        assert sum("contents" in url for _, url in fake.calls) == 3

    def test_connection_error_is_gofile_error(self, tmp_path: Path) -> None:
        """M1: §11.3 — connection reset = loud error quoting that diagnosis."""
        save_guest("g", tmp_path / "g.toml")
        fake = _FakeSession()

        def boom(url: str, **_kw: object) -> object:
            raise requests.ConnectionError("Connection reset by peer")

        fake.get = boom  # type: ignore[method-assign]
        client = _client(fake, state_path=tmp_path / "g.toml")
        with pytest.raises(GofileError, match="Connection reset by peer"):
            client.fetch_contents("abc")

    def test_data_null_is_loud_not_attribute_error(self, tmp_path: Path) -> None:
        """M1: `"data": null` payload must reach the loud children error."""
        save_guest("g", tmp_path / "g.toml")
        fake = _FakeSession()
        fake._queue("contents", _FakeResponse({"status": "ok", "data": None}))
        client = _client(fake, state_path=tmp_path / "g.toml")
        with pytest.raises(GofileError, match="children"):
            client.expand_entries("abc")


class TestFetchContentsTokenRecovery:
    """Live bug 2026-11-02: a stale guest token returned 401 + error-wrongToken,
    but the client checked HTTP before JSON and reported a useless «HTTP 401»."""

    def test_wrong_token_refreshes_guest_once_and_retries(self, tmp_path: Path) -> None:
        state = save_guest("stale-guest", tmp_path / "g.toml")
        fake = _FakeSession()
        fake._queue("contents", _FakeResponse({"status": "error-wrongToken"}, status_code=401))
        fake._queue(
            "accounts",
            _FakeResponse({"status": "ok", "data": {"token": "fresh-guest"}}),
        )
        fake._queue("contents", _FakeResponse({"status": "ok", "data": {"type": "file"}}))
        client = _client(fake, state_path=state)
        assert client.fetch_contents("abc")["data"]["type"] == "file"
        assert load_saved_guest(state) == "fresh-guest"
        assert [call[1] for call in fake.calls].count("https://api.gofile.io/accounts") == 1
        assert sum("contents" in url for _, url in fake.calls) == 2

    def test_error_token_also_refreshes_guest(self, tmp_path: Path) -> None:
        state = save_guest("stale-guest", tmp_path / "g.toml")
        fake = _FakeSession()
        fake._queue("contents", _FakeResponse({"status": "error-token"}, status_code=401))
        fake._queue("accounts", _FakeResponse({"status": "ok", "data": {"token": "fresh"}}))
        fake._queue("contents", _FakeResponse({"status": "ok", "data": {"type": "file"}}))
        client = _client(fake, state_path=state)
        assert client.fetch_contents("abc")["status"] == "ok"
        assert client.token == "fresh"

    def test_second_wrong_token_is_loud_and_names_state_file(self, tmp_path: Path) -> None:
        state = save_guest("stale-guest", tmp_path / "g.toml")
        fake = _FakeSession()
        fake._queue("contents", _FakeResponse({"status": "error-wrongToken"}, status_code=401))
        fake._queue("accounts", _FakeResponse({"status": "ok", "data": {"token": "fresh"}}))
        fake._queue("contents", _FakeResponse({"status": "error-wrongToken"}, status_code=401))
        client = _client(fake, state_path=state)
        with pytest.raises(GofileError, match="error-wrongToken") as exc_info:
            client.fetch_contents("abc")
        assert str(state) in str(exc_info.value)
        assert sum("accounts" in url for _, url in fake.calls) == 1

    def test_account_token_is_never_replaced_by_a_guest(self, tmp_path: Path) -> None:
        state = tmp_path / "g.toml"
        fake = _FakeSession()
        fake._queue("contents", _FakeResponse({"status": "error-wrongToken"}, status_code=401))
        client = _client(fake, account_token="premium-token", state_path=state)
        with pytest.raises(GofileError, match="аккаунт") as exc_info:
            client.fetch_contents("abc")
        assert "premium-token" not in str(exc_info.value)
        assert not state.exists()
        assert not any("accounts" in url for _, url in fake.calls)

    def test_not_premium_order_window_minus_one_then_refresh_then_loud(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state = save_guest("guest", tmp_path / "g.toml")
        monkeypatch.setattr(gofile.time, "sleep", lambda _s: None)
        fake = _FakeSession()
        server_date = format_datetime(datetime.now(UTC) + timedelta(hours=1), usegmt=True)
        fake._queue("contents", _FakeResponse({"status": "error-notPremium"}))
        fake._queue("contents", _FakeResponse({"status": "error-notPremium"}))
        fake._queue("accounts", _FakeResponse({"status": "ok", "data": {"token": "fresh"}}))
        fake._queue(
            "contents",
            _FakeResponse({"status": "error-notPremium"}, headers={"Date": server_date}),
        )
        client = _client(fake, state_path=state)
        with pytest.raises(GofileError, match="wt_salt") as exc_info:
            client.fetch_contents("abc")
        message = str(exc_info.value)
        assert "заголовку Date" in message
        assert "clock" not in message.lower()  # Russian message, not an English log
        contents_calls = [url for _, url in fake.calls if "contents" in url]
        assert len(contents_calls) == 3
        assert sum("accounts" in url for _, url in fake.calls) == 1

    def test_not_premium_without_date_never_invents_an_offset(self, tmp_path: Path) -> None:
        state = save_guest("guest", tmp_path / "g.toml")
        fake = _FakeSession()
        fake._queue("contents", _FakeResponse({"status": "error-notPremium"}))
        fake._queue("contents", _FakeResponse({"status": "error-notPremium"}))
        fake._queue("accounts", _FakeResponse({"status": "ok", "data": {"token": "fresh"}}))
        fake._queue("contents", _FakeResponse({"status": "error-notPremium"}))
        client = _client(fake, state_path=state)
        with pytest.raises(GofileError, match="wt_salt") as exc_info:
            client.fetch_contents("abc")
        message = str(exc_info.value)
        assert "заголовок Date отсутствует" in message
        assert "Измеренный сдвиг" not in message

    def test_non_json_401_quotes_truncated_body(self, tmp_path: Path) -> None:
        save_guest("g", tmp_path / "g.toml")
        fake = _FakeSession()
        body = "<html><body>blocked by provider gateway</body></html>"
        fake._queue(
            "contents",
            _FakeResponse(body, status_code=401, json_raises=True),
        )
        client = _client(fake, state_path=tmp_path / "g.toml")
        with pytest.raises(GofileError, match="HTTP 401") as exc_info:
            client.fetch_contents("abc")
        assert "blocked by provider gateway" in str(exc_info.value)

    def test_unknown_status_is_loud(self, tmp_path: Path) -> None:
        save_guest("g", tmp_path / "g.toml")
        fake = _FakeSession()
        fake._queue("contents", _FakeResponse({"status": "error-mystery"}, status_code=418))
        client = _client(fake, state_path=tmp_path / "g.toml")
        with pytest.raises(GofileError, match="error-mystery"):
            client.fetch_contents("abc")

    def test_refresh_guest_is_public_and_persists(self, tmp_path: Path) -> None:
        state = save_guest("old", tmp_path / "g.toml")
        fake = _FakeSession()
        fake._queue("accounts", _FakeResponse({"status": "ok", "data": {"token": "new"}}))
        client = _client(fake, state_path=state)
        assert client.refresh_guest() == "new"
        assert client.token == "new"
        assert load_saved_guest(state) == "new"


def _folder_payload(children: list[dict], *, password_status: str = "passwordOk") -> dict:
    """Fixture in the REAL live API shape: children is a DICT keyed by child id
    (verified live 2026-10-28, commit e7bb5d1 + the smoke round-trip). Building
    the dict here forces every expansion test through _children_list in the
    production path — list-shaped fixtures masked the F1 bug."""
    children_dict = {child.get("id", f"child-{i}"): child for i, child in enumerate(children)}
    return {
        "status": "ok",
        "data": {"type": "folder", "children": children_dict, "passwordStatus": password_status},
    }


def _file(name: str, link: str = "https://store.gofile.io/download/x") -> dict:
    return {"type": "file", "name": name, "link": link, "id": name}


class TestFolderExpansion:
    def test_nested_expansion_deterministic_order_folders_first(self, tmp_path: Path) -> None:
        save_guest("g", tmp_path / "g.toml")
        fake = _FakeSession()
        fake._queue(
            "contents/root",
            _folder_payload(
                [
                    _file("b_video.mkv"),
                    {"type": "folder", "name": "z_folder", "id": "zf", "children": None},
                    _file("a_video.mp4"),
                    {"type": "folder", "name": "a_folder", "id": "af", "children": None},
                ]
            ),
        )
        fake._queue(
            "contents/af",
            _folder_payload([_file("nested_webm.webm")]),
        )
        fake._queue(
            "contents/zf",
            _folder_payload([_file("z.m4v"), _file("a.m4v")]),
        )
        reporter = _FakeReporter()
        client = _client(fake, state_path=tmp_path / "g.toml")
        children = client.expand_entries("root", reporter)
        names = [c.name for c in children]
        # Folders first (a_folder, z_folder — each alphabetical), then files (a_video, b_video);
        # inside folders: alphabetical (a.m4v, z.m4v; nested_webm).
        assert names == ["nested_webm.webm", "a.m4v", "z.m4v", "a_video.mp4", "b_video.mkv"]

    def test_video_filter_skips_with_notes(self, tmp_path: Path) -> None:
        save_guest("g", tmp_path / "g.toml")
        fake = _FakeSession()
        fake._queue(
            "contents/x",
            _folder_payload([_file("keep.mkv"), _file("image.jpg"), _file("nfo.txt")]),
        )
        reporter = _FakeReporter()
        client = _client(fake, state_path=tmp_path / "g.toml")
        children = client.expand_entries("x", reporter)
        assert [c.name for c in children] == ["keep.mkv"]
        assert "Пропуск (не видео): image.jpg" in reporter.notes
        assert "Пропуск (не видео): nfo.txt" in reporter.notes

    def test_empty_after_filter_is_loud(self, tmp_path: Path) -> None:
        save_guest("g", tmp_path / "g.toml")
        fake = _FakeSession()
        fake._queue("contents/x", _folder_payload([_file("a.jpg"), _file("b.png")]))
        client = _client(fake, state_path=tmp_path / "g.toml")
        with pytest.raises(GofileError, match="нет видеофайлов"):
            client.expand_entries("x")

    def test_password_protected_loud(self, tmp_path: Path) -> None:
        save_guest("g", tmp_path / "g.toml")
        fake = _FakeSession()
        fake._queue(
            "contents/x",
            _folder_payload([_file("a.mkv")], password_status="passwordWrong"),
        )
        client = _client(fake, state_path=tmp_path / "g.toml")
        with pytest.raises(GofileError, match="парол"):
            client.expand_entries("x")

    def test_nested_real_payload_shape(self, tmp_path: Path) -> None:
        """F1 fix-sensitivity guard: the EXACT live payload shape from the smoke
        (children dict keyed by UUID ids, full field set) walked two levels deep.
        Fails against the pre-fix implementation (strict list check on the
        subfolder branch) — revert-check documented in the report."""
        save_guest("g", tmp_path / "g.toml")
        root_children = {
            "e0ad0fde-1111-4ba8-8beb-000000000001": {
                "canAccess": True,
                "id": "e0ad0fde-1111-4ba8-8beb-000000000001",
                "parentFolder": "e0ad0fde-1111-4ba8-8beb-root",
                "type": "folder",
                "name": "season_01",
                "createTime": 1793143713,
                "modTime": 1793143713,
                "code": "AbCdE12",
                "public": True,
                "totalSize": 65562,
                "childrenCount": 1,
            },
            "e0ad0fde-2222-4ba8-8beb-000000000002": {
                "canAccess": True,
                "id": "e0ad0fde-2222-4ba8-8beb-000000000002",
                "parentFolder": "e0ad0fde-1111-4ba8-8beb-root",
                "type": "file",
                "name": "top.mkv",
                "createTime": 1793143713,
                "modTime": 1793143713,
                "code": "Zz9Yy8X",
                "size": 65562,
                "md5": "e410f6d44af0969c2f758594ac9aa63e",
                "mimetype": "video/x-matroska",
                "servers": ["store-eu-par-5"],
                "serverSelected": "store-eu-par-5",
                "link": "https://store-eu-par-5.gofile.io/download/web/topid/top.mkv",
            },
        }
        sub_children = {
            "e0ad0fde-3333-4ba8-8beb-000000000003": {
                "canAccess": True,
                "id": "e0ad0fde-3333-4ba8-8beb-000000000003",
                "parentFolder": "e0ad0fde-1111-4ba8-8beb-000000000001",
                "type": "file",
                "name": "nested.mkv",
                "createTime": 1793143713,
                "modTime": 1793143713,
                "code": "Qq1Ww2Ee",
                "size": 65562,
                "md5": "e410f6d44af0969c2f758594ac9aa63e",
                "mimetype": "video/x-matroska",
                "servers": ["store-eu-par-5"],
                "serverSelected": "store-eu-par-5",
                "link": "https://store-eu-par-5.gofile.io/download/web/nestedid/nested.mkv",
            },
        }
        fake = _FakeSession()
        fake._queue(
            "contents/rootid",
            {
                "status": "ok",
                "data": {
                    "type": "folder",
                    "name": "rootid",
                    "children": root_children,
                    "passwordStatus": "passwordOk",
                },
            },
        )
        fake._queue(
            "contents/e0ad0fde-1111-4ba8-8beb-000000000001",
            {
                "status": "ok",
                "data": {
                    "type": "folder",
                    "name": "season_01",
                    "children": sub_children,
                    "passwordStatus": "passwordOk",
                },
            },
        )
        client = _client(fake, state_path=tmp_path / "g.toml")
        children = client.expand_entries("rootid")
        assert [c.name for c in children] == ["nested.mkv", "top.mkv"]
        assert next(c.link for c in children).endswith("nested.mkv")

    def test_nested_dict_children_real_shape(self, tmp_path: Path) -> None:
        """F1 regression: subfolder children use the SAME dict normalizer as the
        root (the live API returns dict payloads at every nesting level)."""
        save_guest("g", tmp_path / "g.toml")
        fake = _FakeSession()
        fake._queue(
            "contents/root",
            _folder_payload(
                [
                    {"type": "folder", "name": "sub", "id": "subid", "children": None},
                ]
            ),
        )
        fake._queue(
            "contents/subid",
            _folder_payload([_file("inside.mkv", "https://s/in")]),
        )
        client = _client(fake, state_path=tmp_path / "g.toml")
        children = client.expand_entries("root")
        assert [c.name for c in children] == ["inside.mkv"]

    def test_empty_subfolder_skipped_with_note(self, tmp_path: Path) -> None:
        save_guest("g", tmp_path / "g.toml")
        fake = _FakeSession()
        fake._queue(
            "contents/root",
            _folder_payload(
                [
                    _file("a.mkv"),
                    {"type": "folder", "name": "void", "id": "voidid", "children": None},
                ]
            ),
        )
        fake._queue(
            "contents/voidid",
            {"status": "ok", "data": {"type": "folder", "children": {}}},
        )
        reporter = _FakeReporter()
        client = _client(fake, state_path=tmp_path / "g.toml")
        children = client.expand_entries("root", reporter)
        assert [c.name for c in children] == ["a.mkv"]
        assert "Пропуск (пустая папка): void" in reporter.notes

    def test_null_subfolder_children_skipped_with_note(self, tmp_path: Path) -> None:
        save_guest("g", tmp_path / "g.toml")
        fake = _FakeSession()
        fake._queue(
            "contents/root",
            _folder_payload(
                [
                    _file("real.mkv"),
                    {"type": "folder", "name": "ghost", "id": "ghostid", "children": None},
                ]
            ),
        )
        fake._queue(
            "contents/ghostid",
            {"status": "ok", "data": {"type": "folder", "children": None}},
        )
        reporter = _FakeReporter()
        client = _client(fake, state_path=tmp_path / "g.toml")
        children = client.expand_entries("root", reporter)
        assert [c.name for c in children] == ["real.mkv"]
        assert "Пропуск (пустая папка): ghost" in reporter.notes

    def test_child_missing_link_is_loud(self, tmp_path: Path) -> None:
        save_guest("g", tmp_path / "g.toml")
        fake = _FakeSession()
        fake._queue("contents/x", _folder_payload([{"type": "file", "name": "no_link.mkv"}]))
        client = _client(fake, state_path=tmp_path / "g.toml")
        with pytest.raises(GofileError, match=r"нет ссылки link"):
            client.expand_entries("x")

    def test_subfolder_data_null_skipped_as_empty(self, tmp_path: Path) -> None:
        """M1: subfolder payload with `"data": null` = empty folder, not AttributeError."""
        save_guest("g", tmp_path / "g.toml")
        fake = _FakeSession()
        fake._queue(
            "contents/root",
            _folder_payload(
                [
                    _file("real.mkv"),
                    {"type": "folder", "name": "ghost", "id": "ghostid", "children": None},
                ]
            ),
        )
        fake._queue("contents/ghostid", {"status": "ok", "data": None})
        reporter = _FakeReporter()
        client = _client(fake, state_path=tmp_path / "g.toml")
        children = client.expand_entries("root", reporter)
        assert [c.name for c in children] == ["real.mkv"]
        assert "Пропуск (пустая папка): ghost" in reporter.notes


def _doh_script(script: list[object]) -> tuple[object, list[str]]:
    """
    Build a requests.get fake serving `script` items in call order.

    Items are payload dicts (HTTP 200) or exceptions (transport failures);
    the fake logs every requested endpoint URL for ladder-order assertions.
    """
    calls: list[str] = []

    def fake_get(url: str, *a: object, **kw: object) -> object:
        calls.append(url)
        step = script[len(calls) - 1]
        if isinstance(step, Exception):
            raise step
        return _FakeResponse(step)

    return fake_get, calls


class TestDoH:
    def test_ok(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_get, calls = _doh_script(
            [
                {
                    "Status": 0,
                    "Answer": [{"type": 1, "data": "203.0.113.1"}, {"type": 28, "data": "::1"}],
                }
            ]
        )
        monkeypatch.setattr(gofile.requests, "get", fake_get)
        assert resolve_doh("api.gofile.io") == ["203.0.113.1"]
        # First endpoint answered — the ladder never advances.
        assert calls == [gofile.DOH_ENDPOINTS[0]]

    def test_status_not_zero_loud(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A valid answer with Status!=0 is final — no ladder advance (protocol answer)."""
        fake_get, calls = _doh_script([{"Status": 3}])
        monkeypatch.setattr(gofile.requests, "get", fake_get)
        with pytest.raises(GofileError, match="Status=3"):
            resolve_doh("nope.example")
        assert calls == [gofile.DOH_ENDPOINTS[0]]

    def test_ladder_ssl_on_domain_then_google_ip_endpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """#1 (dns.google domain) dies on SSL → #2 (Google by bare IP) wins; #3 not called."""
        fake_get, calls = _doh_script(
            [
                requests.exceptions.SSLError("self-signed certificate in certificate chain"),
                {"Status": 0, "Answer": [{"type": 1, "data": "203.0.113.1"}]},
            ]
        )
        monkeypatch.setattr(gofile.requests, "get", fake_get)
        assert resolve_doh("api.gofile.io") == ["203.0.113.1"]
        assert calls == [gofile.DOH_ENDPOINTS[0], gofile.DOH_ENDPOINTS[1]]

    def test_ladder_cloudflare_ip_endpoint_final_resort(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """#1/#2 transport-dead → #3 (Cloudflare by bare IP, application/dns-json) wins."""
        fake_get, calls = _doh_script(
            [
                requests.exceptions.ConnectionError("connection refused"),
                requests.exceptions.Timeout("timed out"),
                {"Status": 0, "Answer": [{"type": 1, "data": "198.51.100.9"}]},
            ]
        )
        monkeypatch.setattr(gofile.requests, "get", fake_get)
        assert resolve_doh("api.gofile.io") == ["198.51.100.9"]
        assert calls == list(gofile.DOH_ENDPOINTS)

    def test_ladder_all_fail_names_every_endpoint_and_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """All three dead → loud Russian error: each endpoint WITH its own failure reason."""
        fake_get, _ = _doh_script(
            [
                requests.exceptions.SSLError("self-signed certificate in certificate chain"),
                requests.exceptions.ConnectionError("connection refused"),
                requests.exceptions.Timeout("timed out"),
            ]
        )
        monkeypatch.setattr(gofile.requests, "get", fake_get)
        with pytest.raises(GofileError) as excinfo:
            resolve_doh("api.gofile.io")
        text = str(excinfo.value)
        for endpoint, reason in (
            (gofile.DOH_ENDPOINTS[0], "self-signed certificate in certificate chain"),
            (gofile.DOH_ENDPOINTS[1], "connection refused"),
            (gofile.DOH_ENDPOINTS[2], "timed out"),
        ):
            assert endpoint in text, text
            assert reason in text, text
        assert "перехват" in text, text  # MITM diagnosis line
        assert "Системный резолвер не используется" in text, text

    def test_status_not_zero_after_transport_fail_never_advances(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """#1 transport-dead, #2 answers Status=3 → final answer, #3 never contacted."""
        fake_get, calls = _doh_script(
            [requests.exceptions.SSLError("handshake failed"), {"Status": 3}]
        )
        monkeypatch.setattr(gofile.requests, "get", fake_get)
        with pytest.raises(GofileError, match="Status=3"):
            resolve_doh("api.gofile.io")
        assert calls == [gofile.DOH_ENDPOINTS[0], gofile.DOH_ENDPOINTS[1]]


class TestDownloadFile:
    def test_part_then_atomic_rename(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        save_guest("g", tmp_path / "g.toml")
        payload = b"VIDEODATA" * 8

        class _Stream(_FakeResponse):
            headers = {"Content-Length": str(len(payload))}

            def iter_content(self, chunk_size: int) -> bytes:
                yield payload

            def __enter__(self) -> Self:
                return self

            def __exit__(self, *a: object) -> None:
                return None

        client = _client(_FakeSession(), state_path=tmp_path / "g.toml")
        monkeypatch.setattr(client.session, "get", lambda url, **kw: _Stream({}), raising=False)
        # pin() needs a PinnedSession; our fake lacks it -> monkeypatch pin
        monkeypatch.setattr(client.session, "pin", lambda host: "203.0.113.9", raising=False)
        progress: list[tuple[int, int]] = []
        target = tmp_path / "out.bin"
        client.download_file(
            "https://store.gofile.io/download/x",
            target,
            progress=lambda done, total: progress.append((done, total)),
        )
        assert target.read_bytes() == payload
        assert not target.with_name("out.bin.part").exists()
        assert progress[-1] == (len(payload), len(payload))

    def test_failure_deletes_part(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        save_guest("g", tmp_path / "g.toml")

        def boom(*a: object, **kw: object) -> object:
            raise gofile.requests.ConnectionError("reset")

        client = _client(_FakeSession(), state_path=tmp_path / "g.toml")
        monkeypatch.setattr(client.session, "get", boom, raising=False)
        monkeypatch.setattr(client.session, "pin", lambda host: "203.0.113.9", raising=False)
        target = tmp_path / "out.bin"
        with pytest.raises(GofileError, match="скачивани"):
            client.download_file("https://store.gofile.io/download/x", target)
        assert not target.exists()
        assert not target.with_name("out.bin.part").exists()
