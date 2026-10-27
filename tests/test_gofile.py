"""gofile client tests: wt vectors (source), token lifecycle, folder expansion."""

import hashlib
from pathlib import Path
from typing import Self

import pytest

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
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self) -> dict:
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
        client = _client(fake, state_path=tmp_path / "g.toml")
        with pytest.raises(GofileError, match="wt_salt"):
            client.fetch_contents("abc")
        assert len(fake.calls) == 2


def _folder_payload(children: list[dict], *, password_status: str = "passwordOk") -> dict:
    return {
        "status": "ok",
        "data": {"type": "folder", "children": children, "passwordStatus": password_status},
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

    def test_child_missing_link_is_loud(self, tmp_path: Path) -> None:
        save_guest("g", tmp_path / "g.toml")
        fake = _FakeSession()
        fake._queue("contents/x", _folder_payload([{"type": "file", "name": "no_link.mkv"}]))
        client = _client(fake, state_path=tmp_path / "g.toml")
        with pytest.raises(GofileError, match=r"нет ссылки link"):
            client.expand_entries("x")


class TestDoH:
    def test_ok(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            gofile.requests,
            "get",
            lambda *a, **kw: _FakeResponse(
                {
                    "Status": 0,
                    "Answer": [{"type": 1, "data": "203.0.113.1"}, {"type": 28, "data": "::1"}],
                }
            ),
        )
        assert resolve_doh("api.gofile.io") == ["203.0.113.1"]

    def test_status_not_zero_loud(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(gofile.requests, "get", lambda *a, **kw: _FakeResponse({"Status": 3}))
        with pytest.raises(GofileError, match="Status=3"):
            resolve_doh("nope.example")

    def test_no_fallback_on_network_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*a: object, **kw: object) -> object:
            raise gofile.requests.ConnectionError("dead")

        monkeypatch.setattr(gofile.requests, "get", boom)
        with pytest.raises(GofileError, match="DoH"):
            resolve_doh("api.gofile.io")


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
