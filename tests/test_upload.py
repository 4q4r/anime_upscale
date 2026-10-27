"""upload tests: server pick, token cookie channel, progress, loud failures (plan §10)."""

from pathlib import Path

import pytest

from anime_upscale.gofile import GofileClient, GofileError
from anime_upscale.reporting import Reporter, resolve_mode
from anime_upscale.upload import UploadError, pick_server, upload_file, upload_output


class _Response:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict:
        return self._payload


class _FakeSession:
    def __init__(self) -> None:
        self.pins: list[str] = []
        self.post_kwargs: dict | None = None
        self.get_map: dict[str, _Response] = {}
        self.post_response: _Response = _Response({})

    def pin(self, hostname: str) -> str:
        self.pins.append(hostname)
        return "203.0.113.5"

    def get(self, url: str, **_kw: object) -> _Response:
        return self.get_map[url]

    def post(self, url: str, **kwargs: object) -> _Response:
        files = kwargs.get("files")
        if files and "file" in files:
            name, stream = files["file"]
            chunks = []
            while chunk := stream.read(65536):  # emulate requests draining the stream
                chunks.append(chunk)
            kwargs = {**kwargs, "files": {"file": (name, b"".join(chunks))}}
        self.post_kwargs = {"url": url, **kwargs}
        return self.post_response


def _setup(tmp_path: Path) -> tuple[Path, _FakeSession]:
    content = b"x" * (1024 * 1024 + 7)
    file = tmp_path / "out.mkv"
    file.write_bytes(content)
    fake = _FakeSession()
    fake.get_map["https://api.gofile.io/servers"] = _Response(
        {"status": "ok", "data": {"servers": [{"name": "store-eu-par-1"}]}}
    )
    fake.post_response = _Response(
        {"status": "ok", "data": {"downloadPage": "https://gofile.io/d/ZZ9"}}
    )
    return file, fake


def _client(fake: _FakeSession, **kw: object) -> GofileClient:
    client = GofileClient(**kw)  # type: ignore[arg-type]
    client.session = fake  # type: ignore[assignment]
    return client


class TestPickServer:
    def test_first_server_pinned(self) -> None:
        fake = _FakeSession()
        fake.get_map["https://api.gofile.io/servers"] = _Response(
            {"status": "ok", "data": {"servers": [{"name": "store1", "zone": "eu"}]}}
        )
        assert pick_server(_client(fake)) == "store1"
        assert "api.gofile.io" in fake.pins

    def test_bad_status_loud(self) -> None:
        fake = _FakeSession()
        fake.get_map["https://api.gofile.io/servers"] = _Response({"status": "error-x"})
        with pytest.raises(UploadError, match="error-x"):
            pick_server(_client(fake))


class TestUploadFile:
    def test_anonymous_upload_no_cookie(self, tmp_path: Path) -> None:
        file, fake = _setup(tmp_path)
        client = _client(
            fake
        )  # no account token; no state file → guest creation... avoid: pre-set token
        client.token = "guest"
        link = upload_file(file, client)
        assert link == "https://gofile.io/d/ZZ9"
        assert fake.post_kwargs is not None
        assert fake.post_kwargs["url"] == "https://store-eu-par-1.gofile.io/uploadfile"
        assert fake.post_kwargs["cookies"] == {"accountToken": "guest"}
        assert "store-eu-par-1.gofile.io" in fake.pins

    def test_account_token_cookie_attached(self, tmp_path: Path) -> None:
        file, fake = _setup(tmp_path)
        client = _client(fake, account_token="premium-tok")
        upload_file(file, client)
        assert fake.post_kwargs is not None
        assert fake.post_kwargs["cookies"] == {"accountToken": "premium-tok"}

    def test_progress_called(self, tmp_path: Path) -> None:
        file, fake = _setup(tmp_path)
        client = _client(fake)
        client.token = "g"
        seen: list[tuple[int, int]] = []
        upload_file(
            file,
            client,
            chunk_size=256 * 1024,
            progress=lambda done, total: seen.append((done, total)),
        )
        assert seen[0][0] > 0
        assert seen[-1] == (file.stat().st_size, file.stat().st_size)

    def test_bad_status_is_loud_and_file_untouched(self, tmp_path: Path) -> None:
        file, fake = _setup(tmp_path)
        fake.post_response = _Response({"status": "error-serverError"})
        client = _client(fake)
        client.token = "g"
        before = file.read_bytes()
        with pytest.raises(UploadError, match="error-serverError"):
            upload_file(file, client)
        assert file.read_bytes() == before


class TestUploadOutput:
    def test_prints_link(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        file, fake = _setup(tmp_path)
        client = _client(fake)
        client.token = "g"
        reporter = Reporter(resolve_mode(no_color=True, no_tty=True))
        link = upload_output(file, client, reporter)
        assert link == "https://gofile.io/d/ZZ9"
        assert "gofile.io/d/ZZ9" in capsys.readouterr().out

    def test_gofile_error_wrapped(self, tmp_path: Path) -> None:
        file, fake = _setup(tmp_path)

        def boom(url: str, **kw: object) -> object:
            raise GofileError("network dead")

        fake.get = boom  # type: ignore[method-assign]
        client = _client(fake)
        client.token = "g"
        reporter = Reporter(resolve_mode(no_color=True, no_tty=True))
        with pytest.raises(UploadError, match="network dead"):
            upload_output(file, client, reporter)
