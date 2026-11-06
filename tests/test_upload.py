"""upload tests: server pick, token cookie channel, progress, loud failures (plan §10)."""

from pathlib import Path

import pytest

from anime_upscale.gofile import GofileClient, GofileError
from anime_upscale.reporting import Reporter, resolve_mode
from anime_upscale.upload import (
    AdaptiveStreamingMultipart,
    UploadError,
    pick_server,
    upload_file,
    upload_output,
)


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
        data = kwargs.get("data")
        if files and "file" in files:
            name, stream = files["file"]
            chunks = []
            while chunk := stream.read(65536):  # emulate requests draining the stream
                chunks.append(chunk)
            kwargs = {**kwargs, "files": {"file": (name, b"".join(chunks))}}
        elif (
            data is not None
            and hasattr(data, "__iter__")
            and not isinstance(data, (bytes, str, dict))
        ):
            # Emulate streaming body consumption by requests / socket sendall
            streamed = list(data)
            kwargs = {**kwargs, "data": b"".join(streamed)}
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
        {
            "status": "ok",
            "data": {
                "downloadPage": "https://gofile.io/d/ZZ9",
                "parentFolder": "folder-uuid-123",
            },
        }
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

    def test_data_null_is_loud_not_attribute_error(self) -> None:
        """M1: `"data": null` from /servers must hit the loud no-servers branch."""
        fake = _FakeSession()
        fake.get_map["https://api.gofile.io/servers"] = _Response({"status": "ok", "data": None})
        with pytest.raises(UploadError, match="без серверов"):
            pick_server(_client(fake))


class TestUploadFile:
    def test_anonymous_upload_no_cookie(self, tmp_path: Path) -> None:
        file, fake = _setup(tmp_path)
        client = _client(
            fake
        )  # no account token; no state file → guest creation... avoid: pre-set token
        client.token = "guest"
        link, parent = upload_file(file, client)
        assert link == "https://gofile.io/d/ZZ9"
        assert parent == "folder-uuid-123"
        assert fake.post_kwargs is not None
        assert fake.post_kwargs["url"] == "https://store-eu-par-1.gofile.io/uploadfile"
        assert fake.post_kwargs["cookies"] == {"accountToken": "guest"}
        assert "store-eu-par-1.gofile.io" in fake.pins
        assert "headers" in fake.post_kwargs
        assert "Content-Type" in fake.post_kwargs["headers"]
        assert "Content-Length" not in fake.post_kwargs["headers"]  # F11: requests derives it
        assert "multipart/form-data; boundary=" in fake.post_kwargs["headers"]["Content-Type"]
        assert isinstance(fake.post_kwargs["data"], bytes)
        # Check that file content is in the streamed multipart body
        assert file.read_bytes() in fake.post_kwargs["data"]

    def test_manual_content_length_header_absent(self, tmp_path: Path) -> None:
        """F11: requests overrides a manual Content-Length, so we must not send one."""
        file, fake = _setup(tmp_path)
        client = _client(fake)
        client.token = "g"
        upload_file(file, client)
        assert fake.post_kwargs is not None
        assert "Content-Length" not in fake.post_kwargs["headers"]

    def test_missing_parent_folder_is_loud(self, tmp_path: Path) -> None:
        """F12: official gofile contract always carries parentFolder; absence = loud."""
        file, fake = _setup(tmp_path)
        fake.post_response = _Response(
            {"status": "ok", "data": {"downloadPage": "https://gofile.io/d/AA1"}}
        )
        client = _client(fake)
        client.token = "g"
        with pytest.raises(UploadError, match="альбом невозможен"):
            upload_file(file, client)

    def test_account_token_cookie_attached(self, tmp_path: Path) -> None:
        file, fake = _setup(tmp_path)
        client = _client(fake, account_token="premium-tok")
        upload_file(file, client)
        assert fake.post_kwargs is not None
        assert fake.post_kwargs["cookies"] == {"accountToken": "premium-tok"}
        data = fake.post_kwargs["data"]
        assert b'name="token"\r\n\r\npremium-tok\r\n' in data

    def test_folder_id_attached(self, tmp_path: Path) -> None:
        file, fake = _setup(tmp_path)
        client = _client(fake, account_token="tok")
        upload_file(file, client, folder_id="fld-12345")
        assert fake.post_kwargs is not None
        data = fake.post_kwargs["data"]
        # Strict ordering: token before folderId before file
        token_idx = data.find(b'name="token"')
        folder_idx = data.find(b'name="folderId"')
        file_idx = data.find(b'name="file"')
        assert -1 < token_idx < folder_idx < file_idx
        assert b"fld-12345" in data

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

    def test_data_null_ok_payload_is_loud_not_attribute_error(self, tmp_path: Path) -> None:
        """M1: `"data": null` in a 200-OK upload response = loud, not AttributeError."""
        file, fake = _setup(tmp_path)
        fake.post_response = _Response({"status": "ok", "data": None})
        client = _client(fake)
        client.token = "g"
        with pytest.raises(UploadError, match="downloadPage"):
            upload_file(file, client)


class TestUploadOutput:
    def test_prints_link(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        file, fake = _setup(tmp_path)
        client = _client(fake)
        client.token = "g"
        reporter = Reporter(resolve_mode(no_color=True, no_tty=True))
        link, parent = upload_output(file, client, reporter)
        assert link == "https://gofile.io/d/ZZ9"
        assert parent == "folder-uuid-123"
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


class TestAdaptiveStreamingMultipart:
    def test_len_matches_exact_bytes(self, tmp_path: Path) -> None:
        content = b"hello multipart world"
        file = tmp_path / "video.mkv"
        file.write_bytes(content)

        streamer = AdaptiveStreamingMultipart(
            file,
            token="secret_token",
            folder_id="target_folder",
            boundary="CustomBoundary123",
        )
        total_len = len(streamer)
        yielded_bytes = b"".join(list(streamer))
        assert len(yielded_bytes) == total_len

    def test_field_order_and_content_type(self, tmp_path: Path) -> None:
        content = b"fake video bytes"
        file = tmp_path / "episode1.mkv"
        file.write_bytes(content)

        streamer = AdaptiveStreamingMultipart(
            file,
            token="tok123",
            folder_id="fol456",
            boundary="BND",
        )
        payload = b"".join(list(streamer))
        tok_pos = payload.find(b'name="token"\r\n\r\ntok123')
        fol_pos = payload.find(b'name="folderId"\r\n\r\nfol456')
        file_pos = payload.find(b'name="file"; filename="episode1.mkv"')
        ct_pos = payload.find(b"Content-Type: video/x-matroska")

        assert tok_pos != -1
        assert fol_pos != -1
        assert file_pos != -1
        assert ct_pos != -1
        assert tok_pos < fol_pos < file_pos < ct_pos
        assert payload.endswith(b"--BND--\r\n")

    def test_adaptive_scaling_up_and_down(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Create a 20MB file
        chunk_data = b"0" * (1024 * 1024)
        file = tmp_path / "big.mkv"
        file.write_bytes(chunk_data * 20)

        # Mock time.monotonic to simulate fast network (0.01s elapsed per chunk)
        cur_time = [0.0]

        def fake_monotonic() -> float:
            cur_time[0] += 0.05
            return cur_time[0]

        monkeypatch.setattr("time.monotonic", fake_monotonic)

        streamer = AdaptiveStreamingMultipart(file, initial_chunk_size=1 * 1024 * 1024)
        chunks = list(streamer)
        # Check that chunk sizes scaled up
        # chunks: [0]=pre_headers, [1]=file_header, [2..N-1]=file chunks, [N]=closing
        file_chunks = chunks[1:-1]
        chunk_sizes = [len(c) for c in file_chunks]
        # At least one chunk should be larger than 1 MiB
        assert any(size > 1 * 1024 * 1024 for size in chunk_sizes)


class TestMultipartToctou:
    """F1 regression: length and bytes come from one pinned handle; parity is enforced."""

    def test_grow_streams_declared_prefix_with_valid_boundary(self, tmp_path: Path) -> None:
        content = b"A" * 1000
        file = tmp_path / "grow.mkv"
        file.write_bytes(content)

        streamer = AdaptiveStreamingMultipart(file, boundary="BND2")
        declared = len(streamer)
        with file.open("ab") as fh:  # concurrent writer after construction
            fh.write(b"B" * 500)

        payload = b"".join(streamer)
        assert len(payload) == declared  # exactly the declared length, not the grown size
        assert payload.endswith(b"\r\n--BND2--\r\n")  # body ends with a valid boundary
        assert content in payload
        assert b"B" * 500 not in payload  # grown tail is never streamed

    def test_shrink_raises_loud_runtime_error(self, tmp_path: Path) -> None:
        content = b"C" * 1000
        file = tmp_path / "shrink.mkv"
        file.write_bytes(content)

        streamer = AdaptiveStreamingMultipart(file, boundary="BND3")
        with file.open("wb") as fh:  # concurrent truncation after construction
            fh.write(content[:400])

        with pytest.raises(RuntimeError, match="усечённое"):
            b"".join(streamer)

    def test_stream_closes_handle_after_iteration(self, tmp_path: Path) -> None:
        file = tmp_path / "close.mkv"
        file.write_bytes(b"D" * 10)
        streamer = AdaptiveStreamingMultipart(file, boundary="BND4")
        list(streamer)
        assert streamer._file_handle is None  # handle released (Windows unlink safety)
        streamer.close()  # idempotent after full iteration

    def test_second_iteration_is_loud(self, tmp_path: Path) -> None:
        file = tmp_path / "once.mkv"
        file.write_bytes(b"E" * 10)
        streamer = AdaptiveStreamingMultipart(file, boundary="BND5")
        list(streamer)
        with pytest.raises(RuntimeError, match="использован"):
            list(streamer)

    def test_upload_file_closes_streamer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        file, fake = _setup(tmp_path)
        client = _client(fake)
        client.token = "g"
        closed: list[bool] = []
        original_close = AdaptiveStreamingMultipart.close

        def spying_close(self: AdaptiveStreamingMultipart) -> None:
            closed.append(True)
            original_close(self)

        monkeypatch.setattr(AdaptiveStreamingMultipart, "close", spying_close)
        upload_file(file, client)
        assert closed


class TestContentTypeBySuffix:
    """F11: file part Content-Type follows the filename suffix."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("video.mkv", "video/x-matroska"),
            ("video.mp4", "video/mp4"),
            ("video.ts", "video/mp2t"),
            ("video.bin", "application/octet-stream"),
        ],
    )
    def test_content_type_follows_suffix(self, tmp_path: Path, name: str, expected: str) -> None:
        file = tmp_path / name
        file.write_bytes(b"data")
        streamer = AdaptiveStreamingMultipart(file, boundary="B")
        payload = b"".join(streamer)
        assert f"Content-Type: {expected}\r\n".encode() in payload
