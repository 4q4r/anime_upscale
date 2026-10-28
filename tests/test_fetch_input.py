"""fetch_input tests: entry grammar, filename precedence, idempotence, collisions (§11.1/§11.6)."""

from pathlib import Path

import pytest

from anime_upscale.fetch_input import (
    BareName,
    DirectUrl,
    EntryError,
    GofileLink,
    build_queue,
    classify_entry,
    derive_filename,
    disambiguate,
    ensure_local,
    fetch_queue_item,
    sanitize_remote_filename,
)
from anime_upscale.gofile import GofileClient
from anime_upscale.reporting import Reporter, resolve_mode


class _MuteReporter(Reporter):
    def __init__(self) -> None:
        super().__init__(resolve_mode(no_color=True, no_tty=True))
        self.warnings: list[str] = []
        self.notes: list[str] = []

    def warn(self, text: str) -> None:
        self.warnings.append(text)

    def note(self, text: str) -> None:
        self.notes.append(text)


class TestGrammar:
    def test_bare_name(self) -> None:
        assert classify_entry("ep01.mkv") == BareName("ep01.mkv")

    def test_direct_url(self) -> None:
        assert classify_entry("https://example.com/x.mkv") == DirectUrl("https://example.com/x.mkv")

    def test_gofile_share_link(self) -> None:
        assert classify_entry("https://gofile.io/d/AbCdEf") == GofileLink("AbCdEf")
        assert classify_entry("https://www.gofile.io/d/x1?x=2") == GofileLink("x1")

    def test_share_link_trailing_slash_accepted(self) -> None:
        """PR1-review Minor: gofile.io/d/<id>/ (trailing slash) IS the canonical
        share form users paste — accepted with id case preserved; a double
        slash stays a loud spec error."""
        assert classify_entry("https://gofile.io/d/abc/") == GofileLink("abc")
        assert classify_entry("https://www.gofile.io/d/AbCdEf/") == GofileLink("AbCdEf")
        with pytest.raises(EntryError, match=r"gofile\.io/d"):
            classify_entry("https://gofile.io/d/abc//")

    def test_plain_http_entry_rejected_https_only(self) -> None:
        """F4: plain-HTTP ingestion bypasses DoH pinning — loud https-only error."""
        with pytest.raises(EntryError, match="HTTPS"):
            classify_entry("http://example.com/x.mkv")
        with pytest.raises(EntryError, match="HTTPS"):
            classify_entry("http://gofile.io/d/AbCdEf")

    def test_store_gofile_host_rejected_loudly(self) -> None:
        """M6: store.gofile.io is gofile-family — a direct store link is a loud
        spec error (only gofile.io/d/<id> is accepted), never a DirectUrl."""
        with pytest.raises(EntryError, match=r"gofile\.io/d"):
            classify_entry("https://store.gofile.io/download/abc123/ep.mkv")

    @pytest.mark.parametrize(
        "bad",
        [
            "https://api.gofile.io/contents/abc",
            "https://store.gofile.io/download/x/ep.mkv",
            "https://beta.gofile.io/d/abc",  # subdomain ≠ the canonical share host path
        ],
    )
    def test_gofile_family_hosts_require_share_path(self, bad: str) -> None:
        """M6: any *.gofile.io host without a /d/<id> path = loud error."""
        with pytest.raises(EntryError, match=r"gofile\.io/d"):
            classify_entry(bad)

    def test_share_match_anchored_to_url_path_not_query(self) -> None:
        """M6: a share link embedded in another gofile URL's QUERY string must not
        resurrect the entry — the path (/download) is what classifies."""
        with pytest.raises(EntryError, match=r"gofile\.io/d"):
            classify_entry("https://gofile.io/download?u=https://gofile.io/d/abc")
        # the canonical forms keep working (query params are fine, path decides)
        assert classify_entry("https://www.gofile.io/d/x1?x=2") == GofileLink("x1")
        assert classify_entry("https://gofile.io/d/AbCdEf#frag") == GofileLink("AbCdEf")

    def test_entry_grammar_matrix(self) -> None:
        """M10/§11.1: the full four-way grammar — bare name / direct URL /
        gofile share link / store-host rejection."""
        assert classify_entry("ep01.mkv") == BareName("ep01.mkv")
        assert classify_entry("https://example.com/x.mkv") == DirectUrl("https://example.com/x.mkv")
        assert classify_entry("https://gofile.io/d/abc123") == GofileLink("abc123")
        with pytest.raises(EntryError, match=r"gofile\.io/d"):
            classify_entry("https://store.gofile.io/download/x/ep.mkv")

    @pytest.mark.parametrize(
        "bad",
        [
            "https://gofile.io/upload",  # non-share shape
            "https://gofile.io/contents/abc",
            "plainid123",  # no separators → the bare-name rule applies, not a gofile id
        ],
    )
    def test_gofile_shapes(self, bad: str) -> None:
        if bad == "plainid123":
            assert classify_entry(bad) == BareName(bad)  # no separators → bare name rule
        else:
            with pytest.raises(EntryError, match="gofile"):
                classify_entry(bad)

    @pytest.mark.parametrize("bad", ["../evil.mkv", "sub/dir.mkv", "back\\slash.mkv"])
    def test_path_separators_rejected(self, bad: str) -> None:
        with pytest.raises(EntryError, match="имя"):
            classify_entry(bad)


class TestDeriveFilename:
    def test_content_disposition_star_rfc5987(self) -> None:
        headers = {"Content-Disposition": "attachment; filename*=UTF-8''ep%2002.mkv"}
        assert derive_filename(headers, "https://x/y") == "ep 02.mkv"

    def test_content_disposition_plain(self) -> None:
        headers = {"Content-Disposition": 'attachment; filename="ep03.mkv"'}
        assert derive_filename(headers, "https://x/y") == "ep03.mkv"

    def test_star_beats_plain(self) -> None:
        headers = {"Content-Disposition": "attachment; filename=a.mkv; filename*=UTF-8''b.mkv"}
        assert derive_filename(headers, "https://x/y") == "b.mkv"

    def test_url_path_segment_decoded_no_query(self) -> None:
        assert derive_filename({}, "https://x/dir/ep%2004.mkv?token=1") == "ep 04.mkv"

    def test_nothing_derivable_is_loud(self) -> None:
        with pytest.raises(EntryError, match="имя файла"):
            derive_filename({}, "https://x/")


class TestDisambiguate:
    def test_second_collision_gets_dash_two(self) -> None:
        reporter = _MuteReporter()
        assert disambiguate("ep.mkv", {"ep.mkv"}, reporter) == "ep-2.mkv"
        assert any("ep-2.mkv" in w for w in reporter.warnings)

    def test_third_collision(self) -> None:
        reporter = _MuteReporter()
        taken = {"ep.mkv", "ep-2.mkv"}
        assert disambiguate("ep.mkv", taken, reporter) == "ep-3.mkv"

    def test_no_extension(self) -> None:
        reporter = _MuteReporter()
        assert disambiguate("file", {"file"}, reporter) == "file-2"


def _client_factory_fake() -> GofileClient:
    raise AssertionError("client factory must not be called without gofile entries")


class TestBuildQueue:
    def test_local_url_order_preserved(self) -> None:
        reporter = _MuteReporter()
        queue = build_queue(
            ["ep01.mkv", "https://x/e.mkv"],
            _client_factory_fake,
            reporter,
            name_prober=lambda url: "e.mkv",
        )
        assert [item.kind for item in queue] == ["local", "url"]
        assert queue[0].name == "ep01.mkv"
        assert queue[1].name == "e.mkv"

    def test_duplicate_entries_deduplicated_with_warning(self) -> None:
        reporter = _MuteReporter()
        queue = build_queue(["a.mkv", "a.mkv"], _client_factory_fake, reporter)
        assert len(queue) == 1
        assert any("Дублирующийся вход" in w for w in reporter.warnings)

    def test_gofile_folder_expands_in_place_with_collision(self) -> None:
        reporter = _MuteReporter()

        class _FakeClient:
            def expand_entries(self, content_id: str, reporter: object) -> list:
                from anime_upscale.gofile import ResolvedChild

                assert content_id == "folder1"
                return [
                    ResolvedChild(name="ep.mkv", link="https://s/1"),
                    ResolvedChild(name="ep.mkv", link="https://s/2"),
                ]

        queue = build_queue(
            ["local.mkv", "https://gofile.io/d/folder1"],
            _FakeClient,  # type: ignore[arg-type,return-value]
            reporter,
        )
        assert [item.name for item in queue] == ["local.mkv", "ep.mkv", "ep-2.mkv"]
        assert queue[1].kind == "gofile"
        assert "gofile папка folder1" in queue[2].origin
        assert any("ep-2.mkv" in w for w in reporter.warnings)


class TestFetchQueueItem:
    def test_idempotent_skip(self, tmp_path: Path) -> None:
        reporter = _MuteReporter()
        existing = tmp_path / "a.mkv"
        existing.write_bytes(b"x")
        from anime_upscale.fetch_input import QueueItem

        item = QueueItem(source="a.mkv", origin="локальный файл", name="a.mkv", kind="local")
        result = fetch_queue_item(item, tmp_path, _client_factory_fake, reporter)
        assert result == existing
        assert any("загрузка пропущена" in n for n in reporter.notes)

    def test_local_missing_is_loud(self, tmp_path: Path) -> None:
        reporter = _MuteReporter()
        from anime_upscale.fetch_input import QueueItem

        item = QueueItem(source="nope.mkv", origin="локальный файл", name="nope.mkv", kind="local")
        with pytest.raises(EntryError, match="не найден"):
            fetch_queue_item(item, tmp_path, _client_factory_fake, reporter)

    def test_ensure_local_rejects_traversal(self, tmp_path: Path) -> None:
        with pytest.raises(EntryError):
            ensure_local("../secret", tmp_path)

    def test_gofile_kind_existing_file_skips_without_client(self, tmp_path: Path) -> None:
        """M10: the idempotent skip precedes any client work for gofile entries."""
        from anime_upscale.fetch_input import QueueItem

        reporter = _MuteReporter()
        existing = tmp_path / "ep.mkv"
        existing.write_bytes(b"x")
        item = QueueItem(
            source="https://gofile.io/d/f1",
            origin="gofile папка f1 → файл",
            name="ep.mkv",
            kind="gofile",
            url="https://store.gofile.io/download/x/ep.mkv",
        )
        result = fetch_queue_item(item, tmp_path, _client_factory_fake, reporter)
        assert result == existing
        assert any("загрузка пропущена" in n for n in reporter.notes)

    def test_url_kind_existing_file_skips_without_client(self, tmp_path: Path) -> None:
        from anime_upscale.fetch_input import QueueItem

        reporter = _MuteReporter()
        existing = tmp_path / "e.mkv"
        existing.write_bytes(b"x")
        item = QueueItem(
            source="https://x/e.mkv",
            origin="прямая ссылка",
            name="e.mkv",
            kind="url",
            url="https://x/e.mkv",
        )
        result = fetch_queue_item(item, tmp_path, _client_factory_fake, reporter)
        assert result == existing
        assert any("загрузка пропущена" in n for n in reporter.notes)

    def test_gofile_kind_missing_file_downloads_via_client(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """M10: a gofile entry with no local file goes through client.download_file."""
        from anime_upscale.fetch_input import QueueItem

        reporter = _MuteReporter()
        (tmp_path / "input").mkdir()
        item = QueueItem(
            source="https://gofile.io/d/f1",
            origin="gofile папка f1 → файл",
            name="ep.mkv",
            kind="gofile",
            url="https://store.gofile.io/download/x/ep.mkv",
        )
        calls: list[tuple[str, Path]] = []

        class _FakeClient:
            def download_file(self, url: str, destination: Path, progress: object = None) -> None:
                calls.append((url, destination))
                destination.write_bytes(b"data")

        result = fetch_queue_item(
            item,
            tmp_path / "input",
            _FakeClient,  # type: ignore[arg-type,return-value]
            reporter,
        )
        assert result == tmp_path / "input" / "ep.mkv"
        assert calls and calls[0][0] == "https://store.gofile.io/download/x/ep.mkv"


class TestSanitizeRemoteFilename:
    """F2: one gate for every remote-derived name — hostile shapes are loud errors."""

    @pytest.mark.parametrize(
        "bad",
        [
            "../etc/passwd",
            "..\\..\\win",
            "/absolute" + "/path/x.mkv",
            "a/b.mkv",
            "a\\b.mkv",
            "..mkv",
            "..",
            ".hidden.mkv",
            ".",
            "",
            "C:evil.mkv",  # item 2: Windows drive-letter form
            "C:\\path",
            "\\\\server\\share",  # UNC — covered by the backslash rule
        ],
    )
    def test_hostile_names_rejected_with_source(self, bad: str) -> None:
        with pytest.raises(EntryError, match=r"Content-Disposition|gofile|URL|имя"):
            sanitize_remote_filename(bad, "заголовок Content-Disposition (filename)")

    @pytest.mark.parametrize(
        "good",
        [
            "эпизод 01.mkv",
            "файл (2).mp4",
            "Воскрешение [1080p].mkv",
            "normal_name-mixed.ts",
            "серия 01 — полное описание эпизода (2026).mkv",  # benign unicode item 2 control
        ],
    )
    def test_benign_unicode_spaces_pass(self, good: str) -> None:
        assert sanitize_remote_filename(good, "последний сегмент пути URL") == good

    def test_derive_filename_rejects_traversal_naming_header(self) -> None:
        headers = {"Content-Disposition": 'attachment; filename="../etc/passwd"'}
        with pytest.raises(EntryError, match=r"passwd.*(Content-Disposition|filename)"):
            derive_filename(headers, "https://host/x")

    def test_derive_filename_url_segment_traversal_rejected(self) -> None:
        with pytest.raises(EntryError, match="URL"):
            derive_filename({}, "https://host/download/..%2F..%2Fwin")

    def test_remote_name_hostile_cd_is_terminal_not_swallowed(self, monkeypatch) -> None:
        """Item 1: a rejected Content-Disposition name must NOT fall through to the
        raw URL segment — the old `except EntryError: pass` bypassed the gate."""
        from anime_upscale import fetch_input as fi

        class _FakeSession:
            def pin(self, host: str) -> str:
                return "203.0.113.10"

            def head(self, url: str, timeout: int = 30, allow_redirects: bool = True):
                final_url = url

                class _Resp:
                    status_code = 200
                    headers = {"Content-Disposition": 'attachment; filename="../x.mkv"'}
                    url = final_url

                return _Resp()

        monkeypatch.setattr(fi, "PinnedSession", _FakeSession)
        with pytest.raises(EntryError, match=r"\.\./x\.mkv"):
            fi._remote_name("https://host/get/file")

    def test_remote_name_absent_cd_benign_segment_used(self, monkeypatch) -> None:
        """Item 1: absence-only fall-through — no header, benign URL segment wins."""
        from anime_upscale import fetch_input as fi

        class _FakeSession:
            def pin(self, host: str) -> str:
                return "203.0.113.10"

            def head(self, url: str, timeout: int = 30, allow_redirects: bool = True):
                final_url = url

                class _Resp:
                    status_code = 200
                    headers: dict = {}
                    url = final_url

                return _Resp()

        monkeypatch.setattr(fi, "PinnedSession", _FakeSession)
        assert fi._remote_name("https://host/get/ok%20file.mkv") == "ok file.mkv"

    def test_remote_name_hostile_segment_no_header_is_loud(self, monkeypatch) -> None:
        from anime_upscale import fetch_input as fi

        class _FakeSession:
            def pin(self, host: str) -> str:
                return "203.0.113.10"

            def head(self, url: str, timeout: int = 30, allow_redirects: bool = True):
                final_url = url

                class _Resp:
                    status_code = 200
                    headers: dict = {}
                    url = final_url

                return _Resp()

        monkeypatch.setattr(fi, "PinnedSession", _FakeSession)
        with pytest.raises(EntryError, match=r"a/b\.mkv"):
            fi._remote_name("https://host/get/a%2Fb.mkv")  # percent-decodes to a/b.mkv

    def test_gofile_hostile_child_name_rejected_in_queue(self) -> None:
        reporter = _MuteReporter()

        class _FakeClient:
            def expand_entries(self, content_id: str, reporter: object) -> list:
                from anime_upscale.gofile import ResolvedChild

                return [ResolvedChild(name="a/b.mkv", link="https://s/1")]

        with pytest.raises(EntryError, match=r"a/b\.mkv"):
            build_queue(
                ["https://gofile.io/d/folder1"],
                _FakeClient,  # type: ignore[arg-type,return-value]
                reporter,
            )
