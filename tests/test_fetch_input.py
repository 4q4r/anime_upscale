"""fetch_input tests: entry grammar, filename precedence, idempotence, collisions (§11.1/§11.6)."""

from pathlib import Path
from typing import Self

import pytest

from anime_upscale.fetch_input import (
    BareName,
    DirectUrl,
    EntryError,
    FetchCache,
    GofileLink,
    QueueItem,
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
                    ResolvedChild(name="ep.mkv", link="https://s/1", size=100, md5="hash1"),
                    ResolvedChild(name="ep.mkv", link="https://s/2", size=200, md5="hash2"),
                ]

        queue = build_queue(
            ["local.mkv", "https://gofile.io/d/folder1"],
            _FakeClient,  # type: ignore[arg-type,return-value]
            reporter,
        )
        assert [item.name for item in queue] == ["local.mkv", "ep.mkv", "ep-2.mkv"]
        assert queue[1].kind == "gofile"
        assert queue[1].expected_size == 100
        assert queue[1].expected_md5 == "hash1"
        assert queue[2].expected_size == 200
        assert queue[2].expected_md5 == "hash2"
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
            expected_size=1,
        )
        result = fetch_queue_item(item, tmp_path, _client_factory_fake, reporter)
        assert result == existing
        assert any("загрузка пропущена" in n for n in reporter.notes)

    def test_fetch_cache_smart_skip_and_delta(self, tmp_path: Path) -> None:
        """Verify smart caching, skipping unchanged files, redownloading corrupted files."""
        import hashlib

        from anime_upscale.fetch_input import FetchCache, QueueItem

        reporter = _MuteReporter()
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        content = b"valid video payload"
        valid_md5 = hashlib.md5(content, usedforsecurity=False).hexdigest()
        valid_size = len(content)

        # 1. Delta download: new episode from album
        download_calls: list[str] = []

        class _MockClient:
            def download_file(
                self, url: str, destination: Path, progress: object = None, **_kw: object
            ) -> None:
                download_calls.append(url)
                destination.write_bytes(content)

        item1 = QueueItem(
            source="https://gofile.io/d/album1",
            origin="gofile папка album1 → файл",
            name="ep01.mkv",
            kind="gofile",
            url="https://store.gofile.io/ep01.mkv",
            expected_size=valid_size,
            expected_md5=valid_md5,
        )

        res1 = fetch_queue_item(item1, input_dir, _MockClient, reporter)
        assert res1 == input_dir / "ep01.mkv"
        assert len(download_calls) == 1
        assert (input_dir / ".fetch_cache.json").is_file()

        cache = FetchCache(input_dir / ".fetch_cache.json", reporter)
        entry = cache.get("ep01.mkv")
        assert entry is not None
        assert entry["size"] == valid_size
        assert entry["md5"] == valid_md5

        # 2. Skip already downloaded valid file
        reporter.notes.clear()
        res2 = fetch_queue_item(item1, input_dir, _MockClient, reporter)
        assert res2 == input_dir / "ep01.mkv"
        assert len(download_calls) == 1  # No new download
        msg_sub = f"Уже в input/ и актуален: {item1.name} — загрузка пропущена"
        assert any(msg_sub in n for n in reporter.notes)

        # 3. Detect size mismatch -> redownload
        (input_dir / "ep01.mkv").write_bytes(b"short")  # size is 5 instead of valid_size
        reporter.warnings.clear()
        reporter.notes.clear()
        res3 = fetch_queue_item(item1, input_dir, _MockClient, reporter)
        assert res3 == input_dir / "ep01.mkv"
        assert len(download_calls) == 2  # Downloaded again
        assert any("повреждён или недокачан" in w for w in reporter.warnings)
        assert (input_dir / "ep01.mkv").read_bytes() == content

        # 4. Detect MD5 mismatch -> redownload
        corrupt_content = b"corrupted payload!!"  # same length: 19 bytes
        assert len(corrupt_content) == valid_size
        (input_dir / "ep01.mkv").write_bytes(corrupt_content)
        # update cache so mtime matches corrupted file but old md5 was valid_md5
        stat = (input_dir / "ep01.mkv").stat()
        cache.record(
            "ep01.mkv",
            item1.source,
            valid_size,
            "wrongmd5",
            valid_size,
            stat.st_mtime - 100,
        )

        reporter.warnings.clear()
        reporter.notes.clear()
        res4 = fetch_queue_item(item1, input_dir, _MockClient, reporter)
        assert res4 == input_dir / "ep01.mkv"
        assert len(download_calls) == 3  # Downloaded again
        assert any("хэш MD5 не совпадает" in w for w in reporter.warnings)
        assert (input_dir / "ep01.mkv").read_bytes() == content

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
            def download_file(
                self, url: str, destination: Path, progress: object = None, **_kw: object
            ) -> None:
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


class TestCacheIntegrity:
    """Adversarial-review F3/F4/F5/F14 regressions: the fetch cache is
    integrity-checked, source-scoped, and fails audibly."""

    @staticmethod
    def _gofile_item(md5: str | None, size: int | None) -> QueueItem:
        return QueueItem(
            source="https://gofile.io/d/album1",
            origin="gofile папка album1 → файл",
            name="ep01.mkv",
            kind="gofile",
            url="https://store.gofile.io/ep01.mkv",
            expected_size=size,
            expected_md5=md5,
        )

    @staticmethod
    def _recorder(calls: list[str], content: bytes) -> type:
        class _MockClient:
            def download_file(
                self, url: str, destination: Path, progress: object = None, **_kw: object
            ) -> None:
                calls.append(url)
                destination.write_bytes(content)

        return _MockClient

    def test_corrupted_same_size_same_mtime_redownloaded(self, tmp_path: Path) -> None:
        """F3 probe: corrupting content while preserving size+mtime must NOT pass
        verification — with expected_md5 the disk is ALWAYS hashed."""
        import hashlib
        import os

        reporter = _MuteReporter()
        content = b"valid video payload"
        valid_md5 = hashlib.md5(content, usedforsecurity=False).hexdigest()
        calls: list[str] = []
        item = self._gofile_item(valid_md5, len(content))
        fetch_queue_item(item, tmp_path, self._recorder(calls, content), reporter)  # type: ignore[arg-type]
        target = tmp_path / "ep01.mkv"
        saved_mtime = target.stat().st_mtime
        target.write_bytes(b"X" * len(content))
        os.utime(target, (saved_mtime, saved_mtime))
        reporter.warnings.clear()
        reporter.notes.clear()
        fetch_queue_item(item, tmp_path, self._recorder(calls, content), reporter)  # type: ignore[arg-type]
        assert len(calls) == 2
        assert any("MD5" in w for w in reporter.warnings)
        assert target.read_bytes() == content

    def test_url_cache_hit_from_other_source_redownloaded(self, tmp_path: Path) -> None:
        """F4 probe: cache hits are source-scoped — the same name previously
        fetched from another source is re-downloaded with an honest note."""
        reporter = _MuteReporter()
        target = tmp_path / "e.mkv"
        target.write_bytes(b"from source A")
        cache = FetchCache(tmp_path / ".fetch_cache.json", reporter)
        cache.record(
            "e.mkv",
            "https://a/e.mkv",
            len(b"from source A"),
            None,
            target.stat().st_size,
            target.stat().st_mtime,
        )
        calls: list[str] = []
        item = QueueItem(
            source="https://b/e.mkv",
            origin="прямая ссылка",
            name="e.mkv",
            kind="url",
            url="https://b/e.mkv",
        )
        result = fetch_queue_item(item, tmp_path, self._recorder(calls, b"from source B"), reporter)  # type: ignore[arg-type]
        assert result == target
        assert len(calls) == 1
        assert any("другого источника" in w for w in reporter.warnings)
        assert target.read_bytes() == b"from source B"
        reloaded = FetchCache(tmp_path / ".fetch_cache.json", reporter)
        assert reloaded.get("e.mkv") is not None
        assert reloaded.get("e.mkv")["source"] == "https://b/e.mkv"

    def test_url_cache_hit_changed_file_redownloaded(self, tmp_path: Path) -> None:
        """F3 (no-md5 path): the size+mtime fingerprint recorded at download time
        is the only freshness proof — an altered file is re-fetched."""
        reporter = _MuteReporter()
        calls: list[str] = []
        item = QueueItem(
            source="https://a/e.mkv",
            origin="прямая ссылка",
            name="e.mkv",
            kind="url",
            url="https://a/e.mkv",
        )
        fetch_queue_item(item, tmp_path, self._recorder(calls, b"original"), reporter)  # type: ignore[arg-type]
        (tmp_path / "e.mkv").write_bytes(b"tampered!")
        reporter.warnings.clear()
        reporter.notes.clear()
        fetch_queue_item(item, tmp_path, self._recorder(calls, b"original"), reporter)  # type: ignore[arg-type]
        assert len(calls) == 2
        assert any("перекачиваю" in w for w in reporter.warnings)
        assert (tmp_path / "e.mkv").read_bytes() == b"original"

    def test_url_cache_hit_same_source_unchanged_skips(self, tmp_path: Path) -> None:
        """F4 pin: an unchanged same-source fingerprint keeps the smart skip."""
        reporter = _MuteReporter()
        calls: list[str] = []
        item = QueueItem(
            source="https://a/e.mkv",
            origin="прямая ссылка",
            name="e.mkv",
            kind="url",
            url="https://a/e.mkv",
        )
        fetch_queue_item(item, tmp_path, self._recorder(calls, b"original"), reporter)  # type: ignore[arg-type]
        reporter.notes.clear()
        fetch_queue_item(item, tmp_path, self._recorder(calls, b"original"), reporter)  # type: ignore[arg-type]
        assert len(calls) == 1
        assert any("загрузка пропущена" in n for n in reporter.notes)

    def test_build_queue_head_probe_sets_expected_size(self) -> None:
        """F4: url-kind items carry the HEAD Content-Length as expected_size."""
        queue = build_queue(
            ["https://x/e.mkv"],
            _client_factory_fake,
            _MuteReporter(),
            name_prober=lambda url: "e.mkv",
            size_prober=lambda url: 12345,
        )
        assert queue[0].kind == "url"
        assert queue[0].expected_size == 12345

    def test_build_queue_name_prober_only_keeps_expected_size_none(self) -> None:
        """F4 pin: an injected name prober without a sizer keeps None (test
        seams never trigger a live probe)."""
        queue = build_queue(
            ["https://x/e.mkv"],
            _client_factory_fake,
            _MuteReporter(),
            name_prober=lambda url: "e.mkv",
        )
        assert queue[0].expected_size is None

    def test_remote_name_records_content_length_for_queue(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """F4 live path: the single HEAD probe already carries Content-Length —
        it is recorded for the queue build, no second probe happens."""
        from anime_upscale import fetch_input as fi

        class _FakeSession:
            def pin(self, host: str) -> str:
                return "203.0.113.10"

            def head(self, url: str = "", **kwargs: object):
                final_url = url

                class _Resp:
                    status_code = 200
                    headers = {
                        "Content-Length": "4321",
                        "Content-Disposition": 'attachment; filename="e.mkv"',
                    }
                    url = final_url

                return _Resp()

        monkeypatch.setattr(fi, "PinnedSession", _FakeSession)
        assert fi._remote_name("https://host/get/e.mkv") == "e.mkv"
        assert fi._PROBED_CONTENT_LENGTHS["https://host/get/e.mkv"] == 4321

    def test_build_queue_live_head_probe_feeds_expected_size(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """F4: the default (live) queue build scopes url-kind items by the
        Content-Length captured during the filename HEAD — one probe per URL."""
        from anime_upscale import fetch_input as fi

        class _FakeSession:
            def pin(self, host: str) -> str:
                return "203.0.113.10"

            def head(self, url: str = "", **kwargs: object):
                final_url = url

                class _Resp:
                    status_code = 200
                    headers = {
                        "Content-Length": "777",
                        "Content-Disposition": 'attachment; filename="e.mkv"',
                    }
                    url = final_url

                return _Resp()

        monkeypatch.setattr(fi, "PinnedSession", _FakeSession)
        monkeypatch.setattr(fi.gofile, "adaptive_tls_request", lambda fn, **kwargs: fn(**kwargs))
        queue = build_queue(["https://host/get/e.mkv"], _client_factory_fake, _MuteReporter())
        assert [i.name for i in queue] == ["e.mkv"]
        assert queue[0].expected_size == 777

    def test_same_source_md5_mismatch_diagnoses_changed_server(self, tmp_path: Path) -> None:
        """F5: same source + md5 mismatch = «повреждён или изменился на сервере»."""
        import hashlib

        reporter = _MuteReporter()
        content = b"valid video payload"
        valid_md5 = hashlib.md5(content, usedforsecurity=False).hexdigest()
        calls: list[str] = []
        item = self._gofile_item(valid_md5, len(content))
        fetch_queue_item(item, tmp_path, self._recorder(calls, content), reporter)  # type: ignore[arg-type]
        (tmp_path / "ep01.mkv").write_bytes(b"X" * len(content))
        reporter.warnings.clear()
        fetch_queue_item(item, tmp_path, self._recorder(calls, content), reporter)  # type: ignore[arg-type]
        assert len(calls) == 2
        assert any("повреждён или изменился на сервере" in w for w in reporter.warnings)
        assert any("хэш MD5 не совпадает" in w for w in reporter.warnings)
        assert (tmp_path / "ep01.mkv").read_bytes() == content

    def test_cross_source_same_name_is_not_called_corrupted(self, tmp_path: Path) -> None:
        """F5 probe: a same-name file from another album is re-downloaded with
        the honest cross-source message — never a false «повреждён»."""
        import hashlib

        reporter = _MuteReporter()
        content_a = b"album A payload"
        content_b = b"album B payload"
        calls: list[str] = []
        item_a = self._gofile_item(
            hashlib.md5(content_a, usedforsecurity=False).hexdigest(), len(content_a)
        )
        fetch_queue_item(item_a, tmp_path, self._recorder(calls, content_a), reporter)  # type: ignore[arg-type]
        item_b = QueueItem(
            source="https://gofile.io/d/album2",
            origin="gofile папка album2 → файл",
            name="ep01.mkv",
            kind="gofile",
            url="https://store.gofile.io/ep01-b.mkv",
            expected_size=len(content_b),
            expected_md5=hashlib.md5(content_b, usedforsecurity=False).hexdigest(),
        )
        reporter.warnings.clear()
        fetch_queue_item(item_b, tmp_path, self._recorder(calls, content_b), reporter)  # type: ignore[arg-type]
        assert len(calls) == 2
        assert any("другого источника" in w for w in reporter.warnings)
        assert not any("повреждён" in w for w in reporter.warnings)
        assert (tmp_path / "ep01.mkv").read_bytes() == content_b

    def test_cache_save_failure_warns_and_cleans_tmp(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """F14: a failing cache persist is audible and leaves no tmp orphan."""
        reporter = _MuteReporter()
        cache = FetchCache(tmp_path / ".fetch_cache.json", reporter)
        cache.record("e.mkv", "https://a", 1, None, 1, 1.0)
        tmp_file = tmp_path / ".fetch_cache.json.tmp"

        def broken_replace(target: Path, other: Path) -> None:
            msg = "disk full"
            raise OSError(msg)

        monkeypatch.setattr(Path, "replace", broken_replace)
        cache.record("e.mkv", "https://a", 2, None, 2, 2.0)
        assert any("Кэш входа не сохранён" in w for w in reporter.warnings)
        assert not tmp_file.exists()


class TestPostDownloadVerification:
    """Post-download integrity (live-batch wave): a downloaded file is verified
    against expected_size/expected_md5 IMMEDIATELY; a mismatch deletes the bad
    file and re-downloads (same transient-retry discipline); only after the
    attempts are exhausted does it fail loudly. A corrupt file never stays
    under the valid name, and the pipeline never receives unverified bytes."""

    @staticmethod
    def _recorder(calls: list[str], content: bytes) -> type:
        class _MockClient:
            def download_file(
                self, url: str, destination: Path, progress: object = None, **_kw: object
            ) -> None:
                calls.append(url)
                destination.write_bytes(content)

        return _MockClient

    @staticmethod
    def _varying_client(calls: list[str], contents: list[bytes]) -> type:
        class _MockClient:
            def download_file(
                self, url: str, destination: Path, progress: object = None, **_kw: object
            ) -> None:
                calls.append(url)
                destination.write_bytes(contents[min(len(calls) - 1, len(contents) - 1)])

        return _MockClient

    def test_size_mismatch_redownloads(self, tmp_path: Path) -> None:
        good = b"0123456789"
        bad = b"01234"
        calls: list[str] = []
        item = QueueItem(
            source="https://gofile.io/d/album1",
            origin="gofile папка album1 → файл",
            name="ep01.mkv",
            kind="gofile",
            url="https://store.gofile.io/ep01.mkv",
            expected_size=len(good),
            expected_md5=None,
        )
        reporter = _MuteReporter()
        result = fetch_queue_item(
            item, tmp_path, self._varying_client(calls, [bad, good]), reporter
        )
        assert len(calls) == 2
        assert result.read_bytes() == good
        assert any("целостности" in w or "размер" in w for w in reporter.warnings)

    def test_md5_mismatch_redownloads(self, tmp_path: Path) -> None:
        import hashlib

        good = b"valid payload"
        bad = b"corrupt paylo"  # same length as good — isolates the MD5 check
        valid_md5 = hashlib.md5(good, usedforsecurity=False).hexdigest()
        calls: list[str] = []
        item = QueueItem(
            source="https://gofile.io/d/album1",
            origin="gofile папка album1 → файл",
            name="ep01.mkv",
            kind="gofile",
            url="https://store.gofile.io/ep01.mkv",
            expected_size=len(good),
            expected_md5=valid_md5,
        )
        reporter = _MuteReporter()
        result = fetch_queue_item(
            item, tmp_path, self._varying_client(calls, [bad, good]), reporter
        )
        assert len(calls) == 2
        assert result.read_bytes() == good
        assert any("MD5" in w for w in reporter.warnings)

    def test_mismatch_exhausted_is_loud_and_file_removed(self, tmp_path: Path) -> None:
        good = b"0123456789"
        bad = b"01234"
        calls: list[str] = []
        item = QueueItem(
            source="https://gofile.io/d/album1",
            origin="gofile папка album1 → файл",
            name="ep01.mkv",
            kind="gofile",
            url="https://store.gofile.io/ep01.mkv",
            expected_size=len(good),
            expected_md5=None,
        )
        reporter = _MuteReporter()
        with pytest.raises(EntryError, match="целостност"):
            fetch_queue_item(item, tmp_path, self._recorder(calls, bad), reporter)
        assert len(calls) == 3
        assert not (tmp_path / "ep01.mkv").exists()
        assert not (tmp_path / "ep01.mkv.part").exists()
        # A failed verification must never poison the cache.
        assert FetchCache(tmp_path / ".fetch_cache.json", reporter).get("ep01.mkv") is None

    def test_verified_success_records_cache(self, tmp_path: Path) -> None:
        import hashlib

        good = b"valid payload"
        valid_md5 = hashlib.md5(good, usedforsecurity=False).hexdigest()
        calls: list[str] = []
        item = QueueItem(
            source="https://gofile.io/d/album1",
            origin="gofile папка album1 → файл",
            name="ep01.mkv",
            kind="gofile",
            url="https://store.gofile.io/ep01.mkv",
            expected_size=len(good),
            expected_md5=valid_md5,
        )
        reporter = _MuteReporter()
        fetch_queue_item(item, tmp_path, self._recorder(calls, good), reporter)
        assert len(calls) == 1
        cached = FetchCache(tmp_path / ".fetch_cache.json", reporter).get("ep01.mkv")
        assert cached is not None
        assert cached.get("md5") == valid_md5

    def test_download_bar_wired_with_expected_total(self, tmp_path: Path) -> None:
        """Бар скачивания строится через reporter.download_progress_bar с
        ожидаемым размером как total (ETA-механика #41)."""
        good = b"0123456789"
        item = QueueItem(
            source="https://gofile.io/d/album1",
            origin="gofile папка album1 → файл",
            name="ep01.mkv",
            kind="gofile",
            url="https://store.gofile.io/ep01.mkv",
            expected_size=len(good),
            expected_md5=None,
        )
        reporter = _MuteReporter()
        seen_bars: list[dict] = []

        def fake_bar(*, total: int = 0, filename: str = "") -> object:
            seen_bars.append({"total": total, "filename": filename})

            class _Bar:
                def __enter__(self) -> Self:
                    return self

                def __exit__(self, *a: object) -> None:
                    return None

                def callback(self) -> object:
                    return lambda done, total: None

            return _Bar()

        reporter.download_progress_bar = fake_bar  # type: ignore[method-assign]
        fetch_queue_item(item, tmp_path, self._recorder([], good), reporter)
        assert seen_bars and seen_bars[0]["total"] == len(good)
        assert seen_bars[0]["filename"] == "ep01.mkv"
