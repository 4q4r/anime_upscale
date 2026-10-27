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
        assert classify_entry("http://www.gofile.io/d/x1?x=2") == GofileLink("x1")

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
