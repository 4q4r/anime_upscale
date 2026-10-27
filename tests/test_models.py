"""Model manifest/fetch tests (plan §6): parse, verify, atomic fetch, loud mismatch."""

import io
import zipfile
from pathlib import Path
from typing import Self

import pytest

from anime_upscale.models.fetch import FetchError, fetch_model, verify_all
from anime_upscale.models.manifest import (
    ManifestError,
    load_manifest,
    resolve_model_path,
)

REPO = Path(__file__).resolve().parent.parent

GOOD_SHA = "af7307eee19e5982a8014dd0e4650d3bde2e25aa78d2105a4bdfd947636e4c8f"


class TestManifest:
    def test_load_default_manifest(self) -> None:
        entries = load_manifest(REPO / "models" / "manifest.toml")
        entry = entries["2x_AnimeJaNai_HD_V3_Compact"]
        assert entry.sha256 == GOOD_SHA
        assert entry.member == "2x_AnimeJaNai_HD_V3_ModelsOnly/2x_AnimeJaNai_HD_V3_Compact.pth"
        assert "mpv-AnimeJaNai/releases/download/3.0.0" in entry.url
        assert entry.license == "CC-BY-NC-SA-4.0"
        assert entries["rife_4_25"].source_note  # documentational entry

    def test_missing_manifest_loud(self, tmp_path: Path) -> None:
        with pytest.raises(ManifestError, match="не найден"):
            load_manifest(tmp_path / "nope.toml")

    def test_resolve_missing_model_names_fetch_command(self, tmp_path: Path) -> None:
        with pytest.raises(ManifestError, match="models fetch"):
            resolve_model_path(
                REPO / "models" / "manifest.toml", tmp_path, "2x_AnimeJaNai_HD_V3_Compact"
            )


def _zip_bytes(member: str, content: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(member, content)
    return buffer.getvalue()


class TestFetchModel:
    def _entry(self) -> object:
        entries = load_manifest(REPO / "models" / "manifest.toml")
        return entries["2x_AnimeJaNai_HD_V3_Compact"]

    def test_download_verify_atomic(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import anime_upscale.models.fetch as fetch_mod

        content = b"PTHBYTES"
        body = _zip_bytes("2x_AnimeJaNai_HD_V3_ModelsOnly/2x_AnimeJaNai_HD_V3_Compact.pth", content)
        real_sha = __import__("hashlib").sha256(content).hexdigest()
        entry = load_manifest(REPO / "models" / "manifest.toml")["2x_AnimeJaNai_HD_V3_Compact"]
        pinned = type(entry)(
            name=entry.name,
            url=entry.url,
            member=entry.member,
            sha256=real_sha,
            license=entry.license,
        )

        class _Resp:
            status_code = 200

            def iter_content(self, chunk_size: int) -> bytes:
                yield body

            def __enter__(self) -> Self:
                return self

            def __exit__(self, *a: object) -> None:
                return None

        monkeypatch.setattr(fetch_mod.requests, "get", lambda url, **kw: _Resp())
        result = fetch_model(pinned, REPO / "models" / "manifest.toml", tmp_path)
        assert result.read_bytes() == content
        assert not any(p.name.startswith(".") for p in result.parent.iterdir())

    def test_sha_mismatch_no_final_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import anime_upscale.models.fetch as fetch_mod

        body = _zip_bytes(
            "2x_AnimeJaNai_HD_V3_ModelsOnly/2x_AnimeJaNai_HD_V3_Compact.pth", b"tampered"
        )
        entry = self._entry()

        class _Resp:
            status_code = 200

            def iter_content(self, chunk_size: int) -> bytes:
                yield body

            def __enter__(self) -> Self:
                return self

            def __exit__(self, *a: object) -> None:
                return None

        monkeypatch.setattr(fetch_mod.requests, "get", lambda url, **kw: _Resp())
        with pytest.raises(FetchError, match="не совпадает"):
            fetch_model(entry, REPO / "models" / "manifest.toml", tmp_path)
        assert not (tmp_path / "upscale" / "2x_AnimeJaNai_HD_V3_Compact.pth").exists()

    def test_verify_all_reports_ok(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import hashlib

        import anime_upscale.models.fetch as fetch_mod

        content = b"PTHBYTES"
        real_sha = hashlib.sha256(content).hexdigest()
        entry = load_manifest(REPO / "models" / "manifest.toml")["2x_AnimeJaNai_HD_V3_Compact"]
        target = tmp_path / "upscale" / "2x_AnimeJaNai_HD_V3_Compact.pth"
        target.parent.mkdir(parents=True)
        target.write_bytes(content)
        pinned = type(entry)(
            name=entry.name,
            url=entry.url,
            member=entry.member,
            sha256=real_sha,
            license=entry.license,
        )
        monkeypatch.setattr(fetch_mod, "load_manifest", lambda _p: {"m": pinned})
        verified = verify_all(tmp_path / "manifest.toml", tmp_path)
        assert verified["m"] == target

    def test_verify_missing_loud(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import anime_upscale.models.fetch as fetch_mod

        entry = self._entry()
        monkeypatch.setattr(fetch_mod, "load_manifest", lambda _p: {"m": entry})
        with pytest.raises(ManifestError, match="models fetch"):
            verify_all(tmp_path / "manifest.toml", tmp_path)
