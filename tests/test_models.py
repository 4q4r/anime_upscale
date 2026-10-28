"""Model manifest/fetch tests (plan §6): parse, verify, atomic fetch, loud mismatch."""

import io
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Self

import pytest

from anime_upscale.models.fetch import FetchError, fetch_model, verify_all
from anime_upscale.models.manifest import (
    ManifestError,
    expected_model_path,
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

    def test_rife_seed_entry_pinned(self) -> None:
        """The vsrife pre-seed entry: same release asset vsrife itself downloads,
        sha256-pinned, marked as a vsrife cache seed."""
        entries = load_manifest(REPO / "models" / "manifest.toml")
        entry = entries["rife_v4_25"]
        assert entry.vsrife_seed is True
        assert entry.url == (
            "https://github.com/HolyWu/vs-rife/releases/download/model/flownet_v4.25.pkl"
        )
        assert entry.sha256 == "6615790efd627772917205db291f51cd392528a157ecbb2ecaeec3bff8eb6de2"
        assert entry.license == "MIT"

    def test_seed_entry_without_url_is_loud(self) -> None:
        entries = load_manifest(REPO / "models" / "manifest.toml")
        broken = replace(entries["rife_v4_25"], url="")
        with pytest.raises(ManifestError, match="url"):
            expected_model_path(Path("models"), broken)

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


REAL_RIFE_SHA = "6615790efd627772917205db291f51cd392528a157ecbb2ecaeec3bff8eb6de2"


class TestVsrifeSeed:
    """RIFE 4.25 pre-seed into the vsrife package cache (E01/E02 live class)."""

    def _seed_entry(self) -> object:
        return load_manifest(REPO / "models" / "manifest.toml")["rife_v4_25"]

    def test_seed_path_resolves_into_vsrife_package(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import anime_upscale.models.manifest as manifest_mod

        pkg = tmp_path / "site-packages" / "vsrife"
        monkeypatch.setattr(manifest_mod, "_vsrife_package_dir", lambda: pkg)
        entry = self._seed_entry()
        target = expected_model_path(tmp_path / "models", entry)  # models_dir is unused for seeds
        assert target == pkg / "models" / "flownet_v4.25.pkl"

    def test_fetch_replaces_stub_atomically(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import anime_upscale.models.fetch as fetch_mod
        import anime_upscale.models.manifest as manifest_mod

        pkg = tmp_path / "site-packages" / "vsrife"
        stub = pkg / "models" / "flownet_v4.25.pkl"
        stub.parent.mkdir(parents=True)
        stub.write_bytes(b"")  # the wheel ships 0-byte placeholders
        monkeypatch.setattr(manifest_mod, "_vsrife_package_dir", lambda: pkg)

        def fake_download(url: str, destination: Path) -> None:
            assert url.endswith("flownet_v4.25.pkl")
            destination.write_bytes(b"RIFEW" * 1024)

        monkeypatch.setattr(fetch_mod, "_download", fake_download)
        import hashlib

        entry = replace(self._seed_entry(), sha256=hashlib.sha256(b"RIFEW" * 1024).hexdigest())  # ty: ignore[arg-type]
        out = fetch_model(entry, tmp_path / "manifest.toml", tmp_path / "models")
        assert out == stub
        assert stub.read_bytes() == b"RIFEW" * 1024

    def test_fetch_mismatch_keeps_a_valid_stub_state(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed/corrupt download must not leave the cache path ABSENT:
        vsrife probes os.path.getsize (FileNotFoundError on a missing file)."""
        import anime_upscale.models.fetch as fetch_mod
        import anime_upscale.models.manifest as manifest_mod

        pkg = tmp_path / "site-packages" / "vsrife"
        stub = pkg / "models" / "flownet_v4.25.pkl"
        stub.parent.mkdir(parents=True)
        stub.write_bytes(b"")  # 0-byte wheel stub mismatches the pin → refetch
        monkeypatch.setattr(manifest_mod, "_vsrife_package_dir", lambda: pkg)

        def bad_download(url: str, destination: Path) -> None:
            destination.write_bytes(b"CORRUPT")

        monkeypatch.setattr(fetch_mod, "_download", bad_download)
        with pytest.raises(FetchError, match="не совпадает"):
            fetch_model(self._seed_entry(), tmp_path / "manifest.toml", tmp_path / "models")
        assert stub.is_file() and stub.stat().st_size == 0

    def test_fetch_existing_verified_is_kept(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import anime_upscale.models.fetch as fetch_mod
        import anime_upscale.models.manifest as manifest_mod

        pkg = tmp_path / "site-packages" / "vsrife"
        target = pkg / "models" / "flownet_v4.25.pkl"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"REALWEIGHTS")
        monkeypatch.setattr(manifest_mod, "_vsrife_package_dir", lambda: pkg)

        import hashlib

        entry = replace(self._seed_entry(), sha256=hashlib.sha256(b"REALWEIGHTS").hexdigest())  # ty: ignore[arg-type]

        def no_download(url: str, destination: Path) -> None:
            msg = "must not download when the seed is verified"
            raise AssertionError(msg)

        monkeypatch.setattr(fetch_mod, "_download", no_download)
        assert fetch_model(entry, tmp_path / "manifest.toml", tmp_path / "models") == target

    def test_ensure_seed_missing_loud(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import anime_upscale.models.manifest as manifest_mod
        from anime_upscale.models.fetch import ensure_vsrife_seed

        pkg = tmp_path / "site-packages" / "vsrife"
        monkeypatch.setattr(manifest_mod, "_vsrife_package_dir", lambda: pkg)
        with pytest.raises(ManifestError, match="models fetch"):
            ensure_vsrife_seed(REPO / "models" / "manifest.toml")

    def test_ensure_seed_corrupt_loud(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import anime_upscale.models.manifest as manifest_mod
        from anime_upscale.models.fetch import ensure_vsrife_seed

        pkg = tmp_path / "site-packages" / "vsrife"
        target = pkg / "models" / "flownet_v4.25.pkl"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"truncated-or-mitm-swapped")
        monkeypatch.setattr(manifest_mod, "_vsrife_package_dir", lambda: pkg)
        with pytest.raises(ManifestError, match="не совпадает"):
            ensure_vsrife_seed(REPO / "models" / "manifest.toml")

    def test_ensure_seed_ok_quiet(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import anime_upscale.models.manifest as manifest_mod
        from anime_upscale.models.fetch import ensure_vsrife_seed

        pkg = tmp_path / "site-packages" / "vsrife"
        target = pkg / "models" / "flownet_v4.25.pkl"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"x" * 64)
        monkeypatch.setattr(manifest_mod, "_vsrife_package_dir", lambda: pkg)
        import hashlib

        entries = load_manifest(REPO / "models" / "manifest.toml")
        pinned = replace(entries["rife_v4_25"], sha256=hashlib.sha256(b"x" * 64).hexdigest())  # ty: ignore[arg-type]
        monkeypatch.setattr(manifest_mod, "load_manifest", lambda _p: {"rife_v4_25": pinned})
        import anime_upscale.models.fetch as fetch_mod

        monkeypatch.setattr(fetch_mod, "load_manifest", lambda _p: {"rife_v4_25": pinned})
        assert ensure_vsrife_seed(tmp_path / "manifest.toml") == target

    def test_verify_all_covers_seed_entry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import hashlib

        import anime_upscale.models.fetch as fetch_mod
        import anime_upscale.models.manifest as manifest_mod

        pkg = tmp_path / "site-packages" / "vsrife"
        target = pkg / "models" / "flownet_v4.25.pkl"
        target.parent.mkdir(parents=True)
        payload = b"WEIGHTS"
        target.write_bytes(payload)
        monkeypatch.setattr(manifest_mod, "_vsrife_package_dir", lambda: pkg)
        entry = replace(self._seed_entry(), sha256=hashlib.sha256(payload).hexdigest())  # ty: ignore[arg-type]
        monkeypatch.setattr(manifest_mod, "load_manifest", lambda _p: {"rife_v4_25": entry})
        monkeypatch.setattr(fetch_mod, "load_manifest", lambda _p: {"rife_v4_25": entry})
        verified = verify_all(tmp_path / "manifest.toml", tmp_path / "models")
        assert verified["rife_v4_25"] == target
