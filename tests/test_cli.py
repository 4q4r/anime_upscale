"""CLI tests: parsing, flags, profile-not-found, input/ single-file rule (plan §3)."""

from pathlib import Path

import pytest

from anime_upscale.cli import EntryError, _default_inputs, build_parser
from anime_upscale.config import ConfigError, resolve_run
from anime_upscale.tools import ToolError, resolve_tool


class TestParser:
    def test_repeatable_inputs(self) -> None:
        args = build_parser().parse_args(
            [
                "--profile",
                "full",
                "-i",
                "a.mkv",
                "-i",
                "https://x/b.mkv",
                "-i",
                "https://gofile.io/d/zz",
            ]
        )
        assert args.inputs == ["a.mkv", "https://x/b.mkv", "https://gofile.io/d/zz"]

    def test_flags(self) -> None:
        args = build_parser().parse_args(
            ["--profile", "full", "--no-color", "--no-tty", "--upload"]
        )
        assert args.no_color and args.no_tty and args.upload

    def test_models_subcommand(self) -> None:
        args = build_parser().parse_args(["models", "verify"])
        assert args.command == "models"
        assert args.action == "verify"

    def test_stages_and_overrides(self) -> None:
        args = build_parser().parse_args(
            ["--stages", "descale,deband,encode", "--target-fps", "60", "--descale-height", "720"]
        )
        assert args.stages == "descale,deband,encode"
        assert args.target_fps == "60"
        assert args.descale_height == 720


class TestProfileResolutionViaCli:
    def test_profile_not_found_lists_available(self) -> None:
        from anime_upscale.config import load_config

        repo = Path(__file__).resolve().parent.parent
        config = load_config(repo / "config.toml")
        with pytest.raises(ConfigError, match="upscale-only"):
            resolve_run(config, profile="nope")

    def test_no_selection_lists_profiles(self) -> None:
        from anime_upscale.config import load_config

        repo = Path(__file__).resolve().parent.parent
        config = load_config(repo / "config.toml")
        with pytest.raises(ConfigError, match="interp-only"):
            resolve_run(config, profile=None)


class TestDefaultInputs:
    def test_exactly_one_file_selected(self, tmp_path: Path) -> None:
        (tmp_path / "only.mkv").write_bytes(b"x")
        (tmp_path / "notes.txt").write_text("ignored: not media")
        assert _default_inputs(tmp_path) == ["only.mkv"]

    def test_zero_files_is_loud(self, tmp_path: Path) -> None:
        with pytest.raises(Exception, match="пуст"):
            _default_inputs(tmp_path)

    def test_multiple_files_is_loud_with_listing(self, tmp_path: Path) -> None:
        (tmp_path / "a.mkv").write_bytes(b"x")
        (tmp_path / "b.mp4").write_bytes(b"x")
        with pytest.raises(EntryError, match=r"a\.mkv") as excinfo:
            _default_inputs(tmp_path)
        assert "b.mp4" in str(excinfo.value)


def _mute():
    from anime_upscale.reporting import Reporter

    return Reporter.__new__(Reporter)  # never emits in these tests


class TestTools:
    def test_override_missing_file_is_loud_naming_key(self, tmp_path: Path) -> None:
        with pytest.raises(ToolError, match=r"\[tools\] ffmpeg"):
            resolve_tool("ffmpeg", str(tmp_path / "missing.exe"))

    def test_override_used_when_valid(self, tmp_path: Path) -> None:
        fake = tmp_path / "my_ffmpeg"
        fake.write_text("#!/bin/sh\n")
        fake.chmod(0o755)
        assert resolve_tool("ffmpeg", str(fake)) == str(fake)

    def test_path_lookup(self) -> None:
        assert resolve_tool("ffmpeg")  # ffmpeg present on this machine
        with pytest.raises(ToolError, match="PATH"):
            resolve_tool("definitely-not-a-tool-42")
