"""Encode-arg builder tests: keyint math, BT.709 tags, x265 param pass-through."""

from pathlib import Path

import pytest

from anime_upscale.config import EncodeConfig, load_config
from anime_upscale.stages.encode import build_video_args, compute_keyint

REPO_ROOT = Path(__file__).resolve().parent.parent


def _sheet_cfg() -> EncodeConfig:
    """The real [stages.encode] from config.toml (the flag sheet lives there)."""
    return load_config(REPO_ROOT / "config.toml").stages.encode


def _cfg(**over: object) -> EncodeConfig:
    return EncodeConfig(**over)  # type: ignore[arg-type]


class TestKeyint:
    def test_23976(self) -> None:
        assert compute_keyint((24000, 1001)) == (240, 24)

    def test_5994(self) -> None:
        assert compute_keyint((60000, 1001)) == (600, 60)

    def test_exact_60(self) -> None:
        assert compute_keyint((60, 1)) == (600, 60)

    def test_exact_24(self) -> None:
        assert compute_keyint((24, 1)) == (240, 24)


class TestVideoArgs:
    def test_flag_sheet_shape(self) -> None:
        args = build_video_args(_sheet_cfg(), output_fps=(60000, 1001))
        assert args[args.index("-c:v") + 1] == "libx265"
        assert "-preset" in args and args[args.index("-preset") + 1] == "slow"
        assert "-crf" in args and args[args.index("-crf") + 1] == "16"
        assert "-pix_fmt" in args and args[args.index("-pix_fmt") + 1] == "yuv420p10le"
        assert "-profile:v" in args and args[args.index("-profile:v") + 1] == "main10"
        for flag, value in (
            ("-colorspace", "bt709"),
            ("-color_primaries", "bt709"),
            ("-color_trc", "bt709"),
            ("-color_range", "tv"),
        ):
            assert args[args.index(flag) + 1] == value

    def test_x265_params_contain_keyint_and_sheet(self) -> None:
        args = build_video_args(_sheet_cfg(), output_fps=(60000, 1001))
        params = args[args.index("-x265-params") + 1]
        parts = dict(p.split("=", 1) for p in params.split(":"))
        assert parts["keyint"] == "600"
        assert parts["min-keyint"] == "60"
        assert parts["no-sao"] == "1"
        assert parts["psy-rd"] == "2.0"
        assert parts["aq-mode"] == "3"
        assert parts["bframes"] == "8"
        assert parts["rc-lookahead"] == "60"
        assert parts["no-open-gop"] == "1"
        assert parts["tskip"] == "1"
        assert parts["cbqpoffs"] == "-2"
        assert "tune" not in parts

    def test_manual_keyint_override_wins(self) -> None:
        cfg = _cfg(x265_params=(("keyint", "300"),))
        args = build_video_args(cfg, output_fps=(60000, 1001))
        params = args[args.index("-x265-params") + 1]
        parts = dict(p.split("=", 1) for p in params.split(":"))
        assert parts["keyint"] == "300"
        assert parts["min-keyint"] == "60"

    def test_threads_passed_when_given(self) -> None:
        args = build_video_args(_cfg(), output_fps=(60, 1), threads=8)
        assert args[args.index("-threads") + 1] == "8"

    def test_animation_tune_never_emitted(self) -> None:
        args = build_video_args(_cfg(), output_fps=(60, 1))
        assert "-tune" not in args

    @pytest.mark.parametrize("crf", [14, 18])
    def test_crf_values_pass_through(self, crf: int) -> None:
        args = build_video_args(_cfg(crf=crf), output_fps=(60, 1))
        assert args[args.index("-crf") + 1] == str(crf)
