"""Encode-arg builder tests: keyint math, BT.709 tags, x265 param pass-through."""

from pathlib import Path

import pytest

from anime_upscale.config import ConfigError, EncodeConfig, load_config
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

    def test_animation_tune_never_emitted(self) -> None:
        args = build_video_args(_cfg(), output_fps=(60, 1))
        assert "-tune" not in args

    @pytest.mark.parametrize("crf", [14, 18])
    def test_crf_values_pass_through(self, crf: int) -> None:
        args = build_video_args(_cfg(crf=crf), output_fps=(60, 1))
        assert args[args.index("-crf") + 1] == str(crf)


def _params(args: list[str]) -> dict[str, str]:
    raw = args[args.index("-x265-params") + 1]
    return dict(part.split("=", 1) for part in raw.split(":"))


class TestThreadBudget:
    """Live bug 2026-11-02: `-threads N` maps to x265 frameNumThreads and dies
    above 16. The CPU budget belongs in `pools=N`; `-threads` is never emitted."""

    def test_threads_never_emitted(self) -> None:
        args = build_video_args(_cfg(), output_fps=(60, 1), threads=32)
        assert "-threads" not in args

    def test_threads_become_x265_pools(self) -> None:
        args = build_video_args(_cfg(), output_fps=(60, 1), threads=32)
        assert _params(args)["pools"] == "32"

    def test_user_pools_override_wins(self) -> None:
        cfg = _cfg(x265_params=(("pools", "8"),))
        args = build_video_args(cfg, output_fps=(60, 1), threads=32)
        assert _params(args)["pools"] == "8"
        assert "-threads" not in args

    def test_no_pools_without_threads(self) -> None:
        args = build_video_args(_cfg(), output_fps=(60, 1), threads=None)
        assert "pools" not in _params(args)


class TestX265ParamValidation:
    """frame-threads is the x265 knob with a hard build limit; bad config fails loud."""

    def test_frame_threads_above_limit_is_loud(self) -> None:
        cfg = _cfg(x265_params=(("frame-threads", "17"),))
        with pytest.raises(ConfigError, match="frame-threads") as exc_info:
            build_video_args(cfg, output_fps=(60, 1))
        message = str(exc_info.value)
        assert "16" in message
        assert "config.toml" in message

    def test_frame_threads_at_limit_is_legal(self) -> None:
        cfg = _cfg(x265_params=(("frame-threads", "16"),))
        args = build_video_args(cfg, output_fps=(60, 1))
        assert _params(args)["frame-threads"] == "16"

    def test_frame_threads_zero_is_legal(self) -> None:
        cfg = _cfg(x265_params=(("frame-threads", "0"),))
        args = build_video_args(cfg, output_fps=(60, 1))
        assert _params(args)["frame-threads"] == "0"

    def test_threads_key_in_x265_params_is_rejected(self) -> None:
        cfg = _cfg(x265_params=(("threads", "32"),))
        with pytest.raises(ConfigError, match="threads") as exc_info:
            build_video_args(cfg, output_fps=(60, 1))
        assert "pools" in str(exc_info.value)
