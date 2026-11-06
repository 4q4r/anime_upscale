"""Config layer tests: schema validation, merge precedence, rational fps parsing."""

from pathlib import Path

import pytest

from anime_upscale.config import (
    ConfigError,
    load_config,
    parse_fps_argument,
    resolve_run,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_default() -> object:
    return load_config(REPO_ROOT / "config.toml")


class TestDefaults:
    def test_load_default_config_matches_plan_values(self) -> None:
        cfg = _load_default()
        assert cfg.pipeline_order == ("descale", "deband", "upscale", "interpolate", "encode")
        assert cfg.stages.interpolate.target_fps == (60000, 1001)
        assert cfg.stages.interpolate.model == "4.25"
        assert cfg.stages.interpolate.trt is True
        assert cfg.stages.encode.crf == 16
        assert cfg.stages.encode.preset == "medium"
        assert cfg.stages.encode.x265_params == tuple(sorted(cfg.stages.encode.x265_params))
        assert ("no-sao", "1") in cfg.stages.encode.x265_params
        assert ("psy-rd", "2.0") in cfg.stages.encode.x265_params
        assert cfg.stages.upscale.model == "2x_AnimeJaNai_HD_V3_Compact"
        assert cfg.stages.descale.kernel == "bicubic"
        assert cfg.stages.deband.radius == 16
        assert cfg.models.manifest == Path("models/manifest.toml")
        assert set(cfg.profiles) == {"full", "upscale-only", "interp-only"}

    def test_profiles_stage_lists(self) -> None:
        cfg = _load_default()
        assert cfg.profiles["full"].stages == (
            "descale",
            "deband",
            "upscale",
            "interpolate",
            "encode",
        )
        assert cfg.profiles["upscale-only"].stages == ("upscale", "encode")
        assert cfg.profiles["interp-only"].stages == ("interpolate", "encode")
        assert cfg.profiles["full"].upload is True

    def test_gofile_upload_defaults(self) -> None:
        cfg = _load_default()
        assert cfg.upload.gofile.token == ""
        assert cfg.upload.gofile.mode == "album"
        assert cfg.upload.gofile.delete_after_upload is False

    def test_gofile_upload_custom_valid(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text(
            '[upload.gofile]\ntoken = "abc"\nmode = "separate"\ndelete_after_upload = true\n',
            encoding="utf-8",
        )
        cfg = load_config(cfg_file)
        assert cfg.upload.gofile.token == "abc"
        assert cfg.upload.gofile.mode == "separate"
        assert cfg.upload.gofile.delete_after_upload is True

    def test_gofile_upload_invalid_mode_rejected(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text(
            '[upload.gofile]\nmode = "invalid_mode"\n',
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match="недопустимый режим"):
            load_config(cfg_file)

    def test_gofile_upload_invalid_delete_flag_rejected(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text(
            '[upload.gofile]\ndelete_after_upload = "yes"\n',
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match="delete_after_upload должен быть булевым"):
            load_config(cfg_file)

    def test_gofile_upload_unknown_key_rejected(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text(
            "[upload.gofile]\nextra_key = 123\n",
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match=r"Неизвестный ключ в \[upload\.gofile\]"):
            load_config(cfg_file)
        cfg = _load_default()
        assert cfg.fetch.gofile.wt_salt == "12af056dacea0b"
        assert cfg.fetch.gofile.language == "en-US"
        assert "Chrome/126.0.0.0" in cfg.fetch.gofile.user_agent


class TestUnknownKeys:
    def test_unknown_stage_key_is_hard_error(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text(
            '[stages.descale]\nkernel = "bicubic"\nnot_a_key = 1\n',
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match="not_a_key"):
            load_config(cfg_file)

    def test_unknown_top_level_section_is_hard_error(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text("[mystery]\nkey = 1\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="mystery"):
            load_config(cfg_file)

    def test_unknown_profile_key_is_hard_error(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text(
            '[profiles.full]\nstages = ["encode"]\nwhoops = true\n', encoding="utf-8"
        )
        with pytest.raises(ConfigError, match="whoops"):
            load_config(cfg_file)


class TestX265ParamValidation:
    """x265 thread knobs: frame-threads has a build limit, `threads` is not a knob."""

    def test_frame_threads_above_limit_is_loud_at_load(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text(
            "[stages.encode.x265_params]\nframe-threads = 32\n",
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match="frame-threads") as exc_info:
            load_config(cfg_file)
        message = str(exc_info.value)
        assert "16" in message
        assert "config.toml" in message

    def test_threads_key_is_rejected_at_load(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text(
            "[stages.encode.x265_params]\nthreads = 32\n",
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match="threads") as exc_info:
            load_config(cfg_file)
        assert "pools" in str(exc_info.value)

    def test_frame_threads_up_to_limit_loads(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text(
            "[stages.encode.x265_params]\nframe-threads = 16\n",
            encoding="utf-8",
        )
        cfg = load_config(cfg_file)
        assert ("frame-threads", "16") in cfg.stages.encode.x265_params


class TestEncodeValueValidation:
    """Scalar encode values: crf range and preset whitelist fail loudly at load."""

    def test_crf_integer_in_range_loads(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text("[stages.encode]\ncrf = 16\n", encoding="utf-8")
        assert load_config(cfg_file).stages.encode.crf == 16

    def test_crf_float_is_legal(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text("[stages.encode]\ncrf = 16.5\n", encoding="utf-8")
        assert load_config(cfg_file).stages.encode.crf == 16.5

    def test_crf_boundary_values_load(self, tmp_path: Path) -> None:
        low = tmp_path / "low.toml"
        low.write_text("[stages.encode]\ncrf = 0\n", encoding="utf-8")
        assert load_config(low).stages.encode.crf == 0
        high = tmp_path / "high.toml"
        high.write_text("[stages.encode]\ncrf = 51\n", encoding="utf-8")
        assert load_config(high).stages.encode.crf == 51

    def test_crf_string_rejected_with_russian_error(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text('[stages.encode]\ncrf = "high"\n', encoding="utf-8")
        with pytest.raises(ConfigError, match="crf") as exc_info:
            load_config(cfg_file)
        message = str(exc_info.value)
        assert "должен быть числом" in message
        assert "stages.encode" in message

    def test_crf_below_range_rejected(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text("[stages.encode]\ncrf = -1\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="вне допустимого диапазона") as exc_info:
            load_config(cfg_file)
        assert "stages.encode" in str(exc_info.value)

    def test_crf_above_range_rejected(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text("[stages.encode]\ncrf = 52\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="вне допустимого диапазона"):
            load_config(cfg_file)

    def test_crf_bool_rejected(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text("[stages.encode]\ncrf = true\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="crf"):
            load_config(cfg_file)

    def test_crf_non_finite_rejected(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text("[stages.encode]\ncrf = nan\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="вне допустимого диапазона"):
            load_config(cfg_file)

    def test_preset_medium_loads(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text('[stages.encode]\npreset = "medium"\n', encoding="utf-8")
        assert load_config(cfg_file).stages.encode.preset == "medium"

    def test_preset_unknown_rejected_with_russian_error(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text('[stages.encode]\npreset = "werkzeug"\n', encoding="utf-8")
        with pytest.raises(ConfigError, match="недопустимый пресет") as exc_info:
            load_config(cfg_file)
        message = str(exc_info.value)
        assert "werkzeug" in message
        assert "veryslow" in message  # hint carries the full official preset list

    def test_preset_non_string_rejected(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text("[stages.encode]\npreset = 5\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="недопустимый пресет"):
            load_config(cfg_file)

    def test_crf_bad_value_in_profile_override_rejected_at_load(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text(
            '[profiles.bad]\nstages = ["encode"]\n[profiles.bad.overrides.encode]\ncrf = "high"\n',
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match="profiles\\.bad\\.overrides\\.encode"):
            load_config(cfg_file)


class TestRationalFps:
    def test_fraction_string(self) -> None:
        assert parse_fps_argument("60000/1001") == (60000, 1001)

    def test_integer_string(self) -> None:
        assert parse_fps_argument("60") == (60, 1)

    def test_ntsc_decimal_maps_to_exact_rational(self) -> None:
        assert parse_fps_argument("23.976") == (24000, 1001)
        assert parse_fps_argument("59.94") == (60000, 1001)

    def test_plain_decimal_exact(self) -> None:
        assert parse_fps_argument("24.5") == (49, 2)

    def test_zero_and_negative_rejected(self) -> None:
        with pytest.raises(ConfigError):
            parse_fps_argument("0")
        with pytest.raises(ConfigError):
            parse_fps_argument("-5")

    def test_garbage_rejected(self) -> None:
        with pytest.raises(ConfigError):
            parse_fps_argument("sixty")


class TestResolveRun:
    def test_profile_selects_stages(self) -> None:
        cfg = _load_default()
        run = resolve_run(cfg, profile="interp-only")
        assert run.stages == ("interpolate", "encode")
        assert run.interpolate.target_fps == (60, 1)  # per-profile deep override

    def test_no_profile_and_no_stages_is_error_listing_profiles(self) -> None:
        cfg = _load_default()
        with pytest.raises(ConfigError, match="interp-only"):
            resolve_run(cfg, profile=None)

    def test_profile_not_found_lists_available(self) -> None:
        cfg = _load_default()
        with pytest.raises(ConfigError, match="upscale-only"):
            resolve_run(cfg, profile="nonexistent")

    def test_stages_override_profile(self) -> None:
        cfg = _load_default()
        run = resolve_run(cfg, profile="full", stages=("interpolate", "encode"))
        assert run.stages == ("interpolate", "encode")

    def test_unknown_stage_name_rejected(self) -> None:
        cfg = _load_default()
        with pytest.raises(ConfigError, match="teleport"):
            resolve_run(cfg, stages=("teleport", "encode"))

    def test_cli_target_fps_beats_profile(self) -> None:
        cfg = _load_default()
        run = resolve_run(cfg, profile="interp-only", target_fps=(60000, 1001))
        assert run.interpolate.target_fps == (60000, 1001)

    def test_descale_selected_without_height_is_error(self) -> None:
        cfg = _load_default()
        with pytest.raises(ConfigError, match="descale"):
            resolve_run(cfg, stages=("descale", "encode"))

    def test_descale_height_from_profile(self) -> None:
        cfg = _load_default()
        run = resolve_run(cfg, profile="upscale-only")
        assert run.descale_height == 1080

    def test_descale_height_from_cli(self) -> None:
        cfg = _load_default()
        run = resolve_run(cfg, stages=("descale", "encode"), descale_height=720)
        assert run.descale_height == 720

    def test_encode_required_for_file_run(self) -> None:
        cfg = _load_default()
        with pytest.raises(ConfigError, match="encode"):
            resolve_run(cfg, stages=("interpolate", "deband"))

    def test_stage_order_is_canonical_even_if_given_unordered(self) -> None:
        cfg = _load_default()
        run = resolve_run(cfg, stages=("encode", "upscale", "interpolate"))
        assert run.stages == ("upscale", "interpolate", "encode")

    def test_upload_flag_precedence(self) -> None:
        cfg = _load_default()
        assert resolve_run(cfg, profile="full").upload is True
        assert resolve_run(cfg, profile="full", upload=False).upload is False
        assert resolve_run(cfg, profile="interp-only", upload=True).upload is True
        assert resolve_run(cfg, profile="interp-only").upload is False

    def test_run_is_frozen(self) -> None:
        cfg = _load_default()
        run = resolve_run(cfg, profile="interp-only")
        import dataclasses

        with pytest.raises(dataclasses.FrozenInstanceError):
            run.stages = ("encode",)  # type: ignore[misc]


class TestNetworkSection:
    def test_default_strict_tls_false(self) -> None:
        cfg = _load_default()
        assert cfg.network.strict_tls is False

    def test_strict_tls_true_parsed(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text("[network]\nstrict_tls = true\n", encoding="utf-8")
        assert load_config(cfg_file).network.strict_tls is True

    def test_unknown_network_key_is_hard_error(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text("[network]\nnope = 1\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="nope"):
            load_config(cfg_file)

    def test_non_bool_strict_tls_is_hard_error(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text('[network]\nstrict_tls = "yes"\n', encoding="utf-8")
        with pytest.raises(ConfigError, match="strict_tls"):
            load_config(cfg_file)
