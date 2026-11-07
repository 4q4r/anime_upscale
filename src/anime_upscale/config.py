"""
TOML configuration: load, validate, and merge into ONE frozen object.

Contract (architecture plan §3/§4):
- single config file, no environment variables anywhere;
- merge precedence: CLI > profile (deep overrides) > [stages.*] > built-in defaults;
- unknown keys are a hard error — nothing is silently accepted;
- canonical stage order is fixed; profiles and --stages select a subset of it.
"""

import dataclasses
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

STAGE_ORDER: tuple[str, ...] = ("descale", "deband", "upscale", "interpolate", "encode")

# [hardware] memory_fraction: torch caching-allocator cap as a share of VRAM
# (torch.cuda.set_per_process_memory_fraction). 1.0 = uncapped; below 1.0 the
# allocator raises CUDA OOM earlier than the driver would spill to system RAM.
_MEMORY_FRACTION_MIN = 0.1
_MEMORY_FRACTION_MAX = 1.0
_MEMORY_FRACTION_DEFAULT = 0.9

# Common broadcast decimals map to their EXACT rationals (never float-round).
_NTSC_TABLE: dict[Decimal, tuple[int, int]] = {
    Decimal("23.976"): (24000, 1001),
    Decimal("29.97"): (30000, 1001),
    Decimal("47.952"): (48000, 1001),
    Decimal("59.94"): (60000, 1001),
    Decimal("119.88"): (120000, 1001),
}


class ConfigError(Exception):
    """User-facing configuration error (loud, Russian message)."""


@dataclass(frozen=True)
class DescaleConfig:
    """[stages.descale] — ядро и смещения обратного масштабирования."""

    kernel: str = "bicubic"
    src_left: float = 0.0
    src_top: float = 0.0


@dataclass(frozen=True)
class DebandConfig:
    """[stages.deband] — радиус, пороги и зерно f3kdb."""

    radius: int = 16
    threshold: tuple[int, int, int] = (48, 48, 48)
    grain: tuple[int, int] = (0, 0)


@dataclass(frozen=True)
class UpscaleConfig:
    """[stages.upscale] — модель из манифеста и точность."""

    model: str = "2x_AnimeJaNai_HD_V3_Compact"
    fp16: bool = True


@dataclass(frozen=True)
class InterpolateConfig:
    """[stages.interpolate] — RIFE: модель, целевой fps, TensorRT."""

    model: str = "4.25"
    target_fps: tuple[int, int] = (60000, 1001)
    scene_detection: bool = True
    scene_threshold: float = 0.15
    trt: bool = True
    trt_cache_dir: str = "~/.cache/vsrife"


@dataclass(frozen=True)
class EncodeConfig:
    """[stages.encode] — лист флагов libx265 (research/05)."""

    crf: int | float = 16
    preset: str = "slow"
    pix_fmt: str = "yuv420p10le"
    profile: str = "main10"
    colorspace: str = "bt709"
    color_primaries: str = "bt709"
    color_trc: str = "bt709"
    color_range: str = "tv"
    x265_params: tuple[tuple[str, str], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class StagesConfig:
    """Все секции [stages.*] одним объектом."""

    descale: DescaleConfig = field(default_factory=DescaleConfig)
    deband: DebandConfig = field(default_factory=DebandConfig)
    upscale: UpscaleConfig = field(default_factory=UpscaleConfig)
    interpolate: InterpolateConfig = field(default_factory=InterpolateConfig)
    encode: EncodeConfig = field(default_factory=EncodeConfig)


@dataclass(frozen=True)
class HardwareOverrides:
    """[hardware] — явные override'ы автодетекта (§9)."""

    num_streams: int | None = None
    gpus: tuple[int, ...] | None = None
    encode_threads: int | None = None
    # Доля VRAM для аллокатора PyTorch (см. константы диапазона выше);
    # дефолт 0.9 = OOM внутри процесса раньше WDDM-спилла в системную RAM.
    memory_fraction: float = _MEMORY_FRACTION_DEFAULT


@dataclass(frozen=True)
class ModelsConfig:
    """[models] — каталог и манифест моделей."""

    dir: Path = Path("models")
    manifest: Path = Path("models/manifest.toml")


@dataclass(frozen=True)
class GofileUploadConfig:
    """[upload.gofile] — опциональный токен аккаунта, режим загрузки и автоудаление."""

    token: str = ""
    mode: str = "album"
    delete_after_upload: bool = False


@dataclass(frozen=True)
class GofileFetchConfig:
    """[fetch.gofile] — константы wt-схемы с переопределениями."""

    wt_salt: str = "12af056dacea0b"
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    )
    language: str = "en-US"


@dataclass(frozen=True)
class FetchConfig:
    """[fetch] — секция ингеста ссылок."""

    gofile: GofileFetchConfig = field(default_factory=GofileFetchConfig)


@dataclass(frozen=True)
class NetworkConfig:
    """[network] — сетевые политики транспорта (gofile/fetch request layer)."""

    strict_tls: bool = False


@dataclass(frozen=True)
class ToolsConfig:
    """External tool overrides (plan §12.3); None = PATH lookup."""

    ffmpeg: str | None = None
    ffprobe: str | None = None


@dataclass(frozen=True)
class UploadConfig:
    """[upload] — секция пост-шага выгрузки."""

    gofile: GofileUploadConfig = field(default_factory=GofileUploadConfig)


@dataclass(frozen=True)
class Profile:
    """Один [profiles.*]: список стадий и опциональные ключи."""

    name: str
    stages: tuple[str, ...]
    upload: bool = False
    descale_height: int | None = None
    stage_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class Config:
    """Корневой объект config.toml (validated, frozen)."""

    pipeline_order: tuple[str, ...] = STAGE_ORDER
    hardware: HardwareOverrides = field(default_factory=HardwareOverrides)
    stages: StagesConfig = field(default_factory=StagesConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    upload: UploadConfig = field(default_factory=UploadConfig)
    fetch: FetchConfig = field(default_factory=FetchConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    profiles: dict[str, Profile] = field(default_factory=dict)


@dataclass(frozen=True)
class RunConfig:
    """Эффективная конфигурация ОДНОГО прогона (все слои merge применены)."""

    stages: tuple[str, ...]
    descale: DescaleConfig
    deband: DebandConfig
    upscale: UpscaleConfig
    interpolate: InterpolateConfig
    encode: EncodeConfig
    descale_height: int | None
    upload: bool
    memory_fraction: float = _MEMORY_FRACTION_DEFAULT


def parse_fps_argument(value: str) -> tuple[int, int]:
    """
    Parse a target fps into an exact (num, den) rational.

    Accepts "60000/1001", "60", and broadcast decimals ("23.976" → 24000/1001).
    """
    text = value.strip()
    if not text:
        msg = "Целевой fps не задан (пустая строка)."
        raise ConfigError(msg)
    if "/" in text:
        num_text, _, den_text = text.partition("/")
        try:
            num, den = int(num_text), int(den_text)
        except ValueError as exc:
            msg = (
                f"Некорректный fps «{value}»: ожидается формат "
                "ЧИСЛИТЕЛЬ/ЗНАМЕНАТЕЛЬ (например 60000/1001)."
            )
            raise ConfigError(msg) from exc
        if den <= 0 or num <= 0:
            msg = f"Некорректный fps «{value}»: числитель и знаменатель должны быть > 0."
            raise ConfigError(msg)
        return (num, den)
    decimal_value = _parse_decimal(text, value)
    exact = _NTSC_TABLE.get(decimal_value)
    if exact is not None:
        return exact
    fraction = Fraction(decimal_value)
    return (fraction.numerator, fraction.denominator)


def _parse_decimal(text: str, original: str) -> Decimal:
    try:
        dec = Decimal(text)
    except ArithmeticError as exc:
        msg = (
            f"Некорректный fps «{original}»: ожидается целое, "
            "десятичная дробь или ЧИСЛИТЕЛЬ/ЗНАМЕНАТЕЛЬ."
        )
        raise ConfigError(msg) from exc
    if not dec.is_finite() or dec <= 0:
        msg = f"Некорректный fps «{original}»: значение должно быть положительным числом."
        raise ConfigError(msg)
    return dec


# ── Schema walk: unknown keys are hard errors ────────────────────────────────────────────

_THRESHOLD_LEN = 3
_GRAIN_LEN = 2

# x265's hard frame-thread ceiling (X265_MAX_FRAME_THREADS); 0 lets x265 decide.
_X265_MAX_FRAME_THREADS = 16

# Official x265 rate-control range: --crf <0..51.0> (float values are legal).
_X265_CRF_MIN = 0.0
_X265_CRF_MAX = 51.0

# Official x265 --preset names (x265 CLI documentation), speed → quality.
_X265_PRESETS: tuple[str, ...] = (
    "ultrafast",
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",
    "slow",
    "slower",
    "veryslow",
    "placebo",
)

_STAGE_KEYS: dict[str, set[str]] = {
    dataclass_type.__name__.removesuffix("Config").lower(): {f.name for f in fields(dataclass_type)}
    for dataclass_type in (DescaleConfig, DebandConfig, UpscaleConfig, InterpolateConfig)
}
_PROFILE_KEYS = {"stages", "upload", "descale_height", "overrides"}


def _reject_unknown(table: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = set(table) - allowed
    if unknown:
        msg = (
            f"Неизвестный ключ в [{where}]: {', '.join(sorted(unknown))}. "
            f"Разрешены: {', '.join(sorted(allowed))}."
        )
        raise ConfigError(msg)


def _validate_profile_table(name: str, table: dict[str, Any]) -> Profile:
    _reject_unknown(table, _PROFILE_KEYS, f"profiles.{name}")
    stages_raw = table.get("stages")
    if stages_raw is None:
        msg = f"Профиль «{name}»: отсутствует обязательный ключ stages."
        raise ConfigError(msg)
    stages = tuple(_check_stage_list(stages_raw, f"profiles.{name}.stages"))
    overrides_raw = table.get("overrides", {})
    if not isinstance(overrides_raw, dict):
        msg = f"Профиль «{name}»: overrides должен быть таблицей стадий."
        raise ConfigError(msg)
    stage_overrides: dict[str, dict[str, Any]] = {}
    for stage_name, params in overrides_raw.items():
        if stage_name not in STAGE_ORDER:
            msg = f"Профиль «{name}»: неизвестная стадия «{stage_name}» в overrides."
            raise ConfigError(msg)
        allowed = _STAGE_KEYS.get(stage_name, set())
        if stage_name == "encode":
            allowed = {f.name for f in fields(EncodeConfig)} - {"x265_params"}
        where = f"profiles.{name}.overrides.{stage_name}"
        _reject_unknown(params, allowed, where)
        if stage_name == "encode":
            _validate_encode_values(params, where)
        stage_overrides[stage_name] = dict(params)
    descale_height = table.get("descale_height")
    bad_height = descale_height is not None and (
        not isinstance(descale_height, int) or descale_height <= 0
    )
    if bad_height:
        msg = f"Профиль «{name}»: descale_height должен быть положительным целым."
        raise ConfigError(msg)
    return Profile(
        name=name,
        stages=stages,
        upload=bool(table.get("upload", False)),
        descale_height=descale_height,
        stage_overrides=stage_overrides,
    )


def _check_stage_list(raw: Any, where: str) -> list[str]:  # noqa: ANN401
    if not isinstance(raw, list) or not raw or not all(isinstance(s, str) for s in raw):
        msg = f"[{where}]: stages должен быть непустым списком имён стадий."
        raise ConfigError(msg)
    return raw


def _validate_stages(table: dict[str, Any]) -> StagesConfig:
    known = {"descale", "deband", "upscale", "interpolate", "encode"}
    _reject_unknown(table, known, "stages")
    descale = dict(table.get("descale", {}))
    deband = dict(table.get("deband", {}))
    upscale = dict(table.get("upscale", {}))
    interpolate = dict(table.get("interpolate", {}))
    encode = dict(table.get("encode", {}))

    _reject_unknown(descale, _STAGE_KEYS["descale"], "stages.descale")
    _reject_unknown(deband, _STAGE_KEYS["deband"], "stages.deband")
    _reject_unknown(upscale, _STAGE_KEYS["upscale"], "stages.upscale")
    _reject_unknown(interpolate, _STAGE_KEYS["interpolate"], "stages.interpolate")
    encode_extra = encode.pop("x265_params", {})
    encode_allowed = {f.name for f in fields(EncodeConfig)} - {"x265_params"}
    _reject_unknown(encode, encode_allowed, "stages.encode")
    _validate_encode_values(encode, "stages.encode")
    if not isinstance(encode_extra, dict):
        msg = "[stages.encode.x265_params]: ожидается таблица параметров x265."
        raise ConfigError(msg)

    if "target_fps" in interpolate:
        if not isinstance(interpolate["target_fps"], str):
            msg = "[stages.interpolate]: target_fps должен быть строкой вида «60000/1001»."
            raise ConfigError(msg)
        interpolate["target_fps"] = parse_fps_argument(interpolate["target_fps"])
    if "trt_cache_dir" in interpolate and not isinstance(interpolate["trt_cache_dir"], str):
        msg = "[stages.interpolate]: trt_cache_dir должен быть строкой (путём)."
        raise ConfigError(msg)

    threshold = deband.get("threshold", [48, 48, 48])
    grain = deband.get("grain", [0, 0])
    if not isinstance(threshold, (list, tuple)) or len(threshold) != _THRESHOLD_LEN:
        msg = "[stages.deband]: threshold — список из трёх значений (luma, u, v)."
        raise ConfigError(msg)
    if not isinstance(grain, (list, tuple)) or len(grain) != _GRAIN_LEN:
        msg = "[stages.deband]: grain — список из двух значений (luma, chroma)."
        raise ConfigError(msg)

    x265_pairs = tuple(sorted((str(k), _x265_value(v)) for k, v in encode_extra.items()))
    validate_x265_params(x265_pairs)

    return StagesConfig(
        descale=DescaleConfig(**descale),
        deband=DebandConfig(
            radius=deband.get("radius", 16),
            threshold=(int(threshold[0]), int(threshold[1]), int(threshold[2])),
            grain=(int(grain[0]), int(grain[1])),
        ),
        upscale=UpscaleConfig(**upscale),
        interpolate=InterpolateConfig(**interpolate),
        encode=EncodeConfig(**encode, x265_params=x265_pairs),
    )


def _x265_value(value: Any) -> str:  # noqa: ANN401
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float, str)):
        return str(value)
    msg = f"[stages.encode.x265_params]: неподдерживаемый тип значения {value!r}."
    raise ConfigError(msg)


def validate_x265_params(pairs: tuple[tuple[str, str], ...]) -> None:
    """
    Reject x265 thread knobs that break the encoder or the ffmpeg contract.

    ``threads`` is an ffmpeg option, not an x265 parameter — the CPU budget is
    ``pools``. ``frame-threads`` has a hard x265 ceiling (0..16); values above
    it make libx265 refuse to open with a cryptic exit code instead of a
    config error.
    """
    values = dict(pairs)
    if "threads" in values:
        msg = (
            "[stages.encode.x265_params]: ключ threads не поддерживается x265 "
            "(это не ffmpeg -threads). Задайте pools=<N> для бюджета CPU "
            "в config.toml."
        )
        raise ConfigError(msg)
    raw = values.get("frame-threads")
    if raw is None:
        return
    try:
        value = int(raw)
    except ValueError as exc:
        msg = (
            f"[stages.encode.x265_params]: frame-threads={raw!r} — ожидается целое "
            f"0..{_X265_MAX_FRAME_THREADS} (config.toml)."
        )
        raise ConfigError(msg) from exc
    if value < 0 or value > _X265_MAX_FRAME_THREADS:
        msg = (
            f"[stages.encode.x265_params]: frame-threads={value} вне предела x265 "
            f"(0..{_X265_MAX_FRAME_THREADS}). Уменьшите frame-threads в config.toml; "
            "бюджет CPU задаётся pools=<N>."
        )
        raise ConfigError(msg)


def _validate_encode_values(params: dict[str, Any], where: str) -> None:
    """
    Validate libx265 scalar values so bad ones fail at config load, not at encode.

    ``crf`` accepts integers and floats within the official x265 range
    ``--crf <0..51.0>`` (bool is rejected despite being an int subclass);
    ``preset`` must be one of the ten official x265 preset names.
    """
    crf = params.get("crf")
    if crf is not None:
        if isinstance(crf, bool) or not isinstance(crf, (int, float)):
            msg = f"[{where}]: crf должен быть числом (целым или дробным), получено {crf!r}."
            raise ConfigError(msg)
        if not _X265_CRF_MIN <= crf <= _X265_CRF_MAX:
            msg = (
                f"[{where}]: crf={crf} вне допустимого диапазона x265 "
                f"({_X265_CRF_MIN:g}..{_X265_CRF_MAX:g})."
            )
            raise ConfigError(msg)
    preset = params.get("preset")
    if preset is not None and (not isinstance(preset, str) or preset not in _X265_PRESETS):
        shown = preset if isinstance(preset, str) else repr(preset)
        msg = f"[{where}]: недопустимый пресет {shown}. Разрешены: {', '.join(_X265_PRESETS)}."
        raise ConfigError(msg)


def _validate_upload(table: dict[str, Any]) -> UploadConfig:
    """[upload] — секция пост-шага выгрузки."""
    _reject_unknown(table, {"gofile"}, "upload")
    gofile_raw = table.get("gofile", {})
    _reject_unknown(gofile_raw, {"token", "mode", "delete_after_upload"}, "upload.gofile")
    upload_mode = str(gofile_raw.get("mode", "album"))
    if upload_mode not in {"album", "separate"}:
        msg = (
            f"[upload.gofile]: недопустимый режим «{upload_mode}». Разрешены: «album», «separate»."
        )
        raise ConfigError(msg)
    delete_after_upload = gofile_raw.get("delete_after_upload", False)
    if not isinstance(delete_after_upload, bool):
        msg = "[upload.gofile]: delete_after_upload должен быть булевым значением (true/false)."
        raise ConfigError(msg)
    return UploadConfig(
        gofile=GofileUploadConfig(
            token=str(gofile_raw.get("token", "")),
            mode=upload_mode,
            delete_after_upload=delete_after_upload,
        )
    )


def _validate_network(table: dict[str, Any]) -> NetworkConfig:
    """[network] — transport policy section (adaptive TLS kill-switch)."""
    _reject_unknown(table, {"strict_tls"}, "network")
    strict_tls = table.get("strict_tls", False)
    if not isinstance(strict_tls, bool):
        msg = "[network]: strict_tls должен быть булевым значением (true/false)."
        raise ConfigError(msg)
    return NetworkConfig(strict_tls=strict_tls)


def _validate_memory_fraction(raw: dict[str, Any]) -> float:
    """
    Validate [hardware] memory_fraction: allocator cap share of VRAM.

    bool is rejected BEFORE the numeric checks (sharp edge: ``true`` would
    otherwise silently pass as 1.0 — an int subclass). Outside 0.1..1.0 the
    config is loud Russian: below the floor the cap would strangle the
    working set, above 1.0 torch itself rejects the value.
    """
    fraction_raw = raw.get("memory_fraction", _MEMORY_FRACTION_DEFAULT)
    if isinstance(fraction_raw, bool) or not isinstance(fraction_raw, (int, float)):
        msg = (
            "[hardware]: memory_fraction должен быть числом "
            f"({_MEMORY_FRACTION_MIN}..{_MEMORY_FRACTION_MAX}), получено {fraction_raw!r}."
        )
        raise ConfigError(msg)
    if not _MEMORY_FRACTION_MIN <= fraction_raw <= _MEMORY_FRACTION_MAX:
        msg = (
            f"[hardware]: memory_fraction={fraction_raw} вне допустимого диапазона "
            f"{_MEMORY_FRACTION_MIN}..{_MEMORY_FRACTION_MAX}. Значение = доля VRAM, "
            "отданная аллокатору PyTorch; запас ниже 1.0 оставляет место под "
            "CUDA-контекст и фрагментацию, чтобы OOM случился внутри процесса, "
            "а не уходом памяти в системную RAM."
        )
        raise ConfigError(msg)
    return float(fraction_raw)


def load_config(path: Path) -> Config:
    """Load and validate config.toml into ONE frozen Config object."""
    if not path.is_file():
        msg = f"Файл конфигурации не найден: {path}"
        raise ConfigError(msg)
    try:
        # TOML payloads are honestly dynamic until validated below.
        raw: dict[str, Any] = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        msg = f"Не удалось разобрать {path}: {exc}"
        raise ConfigError(msg) from exc

    known_sections = {
        "pipeline",
        "hardware",
        "stages",
        "models",
        "network",
        "upload",
        "fetch",
        "tools",
        "profiles",
    }
    _reject_unknown(raw, known_sections, "config")

    pipeline = raw.get("pipeline", {})
    _reject_unknown(pipeline, {"order"}, "pipeline")
    order_raw = pipeline.get("order", list(STAGE_ORDER))
    order = tuple(_check_stage_list(order_raw, "pipeline"))
    if order != STAGE_ORDER:
        raise ConfigError(
            "[pipeline]: order должен содержать ровно канонический порядок стадий: "
            + " → ".join(STAGE_ORDER)
        )

    hardware_raw = raw.get("hardware", {})
    _reject_unknown(
        hardware_raw, {"num_streams", "gpus", "encode_threads", "memory_fraction"}, "hardware"
    )
    gpus_raw = hardware_raw.get("gpus")
    if gpus_raw is not None and (
        not isinstance(gpus_raw, list) or not all(isinstance(g, int) for g in gpus_raw)
    ):
        msg = "[hardware]: gpus должен быть списком индексов GPU."
        raise ConfigError(msg)
    hardware = HardwareOverrides(
        num_streams=hardware_raw.get("num_streams"),
        gpus=tuple(gpus_raw) if gpus_raw is not None else None,
        encode_threads=hardware_raw.get("encode_threads"),
        memory_fraction=_validate_memory_fraction(hardware_raw),
    )

    models_raw = raw.get("models", {})
    _reject_unknown(models_raw, {"dir", "manifest"}, "models")
    models = ModelsConfig(
        dir=Path(models_raw.get("dir", "models")),
        manifest=Path(models_raw.get("manifest", "models/manifest.toml")),
    )

    network = _validate_network(raw.get("network", {}))
    upload = _validate_upload(raw.get("upload", {}))

    fetch_raw = raw.get("fetch", {})
    _reject_unknown(fetch_raw, {"gofile"}, "fetch")
    gofile_fetch_raw = fetch_raw.get("gofile", {})
    _reject_unknown(gofile_fetch_raw, {"wt_salt", "user_agent", "language"}, "fetch.gofile")
    fetch = FetchConfig(gofile=GofileFetchConfig(**gofile_fetch_raw))

    tools_raw = raw.get("tools", {})
    _reject_unknown(tools_raw, {"ffmpeg", "ffprobe"}, "tools")
    tools = ToolsConfig(
        ffmpeg=tools_raw.get("ffmpeg"),
        ffprobe=tools_raw.get("ffprobe"),
    )

    profiles_raw = raw.get("profiles", {})
    if not isinstance(profiles_raw, dict):
        msg = "[profiles]: ожидается таблица профилей."
        raise ConfigError(msg)
    profiles: dict[str, Profile] = {}
    for name, table in profiles_raw.items():
        if not isinstance(table, dict):
            msg = f"[profiles.{name}]: ожидается таблица профиля."
            raise ConfigError(msg)
        profiles[name] = _validate_profile_table(name, table)

    return Config(
        pipeline_order=order,
        hardware=hardware,
        stages=_validate_stages(raw.get("stages", {})),
        models=models,
        network=network,
        upload=upload,
        fetch=fetch,
        tools=tools,
        profiles=profiles,
    )


# ── Run resolution: CLI > profile > [stages.*] > defaults ────────────────────────────────


def _select_stages(
    config: Config,
    profile: str | None,
    stages: tuple[str, ...] | None,
) -> tuple[tuple[str, ...], Profile | None]:
    """Resolve the run subset: --stages beats --profile; no selection fails loudly."""
    if stages is not None:
        for name in stages:
            if name not in STAGE_ORDER:
                msg = f"Неизвестная стадия «{name}». Доступны: {', '.join(STAGE_ORDER)}."
                raise ConfigError(msg)
        return tuple(sorted(set(stages), key=STAGE_ORDER.index)), None
    if profile is not None:
        chosen = config.profiles.get(profile)
        if chosen is None:
            available = ", ".join(sorted(config.profiles)) or "<профили не заданы>"
            msg = f"Профиль «{profile}» не найден. Доступные профили: {available}."
            raise ConfigError(msg)
        return chosen.stages, chosen
    available = ", ".join(sorted(config.profiles)) or "<профили не заданы>"
    msg = (
        "Не выбран ни профиль, ни набор стадий. Укажите --profile (доступные: "
        f"{available}) или --stages."
    )
    raise ConfigError(msg)


def resolve_run(  # noqa: PLR0913 - mirrors the CLI surface 1:1
    config: Config,
    profile: str | None = None,
    stages: tuple[str, ...] | None = None,
    target_fps: tuple[int, int] | None = None,
    descale_height: int | None = None,
    *,
    upload: bool | None = None,
) -> RunConfig:
    """Resolve the effective per-run configuration (all precedence layers applied)."""
    selected, chosen = _select_stages(config, profile, stages)

    if "encode" not in selected:
        msg = (
            "Стадия encode обязательна: без кодировщика запуск файла не имеет выхода. "
            f"Выбранные стадии: {', '.join(selected)}."
        )
        raise ConfigError(msg)

    effective_interpolate = _apply_overrides(config.stages.interpolate, chosen, "interpolate")
    if target_fps is not None:
        effective_interpolate = _replace(effective_interpolate, target_fps=target_fps)

    height = descale_height
    if height is None and chosen is not None:
        height = chosen.descale_height
    if "descale" in selected and height is None:
        msg = (
            "Для стадии descale нужна целевая высота: задайте --descale-height N "
            "или ключ descale_height в профиле."
        )
        raise ConfigError(msg)

    upload_effective = upload
    if upload_effective is None and chosen is not None:
        upload_effective = chosen.upload
    if upload_effective is None:
        upload_effective = False

    effective_descale = _apply_overrides(config.stages.descale, chosen, "descale")
    effective_deband = _apply_overrides(config.stages.deband, chosen, "deband")
    effective_upscale = _apply_overrides(config.stages.upscale, chosen, "upscale")
    effective_encode = _apply_overrides(config.stages.encode, chosen, "encode")

    return RunConfig(
        stages=selected,
        descale=effective_descale,
        deband=effective_deband,
        upscale=effective_upscale,
        interpolate=effective_interpolate,
        encode=effective_encode,
        descale_height=height,
        upload=upload_effective,
        memory_fraction=config.hardware.memory_fraction,
    )


def _apply_overrides[T: object](base: T, profile: Profile | None, stage_name: str) -> T:
    """Apply one profile's deep overrides ([profiles.NAME.overrides.STAGE]) over defaults."""
    if profile is None or stage_name not in profile.stage_overrides:
        return base
    overrides = dict(profile.stage_overrides[stage_name])
    if stage_name == "interpolate" and "target_fps" in overrides:
        if not isinstance(overrides["target_fps"], str):
            msg = (
                f"[profiles.{profile.name}.overrides.interpolate]: target_fps должен быть строкой."
            )
            raise ConfigError(msg)
        overrides["target_fps"] = parse_fps_argument(overrides["target_fps"])
    if not is_dataclass(base):  # pragma: no cover - all stage configs are dataclasses
        msg = f"Внутренняя ошибка конфигурации: стадия «{stage_name}»."
        raise ConfigError(msg)
    return base.__class__(**overrides)  # type: ignore[arg-type,return-value]


def _replace[T: object](obj: T, **changes: Any) -> T:  # noqa: ANN401
    """Frozen-dataclass replace helper."""
    return dataclasses.replace(obj, **changes)  # ty: ignore[invalid-argument-type]
