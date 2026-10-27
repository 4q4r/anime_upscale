"""
Stage registry: name → builder(ctx, clip) -> clip (plan §2).

Canonical order and subset validation live in config.resolve_run; this registry
only binds names to implementations. Unknown stage = loud error.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import vapoursynth as vs

from anime_upscale.config import RunConfig
from anime_upscale.hardware import HardwareProfile
from anime_upscale.reporting import Reporter


@dataclass(frozen=True)
class StageContext:
    """Read-only bundle every stage builder receives."""

    run: RunConfig
    hardware: HardwareProfile
    project_root: Path
    reporter: Reporter
    models_dir: Path = Path("models")
    models_manifest: Path = Path("models/manifest.toml")


StageBuilder = Callable[[StageContext, vs.VideoNode], vs.VideoNode]

STAGE_BUILDERS: dict[str, StageBuilder] = {}


def register(name: str) -> Callable[[StageBuilder], StageBuilder]:
    """Bind a builder under a canonical stage name (duplicate = loud error)."""

    def wrap(builder: StageBuilder) -> StageBuilder:
        if name in STAGE_BUILDERS:
            msg = f"Стадия «{name}» уже зарегистрирована."
            raise RuntimeError(msg)
        STAGE_BUILDERS[name] = builder
        return builder

    return wrap


def get_builder(name: str) -> StageBuilder:
    """Look up a registered builder; unknown name = loud error."""
    try:
        return STAGE_BUILDERS[name]
    except KeyError:
        msg = f"Неизвестная стадия «{name}». Зарегистрированы: {', '.join(sorted(STAGE_BUILDERS))}."
        raise RuntimeError(msg) from None
