#!/usr/bin/env bash
# Build the VapourSynth `misc` plugin (SCDetect) into the project .venv.
#
# Why: vsrife's scene-detection path calls misc.SCDetect; the plugin ships with
# the official VS installers but not with the PyPI vapoursynth wheel, and no
# PyPI package provides it. This script compiles it from the canonical source
# (vapoursynth/vs-miscfilters-obsolete, LGPL-2.1) against the VS headers that
# ship inside the wheel. Re-run after `uv sync` wipes/recreates .venv.
#
# Supply-chain pins: miscfilters.cpp is fetched by COMMIT (a moving branch head
# is not acceptable for code compiled into the process); the VS headers come
# from the R81 release tag. The plugin directory is derived from the RUNNING
# project interpreter — no hardcoded python version.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO_ROOT/.venv/bin/python"

MISC_PIN="07e0589a381f7deb3bf533bb459a94482bccc5c7"  # vs-miscfilters-obsolete@master, 2022-01-24 (last upstream commit)
VS_TAG="R81"

PLUGIN_DIR="$("$PY" -c 'import vapoursynth, os; print(os.path.dirname(vapoursynth.__file__))')/plugins/misc"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

BASE="https://raw.githubusercontent.com/vapoursynth"
curl -fsSL "$BASE/vs-miscfilters-obsolete/$MISC_PIN/src/miscfilters.cpp" -o "$WORK/miscfilters.cpp"
curl -fsSL "$BASE/vapoursynth/$VS_TAG/include/VSHelper4.h" -o "$WORK/VSHelper4.h"
curl -fsSL "$BASE/vapoursynth/$VS_TAG/include/VSConstants4.h" -o "$WORK/VSConstants4.h"
curl -fsSL "$BASE/vapoursynth/$VS_TAG/include/VapourSynth4.h" -o "$WORK/VapourSynth4.h"

mkdir -p "$PLUGIN_DIR"
g++ -std=c++17 -O2 -fPIC -shared -I"$WORK" -o "$PLUGIN_DIR/libmisc.so" "$WORK/miscfilters.cpp"

echo "built: $PLUGIN_DIR/libmisc.so"
"$PY" - <<'PY'
import vapoursynth as vs  # noqa: PLC0415

assert hasattr(vs.core, "misc"), "misc namespace not visible after build"
print("verified: core.misc is available")
PY
