#!/usr/bin/env bash
# Build the VapourSynth `misc` plugin (SCDetect) into the project .venv.
#
# Why: vsrife's scene-detection path calls misc.SCDetect; the plugin ships with
# the official VS installers but not with the PyPI vapoursynth wheel, and no
# PyPI package provides it. This script compiles it from the canonical source
# (vapoursynth/vs-miscfilters-obsolete, LGPL-2.1) against the VS headers that
# ship inside the wheel. Re-run after `uv sync` wipes/recreates .venv.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLUGIN_DIR="$REPO_ROOT/.venv/lib/python3.14/site-packages/vapoursynth/plugins/misc"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

BASE="https://raw.githubusercontent.com/vapoursynth"
curl -fsSL "$BASE/vs-miscfilters-obsolete/master/src/miscfilters.cpp" -o "$WORK/miscfilters.cpp"
curl -fsSL "$BASE/vapoursynth/R81/include/VSHelper4.h" -o "$WORK/VSHelper4.h"
curl -fsSL "$BASE/vapoursynth/R81/include/VSConstants4.h" -o "$WORK/VSConstants4.h"
curl -fsSL "$BASE/vapoursynth/R81/include/VapourSynth4.h" -o "$WORK/VapourSynth4.h"

mkdir -p "$PLUGIN_DIR"
g++ -std=c++17 -O2 -fPIC -shared -I"$WORK" -o "$PLUGIN_DIR/libmisc.so" "$WORK/miscfilters.cpp"

echo "built: $PLUGIN_DIR/libmisc.so"
"$REPO_ROOT/.venv/bin/python" - <<'PY'
import vapoursynth as vs  # noqa: PLC0415

assert hasattr(vs.core, "misc"), "misc namespace not visible after build"
print("verified: core.misc is available")
PY
