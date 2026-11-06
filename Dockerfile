# Multi-stage build for the anime-upscale pipeline.
#
# Contract (docs/docker.md):
#   * builder  = python:3.14-slim-trixie + gcc/g++/make + uv (from the pinned
#     official uv image); deps layer split for cache reuse; misc.SCDetect plugin
#     compiled into the venv; model weights fetched + sha256-verified at build.
#   * final    = python:3.14-slim-trixie + apt ffmpeg/ffprobe (libx265 10-bit)
#     + ca-certificates; only the built venv and models/ are copied; non-root
#     uid/gid 1000 `app`; ENTRYPOINT = the pipeline CLI.
#   * No HEALTHCHECK: this is a run-once CLI tool — the process exit code IS the
#     health signal (a healthcheck would only re-run the binary).
#   * No ENV configuration knobs: HOME is filesystem mechanics, not config; the
#     pipeline reads everything from config.toml (owner ruling: zero env vars).
#   * The final stage intentionally does NOT copy config.toml — it arrives as a
#     bind mount at runtime (single source of truth on the host).

# syntax=docker/dockerfile:1

ARG BASE_IMAGE=python:3.14-slim-trixie
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.13

# Named stage so builder can COPY --from without variable expansion
# (BuildKit does not support --from=${VAR}).
FROM ${UV_IMAGE} AS uv-sources

# ---------------------------------------------------------------- builder ----
FROM ${BASE_IMAGE} AS builder

# Reuse the base image's system Python across stages; copy (not hardlink) from
# the cache mount, which lives on a different filesystem (official uv pattern).
ENV UV_PYTHON_DOWNLOADS=0 \
    UV_LINK_MODE=copy

# gcc/g++/make: misc.SCDetect plugin build (scripts/build_misc_plugin.sh needs
# g++). curl: the script fetches pinned sources from raw.githubusercontent.com.
# git: uv resolves the vsdeband git dependency (tool.uv.sources) through the
# git CLI during uv sync.
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc g++ make curl git \
 && rm -rf /var/lib/apt/lists/*

COPY --from=uv-sources /uv /uvx /bin/

WORKDIR /app

# Deps layer: only the manifests, so source edits never invalidate the heavy
# torch cu130 layer. `--frozen` installs exactly the universal lock (which
# carries the linux cu130 wheels); `--no-editable` keeps the venv relocatable.
COPY pyproject.toml README.md uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-editable

# Project sources + scripts (misc plugin) + model manifests/config.
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY models/ ./models/
COPY config.toml ./

# Full sync (builds the project wheel into the venv), then bake everything
# version-specific into the venv/image:
#   1. misc.SCDetect compiled into venv site-packages (needs g++ — builder only).
#   2. `models fetch` downloads the sha256-pinned AnimeJaNai .pth and RIFE 4.25
#      weights, verifies hashes, and seeds the RIFE pickle into venv
#      site-packages vsrife — the runtime never downloads weights on first run.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-editable \
 && bash scripts/build_misc_plugin.sh \
 && .venv/bin/anime-pipeline models fetch

# ------------------------------------------------------------------ final ----
FROM ${BASE_IMAGE}

# ffmpeg/ffprobe from Debian trixie: libx265 10-bit (main10) included.
# ca-certificates: outbound TLS for gofile upload/fetch + DoH ladder.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Non-root runtime user; uid/gid 1000 matches the typical first user on Linux
# hosts and the default WSL2 user, so bind-mounted input/output stay usable.
RUN groupadd --system --gid 1000 app \
 && useradd --system --uid 1000 --gid 1000 --create-home --home-dir /home/app app

# The venv is self-contained (uv --no-editable: project installed as a wheel;
# misc plugin and RIFE weights live inside site-packages). models/ holds the
# sha256-verified weight files fetched at build time. Both stay root-owned —
# the app only needs read/execute; all runtime writes go to mounts + HOME.
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/models /app/models

# HOME must be writable (TensorRT engine cache lives in ~/.cache/vsrife,
# config: [hardware] trt_cache_dir). No other env vars by design.
ENV HOME=/home/app

WORKDIR /app

USER app

ENTRYPOINT ["/app/.venv/bin/anime-pipeline"]
