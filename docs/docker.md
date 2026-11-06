<div align="center">

# 🐳 Docker deployment

**Run the whole pipeline from one image: Python 3.14 venv (torch cu130, TensorRT, VapourSynth), compiled misc plugin, sha256-pinned model weights, and ffmpeg — baked at build time. The host provides only NVIDIA driver + Docker; `input/`, `output/`, `config.toml` cross the boundary as bind mounts, and every GPU is injected automatically.**

![Docker](https://img.shields.io/badge/Docker-multi--stage-2496ED?logo=docker&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.14--slim--trixie-3776AB?logo=python&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-cu130-76B900?logo=nvidia&logoColor=white)
![GPU](https://img.shields.io/badge/GPU-gpus%3A%20all-76B900?logo=nvidia&logoColor=white)

</div>

## 📑 Table of Contents

- [📦 Image contract](#-image-contract)
- [🚀 Running the pipeline](#-running-the-pipeline)
- [🖥️ Host setup](#️-host-setup)
- [🩺 Doctor checklist](#-doctor-checklist)
- [🏗️ Building the image](#️-building-the-image)
- [💾 TensorRT engine cache](#-tensorrt-engine-cache)
- [📉 Degradation without a GPU](#-degradation-without-a-gpu)
- [🔐 User mapping and ownership](#-user-mapping-and-ownership)
- [📏 Size budget](#-size-budget)

## 📦 Image contract

Everything version-specific is baked into the image at build time (multi-stage
`Dockerfile`, official [uv pattern](https://docs.astral.sh/uv/guides/integration/docker/)):
the `.venv` is synced from the committed `uv.lock` (`--frozen`/`--locked`, non-editable),
the VapourSynth `misc.SCDetect` plugin is compiled into the venv
(`scripts/build_misc_plugin.sh`, commit-pinned sources), and `anime-pipeline models fetch`
downloads and sha256-verifies the AnimeJaNai `.pth` plus the RIFE 4.25 weights —
including the seed into venv site-packages `vsrife` — so **the runtime never downloads
model binaries on first use**.

**Inside the image:**

| Component | Delivery |
| :-- | :-- |
| Python 3.14 venv | `uv sync` from committed `uv.lock` (torch cu130, torch-tensorrt, vapoursynth, bestsource, descale, vsdeband, vsrife, spandrel, numpy, tqdm, requests) |
| `misc.SCDetect` plugin | compiled into venv site-packages at build (builder stage only — g++ never ships in the final image) |
| Model weights | `models fetch` at build, sha256-pinned (`models/manifest.toml`) |
| `ffmpeg` / `ffprobe` | Debian trixie apt (libx265, 10-bit main10 included) |
| `ca-certificates` | outbound TLS for gofile upload/fetch and the DoH ladder |
| Runtime user | non-root `app` (uid/gid 1000), `HOME=/home/app` |
| Entrypoint | `["/app/.venv/bin/anime-pipeline"]`, `WORKDIR /app` |
| `config.toml` | **not baked** — bind-mounted at runtime (single source of truth on the host) |

**Outside the image (bind mounts only):**

| Host path | Container path | Purpose |
| :-- | :-- | :-- |
| `./input` | `/app/input` | source video (the CLI resolves everything from CWD) |
| `./output` | `/app/output` | finished `.mkv` files |
| `./config.toml` | `/app/config.toml` | the one configuration file (no env vars, by owner ruling) |
| `./cache/trt` *(optional)* | `/home/app/.cache/vsrife` | persistent TensorRT engine cache — see [below](#-tensorrt-engine-cache) |

> **Base image deviation note.** The house default is a distroless final stage.
> This image deliberately uses `python:3.14-slim-trixie` as a non-root runtime:
> the stack is venv-self-contained (torch pulls all CUDA userspace from wheels;
> TensorRT runtime ships as `tensorrt-cu13-libs`), but the delivery of
> ffmpeg-with-libx265 via apt, the compiled misc plugin's glibc linkage, and
> torch/TRT dynamic-library edge cases make strict distroless a fragile
> static-ffmpeg + library-surgery exercise. Slim + apt + non-root is the
> documented, pragmatic deviation; a distroless variant remains a possible
> future optimization wave.

No `HEALTHCHECK` by design: this is a run-once CLI tool — the process exit code
**is** the health signal, and a healthcheck could only re-run the binary. No
`environment:` keys by design: the pipeline reads everything from `config.toml`;
GPU injection happens via runtime flags, not env config.

## 🚀 Running the pipeline

`docker compose run` allocates a TTY (Russian progress bars live); `--rm` keeps
no stray containers. This is a CLI tool — **never `docker compose up -d`**.

```bash
# Interpolation-only profile with gofile upload after success
docker compose run --rm anime-pipeline --profile interp-only --upload

# Arbitrary stage subset with explicit inputs (names with extensions, not paths)
docker compose run --rm anime-pipeline --stages interpolate,encode -i episode01.mkv -i episode02.mkv

# CLI help inside the image
docker compose run --rm anime-pipeline --help
```

The same shape works identically on Linux and Windows (WSL2). Bare-docker
equivalent (Linux host):

```bash
docker run --rm -it --gpus all \
  -v ./input:/app/input \
  -v ./output:/app/output \
  -v ./config.toml:/app/config.toml \
  anime-upscale:local --profile interp-only
```

`gpus: all` in `compose.yaml` (Compose Specification service attribute) allocates
**every host GPU** with zero device pinning — the pipeline's own hardware
autodetect decides which GPU to use. On WSL2 hosts `--gpus all` is in any case
the only supported form (no per-GPU filtering through GPU-PV), which is exactly
what this design uses.

## 🖥️ Host setup

The container needs only the **NVIDIA driver** on the host — CUDA userspace
comes from wheels inside the venv. Docker gets GPU access via the
[NVIDIA Container Toolkit](https://docs.nvidia.com/data-center/cloud-native/container-toolkit/latest/install-guide.html).

### Linux (native)

```bash
# 1. NVIDIA driver (distribution package manager or .run installer)
# 2. NVIDIA Container Toolkit (repo setup per NVIDIA install guide)
sudo apt-get install -y nvidia-container-toolkit
# 3. Wire the runtime into Docker and restart the daemon
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
# 4. Verify
docker run --rm --gpus all ubuntu nvidia-smi -L
```

Modern setups (Docker ≥ 27, toolkit ≥ 1.18) additionally get automatic CDI
spec generation (`nvidia-cdi-refresh` regenerates `/var/run/cdi/nvidia.yaml` on
driver/toolkit install and reboot) — an optional modernization, not required:
`nvidia-ctk runtime configure` remains the canonical path.

### Windows Server (via WSL2)

Docker Desktop is **not supported** on Windows Server, and native Docker Engine
on Windows runs Windows containers only (no Linux containers, no NVIDIA GPU for
Linux workloads). The sanctioned path for Linux containers + GPU on Windows
Server **2022 or newer** is WSL2 + Docker Engine + the toolkit **inside the
distro**:

```powershell
# 1. From elevated PowerShell: confirm Server 2022+ (WSL2 needs KB5014678+, June 2022)
Get-ComputerInfo | Select-Object WindowsProductName, OsVersion

# 2. Install WSL2 and a distro (Ubuntu)
wsl --install
# reboot, then enter the distro; keep it current:
wsl --update
```

```bash
# 3. INSIDE the distro: Docker Engine (docker-ce) + NVIDIA Container Toolkit
#    (repo setup per NVIDIA/Docker install guides)
sudo apt-get install -y docker-ce docker-ce-cli containerd.io \
    docker-buildx-plugin docker-compose-plugin nvidia-container-toolkit

# 4. Wire the NVIDIA runtime into the in-distro Docker and restart it
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
# 5. Verify from inside the distro
docker run --rm --gpus all ubuntu nvidia-smi -L
```

> ⚠️ **Never install a Linux NVIDIA driver inside WSL.** The Windows host driver
> is stubbed into WSL as `libcuda.so`; installing `cuda-drivers`/`nvidia-open`
> inside the distro overwrites the projection and breaks GPU access. The
> **Windows NVIDIA display driver on the host is the only driver you install**.
> Also expect `nvidia-smi` to expose a limited feature set under WSL — that is
> documented NVIDIA behavior, not a setup defect.

Keep the project (repo, `input/`, `output/`, `cache/`) inside the WSL2 ext4
filesystem rather than on `C:\`: bind mounts crossing WSL2↔NTFS go over 9P and
pay a real tax on many-small-file operations (large sequential reads are only
moderately affected — acceptable for this pipeline, but ext4 is the
Docker-best-practice layout).

## 🩺 Doctor checklist

First-run verification, in order — every step is loud about its failure:

```bash
# 1. The CLI itself runs inside the image
docker compose run --rm anime-pipeline --help

# 2. Baked weights match their sha256 pins
docker compose run --rm anime-pipeline models verify

# 3. GPU reaches the container (nvidia-smi is injected by the toolkit,
#    not baked into the image — hence the entrypoint override)
docker compose run --rm --entrypoint nvidia-smi anime-pipeline -L
```

If step 3 prints **nothing** (or "could not select device driver"), the toolkit
side is misconfigured: re-run `sudo nvidia-ctk runtime configure --runtime=docker`
and `sudo systemctl restart docker` on the host (inside the WSL2 distro on
Windows), then re-check. An empty `-L` listing is never a silent-pass situation.

## 🏗️ Building the image

```bash
docker compose build            # or: docker build -t anime-upscale:local .
```

After a build (or after any Docker-affecting change), the repo smoke script
exercises the whole contract end to end — build, compose config, CLI, sha256
verification, GPU doctor, and a synthetic 5-second clip through the mounted
`input/`/`output/` plus a container-layer hygiene check:

```bash
scripts/docker-smoke.sh    # needs a toolkit-enabled GPU host for the full run;
                           # SMOKE_SKIP_GPU=1 skips the GPU-dependent steps explicitly
```

**Build on the target device** — owner decision (2026-11-06): build where the
image will run; if that network's TLS interception breaks the build, use the
`docker save` fallback below. The image is
GPU-architecture-agnostic (weights and wheels are universal; TensorRT engines
are built at first run and cached), so there is no cross-machine image constraint.

⚠️ **Honest caveat — TLS interception can break `docker build`.** The pipeline's
adaptive-TLS logic (DoH ladder, certificate-verify-failure retry) lives in the
**application** and does not exist during an image build: the build stage does
plain `apt`, `uv`, `curl`, and `models fetch` downloads over the build host's
network. On a network with TLS interception (self-signed chains on pypi.org /
download.pytorch.org / Debian mirrors), the build fails loudly. The fallback
recipe is to **build elsewhere and transfer**:

```bash
# On the build machine
docker save anime-upscale:local | zstd > anime-upscale-image.tar.zst
# Transfer (scp/USB), then on the target machine
zstd -d anime-upscale-image.tar.zst | docker load
```

(or push/pull through a registry instead of `docker save`).

**Rebuild hygiene:** source edits only invalidate the light layers; the heavy
torch cu130 dependency layer survives in the build cache thanks to the
manifests-only deps stage. The uv download cache persists across builds via the
`type=cache` mount.

## 💾 TensorRT engine cache

TensorRT engines are GPU-architecture- and TRT-version-specific and take
minutes to build on a 5080-class GPU. The pipeline stores them in
`~/.cache/vsrife` (config: `[hardware] trt_cache_dir`), and the compose file
bind-mounts `./cache/trt` there so **engines survive container recreation**.
Pre-create the directory on the host before the first run:

```bash
mkdir -p cache/trt     # owned by your user (uid 1000) — see ownership note below
```

A Docker auto-created bind directory is root-owned, and the container runs as
uid 1000 — engine writes would fail with a loud permission error. Engines are
invalidated by GPU or TensorRT version changes; the directory is gitignored.

## 📉 Degradation without a GPU

No separate CPU mode exists — the pipeline's standing contract applies unchanged:

- zero GPU + any NN stage (`upscale`/`interpolate`) → **loud Russian error**
  naming the cause (in the container case, the toolkit hint above is the first
  suspect);
- encode-only subsets (`--stages encode`) still work CPU-only — libx265 encode
  never touched the GPU.

The image itself does not require a GPU to build or to print `--help`/run
`models verify`; only NN stages demand one at runtime.

## 🔐 User mapping and ownership

The container runs as non-root uid/gid **1000** (`app`) — matching the typical
first user on Linux hosts and the default WSL2 user, so files written to
`./output` land with your own ownership. The baked `.venv` and `models/` stay
root-owned on purpose: the app needs only read/execute there; every runtime
write goes to bind mounts or `$HOME`.

If your host user is **not** uid 1000 (files in `output/` end up owned by
`1000:1000`), either align ownership on the host
(`sudo chown -R "$(id -u):$(id -g)" output/`) or override the user for a run:
`docker compose run --rm --user "$(id -u):$(id -g)" anime-pipeline …`
(the `HOME` bind mount for the TRT cache should then match that uid too).
Pre-create bind targets (`mkdir -p cache/trt`) as your user so Docker never
auto-creates them root-owned.

## 📏 Size budget

Expect roughly **11–12 GB** uncompressed: the host venv measured ~9.9 GB
(torch cu130 wheels + CUDA userspace + torch-tensorrt/TRT-11.1 + VapourSynth
stack), plus apt ffmpeg (~0.5 GB) and the baked weights (~28 MB). Measured on
the reference build: ~11.0 GB by per-layer history (venv layer 10.4 GB +
ffmpeg 0.46 GB + base 0.13 GB + models) — right in budget. Note that
`docker images` under Docker 29 (containerd storage) may print a larger
aggregate figure for the same image; the layer history above is the honest
accounting, and the transferable artifact (`docker save`) follows it. Disk
space and transfer time are first-class expectations of this image — use the
`docker save`/registry recipe above for moving it between machines.
