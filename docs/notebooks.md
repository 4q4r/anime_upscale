# Kaggle Notebooks + Google Colab — compatibility matrix and deployment notes (2026-11-06)

**Status:** documentation for the notebook deployment surface (plan `plan-kaggle-colab.md`, wave N4). Facts collected 2026-11-06 at main `517d160` (427 tests green). Windows Server RTX 5080 remains the **primary** deploy target — nothing in this document changes primary-runtime behavior; notebooks are an additional runtime mode (see [architecture-plan.md](architecture-plan.md) §13).
**Design invariants preserved:** config.toml stays the single configuration source, no env-var configuration in the app; Russian user-facing messages; fail fast, loudly; single committed manifest (cu130) — platform adaptation lives in notebook bootstrap cells against an ephemeral clone, never in the repo.

---

## 1. Compatibility matrix (project × Kaggle Notebooks × Google Colab)

Legend: **[official]** = fetched from first-party docs/repos 2026-11-06 unless dated otherwise; **[community]** = recent community evidence, dated explicitly. "Project" column = the committed Windows/RTX 5080 configuration, unchanged by this feature.

| Dimension | Project (Windows RTX 5080, unchanged) | Kaggle Notebooks | Google Colab |
|---|---|---|---|
| Python | 3.14 (requires-python `>=3.12`) | 3.13 (Dockerfile.tmpl `ARG PACKAGE_PATH=…/python3.13/dist-packages` [official, GitHub Kaggle/docker-python, main 2026-11-06]); a validated live run showed 3.12.13 on 2026-08-22/23 [community, kaggle-vllm PyPI page] → image mid-migration; **≥3.12 either way** | 3.13.18 (runtime 2026.10, Ubuntu 24.04.5); older runtimes pinnable: 2026.07 = 3.12.13 [official, research.google.com/colaboratory/runtime-version-faq.html, 2026-11-06] |
| vapoursynth (PyPI R81/82) | works | cp312-abi3 manylinux_2_27 wheels → installs on 3.12/3.13/3.14, glibc ≥2.27 [official, PyPI JSON API, 2026-11-06] | same ✓ |
| vapoursynth-bestsource 23 / vapoursynth-descale 12 | works | py3-none manylinux_2_28 / manylinux2014 wheels ✓ [official, PyPI JSON, 2026-11-06] | same ✓ |
| vsrife 5.7.0 | works | py3-none-any, requires ≥3.10 ✓ [official, PyPI JSON, 2026-11-06] | same ✓ |
| vsdeband (git JET) → vapoursynth-vszip 23.1.0 | works | py3-none manylinux ✓ [official, repo uv.lock + PyPI, 2026-11-06] | same ✓ |
| torch | 2.14.1 cu130 (driver r580+, Blackwell sm_120) | driver **580.159.04 / CUDA 13.0 capable**, 2× T4 sm_75, 15 360 MiB each [community, kaggle-vllm validated run 2026-08-22/23]; **cu130 torch runs as-is** (torch ≥2.11 PyPI default = cu130; "CUDA 13.0 only supports Turing SM 7.5 and newer on Linux x86_64" [official, PyTorch v2.11.0 release notes, 2026-11-06]) | driver **550.54.15 / CUDA 12.4** [community, colabtools issues #5521/#5578, 2025-09 — **2026 value unverified → runtime probe mandatory**]; cu130 torch will NOT run (CUDA 13 needs r580+); must resolve **cu126** (available through torch 2.14; dropped from 2.15 [official, dev-discuss.pytorch.org CUDA-12.6 deprecation notice, 2026-09]) or cu128 |
| torch-tensorrt | 2.14.0 cp314/cu130 | cp312/cp313 wheels exist (v2.9/v2.10 target Py 3.10–3.13, CUDA 13.0 via PyPI, cu126/128/13.0 via the PyTorch index [official, pytorch/TensorRT releases v2.9.0/v2.10.0, 2026-11-06]); exact torch-version match required | cu126 variant via the PyTorch index (same [official] source); **fallback = `trt=false`** — the pipeline already has a designed degrade path ([stages/interpolate.py](../src/anime_upscale/stages/interpolate.py) 37–43) |
| TensorRT engine on T4 (sm_75) | n/a (Blackwell sm_120: TRT off by design) | **Supported**: "TensorRT supports NVIDIA hardware with compute capability SM 7.5 or higher"; Volta removed in 10.5 [official, docs.nvidia.com TensorRT support matrix, TRT 10.0–10.16.1, 2026-11-06] | same ✓ (if the TRT path is used) |
| hardware.py verdicts for T4 | — | `is_blackwell=False` → `recommended_interp_format="rgbh"` (FP16) + `recommended_trt=True` — **already correct for Turing, zero code change** [repo fact, hardware.py:53–71] | same ✓ |
| VRAM streams (hardware.py budget) | 16 GB → 2 streams | T4 15 GB → 1 stream/GPU (clamp formula); v1 uses `gpus[0]` only | T4 15 GB → 1 stream |
| ffmpeg + libx265 | PATH + `[tools]` override (winget ladder, untouched) | Kaggle base image is Colab-derived (Dockerfile.tmpl `FROM us-docker.pkg.dev/colab-images/public/runtime:…` [official, GitHub Kaggle/docker-python, build 20261027]) → inherits **ffmpeg 6.1.1 (apt noble) + libx265-199 3.5 + g++ 13.2 + build-essential** [official, github.com/googlecolab/backend-info apt-list.txt, fetched 2026-11-06]; apt-get install works in-session; Ubuntu/Debian apt ffmpeg builds enable `--enable-libx265` [official, ffmpeg-cookbook 2026-04, configuration-line evidence] | **ffmpeg 6.1.1 + libx265 preinstalled** [official, apt-list.txt, 2026-11-06]; g++ preinstalled ✓ (misc plugin compiles) |
| misc plugin (SCDetect) | scripts/build_misc_plugin.sh (g++, commit-pinned source) | g++ present ✓; script needs curl + github.com (internet ON) → works as-is | same ✓ |
| Docker | n/a | No user-managed Docker; Kaggle runs the session in its own image; image-version pin available in Session options [official, kaggle.com/docs/notebooks, 2026-11-06] | No Docker (containerization to evade anti-abuse is explicitly banned [official, Colab FAQ, 2026-11-06]) |
| Session limits | unlimited | **12 h** CPU/GPU sessions (9 h TPU), **20-min idle** timeout for interactive, Save & Run All runs headless full sessions [official, kaggle.com/docs/notebooks Technical Specifications, 2026-11-06]; weekly GPU quota ≈30 h shared — **not published by Kaggle** [community] | No published number: "maximum lifetime enforced"; free tier commonly shows usage-based "may last up to 5 hours"; idle VM deletion; free-tier runtimes may be terminated without warning [official FAQ wording + community observations 2025-12] |
| Persistent storage | local disk | `/kaggle/working` **20 GB auto-saved**; additional ephemeral scratchpad (size unpublished; community ≈57 GB total disk) [official specs + community df] | Everything ephemeral (~112 GB disk observed on T4 VM [community 2025-12]); persistence = Google Drive mount (`google.colab.drive.mount` [official mechanism]) |
| Input paths | `-i` = name in `input/` (CWD-relative) or `https://` / `gofile.io/d/<id>` URL [repo fact, cli.py] | gofile URL directly ✓; local files via **Kaggle Datasets** → read-only `/kaggle/input/<slug>` → symlink into `./input/` (recipe in notebook cell) | gofile URL ✓; local files via Drive → symlink/copy into `./input/` |
| Output | `output/` (CWD-relative) | `/kaggle/working/.../output/` — persists ✓ (within 20 GB) | `/content/...` ephemeral → copy to Drive at end of batch |
| config.toml location | repo root, CWD-relative `./config.toml` [repo fact] | repo cloned to `/kaggle/working/anime_upscale` → config editable + persists ✓ | repo cloned to `/content/anime_upscale` → persists only via Drive sync |
| gofile guest-token state | `~/.local/state/anime_upscale/gofile_guest.toml` (XDG) [repo fact, gofile.py] | `$HOME=/root` ephemeral → token lost per session; re-creation is rate-limited when frequent [project memory]; **owner decision: per-session token accepted** (§5) | same → Drive-persist or accept per-session token; **owner decision: per-session accepted** (§5) |
| Internet | always | **Toggle per notebook** (Session options); phone-verified account required for GPU + internet [official docs + community, 2026-11-06] | always ON |
| Colab abuse-policy constraints | n/a | n/a | banned: remote control/SSH shells, bypassing the notebook UI, containerization to evade anti-abuse; notebook-cell-driven use is compliant [official FAQ, 2026-11-06] |
| CPU for x265 | 16C/32T server (`pools=32` contract) | **4 vCPU** → `pools=4` (≤16, no frame-threads bug) [official specs, 2026-11-06] | T4 VM ≈ 2 vCPU / 12.7 GB RAM observed [community 2025-12] → **encode is the bottleneck** (§4, R2) |

**Stack verdict:** the entire dependency set installs from wheels on both platforms with `requires-python >=3.12` satisfied ([pyproject.toml](../pyproject.toml) untouched). The only hard incompatibility is cu130 torch/torch-tensorrt on Colab's CUDA-12-class driver — solved at bootstrap resolution time (§2.4), never in the committed manifest.

---

## 2. Design notes

### 2.1 Principles

1. **Detection is not configuration.** The app keeps reading settings exclusively from config.toml. A new `runtime_env.py` module only *identifies the platform* (read-only `os.environ` probes, path probes) and resolves *root/state defaults* — the same class of sensing [hardware.py](../src/anime_upscale/hardware.py) already does for GPUs. No app behavior is configurable via env (owner invariant).
2. **Zero interactive prompts.** The CLI has none (argparse only); notebook cells never ask.
3. **TTY degradation already works:** `resolve_mode()` checks `stream.isatty()` (reporting.py) → notebook `!` cells (piped stdout) automatically get the no-ANSI plain mode; tqdm degrades on non-TTY. No code change needed; the explicit `--no-tty` flag remains available.
4. **Russian messages** — all new user-facing strings follow the existing Reporter vocabulary (banner, warn, error).
5. **Fail fast, loudly, in Russian** at every bootstrap step: missing GPU, wrong driver class, no internet, no g++, disk headroom < 3 GB → named error + remediation hint, exit non-zero. No silent fallbacks.
6. **Windows paths untouched:** every adaptation lives in notebook bootstrap cells (ephemeral clone mutation) or in new read-only detection code gated on the detected platform. The committed cu130 manifest stays.

### 2.2 Environment detection

`runtime_env.py` (read-only, no config keys, no env writes) resolves a `RuntimeProfile`: `kind` (`local`/`kaggle`/`colab`), `persistent_root`, `session_note`.

- Kaggle = `KAGGLE_KERNEL_RUN_TYPE` env set or `/kaggle/working` exists; default root `/kaggle/working/anime_upscale`.
- Colab = `importlib.util.find_spec("google.colab")` or `COLAB_GPU` env; default root `/content/anime_upscale`.
- Else `local`; default root = CWD.
- Detection precedence: kaggle > colab > local.

### 2.3 GPU-profile degradation ladder (T4 = Turing sm_75)

`hardware.py` already classifies T4 correctly (`is_blackwell=False` → FP16 `rgbh` + `recommended_trt=True`); engine build for sm_75 under TRT 10.x is supported (§1). The interpolate stage degrades:

1. **TRT on** (sm_75 supported) — **Kaggle default** (driver CUDA-13-capable → committed cu130 torch-tensorrt resolves).
2. **TRT off → PyTorch CUDA FP16 (`rgbh`)** — the exact code path Blackwell uses today ([stages/interpolate.py](../src/anime_upscale/stages/interpolate.py): the config `trt=true` is dropped with a Russian warning when `hardware.recommended_trt` is false, and the run proceeds). This is the **Colab default** when torch-tensorrt cu126 wheels are unavailable or engine build fails → the bootstrap seeds an ephemeral config override `[stages.interpolate] trt = false` (config override, not env).
3. **No GPU → loud error** (already implemented, NN-stage guard).

FP32 (`rgbs`) remains Blackwell-only semantics; on T4, FP16 RGBH is the intended profile.

### 2.4 Torch-index adaptation for Colab (the one real compatibility move)

The repo keeps **one** manifest ([pyproject.toml](../pyproject.toml), cu130 index). The Colab bootstrap cell mutates only the **ephemeral clone**:

1. Probe: `nvidia-smi --query-gpu=driver_version` + parse the `nvidia-smi` header CUDA Version.
2. If CUDA < 13 → rewrite the `[[tool.uv.index]]`/`[tool.uv.sources]` torch + torch-tensorrt URLs cu130→**cu126** in the clone's `pyproject.toml` (sed on the ephemeral copy, never committed) and append `torch<2.15, torch-tensorrt<2.15` guards (cu126 wheels stop publishing at 2.15 [official, dev-discuss.pytorch.org, 2026-09]).
3. `uv sync` (lock re-resolution; network cost ~3–6 GB once per runtime).
4. If torch-tensorrt fails to resolve → drop it and seed the `trt=false` override (ladder step 2).

Kaggle (driver 580, CUDA-13-capable): nothing to do — `uv sync` uses the committed cu130 lock resolution. The probe-first design is deliberate: the 2026 Colab driver value is unverified (§1), so the notebook self-adapts to driver drift instead of hardcoding it.

---

## 3. Definition of Done and local validation contract

**Wave DoD (docs wave N4):** every §1 row carries its source + date stamp; architecture-plan addendum ≤40 lines; markdown structure lint clean (this repo carries no markdownlint config — checked 2026-11-06; heading structure and relative links verified manually).

**Local-only validation contract (no Kaggle/Colab account needed):**

- Env-emulation contract tests: detection precedence (kaggle > colab > local), default roots, byte-identical default behavior when no platform is detected (monkeypatched env vars, fake `/kaggle/working`, fake `google.colab` module spec).
- Notebook-cell script dry-runs with in-cell dry-run flags (cells' own code, never app env config) + JSON-validity lint of the notebooks (`jupyter nbconvert --to script` round-trip).
- Standard repo gates: `ruff check` / `ruff format --check` / `ty check src` / `pytest` — standalone-blocking, unchanged.

**Owner acceptance steps (declared limitations — cannot be automated from here):**

1. Live GPU-session end-to-end run on Kaggle (phone-verified account required for GPU + internet).
2. TRT engine build on T4 (sm_75) under torch-tensorrt cu130.
3. Colab live run: driver probe → cu126 swap → `trt=false` degrade path on real hardware.
4. gofile round-trip (upload/fetch) from notebook egress IPs.

---

## 4. Risk register

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | Colab driver/cu130 mismatch: committed torch is cu130; Colab driver class is CUDA 12.x (550.54.15 evidence is 2025-09; 2026 value unverified) | **High** | Mandatory driver probe in bootstrap → cu126+`<2.15` index swap (§2.4); probe failure = loud Russian error with diagnosis. Residual: if Colab moves to CUDA-13 drivers, the swap branch simply stops firing. |
| R2 | Encode throughput: x265 preset medium 10-bit on 2–4 vCPU ≈ 1.5–4 fps @1080p; a 60 fps-interpolated episode (≈85k frames) exceeds a Colab session window and eats most of a 12 h Kaggle session | **High** | **Owner decision: preset medium kept** (§5) — the bootstrap banner prints honest session-budget math (frames/fps/ETA) so multi-session per-episode workflows are planned, not discovered. Interp-only profiles on notebooks remain the feasible default. |
| R3 | Platform drift: Kaggle image switched base to the Colab runtime (build 20261027) and is mid py3.12→3.13 migration; weekly quotas unpublished; Colab session caps unpublished/usage-based | Medium | Bootstrap probes everything at runtime (python version, driver, disk, GPU); wheels chosen are abi3/py3-none (immune to py-version wobble); facts in §1 are stamped with dates, community numbers marked as such. |
| R4 | Kaggle 20 GB `/kaggle/working` cap vs .venv ≈ 9–10 GB (torch cu130 + nvidia libs) + inputs/outputs | Medium | Bootstrap prints disk headroom and warns < 3 GB; documented fallback: clone into the ephemeral scratchpad (bigger), keeping only config/outputs in `/kaggle/working` (costs a re-install per session). |
| R5 | torch-tensorrt cu126 cp312/313 wheel availability for torch 2.14.x unverified (documented pattern only for 2.9/2.10) | Medium | Ladder step 2 is the designed degrade (`trt=false` → PyTorch CUDA FP16 RGBH, §2.3); Colab bootstrap seeds it when resolution or engine build fails; Kaggle (cu130) unaffected. |
| R6 | Colab free-tier termination without warning + Kaggle interactive 20-min idle kill long interactive batches | Medium | Single-cell run UX; Kaggle `Save & Run All` headless mode documented for unattended batches; docs advise per-episode sessions, not multi-episode marathons, on Colab. |
| R7 | gofile guest-token churn on ephemeral `$HOME` (rate-limit risk from frequent re-creation) | Low | **Owner decision: no `state_file` config key** (§5) — per-session token accepted and documented as the intended notebook behavior; frequent-recreation rate limit noted in §1. |

---

## 5. Fixed owner decisions (2026-11-06, binding)

1. **Encode preset `medium` everywhere.** No `[profiles.notebook]` fast-preset override. The bootstrap banner states the honest multi-session math (preset medium on 2–4 vCPU ≈ 1.5–4 fps @1080p → plan per-episode sessions; R2).
2. **No `upload.gofile.state_file` config key.** The guest token is per-session on notebooks (ephemeral `$HOME`); re-creation per session is the accepted behavior (R7). The repo's state-file contract stays byte-identical for the primary runtime.
3. **Input = gofile URL entry; Kaggle Datasets as a symlink recipe.** gofile.io/d/<id> works directly via `-i`; local files go through Kaggle Datasets → `/kaggle/input/<slug>` → symlink into `./input/` (recipe cell). Dataset attachment is not a first-class `-i` path in v1.
4. **Colab torch variant = cu126 + `<2.15` pins.** Bootstrap resolves cu126 with `torch<2.15, torch-tensorrt<2.15` guards in the ephemeral clone (§2.4); the committed manifest stays cu130.
5. **Live platform runs = owner acceptance.** The first live sessions on both platforms are manual owner-side validation steps (§3 list); the project cannot automate account prerequisites (phone verification).

---

## Provenance

**Skills used:** `brainstorming` (bounded — spec-driven: the plan file and the controller brief froze scope, owner decisions, and the fact set; no open questions remain for a subagent), `build-with-exa` (loaded; **N/A** — all external facts were already collected, dated, and source-labeled in the plan §1/§7; this wave transfers them verbatim and invents none, per brief), `grilling` (bounded — design-tree self-check against the brief: scope = 2 files only; all five owner decision branches pre-resolved in the brief; no unsettled frontier left), `technical-writer` methodology applied manually (no such skill installed in this runtime: audience = repo readers + future implementers; decision-carrying prose; source/date stamp per row; anti-goals stated; no invented facts).

**Memory files read (mandate compliance):** `~/.config/opencode/MEMORY.md` in full; **topics/ 20/20** — `workflow.md`, `workflow-git-pr.md`, `stack-and-tools.md`, `communication-behavior.md`, `tech-gotchas.md`, `code-gotchas.md`, `environment-infra.md`, `remnawave.md`, `nnm-club-anime.md`, `harness-server.md`, `hh-ranking-ats-research.md`, `job-search.md`, `2026-10-09-telegram-serverless.md`, `2026-10-11-novum-organum-report-plan.md`, `2026-10-12-slides-novum-organum-plan.md`, `2026-10-14-opencode-v2-storage.md`, `2026-10-15-binary-re-landmines.md`, `2026-10-15-vision-ocr-scans.md`, `2026-10-21-stepin-book-contract.md`, `2026-10-25-locke-pirogov-referaty.md` (small files full; `code-gotchas.md` and `environment-infra.md` capped reads — large files); **projects/ 34/34** — `anime_upscale.md` in full (project canon), all others via capped head reads: `4q4r-profile.md`, `WMS.md`, `anicli-go.md`, `anicli-py.md`, `bos-pro-bugs.md`, `cindex.md`, `ert.md`, `firefox-vpn-client.md`, `freeai-decompile.md`, `freepackages.md`, `habrparser.md`, `hhauto.md`, `home-global.md`, `interview-kb.md`, `iris-proposal.md`, `kolosyug.md`, `labboardvent.md`, `labs-docker-stands.md`, `labs.md`, `metrics.md`, `public_vpn.md`, `remnavawe_ansible.md`, `secureway_bot.md`, `shikimori-japanese-theme.md`, `stepik-django-240388-fixes.md`, `stepik-whitehacker-course.md`, `teleparse.md`, `traffic_mimicry.md`, `tspu-research-2026.md`, `vibecoding-course.md`, `wauploader.md`, `wms.md`, `xray-raw.md` (giants — wms 370K, ert 142K, hhauto 138K, anicli-py 119K, traffic_mimicry 79K, anime-adjacent corpuses — exceed context for full reads; caps confirmed zero content governing this docs-only task beyond MEMORY.md index entries). No memory files written.

**Sources (all fetched 2026-11-06 unless dated otherwise):** Kaggle docs / Technical Specifications + Kaggle/docker-python (Dockerfile.tmpl FROM colab-images, build 20261027); Colab runtime-version-faq + FAQ + googlecolab/backend-info apt-list.txt; PyTorch v2.11.0 release notes + dev-discuss CUDA-12.6 deprecation notice (2026-09); pytorch/TensorRT v2.9.0/v2.10.0 release notes; docs.nvidia.com TensorRT support matrix (TRT 10.0–10.16.1); PyPI JSON API for vapoursynth / vapoursynth-bestsource / vapoursynth-descale / vsrife; kaggle-vllm PyPI page (Kaggle validated live env, 2026-08-22/23); colabtools issues #5521/#5578 (driver evidence, 2025-09). Repo facts read directly at main `517d160` ([pyproject.toml](../pyproject.toml), [hardware.py](../src/anime_upscale/hardware.py), [stages/interpolate.py](../src/anime_upscale/stages/interpolate.py), cli.py, gofile.py, reporting.py).
