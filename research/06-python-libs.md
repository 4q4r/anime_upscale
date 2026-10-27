# 06 — Python ecosystem & Python 3.15 fit for the VS-graph rewrite

Research date: 2026-10-27. Subagent: python-ecosystem researcher (controller wave for ~/projects/anime_upscale).
Approved architecture (fixed): VapourSynth R81 embedded graph + vsrife 5.7.0 (RIFE 4.25, `trt=True` Torch-TensorRT) + spandrel upscale in `std.ModifyFrame` + ffmpeg subprocess (libx265 per research/05).

Verification method: Context7 first (owner mandate) → PyPI JSON API + download.pytorch.org wheel indexes (primary sources) → websearch (python.org / PEPs). Exa returned **402 credits-exhausted** on first probe (matches memory 2026-10-22) → fallback chain used per standing rule. Labels: **[V]** = verified today against the named URL/API; **[OP]** = operator/owner prior fact, not re-verified; **[UNVERIFIED]** = stated as unknown.

---

## 1. Legacy script dependency inventory vs current majors

Import audit re-verified by direct read of both scripts today (full files, not grep): `interpolate_only.py` imports os, sys, subprocess, argparse, numpy, torch, torch.nn.functional, tqdm; `process_anime.py` adds threading, queue, spandrel. **Neither script imports moviepy.**

| Legacy import | Current stable (2026-10-27) | Breaking since ~2023–24 era | Fits new stack? |
|---|---|---|---|
| numpy | **2.5.4** [V: pypi.org/pypi/numpy/json] | 1.x→2.0 (2024-06) ABI+API break (alias removals, copy-semantics warnings), long settled by 2.5.x; `requires_python >=3.12`; wheels cp312–**cp315, cp314t, cp315t** | Yes — as VS/spandrel/torch transitive + direct use; scripts' surface (`frombuffer/reshape/clip/mean`) unchanged across 2.0 |
| torch | **2.14.1** on PyPI (2026-09-30) [V]; 2.14.x docs on context7 [V] | No API break for the scripts' surface: `torch.backends.cudnn.benchmark` / `allow_tf32`, `inference_mode`, `F.pad` all current; the pain points are architectural (rawvideo pipe loop, sys.path RIFE hacks, NVENC), replaced wholesale | Yes — cu130 lane required for RTX 5080 (Blackwell, CUDA 12.8+) [OP + V wheel index] |
| spandrel | **0.4.2** (2026-02-21) [V] | None affecting script: `ModelLoader(device=…).load_from_file` is still the current documented API (context7 /chainner-org/spandrel, 830 snippets); py3-none-any wheel, `requires_python >=3.10`, deps: torch, torchvision, safetensors, numpy, einops, typing_extensions [V] | Yes — unchanged role (upscale model loader) |
| tqdm | **4.70.1** (2026-09-11), no 5.x exists [V] | None relevant | Mostly superseded — VS `frames()` iteration replaces per-frame progress; keep only for CLI-level job progress if wanted |
| threading, queue (stdlib) | — | — | Dropped — VS `frames(prefetch=N)` replaces the producer-thread + queue pattern natively |
| os/sys/subprocess/argparse | stdlib | — | Kept (CLI/config/ffmpeg plumbing) |
| **moviepy** | **2.2.1** (2025-05-21), major 2.x [V] | 1.x→2.x is a hard break [V: context7 /zulko/moviepy → docs/getting_started/updating_to_v2.md]: `.set_*` → `.with_*` (out-of-place copies), `resize/crop/rotate` → `resized/cropped/rotated`, `clip.fx` → `with_effects()`, TextClip requires font file path + new arg names | **NOT NEEDED — DROP.** Not imported by either script (the owner's "one major behind" moviepy lives only in the old environment, not in the code). Under the new stack there is nothing for it to do: muxing/subs/attachments = ffmpeg two-input map (research/05), cutting/timeline = VS graph (Trim), encoding = libx265 subprocess. moviepy is itself an ffmpeg wrapper with none of the 10-bit/x265 flag control research/05 demands, and v2.2.1 has been idle since 2025-05 |

**Bottom line §1:** zero legacy imports block the rewrite; all four real deps are current-major and API-compatible at the usage surface. moviepy is removed by decision, not by migration effort. The legacy problems were never the library list — they were pre-4.22 RIFE weights, x2-only midpoint interpolation, rawvideo double-piping, and NVENC — all of which the VS-graph architecture eliminates.

## 2. Context7 verification report (mandated order)

| Library | context7 ID (resolve result) | Queried concepts | What context7 established | Gaps → filled from |
|---|---|---|---|---|
| VapourSynth | `/vapoursynth/vapoursynth` (1646 snippets, High rep) [V] | (a) Python version requirements; (b) frame delivery / GIL / prefetch | (a) **Python 3.12+**, "support for Python 3.12 and all later versions, including 3.13 and 3.14" (installation.rst); pip install + `vapoursynth config` flow. (b) `RawNode.get_frame` calls core `getFrame` **with GIL released** (`with nogil:` in vapoursynth.pyx); `frames()` "renders multiple frames concurrently", `prefetch` = concurrent frame count ("debugging only, should never need to be changed"), `backlog` caps memory; `get_frame_async` renders in another thread (R58+); Vulkan bindings release GIL around device creation | No 3.15 statement in docs (predates) → PyPI JSON today: R81 ships **cp312-abi3** wheels (install on 3.12–3.15 GIL builds) + **cp315-cp315t** free-threaded wheels incl. manylinux_2_27_x86_64 [V]. No cp314t wheel exists [V] |
| spandrel | `/chainner-org/spandrel` (830 snippets) [V] | (a) model loading API currency; (b) supported torch versions / CUDA 12.8+ | (a) `ModelLoader(device="cuda:0").load_from_file(path)` → `ImageModelDescriptor` with `.scale/.eval()/__call__` under `torch.inference_mode`; auto padding via size_requirements; errors `UnsupportedModelError`. API identical to the legacy script's usage — nothing to migrate. (b) context7 only carried repo-level metadata (version 0.4.2, "Python >=3.8,<3.12" — **stale/incorrect for the PyPI package**) | PyPI JSON: actual `requires_python >=3.10`, pure-python wheel, deps torch/torchvision/safetensors/numpy/einops/typing_extensions; no torch version cap [V]. CUDA support comes from whatever torch wheel is installed → RTX 5080 needs cu128+/cu130 torch [OP+V] |
| PyTorch | `/pytorch/pytorch` + `/websites/pytorch` (docs up to **2.14.x**) [V] | supported Python versions, cu130/Blackwell wheels | Docs carry a standing warning that `torch.compile` "may not support recently released major versions of Python" (generic 3.xx guard) — not used in our stack. No wheel matrix in docs | download.pytorch.org indexes [V]: **cu130** lane has torch 2.13.0+/2.14.x wheels for **cp310–cp315 AND cp314t/cp315t** (manylinux_2_28_x86_64 confirmed for 2.14.1+cu130 incl. cp315 & cp315t); **cu128** tops out at cp314/cp314t (no 3.15). PyPI torch 2.14.1: cp310–cp314 + cp314t (no cp315 on PyPI yet). **torch_tensorrt 2.14.0** (2026-09-23): cp310–**cp314 only — no cp315, no cp315t** [V: pypi.org/pypi/torch_tensorrt/json] |
| moviepy | `/zulko/moviepy` (1506 snippets; versions v2.2/v2.2.1) [V] | 2.x migration notes | Full migration doc indexed: renames above + "substantial breaking modifications… programs written for pre-v2.0 will likely require updates" | Nothing further needed (verdict: drop) |
| vsrife | (not indexed in context7) | — | — | PyPI JSON [V]: 5.7.0 (2026-02-09), py3-none-any, `requires_dist: numpy, requests, torch>=2.10.0, tqdm, vapoursynth>=69` — confirms the "PyTorch 2.10+ baseline" prior fact [OP→V] |

Context7 gaps found and closed: VS 3.15 support (→PyPI wheel tags), spandrel's real requires_python (context7's repo metadata was wrong for the PyPI package), torch Python-version wheel matrix (→download.pytorch.org indexes), torch_tensorrt 3.15 support (→PyPI).

## 3. Python 3.15 & free-threading — status and wheel reality

**Release status [V: python.org + PEP 790 + Python Insider blog]:** 3.15.0 final released **2026-10-09**; 3.15.1 2026-10-21; **3.15.2 2026-10-26** (expedited security release, eMudhra CA incident) → current as of today. 3.10 went EOL 2026-10-01. 3.15 highlights: lazy imports, frozendict/sentinel builtins, UTF-8 default encoding, faster JIT, **PEP 803 abi3t (stable ABI for free-threaded builds)**.

**Free-threading maturity [V: PEP 779 + SC acceptance + free-threading HOWTO]:**
- Phase I (experimental) = 3.13. **Phase II (officially supported, NOT default)** = achieved with 3.14 (SC approval 2025-06-16; perf guardrail ≈10% observed, 15% hard cap). 
- **Phase III (free-threading as default) is still undecided — explicitly "left for a future PEP"**; not done in 3.15. What 3.15 added: macOS installers install free-threading support by default; abi3t stable ABI for C extensions.
- Extensions not marked free-threading-safe silently re-enable the GIL at import (with a warning) — free-threaded ≠ automatically parallel.

**Wheel matrix, verified today from PyPI JSON + download.pytorch.org/whl/cu130 [V]:**

| Package (release date) | cp312 | cp313 | cp314 | cp315 (GIL) | cp314t (FT) | cp315t (FT) |
|---|---|---|---|---|---|---|
| vapoursynth 81 (10-04) | ✓ abi3 | ✓ abi3 | ✓ abi3 | ✓ abi3 | ✗ | ✓ (cp315-cp315t) |
| torch 2.14.1 PyPI (09-30) | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ |
| torch 2.14.1+**cu130** index | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **torch_tensorrt 2.14.0** (09-23) | ✓ | ✓ | ✓ | **✗** | **✗** | **✗** |
| numpy 2.5.4 (10-10) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| spandrel 0.4.2 | pure-py (py3-none-any, ≥3.10) — all builds | 〃 | 〃 | 〃 | 〃 | 〃 |
| vsrife 5.7.0 | pure-py, torch>=2.10, vapoursynth>=69 — all builds | 〃 | 〃 | 〃 | 〃 | 〃 |
| tqdm 4.70.1 | pure-py, ≥3.8 — all builds | 〃 | 〃 | 〃 | 〃 | 〃 |

Reading the matrix:
- **Python 3.15 GIL:** everything installs **except torch_tensorrt** — which is exactly what vsrife `trt=True` imports at runtime. 3.15 = blocked for the approved interpolation path until torch-tensorrt ships cp315 wheels (they track torch majors; earliest realistic window ≈ torch 2.15, ~Q1 2027 [OP inference, not verified]). Escape hatch: vs-mlrt `BackendV2.TRT` (native plugin, no torch_tensorrt) would unblock 3.15, but that is the documented post-benchmark escape hatch, not the primary path.
- **Free-threaded builds:** the only fully-wheeled FT combo is 3.15t (VS has cp315t but no cp314t; torch cu130 has both) — and 3.15t is precisely the build torch_tensorrt does not support. Free-threading is therefore a **wheel-provable dead end for this stack today**.

**Engineering verdict — does free-threading buy anything for THIS pipeline?**
No, and the reasons are structural, not maturity-nitpicking:
1. The parallelism already lives below Python: VS core renders `prefetch` frames concurrently on native threads, releasing the GIL around `getFrame` (context7 pyx source); TRT inference overlaps via `num_streams` CUDA streams; libx265 runs its own CPU thread pool; ffmpeg decode/mux are subprocesses. The legacy script's `threading.Thread` producer was already effectively GIL-free while torch ops ran — and that whole pattern is deleted by the VS graph.
2. Python-level work in the new architecture is config, orchestration, thin tensor glue, and progress reporting — microseconds per frame, not the bottleneck. Free-threading accelerates exactly the layer that isn't loaded.
3. The realistic parallelism need — 2–3 titles at once — is better served by **process-level parallelism**: one `uv run` invocation per title gives independent crash domains, clean per-job CUDA/TensorRT memory accounting (shared-process FT threads would contend for the same device), simple per-job x265 thread budgeting (threads/jobs), zero dependence on every C-extension being FT-clean, and works identically on the GIL build. Free-threading would couple jobs into one address space to save fork overhead we don't pay for.
4. Costs avoided by staying GIL: ~10% single-thread interpreter overhead (Phase II guardrail), torch_tensorrt/extension FT-verification burden, and the least-mature wheel lane.

**Recommendation:** `requires-python = ">=3.12"` (floor set by vapoursynth). Pin the working interpreter to **CPython 3.14, standard GIL build** (`.python-version` / `uv python pin 3.14`): it is the only version where every package in the approved path — including torch_tensorrt for vsrife trt — has wheels (torch cu130 cp314 + torch_tensorrt cp314 + VS abi3 + numpy 2.5.4). Revisit 3.15 (GIL) when torch-tensorrt publishes cp315, and free-threading only if a future all-native-plugin variant (vs-mlrt TRT) plus a measured multi-job win ever justify it.

## 4. Dependency manifest proposal (pyproject.toml — names only, owner rule)

```toml
[project]
name = "anime-upscale"
requires-python = ">=3.12"          # floor: vapoursynth PyPI wheels; pin interpreter 3.14 via .python-version
dependencies = [
  "vapoursynth",                     # R81 line (PyPI official wheels)
  "vsrife",                          # pulls torch>=2.10, numpy, requests, tqdm, vapoursynth>=69
  "spandrel",                        # upscale model loading (pulls torch, torchvision, safetensors, numpy, einops)
  "numpy",
  "torch",                           # must resolve from the cu130 index for RTX 5080 — index config, not a pin
  "torch-tensorrt",                  # required at runtime by vsrife trt=True; cp314 max today
  "tqdm",                            # CLI-level job progress only (frame loop replaced by VS frames())
]

[dependency-groups]
dev = ["ruff", "ty", "pytest", "pytest-cov", "pip-audit"]
```

Implementer notes:
- No version pins (owner rule) — `uv.lock` commits the resolution; `uv python pin 3.14` fixes the interpreter.
- torch from PyPI's default build is wrong for Blackwell: configure the cu130 index in `[tool.uv]` (index/source mapping is environment config, not a version pin). PyPI-only torch would also silently lose cp315 later; cu130 lane is the forward lane.
- vs-mlrt stays OUT of the manifest (not on PyPI [OP+prior research]; 7z into plugin dir when the TRT escape hatch is benched).
- moviepy deliberately absent.

## 5. Provenance

- Context7: resolve+query for /vapoursynth/vapoursynth, /chainner-org/spandrel, /pytorch/pytorch + /websites/pytorch, /zulko/moviepy (details §2).
- Primary sources [V, 2026-10-27]: pypi.org/pypi/{vapoursynth,spandrel,numpy,vsrife,moviepy,tqdm,torch_tensorrt,tensorrt}/json (+ per-version for torch 2.14.1, vapoursynth 81, vsrife 5.7.0, spandrel 0.4.2); download.pytorch.org/whl/cu130/torch/ + /whl/cu128/torch/; python.org downloads + PEP 790 + PEP 779 + PEP 803 + docs.python.org free-threading HOWTO + blog.python.org release posts (via websearch).
- Exa: web_search_exa → **402 credits exhausted** (first probe, logged 2026-10-27T20:49Z); fallback chain (websearch + PyPI JSON) used per standing rule. No cloakbrowser needed.
- [OP] carried from controller/prior verified research: RTX 5080 needs CUDA 12.8+/cu130; vs-mlrt not on PyPI; owner rules (uv, PEP 621, no pins).
- [UNVERIFIED]: torch-tensorrt cp315 ETA (inference from its torch-major tracking cadence); FT single-thread overhead on this exact CPU (PEP 779's ~10% is the project's own number).

## 6. Skills used
brainstorming (verdict framing + decision inventory), grilling (design-tree discipline applied to the controller's three sub-questions), build-with-exa (search discipline + documented 402 fallback), modern-python (uv/PEP 621/dependency-groups manifest), research (primary-source method, Markdown deliverable).

## 7. Memory files read
Full reads: `MEMORY.md`; topics/: stack-and-tools, tech-gotchas, code-gotchas (lines 1–87), communication-behavior, environment-infra, workflow, workflow-git-pr; projects/: anime_upscale.
Lead-section registration (head -n 3, all remaining files — none skipped): topics/: remnawave, 2026-10-09-telegram-serverless, 2026-10-11-novum-organum-report-plan, 2026-10-12-slides-novum-organum-plan, 2026-10-14-opencode-v2-storage, 2026-10-15-binary-re-landmines, 2026-10-15-vision-ocr-scans, 2026-10-21-stepin-book-contract, 2026-10-25-locke-pirogov-referaty, harness-server, hh-ranking-ats-research, job-search; projects/: 4q4r-profile, WMS, anicli-go, anicli-py, bos-pro-bugs, cindex, ert, firefox-vpn-client, freeai-decompile, freepackages, habrparser, hhauto, home-global, interview-kb, iris-proposal, kolosyug, labboardvent, labs-docker-stands, labs, remnavawe_ansible, secureway_bot, shikimori-japanese-theme, stepik-whitehacker-course, teleparse, traffic_mimicry, tspu-research-2026, vibecoding-course, wauploader, wms, xray-raw.
No memory files were written (facts returned to controller only).
