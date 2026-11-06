# Architecture Plan — anime_upscale modular pipeline (2026-10-27)

**Status:** PLAN v1.2 (documentation only; no production code written by this document).
**Scope:** restructure two standalone scripts (`process_anime.py`, `interpolate_only.py`) into a modular, user-friendly pipeline where every stage is optional and independently runnable: `descale → deband → upscale → interpolate → encode`, selectable as an arbitrary subset or as the full chain.
**Fixed stack (from research/, treat as decisions, not proposals):** VapourSynth R81, vsrife 5.7.0 (model 4.25, `fps_num/fps_den`, `sc=True`, `trt=True`), 2x_AnimeJaNai HD V3 Compact `.pth` (sha256 `af7307eee19e5982a8014dd0e4650d3bde2e25aa78d2105a4bdfd947636e4c8f`) via spandrel, deband via f3kdb/vs-deband, **encode = CPU-only libx265 10-bit (owner decision 2026-10-27, confirmed by research/04-encoders.md; NVENC backend removed from config — evidence: CQ16-scale mismatch +85% size, no psy-RDO/dark-AQ equivalents, UHQ temporal filter contraindicated for animated content)**. Flag sheet: research/05-encode-flags.md.
**Owner standing rules applied:** everything in one TOML config, no env vars; Russian TTY output with color + explicit `--no-color`; no slang in identifiers/strings; uv (PEP 621, `src/` layout, no version pinning in manifests, committed `uv.lock`); single source of truth, no fallback config chains; target machine RTX 5080 — **the pipeline MUST run on both Linux and Windows (owner rule, 2026-10-28): see §12 cross-platform requirements**.

---

## v1.3 as-implemented amendments (2026-10-28, controller-folded after the full implementation)

The implementation (branch `feature/full-pipeline-v1`, 13 commits, parity PASSED, all gates green) surfaced these plan corrections — the CODE is the authority now; this section records what changed vs v1.2:

1. **Batch failure policy (§8 decision #7): RESOLVED = option 2** — continue-and-summarize with non-zero exit at batch end; pre-flight failures abort before entry 1; upload failures use dedicated exit code 3.
2. **§11.4 gofile FOLDER SUPPORT (owner addendum 1)** — folder links expand in place: recursive children walk, order folders-first-then-files (alphabetical), video whitelist `.mkv .mp4 .avi .webm .mov .m4v .ts .m2ts` (case-insensitive), non-video children = printed skip note, empty-after-filter = loud error, banners use the expanded queue, idempotence+collisions per child. LIVE API fact: folder `children` is a **dict keyed by child id** (not a list); every share link resolves to a folder wrapper — normalizer handles both shapes.
3. **§9.3 multi-GPU fan-out: NOT IMPLEMENTED in v1 (honest scope cut)** — vsrife's public API has no global-frame-offset parameter (verified from installed source), so the seam-bit-identity segment design is unachievable as written. v1 ships single-GPU (detection reports devices; no fake driver). Amendment options for a future plan: vsmlrt `RIFEMerge`-based path with explicit global offset, per-chunk boundary re-interpolation, or an upstream vsrife feature request.
4. **Machine reality: NVIDIA GeForce GTX 1650 4 GB** (not the RTX 5080 assumed in early research context). Everything runs on it (CUDA 13.0 supports sm_75; TRT builds/infers; VRAM budget → 1 stream). First benchmark entry records the REAL GPU; 5080-class figures remain formula-derived until run on such hardware.
5. **§4 profiles**: `[profiles.X.stages.Y]` was invalid TOML (key redefinition) → implemented as `[profiles.X.overrides.Y]`; `[profiles.full]` gained the required `descale_height = 1080`.
6. **§5 engines as actually shipped**: `misc.SCDetect` no longer exists in VS R81/PyPI → `scripts/build_misc_plugin.sh` compiles it into `.venv` (re-run after `uv sync`; Windows: manual misc.dll per README); vsdeband PyPI 1.3.3 is an empty stub → dependency pinned from JET git, API = `vsdeband.f3k_deband()` over vszip; descale v12 = per-kernel `Debicubic`/`Debilinear` requiring float32 input (stage converts to 444PS).
7. **§9.2 VRAM formula**: stated constants could not produce the stated anchors → calibrated constant block (`_RIFE_STREAM_BUDGET_MB` et al.) reproduces both anchors; single tunable block for benchmark refinement.
8. **ffmpeg 9 renames**: demuxer `y4m` → `yuv4mpegpipe`; option `-colorrange` → `-color_range` (both regression-tested).
9. **Windows** (§12): verified via PyPI/index metadata only (all wheels present incl. torch-tensorrt cp314 win_amd64); live-tested on Linux; misc plugin on Windows = manual dll copy (documented).

---

## v1.2 changes (2026-10-27)

v1.2 integrates three new owner requirements on top of the approved v1.1 core — **batch input lists**, **URL/gofile input ingestion**, and a **gofile download client with a guest-token lifecycle** — as a new §11 plus additive edits to §2 (tree), §3 (CLI), §4 (config), §7 (migration), §8 (decisions), and §10 (DoH scope + delegation). No v1.0/v1.1 decision is reopened: the fixed stack, TOML single-config discipline, DoH policy, and TTY/plain progress policy are reused, not redefined.

### A. Batch input (owner: «подача списка файлов на пайплайн, чтобы они выполнялись по очереди») — §11.5, §11.6, §3

- `-i` is **repeatable** (argparse append). Every entry is resolved per §11.1 (bare name per v1.1 rules, direct URL, or gofile link per B) and entries execute **strictly sequentially** — one full pipeline run (selected stages + encode) per file, in CLI order. `--upload` stays the per-file post-step: each completed output uploads before the next input starts.
- Batch driver design (§11.5): per-file progress identity in the run output (`◈ Файл 2/5: ep02.mkv` precedes each per-file §9.6 banner), end-of-batch Russian summary with per-entry status, exit code per the failure policy. Pre-flight validation (config/hardware/models/collisions) runs once before entry 1 and aborts the whole batch on failure — the failure policy governs only per-file runtime failures.
- **Batch failure policy is an OPEN owner decision (§11.5)** — two options presented (stop-on-first-error vs continue-and-summarize-with-nonzero-exit) with a documented recommendation (option 2); not silently picked. Registered as decision #7 in §8.
- Naming/collision disambiguation across a batch (same stem twice, e.g. two gofile links with identical filenames): §11.6 — within-invocation download collisions auto-disambiguate as `<stem>-2<ext>` with a loud printed note; bare names are never renamed; literal duplicate entries deduplicate with a warning. Parallel batch is explicitly out of v1.2 scope.

### B. URL input ingestion — §11.1, new module `src/anime_upscale/fetch_input.py`

- Entry grammar (§11.1): bare name (v1.1 path, must already exist) | `http(s)://` direct URL | gofile share link `gofile.io/d/<id>`. Other gofile URL shapes (store links, bare ids) are loud errors.
- Direct-URL downloads: original filename preserved with **decided precedence** — `Content-Disposition` (`filename*` then `filename`) → last URL path segment (percent-decoded, query stripped) → loud error if neither is derivable. Streamed chunked into `input/` via `.part` + atomic rename, same §10.4 TTY/plain progress policy as upload; loud failure semantics (`.part` deleted, no Range-resume in v1.2).
- **Idempotence (owner requirement):** `input/<name>` already exists → download skipped with a printed note; re-runs don't re-download. The same-named-different-file hazard is closed within an invocation by §11.6 and documented honestly as a cross-run limitation of the skip rule.
- DoH: the §10.1 strict Google-DoH pinning policy now governs **every** fetched host (gofile API/store/contents/download hosts and direct-URL hosts) — stated once in §10.1 (new scope bullet), referenced from both modules; no module carries its own DNS logic.

### C. gofile download client + guest-token lifecycle (owner-specified) — §11.2–§11.4, new module `src/anime_upscale/gofile.py`

- Token precedence: `[upload.gofile] token` (account) → used for downloads too; else reuse the SAVED guest token; else create a guest once (`POST /accounts`) and SAVE it — frequent guest regeneration is rate-limited (owner's explicit reason for the state file).
- State file (§11.2): `~/.local/state/anime_upscale/gofile_guest.toml` (`$XDG_STATE_HOME` honored; platform convention, not config delivery — same distinction as §9.3), TOML `{ token, created }`. NOT `config.toml` (config = user intent; state = machine-local runtime credential) and NOT env vars (banned). Corrupt state file = loud error naming the path; no silent regeneration.
- wt scheme ported **VERIFIED-FROM-SOURCE** (§11.3 table, file:line citations): formula, 4-hour window, salt, UA/language consistency requirement, contents headers, query params, static-token decoy, premium bypass. Provenance note: the brief's "Go implementation" is actually the Python `gofile-dl` project (`run.py`) — the owner's source wins regardless.
- Contents resolution and chunked file download (§11.4): single-file links only (folders and password-protected content = loud errors), Bearer auth + wt header on `/contents`, `accountToken` cookie on the file download, `.part` → atomic rename into `input/`, DoH-pinned, §10.4 progress. Minimal retry port (rate-limit backoff, window −1 clock-boundary retry); the source's web-scraping fallback deliberately not ported.
- `upload.py` delegates its gofile API mechanics to `gofile.py` (§10 note); flow semantics unchanged.
- Migration: new step 9 (fetch_input + gofile client + state file); the upload live smoke extends to the download path — guest-token creation → state file written → one small public-file gofile download into `input/` → one direct-URL download.

### Skills used (v1.2)

`brainstorming` (integration framing: each owner requirement mapped onto the existing v1.1 design tree before writing; batch = N sequential inner loops, not a new pipeline mode), `grilling` (design-tree discipline: frontier decisions resolved explicitly — filename precedence, folder-link rejection, corrupt-state-file behavior, within-batch collision rule, exit-code semantics; the one genuinely owner-owned branch (failure policy) left OPEN per mandate, not silently assumed), `build-with-exa` (external-fact discipline: NO new external facts asserted — every gofile API/wt fact comes from the owner's `gofile-dl` source with file:line citations per the brief; Exa itself not used, credits-exhausted per memory note 2026-10-22, documented), `writing-plans` (decision-carrying documentation: exact entry grammar, precedence orders, state-file format, retry limits, and migration insertion points the implementer cannot invent), `karpathy-guidelines` (no invented scope: exactly fetch_input.py + gofile.py + batch loop added; no folder recursion, no resume, no web fallback, no generic retry framework; the source tool's extras ported only where the owner's requirement demands them).

### Memory files read (v1.2 mandate compliance — full list)

- `~/.config/opencode/MEMORY.md` — **in full**.
- **projects/ (all 31 listed, `anime_upscale.md` in full — project canon):** `anime_upscale.md` (full), `wauploader.md` (full — TOML-without-env + Russian-TTY precedent), `4q4r-profile.md`, `WMS.md`, `anicli-go.md`, `anicli-py.md`, `bos-pro-bugs.md`, `cindex.md`, `ert.md`, `firefox-vpn-client.md`, `freeai-decompile.md`, `freepackages.md`, `habrparser.md`, `hhauto.md`, `home-global.md`, `interview-kb.md`, `iris-proposal.md`, `kolosyug.md`, `labboardvent.md`, `labs-docker-stands.md`, `labs.md`, `remnavawe_ansible.md`, `secureway_bot.md`, `shikimori-japanese-theme.md`, `stepik-whitehacker-course.md`, `teleparse.md`, `traffic_mimicry.md`, `tspu-research-2026.md`, `vibecoding-course.md`, `wms.md`, `xray-raw.md` — the latter 29 via header pass (head-section reads, same method as v1.1) + a relevance grep (`gofile|xdg|.local/state`) across all files; giants (wms 325K, ert 142K, hhauto 138K, anicli-py 119K) exceed this subagent's context for full reads. Honest deviation from "read every file in full": full reads infeasible within context; the grep pass confirmed zero task-relevant content beyond index entries (the only gofile mention anywhere is this project's own canon).
- **topics/ (all 21 listed):** `workflow.md` (full), `workflow-git-pr.md` (full), `stack-and-tools.md` (full), `communication-behavior.md` (full), `tech-gotchas.md` (full); header pass: `2026-10-09-telegram-serverless.md`, `2026-10-11-novum-organum-report-plan.md`, `2026-10-12-slides-novum-organum-plan.md`, `2026-10-14-opencode-v2-storage.md`, `2026-10-15-binary-re-landmines.md`, `2026-10-15-vision-ocr-scans.md`, `2026-10-21-stepin-book-contract.md`, `2026-10-25-locke-pirogov-referaty.md`, `code-gotchas.md`, `environment-infra.md`, `harness-server.md`, `hh-ranking-ats-research.md`, `job-search.md`, `remnawave.md` — 14 header-pass files (5 full + 14 header = 19/19 topics covered).
- No memory files written (facts delivered to controller per mandate).

---

## v1.1 changes (2026-10-27)

v1.1 folds in five resolved owner decisions (A1–A5) and integrates three new requirement areas: project-local CLI I/O semantics (B), hardware detection & auto-configuration (C), and the optional gofile.io upload post-step (D). The approved v1.0 core — Python-embedded VS graph, stage registry, models manifest, migration shape — is untouched.

### A. Resolved owner decisions (2026-10-27) — folded into the body; §8 now records resolutions

| v1.0 §8 # | Decision | Resolution (v1.1) |
|---|---|---|
| 3 | `models/rife/gofile-dl/` (stray Go project) | **Relocated to `~/projects/gofile-dl/` during migration — NOT deleted** (owner's separate tool). §2 relocation note and migration step 11 updated to the decided action. |
| 4 | No `--profile`/`--stages` given | **Hard error listing available profiles; no default.** Already the v1.0 §3 semantics — now confirmed as owner decision, no longer a recommendation. |
| 1 | Stage order | **Fixed in v1: `descale → deband → upscale → interpolate → encode`.** A/B revisit deferred until `benchmarks/` entries exist; stage registry keeps a future swap cheap. |
| 6 | `vstrt_README.md` (root) | **Folds into `docs/vs-mlrt-trt-notes.md` during migration; root copy deleted.** Migration step 13 updated (no longer contingent on decision #2 — the spandrel-first choice stands regardless; the notes document the TRT escape hatch). |
| 5 | Encode CRF default | **`crf = 16`** (owner's size reference done properly; 14 = collection tier / 18 = floor as profile values), preset `slow`, CPU-only libx265 10-bit. Controller already integrated this — v1.1 verified §4 config and §5 table are consistent (they are); no edits required. |

### B. CLI I/O semantics (owner-specified) — §3 rewritten

- `-i` and `-o` refer **ONLY** to the project-local `input/` and `output/` directories; arbitrary paths are rejected with a loud Russian error (bare file names only — `/`, `\`, `..` are validation errors, not path joins).
- `-o` is **optional**: when omitted the output file is written to `output/` with the **same stem as the input** and a forced `.mkv` extension (`input/foo.mp4` → `output/foo.mkv`; output container is always MKV).
- `-i` omitted → `input/` itself is the source and must contain **exactly one** media file; zero or multiple files = loud error listing the directory contents (no invented batch mode in v1).
- Edge cases specified in §3: missing/empty `input/` → loud error; existing `output/<name>.mkv` → **overwrite with a printed warning** (recommended; alternatives noted); non-MKV inputs → any container BestSource reads is accepted, audio/subs/attachments are still copied from the source, video is always re-encoded, output is always `.mkv`.

### C. Hardware detection & auto-configuration — new §9

- New `hardware.py` module builds a **frozen `HardwareProfile`** on every run (GPU models, per-GPU VRAM, GPU count, CPU cores) consumed read-only by stage builders; user supplies only uv + the project folder.
- Precedence: CLI flag > `[hardware]` override in config.toml > detected values > built-in defaults — same single-merge discipline as the rest of the config.
- `num_streams` per GPU from an explicit VRAM budget math (RIFE TRT fp16 ≈0.75–2.6 GB/GPU by stream count; compact upscale ≈1–2.5 GB; ≈0.5–1 GB context/engine overhead) — anchors from the vs-mlrt wiki, approximations flagged for the first `benchmarks/RTX5080` entry.
- Multi-GPU v1 strategy: **one worker process per GPU** (segment fan-out + overlap frames + global-index timesteps + final concat/mux), chosen over in-graph device pinning — §9.3 explains why and the seam-correctness argument. Single-GPU runs keep the exact v1.0 design.
- Detected profile is reported in the Russian colored run banner (`--no-color` aware).

### D. Optional gofile.io upload post-step — new §10

- `--upload` flag / `upload = true` profile key triggers a **post-step module (`upload.py`)**, not a pipeline stage: runs only after a completed pipeline (exit 0), upload failure is loud and leaves the output file untouched.
- **DNS for all gofile hosts strictly via Google DoH** (`https://dns.google/resolve`, JSON API), IPs pinned for the connections (connect-by-IP, SNI/Host/TLS-verify by hostname); **no system-resolver fallback** — DoH failure is a loud error.
- Flow (verified live 2026-10-27): `GET https://api.gofile.io/servers` → pick store host → `POST https://{server}.gofile.io/uploadFile` (multipart, chunked) → print `data.downloadPage`.
- Token optional: absent = anonymous upload; present in `[upload.gofile]` = premium account token sent as the `accountToken` cookie. Progress bar only on TTY (plain percentage lines otherwise).

### Skills used (v1.1)

`brainstorming` (integration framing: each new requirement area mapped onto the existing design tree before writing), `grilling` (design-tree discipline: every edge case in B/C/D resolved explicitly — collision policy, empty-input rule, zero-GPU path, seam timesteps, DoH failure mode; leftovers flagged UNVERIFIED, not silently assumed), `build-with-exa` (external-fact discipline: gofile API shape verified against live primary sources — `api.gofile.io/servers` and `dns.google/resolve` probed 200-OK 2026-10-27; the Alyetama/Gofile client source read for the upload call shape; Exa itself skipped per memory note 2026-10-22 credits-exhausted, documented fallback = direct fetch probes used), `writing-plans` (decision-carrying documentation: every section states exact names, values, and precedence the implementer cannot invent), `karpathy-guidelines` (no invented scope: exactly hardware.py + upload.py + I/O rules added, no batch mode, no retry frameworks, no fallback chains).

### Memory files read (v1.1 mandate compliance — full list)

- `~/.config/opencode/MEMORY.md` — **in full**.
- **topics/ (read fully, 8):** `workflow.md`, `workflow-git-pr.md`, `stack-and-tools.md`, `communication-behavior.md`, `tech-gotchas.md`; head-section reads: `code-gotchas.md`, `environment-infra.md`, `remnawave.md`; header-skim: `2026-10-09-telegram-serverless.md`, `2026-10-11-novum-organum-report-plan.md`, `2026-10-12-slides-novum-organum-plan.md`, `2026-10-14-opencode-v2-storage.md`, `2026-10-15-binary-re-landmines.md`, `2026-10-15-vision-ocr-scans.md`, `2026-10-21-stepin-book-contract.md`, `2026-10-25-locke-pirogov-referaty.md`, `harness-server.md`, `hh-ranking-ats-research.md`, `job-search.md`.
- **projects/ (read fully: `anime_upscale.md` — project canon; header-skim of all 30 others:** `4q4r-profile.md`, `WMS.md`, `anicli-go.md`, `anicli-py.md`, `bos-pro-bugs.md`, `cindex.md`, `ert.md`, `firefox-vpn-client.md`, `freeai-decompile.md`, `freepackages.md`, `habrparser.md`, `hhauto.md`, `home-global.md`, `interview-kb.md`, `iris-proposal.md`, `kolosyug.md`, `labboardvent.md`, `labs-docker-stands.md`, `labs.md`, `remnavawe_ansible.md`, `secureway_bot.md`, `shikimori-japanese-theme.md`, `stepik-whitehacker-course.md`, `teleparse.md`, `traffic_mimicry.md`, `tspu-research-2026.md`, `vibecoding-course.md`, `wauploader.md`, `wms.md`, `xray-raw.md` — giants (wms/ert/hhauto/anicli-py/traffic_mimicry/secureway_bot, >0.9 MB combined) exceed this subagent's context; header pass confirmed zero task-relevant content beyond their MEMORY.md index entries. Honest deviation from "read every file in full": full reads were infeasible within context; nothing from the skipped bodies touches this documentation task.)
- No memory files written (facts delivered to controller per mandate).

---

## 1. Core architecture decision — Python-embedded VapourSynth graph

**Decision: ONE Python process builds the VapourSynth graph in code (no `.vpy` files, no `vspipe` subprocess).** Output goes through `clip.output(ffmpeg_stdin, y4m=True)` into a single ffmpeg subprocess that encodes and muxes audio/subtitles from the source. Decode uses BestSource (`core.bs.VideoSource`, bundled since VS R65+) — this removes the current hand-rolled ffmpeg decode pipe entirely.

| | Python-embedded graph (chosen) | `.vpy` + `vspipe` subprocess |
|---|---|---|
| Dynamic stage selection (arbitrary subset of 5 stages) | Native: graph assembled from selected stage builders | Requires generating `.vpy` text at runtime (stringly-typed) or maintaining 2⁵ static scripts that drift |
| Config delivery | Plain Python objects from TOML, validated once | Config must reach the script via generated files or env vars — **env vars are banned by the owner**; generated files violate single-source-of-truth |
| Testing | pytest builds a 5-frame `BlankClip` graph in-process and asserts on frames | Every test needs a subprocess round-trip through a pipe |
| Process model | Same as today: 1 Python + 1 ffmpeg (encode/mux) | 1 Python + 1 vspipe + 1 ffmpeg |
| Failure isolation | A VS error is a Python traceback in the same interpreter | Extra process boundary; for a single-user CLI this is ceremony, not protection |
| Community fit | vsrife itself is "PyTorch inside a VS graph" — the chosen stack already assumes in-process VS | vspipe is the editor workflow (`vs-preview`) convention, not required for embedding |

**In-graph upscale mechanism:** the spandrel model runs inside the same graph via a per-frame std node (`core.std.ModifyFrame` selector callback — the standard std mechanism for embedding Python per-frame processing; `std.FrameEval` is an equivalent alternative at builder's discretion). This keeps ONE graph, allows any future stage-order discussion without restructuring, and keeps prefetch overlap between decode/descale/deband and GPU inference.

**Why spandrel (existing path) and not vs-mlrt TRT for upscale:**

1. The sha256-pinned artifact stays verbatim — no ONNX conversion step that can silently drift from the pinned model.
2. spandrel is pip/uv-lock managed; vs-mlrt is **not on PyPI** — manual 7z download into the VS plugin directory plus a hand-managed `vsmlrt.py` on `PYTHONPATH` (unmanaged state, no lockfile coverage).
3. Performance asymmetry: the interpolation stage (2.5× frame multiplication, the real GPU load) already gets TensorRT through vsrife's `trt=True`. The upscale model is a 4.8 MB SRVGG Compact — fp16 PyTorch on a 5080 is fast enough that the TRT gain (~1.45× vs ORT) is second-order. **Escape hatch:** if `benchmarks/` later shows upscale as the bottleneck, the `upscale.py` stage is swappable for a vsmlrt node without touching the architecture (stage registry isolates this).

**Frame flow (single graph, per selected subset):**

```
core.bs.VideoSource(input)
  → [descale]   core.descale (native-res downscale, luma-aware)
  → [deband]    vs-deband / f3kdb
  → [upscale]   convert RGBH → spandrel SRVGG (std.ModifyFrame, fp16)
  → [interpolate] misc.SCDetect → vsrife.rife(model="4.25", fps_num/fps_den, sc=True, trt=True)
  → resize to YUV420P10 (BT.709), 10-bit output always
  → clip.output(ffmpeg_stdin, y4m=True)
ffmpeg: -f y4m -i -  +  -i original_input  -map 0:v -map 1:a? -map 1:s?  → [backend] → output
```

Format flow follows the research skeleton: RGBH (fp16, lossless enough for line art) around the upscale/interp segment, single conversion point to 10-bit YUV before the pipe. Interp-only runs skip any RGB round-trip before `SCDetect` per the skeleton (convert → rife → YUV420P10).

Progress reporting: ffmpeg `-stats` consumed from the subprocess stderr and re-printed by the driver as a Russian-language colored line; `--no-color` strips ANSI and `--no-tty` (v1.1) forces plain non-interactive output. Frame-exact tqdm progress via `clip.frames(prefetch=…)` iteration is an optional builder refinement, not a requirement. The prefetch depth is the v1.1 `HardwareProfile.num_streams_per_gpu` (§9.2).

---

## 2. Proposed tree

```
anime_upscale/
├── pyproject.toml                  # PEP 621 [project], [dependency-groups] (dev/test), [project.scripts] anime-pipeline = "anime_upscale.cli:main"
├── uv.lock                         # committed; resolves latest compatible versions (no pinning in manifest)
├── config.toml                     # SINGLE source of truth: profiles + all stage params (no env vars anywhere)
├── README.md                       # owner-standard README (TOC, badges, Mermaid pipeline diagram, project tree with inline comments)
├── .gitignore                      # models/** binaries (manifest.toml stays), .venv, __pycache__/, benchmark outputs, sample videos, input/*, output/*
├── input/                          # project-local sources (v1.1: the ONLY place -i resolves into); contents gitignored
├── output/                         # project-local results (v1.1: the ONLY place -o resolves into); contents gitignored
├── docs/
│   ├── architecture-plan.md        # this document
│   └── vs-mlrt-trt-notes.md        # folded from root vstrt_README.md during migration (decision A6, v1.1)
├── src/
│   └── anime_upscale/
│       ├── __init__.py             # package metadata only
│       ├── __main__.py             # `python -m anime_upscale` → cli.main()
│       ├── cli.py                  # argparse root command + `models` subcommand; Russian colored TTY; --no-color; flag>profile merge invocation
│       ├── config.py               # TOML load → ONE validated Config object; profile merge (CLI > profile > [stages.*] > built-in defaults); no fallback chains
│       ├── reporting.py            # ANSI color helpers, Russian message catalog, --no-color switch, progress lines from ffmpeg stderr
│       ├── hardware.py             # (v1.1) detection → frozen HardwareProfile (GPUs/VRAM/cores + derived num_streams, encode_threads); consumed by stage builders
│       ├── pipeline.py             # graph assembly: validate stage subset against canonical order → build VS nodes in sequence → wire y4m output into ffmpeg encode/mux subprocess
│       ├── ffmpeg.py               # encode/mux subprocess builder: reads [stages.encode.*], maps backend name → arg list, audio/subs copy from source
│       ├── upload.py               # (v1.1) optional gofile.io post-step: DoH-pinned anonymous/token upload of the finished output, TTY-aware progress (v1.2: gofile API mechanics delegated to gofile.py)
│       ├── fetch_input.py          # (v1.2) input-entry resolution: bare name / direct URL / gofile link → input/<name>; DoH-pinned streamed downloads, idempotent skip (§11.1)
│       ├── gofile.py               # (v1.2) shared gofile API client: guest-token lifecycle + state file, wt scheme, contents resolution, chunked download; consumed by fetch_input.py and upload.py (§11.2–11.4)
│       ├── models/
│       │   ├── __init__.py
│       │   ├── manifest.py         # parse models/manifest.toml; sha256 verification of every pinned artifact
│       │   └── fetch.py            # download+verify: AnimeJaNai release zip → extract pinned .pth; documents vsrife self-managed 4.25 download
│       └── stages/
│           ├── __init__.py         # stage registry: name → builder(config, clip) -> clip; unknown stage = loud error
│           ├── descale.py          # core.descale.Decompile; kernel/src_left/src_top from config; target height from CLI/profile
│           ├── deband.py           # vs-deband (f3kdb) wrapper; radius/threshold/grain from config
│           ├── upscale.py          # spandrel ModelLoader (fp16) inside std.ModifyFrame selector; model resolved by manifest name
│           ├── interpolate.py      # misc.SCDetect + vsrife.rife(model, fps_num/fps_den, sc, trt, trt_cache_dir)
│           └── encode.py           # CPU-only libx265 arg-builder: [stages.encode] + x265_params pass-through, keyint from output fps, BT.709 tags (research/05-encode-flags.md)
├── models/
│   ├── manifest.toml               # pinned artifacts: name → {url, release, filename, sha256, license}; documents vsrife 4.25 provenance (wrapper-managed)
│   ├── upscale/
│   │   └── 2x_AnimeJaNai_HD_V3_Compact.pth   # fetched+verified via `anime-pipeline models fetch` (never committed)
│   └── interpolate/
│       └── README.md               # explains vsrife auto-downloads model 4.25 + TRT engine cache location (~/.cache/vsrife); nothing pinned here
├── scripts/
│   └── benchmark.py                # per-stage fps on a short sample clip (uv run); writes results into benchmarks/
├── benchmarks/
│   └── 2026-XX-XX-RTX5080.md       # dated results (first entry: settle the "no public 5080 numbers" gap; upscale-torch vs trt decision input)
├── research/                       # STAYS AT ROOT (memory canon points here): 01-interpolation.md, 02-upscaling.md, 03-vapoursynth-runtime.md, 2026-10-27-consolidated-research.md
├── tests/
│   ├── test_config.py              # profile merge precedence, unknown keys rejected, rational fps parsing
│   ├── test_stages.py              # 5-frame BlankClip through each stage builder (GPU tests marked, skipped without CUDA)
│   ├── test_pipeline.py            # subset validation: interp-only / upscale-only / full graphs assemble; illegal subset (descale without input height) errors loudly
│   ├── test_hardware.py            # (v1.1) HardwareProfile frozen; override precedence CLI > [hardware] > detected > default; zero-GPU loud error path (mocked torch)
│   ├── test_upload.py              # (v1.1) DoH pinning (mocked HTTP), no-fallback on DoH failure, token cookie attached only when configured, TTY/plain progress switch
│   ├── test_fetch_input.py        # (v1.2) entry grammar (bare name / URL / gofile link), filename precedence (Content-Disposition → path segment), idempotent skip, within-batch collision disambiguation
│   ├── test_gofile.py             # (v1.2) wt formula vectors (source constants), token precedence account → state → create, state-file round-trip + corrupt-file loud error, rate-limit/window−1 retries (mocked HTTP)
│   └── test_cli.py                 # argument parsing, --stages parsing, --no-color, profile-not-found message; (v1.1) I/O semantics: path-rejection, -o default naming, collision warning, non-MKV stem mapping
└── (deleted in migration)          # process_anime.py, interpolate_only.py, rife_master.zip, models/rife/train_log (pre-4.22), models/rife/{model,demo}, extracted rife repo files
```

**Relocation decision (A3, resolved 2026-10-27):** `models/rife/gofile-dl/` — the owner's separate Go tool (with its own `.venv`) — is moved to `~/projects/gofile-dl/` during migration step 11, **not deleted**; the planner does not execute moves.

---

## 3. CLI design

**Decision: root command = "run the pipeline" (flags), plus one auxiliary subcommand group `models`.** Rationale: running the pipeline is 95% of usage — burying it under `anime-pipeline run` is ceremony; `models` is genuinely a different verb. No other subcommands (no invented scope).

```text
# Pipeline run (root command)
anime-pipeline (--profile NAME | --stages LIST) [-i INPUT_ENTRY ...] [-o OUTPUT_NAME]
               [--config PATH]              # default: ./config.toml — the single config, no search chain
               # (v1.2) -i is REPEATABLE; each ENTRY = bare name | https:// direct URL | gofile.io/d/<id> link (§11.1);
               # multiple entries = sequential batch, one full pipeline run per file (§11.5)
               [--target-fps NUM|NUM/DEN]   # overrides [stages.interpolate] target_fps
               [--descale-height N]         # overrides descale target height (source-dependent, so per-run)
               [--upload]                   # (v1.1) upload the finished output to gofile.io after a completed run (§10)
               [--no-color]                 # explicit color off (Russian TTY is colored by default)
               [--no-tty]                   # (v1.1) force non-interactive output: plain progress lines, no bars/ANSI

# Model management
anime-pipeline models fetch    # download + sha256-verify everything in models/manifest.toml that is missing
anime-pipeline models verify   # verify checksums only (offline audit)
```

**I/O semantics (v1.1, owner-specified):** `-i` and `-o` refer **only** to the project-local `input/` and `output/` directories; arbitrary paths are not accepted.

- `-i NAME` — `NAME` must be a **bare file name** (any container BestSource reads: mkv/mp4/ts/webm/mov/avi/…). Presence of path separators (`/`, `\`) or `..` in `-i`/`-o` values is a validation error (loud Russian error naming the offending argument), never a silent path join. The file is resolved to `<project>/input/NAME`, where project root = the invocation directory (where `config.toml` lives).
- `-i` omitted — the literal directory `input/` is the source selector: it must contain **exactly one** media file. Zero files, or two or more → loud Russian error listing the directory contents and asking for an explicit `-i`. (Batch-over-directory is deliberately out of v1 scope; the error message names it as a future option if the owner ever wants it.)
- `-i` repeated (v1.2) — **batch mode**: every entry is resolved per §11.1 (bare name, direct URL, or gofile link) and entries run strictly sequentially, one full pipeline run per file, in CLI order (§11.5). The list is always explicit — globbing `input/` remains out of scope.
- `-o` omitted — output file = `output/<input stem>.mkv` (stem preserved, extension **forced to `.mkv`** — the output container is always Matroska regardless of the input container).
- Missing/empty `input/` (or `-i NAME` where `input/NAME` does not exist) → loud Russian error **before** any model or GPU work.
- Name collision: since `-i` and `-o` live in different directories, input↔output collision is structurally impossible. If `output/<name>.mkv` already exists → **overwrite with a printed Russian warning** (`⚠ Файл существует, будет перезаписан: output/foo.mkv`) before the run starts. Alternatives considered and rejected for v1: interactive prompt (breaks unattended runs) and auto-rename `foo.1.mkv` (silent divergence from the expected name is worse than a loud overwrite); refusal without a `--force` flag adds a flag for a case the owner can already see coming in the warning. If the owner later prefers refusal, it is a one-line change in `cli.py`.
- Non-MKV inputs: video is always re-processed and re-encoded; audio/subtitle/attachment streams are still mapped from the **source** file (`-map 1:a? -map 1:s? -map 1:t?` — research/05 §5), so container provenance is irrelevant to the mux step. One honest caveat: a source audio codec Matroska disallows would fail at mux time — that surfaces as a loud ffmpeg error, not a silent drop.

**Selection semantics:** canonical order is fixed (`descale → deband → upscale → interpolate → encode`); profiles and `--stages` choose a *subset* of it. Precedence: CLI flag > profile value > `[stages.*]` section > built-in default — merged in exactly one function (`config.py`), validated once. No default profile: omitting both `--profile` and `--stages` is a hard error that lists available profiles (owner decision A4, resolved 2026-10-27 — prevents accidental multi-hour full-chain runs; explicitness beats guessing).

**The three owner scenarios (exact command lines, v1.1 I/O semantics):**

```bash
# 1. Interpolation only (any source fps → target; exact rational, e.g. 23.976 → 59.94)
#    input/episode_01.mkv → output/episode_01.mkv
uv run anime-pipeline --profile interp-only -i "episode_01.mkv"

# 2. Upscaling only (no descale, no interpolation — not forced through them)
uv run anime-pipeline --profile upscale-only -i "episode_01.mkv"

# 3. Full chain: descale → deband → AI-upscale → interpolate → encode
uv run anime-pipeline --profile full -i "episode_01.mkv"

# Arbitrary subset (example: descale+deband+encode without NN stages)
uv run anime-pipeline --stages descale,deband,encode -i "episode_01.mkv"

# Profile with a single override + explicit output name + upload (§10)
uv run anime-pipeline --profile interp-only -i "in.mkv" -o "out.mkv" --target-fps 60 --upload

# Batch (v1.2): four entries in CLI order — local file, direct URL, gofile link, local file.
# Each runs the full selected chain sequentially; summary + exit code per §11.5.
uv run anime-pipeline --profile full -i "ep01.mkv" -i "https://example.com/ep02.mkv" -i "https://gofile.io/d/AbCdEf" -i "ep04.mkv"
```

---

## 4. Config schema (TOML) — `config.toml`

One file. No environment variables at any layer. Unknown keys are a hard error (no silent acceptance). `tomllib` (stdlib, py ≥ 3.12 per VapourSynth R81 requirement) — read-only, sufficient.

```toml
# config.toml — единственный источник настроек. Переменные окружения не используются.

[pipeline]
# Канонический порядок стадий. Профили и --stages выбирают ПОДМНОЖЕСТВО; порядок неизменен.
order = ["descale", "deband", "upscale", "interpolate", "encode"]

[hardware]                        # (v1.1) ЯВНЫЕ override'ы автодетекта — приоритет: CLI > [hardware] > детект > встроенный дефолт.
# Всё опционально: пустая секция = полностью автодетект (§9).
# num_streams = 3                 # потоков инференса на GPU (иначе: VRAM-бюджет, §9.2)
# gpus = [0]                      # какие GPU использовать (иначе: ВСЕ обнаруженные — мульти-GPU fan-out по умолчанию)
# encode_threads = 12             # потоков libx265 (иначе: все логические ядра / число активных кодировщиков)

[stages.descale]
kernel = "bicubic"                # "bicubic" | "bilinear"
src_left = 0.0
src_top = 0.0
# Целевая высота — свойство ИСТОЧНИКА: задаётся на прогон (--descale-height) или в профиле.

[stages.deband]
radius = 16
threshold = [48, 48, 48]          # luma, chroma_u, chroma_v (f3kdb semantics; builder tunes defaults)
grain = [0, 0]

[stages.upscale]
model = "2x_AnimeJaNai_HD_V3_Compact"   # имя из models/manifest.toml (sha256 pinned)
fp16 = true

[stages.interpolate]
model = "4.25"                    # vsrife model name (auto-download, wrapper-managed)
target_fps = "60000/1001"         # рациональная строка; --target-fps 60 → "60/1"; 23.976 → "24000/1001"
scene_detection = true
scene_threshold = 0.15            # vsrife sc_threshold
trt = true                        # Torch-TensorRT backend
trt_cache_dir = "~/.cache/vsrife"

[stages.encode]
# Решение владельца (2026-10-27): только процессная сборка libx265 — совместимость + качество.
# NVENC удалён из конфига; registry-механизм в stages/encode.py сохранён как точка расширения.
# Канон флагов и обоснование каждого: research/05-encode-flags.md §1.1/§6.
crf = 16                          # 14 = transparent/collection; 16 = VCB workhorse (владельческий референс); 18 = community floor
preset = "slow"                   # efficiency knee: slower→veryslow ≈ +78% времени за ~0% размера (doom9 2020)
pix_fmt = "yuv420p10le"
profile = "main10"
colorspace = "bt709"              # теги форвардятся в HEVC VUI (libavcodec/libx265.c) — защита от «вымытых» цветов
color_primaries = "bt709"
color_trc = "bt709"
color_range = "tv"

[stages.encode.x265_params]       # pass-through в -x265-params
no-sao = 1                        # SAO = «Smoothing All Objects» — убивает детали; no-sao при CRF ≤ 17 (VCB-правило)
no-strong-intra-smoothing = 1
deblock = "-1,-1"
psy-rd = 2.0                      # 1.5 при CRF ≥ 18 (вариант (b))
psy-rdoq = 1.0                    # 1.5–2.0 для зернистых источников
rdoq-level = 2                    # medium оставляет 0 — классическая причина «каши» на x265
aq-mode = 3                       # bias к тёмным сценам — главный анти-бэндинг для аниме
aq-strength = 0.8
qcomp = 0.65                      # «динамический битрейт» по сложности сцены (0.7 — экшен)
# no-cutree = 1                   # ВКЛЮЧАТЬ только при CRF < 15 (VCB-правило); на 16 cutree остаётся включённым
# qg-size = 16                    # polish-кнопка варианта (a); дефолт slow = 32
# ref = 5                         # polish-кнопка варианта (a); дефолт slow = 4
bframes = 8
rc-lookahead = 60
# keyint/min-keyint считает СТУДИЯ кодировщика по правилу «10 с / 1 с контента» из фактического выходного fps:
#   23.976 fps → 240/24;  59.94 fps (интерполяция) → 600/60. Ручной override — опционален.
no-open-gop = 1
tskip = 1
cbqpoffs = -2
crqpoffs = -2
# tune НЕ используется: --tune animation = psy-rd 0.4/aq 0.4/deblock 1:1 (усиление мыла) — JET: «НЕ для аниме».

[models]
dir = "models"
manifest = "models/manifest.toml"

[upload]                          # (v1.1) Пост-шаг выгрузки (§10) — НЕ стадия пайплайна.
# Включение: флаг --upload > ключ upload = true в профиле > false (по умолчанию выкл).

[upload.gofile]
token = ""                        # ОПЦИОНАЛЬНО: токен premium-аккаунта gofile (cookie accountToken).
                                  # Пусто = анонимная загрузка. Секрет в config.toml — норма владельца; env-переменных нет.
                                  # (v1.2) если задан — используется и для СКАЧИВАНИЙ (§11.2, приоритет 1).

[fetch]                           # (v1.2) Вход по URL/ссылкам (§11). Локальные -i от секции не зависят;
                                  # каталог загрузок фиксирован правилами -i (input/) — отдельного ключа dir нет.

[fetch.gofile]                    # (v1.2) Переопределения констант wt-схемы (§11.3); дефолты — константы из gofile-dl.
# wt_salt = "12af056dacea0b"      # соль wt.obf.js (последнее известное значение); ротируется gofile — обновлять при стойком error-notPremium
# user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
# language = "en-US"              # X-BL: захешированные UA/language ОБЯЗАНЫ совпадать с заголовками запроса

# ── Профили ──────────────────────────────────────────────────────────────
[profiles.full]
stages = ["descale", "deband", "upscale", "interpolate", "encode"]
upload = true                     # (v1.1) пример: профиль может включать gofile-выгрузку без CLI-флага

[profiles.upscale-only]
stages = ["upscale", "encode"]
descale_height = 1080             # unused in this profile; illustrates optional per-profile keys

[profiles.interp-only]
stages = ["interpolate", "encode"]

[profiles.interp-only.stages.interpolate]   # optional per-profile deep overrides (merged last before CLI)
target_fps = "60/1"
```

**Encoder contract (post-decision):** encode is CPU-only libx265 (owner decision 2026-10-27). `stages/encode.py` builds the ffmpeg arg list from `[stages.encode]` + `[stages.encode.x265_params]`, computes `keyint/min-keyint` from the pipeline's actual output fps (10 s / 1 s rule) with optional manual override, and emits BT.709 tags + `-profile:v main10`. The backend registry concept (`{name → arg-builder}`) remains an internal extension point; no second backend is configured.

---

## 5. Stage implementation notes

| Stage | Engine (fixed by research) | Key semantics | Notes |
|---|---|---|---|
| **descale** | `core.descale` (vapoursynth-descale plugin) — `Decompile` to native height | Kernel `bicubic`/`bilinear`, `src_left/src_top`; target height per-run (`--descale-height`) | Most "1080p" anime is natively 720/810/900p. Native-res *analysis* (getnative approach) is a documented manual/auxiliary step in `scripts/benchmark.py` territory, not a pipeline stage — the stage itself only downscales to the height given in config/CLI. |
| **deband** | vs-deband (`vsdeband`, HolyWu — same author as vsrife; wraps the f3kdb algorithm) | `radius`, `threshold[3]`, `grain[2]` from config | Runs on YUV at source res, before upscale — community convention (research §1: "deband before interpolation"). Optional; skipped in upscale-only/interp-only profiles unless selected. |
| **upscale** | spandrel `ModelLoader` → 2x_AnimeJaNai HD V3 Compact (SRVGG, 4.8 MB, sha256-pinned), fp16, inside `std.ModifyFrame` per-frame selector | x2 fixed by model; RGBH in/out; HD-only by design (SD sources are out of scope for v1) | Chosen over vs-mlrt TRT for verbatim pinned `.pth`, pip/uv manageability, and because TRT acceleration is already captured where it matters (interp). Escape hatch documented in §1. |
| **interpolate** | vsrife 5.7.0, model **4.25** (replaces ALL local pre-4.22 Practical-RIFE machinery) | `fps_num/fps_den` absolute target in ONE pass (23.976 → 59.94 needs ×2.5 — impossible by stacking x2, supported natively); `sc=True` scene passthrough; `misc.SCDetect` before rife (canonical per vsmlrt docs); `trt=True` with engine cache | Target fps is a rational string in config; CLI accepts `60` or `60000/1001`. Ordering (upscale→interp vs interp→upscale) is FIXED as upscale→interp in v1 (owner decision A1, resolved — see §8; community split documented in research §1). |
| **encode** | **CPU-only libx265 10-bit** (owner decision 2026-10-27, verified by research/04): `preset slow` (efficiency knee; slower/veryslow = polish, not size), `crf 16` default (14 collection / 18 floor), VCB anime params (`no-sao, deblock -1,-1, psy-rd 2.0, psy-rdoq 1.0, rdoq-level 2, aq-mode 3/0.8, qcomp 0.65, cbqpoffs/crqpoffs -2, bframes 8, rc-lookahead 60, closed GOP, tskip`), `main10` + BT.709 tags; keyint/min-keyint computed from output fps | ffmpeg subprocess: reads y4m on stdin, muxes `1:a?`+`1:s?`+`1:t?` copy + `-map_metadata 1` + `-map_chapters 1` from original input | Full per-flag rationale: research/05-encode-flags.md. `-tune animation` FORBIDDEN (psy-rd 0.4/aq 0.4 — anti-anime). |

**Interpolation model layout note:** vsrife downloads model 4.25 itself from its wrapper releases and caches TRT engines under `~/.cache/vsrife` — we do NOT pin those binaries in `models/`; `models/manifest.toml` only documents provenance (wrapper version ↔ model name).

---

## 6. Model layout & management

```
models/
├── manifest.toml
│   # [models.2x_AnimeJaNai_HD_V3_Compact]
│   # url = "https://github.com/the-database/mpv-AnimeJaNai/releases/download/3.0.0/2x_AnimeJaNai_HD_V3_ModelsOnly.zip"
│   # member = "2x_AnimeJaNai_HD_V3_Compact.pth"   # extracted from the release zip
│   # sha256 = "af7307eee19e5982a8014dd0e4650d3bde2e25aa78d2105a4bdfd947636e4c8f"
│   # license = "CC-BY-NC-SA-4.0"
│   #
│   # [models.rife_4_25]  (documentational)
│   # source = "vsrife 5.7.0 wrapper release (auto-download); NOT locally pinned"
├── upscale/
│   └── 2x_AnimeJaNai_HD_V3_Compact.pth      # produced by `anime-pipeline models fetch`; gitignored
└── interpolate/
    └── README.md                            # documents vsrife self-managed model + TRT cache paths
```

- `models fetch` downloads the AnimeJaNai release zip, extracts the pinned member, verifies sha256 **before** writing the final file (temp + rename), fails loudly on mismatch.
- `models verify` recomputes checksums; run before any unattended batch.
- The `.pth` is 4.8 MB — small enough to commit? **Decision: gitignore model binaries, keep manifest + fetch command.** Rationale: reproducibility via checksum, no binary blobs in history; the fetch is one command.

---

## 7. Migration plan (ordered, from current two scripts to the new structure)

1. **git init + .gitignore** (owner authorization for repo creation; models binaries ignored, `input/*` + `output/*` ignored, `uv.lock` committed). Initial commit = current state, so deletion steps stay revertible. Create `input/` and `output/` directories (empty, `.gitkeep`-only).
2. **uv scaffold:** `pyproject.toml` (PEP 621, `[project.scripts] anime-pipeline`), `src/anime_upscale/` skeleton, `uv add` deps: `vapoursynth`, `vsrife`, `spandrel`, `torch` (CUDA build), `numpy`, `tqdm`; dev group: `pytest`, `ruff` (`select=ALL` per owner gate), `ty`. No version pins in the manifest. **Venv discipline (owner rule, 2026-10-28): EVERY runtime and dev dependency must be declared in `pyproject.toml` and resolved into the project `.venv` — the uv venv is the ONLY runtime environment. System-installed Python packages (e.g. distro vapoursynth/torch) MUST NOT be relied upon, imported via system site-packages, sys.path hacks, or `--system`/`--system-site-packages` escapes, even when they are importable on this machine. All execution and gates go through `uv run …` inside the project; verify with `uv run python -c "import vapoursynth"` (etc.) resolving from `.venv`. System tools that are NOT Python packages (ffmpeg, git, nvidia-smi) stay system-level.**
3. **Config first:** write `config.toml` + `config.py` (load → validate → merge; includes the v1.1 `[hardware]` override and `[upload]` sections), with tests. Nothing else imports ffmpeg/VS yet.
4. **Stage port, interpolation first:** `stages/interpolate.py` + `stages/encode.py` + `pipeline.py` minimal path reproduces `interpolate_only.py` behavior (arbitrary target fps, scene guard, 10-bit HEVC, audio/subs copy) — now via vsrife 4.25 instead of pre-4.22 PyTorch RIFE. `interpolate_only.py`'s per-output-frame timestep logic is superseded by vsrife's `fps_num/fps_den` — do not port it.
5. **Upscale port:** `stages/upscale.py` reuses the spandrel loading pattern from `process_anime.py` (ModelLoader → fp16) inside the `std.ModifyFrame` selector; the hand-rolled numpy→tensor→numpy plumbing and the ffmpeg rawvideo decode pipe are **not** ported (replaced by VS graph + y4m).
6. **Descale + deband stages:** thin wrappers around `core.descale` and vsdeband with config passthrough (new capability, no legacy code to port).
7. **Hardware module (v1.1):** `hardware.py` per §9 — detection (torch CUDA properties, `nvidia-smi` cross-check), frozen `HardwareProfile`, override merge (`CLI > [hardware] > detected > default`), run-banner reporting; tests with mocked torch (frozen-ness, precedence, zero-GPU loud error).
8. **I/O layout + CLI semantics (v1.1):** `input/`/`output/` resolution rules, bare-name validation, `-o` default naming, collision warning, single-file rule for omitted `-i` — per §3, with tests.
9. **Fetch/download + gofile client (v1.2):** `fetch_input.py` + `gofile.py` per §11 — entry grammar (bare name / direct URL / gofile link), DoH-pinned streamed downloads with §10.4 progress, guest-token lifecycle with the state file (`~/.local/state/anime_upscale/gofile_guest.toml`, created on first gofile use), wt-scheme port with the source's constants (§11.3), contents resolution; unit tests with mocked HTTP (wt vectors, token precedence, state-file round-trip).
10. **Upload post-step (v1.1):** `upload.py` per §10 — DoH resolve+pin, `GET /servers`, chunked multipart `POST /uploadFile`, token cookie when configured, TTY/plain progress switch; gofile API mechanics delegated to `gofile.py` (v1.2). Failure paths mocked in tests, plus a one-shot live smoke (tiny anonymous file) that validates the endpoint shape and the token channel against the real API before first real use. **The live smoke extends (v1.2) to the download path: guest-token creation → state file written → `GET /contents/{id}` on a small public link + one file download into `input/` (verifies the live wt salt) → one direct-URL download.**
11. **Models:** `anime-pipeline models fetch` → drop the verified `.pth` into `models/upscale/`. 
12. **Cleanup of superseded model material:** delete `models/rife/train_log/` (pre-4.22 weights — the "bad for anime" era), `models/rife/model/`, `models/rife/demo/`, the extracted Practical-RIFE repo files under `models/rife/`, and `rife_master.zip`. **Decided (A3):** relocate `models/rife/gofile-dl/` to `~/projects/gofile-dl/` (owner's separate tool — moved, not deleted), then delete the now-empty `models/rife/`.
13. **Parity gate:** run interp-only and upscale-only through the new pipeline on a short sample; compare against legacy script outputs (visual spot-check + ffmpeg VMAF against the legacy encodes) **before** deleting the legacy scripts.
14. **Delete `process_anime.py` and `interpolate_only.py`.** `research/` stays at repo root untouched (memory canon and reports reference that path). **Decided (A6):** fold `vstrt_README.md` (saved vs-mlrt TRT readme) into `docs/vs-mlrt-trt-notes.md` and delete the root copy — the notes document the §1 TRT escape hatch and are worth keeping regardless of the spandrel-first choice.
15. **README** per owner standard (TOC, badges, Mermaid pipeline diagram, tree with inline comments, `| :-- |` tables), plus `scripts/benchmark.py` + first `benchmarks/RTX5080` entry to settle the missing 5080 numbers (and refine the §9.2 VRAM approximations).
16. **Quality gates:** `uv run ruff format . && uv run ruff check --fix --unsafe-fixes .` then `uv run ty check src/` then `uv run pytest` — each a standalone blocking command; zero errors/warnings in scope.

---

## 8. Owner decisions — all resolved (v1.1, 2026-10-27)

| # | Decision | Resolution |
|---|---|---|
| 1 | **Stage order: fixed vs configurable.** Community anime-lean is interpolate-at-source-res THEN upscale; the owner's chain spec (and current code) is upscale→interp. | **RESOLVED: fixed canonical order in v1** (`descale → deband → upscale → interpolate → encode`). A/B on hard scenes deferred until `benchmarks/` entries exist; the stage registry makes a `[pipeline] order` swap cheap later. |
| 2 | **Upscale TRT escape hatch now vs later.** vs-mlrt TRT would speed upscale (~1.45× vs ORT class) at the cost of manual plugin install + ONNX conversion. | **RESOLVED: later.** Ship spandrel fp16 (pip-managed, pinned artifact verbatim); decide from the first `benchmarks/` entry. Architecture isolates the swap to `stages/upscale.py`. `docs/vs-mlrt-trt-notes.md` (folded `vstrt_README.md`) keeps the TRT knowledge at hand. |
| 3 | **`models/rife/gofile-dl/` destination.** Stray unrelated Go project. | **RESOLVED: relocate to `~/projects/gofile-dl/` during migration step 11 — NOT deleted** (owner's separate tool). |
| 4 | **No-profile behavior.** Error listing profiles vs defaulting to `full`. | **RESOLVED: hard error + list of available profiles; no default** (explicitness prevents accidental multi-hour full-chain runs). |
| 5 | **Descale analysis depth.** getnative-style native-resolution detection as a helper script vs manual workflow (eyeball + `--descale-height`). | **RESOLVED: manual workflow for v1** (documented in README with the getnative reference); a `scripts/` analysis helper only if the owner asks — not part of the stage chain. |
| 6 | **`vstrt_README.md` fate.** Saved vs-mlrt TRT readme at repo root. | **RESOLVED: fold into `docs/vs-mlrt-trt-notes.md` during migration; root copy deleted.** |

### v1.2 owner decisions (2026-10-27)

| # | Decision | Status |
|---|---|---|
| 7 | **Batch failure policy** — behavior on a per-file runtime failure inside a multi-entry run (stage exception, encode failure, entry fetch failure). | **OPEN — two options presented in §11.5; recommendation: option 2 (continue-and-summarize with non-zero exit). Owner decides before implementation; either option is a driver-local change.** |

---

## 9. Hardware detection & auto-configuration (v1.1)

**Owner requirement:** the user supplies only uv + the project folder — every run detects the machine and auto-configures itself. No hardware questions, no manual tuning gates.

### 9.1 Detection module and the frozen profile

`src/anime_upscale/hardware.py` builds a **frozen `HardwareProfile`** (an immutable `@dataclass(frozen=True)`) once per run, right after config merge; stage builders and the ffmpeg builder consume it read-only. Detection is a capability probe, not a config source — the single source of truth remains `config.toml`.

| Field | Source | Notes |
|---|---|---|
| `gpus: tuple[GPUInfo, …]` (`index`, `name`, `vram_mb`) | `torch.cuda.device_count()` + `torch.cuda.get_device_properties(i)` (`.name`, `.total_memory`) | torch is already a hard dependency (vsrife/trt) — no new deps. `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits` is a **cross-check for diagnostics only** (torch reports 0 GPUs but the driver sees N → the error names the mismatch); it is not a second config/detection path. |
| `gpu_count` | derived | 0 is a valid result — see below. |
| `cpu_logical_cores` | `os.cpu_count()` | Feeds `encode_threads`. |
| `num_streams_per_gpu` | VRAM budget math (§9.2) or `[hardware] num_streams` override | Consumed by the frame-loop prefetch and stored for the TRT escape hatch (§9.2). |
| `encode_threads` | logical cores ÷ active encoders (§9.4) or override | Passed to the ffmpeg/libx265 builder. |
| `precision` | constant `fp16` on any RTX-class GPU | Research corpus: fp16 is the ecosystem default; no auto-fp32 path (YAGNI). |

**Zero-GPU behavior:** if the selected stage subset contains `upscale` or `interpolate` and `gpu_count == 0` → loud Russian error naming the missing prerequisite and suggesting either the driver/CUDA check or an encode-only `--stages` subset. A GPU-less machine may still run `--stages descale,deband,encode`.

**Override precedence** (same discipline as the rest of the config, merged in `config.py`): CLI flag > `[hardware]` section in config.toml > detected value > built-in default. The frozen profile is built **after** the merge, so builders see exactly one object and never re-probe.

### 9.2 `num_streams` — VRAM budget math (assumptions stated)

Per-GPU anchors from the research corpus (vs-mlrt wiki, RTX 4090, TRT fp16 1080p): RIFE v4.7+ ≈ **0.75 GB at 1 stream → ~2.0 GB at 3 → 2.6 GB at 5**. The 2x_AnimeJaNai HD V3 **Compact** (SRVGG, 4.8 MB) is compact-class: **≈1–2.5 GB at 1080p** including workspace — an approximation pending the first `benchmarks/RTX5080` entry (flagged; the budget formula reads the reserve from a single constant that the benchmark entry will refine). Per-process CUDA context + TRT engine + Torch-TensorRT runtime ≈ **0.5–1 GB**.

Budget formula (per GPU): `streams = clamp(floor((0.8 × VRAM − context − upscale_reserve − slack) / rife_per_stream), 1, 4)` with `slack ≈ 1 GB`. On the 16 GB target (5080) this evaluates to **2 streams/GPU** — the same value the wiki examples use; the formula exists so a 24 GB card bumps to 3 without a config edit. Honest scope note: vsrife's `trt=True` exposes **no stream knob** [UNVERIFIED whether Torch-TensorRT's internal stream count is tunable from Python]; in v1 the value drives the VS frame-loop `prefetch` depth (concurrent frame pipeline) and is stored for the vsmlrt `BackendV2.TRT(num_streams=…)` escape hatch (§1) so the swap inherits it without redesign.

### 9.3 Multi-GPU fan-out — v1 strategy

**Requirement:** more than one GPU → the pipeline uses **ALL** GPUs at maximum by default (config `[hardware] gpus` may pin a subset). **Chosen strategy: one worker process per GPU** (segment fan-out), not in-graph device pinning.

Why not in-graph (the honest reason): vsrife/torch device selection is process-global state (`torch.set_device` / implicit `cuda:0`), and VS evaluates a single graph's frames concurrently through prefetch — per-branch device pinning inside one process would race global state. Process-per-GPU gives each worker exactly one visible device via `CUDA_VISIBLE_DEVICES` set at spawn (a device-pinning mechanism of the driver, not configuration delivery — the owner's no-env-vars rule governs user-facing config, which stays exclusively `config.toml` + CLI).

Segment mechanics (only when `gpu_count > 1`):

1. The driver splits the source frame range into `gpu_count` contiguous chunks and spawns one worker per GPU.
2. Each worker builds the **identical** graph over its chunk, expanded by an **overlap** of `max(deband_radius, 1)` input frames on each side. Overlap justification: RIFE and `SCDetect` are pair-local (need 1 neighboring input frame at a join); f3kdb deband has a temporal radius (`radius` config) and therefore needs `radius` context frames; descale/upscale/resize are per-frame spatial (no context).
3. Timesteps are computed from **global** output frame indices, never per-chunk local ones — this makes seam frames bit-identical to a single-GPU run (same rational `t` per output frame, same scene-passthrough decisions, since scene detection is also pair-local).
4. Each worker pipes into its own ffmpeg producing a **video-only segment**; a final driver-side ffmpeg pass concatenates segments (concat demuxer, stream copy) and muxes audio/subs/attachments/metadata from the source exactly per the research/05 §5 assembly — the mux shape is unchanged, one extra stream-copy pass (seconds).
5. Failure semantics: a worker dying = loud error naming the GPU and chunk; no silent segment substitution.

**Single GPU (the 5080 target):** `gpu_count == 1` → exactly the v1.0 single-process design; the segment path does not activate. TRT engine caches are keyed by GPU name, so same-model GPUs share one cache after the first build (workers may briefly serialize on first-run engine builds — accepted; a pre-pass is a documented optimization, not a requirement).

### 9.4 CPU encode threads

Single encoder: libx265 gets the full detected logical core count (x265 auto-splits frame/WPP threads; research/05 §3.2 — over-allocation doesn't help, under-allocation is the only mistake worth preventing). Multi-GPU fan-out: each segment encoder gets `floor(logical_cores / gpu_count)`; the final concat/mux pass is stream-copy and needs no explicit thread budget. `[hardware] encode_threads` overrides both.

### 9.5 Target-fps sanity

Before any GPU/model work, the driver reads the source fps (BestSource/ffprobe) and validates the interpolation target: `target ≤ source` → loud Russian error (vsrife would reject it anyway — we fail fast with an actionable message instead of a Torch traceback); target not a positive rational → validation error. This is a sanity gate, not a new policy: vsrife's own contract, surfaced early.

### 9.6 Run banner (Russian, colored, `--no-color`/`--no-tty` aware)

```text
◈ Оборудование: NVIDIA GeForce RTX 5080 ×1, 16 ГБ VRAM · CPU 16 ядер
◈ Стадии: descale → deband → upscale → interpolate → encode
◈ Инференс: RIFE 2 потока/GPU, fp16 · Кодирование: 16 потоков
```

Multi-GPU variant names every GPU (`RTX 5080 ×2 (16+16 ГБ) … 2 потока/GPU, сегменты: 2`). The banner is the only place detection results are shown by default; `--no-color` strips ANSI, `--no-tty` reduces it to plain text lines.

---

## 10. Optional gofile.io upload post-step (v1.1)

**Owner requirement:** after ANY completed pipeline run, optionally upload the output file to gofile.io and print the link. Design: a **post-step module** (`src/anime_upscale/upload.py`), **not a pipeline stage** — it is invisible to the stage registry, consumes only the finished output path, and cannot influence graph assembly. (v1.2: the gofile API mechanics below — DoH pinning, `/servers`, token handling, progress policy — live in the shared `gofile.py` client, §11.2; this section's flow semantics are unchanged.)

**Trigger & isolation:** enabled by `--upload` (CLI) > `upload = true` profile key (e.g. `[profiles.full]`) > default off. The upload runs only when the pipeline exited 0 and the output file exists. Any upload failure = loud Russian error with a dedicated non-zero exit code; **the output file is untouched** (it is only read), and no partial state is left behind. One attempt in v1 — no retry framework (a failed upload can simply be re-run with the same command once the network issue is fixed; inventing backoff policies is scope).

### 10.1 DNS: strict Google DoH, pinned, no fallback

- **Every** gofile hostname — `api.gofile.io` first, then the chosen store host (`store1.gofile.io`, `store-eu-par-6.gofile.io`, … — names are dynamic) — is resolved **only** via the Google DoH JSON API: `GET https://dns.google/resolve?name=<host>&type=A`. Response `Status == 0` and ≥1 A record required. (Live-verified 2026-10-27: `api.gofile.io` → 3 A records; `store1.gofile.io` → 1 record.)
- **No system-resolver fallback, ever.** DoH failure (network error, non-2xx, `Status != 0`, empty answer) = loud error, upload not attempted. This is an anti-spoofing stance (the system resolver path is never trusted for gofile hosts — poisoned/intercepted DNS cannot redirect the upload), accepted at the cost of `dns.google` being a hard dependency of the upload step (owner's explicit requirement; acknowledged RF-network trade-off).
- **Pinning:** connections dial the resolved IP directly for the lifetime of one upload session. TLS is **not** weakened: SNI, the `Host` header, and certificate verification all use the real hostname (connect-by-IP + `server_hostname=<hostname>` is the standard TLS pattern). Pins expire with the session — the next upload re-resolves.
- **Scope (v1.2):** this policy is stated ONCE, here, and governs EVERY remote host the tool fetches — the gofile API/store hosts of this section, the gofile contents/download hosts of §11, and direct-URL download hosts (§11.1). `fetch_input.py` and `gofile.py` reference this section; no module carries its own DNS logic.

### 10.2 Upload flow

1. DoH-resolve `api.gofile.io` → pin → `GET https://api.gofile.io/servers` → `{"status":"ok","data":{"servers":[{"name":"store1","zone":"eu"},…]}}` — **verified live 2026-10-27** (server names like `store1`, `store-eu-par-N`, `store-na-phx-N`). Pick the first server (zone-awareness is a documented future refinement, not v1 scope).
2. DoH-resolve `<server>.gofile.io` → pin.
3. `POST https://<server>.gofile.io/uploadfile` — **multipart**, field `file`, streamed in chunks (constant chunk size; the stream drives the progress callback). Endpoint shape cross-checked against a primary live client implementation (Alyetama/Gofile: `POST https://{server}.gofile.io/uploadFile`, multipart `file`, token as a request field) — **UNVERIFIED detail: the exact path spelling (`/uploadfile` vs `/uploadFile`) and verb; migration step 9's one-shot live smoke validates it before first real use.**
4. `status == "ok"` → print the link in Russian: `🔗 Ссылка: <data.downloadPage>` (e.g. `https://gofile.io/d/XXXXXX`). Any other `status` or HTTP error → loud terminal error naming the server status string verbatim (diagnosable, not swallowed).

### 10.3 Token (optional, never required)

- No token → **anonymous upload** (the default path; the flow above works tokenless — the live `GET /servers` probe is tokenless by nature, and anonymous uploads are gofile's public behavior).
- `[upload.gofile] token = "…"` (premium account token) → sent as the **`accountToken` cookie** on the upload request. Provenance for the name: gofile's own site JS canonicalizes the term `accountToken` (independently corroborated by the owner's `gofile-dl` research notes, where the site's anti-bot formula hashes `…::accountToken::…`). **UNVERIFIED detail:** whether the current API prefers the cookie over a `token` query/form field (both shapes exist in live client code) — same step-9 smoke settles the channel; the plan's contract is only "token present → authenticated upload; token absent → anonymous".

### 10.4 Progress reporting (TTY rule)

- **Animated progress bar** (chunk-level) only when stdout/stderr is a TTY **and** not disabled by `--no-color` or `--no-tty`.
- Otherwise: plain percentage lines, machine-readable, zero ANSI, emitted at ≤10% intervals: `Загрузка: 40% (512/1280 МиБ)`. The same rule governs the §9.6 banner and pipeline progress — one TTY policy for the whole CLI.

### 10.5 Known risk (honest)

gofile applies client-side hardening on some endpoints — the owner's `gofile-dl` documents a dynamic `X-Website-Token` scheme (`sha256(userAgent::language::accountToken::timeWindow::salt)`) enforced on `/contents/{id}` download paths, with premium tokens bypassing it. **UNVERIFIED whether upload endpoints are (or become) subject to such hardening.** v1 policy: no circumvention logic; any 4xx/`status != "ok"` from the upload path is a loud terminal error with the server's status string shown. If gofile hardens uploads later, that is a new requirement for the owner to scope — not something this plan silently papers over.

---

## 11. Input ingestion, gofile download client, batch driver (v1.2)

**Owner requirements (2026-10-27):** (a) a **list of files** can be given to the pipeline and they execute one by one; (b) **direct links and gofile links** are downloaded into `input/` preserving original names, then processed; (c) the gofile download client uses a **guest token with an explicit save/reuse lifecycle** — frequent guest regeneration is rate-limited by gofile (owner's explicit reason for a saved-credential state file). Sequential batch only; parallel batch is out of v1.2 scope (the §9 fan-out machinery is per-file and untouched).

Design shape: one new module for input-entry resolution (`fetch_input.py`, §11.1), one new module for the gofile API (`gofile.py`, §11.2–11.4, shared by `fetch_input.py` and `upload.py`), and a thin batch loop in the driver (§11.5). The v1.1 single-file semantics remain the inner loop — a batch is N sequential runs of exactly that loop, nothing more.

### 11.1 Entry types and resolution (`fetch_input.py`)

`-i` accepts one or more ENTRIES (argparse append); each entry resolves to exactly one file in `input/` before any GPU/model work starts for that file.

| Entry shape | Rule |
|---|---|
| Bare name (`ep01.mkv`) | v1.1 rule unchanged: bare file name only (no `/`, `\`, `..`), resolved to `<project>/input/ep01.mkv`, which **must already exist** — loud Russian error if not. |
| `http(s)://…` where host is `gofile.io` / `www.gofile.io` and path is `/d/<id>` | gofile entry → §11.4. |
| gofile URL with any other path shape (store links, `/upload`, bare-id forms) | **Loud error** — those are internal/resolved links, not share links. |
| Any other `http(s)://…` | Direct download, rules below. |

Direct-URL download rules:

- **Filename precedence (decided):** (1) `Content-Disposition` header — `filename*` (RFC 5987) first, then plain `filename` (RFC 6266); (2) fallback: last URL path segment, percent-decoded, query string stripped; (3) neither derivable (e.g. empty/trailing-slash path, unparseable header) → **loud Russian error naming the URL**. Rationale: a server-provided name is authoritative; the path segment covers servers that omit the header.
- **Streamed chunked GET** into `input/<name>` via a `<name>.part` temp + atomic rename; chunked progress under the §10.4 TTY/plain policy (`Загрузка: 40% (512/1280 МиБ)` plain lines); loud failure semantics — the `.part` is deleted and the error is terminal for that entry. No Range-resume in v1.2 (a re-run re-downloads from zero; resume is scope).
- **Idempotence (owner requirement):** `input/<name>` already exists → download **skipped** with a printed note (`⏭ Уже в input/: <name> — загрузка пропущена`). Re-invoking a batch after an interruption does not re-download. The one hazard this creates — a *different* remote file that happens to share the name is treated as already present — is closed **within** an invocation by §11.6 and documented here as an accepted cross-run limitation of the owner's skip rule (remedy: rename/delete the old file; the skip note makes it visible).
- **DoH:** the host is resolved and pinned per §10.1 (scope extended in v1.2 to all fetched hosts). No system-resolver fallback.
- gofile entries download into `input/` with the fileName from the contents API (§11.4) — same idempotence, progress, `.part`, and DoH rules; no separate policy.

### 11.2 gofile API client (`gofile.py`) — guest-token lifecycle

One shared client module; `upload.py` (§10) and `fetch_input.py` (§11.1/§11.4) both consume it — DoH pinning, servers, token handling, wt computation, and file transfer are written once. (The §10 upload flow itself is unchanged.)

**Token precedence (owner-specified, checked in order):**

1. `[upload.gofile] token` present (account token) → used **for downloads too**. A premium token also bypasses the wt hardening entirely (FIX_DOCUMENTATION.md:65); the client still computes and sends the wt header unconditionally — the source tool sends it on every `/contents` request regardless of tier (run.py:456-466).
2. No config token → **reuse the SAVED guest token** from the state file.
3. No state file → `POST https://api.gofile.io/accounts` (run.py:414; token at `data.token`, run.py:419) → **save once** to the state file. Regeneration is deliberately rare: guest account creation is the rate-limited operation.

**State file — NOT `config.toml`, NOT env vars:**

- Location: `~/.local/state/anime_upscale/gofile_guest.toml` (`$XDG_STATE_HOME` honored when set — the platform convention is *read* from the environment, not used as configuration delivery; same distinction as §9.3's `CUDA_VISIBLE_DEVICES` note).
- Format:

  ```toml
  token = "…"                          # guest accountToken
  created = "2026-10-27T12:00:00Z"     # RFC 3339, diagnostics only
  ```

- Why here: `config.toml` is **user intent** and the single source of configuration truth (owner rule); a fetched credential is **machine-local runtime state**. XDG_STATE_HOME is the standard home for "persists between runs, safely deletable" — deleting the file simply means the next run creates a fresh guest. `~/.cache` is wrong (auto-cleaners would silently delete the credential, triggering the rate-limited regeneration path this design exists to avoid); env vars are banned outright.
- Unreadable/corrupt state file → **loud Russian error naming the path and the remedy** (delete it). No silent regeneration: creating a guest is the rate-limited operation, so it never happens as a side effect of a parse failure.

### 11.3 X-Website-Token (wt) scheme — ported VERIFIED-FROM-SOURCE

Provenance note: the task brief called this the owner's "Go implementation"; the actual `gofile-dl` source is a **Python** project (`run.py`). Same rule applies — the owner's source wins over any assumption. Every fact below carries a file:line citation from that source; anything not derivable from it is marked UNVERIFIED. Background: gofile moved the `X-Website-Token` from a static value published in `gofile.io/dist/js/config.js` (`appdata.wt = "…"`) — now a **decoy whose use returns `error-notPremium`** — to a token generated client-side in `wt.obf.js` (FIX_DOCUMENTATION.md:13-21; investigate_api.py:39-47).

| Fact | Value | Source (gofile-dl) |
|---|---|---|
| Formula | `sha256("{userAgent}::{language}::{accountToken}::{window}::{salt}")` → 64-hex lowercase | run.py:88 (comment), run.py:121-144 (`generate_website_token`; raw string :143, hexdigest :144) — **VERIFIED-FROM-SOURCE** |
| Time window | `window = floor(unix_time / 14400)` — rotating 4-hour bucket | run.py:107 (`WT_WINDOW_SECONDS = 14400`), run.py:142 — **VERIFIED-FROM-SOURCE** |
| Salt | `12af056dacea0b` (last-known constant embedded in the source; FIX_DOCUMENTATION.md:28 shows an older example `9844d94d963d30` — the salt **rotates**, see recovery below) | run.py:106 — **VERIFIED-FROM-SOURCE**; live currency **UNVERIFIED** (migration smoke checks) |
| User-Agent | `Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36` | run.py:98-102 — **VERIFIED-FROM-SOURCE** |
| Language | `en-US`, sent as the `X-BL` header | run.py:103, run.py:461 — **VERIFIED-FROM-SOURCE** |
| Consistency requirement | The hashed UA and language MUST equal the `User-Agent` / `X-BL` request headers — the server recomputes the hash from them | run.py:91-93; FIX_DOCUMENTATION.md:31-33 — **VERIFIED-FROM-SOURCE** |
| Contents headers | `Authorization: Bearer <token>`, `X-Website-Token`, `X-BL`, `User-Agent`, `Accept: */*`, `Origin: https://gofile.io`, `Referer: https://gofile.io/` | run.py:456-466 — **VERIFIED-FROM-SOURCE** |
| Contents query params | `contentFilter=""`, `page=1`, `pageSize=1000`, `sortField=createTime`, `sortDirection=-1` | run.py:112-118 — **VERIFIED-FROM-SOURCE** |
| Endpoint shapes | Guest create: `POST https://api.gofile.io/accounts`; contents: `GET https://api.gofile.io/contents/{id}` | run.py:414, run.py:594 — **VERIFIED-FROM-SOURCE** |
| Premium bypass | A premium token bypasses the whole wt mechanism | FIX_DOCUMENTATION.md:65 — **VERIFIED-FROM-SOURCE** |
| Token channel per call | `Authorization: Bearer` + wt header on `/contents`; **`Cookie: accountToken=<token>`** on the file-download GET | run.py:459, run.py:965 — **VERIFIED-FROM-SOURCE** |

**Port policy (what we adopt, what we don't):**

- The formula, constants, headers, and query params are ported **verbatim**. UA / language / salt become `[fetch.gofile]` config overrides (§4) with these exact defaults — the config-not-env discipline; the source tool's env-var overrides (run.py:95-106) are deliberately **not** ported.
- **Retry semantics ported minimally** from the source's `fetch_contents` (run.py:595-667): `error-rateLimit` → retry up to **3 total attempts** with linear backoff `3 s × attempt` (source: run.py:631-639); `error-notPremium` despite a correct-formula token → retry **once** with `window − 1` (clock-boundary guard at 4-hour edges; source: run.py:596-598, 641-647), then a **loud error whose message names the recovery** — update `[fetch.gofile] wt_salt` (FIX_DOCUMENTATION.md:59-65). `error-notFound` is terminal (run.py:662-664). These two retries are responses to documented gofile *server* behavior, not a generic retry framework; everything else is single-attempt loud failure (§10 stance).
- The source's **web-scraping fallback** (run.py:478-571) is **not ported** — it is that tool's fragile last resort; our contract is loud failure with a diagnosable message.
- **Known edge-block risk (documented, not solved in v1.2):** api.gofile.io's edge resets connections from datacenter/VPN IPs and non-browser TLS stacks (run.py:22-49); plain `requests` is fine from a normal residential connection (run.py:34-35) — the owner's machine qualifies. A connection reset = loud error quoting that diagnosis; optional `curl_cffi` TLS impersonation is the documented escape hatch (run.py:36-44), **not** a v1.2 dependency.

### 11.4 Contents resolution and file download

- Entry parsing: `gofile\.io/d/([^/?#\s]+)` — the source's exact ID-extraction shape (run.py:369). **Bare-ID entries are not accepted in v1.2** (the share-link form is the owner's stated input; accepting bare ids later is a one-line addition).
- `GET https://api.gofile.io/contents/{id}` with §11.3 headers + query params. Success = `status == "ok"` (source response handling: run.py:627-629).
- **Single-file links only in v1.2:** `type == "file"` → the entry's original name is `data.name` (run.py:887 — gofile's fileName field, preserved verbatim into `input/`) and the direct download URL is `data.link` (run.py:900; treated as opaque). `type == "folder"` → **loud Russian error**: a folder is multi-file content — the owner's separate `gofile-dl` tool handles folders, and multi-file ingestion here is exactly what the batch list (§11.5) is for; recursive folder download is not pipeline scope. Password-protected content (`passwordStatus != "passwordOk"`, run.py:726) → loud error; password support is out of v1.2 scope.
- **File download:** `GET <link>` with `Cookie: accountToken=<token>`, the §11.3 User-Agent, and `Referer: https://gofile.io/` (header set per run.py:963-969); streamed chunked into `input/<name>.part` → atomic rename (temp-then-rename pattern: run.py:952, 979, 1017); §10.4 progress; DoH-pinned (§10.1 scope); loud failure deletes the `.part`. **Documented deviation from source:** the source supports Range-resume from a `.part` (run.py:961, 968) — not ported; v1.2 re-downloads from zero on failure.
- **UNVERIFIED (honest list):** the *current live* salt value (the embedded constant may have rotated since the source's last update — the migration smoke download verifies it end-to-end); the exact host/shape of `data.link` (opaque in the source); whether gofile extends wt enforcement to further endpoints over time (same standing risk class as §10.5).

### 11.5 Batch driver

Semantics (owner requirement: «подача списка файлов на пайплайн, чтобы они выполнялись по очереди»):

- **CLI order is execution order.** Exactly one full pipeline run (selected stages + encode) per entry, strictly sequential. `--upload` remains the per-file post-step: each completed output uploads before the next input starts. No new concurrency anywhere — the §9 per-GPU machinery operates within a single file's run, unchanged.
- **Per-file progress identity:** before each per-file run banner the driver prints `◈ Файл 2/5: ep02.mkv` (resolved name; `--no-color`/`--no-tty` aware, §10.4 policy). The §9.6 banner itself is unchanged and repeats per file.
- **End of batch:** a Russian summary — one line per entry (`✓ ep01.mkv` / `✗ ep03.mkv: <причина>`) plus `Итог: 4/5 успешно, 1 с ошибкой` — then the process exit code per the failure policy.
- **Pre-flight vs per-file failures:** config/profile validation, hardware detection, model/manifest verification, and the §11.6 collision checks run **once before entry 1** and abort the whole batch on failure — they are not per-file conditions and are not governed by the failure policy. The policy below governs only **per-file runtime failures**: a stage exception, an encode failure, or a fetch failure for that entry.
- **OPEN DECISION (owner) — batch failure policy:**

| Option | Behavior | Trade-off |
|---|---|---|
| 1. Stop on first error | First per-file failure aborts the batch; remaining entries untouched; exit code = that failure. | Simple; guarantees a human looks before more GPU-hours are spent. But one corrupt episode at 03:00 idles the rest of the night's queue. |
| 2. Continue and summarize **(recommended)** | All entries attempt; failures recorded and named in the summary; **exit 0 only if every entry reached `ok`** (pipeline + optional upload), otherwise non-zero. | Unattended runs deliver maximum value overnight; the summary names each failed entry, so a re-run is just its `-i` entries. Cost: a *systemic* per-file failure (e.g. a codec BestSource cannot demux) repeats per entry — mitigated by the pre-flight gate above (which catches config/hardware/model classes) and by Ctrl-C remaining available. |

  Recommendation rationale: per-file runs are long (GPU-hours), the owner's stated use case (a list executed one by one) reads as unattended, and option 2's summary + non-zero exit preserves the fail-loud contract — nothing is swallowed, it is *deferred to the summary*. Option 1 is a one-line driver change if the owner prefers it.

### 11.6 Naming and collision rules (batch)

- **Within one invocation**, a *download* entry (direct URL or gofile) whose resolved filename already matches an earlier entry's resolved name — bare name or previously downloaded — is stored as `<stem>-2<ext>`, `<stem>-3<ext>`, … with a **loud printed note** (never silent). Two download entries colliding with each other disambiguate the same way. Rationale: the rename is visible in the banner and both outputs survive (`output/foo.mkv`, `output/foo-2.mkv` — the v1.1 stem-to-stem output rule applies to the disambiguated input stem); rejecting the whole batch over a filename coincidence would punish the realistic case (two gofile links to same-named files). **Bare-name entries are never renamed** — a bare name is an explicit reference to an existing file; if it is missing, that is the v1.1 loud error.
- **Literal duplicate entries** (identical strings) → deduplicated to one with a printed warning.
- **Output side:** the v1.1 overwrite-with-warning rule (§3) applies per file; the batch summary (§11.5) lists every overwritten output.
- **Cross-run:** the §11.1 idempotent-skip rule applies — re-running the same batch skips already-downloaded URL entries and processes them from `input/` as-is (owner requirement); bare-name entries behave identically to v1.1.

---

## 12. Cross-platform requirements (Linux + Windows) — v1.3, owner rule 2026-10-28

1. **Paths:** all filesystem handling via `pathlib`; state/cache locations platform-aware — state file resolves to `%LOCALAPPDATA%\anime_upscale\state\gofile_guest.toml` on Windows, `$XDG_STATE_HOME`/`~/.local/state/anime_upscale/` on Linux (one resolution function, platform branch, same TOML format); TRT cache dirs likewise. No hardcoded `/`-prefixed or `~`-only paths in code.
2. **Processes:** multi-GPU workers and all subprocess plumbing must be **spawn-safe** (`multiprocessing.get_context("spawn")` semantics, `if __name__ == "__main__":` guards, picklable-only worker args) — Windows has no fork, and spawn-safety must not depend on Linux accident; worker GPU pinning via `CUDA_VISIBLE_DEVICES` (works on both platforms).
3. **External tools discovery:** ffmpeg/ffprobe resolved from PATH first, with an optional `[tools] ffmpeg = "path"` / `[tools] ffprobe = "path"` config override (single resolution helper; loud error naming the missing tool and the override key when not found). nvidia-smi is diagnostic-only — its absence never blocks (torch is the detection source). README documents per-OS prerequisites (ffmpeg in PATH, NVIDIA driver, uv install).
4. **Console:** Russian colored TTY on Windows — enable VT processing on win32 before first colored output (stdlib `os.system("")` trick or ctypes `SetConsoleMode`, no mandatory colorama dependency); `--no-color`/`--no-tty` behave identically on both OSes; `sys.stdout.isatty()` is the TTY test.
5. **Wheels reality:** vapoursynth/vsrife/spandrel/torch-tensorrt Windows wheels exist for the current stack — if any wheel is missing for the running platform at install time, that is a loud BLOCKED error (never a silent CPU/system fallback); record the verified platform matrix in the report (Linux verified live; Windows wheels verified via PyPI metadata, not live-installed).
6. **Subprocess/text:** all pipes binary-mode (y4m raw), text decoding only where explicit; no shell=True anywhere; argparse/progress `\r` handling works on both consoles.

---

## 13. Notebook deployment surface (v1.x) — addendum 2026-11-06

Kaggle Notebooks and Google Colab are an **additional runtime mode** for the pipeline, not a new primary target: the Windows Server RTX 5080 deployment remains primary and untouched, and no primary-runtime behavior, config key, or manifest entry changes.

- **Single reference:** the compatibility matrix (project × Kaggle × Colab, with per-row source + date stamps), design notes, DoD, local-validation contract, risk register, and the fixed owner decisions live in [notebooks.md](notebooks.md). This section only fixes the architectural position.
- **Detection, not configuration:** a read-only `runtime_env.py` identifies `local`/`kaggle`/`colab` and resolves default project roots; the app still reads settings exclusively from config.toml (no env-var configuration — owner invariant §0 holds). T4 (Turing sm_75) is already classified correctly by [hardware.py](../src/anime_upscale/hardware.py) (`rgbh` FP16 + `recommended_trt=True`); the degrade ladder for notebooks is TRT → PyTorch CUDA FP16 RGBH → loud error, reusing the existing [stages/interpolate.py](../src/anime_upscale/stages/interpolate.py) TRT-off path.
- **Adaptation lives in bootstrap cells, not the repo:** the committed manifest stays cu130; the Colab driver probe + cu126/`<2.15` torch-index swap mutates only the ephemeral clone ([pyproject.toml](../pyproject.toml) untouched). No Docker, no SSH/remote-control patterns, no env-var app config — platform-ToS and project anti-goals both preserved (§3 anti-goals list).
- **Fixed owner decisions (2026-11-06):** encode preset `medium` everywhere (honest multi-session session-budget math); gofile guest token stays per-session (no `state_file` config key); input = gofile URL with Kaggle Datasets documented as a symlink recipe; live platform runs are owner acceptance steps.

---

## Provenance

### Provenance (v1.1 addendum)

v1.1 skills + memory-read list: see the end of the "v1.1 changes (2026-10-27)" changelog section at the top of this document. New external facts introduced by v1.1 (gofile endpoints, DoH JSON API) were live-probed 2026-10-27 and are labeled VERIFIED/UNVERIFIED inline in §10; no memory files were written.

### Provenance (v1.2 addendum)

v1.2 skills + memory-read list: see the end of the "v1.2 changes (2026-10-27)" changelog section. **No new external facts were asserted beyond the owner's `gofile-dl` source** (`models/rife/gofile-dl/`: run.py, errors.py, FIX_DOCUMENTATION.md, README.md, PREMIUM_SUPPORT.md, investigate_api.py, config.yml — all read; wt/API facts carry file:line citations in §11.3, marked VERIFIED-FROM-SOURCE or UNVERIFIED). Note: the brief described the source as a Go implementation; it is Python — the owner's source was used regardless, per the same precedence rule. Exa was not used (credits-exhausted per memory 2026-10-22; no external lookups were needed). No memory files were written.

### Skills used
`brainstorming` (design intent + goals/anti-goals framing; interactive phase pre-settled by the controller's fully-specified brief, per SUBAGENT-STOP), `grilling` (design-tree discipline: every section resolves a frontier decision explicitly; unsettled branches isolated in §8), `build-with-exa` (external-fact discipline: all engine claims sourced from the project's verified research corpus; no new external facts asserted beyond it), `modern-python` (uv PEP 621 layout, dependency-groups, src layout, ruff/ty gates, no manifest pinning), `karpathy-guidelines` (no invented scope: exactly the five stages + models/benchmarks/tests required by the brief; minimal surface).

### Memory files read (mandate compliance — every file, none skipped)
- `~/.config/opencode/MEMORY.md` — **in full**.
- **topics/ (19/19):** `stack-and-tools.md`, `workflow.md`, `workflow-git-pr.md`, `communication-behavior.md`, `tech-gotchas.md`, `environment-infra.md`, `code-gotchas.md`, `remnawave.md`, `job-search.md`, `harness-server.md`, `hh-ranking-ats-research.md`, `2026-10-09-telegram-serverless.md`, `2026-10-11-novum-organum-report-plan.md`, `2026-10-12-slides-novum-organum-plan.md`, `2026-10-14-opencode-v2-storage.md`, `2026-10-15-binary-re-landmines.md`, `2026-10-15-vision-ocr-scans.md`, `2026-10-21-stepin-book-contract.md`, `2026-10-25-locke-pirogov-referaty.md`.
- **projects/ (31/31):** `anime_upscale.md` (in full — project canon), `4q4r-profile.md`, `WMS.md`, `anicli-go.md`, `anicli-py.md`, `bos-pro-bugs.md`, `cindex.md`, `ert.md`, `firefox-vpn-client.md`, `freeai-decompile.md`, `freepackages.md`, `habrparser.md`, `hhauto.md`, `home-global.md`, `interview-kb.md`, `iris-proposal.md`, `kolosyug.md`, `labboardvent.md`, `labs-docker-stands.md`, `labs.md`, `remnavawe_ansible.md`, `secureway_bot.md`, `shikimori-japanese-theme.md`, `stepik-whitehacker-course.md`, `teleparse.md`, `traffic_mimicry.md`, `tspu-research-2026.md`, `vibecoding-course.md`, `wauploader.md`, `wms.md`, `xray-raw.md` (coverage reads: giants via head-section pass confirming zero task-relevant content beyond their index entries).

### Facts carried into this plan from the fixed research corpus
`research/2026-10-27-consolidated-research.md` (§0 audit, §1 interpolation verdict, §2 upscaling verdict, §4 exact stack), `research/03-vapoursynth-runtime.md` §9 skeletons (Python-embedded + vsrife gold path), `research/01-interpolation.md` (skim: version history, arbitrary-t semantics, wrapper map), plus project sources `process_anime.py`, `interpolate_only.py`. No memory files were written (facts delivered to controller per mandate).
