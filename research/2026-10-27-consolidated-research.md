# Anime Upscale & Interpolation — Consolidated Research (2026-10-27)

Consolidated controller report from three parallel research agents. Full deep reports live alongside this file:

- `01-interpolation.md` — neural frame interpolation sphere (RIFE/GMFSS/GIMM/IFRNet/AMT/FILM/CAIN/DAIN, wrappers, benchmarks, community signals)
- `02-upscaling.md` — anime upscaling sphere (Real-ESRGAN, Anime4K, Real-CUGAN, OpenModelDB, AnimeJaNai, SOTA archs, archival pipelines)
- `03-vapoursynth-runtime.md` — VapourSynth runtime + vs-mlrt + wrapper semantics + performance reality + skeletons

Provenance: every claim in the sub-reports carries `[V]` (verified via GitHub/PyPI/Steam API or primary source, 2026-10-27), `[O]`/`[OP]` (community opinion with quoted source) or `[UNVERIFIED]`. Exa returned HTTP 402 during the run; agents used the documented fallback chain (GitHub REST, PyPI JSON, raw.githubusercontent, DDG HTML, proxy curl, cloakbrowser).

---

## 0. Local project audit (controller, pre-research)

- `process_anime.py` — ffmpeg rawvideo pipe → spandrel `ModelLoader` upscale (`models/upscale/model.pth`, **currently missing**) → optional RIFE interpolation **on upscaled tensors** → ffmpeg `hevc_nvenc p7 hq` HEVC **10-bit** (p010le, CQ16). Interpolation midpoint-only (x2), scene guard `mean|Δ| > 0.18`, 64px padding.
- `interpolate_only.py` — RIFE-only, **arbitrary target FPS already implemented** (per-output-frame timestep computation), default 60.
- `models/rife/train_log/` — `flownet.pkl` (23.5 MB) + `IFNet_HDv3.py` with `scale_list=[8,4,2,1]` → **pre-4.22 RIFE weights** (4 flow blocks). The 4.25 anime fix changed the architecture to 5 blocks (`[16,8,4,2,1]`) — i.e. the local weights are exactly the era the model author replaced because of anime.
- `rife_master.zip` — Practical-RIFE-main snapshot (2026-08-27, commit bbfd2ea9).
- Landmine: `models/rife/gofile-dl/` — an **unrelated Go project** (gofile downloader, with its own .venv) sitting inside `models/rife/`. Likely accidental; recommend relocating.

---

## 1. VERDICT — Interpolation (any anime, no soap, no artifacts, not glacial, target-FPS flexible)

### #1: RIFE v4.25 via HolyWu/vs-rife (`pip install vsrife`, v5.7.0, 2026-02-09, MIT, active)

```python
from vsrife import rife

out = rife(
    rgb_h,
    model="4.25",
    fps_num=60000,
    fps_den=1001,  # exact 59.94 target — ONE pass, no x2 stacking
    sc=True,
    sc_threshold=0.15,  # scene-change passthrough
    trt=True,
    trt_opt_shape=[1920, 1080],
    trt_cache_dir="~/.cache/vsrife",
)
```

Why (all verified):
1. **The "bad RIFE" is the owner's current weights.** Pre-4.22 `scale_list=[8,4,2,1]` era = the ghosting era. hzwer (author), 4.25 release note: *"more flow blocks, scale_list changes… **anime scenes have been significantly improved**"*; README: *"recommended to choose 4.25 by default for most scenes"*. SVP forum (2025-05): *"Rife 4.25/4.26 are the best followed by 4.18"*. In-vs-rife fallbacks: 4.26 / 4.26.heavy / 4.13–4.15-lite (old-school sharp-lines anime camp).
2. **Arbitrary target FPS is first-class in the wrapper** (verified in `vsrife/__init__.py` source): `fps_num/fps_den` absolute target overrides `factor_num/factor_den`; per-output-frame `t = n*factor_den % factor_num / factor_num`; `t==0` → passthrough; scene change → passthrough. 23.976→59.94/60 in one pass. **Stacked x2 is both unnecessary and unable to reach 60 from 23.976 (needs ×2.5)**; no source endorses stacking.
3. **Speed**: vs-mlrt wiki official bench, RTX 4090 TRT FP16 1080p: RIFE v4.7+ 132 fps (1 stream) → 268 (2) → 496 (5); v4.4–4.6 up to 574–787 fps. TRT ≈1.45× faster than ORT_CUDA. 5080 expectation: between 3090 and 4090 rows (≈100–150 fps single-stream output) — faster than realtime; **no public 5080 benchmark exists [UNVERIFIED]** — settle with a 5-min `rifebench`/vspipe run.
4. **Recognition**: the most-deployed open VFI (SVP embeds vsmlrt itself; Flowframes 2,076★, Waifu2x-Extension-GUI 17,156★, enhancr package it; hzwer's README links vs-rife officially).

### Runner-up: GMFSS Fortuna (union) via HolyWu/vs-gmfss_fortuna
- **Best artifact-free anime quality** — anime-dedicated (AnimeRun flow finetune, SVFI-team-supported). Controlled A/B (Practical-RIFE issue #110): beats RIFE 4.26 on ghosting/warping at all playback speeds; RIFE keeps lines sharper. SVFI's anime "Ultra High Quality" preset = GMFSS pg104.
- Same `fps_num/fps_den` arbitrary-FPS API (verified in source).
- **Costs**: 3.5–10× slower than RIFE (16 fps 1080p on a 4090 PyTorch); wrapper **frozen since 2023-04** (PyTorch 1.13/CUDA 11 era — friction on a 5080/CUDA 13 stack). Quality pick for hard episodes, not the batch default.

### Not viable in VS today
- **GIMM-VFI** (NeurIPS 2024, natively arbitrary-t, best hard-motion quality among RIFE alternatives): **no VapourSynth wrapper exists** (verified absent in vs-mlrt + GitHub searches); ComfyUI-only.
- vs-gimm / vs-ifrnet / vs-amt / vs-film wrappers **do not exist** (verified 404/search-zero).
- IFRNet/AMT/FILM/CAIN/DAIN: no anime-community following or no VS path; FILM repo archived; DAIN original repo gone.

### Community facts that change quality more than model choice
- **Scene-change detection before interpolation is canonical** (vsmlrt.RIFE docstring mandates `misc.SCDetect`/`mv.SCDetection`).
- **Anime dedup matters more than stacking debates**: anime is drawn on twos/threes (8–12 effective fps); SVP users report "duplicate frames removal … felt like going from 8 fps to 120 fps"; SVFI ships dedup presets. A dedup-aware/retime pipeline is the real smoothness lever.
- Interpolate-before-upscale vs after: split community. Anime-dominant sources lean **interpolate at source res, then NN upscale** (interpolation quality better on native line art; upscale runs on fewer frames); ComfyUI/modern general-video leans upscale-first. Both cited with dates in `01-interpolation.md` §8. The owner's current code interpolates AFTER upscale — defensible, but the anime-lean is the opposite order.
- Deband before interpolation is standard VS anime practice (flagged `[UNVERIFIED convention]` — consistent with all pipeline evidence).

---

## 2. VERDICT — Upscaling (one universal spandrel-loadable model, any anime)

### #1: `2x_AnimeJaNai_HD_V3` Compact (the-database/mpv-AnimeJaNai)
- Exact artifact: `.pth` **4,838,742 B**, sha256 `af7307eee19e5982a8014dd0e4650d3bde2e25aa78d2105a4bdfd947636e4c8f`, from GitHub release **3.0.0** asset `2x_AnimeJaNai_HD_V3_ModelsOnly.zip` (2024-01-17). Drop-in for `models/upscale/model.pth`.
- SRVGGNet Compact (64nf/16nc, **4.8 MB**) — spandrel loads it natively via the project's existing `ModelLoader().load_from_file()` path. x2 scale.
- **Recognition (widest live footprint of any anime-video model)**: mpv-AnimeJaNai **770★ pushed 2026-10-27** (release 3.9.0 same day); bundled as default brain in VideoJaNai (281★) and Real-Video-Enhancer (2,358★); first named recommendation in AMV101's canonical guide (updated 2025-06-27); rated in SVFI's commercial table; active author + Discord.
- **Quality design = the anti-"мыло" choice**: V3 = 7 months, 100+ release candidates; explicit goals: faithful appearance, **no oversharpening/ringing/aliasing, preserve intentional DOF blur**, accurate line colors/darkness/thickness. Speed class: compact = 17→38 FPS 1080p on a 4090 TRT — trivial for a 5080.
- Tradeoff: **HD-only by design** (same author ships SD V1 beta; community pairing for DVD = AniSD Suite / LD-Anime_Compact). License CC-BY-NC-SA-4.0 (fine for personal archival).

### Runner-up: `4x-AnimeSharp` (Kim2091, 2021, ESRGAN RRDB, CC-BY-NC-SA)
- **The all-time recognition king** of dedicated anime models; the only single file that also tolerates SD/DVD sources → pick this instead if "ANY anime" must include a heavy SD library in one never-think-about-it x4 model. Costs: 2021-gen sharpening bias, RRDB-class speed at 4x. Modern successor line: 2x-AnimeSharpV3 (2024-10), V4/RCAN (2025-01).

### Top-5 table (details and sources in `02-upscaling.md` §9)
| # | Model | Arch | Why / why not |
|---|---|---|---|
| 1 | 2x_AnimeJaNai HD V3 Compact | SRVGG 2x | fidelity-designed, fastest recognized tier, widest live ecosystem |
| 2 | 4x-AnimeSharp | ESRGAN 4x | universality incl. SD, legend status, but sharpening-forward + slow |
| 3 | Real-CUGAN Pro set | CUNet 2x/3x/4x + denoise | technical universality champion (SVFI ★★★★★ anime), but frozen 2023, repo license ambiguous |
| 4 | 2x-AnimeSharpV3 | ESRGAN 2x | modern Kim2091 line |
| 5 | AniScale 2 (ESRGAN variant) | ESRGAN 2x | clean 2023 suite (OmniSR sibling for speed) |

### Explicitly not for this pipeline
- **Topaz Video AI** — community-canonical "not for anime" (AMV101: models target mainstream/live-action).
- **Anime4K v4** (21,570★) — brilliant mpv realtime GLSL, but not a `.pth`; README itself says re-encoding to 4K should be avoided → playback complement only.
- **Real-ESRGAN anime_6B** — foundational (37,247★) but frozen since 2022, oversharpening/hallucination is the recurring complaint; `realesr-animevideov3` remains the only BSD-licensed fallback (conservative look).
- **Transformer archs (SwinIR-L/HAT)** — 0.61 FPS @ 15.4 GB for SwinIR-L 4x on a 4090 — archival-brutal; not for batch.

---

## 3. Canonical community pipeline (HD anime → archival)

1. **Descale** to native production res (most "1080p" anime is natively 720/810/900p) — guide.encode.moe / getnative.
2. **Deband** (f3kdb/vs-deband).
3. **AI upscale** — #1 model above (chaiNNer batch, VS+vs-mlrt TRT, Real-Video-Enhancer GUI…).
4. **Interpolate** (RIFE 4.25 via vsrife; scene detect; optional dedup/retime).
5. **Encode** — community x265 10-bit; owner NVENC HEVC 10-bit (current `-cq 16 p7` posture consistent with community).

Order note: community sources split on upscale↔interpolate order; anime-lean = interpolate at source res then upscale (see §1). Either is defensible; pick one, verify visually on hard scenes.

---

## 4. Recommended exact stack

| Layer | Choice | Verified state 2026-10-27 |
|---|---|---|
| Frameserver | **VapourSynth R81** (PyPI, LGPL-2.1, monthly cadence, pushed today) | 2,122★ |
| Interpolation | **vsrife 5.7.0** (`pip`), model **4.25**, `fps_num/fps_den`, `trt=True`, `sc=True` | MIT, active, models auto-download |
| Alternative runtime for both steps | **vs-mlrt v16.3** (AmusementClub — *not* "AmusementLAB"; GPL-3.0; ONNX RIFE 4.0–4.26 + RealESRGANv2/CUGAN/waifu2x/SwinIR ships in release assets) | `BackendV2.TRT(fp16=True, num_streams=2, output_format=1, use_cuda_graph=True)` |
| Upscale model | **2x_AnimeJaNai HD V3 Compact** `.pth` (sha256 pinned) via existing spandrel path | release 3.0.0 |
| Pipe | `vspipe -c y4m script.vpy - \| ffmpeg -i - -c:v hevc_nvenc …` or in-Python `clip.output()`/`frames()` | official docs + context7 covered (`/vapoursynth/vapoursynth` 1646 snippets, `/amusementclub/vs-mlrt` 226, spandrel 830) |

Both skeletons (vsrife gold path, vs-mlrt graph) in `03-vapoursynth-runtime.md` §9.

---

## 5. Honest gaps

- **No public RTX 5080 benchmarks** for RIFE/upscale models (all 5080 numbers extrapolated from 3090/4090 TRT rows).
- OpenModelDB (671 models, 241 anime-tagged) has **no public rating/download counters** — "top by rating" claims anywhere are unfaithful; ranking here uses recognition signals instead.
- Reddit bodies were unreachable from this network on every route (incl. Wayback); thread evidence comes from search snippets and mirrors, dates flagged `[UNVERIFIED]` where applicable.
- doom9/Habr are thin in this niche post-2024 (Habr: one 2022 RIFE walkthrough, Selectel).
- VSEDB unreachable; "st4rkhit GMFSS ONNX pack" unlocatable (folklore until linked).

---

## 6. Corrections to commonly repeated claims (all API-verified)

- vs-mlrt owner = **AmusementClub**, not AmusementLAB.
- Flowframes owner = **n00mkrad** (brand "nmkd"), 2,076★.
- GIMM-VFI = **NeurIPS 2024**, S-Lab NTU (not Google/ICML). AMT = CVPR 2023. IFRNet = CVPR 2022.
- Anime4K v4 shipped **2021-09**, not 2022. Real-ESRGAN frozen since release 0.3.0 (2022-09-20).
- `HomeOfVapourSynthEvolution/VapourSynth-RIFE-ncnn-Vulkan` is deleted; living fork = styler00dollar.
- Original baoteng/DAIN repo is gone (only nihui's ncnn port remains).

---

## 7. Encoder decision (2026-10-27, owner-confirmed): CPU-only libx265 10-bit

Full reports: `04-encoders.md` (x265 vs NVENC Blackwell) and `05-encode-flags.md` (flag sheet + MKV assembly).

- **Owner instinct confirmed by evidence.** NVENC CQ16 file-size explosion = CQ/CRF scale mismatch: at index 16 NVENC HEVC produced **+85% size** vs x265 (rigaya anime bench, 14.9 vs 8.1 Mbps). x265-CRF16 quality ↔ NVENC **CQ ≈ 20–22**.
- **The forgotten "50-series generation" = Ninth-generation NVENC (Blackwell)**: +5% BD-BR, AV1 lookahead, UHQ mode; 4:2:2 (HEVC/H.264 only) irrelevant to a 4:2:0 deliverable. On Blackwell UHQ, NVENC reaches ±10% of *default* x265 at the CRF16 anchor — but has **no psy-RDO / dark-AQ (aq-mode 3) equivalents**, and UHQ's temporal filter is NVIDIA-documented as "for natural content, not synthetic/animated". At CRF24+ anchors the gap re-opens to +25–45%. Verdict: NVENC fails the archival bar → removed from pipeline.
- **Default encode line** (flag rationale per-flag in 05 §1.1): `libx265 -preset slow -crf 16 -profile:v main10 -pix_fmt yuv420p10le` + BT.709 tags + `-x265-params no-sao=1:no-strong-intra-smoothing=1:deblock=-1,-1:psy-rd=2.0:psy-rdoq=1.0:rdoq-level=2:aq-mode=3:aq-strength=0.8:qcomp=0.65:bframes=8:rc-lookahead=60:keyint=<10s fps>:min-keyint=<1s fps>:no-open-gop=1:tskip=1:cbqpoffs=-2:crqpoffs=-2`. Cutree stays ON at CRF ≥15 (VCB rule); `no-cutree` + `qg-size 16` + `ref 5` are the (a)-collection-tier extras at CRF 14. `-tune animation` FORBIDDEN (JET: psy-rd 0.4/aq 0.4/deblock 1:1 = anti-anime).
- **Preset ladder reality** (answers "is slow going to tank it?"): medium CRF16 is *worse* than slow CRF18 while ~50% larger (Kokomins, anime-measured); slow→slower = +81% time for +4% size; slower→veryslow = +78% time for −0.4% size (doom9 2020 bench). **Slow = the efficiency knee**; veryslow is polish, not a size lever.
- **CRF anchors** (10-bit 1080p anime): 12–14 transparent/collection (JET, VCB BDRip), **16 = VCB workhorse = owner's size reference done properly**, 18 = cross-community floor (encode.moe/Kokomins). CRF (not 2-pass, not CRF+VBV — VBV enables "emergency denoising" blur) is the right rate control; "dynamic bitrate" = qcomp 0.65 (temporal) + aq-mode 3 (spatial dark bias) + cutree (causal).
- **MKV assembly** (verified from ffmpeg docs): two-input shape `-i pipe -i source.mkv`, `-map 0:v:0 -map 1:a? -map 1:s? -map 1:t?` (attachments need explicit maps), `-map_metadata 1 -map_chapters 1`, `-c:a copy -c:s copy`, `-disposition:a:0 default` optional; BT.709 color tags forwarded to HEVC VUI via libx265.c. Grain synthesis: x265 has SEI-signaling only — not warranted for typical anime.
- **Expected outcome vs the owner's NVENC CQ16 baseline: ~45–55% smaller files at equal-or-better quality.**
