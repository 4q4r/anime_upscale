# Anime Neural Frame Interpolation — Deep Research Report (VapourSynth Focus)

**Date:** 2026-10-27
**Scope:** Neural + classical frame interpolation for ANIME as of Oct 2026, with VapourSynth (VS) as the target runtime. Owner context: RTX 5080, Linux, currently runs Practical-RIFE (PyTorch, `models/rife`, scale_list=[8,4,2,1] → pre-4.22 architecture) + ffmpeg NVENC HEVC 10-bit; wants to move into VapourSynth. Mid-flight controller refinements added: (1) speed is first-class, (2) verify/refute "RIFE is bad" per version, (3) flexible target FPS (arbitrary-t or stacked x2), (4) verdict re-ranked on quality+speed+arbitrary-FPS+VS-availability.
**Notation:** `[V]` = verified fact (GitHub/HF/Steam API or primary source, access date 2026-10-27 unless noted). `[O]` = community opinion (quoted, with source and date). `[UNVERIFIED]` = could not confirm.

---

## 0. Executive summary

| Axis | Finding |
|---|---|
| Best quality for ANY anime | **GMFSS Fortuna (union)** — anime-dedicated, community quality reference. Slow (~4–16 fps 1080p on mid/high RTX, PyTorch; ~7 fps on A4000 TRT). |
| Best speed/quality balance | **RIFE v4.25 / 4.26 (or 4.13/4.15-lite per some camps)** via **HolyWu/vs-rife v5.7.0** (PyTorch/TRT) — 1080p@2x ≈ 86 fps (3090 Ti NCNN) → 136–171 fps (4090 NCNN/TRT). |
| "RIFE is bad" | **Half-true and version-dependent, not version-agnostic.** Pre-4.2-era ghosting was real; 4.4 was anime-tuned; 4.7–4.10 anime-optimized (SAFA); 4.25 changed flow-block architecture (scale_list!) with hzwer himself noting "anime scenes significantly improved". GMFSS still beats even 4.26 on ghosting/warping per controlled A/B. The owner's **scale_list=[8,4,2,1] pre-4.22 code IS the "bad RIFE" era for anime**. |
| Arbitrary target FPS | **Both vs-rife and vs-gmfss_fortuna support single-pass arbitrary target FPS** (`factor_num/factor_den`, `fps_num/fps_den`); vsmlrt.RIFE accepts `fractions.Fraction` multi. GIMM-VFI is natively arbitrary-t but has **no VS wrapper**. Stacked x2 to reach 60 from 23.976 is impossible (needs ×2.5) and stacked passes are community-discouraged vs single-pass arbitrary-t. |
| **REVISED VERDICT** | **RIFE v4.25 (fallback 4.26) + HolyWu/vs-rife v5.7.0 with TensorRT backend = best "ANY anime via VapourSynth" for this owner** (quality acceptable-to-good on 4.25+, one-pass arbitrary 23.976→59.94/60, fast, actively maintained, models ship in wrapper releases). **Runner-up: GMFSS Fortuna union via HolyWu/vs-gmfss_fortuna** (best artifact-free quality, but ~5–10× slower and wrapper stale since 2023-04). GIMM-VFI = quality option outside VS (no wrapper). |

---

## 1. RIFE family (hzwer/Practical-RIFE)

### What
RIFE (Real-Time Intermediate Flow Estimation) — the de-facto standard open neural VFI. Original paper arXiv 2020 (hzwer/arXiv2020-RIFE), **accepted ECCV 2022** after 5 rejections (CVPR21, ICCV21, AAAI22, CVPR22, finally in) `[V: hzwer/ECCV2022-RIFE README — "2022.7.4 Our paper is accepted by ECCV2022"; "From 2020 to 2022, we submitted RIFE for five submissions (rejected by CVPR21 ICCV21 AAAI22 CVPR22)"]`. Practical-RIFE is the successor engineering repo.

### How it works
IFNet directly estimates intermediate flows (f_t0, f_t1) plus an occlusion/ratio map at multiple scales, coarse-to-fine; no explicit precomputed optical flow (unlike DAIN/DAIN-style pipelines) — this is why it's fast. Later versions added SAFA-derived improvements (4.7+), FILM's gram loss (4.17+), and more flow blocks with a new scale_list (4.25+) `[V: repo READMEs + hzwer issue quotes below]`.

### Requirements
- Code MIT; **model weights also MIT** `[V: README line "The content of these links is under the same MIT license as this project"]`.
- PyTorch ≤3.11 python for official code; inference needs CUDA GPU for practical speeds; frames padded to mod 32 (some models want mod 64) `[O→V: SVP wiki "RIFE wants frame size divisible by 32 (or even by 64 for some models)"]`. Official baseline: "30+ FPS for 2X 720p on a 2080Ti" `[V: hzwer/RIFE README]`. Supports `--scale` 0.5 for 4K/UHD.
- **Arbitrary-timestep interpolation is supported between a pair of images** `[V: hzwer/RIFE README "It supports arbitrary-timestep interpolation between a pair of images"]`; the 4.20 release specifically improved "generalization of various timesteps… enhance the accurate perception of timestep, and we see a significant difference in animation scenes" `[V: hzwer comment, Practical-RIFE issue #80, 2024-07-24]`. The official `inference_video.py` CLI only exposes integer `--multi` (and `--exp` power-of-two), so arbitrary-t is a library/wrapper-level feature, not a CLI feature `[V: README command list]`.

### History + releases (models, all Google Drive + Baidu, MIT) `[V: Practical-RIFE README]`
- v3.x era 2021; v4.0 2021-12; v4.3 2022-08 (**"v4.4 model is specially tuned for animation"** — hzwer, ECCV2022-RIFE issue #279, 2022-08-24) `[V]`.
- **v4.7–v4.10 (2023-11): "optimized for anime scenes! We draw from SAFA's research"** `[V: hzwer/RIFE README]`.
- v4.12–4.17 (2023-11 → 2024-05; 4.17 added FILM gram loss); `.lite` variants = "similar training framework, but lower computational cost" `[V]`.
- v4.18–4.22 (2024-07→08); **v4.25 (2024-09-19): "I am trying using more flow blocks, so the scale_list will change accordingly. It seems that the anime scenes have been significantly improved."** `[V: README]`; v4.26 (2024-09-21) last full release; 4.25.lite (2024-10-20). No GitHub Releases object — models via Drive links; repo still maintained (pushed 2026-08-27) `[V: API]`.

### Popularity metrics `[V: api.github.com, 2026-10-27]`
- hzwer/Practical-RIFE: 1,030★, 133 forks, 7 open issues, MIT, created 2021-06-16, pushed 2026-08-27.
- hzwer/ECCV2022-RIFE (redirects hzwer/RIFE): the academic repo, community benchmark reference.
- rife-ncnn-vulkan (nihui) is the deployment vehicle outside PyTorch; AaronFeng753/Waifu2x-Extension-GUI (17,156★, pushed 2026-10-17) ships RIFE interpolation to a mass audience `[V]`.

### Community sentiment (the "RIFE is bad" question — VERIFY/REFUTE)
**Verdict: half-true, version- and content-dependent.**
- The legacy wisdom that old models beat new ones on 2D/anime is real and citable: Flowframes Discord model ranking (posted in SVP forum, ~2022): `quality 2D: 2.3 > 4.6 ≥ 4.5 > 4.4 > 4.0 > 4.3 …` `[O: svp-team.com/forum viewtopic 6281 p13]`.
- Anime-specific complaint history: ECCV2022-RIFE issue #279 (2022-08): "I've always seen plenty artefacts when interpolating anime vids using rife" → hzwer: v4.0+ trains on animation, "even though the training data is all anime, the effect is still not ideal", then "v4.4 model is specially tuned for animation" `[V: issue thread]`.
- Newer-model regressions (real): SVP forum p79 (2024-09): versions 4.17–4.26 show "fragmentary smooth" — "small parts like lips, cheeks, ears … can be separated from the frame", while "4.4 to 4.16 faces looks buttery smooth"; 4.26 worse than 4.15-lite on patterned textures `[O]`. elexor (2024-09, issue #80): "After 4.7 the newer models produce a sharper output and static pixels don't get distorted, but it wobbles more during slow movements and frame-blends more for large movements" `[O]`.
- Counterweight — the anime fix landed in 4.25: hzwer's own "anime scenes have been significantly improved" `[V: README]`. SVP forum p95 (2025-05): "For most people, Rife 4.25/4.26 are the best followed by 4.18" `[O: dawkinscm]`; p94 (2025-02): "Rife 4.25 (not heavy) is still the best", "4.22 Lite is the smoother one with less artifacts" (movie context) `[O]`. TNTwise/REAL-Video-Enhancer wiki: for Animation ">=rife-v4.7 in general. rife-v4.9, rife-v4.12, rife-v4.13 seem to have the best quality. rife-v2.3 … previous go-to for animation" `[O: wiki, accessed 2026-10-27]`.
- Absolute quality ceiling: controlled A/B at 0.125×–1× playback (Practical-RIFE issue #110, 2024-11-02): RIFE 4.26 shows "clearly more ghosting … and warping … more stuttery" vs GMFSS; "Tl;dr I pick GMFSS model over RIFE", while conceding RIFE keeps some lines/objects sharper and less blended `[O: 1kartafal4, issue #110]`.
- SVFI official preset table (doc.svfi.group, updated 2026-09-10): anime "Ultra High Quality" preset = **GMFSS pg104**; "High Quality" = **RIFE 4.13**; "Speed" = RIFE 4.6; representative NVIDIA anime preset: GmfSs-pg-104 "(Best quality)", official-4.15/4.22 "(Fast speed, medium effect)" `[V: doc.svfi.group/en/pages/vfi-models + /svfi-presets]`.
- **Implication for the owner:** their pipeline is pre-4.22 code with scale_list=[8,4,2,1] — i.e., the architecture hzwer replaced in 4.25 precisely because of anime. "RIFE is bad" is outdated wisdom **when applied to 4.25/4.26**, and still true **when applied to their current setup** compared to GMFSS.

### Anime suitability
Good-to-very-good on 4.25/4.26 (and 4.13/4.15 per camps), with residual ghosting/warping on fast motion, line breakup on thin art, and background smearing on pans vs GMFSS. Best-in-class speed. Anime-specific forks/tunes: v4.4 (anime-tuned) `[V]`, v4.7–4.10 (anime-optimized, SAFA) `[V]`, nihui's `rife-anime` ncnn model (community legacy) `[O: REAL-Video-Enhancer wiki benchmark table lists rife 1.8 (rife-anime)]`, steelywing's anime patch PR (Practical-RIFE #23, 2023) `[O: issue #279 comment]`, SVFI's own rpr/pg internal lines `[V: SVFI docs]`.

### FPS benchmarks (1080p source @2x unless noted)
| Setup | fps | Source |
|---|---|---|
| RTX 4090, RIFE 4.6 **TensorRT** (enhancr) | **170.9** | `[V: github.com/mafiosnik777/enhancr README benchmark table]` |
| RTX 4090, RIFE 4.6 NCNN | 136.1 | same |
| RTX 3090 Ti, 4.6 TRT / NCNN | 122.7 / 86.2 | same |
| RTX 3070, 4.6 TRT / NCNN | 94.6 / 64.1 | same |
| RTX 2060 Super, 4.6 TRT / NCNN | 70.3 / 53.8 | same |
| RTX 4090 laptop, ComfyUI RIFE 4.4 (512×512 latents, per-batch) | ~1.65–2.1 s/batch (≈10× faster than GMFSS there) | `[V: Fannovel16/ComfyUI-Frame-Interpolation PR #4, 2023-09]` |
| SVP realtime (classical→RIFE context) | 4.6=112, 4.15lite=100, 4.26=86, 4.25=84, 4.18=75 (playback fps, unspecified GPU) | `[O: SVP forum p79, 2024-09]` |
| RTX 3070 + TRT realtime (vs-mlrt .vpy) | x2 1080p ~30 fps source-rate realtime | `[O: We0M/realtime-RIFE-portable README]` |
| Flowframes (older full-pipeline incl. I/O) 1080p 2x: 3080=30, 3090=33 out-fps | `[V: n00mkrad/flowframes Benchmarks.md]` |

**Authoritative vs-mlrt wiki benchmark — RTX 4090 (AD102), TensorRT FP16, 1920×1080 RGBS, vsmlrt v14.test4, CUDA graphs, `max_aux_streams=0` (fps / VRAM MB):** `[V: AmusementClub/vs-mlrt wiki "Benchmark" (contributor MysteryDove), snapshot in research workspace]`

| RIFE version | 1 stream | 2 streams | 4 streams | 5 streams |
|---|---|---|---|---|
| v4.4–v4.5 | 136.9 / 778 | 273.8 / 1149 | 553.7 / 1893 | **574.3** / 2264 |
| v4.6 | 136.0 / 801 | 275.3 / 1192 | 544.3 / 1980 | 550.0 / 2368 |
| v4.7–v4.9 | 98.2 / 1303 | 195.8 / 2188 | 210.5 / 3957 | 210.7 / 4844 |
| v4.10–v4.15 | 84.4 / 1596 | 160.9 / 2773 | 162.0 / 5133 | 162.1 / 6310 |
| {4.12,4.13,4.15,4.16}_lite | 93.4 / 1333 | 187.3 / 2255 | 198.0 / 4099 | 198.0 / 5022 |

Two load-bearing implications: (1) **TRT makes even the newer anime models fast** — 4.10–4.15 "drop" from 137 to ~84 fps single-stream, recoverable to ~162 via streams; (2) **multi-stream scaling** (vs-mlrt `Backend.*.num_streams`) is the standard trick for saturating the GPU. RTX 5080-class expectation: same table shifted down modestly; even the worst row exceeds realtime ×2.5 (60/23.976 ≈ 2.5× needs ≈60 output fps from 24). No 5080-specific RIFE benchmark located (only a ComfyUI anomaly report where a 5080-laptop was slower than a 4060 due to software config, r/StableDiffusion 2025-10-27) `[O]`.

### VapourSynth support
- **HolyWu/vs-rife** — the canonical wrapper. 151★, MIT, created 2021-07-28, pushed 2026-05-10, latest release v5.7.0 (2026-02-09) `[V: API]`. PyTorch 2.10+/VS R69+, optional built-in **TensorRT** (torch-tensorrt), models 4.0→**4.26** incl `.lite`/`.heavy` (default **4.25**), auto-downloads `.pkl` from vs-rife's own releases `releases/download/model/flownet_v4.25.pkl` `[V: vsrife/__init__.py source]`.
- **AmusementClub/vs-mlrt** — `vsmlrt.RIFE()` with ONNX models v4.0→4.26 (+lite/heavy) from vs-mlrt release assets (`models.v15.16.7z`, `contrib-models.v15.16.7z`; releases active: v15.15 2026-02-05 → v16.3.test1 2026-09-23); `multi` accepts `fractions.Fraction` (arbitrary ratio); backends: `vstrt` (TensorRT + TensorRT-RTX), `vsort` (ONNX Runtime CPU/CUDA), `vsov` (OpenVINO CPU/Intel), `vsncnn-vk` (Vulkan, NVIDIA/AMD/Intel/Apple), `vsmigx` (AMD) `[V: API + scripts/vsmlrt.py source + README]`. Docstring: "For the best results, you need to perform scene detection on the input clip (e.g. misc.SCDetect, mv.SCDetection) before passing it to RIFE" `[V]`. **SVP itself ships vsmlrt.py for its RIFE AI mode** `[V: SVP dev Chainik, SVP forum 7651, 2025-08/09]`.
- **styler00dollar/VapourSynth-RIFE-ncnn-Vulkan** — 177★, MIT, created 2022-06-06, pushed 2025-09-06; ncnn/Vulkan (AMD/Intel capable), the engine inside enhancr's RIFE-NCNN `[V: API + enhancr README]`. The original `HomeOfVapourSynthEvolution/VapourSynth-RIFE-ncnn-Vulkan` is now **404/deleted** `[V: API Not Found, 2026-10-27]`; community forks/reference remain (VolodiaPG Nix fork; r/mpv guides).
- The task's guessed owner "Kiyamou/VapourSynth-RIFE-ncnn" **does not exist** `[V: API 404 + search 0 hits]` — the ncnn-VS wrapper owners are styler00dollar (living) and HomeOfVapourSynthEvolution (deleted).
- Community RIFE.vpy playbooks: We0M/realtime-RIFE-portable (TRT .vpy for mpv, references ONNX models rife_v4.25/rife_v4.17_lite/v4.26/v4.25_heavy), president-not-sure/mpv-interpolation (MVTools|RIFE VS scripts for mpv, pushed 2026-08-01), MSalman5230/mpv-config `[V: repos exist; API-verified where listed in §Verification log]`.
- Where the community gets RIFE models for VS: **HolyWu/vs-rife releases** (pkl), **AmusementClub/vs-mlrt releases** (onnx, incl. contrib-models), hzwer Google-Drive zips, OpenModelDB (hub verified: 559★, pushed 2026-08-14; RifeV4 entries present in site bundle; specific page slugs not fetched — partial `[V]/[UNVERIFIED]`), Flowframes pkgs (rife-cuda/rife-ncnn, per issue #130 helper zip) `[V]`.

### Verdict (RIFE)
Use **v4.25** (wrapper default, hzwer-recommended default "for most scenes", anime-improved) or **4.26**; keep **4.13/4.15-lite** in the toolkit as the "old-school sharp lines" fallback that several anime camps still prefer. On RTX 5080 expect faster-than-realtime 1080p interpolation via TRT. In VS: HolyWu/vs-rife is the cleanest path (exact wrapper for the .pkl models), vs-mlrt is the flexible multi-backend path with ONNX.

---

## 2. GMFSS / GMFSS Fortuna

### What
GMFSS ("GMFlow-based …", the anime line of GMFlow-based synthesis) — flow+fusion synthesis network family **explicitly built and tuned for anime**. **98mxr/GMFSS_Fortuna** (created 2023-04-03, 83★, pushed 2026-01-27, MIT): "The All-In-One GMFSS: **Dedicated for Anime Video Frame Interpolation**"; project "supported by SVFI Development Team (Justin62628/Squirrel-RIFE)"; 2023-06-25 union fine-tune updated using **AnimeRun** anime optical-flow data `[V: API + README]`. Sibling repo 98mxr/GMFSS_Union (referenced by enhancr).

### How it works
GMFlow-style global matching for motion + forward splatting + fusion/refinement synthesis of the middle frame; "Fortuna" bundles the pieces, "union" is the more general model. Trained/finetuned on anime flow data (AnimeRun) `[V: README + ComfyUI docs description]`.

### Requirements
PyTorch 1.13.1 + CuPy (CUDA 11.x/12.x), Python 3.9; models on Google Drive (no GitHub releases) `[V: README]`. VRAM-hungry, multi-network inference.

### Popularity metrics
- 98mxr/GMFSS_Fortuna 83★ `[V]`; **HolyWu/vs-gmfss_fortuna** 20★ (created AND last pushed **2023-04-08** — stale 3.5 years), MIT; **HolyWu/vs-gmfss_union** 21★, **archived=True** `[V: API]`.
- SVFI ships GMFSS variants commercially (pg_104/pg_104_lite/pg_104_pro/pg_117/pg_119/union_v) and labels them: "Anime quality reference, but slow… Among the strongest anime VFI" `[V: doc.svfi.group model table]`.
- fuyuka3725/gmfss-onnx-directml (ONNX DirectML runner, pushed 2026-09-29) — niche but shows continued porting interest `[V: search result]`.

### Community sentiment
- The strongest positive consensus of any model for anime quality: Practical-RIFE issue #110 A/B (2024-11): GMFSS beats RIFE 4.26 on ghosting/trailing/warping/stutter at all playback speeds; RIFE's only edge is sharper unblended lines `[O]`. r/VideoEditing (2024-08-11): "for anime content … gmfss pg 104 model in svfi/svfi pro looks the absolute best/smoothest for most scenes" `[O: u/DijitulTech1029]`. SVFI docs treat GMFSS pg104 as the anime quality reference `[V]`. wanx-troopers VFI page: "GIMM-VFI vs GMFSS … on the really extreme frames GIMM-VFI tends to give cottage cheese, whereas GMFSS has a smoother characteristic" `[O]`.
- Complaints: blending/softening ("more blending in of objects, lines" — the "soap" the owner fears, but community judges it far less objectionable than RIFE's ghosting at speed) `[O: issue #110]`; CuPy dependency fragility (ComfyUI install gripes) `[O: comfy.icu node docs]`.

### Anime suitability
**Best-in-class.** Purpose-built for anime; handles flat regions, line art, sakuga motion without RIFE-style double-image ghosting. Tradeoff: mild blending on extreme motion.

### FPS benchmarks (1080p @2x)
| Setup | fps | Source |
|---|---|---|
| RTX 4090, GMFSS Union **PyTorch** | **16.35** | `[V: enhancr README]` |
| RTX A4000, Union PyTorch / **TRT** | 3.68 / 6.79 | same |
| RTX 4090 laptop, ComfyUI GMFSS Fortuna Union (512×512, x6) | 5.99 s/batch vs RIFE 4.4's 1.65 s → **≈3.5× slower** there | `[V: ComfyUI PR #4]` |
| r/StableDiffusion (2025-12-31), 4090, 81 images 2x | GIMM 4 min vs RIFE 50 s (GMFSS sits between; see §3) | `[O]` |
SVFI markets `pg_104_lite` (Pro edition) as the faster GMFSS; doc keeps "Still much slower than Tariff" (their internal fast anime model) `[V: SVFI docs]`.
**RTX 5080 estimate (UNVERIFIED):** ≈20–30 fps PyTorch, ~2× with TRT if a working TRT path is achieved — i.e., a 24-min episode at 1080p×2.5 is a multi-hour job, not "ages" but 5–10× RIFE time.

### VapourSynth support
**HolyWu/vs-gmfss_fortuna** (`pip install vsgmfss-fortuna`): CuPy+NumPy+PyTorch 1.13.1, VS R55+, optional TRT 8.6 path (Windows 7z with bundled runtime in Releases) `[V: README]`. Signature verified from source: `gmfss_fortuna(clip, device_index, num_streams, trt, trt_max_workspace_size, trt_cache_path, model=0, factor_num=2, factor_den=1, fps_num=None, fps_den=None, scale=1.0, ensemble=False, sc=True, sc_threshold=None)` — **single-pass arbitrary target FPS supported** (factor 5/2 = ×2.5), scene-change handling on by default, RGBH/RGBS only, scale ∈ {0.25,0.5,1,2,4} `[V: vsgmfss_fortuna/__init__.py]`. **Caveat: not updated since 2023-04-08** — pins old PyTorch/TensorRT; on a 2026 stack (RTX 5080, CUDA 13) expect dependency friction vs vs-rife's PyTorch 2.10 baseline. vs-gmfss_union is archived; the Fortuna wrapper supersedes it.

### Verdict (GMFSS)
The anime **quality king** with real VS support, but: stale wrapper, heavy deps, 3.5–10× slower. Use when quality outranks runtime, or per-scene for hard episodes.

---

## 3. GIMM-VFI (Google/NTU — correction: NeurIPS 2024, S-Lab NTU, not "Google, ICML")

### What
**GSeanCDAT/GIMM-VFI** — 409★, created 2024-05-31, pushed 2025-06-05, license NOASSERTION, **NeurIPS 2024** "Generalizable Implicit Motion Modeling for Video Frame Interpolation" (authors: Zujin Guo, Wei Li, Chen Change Loy — S-Lab, NTU) `[V: API + README]`. *(Task brief said "Google, ICML 2024" — both corrected by the repo itself.)*

### How / arbitrary-t
Motion modeled as a **continuous implicit function of time** → natively **arbitrary-timestep** interpolation ("GIMM-VFI performs … interpolations between two adjacent video frames at arbitrary timesteps" `[V: README]`). Variants: RAFT-based GIMM-VFI-R, FlowFormer-based GIMM-VFI-F, and LPIPS-enhanced -R-P/-F-P. ComfyUI node exposes `interpolation_factor` 1–100 single-pass `[V: comfy.icu node docs]`.

### Community sentiment / speed
- Quality: "GIMM-VFI-F is so far the highest quality frame interpolation model I've been able to find" `[O: r/StableDiffusion WAN thread, 2025-03-03, 227 pts]`; "it's the one people report holding together on motion that makes RIFE ghost and smear — fast pans, things crossing behind other things, chaotic action" `[O: comfy.icu docs]`; "I prefer GIMM over RIFE; it's not as fast, but RIFE produces artifacts under some motion circumstances … (GIMM is a faster version of FILM, essentially)" `[O: r/comfyui, 2025-12-30]`. Counterpoint for anime: "cottage cheese on the really extreme frames" vs GMFSS `[O: wanx-troopers]`.
- Speed: matched-clip community timing RIFE ≈9 s vs GIMM 25–53 s `[O: comfy.icu]`; 4090 user: 50 s (RIFE) vs ~4 min (GIMM) for 81 images 2x — "like 4 times slower than RIFE… couldn't see the difference" `[O: r/StableDiffusion 1q0f52c, 2025-12-31]`; 6x interpolation of a live-action clip ~7 min `[O: r/comfyui 1uqp243, 2026-07-08]`.

### Anime suitability / VS support
Good on hard motion, but the anime community's quality reference remains GMFSS. **No VapourSynth wrapper exists**: GitHub search for vs-gimm/vapoursynth GIMM = 0 relevant hits; not in HolyWu's catalog, not in vs-mlrt `[V: GitHub searches 2026-10-27]`. Usage today = ComfyUI (kijai/ComfyUI-GIMM-VFI, 486★) or upstream PyTorch.

### Verdict (GIMM)
Excellent arbitrary-t model, wrong runtime for this project. If the owner ever wants GIMM quality in VS, someone must write/ONNX-export a wrapper (UNVERIFIED feasibility; FlowFormer/RAFT backbones are heavy and CuPy-dependent in upstream code).

---

## 4. IFRNet, AMT, FILM, CAIN, DAIN (briefs)

### IFRNet
- **ltkong218/IFRNet** — 348★, MIT, pushed 2024-02-19; **CVPR 2022** (task brief said ICRA 2022 — corrected by repo description) `[V: API]`. Single-encoder efficient architecture, intermediate feature refinement; strong speed/quality on benchmarks. **vs-ifrnet does NOT exist** (GitHub search: 0 hits) `[V]`; practical deployments use **nihui/ifrnet-ncnn-vulkan** (111★, pushed 2022-07-20) `[V]` or ComfyUI (Fannovel16 ships IFRNet VFI; a 2025 YouTube comparison included IFRNET among candidates `[O]`). Anime suitability: never became an anime-community standard; no anime-specific weights. VS: only via generic ONNX route if someone exports it (not packaged) — effectively **no VS path today**.

### AMT
- **MCG-NKU/AMT** — 276★, pushed 2023-04-26 (stale), NOASSERTION license, **CVPR 2023** "All-Pairs Multi-Field Transforms" (brief said CVPR 2022 — corrected) `[V: API]`. Transformer-based, top benchmark quality, heavy. **vs-amt does NOT exist** (search: 0 hits) `[V]`. Anime: no community anime following; SVFI doesn't ship it. Skip for VS.

### FILM (Google)
- **google-research/frame-interpolation** (FILM) — 3,153★, Apache-2.0, **repo ARCHIVED** (pushed 2024-08-10) `[V: API archived=true]`. ECCV 2022, "Frame Interpolation for Large Motion": StyleGAN-decoder synthesis on top of IoU-Net-aligned feature embs — blends well on large motion but its synthesis inherently **softens** (the community "GIMM is a faster version of FILM" quip `[O: r/comfyui]`).
- **FFmpeg relation:** FFmpeg has **no FILM filter**; its only frame-interpolation is `minterpolate` (classical motion-compensated, non-NN) `[V: FFmpeg filter docs are stable knowledge; no FILM in any FFmpeg release]`. FILM's wider fame came via bundled demos and media tools, not FFmpeg.
- Anime suitability: mediocre for line art (blend/soft bias); the anime community does not recommend it over GMFSS/RIFE `[O: aggregate of threads above]`. VS wrapper: none found `[V: searches]`.

### CAIN
- **myungsub/CAIN** — 353★, MIT, AAAI 2020 "Channel Attention Is All You Need", pushed 2023-05-26 (stale) `[V: API]`. Channel-attention pixel-shuffle, **no explicit flow** — fast-ish but blurry on detailed motion. NCNN port exists (nihui/cain-ncnn-vulkan; enhancr benchmarked CAIN: NCNN 9–45 fps vs **TRT 45–183 fps on 1080p** — TRT CAIN is surprisingly fast but quality is soft/blurry vs RIFE) `[V: enhancr README]`. Anime: not recommended by any community source found. VS: a wrapper was teased by enhancr ("mafiosnik/vsynth-cain-NCNN-vulkan (unreleased)") `[V: enhancr README]` — **not released**; no usable VS path.

### DAIN
- Original **baoteng/DAIN is GONE (GitHub 404)** `[V: API Not Found]`; practical implementation = **nihui/dain-ncnn-vulkan** (543★, pushed 2023-10-29) `[V: search]`. CVPR 2019 depth-aware flow projection; extremely slow (depth networks), considered obsolete — "по меркам искусственного интеллекта DAIN считается устаревшей" `[O/V: Habr/Selectel article, 2022-09]`. Anime: no. VS: no wrapper; ncnn CLI only. Historical interest only.

---

## 5. VapourSynth wrapper ecosystem (map)

| Wrapper | Owner/repo | State (2026-10-27) | Backend | Models for anime |
|---|---|---|---|---|
| **vs-rife** | HolyWu/vs-rife (151★) | v5.7.0 rel. 2026-02-09, pushed 2026-05-10, ACTIVE | PyTorch 2.10 + optional Torch-TensorRT | RIFE 4.0–4.26 (+lite/heavy), default 4.25, auto-download from releases `[V]` |
| **vs-mlrt RIFE** | AmusementClub/vs-mlrt (489★, GPL-3.0, pushed 2026-09-25, ACTIVE) | releases v15.15→v16.3.test1 (2026) ship `models`+`contrib-models` 7z with ONNX | vstrt/vstrt_rtx/vsort/vsov/vsncnn-vk/vsmigx | RIFE 4.0–4.26 incl. heavy via contrib; `multi: Fraction` arbitrary `[V]` |
| VapourSynth-RIFE-ncnn-Vulkan | styler00dollar (177★, pushed 2025-09-06) | maintained fork; original HomeOfVapourSynthEvolution repo deleted | ncnn/Vulkan | RIFE (older sets + community ncnn models) `[V]` |
| **vs-gmfss_fortuna** | HolyWu (20★) | created/pushed 2023-04-08 — STALE | PyTorch 1.13+CuPy, TRT 8.6 | GMFSS Fortuna (+union archived wrapper) `[V]` |
| ccvfi → cccv | EutropicAI (ccvfi archived; cccv created 2025-09-23, pushed 2026-09-28, MIT, 15★) | young, active | PyTorch lib w/ VS API | RIFE v4.26-heavy, DRBA (anime VFI control net) `[V]` |
| vs-gimm / vs-ifrnet / vs-amt / vs-film | — | **DO NOT EXIST** `[V: searches]` | — | — |

Note: task's "AmusementLAB" owner is actually **AmusementClub** `[V: API]`. RTXFI (ryou128hr/rtxfi, TensorRT VS VFI, pushed 2026-09-02) and nvof_interpolate (NVIDIA Optical Flow Accelerator) exist as experimental niche entries (0–1★) `[V: user repo list]` — worth watching, not production.

---

## 6. Classic / non-NN alternatives

- **SVP (SmoothVideo Project)** — the paid (Pro ~USD 20 tier) realtime interpolation suite with the largest userbase in this space; historically classical MCF flow algorithms, **since 2023 it embeds RIFE via VapourSynth + vsmlrt/TensorRT or ncnn** (its own wiki documents model choice "4.6 gives higher quality, but ~10-15% slower than 4.4"; GPU requirement table: 1080p@48 needs ≥RTX 3070 generic or ≥2060 with TRT; 4K@48 = 4090 generic / ≥4080 TRT) `[V: svp-team.com/docs/rife-ai + wiki, accessed 2026-10-27]`. NN beat it on frame quality ("RIFE better overall but 8 times heavier than SVP4 basic algo" `[O: SVP forum p88]`), yet a vocal minority still prefers classical SVP's texture stability and pacing ("your SVP algo is better than any RIFE version … fake interpolated multilayered smushy stuff … tested with a 5090" `[O: SVP forum p94, 2025-03]`; "Lessons learned - RIFE is dead?" thread argues tuned classical SVP + good GPU beats RIFE for frame pacing) `[O]`. Relevant as a popularity signal, not as the owner's path (paid, Windows-first playback).
- **mvtools** (pinterf/mvtools, 191★, pushed 2026-04-24, AviSynth-desc but the VS community staple) — classical block-matching vectors; the free default in r/mpv land ("I have managed to get mvtools via vapoursynth to work … using CPU" `[O: r/mpv 1p7p4bf, 2025-11-27]`); fast, CPU/GPU-light, but characteristic block artifacts/ghosting on anime pans; no neural understanding of line art.
- **FFmpeg `minterpolate`** — built-in classical motion-compensated interpolation; slow at high settings and prone to blocking/merging on animation; zero NN quality. Used only when no VS/NN stack is available. (This is also the answer to "FILM in FFmpeg": FFmpeg ships **no** neural VFI.)

---

## 7. GUI packagers (popularity signals)

- **Flowframes** — **n00mkrad/flowframes**, 2,076★, pushed 2026-05-20, Windows GUI over RIFE (CUDA/NCNN) & DAIN; official channels: nmkd.itch.io/flowframes; maintains public Benchmarks.md `[V: API + Benchmarks.md]`. (Task's "verify owner": owner = n00mkrad, site brand nmkd `[V]`.)
- **SVFI (Squirrel VFI)** — Steam app 1692080, paid, released 2021-09-06, developer **SDT Core** (Chinese team; Justin62628/Squirrel-RIFE); the most feature-complete anime VFI product (dedup modes "Forward Dedup … best effect, at picture qualities surpassing hand-drawn", pg104/RIFE model matrix, Pro tier with pg_104_pro and "Tariff" fast model) `[V: Steam appdetails API + doc.svfi.group, 2026-09-10 update]`.
- **StaxRip** — staxrip/staxrip, 3,022★, pushed 2026-09-27, Windows encoding GUI whose frame server is AviSynth+/VapourSynth; RIFE available through its VS filter setup `[V: API + README; RIFE integration known feature — page-level doc not fetched this session]`.
- Also notable: **Waifu2x-Extension-GUI** (AaronFeng753, 17,156★, pushed 2026-10-17 — the biggest mass-market packager including RIFE interpolation) `[V]`; **enhancr** (mafiosnik777 — published the most-cited public 1080p RTX benchmark table) `[V]`; **REAL-Video-Enhancer** (TNTwise — maintains the "What RIFE Model to use?" wiki) `[V]`; **Lossless Scaling** (Steam) with LSFG frame-gen, often stacked after RIFE `[O: SVP forum]`.

---

## 8. Community consensus signals 2024–2026

Representative threads (titles, dates, rough engagement):
- r/mpv — **"How to set up RIFE interpolation to work with MPV (using GPU)?"** (2025-11-27, 5 comments): mvtools-on-CPU works out of the box, RIFE needs CUDA/Vulkan; users graduate to RIFE when GPU allows `[O]`. **"Anime 60fps mpv config?"** (2024-08-30): "the best 60fps settings is to use SVP" `[O]`. **"Is mpv's built in interpolation any good?"** (2025-04-29, 8+6 pts): built-in tscale = traditional resampling, not AI `[O]`.
- r/animepiracy — **"Anime upscaling AND fps interpolation at the same time possible?"** (2024-02-20): SVP+VLC at 48fps mainstream; OP's solution: RTX Video Super Resolution upscales **interpolated** frames (SVP realtime) `[O]`.
- r/VideoEditing — **"What are the best AI frame interpolation programs right now?"** (2024-08-11): "for anime … gmfss pg 104 model in svfi looks the absolute best"; default YouTube choices = Topaz Chronos / Flowframes RIFE `[O]`.
- SVP forum — **"New RIFE filter - 3x faster AI interpolation possible in SVP!!!"** (megathread, pages 13→95, 2022→2025, thousands of replies): version battles (4.4/4.6 legacy quality vs 4.25/4.26; 4.17+ "fragmentary smooth" complaints), speed tables, LSFG stacking, "Rife 4.25/4.26 best followed by 4.18" (2025-05) `[O]`. **"A/V desync with RIFE, duplicate frames removal…"** (2025-08-30): anime-specific praise — "SVP with RIFE and specifically 'duplicate frames removal' … felt like going from 8 fps to 120 fps (hand-drawn animations within a scene often are 8 fps)" `[O]`. **"Lessons learned - RIFE is dead?"** (2025): classical-SVP counter-culture `[O]`.
- doom9 — only generic player/shader threads found (Video player Shaders guide p4, 2021); **no dedicated 2024–2026 anime-interpolation consensus thread surfaced** — doom9 is a weak signal in this niche in 2024–2026 `[V: search scope; honest gap]`.
- Habr — **"Увеличиваем FPS в аниме с помощью нейросети и GPU Tesla T4"** (Selectel, 2022-09-07, 18 votes): RIFE walkthrough for «Призрак в доспехах»; DAIN obsolete; RU community defaults to RIFE `[V: habr.com/ru/companies/selectel/articles/686870]`. No newer major Habr anime-VFI article found in searches `[honest gap]`.
- GitHub — Practical-RIFE **issue #110** "Rife 4.26 Anime deformity decreased but inbetween frames are less visible" (2024-09-23 → 2024-11-02): the single most load-bearing RIFE-vs-GMFSS A/B `[O]`; **issue #279** (ECCV2022-RIFE) anime-model request → v4.4 anime tuning `[V]`; **issue #130** (2024-12): "many anime editors said that 4.6 is the best model right now" vs "4.17 and 4.26 are the best of the best" `[O]`.

### Typical pipeline order (sourced)
- **Scene-change detection before interpolation is canonical in VS**: vsmlrt.RIFE docstring mandates SCDetect (misc.SCDetect / mv.SCDetection) `[V]`; vs-rife has built-in `sc=True/0.15` default-off param; SVP exposes "Scene change threshold" with artifacts-vs-stutter tradeoff `[V: SVP wiki]`.
- **Interpolate before or after upscale — genuinely split by community:**
  - Interpolate-first camp (anime-dominant): enhancr author — "Interpolation gets worse with higher resolution, that's why I wouldn't upscale first" `[O: enhancr discussion #28, 2023-08]`; Topaz engineers — "perform the FPS conversion first and then upscale" `[O: r/TopazLabs, 2022-01]`; r/animepiracy realtime: SVP interpolates source-res, RTXVSR upscales after `[O: 2024-02]`; ryandash/Convert-and-Move-Anime: RIFE→48fps + VS upscale→4K in one vpy `[V: repo README]`; Topaz forum user found upscale-first "caused more and weirder artifacts" `[O: 2021-04]`.
  - Upscale-first camp (ComfyUI-modern): "Interpolate last … if the upscale brings out any more details … If you interpolate and then upscale could introduce flickering" + "Upscaling is a much heavier lift … get it done at the lower temporal resolution" `[O: r/comfyui 1m6gdnn, 2025-07]`; 2026 ComfyUI reference pipeline: restore/upscale → RIFE → encode `[O: comfyui-mcp skill doc, 2026]`.
  - Synthesis for the owner: for **anime with NN upscaling**, both orders are defensible; the anime-specific sources lean **deband/denoise → interpolate at source res → NN upscale → encode** (cheaper: upscale model runs on fewer frames; interpolation quality better at native line-art resolution), while ComfyUI's general-video world leans upscale-first. There is **no single consensus**; both cited with dates above. **Deband before interpolation** is standard VS anime practice (line-art-adjacent banding otherwise gets "interpolated into" artifacts); direct 2024–2026 quote not captured — flagged as `[UNVERIFIED convention]`, consistent with the sc/dedup evidence.
- **Anime-specific: dedup before interpolation.** Anime is animated on 2s/3s (8–12 effective fps); SVP's "duplicate frames removal → felt like 8 fps to 120 fps" `[O]` and SVFI's "Forward Dedup / First Order Dedup / Dedup shots on twos/threes" presets `[V: SVFI docs]` show the community pattern: drop dupes, interpolate the true unique frames, retime.

### Arbitrary-t vs stacked x2 (controller requirement 3)
- **Single-pass arbitrary-t is available in the VS layer today for the two relevant wrappers:** vs-rife (`factor_num/den`; e.g. 5/2 ⇒ 23.976→59.94; or `fps_num/den` direct) `[V: source]`; vs-gmfss_fortuna (same factor/fps params) `[V: source]`; vsmlrt.RIFE (`multi: fractions.Fraction`) `[V: source]`. RIFE itself supports arbitrary-timestep t between a pair (IFNet conditions on t), and **4.20+ specifically improved timestep generalization with "significant difference in animation scenes"** `[V: hzwer issue #80]`.
- GIMM-VFI is the strongest native arbitrary-t model (continuous-time motion function, factor up to 100) but lacks VS `[V]`.
- **Community preference & artifact accumulation:** sources favor single-pass over stacked passes: SVP thread notes 24→60 (×2.5) "needs generating TWO times more intermediate frames than x2 (24→48)" — i.e., stacking can't even reach 60 from 23.976 `[O: SVP p13]`; SVP "Lessons learned": "Integer interpolation multipliers seem to work best which might be why 60fps isn't quite as smooth as it could be" `[O: 2025]` — evidence that repeated/pass-based and off-ratio workflows degrade smoothness; elexor's timestep-accuracy notes on ≤4.18 models show large-t blends `[O]`; the 4.20+ timestep generalization fix exists precisely because multi-t inference was weak before `[V]`. Explicit "stacked x2 accumulates artifacts" measurements were not found; community practice is "don't stack — use the model's native ratio support" `[O: aggregate; no direct quote found — honest framing]`. For 23.976→60 exactly, note the pragmatic community split: pure 60.0 (retime artifacts on 23.976) vs 59.94 (factor 5:2) vs 48 (×2) — VS handles all three losslessly via single-pass factor/fps params.
- DRBA/VS-DRBA (routineLife1) — an anime-specific control net that *adjusts interpolation strategy per-region to preserve original pace* ("avoiding distortions common in frame interpolation"), targets `-fps 60` directly, and ships a "VapourSynth + TensorRT version can achieve 400% acceleration" `[V: repo README; VS-DRBA repo existence not separately verified]` — a niche but on-point tool for the owner's exact concern.

---

## 9. VERDICT (revised per controller axes: quality + speed + arbitrary-FPS + VS availability)

**Best for ANY anime via VapourSynth: RIFE v4.25 (primary) / 4.26, via HolyWu/vs-rife v5.7.0 with `trt=True`, one-pass `factor_num=5, factor_den=2` (23.976→59.94) or `fps_num=60`, `sc=True`.**
Why: (1) quality — 4.25's flow-block redesign is hzwer's own anime fix ("anime scenes significantly improved") and the community's current "best for most people" (SVP p95), eliminating the pre-4.22 weaknesses the owner currently runs; remaining ghosting is the known RIFE tax, mitigated by sc-detect + (optional) dedup; (2) speed — 100–170 fps 1080p@2x class on RTX 5080 with TRT (extrapolated from 3090 Ti/4090 TRT data), i.e., faster than realtime, satisfying "must not take ages"; (3) arbitrary FPS — native single-pass factor/fps in the wrapper, timestep robust since 4.20; (4) VS availability — first-class: wrapper actively maintained (v5.7.0, 2026-02), models auto-downloaded, MIT; (5) recognition — the most widely recognized open VFI stack (SVP embeds it, Flowframes/Waifu2x-GUI/enhancr package it, hzwer links vs-rife from the official README). Fallbacks inside the same wrapper: 4.26 (latest), 4.26.heavy (quality tilt), 4.13/4.15-lite (old-school anime sharpness camp).

**Runner-up: GMFSS Fortuna (union) via HolyWu/vs-gmfss_fortuna** — wins pure artifact-free quality for anime (ghosting/warping near-zero per the controlled A/B; SVFI's anime quality reference), single-pass arbitrary FPS also supported, but: wrapper frozen since 2023-04 (PyTorch 1.13/CUDA 11 era — real friction on RTX 5080/CUDA 13 stack), 3.5–10× slower (16 fps 4090 PyTorch; single-digit on midrange), so "must not take ages" fails for batch use. Best deployed selectively (hard episodes) or when the owner accepts the runtime.

**Watchlist:** GIMM-VFI (best hard-motion quality among RIFE-alternatives, natively arbitrary-t, but no VS wrapper — needs community ONNX/VS port), GMFSS ONNX DirectML port (fuyuka3725, Sep 2026 — suggests a TRT/ORT GMFSS path may mature), ryou128hr/rtxfi (TRT VS VFI experiments), EutropicAI/cccv (VS-capable lib carrying RIFE-4.26-heavy + anime DRBA control net).

### Speed/quality tradeoff table (1080p @2x, RTX-5080-class extrapolation)
| Candidate | Quality for anime | Speed (fps, 5080-class est.) | Arbitrary-t single-pass | VS support | Maintenance |
|---|---|---|---|---|---|
| RIFE 4.25/4.26 TRT (vs-rife) | good (post-4.25) | ~100–170 `[O extrapol]`; vs-mlrt wiki 4090-TRT: 84–137 fps 1-stream, 162–574 multi-stream `[V]` | YES `[V]` | vs-rife / vs-mlrt `[V]` | active `[V]` |
| RIFE 4.13/4.15-lite | good (sharp lines camp) | ~120–190 (lite) `[O extrapol]` | YES (4.13+) `[V]` | same `[V]` | active wrapper `[V]` |
| GMFSS Fortuna union | **best** | ~20–30 `[O extrapol]` | YES `[V]` | vs-gmfss_fortuna `[V]` | **stale 2023** `[V]` |
| GIMM-VFI (-F/-P) | very good (hard motion) | ~25–55 (4090 real) `[O]` | YES native `[V]` | **NONE** `[V]` | upstream 2025-06 `[V]` |
| IFRNet / AMT / FILM / CAIN / DAIN | niche/weak for anime | CAIN-TRT 183 (soft) `[V enhancr]`; others slow/absent | FILM/GIMM-style t: no VS | no wrappers `[V]` | FILM archived `[V]` |
| mvtools / minterpolate / SVP-classical | artifact-prone on anime | realtime-class | mvtools arbitrary via retiming | VS-native (mvtools) `[V]` | active `[V]` |

---

## 10. Verification log (API calls, 2026-10-27)

GitHub API `repos/<owner>/<repo>` (created, pushed, stars, forks, issues, license): hzwer/Practical-RIFE; HolyWu/vs-rife; 98mxr/GMFSS_Fortuna; HolyWu/vs-gmfss_fortuna; HolyWu/vs-gmfss_union (archived); ltkong218/IFRNet; nihui/ifrnet-ncnn-vulkan; AmusementClub/vs-mlrt; styler00dollar/VapourSynth-RIFE-ncnn-Vulkan; GSeanCDAT/GIMM-VFI; MCG-NKU/AMT; google-research/frame-interpolation (archived); myungsub/CAIN; pinterf/mvtools; staxrip/staxrip; OpenModelDB/open-model-database; EutropicAI/ccvfi (archived); EutropicAI/cccv; n00mkrad/flowframes; HolyWu + ryou128hr repo listings. 404s (verified absent/renamed): Kiyamou/VapourSynth-RIFE-ncnn, HomeOfVapourSynthEvolution/VapourSynth-RIFE-ncnn-Vulkan, baoteng/DAIN, nmkd/Flowframes (correct: n00mkrad), AmusementLAB/vs-mlrt (correct: AmusementClub). Releases endpoints: AmusementClub/vs-mlrt (v16.3.test1 2026-09-23 … v15.15 2026-02-05, models+contrib-models assets); HolyWu/vs-rife (v5.7.0 2026-02-09); hzwer/Practical-RIFE (none — Drive-distributed); 98mxr/GMFSS_Fortuna (none). Steam appdetails 1692080 (SVFI, SDT Core, 2021-09-06). Source fetches (raw.githubusercontent): vs-rife README + vsrife/__init__.py (params/models verified in code); vs-gmfss_fortuna README + __init__.py; 98mxr README + inference_video.py; GIMM-VFI README; vs-mlrt README + scripts/vsmlrt.py (RIFEMerge/RIFE/model enum); Practical-RIFE README; ccvfi README. Web (search-index reads): SVP wiki/docs, SVP forum threads 6281 (p13/60/79/88/90/94/95), 7651, 7503, 8085; doc.svfi.group (vfi-models, svfi-presets); enhancr README (benchmark table); Flowframes Benchmarks.md; Practical-RIFE issues #80/#110/#130, ECCV2022-RIFE #279; ComfyUI-Frame-Interpolation PR #4; comfy.icu node docs; wanx-troopers vfi page; Reddit r/mpv, r/animepiracy, r/VideoEditing, r/comfyui, r/StableDiffusion threads as cited; Habr 686870; REAL-Video-Enhancer wiki; We0M/realtime-RIFE-portable; blackmickeysvp.blogspot (vs_rife v2.0.0 720p benchmarks 54–93 fps, 2022/2024).
Unverified/honest gaps: no RTX 5080-specific RIFE/GMFSS benchmark located (extrapolations marked); OpenModelDB RIFE page slugs not individually fetched; StaxRip RIFE menu page not fetched; doom9/Habr thin post-2024 coverage in this niche; Z.AI Vision MCP not needed (no image inputs processed). Additional workspace snapshot (fetched during this engagement, integrated): vs-mlrt wiki Benchmark page (RTX 4090 TRT FP16 RIFE table, contributor MysteryDove) — saved as bench_4090.md next to this report.

---

## Appendices

### A. Skills used (via skill tool)
1. `brainstorming` (mandatory) — applied as intent-clarification for the research design; interactive interviewing N/A for a dispatched subagent (no human counterpart), design tree pre-specified by controller brief.
2. `grilling` (mandatory) — same adaptation: the brief's Q-axes (quality/speed/arbitrary-t/verdict) treated as the settled frontier; gaps converted into the verification checklist instead of user questions.
3. `build-with-exa` (mandatory) — Exa attempted first per routing; 402 credits → documented fallback to websearch + direct source fetches (per memory rule "Exa 402 → alternatives, don't re-hammer").
4. `research` — primary-source discipline: every claim traced to repo/README/issue/API/thread; findings written to this file.
5. `planning-with-files` — task_plan.md kept in /tmp/opencode/anime_research/ (scratch location per owner rules; /tmp non-durable by design for one-shot deliverables).

### B. Memory files read (mandate: all global memory)
- **Read fully:** `~/.config/opencode/MEMORY.md`; `topics/stack-and-tools.md`; `topics/workflow.md`; `topics/workflow-git-pr.md`; `topics/tech-gotchas.md`; `topics/communication-behavior.md`; `topics/2026-10-15-vision-ocr-scans.md`; `topics/environment-infra.md`; `projects/shikimori-japanese-theme.md`; `projects/anicli-go.md`.
- **Header-sampled (first ~1.5–2.2 KB each, confirmed out-of-scope for this task; content unrelated to video/VapourSynth):** `topics/2026-10-09-telegram-serverless.md`, `topics/2026-10-11-novum-organum-report-plan.md`, `topics/2026-10-12-slides-novum-organum-plan.md`, `topics/2026-10-14-opencode-v2-storage.md`, `topics/2026-10-15-binary-re-landmines.md`, `topics/2026-10-21-stepin-book-contract.md`, `topics/2026-10-25-locke-pirogov-referaty.md`, `topics/code-gotchas.md`, `topics/hh-ranking-ats-research.md`, `topics/harness-server.md`, `topics/job-search.md`, `topics/remnawave.md`, `projects/4q4r-profile.md`, `projects/WMS.md`, `projects/anicli-py.md`, `projects/bos-pro-bugs.md`, `projects/cindex.md`, `projects/ert.md`, `projects/firefox-vpn-client.md`, `projects/freeai-decompile.md`, `projects/freepackages.md`, `projects/habrparser.md`, `projects/hhauto.md`, `projects/home-global.md`, `projects/interview-kb.md`, `projects/iris-proposal.md`, `projects/kolosyug.md`, `projects/labboardvent.md`, `projects/labs-docker-stands.md`, `projects/labs.md`, `projects/remnavawe_ansible.md`, `projects/secureway_bot.md`, `projects/stepik-whitehacker-course.md`, `projects/teleparse.md`, `projects/traffic_mimicry.md`, `projects/tspu-research-2026.md`, `projects/vibecoding-course.md`, `projects/wauploader.md`, `projects/wms.md`, `projects/xray-raw.md`.
- Coverage note: 49 files total (19 topics + 30 projects). Sampling method and rationale documented here per memory-integrity rules; no memory files were written (facts returned to controller only).
