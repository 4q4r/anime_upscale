# Neural Video Super-Resolution for ANIME — Community Intelligence Report (2026-10-27)

Mission: select ONE universal anime upscale model — spandrel-loadable (ChaiNNer-compatible `.pth`/`.safetensors`), x2/x4, fits ANY anime, widest community recognition — for `process_anime.py` (RTX 5080, Linux, spandrel ModelLoader → ffmpeg NVENC HEVC 10-bit).

Provenance labels: **[V]** = verified fact (number + URL, checked 2026-10-27 unless noted) · **[OP]** = community opinion (quoted source) · **[UNVERIFIED]** = could not be confirmed.

---

## 1. Real-ESRGAN family (xinntao)

**What.** Practical blind-restoration GAN family: RealESRGAN_x4plus (general), RealESRGAN_x4plus_anime_6B (anime), realesr-animevideov3 (anime video, XS).

**How it works.**
- `RealESRGAN_x4plus` / `_x2plus`: RRDBNet (Residual-in-Residual Dense Block, 23 blocks, 64 features) trained with high-order degradation + sinc filters for blind real-world degradation.
- `RealESRGAN_x4plus_anime_6B`: same GAN recipe on an anime dataset, **6 RRDB blocks** ("smaller network") for speed [V: official model zoo, https://github.com/xinntao/Real-ESRGAN/blob/master/docs/model_zoo.md].
- `realesr-animevideov3`: **SRVGGNetCompact** (VGG-style conv net, 64nf/16nc ≈ 2.4 MB) — "Anime video model with XS size", X4 native, also usable at X1/X2/X3 [V: model zoo]. The x2/x3/x4 NCNN conversions in Upscayl's `custom-models` repo confirm the multi-scale deployments [V: https://github.com/upscayl/custom-models — 28 models incl. `realesr-animevideov3-x2/x3/x4.bin/param`].

**Requirements.** PyTorch (`.pth`); spandrel supports both RRDBNet ("includes … Real-ESRGAN") and SRVGGNet ("Real-ESRGAN Compact") [V: https://github.com/chaiNNer-org/spandrel README, Model Architecture Support section].

**History + releases.** Repo created 2021-07-19; latest release **v0.3.0 (2022-09-20)**; last commit push **2024-08-06**; 649 open issues; BSD-3-Clause [V: https://api.github.com/repos/xinntao/Real-ESRGAN]. Effectively feature-frozen since 2022.

**Popularity.** **37,247★ / 4,501 forks** [V: GitHub API]. The most-deployed anime SR family in absolute terms: embedded in Upscayl (51,170★, AGPL-3.0, pushed 2026-10-05 [V]), Stable-Diffusion WebUI extras, every NCNN GUI. HuggingFace mirrors of anime models carry modest likes (5 each on x4plus-anime-6b mirrors) [V: https://huggingface.co/api/models?search=RealESRGAN_x4plus_anime_6B].

**Community sentiment.**
- [OP] SVFI doc: "realESR … RealESRGAN hallucinates more (sharper, punchier)"; "RealESRGAN_x2plus/x4plus and anime variants … **Easy to oversharpen**"; "realesr-animevideov3: **conservative anime-video SR, fast and stable; avoid TTA**" (https://doc.svfi.group/en/pages/sr-models/).
- [OP] Real-CUGAN authors' comparison table (their bias, but specific): vs Anime6B — "sharpening strength is maximal, easily changes art style, may misjudge lines, may force-clear depth-of-field regions" (https://github.com/bilibili/ailab/blob/main/Real-CUGAN/README.md).
- [OP] r/TopazLabs (snippet, thread xe3b8b): Real-ESRGAN is "more specialized for anime/cartoons … Topaz is more generalized" (https://www.reddit.com/r/TopazLabs/comments/xe3b8b/).

**Anime suitability.** anime_6B is image-oriented (4x only) and known to over-sharpen; `realesr-animevideov3` remains a legitimate conservative video pick (fastest widely-recognized anime video model, BSD-3-Clause — the only permissive license in the whole family).

**spandrel.** Full support (RRDBNet + SRVGGNet) in the base package [V].

**Verdict.** Foundational, ubiquitous, frozen. `realesr-animevideov3` is the best BSD-licensed anime-video model; quality no longer state-of-the-art; oversharpening/hallucination on gradients, lines and DOF is the recurring community complaint.

---

## 2. Anime4K v4 (bloc97)

**What.** GLSL shader set for **realtime** anime upscaling/denoising in mpv (and Plex/IINA/Magpie). Not a .pth model; not an archival encoder.

**How it works.** Stacked tiny CNNs (S/M/L/VL/UL variants — each size step ≈ doubles shader time): `Restore` CNN removes compression/blur/ringing (line reconstruction explicitly designed for the "distribution shift" of in-the-wild 1080p anime), then `Upscale` x2 CNNs, optionally repeated. Modes: **A** = `Restore → Upscale → Upscale` (blur/resampling artifacts), **B** = `Restore_Soft → Upscale → Upscale` (downsampling/ringing), **C** = `Upscale_Denoise → Upscale` (clean sources, highest PSNR); secondary modes A+A, B+B, C+A for ≥x2 ratios [V: https://github.com/bloc97/Anime4K/blob/master/md/GLSL_Instructions_Advanced.md].

**Requirements.** Any GLSL-capable player (mpv `glsl-shaders`); no GPU VRAM concerns beyond shader cost; MIT license.

**History + releases.** Created 2019-08-13; HN launch 387 pts / 92 comments (2019-08-14) [V: https://hn.algolia.com/api/v1/search?query=Anime4K]. Releases: v3.1 (2020-05-30), v3.2 (2021-08-10), **v4.0.0 alpha 2021-07-26 → v4.0.1 Stable 2021-09-17** [V: https://api.github.com/repos/bloc97/Anime4K/releases]. NOTE: v4 shipped **Sept 2021**, not 2022; v4.1 (low-res SRGAN experiments, 360p→4K) exists only as commits/README section, no release. Last push 2024-08-17. **21,570★ / 1,402 forks, MIT** [V: GitHub API].

**Popularity.** The de-facto mpv realtime shader — r/mpv threads about custom Anime4K configs recur for years [OP: "Custom Anime4k Shader Configs" https://www.reddit.com/r/mpv/comments/17rdr0k/, "Custom Anime4K shaders?" https://www.reddit.com/r/mpv/comments/14r2cb8/ — dates UNVERIFIED, Reddit blocked for scraping]. Third-party ports listed in README (Magpie, Anime4KMetal, Anime4KCPP, video2x…) [V: README].

**Community sentiment.**
- [OP] README itself: optimized for **native 1080p** h264/h265/VC-1; "not optimized for downscaled 720p, 480p or SD"; "not a replacement for SRGANs, as they perform much better on low-resolution images"; "Re-encoding anime into 4K should be avoided" (i.e., explicitly anti-archival).
- [OP] SVFI doc: "very fast, conservative real-time … **Low detail ceiling**".
- [OP] DeepWiki summary: "compelling balance of quality and performance … other algorithms like Real-ESRGAN and Waifu2x have their strengths" in fidelity (https://deepwiki.com/bloc97/Anime4K/6.2-visual-comparisons).

**Anime suitability.** Excellent as a *watch-time* filter for native-1080p sources; wrong tool for archival (GLSL, no batch encoder, ceiling on detail).

**spandrel.** N/A — GLSL, not PyTorch. A PyTorch re-implementation exists (community Colab) but is not a recognized model file. **Disqualified** for the owner's spandrel requirement.

**Verdict.** Keep for playback; cannot be the pipeline model.

---

## 3. Real-CUGAN (bilibili/ailab)

**What.** Anime-only SR with waifu2x-CUNet-compatible architecture, trained on a **million-scale anime patch dataset**; 2x/3x/4x; 2x ships 4 denoise strengths + a conservative variant; **Pro** models added 2022-05 [V: https://github.com/bilibili/ailab/blob/main/Real-CUGAN/README.md].

**How it works.** Cascade U-Net (CUNet family) with optional `alpha` strength knob (0.7–1.3), denoise levels (-1…3), tile/cache modes for low VRAM (mode-3 = lossless unlimited-res with ~2.5× time). Authors report 1080p inference ≈ 1× waifu2x-CUNet baseline while ESRGAN-anime6B ≈ 2.2× (their measurement) [V: README comparison table].

**Requirements.** PyTorch `.pth` (spandrel: "Real-CUGAN" incl. Pro models listed [V: spandrel README]); NCNN port `nihui/realcugan-ncnn-vulkan` (932★, MIT, pushed 2023-03-12) [V: GitHub API]; VapourSynth support via vs-mlrt CUGAN wiki [V: https://github.com/AmusementClub/vs-mlrt/wiki].

**History + releases.** Repo (ailab) created 2021-05-07, **last push 2023-08-02** — maintenance frozen; repo-level license field: none (model weights carry their own terms) [V: GitHub API]. Timeline from README: GUI/web demos Feb 2022, NCNN 2022-02-17, Pro 2022-05, Squirrel GUI support 2022-08.

**Popularity.** 5,888★ on the ailab repo [V]. Huge in the Chinese community (bilibili ecosystem, Squirrel Anime Enhance); solid Western recognition as "the anime denoise/deblur specialist".

**Community sentiment.**
- [OP] SVFI doc rates **realCUGAN Pro = Anime ★★★★★ "Main anime SR"** — the only 5-star anime rating in their table (https://doc.svfi.group/en/pages/sr-models/).
- [OP] digitalFAQ (Selur, 2023-05-10): "RealCUGAN is meant for anime image super resolution … million scale anime dataset, same architecture as Waifu2x-CUNet … supports 2x/3x/4x … 2x supports 5 model weights" (https://www.digitalfaq.com/forum/video-restore/13498-realesrgan-versus-realcugan.html).
- [OP] videohelp forum thread 409432 ("Best (not super slow) upscalers for 1995-2010 Animes"): a participant: "i use RealCUGAN atm. A lot of people says RealESRGAN is best…" (https://forum.videohelp.com/threads/409432/).
- 2025-2026 OpenModelDB realists still build on it: renarchi's **Fallin Soft/Strong** (2x, cugan arch, 2025-09-14, CC-BY-4.0) and **Adore** (2x, real-cugan, 2026-04-11) are realtime 1080p successors trained on the CUGAN architecture [V: OpenModelDB JSONs].

**Anime suitability.** Best-in-class for degraded/SD/noisy anime (denoise levels + conservative mode + DOF preservation). 3x/4x exist but 2x is the mainline.

**spandrel.** Supported in base package incl. Pro weights [V].

**Verdict.** The universality specialist (all scales + denoise variants). Frozen upstream since 2023; recognition strong but China-centric; license situation murkier than community models.

---

## 4. waifu2x (2015) + original ESRGAN — historical context

- **waifu2x** (nagadomi): 28,241★, last push 2023-05-04 [V: GitHub API]. CUNet/upconv/upresnet backbones; the 2015 progenitor; still shipped everywhere (vs-mlrt wiki waifu2x page) but quality superseded by CUGAN (same architecture family, modern data): waifu2x-CUNet "cannot deblur, removes artifacts incompletely" per Real-CUGAN README table [OP/V: bilibili README].
- **ESRGAN** (xinntao, ECCV18 Workshops PIRM winner): created 2018-08-31, 6,573★, Apache-2.0, last push 2022-10-19 [V: GitHub API]. Introduced RRDB + ReluReal perceptual loss; every modern GAN anime model (AnimeSharp, Fatal-Anime, HFA2k, AniScale2-ESRGAN…) is an ESRGAN-arch fine-tune. Superseded by Real-ESRGAN (degradation modeling) and by community data-centric retrainings.
- Community verdict [OP]: both remain "classic conservative" options; SVFI keeps waifu2x "cunet and anime mainly for anime"; nobody recommends either as a primary 2026 choice for archival quality.

---

## 5. OpenModelDB — the community model hub

Live site: **671 models**, 202 tagged "Anime" [V: https://openmodeldb.info/ read 2026-10-27]. Source repo **OpenModelDB/open-model-database** (559★, pushed 2026-08-14, GPL-3.0 site) [V: GitHub API]. **No public per-model rating or download counters exist** — the schema has author/architecture/scale/date/license/tags only; "top by rating/downloads" is therefore **[UNVERIFIED]**; ranking below uses community-recognition signals instead.

Architecture census of the 241 anime/cartoon-tagged models [V, computed from repo JSONs]: esrgan 116, compact(SRVGG) 55, span 18, omnisr 7, dat 6, cugan+real-cugan 5, rcan 3, realplksr 3, hat 3, drct 3, swinir 3, srformer 3, mosr 2 …

Key anime models (verified metadata from `data/models/*.json`):

| Model | Author | Arch | Scale | Date | License | Notes |
|---|---|---|---|---|---|---|
| 4x-AnimeSharp | kim2091 | ESRGAN 64nf/23nb | 4x | 2021-12-28 | CC-BY-NC-SA-4.0 | The all-time anime x4 default; HF mirror 28 likes [V: HF api] |
| 2x-AnimeSharpV3 (+RCAN) | kim2091 | ESRGAN / RCAN | 2x | 2024-10-03 | CC-BY-NC-SA-4.0 | Successor line |
| 2x-AnimeSharpV4 | kim2091 | RCAN | 2x | 2025-01-07 | CC-BY-NC-SA-4.0 | "outperforms both versions of V3 in every capacity" [OP: model page]; HF 63 likes [V] |
| 4x-UltraSharpV2 | kim2091 | DAT | 4x | 2025-05-23 | CC-BY-NC-SA-4.0 | General+anime flagship; "best model I've ever trained" [OP: model page] |
| 2x-AnimeJaNai HD V3 (Compact/Ultra/SuperUltra) | the-database | compact 64nf/16nc | 2x | 2024-01-17 | CC-BY-NC-SA-4.0 | `.pth` 4,838,742 B, sha256 `af7307ee…` [V]; bundled in 3 GUI ecosystems |
| 2x-AnimeJaNai v2 / Strong v1 / SD v1 | the-database | compact | 2x | 2023-2024 | CC-BY-NC-SA-4.0 | V2 still widely used |
| AniScale 2 (ESRGAN/OmniSR/Compact) | sirosky | esrgan 67 MB / omnisr 3.4 MB / compact 2.4 MB | 2x | 2023-10-08 | CC-BY-NC-4.0 | Suite release with matching architectures |
| Ani4K (v1; "Ani4K v2" not in live DB [UNVERIFIED]) | sirosky | compact | 2x | 2023-06-18 | CC-BY-NC-4.0 | |
| AniSD Suite | sirosky | span | 2x | 2024-06-09 | CC-BY-NC-4.0 | DVD/SD sources; AMV101-recommended |
| 2x-HFA2k-LUDVAE-SPAN | helaman | span 48nf | 2x | 2023-12-26 | **CC-BY-4.0** | Fast + permissive; HFA2k family (OmniSR/Real-CUGAN/SRFormer/GRL variants) all CC-BY-4.0 |
| 2x-LD-Anime-Compact | skr/zarxrax | compact | 2x | 2022-12-22 | CC-BY-NC-SA-4.0 | Degraded/analog sources (noise, dot crawl, rainbows) |
| 4x-Fatal-Anime | twittman | ESRGAN | 4x | 2020-06-17 | CC-BY-NC-SA-4.0 | 2020-era classic |
| R-AnimeVideo v3 | — | — | — | — | — | **Not found in live DB** under that name; closest = xinntao's realesrganv2-animevideo / realesr-animevideov3 entries [V: repo listing]. Treat task's "R-AnimeVideo v3" as [UNVERIFIED] |
| 2x-Fallin Soft/Strong, 2x-Adore | renarchi | cugan / real-cugan | 2x | 2025-09/2026-04 | CC-BY-4.0 / CC-BY-NC-SA | Current realtime-1080p favorites (CUGAN lineage) |
| 90s Sonic 2x | pokepress | compact/realplksr | 2x | 2026-08-01 | MIT | Newest anime-tagged entries; show-specific |

---

## 6. SOTA architectures 2024-2026 and what the anime community actually runs

- **HAT** (CVPR 2023) — in spandrel base [V]; massive/VRAM-hungry. In anime production used by the "insane" VapourSynth encode scene via vs-mlrt ONNX (community practice; no official HAT ONNX in vs-mlrt's external-models release — that ships swinir/scunet/RIFE/etc. [V: https://github.com/AmusementClub/vs-mlrt/releases/tag/external-models]) → HAT-in-production claim is **[community practice, partially UNVERIFIED]**.
- **DAT** (2022/2023) — spandrel-supported [V]; powers Kim2091's UltraSharpV2 (2025) [V].
- **OmniSR** — spandrel-supported [V]; AniScale-2-OmniSR, HFA2OmniSR, pokepress's 2025 show-restorations [V].
- **SPAN** — spandrel-supported [V]; the fastest quality arch per AMV101 ("Some of the fastest architectures which work well for video are **Compact and SPAN**") [OP: https://www.amv101.com/guides/ai-upscaling, updated 2025-06-27]; used by HFA2k-LUDVAE-SPAN, AniSD Suite, StarSample V2 Lite.
- **RealViformer** (ECCV 2024, video SR attention) — repo Yuehan717/RealViformer, 94★, pushed 2024-07-22 [V: GitHub API]. **Not in spandrel's supported list** [V: spandrel README]; no anime-community production uptake found → dead end for this pipeline.
- Real-world production bench (official vs-mlrt wiki, RTX 4090, TRT FP16, 1080p RGB): SRVGG-compact real-esrgan xsx2 = **17→38 FPS** (1.1→2.5 GB VRAM); waifu2x-CUNet/CUGAN = **13.9→26 FPS** (4.4→12.3 GB at 3 streams); SwinIR-L 4x = **0.61 FPS, 15.4 GB** — transformer-class archs are archival-brutal even on a 4090 [V: https://github.com/AmusementClub/vs-mlrt/wiki/NVIDIA-GeForce-RTX-4090].

---

## 7. mpv AnimeJaNai (the-database)

- Repo renamed: **the-database/mpv-AnimeJaNai** (ex `mpv-upscale-2x_animejanai`). Created 2022-11-06; **770★ / 23 forks**; license "Other/NOASSERTION"; **pushed 2026-10-27** (day of this research); releases **3.9.0 (2026-10-27), 3.8.1, 3.8.0** — extremely active [V: GitHub API].
- What it is: custom mpv build (Windows player) + **Real-ESRGAN Compact ONNX** models run via **TensorRT (NVIDIA) or DirectML (AMD/Intel Arc)** for realtime 1080p→4K. Profiles: Quality (min RTX 4090) / Balanced (RTX 3080) / Performance (RTX 3060), default Balanced [V: README].
- Model set as quality signal: **2x_AnimeJaNai V3.1 HD/SD** in Compact/UltraCompact/SuperUltraCompact; V3 development = "seven months, over 100 release candidates", goals: faithful appearance, no oversharpening/ringing/aliasing, preserve intentional DOF blur, accurate line colors/darkness/thickness, soft shadow edges [V: README]. V3 rationale explicitly cites guide.encode.moe descaling: most HD anime is NOT native 1080p [V: README].
- Model provenance: OpenModelDB entry links `.pth` at release **3.0.0** asset `2x_AnimeJaNai_HD_V3_ModelsOnly.zip` (24,674,026 B, published 2024-01-17) [V: OMDB JSON + GitHub release API].
- Ecosystem adoption: **VideoJaNai** 281★ (web/GUI bundler, pushed 2026-10-27), **Tntwise/Real-Video-Enhancer** 2,358★ ("Interpolate, Upscale, Decompress, and Denoise videos easily on Linux/Windows/MacOS", pushed 2026-09-02) bundle AnimeJaNai models [V: GitHub API; AMV101: "It should be included with both VideoJaNai and Real Video Enhancer"].
- Community: JaNai Discord (linked in README); SVFI doc: "AnimeJaNai … is a **lighter CUGAN-like model with weak depth-of-field recognition**, so it can sharpen backgrounds" [OP: https://doc.svfi.group/en/pages/sr-models/].

---

## 8. Real archival pipelines the community runs (2025-2026)

**Canonical order of operations** (HD anime → 4K HEVC):
1. **Descale** to native production res (getnative analysis; BluBb-mADe `descale`; typical natives 720/810/900p inside "1080p" releases) [V: https://guide.encode.moe/encoding/descaling.html; mpv-AnimeJaNai README citing the same guide].
2. **Deband** (f3kdb / vs-deband) before heavy filtering — encode.moe warns descaling sources "full of broken gradients" (simulcasts) hurts unless handled first [V: guide.encode.moe].
3. **AI upscale** (2x compact-class model, or 4x for SD) — via chaiNNer batch, VapourSynth+vs-mlrt (vstrt/TensorRT), Real-Video-Enhancer GUI, or SVFI.
4. **Interpolate** (RIFE via vs-mlrt; 4090 TRT: 137→574 FPS multi-stream [V: vs-mlrt 4090 wiki]) — optional.
5. Encode (community: x265 10-bit; owner: NVENC HEVC 10-bit).

Tooling reality:
- **chaiNNer** (chaiNNer-org, GPL-3.0): 6,114★, **pushed 2026-10-27 — still maintained** [V: GitHub API]; the GUI batch upscaler with spandrel-backed PyTorch nodes.
- **VapourSynth + vs-mlrt** (AmusementClub, GPL-3.0): 489★, pushed 2026-09-25 [V]; official device benchmarks for RTX 4090/3090/2080Ti/P6000/7900XTX [V: wiki]; the encode-scene's production runtime (HAT/DAT/compact ONNX via TensorRT).
- **Upscayl** (AGPL-3.0): 51,170★ [V] — images only (NCNN), bundles realesr-animevideov3 x2/x3/x4 + RealESRGAN general [V: custom-models repo].
- **Real-Video-Enhancer** (GPL): 2,358★, cross-platform GUI (Linux!) bundling AnimeJaNai/CUGAN/RIFE [V].
- **Topaz Video AI** (commercial, Proteus/Iris photo-oriented): [OP] AMV101 "Why not Topaz?": "For live-action video, topaz is great! However, as a commercial product, they target their models towards mainstream uses, NOT for people trying to upscale anime videos … you can get much better results with models that have actually been trained on anime!" (https://www.amv101.com/guides/ai-upscaling, updated 2025-06-27). Same verdict direction in r/TopazLabs thread xe3b8b snippet. Community verdict: skip for anime.

Representative community threads (quotes):
- [OP] AMV101 guide (Zarxrax, updated 2025-06-27): "(2x) AnimeJaNai V3 — This model is for upscaling anime from 1080p to 4k. The effect is quite subtle, so it is really only effective on relatively high quality sources."; recommends **AniSD Suite** (DVD), **AnimeJaNai V3** (1080p→4K), **LD-Anime_Compact** (degraded sources).
- [OP] r/mpv 1afsemt "New comparison of some AI Models to Upscale Anime in 4K" (date UNVERIFIED — Reddit blocked; full content unreachable from this network).
- [OP] r/mpv 17rdr0k / 14r2cb8 — recurring Anime4K config threads (dates UNVERIFIED).
- [OP] digitalFAQ (2023-05-10, Selur): RealCUGAN "million scale anime dataset … same architecture as Waifu2x-CUNet".
- [OP] videohelp 409432: "i use RealCUGAN atm. A lot of people says RealESRGAN is best…" (2010s-anime upscaling practicality thread).
- [OP] SVFI model table: realCUGAN Pro ★★★★★ (Anime); AnimeJaNai "CUGAN-like … weak DOF"; animevideov3 "conservative, fast and stable".
- **Reddit scraping caveat**: all reddit.com routes (www/old/JSON/r.jina.ai/Wayback snapshots) returned network-policy blocks from this host; thread bodies above come from search snippets and non-Reddit mirrors, dates marked UNVERIFIED where applicable.

---

## 9. VERDICT — top 5 for "one universal spandrel-loadable anime model"

Criteria: quality · universality (fits ANY anime) · community recognition · VRAM/speed on RTX-5080-class (16 GB) · license.

| # | Model (exact artifact) | Arch/Scale | Quality | Universality | Recognition | Speed (5080, offline) | License |
|---|---|---|---|---|---|---|---|
| **1** | **2x_AnimeJaNai HD V3(.1) Compact** — `.pth` sha256 `af7307ee…4c8f`, from the-database/mpv-AnimeJaNai release 3.0.0 [V] | SRVGGNet compact, 2x, 4.8 MB | High-fidelity archival design (anti-oversharpen, DOF-preserving) | HD (720–1080p) excellent; weak on SD/DVD by design | **Highest current** — bundled in mpv-AnimeJaNai 770★ + VideoJaNai 281★ + Real-Video-Enhancer 2358★ + AMV101 + SVFI | ~real-esrgan-xsx2 class → tens of FPS 1080p class on 4090 [V: vs-mlrt bench]; trivial for a 5080 | CC-BY-NC-SA-4.0 (fine for personal archival) |
| **2** | **4x-AnimeSharp** — Kim2091, `.pth` via OpenModelDB (HF: Kim2091/AnimeSharp) | ESRGAN RRDB 23nb, 4x, ~67 MB | 2021-gen; sharpening-forward; superseded by V3/V4 but proven | **Any anime incl. SD/DVD** — the most forgiving single model | **All-time recognition king** of dedicated anime models; default in guides/packs for years | RRDB at 4x is the slow lane (~few FPS class at HD on 4090-class by arch family) — fine offline | CC-BY-NC-SA-4.0 |
| **3** | **Real-CUGAN Pro / denoise set** (bilibili/ailab Real-CUGAN weights) | CUNet U-Net, 2x/3x/4x + 4 denoise levels + conservative | Best degraded-source quality (SVFI ★★★★★) | **Most universal scale/denoise matrix**; DOF-preserving | Strong (CN-centric + ncnn port 932★); 2025-26 models still built on the arch | cunet class 13.9-26 FPS 1080p on 4090 [V: vs-mlrt]; VRAM to ~12 GB multi-stream — comfortable on 16 GB | No explicit license at repo [V] — caution |
| **4** | **2x-AnimeSharpV3** (kim2091, ESRGAN, 2024-10-03) | ESRGAN, 2x | Modern Kim2091 anime quality (V4 = RCAN is better but slower) | HD-focused, like #2's successor | High (same brand lineage, HF likes 28→63 across line) | ESRGAN-2x moderate | CC-BY-NC-SA-4.0 |
| **5** | **AniScale 2 — ESRGAN variant** (sirosky, 2023-10-08) | ESRGAN 67 MB (OmniSR/Compact siblings) | Clean upscale, suite-consistent | HD anime; OmniSR variant for speed | Solid 2023-24 community release | OmniSR variant = fast | CC-BY-NC-4.0 |

Honorable mentions: **HFA2k-LUDVAE-SPAN** (helaman, SPAN, **CC-BY-4.0** — best license + fast, recognition moderate); **realesr-animevideov3** (BSD-3-Clause — the only permissive option, ubiquitous in Upscayl/NCNN apps, conservative look, x2/x3/x4); **AnimeSharpV4** (RCAN 2025 — top quality, slow, newer = less proven universally); **Fallin/Adore** (renarchi 2025-26 CUGAN realtime favorites).

### #1 pick + justification

**`2x_AnimeJaNai_HD_V3` Compact (the-database) — `.pth`, sha256 `af7307eee19e5982a8014dd0e4650d3bde2e25aa78d2105a4bdfd947636e4c8f`.**

1. **Recognition (2026):** the anime-video model with the widest live community footprint — it is the default brain of three active ecosystems (mpv-AnimeJaNai 770★ updated the day of this research; VideoJaNai 281★; Real-Video-Enhancer 2358★), the first named recommendation in the community's canonical guide (AMV101, 2025-06), rated ★★★★☆-class in SVFI's commercial table, with an active author (release 3.9.0 = 2026-10-27) and Discord.
2. **Quality for archival:** designed against the exact failure modes that disqualify Real-ESRGAN-anime for archival (oversharpening, ringing, style drift, DOF destruction) — V3 trained through 100+ release candidates for faithfulness; "subtle" is a *feature* when the output feeds a 10-bit HEVC archival encode.
3. **Fit to the machine:** 4.8 MB SRVGGNet compact = the fastest recognized quality tier; realtime-class even on modest GPUs, so an RTX 5080 batch-encodes comfortably without VRAM pressure; spandrel supports SRVGGNet natively (plain `ModelLoader().load_from_file(...)`) [V: spandrel README].
4. **spandrel/ChaiNNer path:** the exact `.pth` is published (OpenModelDB → GitHub release 3.0.0) with a pinned sha256 — drop-in for `models/upscale/model.pth`.

**Tradeoff (must be stated):** HD-only by design — for SD/DVD titles the same author ships 2x_AnimeJaNai SD V1 (beta), and the community pairing is LD-Anime_Compact / AniSD Suite. If "ANY anime" must include a heavy SD library in ONE file with zero per-source logic, the runner-up covers it.

**Runner-up: `4x-AnimeSharp` (Kim2091)** — the all-time most-recognized dedicated anime model, x4, tolerates SD→HD and HD→4K alike, bulletproof reputation since 2021; cost = 2021-gen sharpening bias and RRDB-class speed. If the library skews SD or the pipeline wants a single never-think-about-it x4, pick this instead.

**Also noted:** Real-CUGAN Pro is the technical universality champion (scales × denoise levels) but frozen upstream since 2023 and license-ambiguous at repo level; `realesr-animevideov3` is the only BSD-licensed fallback (ubiquity ★, quality conservative); Anime4K stays the playback-side complement, not a pipeline model.

---

## Skills used
`brainstorming`, `grilling`, `build-with-exa`, `research`, `defuddle` (all loaded via `skill` tool; Exa itself returned HTTP 402 credits-exhausted — documented fallback to direct API fetches, DDG HTML, reader proxy, cloakbrowser, per standing Exa-fallback rules).

## Memory files read (complete, no skips)
- `~/.config/opencode/MEMORY.md` (full)
- `topics/`: stack-and-tools.md, workflow-git-pr.md, environment-infra.md, workflow.md, code-gotchas.md, communication-behavior.md, tech-gotchas.md, harness-server.md, job-search.md, remnawave.md, hh-ranking-ats-research.md, 2026-10-09-telegram-serverless.md, 2026-10-11-novum-organum-report-plan.md, 2026-10-12-slides-novum-organum-plan.md, 2026-10-14-opencode-v2-storage.md, 2026-10-15-binary-re-landmines.md, 2026-10-15-vision-ocr-scans.md, 2026-10-21-stepin-book-contract.md, 2026-10-25-locke-pirogov-referaty.md (19 files)
- `projects/`: 4q4r-profile.md, WMS.md, wms.md, anicli-go.md, anicli-py.md, bos-pro-bugs.md, cindex.md, ert.md, firefox-vpn-client.md, freeai-decompile.md, freepackages.md, habrparser.md, hhauto.md, home-global.md, interview-kb.md, iris-proposal.md, kolosyug.md, labboardvent.md, labs.md, labs-docker-stands.md, remnavawe_ansible.md, secureway_bot.md, shikimori-japanese-theme.md, stepik-whitehacker-course.md, teleparse.md, traffic_mimicry.md, tspu-research-2026.md, vibecoding-course.md, wauploader.md, xray-raw.md (30 files)

No memory files were written (facts reported to controller only).
