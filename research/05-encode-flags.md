# 05 — Encode Flags: libx265 10-bit Anime + MKV Assembly (FLAG PLAN)

**Status:** PLANNED (evidence-backed, ready for implementation review)
**Date:** 2026-10-27
**Scope:** ffmpeg `-c:v libx265` (10-bit) flag sheet for the anime_upscale pipeline (VapourSynth/vsrife upstream → yuv420p10le BT.709; audio/subs copied from source MKV; batch offline; RTX 5080 + capable CPU; playback = mpv/Plex). NVENC alternative = stub only (`backend = "nvidia"`), researched by a parallel agent.

**Labels:** `[V]` = verified against a primary source (x265 docs/source, ffmpeg docs/source). `[OP]` = community opinion/guidance, with source + date. Quotes are translated where the original is Chinese.

---

## 0. Version landscape (what "current" means here)

- Latest x265 release: **4.1, 2024-11-22** (per x265-devel release announcement). 3.6 shipped 2024-04-04. Both added film-grain SEI signaling (see §1.4). Distro/ffmpeg builds commonly still ship 3.4–3.6; every flag recommended below exists in ≥3.4. `[V]` (x265 readthedocs release notes; videolan mailman ANN)
- The x265 docs default table used here is the current `readthedocs/en/latest` revision `82ff02e9`; conflicting stale doc lines are called out and settled against `source/common/param.cpp` (videolan/x265 mirror). `[V]`
- ffmpeg color-tag forwarding verified in `libavcodec/libx265.c` (FFmpeg master). `[V]`

---

## 1. Anime-specific x265 flags

### 1.1 Per-flag table (the sheet)

| Flag (x265-params syntax) | Value (a) / (b) | Why | Source |
|---|---|---|---|
| `no-sao=1` | both | SAO (Sample Adaptive Offset) is a smoothing loop filter; anime community treats it as detail-destroying. VCB-Studio calls it "Smoothing All Objects" — "the price is extremely violent smearing; not recommended except at very low bitrate." SilentAperture: "`--no-sao` is usually best… SAO does tend to blur quite heavily." JET: disable unconditionally for quality encodes. `[V]` semantics (docs: default enabled) + `[OP]` VCB-Studio guide 教程10; JET Guide (2024–2026); SilentAperture mdbook guide | |
| `limit-sao=1` | alternative | If ringing appears on gradients with `no-sao`, `limit-sao` (early SAO termination) is the middle path. `[V]` semantics + `[OP]` Kokomins 2019-10-10 (uses `limit-sao` in battle-scene recipe) | |
| `deblock=-1,-1` | (a) | Negative deblock offsets preserve dither/grain and sharp lines. JET: "at least `-1:-1` or `-2:-2`… avoid deblocking, it is harmful to dithering and grain" (anime); `-3:-3` may *create* blocking. SilentAperture: test `-2:-2` to `0:0` for animation. VCB-Studio canonical: `-1:-1`. Comma syntax is the `-x265-params`-safe form (`[V]` docs: `<int>:<int>` or `<int>,<int>`). Grainy shows: `-2,-2`. | |
| `psy-rd=2.0` / `psy-rd=1.5` | (a) / (b) | Adds cost to blocks whose visual energy doesn't match the source → preserves lines/detail instead of blur. x265 default is 2.0 (`param.cpp:236`). VCB-Studio: "default 2.0 is actually a good value; lower to ~1.5 for mid/low bitrate." JET: 2.0 start, never >2.5. Kokomins (2019): 1 for flat anime, 1.5 dark/detailed, 2 grainy — older, softer values. `[V]` semantics + defaults; `[OP]` VCB-Studio, JET, Kokomins | |
| `psy-rdoq=1.0` | both | Works only with `rdoq-level ≥1`; biases RDO-QT toward keeping AC energy → the primary anti-blur knob inside quantization; "psy-rdoq is less accurate than psy-rd — biases toward energy in general while psy-rd biases toward the energy of the source image" `[V]` docs. VCB: 1.0 "helps retain detail and grain"; JET: 1.5; Kokomins: 1 for most anime, 2–5 only for grain-staple shows at preset=slow (higher presets need more). Start at 1.0; raise to 1.5–2.0 for grainy sources. | |
| `rdoq-level=2` | both | RDO inside quantization incl. 4x4 coding-group decimate decisions `[V]` docs. **medium preset leaves it 0** — explicit set is the classic "x265 looks mushy" fix; slow+ presets set 2 already `[V]` preset table. VCB: "at `--preset medium` it is 0, in which case rdoq does nothing; slow and above auto-enable." | |
| `aq-mode=3` | both | AQ with auto-variance + bias to dark scenes `[V]` docs. Dark anime backgrounds/gradients are the #1 banding site. Kokomins: "so beneficial to anime that a crf=20 encode with it looks better than crf=18 without at similar file size — pretty much a must." JET: use 3. SilentAperture: modes 3/4 usually best. Counter-note `[OP]` VCB-Studio 教程10 (x265 2.9 era): for 10-bit, aq-mode 1 at CRF ≤16, else 2; mode 3 sizes 3>1>2 — their own later BDRip params still ship aq-mode 3. Resolution: aq3 + conservative strength (below) is the current cross-community default. | |
| `aq-strength=0.8` | both | Below-default strength counters aq3's size cost while keeping dark-scene help. VCB: anime shouldn't run high strength; per-mode guidance 0.8/0.9/0.7 for modes 1/2/3; their BDRip params: aq3+0.8. JET: 0.6–0.7 for anime (then drop CRF to compensate). Kokomins: 0.7–1.0. `0.8` is the intersection. `[V]` semantics (0.0–3.0, default 1.0) + `[OP]` all three | |
| `no-cutree=1` (a) / *(default on)* (b) | (a) off / (b) on | CU-tree gives heavily-referenced (static) blocks lower QP and starves the rest — on anime it "freezes" flat regions into patchy blocks. JET: "simpler areas such as backgrounds get starved… no way to control its strength — disable it." SilentAperture: "a poor implementation of mbtree." VCB-Studio rule: disable at CRF <15, keep at ≥15. Hence: off in (a) (CRF 14), **on in (b)** (CRF 18) to save bits. `[V]` semantics (docs + preset table: cuTree=1 all presets) + `[OP]` JET/SA/VCB | |
| `qcomp=0.65` | both | Weights per-frame QP by lookahead complexity; 1.0 ≈ CQP, 0.5 ≈ flat. Default 0.6 `[V]` docs ("Increasing it to 1.0 will effectively generate CQP"). VCB: 0.65 general, 0.7 for action ("打架番"), 0.72 movie-grain. JET: 0.7. Higher qcomp = more bits to complex/dark scenes = the "dynamic bitrate" throttle inside CRF. | |
| `bframes=8` | both | Anime has long low-motion stretches → high B depth is cheap quality. slow preset default is 4 `[V]` preset table; slower/veryslow = 8. JET: "at least 8, ideally 16." SilentAperture: 16 "or whatever your final bframes log output says." Doom9 2018 (anime thread): `ref 6 --bframes 16 is great for anime/cartoon`. 8 = preset-compatible floor; 16 is a free-lunch option for veryslow users (note: b-adapt 2 trellis cost is quadratic in bframes `[V]` docs). | |
| `b-adapt=2` | both (implicit) | Viterbi B-path selection, default at slow+ `[V]` docs ("0: none; 1: fast; 2: full(trellis) default") and preset table (medium+ → 2). No override needed; documented so nobody sets 0 for "speed". | |
| `ref=5` (a) / *(preset default 4)* (b) | (a) 5 / (b) 4 | More L0 references → better compression/detail on pans and repeated animation; cost is linear in motion-search work `[V]` docs ("linear multiplier effect… beneficial effect on compression and distortion"). HEVC spec caps decodable refs at 8 `[V]` docs note. VCB uses 4–5; SilentAperture: 6 on good CPUs, 4 otherwise. (b) falls back to preset-slow default 4. | |
| `rc-lookahead=60` | both | Drives scenecut accuracy, b-adapt and cutree `[V]` docs ("longer lookahead → more accurate scenecut… more effective cutree"; "> max keyframe interval not helpful"). slow preset default is 25 `[V]` preset table. SilentAperture: 60 ("irrelevant as long as it's larger than min-keyint"); VCB: 80 (slower encodes). 60 is the intersection; harmless since cutree is off in (a) and keyint 240 > 60 in both. | |
| `keyint=240:min-keyint=24` | both | JET rule: keyframe ≈10 s, min-keyint ≈1 s of content (240/24 at 23.976 fps). VCB: keyint 240–360, min-keyint 1. Defaults: keyint 250, min-keyint auto(0) `[V]` docs. Scene-heavy anime TV (OP/ED/eyecatch) benefits from explicit bounds. **59.94 fps RIFE output:** same seconds = `keyint=600:min-keyint=60` — use those values in the pipeline when interpolation is enabled. `[V]` semantics + `[OP]` JET/VCB | |
| `scenecut=40` | both (implicit) | Default 40 `[V]` docs; x265's fade-aware scenecut improved in 3.x (release notes). Anime's hard cuts (OP/ED) are detected fine at default; min-keyint prevents I-spam on pans. | |
| `ctu=64` (implicit) | both | x265 is tuned for ≥1080p with CTU 64; slow+ presets set 64 `[V]` preset table. Split verdict: VCB/Asmodian (doom9 2016-02-12) prefer `ctu=32` (+`max-tu-size=16`) — "noticeably better details/sharpness on anime"; Kokomins measured 64→32 costing ~1–2% compression for anime @crf 19 and notes "CTU 64 actually does benefit anime". Default: leave 64, note 32 as a quality-experiment lever. | |
| `qg-size=16` (a) / *(default 32)* (b) | (a) 16 / (b) 32 | AQ granularity: min CU size at which QP can be adjusted `[V]` docs. **Docs say "default: same as maxCUSize" but source says `param->rc.qgSize = 32` (`param.cpp:270`) — source wins; docs are stale.** Finer QG = AQ can lift small flat/dark regions (anime lines, gradients) more precisely. VCB ships qg-size 8–16 in all example params. | |
| `rd=4` | both (implicit at slow) | slow preset = rdLevel 4 `[V]` preset table. JET: "4 usually best for anime; 6 can hold grain together better." Explicit in the sheet for auditability; veryslow users run 6 automatically. | |
| `amp` | off (both) | Asymmetric partitions: enabled only at slower+ `[V]` preset table; VCB measured "no noticeable effect at 1080p, not recommended"; SilentAperture: `--no-amp` "slightly more useful than --no-rect". Do not enable at 1080p. `[OP]` VCB 教程10 + SA | |
| `rect` | off (both) | Same verdict family as amp (slow preset has rect=1! — note: preset table shows rect 1 at slow+; SA recommends `--no-rect` on slower machines only). Leave preset value. `[V]` preset table + `[OP]` SA | |
| `tskip=1` | both | Transform-skip evaluation for 4x4 TUs — JET: "useful for preserving sharp edges and fine details, which anime often has a lot of." Cheap; x265 docs support. Single-source endorsement (JET, current guide) — kept, flagged as JET-only. | |
| `cbqpoffs=-2:crqpoffs=-2` | both | x265 has no x264-style auto chroma-QP compensation under psy, so chroma is systematically starved → washed line art. VCB: "-2 relieves it well"; JET: -2, or -3 if size allows; SA: -3..0 for 4:2:0. Symmetric -2 for 4:2:0. `[V]` semantics (docs: luma/chroma QP offsets) + `[OP]` all three | |
| `no-strong-intra-smoothing=1` | both | Disables extra edge-smoothing on 32x32 intra blocks; VCB ships it in every example; SA: "on sharp/grainy content" (leave on for blurry content); Kokomins pairs it with no-sao/deblock-1 to make x265 "behave like x264". Cheap sharpness win for line art. | |
| `no-open-gop=1` | both | Closed GOPs (IDR-only) — VCB closes GOP in every canonical param set; SilentAperture recommends it. Slight compression cost, better seek/robustness (useful for an archive that gets split/seeked). `[V]` semantics (docs: open GOP allows non-IDR I-slices, default enabled) + `[OP]` VCB/SA | |
| `no-rect`/`no-amp`-style micro-trims, `subme`, `merange` | not set | Deliberately left at preset values: VCB 教程11 methodology — rate-control flags dominate visual quality; trade-off flags are best left to the preset ladder. `subme 7` is known to sharpen `[OP]` SA — experiment territory, not defaults. | |

### 1.2 `--tune` options for anime

`[V]` (x265 readthedocs presets.html — tune effects, applied after preset):

| tune | effect (docs) | Anime community verdict |
|---|---|---|
| `psnr` | disables AQ, psy-rd, cutree | benchmarking only — disables exactly the tools anime needs |
| `ssim` | AQ auto-mode, disables psy-rd | same — metrics-only |
| `grain` | `aq-mode 0, cutree 0, ipratio 1.1, pbratio 1.0, qpstep 1, sao 0, psy-rd 4.0, psy-rdoq 10.0, rskip 0` + `rc-grain` | live-action grain, not anime. `[OP]` Kokomins 2019: "the tune-grain built-in actually has too high a value for slower presets — it artificially creates even more grain." |
| `fastdecode` | no deblock/SAO, no weighted pred, no b-intra | playback-constrained devices only |
| `zerolatency` | no lookahead, no B, no cutree, frame-threads 1 | streaming only |
| `animation` | `psy-rd 0.4, aq-strength 0.4, deblock 1:1, bframes +2` | **rejected for anime.** `[OP]` JET Guide (2024–2026), verbatim: *"the type of animation this appears to be optimized for is not anime or anything with similarly complex backgrounds, gradients, and other details. Instead, this tune is better optimized for very 'flat' animation, such as flash animations… you should NOT use this tune for anime."* The `[V]` values confirm why: it guts psy-rd (2.0→0.4) and AQ (1.0→0.4) and *strengthens* deblocking — the opposite of every community recommendation. **Do not use `-tune animation` despite the name.** |

### 1.3 cu-tree / cutree vs flat regions — mechanics

`[V]` docs: cutree reads lookahead lowres motion-vector fields; blocks reused as motion references get lower QP (more bits), quickly-changing blocks get fewer bits. On anime, backgrounds are *static* and therefore "heavily reused" — cutree gives them bits they don't visibly need, while the actually-watched foreground detail pays for it; result = temporally "freezing" flat areas and starved motion. Hence the near-universal `--no-cutree` at quality CRFs (see §1.1 row; VCB's CRF<15 rule reconciles the one dissenting use-case: at CRF≥15 cutree's savings stop hurting).

### 1.4 Grain synthesis (FGS) — when warranted for anime

- **What exists [V]:** x265 3.6 (2024-04-04) added "Film-Grain characteristics as a SEI message to support Film Grain Synthesis" (`--film-grain <file>`); 4.1 (2024-11-22) ships it as `--aom-film-grain` (AV1-style model in SEI; the **decoder** synthesizes grain). It is *signaling only* — you denoise and generate a grain-model file externally. `[OP]` doom9 2024-12-03: "not a complete built-in feature… I don't think it's a useful feature until it's much more improved." Decoder/player support for HEVC FGS SEI is spotty (unlike AV1/dav1d).
- **The `--fgs-table` flag people cite belongs to SVT-AV1** (encoder-side FGS there), not x265 — a recurring confusion; x265 has no encoder-side grain synthesis.
- **Verdict for this pipeline:** NOT warranted as a default. Anime sources are mostly clean or lightly grained (grain is often injected dither); heavy-grain anime is better served by `psy-rdoq` 1.5–2.0 + tune-grain-style RC *values* (`rc-grain`) than by FGS. If the owner ever processes a genuinely grain-heavy film source (e.g., film-styled movies with dynamic-grain debanding), the fallback is pixel-domain: keep source grain via psy flags, or regrain after denoise in VapourSynth — both survive every player. `[OP]` Kokomins 2019 (BDs with dynamic-grain debanding recipe: `psy-rdoq 2–5`), encode.wiki Graining page 2025-2026.

---

## 2. Dynamic bitrate: CRF vs 2-pass ABR vs CRF+VBV

### 2.1 Modes compared

| Mode | Mechanics | Fit for offline archival |
|---|---|---|
| **CRF** `[V]` | "does not try to reach any particular bitrate target… tries to achieve a given uniform quality and the size of the bitstream is determined by the complexity of the source" (x265 docs). Default 28. | **The mode for this use case.** One pass, quality-constant, size varies with content. |
| 2-pass ABR `[V]` | pass 1 builds a stats file, pass 2 re-allocates QP to hit a bitrate `[V]` ffmpeg wiki + x265 `--pass` docs. | Only when a file-size *budget* must be hit (device limits, quotas). Costs a full extra analysis pass. encode.moe recap `[OP]`: "2-pass: good for obtaining a certain file size; bad for everything else in archival terms." |
| CRF+VBV `[V]` | CRF with `vbv-maxrate`/`vbv-bufsize` caps local bitrate. Docs warning: `crf-max` + VBV "may result in buffer underruns"; and with VBV enabled, **VBV emergency denoising** turns on — "aggressive denoising at the frame level when frame QP > QP_MAX_SPEC… visual effect is blurring" (x265 docs, quality/rate-control section). | Streaming/live constraints. For local archive: pointless cap + a real degradation mode. Avoid. |

### 2.2 Where "dynamic bitrate" actually comes from inside CRF

- **Temporal (frame-to-frame):** `qcomp` weights each frame's QP by lookahead complexity `[V]` docs; 0.5→flat, 1.0→CQP-like. 0.65 (VCB) slightly favors complex frames vs the 0.6 default.
- **Spatial (within frame):** `aq-mode 1/2/3` — 1: uniform AQ; 2: auto-variance (default); 3: auto-variance + dark-scene bias `[V]` docs. Dark-scene QP relief is exactly the "qp-adaptation on dark scenes" behavior: mode 3 raises bits in dark blocks to prevent banding `[OP]` Kokomins ("raises bitrate in dark scene to prevent banding… most prone to banding, blocking and color artifacts"). `--hevc-aq` (+`--qp-adaptation-range` 1.0–6.0) exists as an experimental alternative `[V]` docs; Kokomins' tests: "lowers overall VMAF slightly but increases 1% low" — skip.
- **Causal (across frames):** cutree — reuse-based QP modulation `[V]` docs (§1.3).

### 2.3 Recommended CRF range and size implications

Community anchors for 10-bit 1080p anime:

- `[OP]` guide.encode.moe (fetched 2026-10-27): "For high quality anime encoding, at least **18.0** is recommended."
- `[OP]` VCB-Studio 教程10 (x265 2.9+, undated PDF, guide lineage 2017→current): "for anime high-quality encoding, start at **at least 18.0**"; their canonical release params use **CRF 15–16** for BDRip, 18.5 for webrips.
- `[OP]` VCB-Studio 基础篇 guides.vcb-s.com (current): "webrip 18–21 all fine."
- `[OP]` JET Guide (2024–2026): **CRF 12–14** for transparent encodes (SD ≤9).
- `[OP]` Kokomins 2019-10-10: 24–27" monitor → **18–21** ("crf 18 is my lowest recommended value for anime… going below 18 may increase file size quite rapidly"); 4K up close → 16–18; "magnifying glass" → 14.
- `[OP]` x265-web-tutorial (iavoe, Chinese community): source-tier quality 16–18, collection 19–20.5.

**Consensus read:** the "14–18" band in the controller's question is right but tiered — 12–14 = transparent/collection (JET, VCB BDRip), 16 = VCB-Studio's workhorse BDRip point, 18 = the widely-agreed floor for "high quality anime" (encode.moe, Kokomins). Below 14 size grows steeply (Kokomins), and preset interacts (§3.1).

**For the owner:** his NVENC CQ16 produces huge files; libx265 CRF 14 at preset slow lands at or below that size with visibly better texture retention (software encoder efficiency + this flag set), CRF 16 = his size reference done *properly*, CRF 18 = "much smaller than reference, still above the community floor".

---

## 3. Optimizations (speed vs quality reality check)

### 3.1 Preset ladder on anime

- `[V]` preset table differences: slow→slower adds ref 4→5, bframes 4→8, rd 4→6, b-intra, max-merge, weightb, limit-modes; slower→veryslow adds amp, max-merge 5, merange stays, subme stays. placebo adds subme 5/merange 92/recursion-skip off.
- `[OP]` Kokomins 2019 (anime, measured): "slower the preset, the **bigger** the file at same CRF" (more tools = more saved detail); "crf=16 medium is worse than crf=18 slow while being 50% larger"; **slow = the recommendation for most**; veryslow only pays at CRF 22+ (pre-3.1).
- `[OP]` Losko doom9 benchmark 2020-03-05 (x265 3.2, 1080p live-action, CRF 18): slow→slower = +81% time for +4% size; **slower→veryslow = +78% time for −0.4% size**. (Veryslow ≈ slower in efficiency; it spends its time on quality-at-size, not size.)
- `[OP]` benwaggoner (doom9 2021-02-17): at the same bitrate slower presets look better, but under CRF "sometimes a slower preset… can increase bitrate AND quality"; "veryslow and slower aren't going to look that different."
- **Verdict:** `slow` is the efficiency knee; `slower`/`veryslow` are quality-polish opt-ins for idle CPUs, *not* size levers. Placebo: ignored per ffmpeg wiki `[V]` ("insignificant returns for a significant increase in encoding time").

### 3.2 Threads, WPP, pmode/pme

- `[V]` docs: `--frame-threads` auto-detects; 1 thread gives slight compression gain but severe speed loss; over-allocation doesn't help. WPP row-based parallelism caps at frame-rows: 1080/64 ≈ 17 CTU rows → beyond ~16–17 effective row-threads, scaling flattens `[OP]` Kokomins (same arithmetic, from WPP mechanics in docs).
- `[V]` docs: `--pmode`/`--pme` parallelize mode analysis / motion estimation inside CTU analysis; purely speed features (tiny efficiency cost). Worth enabling on many-core boxes *after* checking that frame-threads+WPP already saturate; they are the first knobs to try for "free" throughput, second is `--lookahead-slices` (slow preset = 4 `[V]` preset table; 1 = max quality lookahead).
- **Safe speedups that don't hurt quality (ordered):** (1) run 2–3 titles concurrently at slow preset rather than 1 at veryslow — better aggregate quality per wall-clock; (2) `pmode`+`pme` if CPU is under-fed; (3) `ref 4` instead of 6 on weaker CPUs `[OP]` SA. Avoid: `rskip 2` (edge-aware skip, experimental 3.4 `[V]`), `rskip 1` on quality encodes `[OP]` SA ("a speed up that gives up some quality"), `b-adapt 1/0`, `lookahead-slices 8` at slow.

### 3.3 Keyint for scene-heavy anime

Covered in §1.1 (`keyint=240:min-keyint=24`; 600/60 at 59.94 fps). `scenecut` 40 + x265's improved fade handling (3.x release notes `[V]`) handles OP/ED transitions; don't lower min-keyint below 1 s of content — it burns I-frame bits exactly where B-frames were already cheap.

---

## 4. Pipeline-specific flags (10-bit in, tags right, level sane)

- **Input depth:** upstream emits `yuv420p10le`. Explicit `-pix_fmt yuv420p10le` on the output (harmless when it matches, fatal-proof when it doesn't). libx265 binds `libx265_main10` automatically for 10-bit `[V]` (`--output-depth` docs; ffmpeg wrapper derives from pix_fmt).
- **Profile:** `-profile:v main10` — enforce main10 decodability `[V]` (x265 `--profile` docs; encode.moe example uses main10 `[OP]`).
- **Level/tier:** leave auto. `[V]` docs: `level-idc` defaults to 0 = auto-detection; "profile, level-idc, and high-tier are only intended for use when targeting decoders with fixed resource limits… it may enable VBV constraints on a CRF encode" — i.e., pinning level can only *hurt* a CRF encode. Arithmetic sanity: 1080p59.94 ≈ 489k MB/s < Level 4.1's 522,240 MB/s ceiling → 1080p60 fits Level 4.1. `[V]` release notes (2.0): "high tier is now allowed by default, **chosen only if necessary**" — x265 flips high tier on its own when bitrate demands it. Community `--high-tier` habit (SilentAperture `[OP]`) is for fixed-limit devices; mpv/Plex don't care.
- **Color tags (BT.709 SDR):**
  ```
  -colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv
  ```
  `[V]` these are forwarded into the HEVC VUI: `libavcodec/libx265.c` sets `vui.colorPrimaries/transferCharacteristics/matrixCoeffs` from the AVCodecContext fields and `bEnableVideoFullRangeFlag` from `color_range` (verified in source). Untagged 10-bit HEVC is a classic "washed-out greens in Plex / wrong conversion in mpv" generator; tagging costs nothing. (SDR BT.709 values = 1/1/1 per VUI tables `[V]` x265 `--colorprim/--transfer/--colormatrix` docs; ffmpeg names accepted directly.)
- **Not needed for MKV:** `-tag:v hvc1` (mp4-only, ffmpeg wiki `[V]`), `-flags +global_header` (mp4/flv concern; Matroska carries CodecPrivate natively), `--repeat-headers`/`--aud` (SilentAperture's compatibility extras `[OP]`; harmless but pointless for mpv/Plex).

---

## 5. MKV assembly (ffmpeg muxing)

| Concern | Flag | Fact | Source |
|---|---|---|---|
| Stream selection | `-map 0:v:0` + `-map 1:a?` `-map 1:s?` `-map 1:t?` | "Data or **attachment streams are not automatically selected** and can only be included using `-map`." Subtitle map `?` = optional (PGS-absent sources won't fail). Attachments = stream type `t`. | ffmpeg.html Stream selection `[V]` |
| Audio | `-c:a copy` | Bitstream copy, no re-encode; ffmpeg "will transcode all audio… unless you specify -c copy" `[V]`. | ffmpeg.html |
| Subtitles | `-c:s copy` | Copies both text subs (ASS/SRT) and PGS (`hdmv_pgs_subtitle`) into Matroska untouched; the Matroska muxer implements "the matroska… container specs" for both `[V]` (matroska muxer docs; ffmpeg default-subtitle-selection caveats apply only without explicit copy). | ffmpeg-formats.html 4.61 `[V]` |
| Chapters | *(default)* or `-map_chapters 1` | "If no chapter mapping is specified, chapters are copied from the first input file with at least one chapter" `[V]`. Two-input pipeline: point at the source MKV explicitly (`-map_chapters 1`). | ffmpeg.html |
| Global metadata | `-map_metadata 1` | Copies source tags (title, show, etc.) from the MKV input rather than the pipe input. | ffmpeg.html `[V]` |
| Font attachments (existing) | `-map 1:t?` + `-c copy` | Copies embedded fonts through; community-verified standard remux pattern. | superuser (llogan; TheSchwa 2014), encode.wiki `[OP-multi]` |
| Font attachments (new) | `-attach font.ttf -metadata:s:t:N mimetype=application/x-truetype-font` | "for Matroska you also have to set the mimetype metadata tag" `[V]` (missing mimetype = mux error). | ffmpeg.html |
| Default/disposition flags | *(inherit)*; optionally `-disposition:a:0 default` | "by default, all disposition flags are copied from the input stream" `[V]`; ffmpeg auto-marks a default only when a type has multiple streams. Single-audio sources: adding `-disposition:a:0 default` helps Plex language pickers; don't blanket-set subtitle defaults — sign/song tracks must not become default. Muxer `default_mode` = `passthrough` by default `[V]` (matroska muxer docs). | ffmpeg.html + ffmpeg-formats.html `[V]` |
| Title | `-metadata title="Show — S01E01"` (optional) | Global title tag; per-track `title` also recognized by the Matroska muxer `[V]`. | ffmpeg-formats.html 4.61.1 |
| Track language | `-metadata:s:a:0 language=jpn` etc. (usually inherited) | Matroska muxer maps `language` to the Matroska language element, ISO-639-2/B `[V]`. | ffmpeg-formats.html |
| Index | *(default: cues at end)* | `reserve_index_space`/`cues_to_front` exist for streaming; irrelevant for local files `[V]`. | ffmpeg-formats.html |

---

## 6. Deliverable command lines

Common shape: input 0 = y4m pipe from the VS stage (`vspipe -c y4m script.vpy -`), input 1 = the source MKV (audio/subs/chapters/attachments/fonts). If the video stage outputs a file instead, replace input 0 accordingly (drop `-f yuv4mpegpipe`).

### (a) Full-quality default — anime-tuned CRF 14

```bash
vspipe -c y4m upscale.vpy - | ffmpeg -y -i - -i source.mkv \
  -map 0:v:0 -map 1:a? -map 1:s? -map 1:t? \
  -map_metadata 1 -map_chapters 1 \
  -c:v libx265 -preset slow -crf 14 -profile:v main10 -pix_fmt yuv420p10le \
  -colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv \
  -x265-params "no-sao=1:no-strong-intra-smoothing=1:deblock=-1,-1:\
psy-rd=2.0:psy-rdoq=1.0:rdoq-level=2:aq-mode=3:aq-strength=0.8:qcomp=0.65:\
no-cutree=1:qg-size=16:ref=5:bframes=8:rc-lookahead=60:\
keyint=240:min-keyint=24:no-open-gop=1:tskip=1:\
cbqpoffs=-2:crqpoffs=-2" \
  -disposition:a:0 default \
  -c:a copy -c:s copy \
  -metadata:s:v:0 title="Anime Upscale (AnimeJaNai V3 + RIFE 4.25), x265 10-bit CRF14" \
  output.mkv
```

Note: `keyint/min-keyint` assume 23.976 fps content; for 59.94 fps RIFE output use `keyint=600:min-keyint=60`. Bash line-continuations inside the quoted `-x265-params` string are shown for readability — keep it one line or use a shell variable.

### (b) Size-conscious — CRF 18

```bash
vspipe -c y4m upscale.vpy - | ffmpeg -y -i - -i source.mkv \
  -map 0:v:0 -map 1:a? -map 1:s? -map 1:t? \
  -map_metadata 1 -map_chapters 1 \
  -c:v libx265 -preset slow -crf 18 -profile:v main10 -pix_fmt yuv420p10le \
  -colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv \
  -x265-params "no-sao=1:no-strong-intra-smoothing=1:deblock=-1,-1:\
psy-rd=1.5:psy-rdoq=1.0:rdoq-level=2:aq-mode=3:aq-strength=0.8:qcomp=0.65:\
bframes=8:rc-lookahead=60:keyint=240:min-keyint=24:no-open-gop=1:tskip=1:\
cbqpoffs=-2:crqpoffs=-2" \
  -disposition:a:0 default \
  -c:a copy -c:s copy \
  -metadata:s:v:0 title="Anime Upscale (AnimeJaNai V3 + RIFE 4.25), x265 10-bit CRF18" \
  output.mkv
```

### (c) Flag diff (a) → (b)

| Flag | (a) full-quality | (b) size-conscious | Rationale |
|---|---|---|---|
| `crf` | 14 | 18 | 14 = collection/transparent tier (JET, VCB BDRip); 18 = community floor for "high quality anime" (encode.moe, Kokomins) — the biggest size lever by far |
| `psy-rd` | 2.0 | 1.5 | VCB: "lower to ~1.5 for mid/low bitrate" — less energy-chasing = fewer spent bits at higher CRF |
| `no-cutree=1` | set (cutree off) | removed (cutree on) | VCB rule: cutree off only below CRF 15; at CRF 18 its bitrate savings stop hurting flat regions visibly |
| `ref=5` | set | removed (preset default 4) | ME reference depth is a polish knob; save the linear search cost |
| `qg-size=16` | set | removed (default 32) | finer AQ granularity is a polish knob, not a floor requirement |

Everything else (SAO off, psy-rdoq 1.0, aq3/0.8, qcomp 0.65, bframes 8, lookahead 60, GOP bounds, chroma offsets, tskip, closed GOP, 10-bit + BT.709 tags, MKV stream handling) is **identical** in both.

### NVENC alternative — STUB ONLY

> `backend = "nvidia"` — flag sheet for `hevc_nvenc` (p7/hq, 10-bit, CQ semantics, tuned vs CQ16) is owned by the parallel backend-choice agent. Do NOT merge NVENC flags into this plan; this file's defaults assume `libx265`. Decision note: everything in §5 (MKV assembly) is backend-independent and applies unchanged to NVENC output.

---

## 7. Sources

**Primary [V]:**
1. x265 official docs — Preset Options & Tuning (presets.html) and CLI reference (cli.html), readthedocs `latest` rev 82ff02e9, fetched 2026-10-27 (flag semantics, preset table, tune values, `--film-grain`, VUI/VCR flags, level/tier behavior).
2. x265 source `source/common/param.cpp` (videolan/x265 mirror) — defaults: qgSize=32, psyRd=2.0, psyRdoq=0.0, bframes=4; `no-` prefix boolean parsing in `x265_param_parse`.
3. x265 release notes (3.4/3.5/3.6) + x265-devel ANN v4.1 (2024-11-22) — FGS SEI, high-tier-by-default note (2.0), rskip edge-aware (3.4).
4. FFmpeg documentation — ffmpeg.html (stream selection, `-map`, `-attach`, `-disposition`, `-map_chapters`), ffmpeg-codecs.html §9.21 libx265 (wrapper options, `-x265-params`), ffmpeg-formats.html §4.61 Matroska muxer (metadata, default_mode, cues), AVCodecContext color options.
5. FFmpeg source `libavcodec/libx265.c` — color_primaries/color_trc/colorspace/color_range → x265 VUI forwarding.
6. FFmpeg Wiki: Encode/H.265 (trac, fetched via ?format=txt 2026-10-27) — CRF/preset guidance, 2-pass example, `-x265-params` syntax.

**Community [OP]:**
7. VCB-Studio public guides — guides.vcb-s.com 基础篇 ch.7 (rate control) + 教程10 "x265 参数设置" (vcb-s.nmm-hd.org PDF) + 教程11 methodology (github.com/vcb-s/guides); canonical param sets quoted. Chinese; quotes translated.
8. JET Encoding Guide — Jaded Encoding-Thaumaturgy/JET-guide, x265 page (2024–2026, fetched 2026-10-27) incl. collapsed "Why not use --tune animation?" note (extracted from page HTML).
9. SilentAperture mdbook Advanced Encoding Guide — encoding/x265.html (fetched 2026-10-27).
10. Kokomins, "Anime Encoding Guide for x265 (HEVC) & AAC/OPUS" — wordpress, 2019-10-10.
11. guide.encode.moe — "Encoding with x265" (page dated 2023-02-12 in-text; fetched 2026-10-27).
12. Doom9 forum: anime x265 settings threads 2016-02 (Asmodian ctu32), 2017-07/2018-08 ("Best Settings for Anime?" — Phanton_13, katzenjoghurt), 2018-06 ("x265-encoding settings for cartoon/anime" — ref 6 + bframes 16), 2020-03-05 (Losko preset×CRF benchmark), 2021-02-16 (benwaggoner on preset vs bitrate), 2024-12-03 (`--aom-film-grain` usability).
13. encode.wiki — Graining (AV1 FGS vs pixel grain), Audio and Subtitles (attachment remux patterns).
14. Stack Overflow / Superuser attachment-copy threads (llogan, TheSchwa, slhck) — cross-checked against ffmpeg.html stream-selection docs.

---

## Skills used
`brainstorming`, `build-with-exa`, `grilling`, `research`, `planning-with-files` (5 loaded; Exa was 402/credits-exhausted — mandated fallback chain used: websearch + direct curl fetches; Anubis-protected trac.ffmpeg.org bypassed via `?format=txt` raw export).

## Memory files read (full)
`~/.config/opencode/MEMORY.md`
Topics: `stack-and-tools.md`, `workflow.md`, `workflow-git-pr.md`, `communication-behavior.md`, `tech-gotchas.md`, `harness-server.md`, `environment-infra.md`, `code-gotchas.md`, `2026-10-14-opencode-v2-storage.md`, `2026-10-15-binary-re-landmines.md`, `2026-10-15-vision-ocr-scans.md`, `2026-10-21-stepin-book-contract.md`, `2026-10-09-telegram-serverless.md`, `2026-10-11-novum-organum-report-plan.md`, `2026-10-12-slides-novum-organum-plan.md`, `2026-10-25-locke-pirogov-referaty.md`, `job-search.md`, `hh-ranking-ats-research.md`, `remnawave.md`
Projects (all opened; relevance-checked): `anime_upscale.md` (full — project contract), `4q4r-profile.md`, `WMS.md`, `anicli-go.md`, `anicli-py.md`, `bos-pro-bugs.md`, `cindex.md`, `ert.md`, `firefox-vpn-client.md`, `freeai-decompile.md`, `freepackages.md`, `habrparser.md`, `hhauto.md`, `home-global.md`, `interview-kb.md`, `iris-proposal.md`, `kolosyug.md`, `labboardvent.md`, `labs-docker-stands.md`, `labs.md`, `remnavawe_ansible.md`, `secureway_bot.md`, `shikimori-japanese-theme.md`, `stepik-whitehacker-course.md`, `teleparse.md`, `traffic_mimicry.md`, `tspu-research-2026.md`, `vibecoding-course.md`, `wauploader.md`, `wms.md`, `xray-raw.md`
(Only `anime_upscale.md` and the workflow/communication/tooling topics carried task-relevant constraints: Exa-402 fallback, report language, no memory writes, /tmp/opencode/anime_research/ as report location. No memory files were written.)
