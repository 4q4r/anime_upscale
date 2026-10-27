# 04 — Encoder Technology Research: x265 10-bit vs NVENC (RTX 50-series era) for Anime Archival

Date: 2026-10-27. Scope: settle the owner decision "NVENC is trash, we pack into libx265 10-bit" with 2025–2026 evidence.
Pipeline context: RTX 5080 (Blackwell, 2× NVENC) + capable CPU; source = debanded/upscaled 10-bit anime; deliverable = personal archive (Plex/mpv); batch offline; speed secondary to size/quality.

Provenance labels: **[V]** = verified fact from a primary source (URL + number), **[C]** = community opinion (quoted source + date), **[U]** = unverified/thin evidence.

---

## 1. Blackwell NVENC generation: exact name and capabilities

**Name: "Ninth-generation NVENC" (9th gen), on Blackwell GPU cores GB202/GB203/GB205/GB206/GB207.** RTX 5080 = GB203 die with **2× NVENC + 2× NVDEC** engines [V].

NVIDIA Video Codec SDK 13.0/13.1 — NVENC Application Note capability tables (docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvenc-application-note/):

| Feature | Ada (8th gen) | **Blackwell (9th gen)** | Since |
|---|---|---|---|
| H.264 4:2:0 8/10-bit | Y / N | **Y / Y** (High10 encode is NEW in Blackwell) | 10-bit H.264: Blackwell only |
| H.264 4:2:2 | N | **Y** | Blackwell |
| HEVC 4:2:0 Main10 | Y | Y | Pascal |
| HEVC 4:2:2 | N | **Y** | Blackwell |
| HEVC B-frames | Y | Y (hw max 5, ref modes each+middle) | Ampere/Turing |
| HEVC MV-HEVC (stereo) | N | Y | Blackwell |
| AV1 8/10-bit 4:2:0 ≤8192×8192 | Y | Y | Ada |
| AV1 4:2:2 / 4:4:4 | N | **N** (AV1 remains 4:2:0 only) | — |
| AV1 B-frames | Y (hw max 7) | Y | Ada |
| AV1 lookahead + lookahead_level | N | **Y** (new in Blackwell) | Blackwell |
| Temporal filtering (tf_level 0/4) | Y (HEVC) | Y (HEVC + AV1 + H.264) | SDK 12.x/13.x |
| Lookahead depth | Y | Y (max = 31 − B-frames; lookahead_level 0–3) | SDK 7 |

Quality deltas (NVIDIA's own numbers) [V]:
- "Ninth-generation NVENC … offers a **5% improvement in video quality on HEVC and AV1 encoding (BD-BR)**" vs Ada (blogs.nvidia.com, CES 2025 post, 2025-01-07).
- "**AV1 Ultra High Quality (UHQ) mode** … achieves **5% more compression at the same quality**"; UHQ is "backwards-compatible with the GeForce RTX 40 Series" (same blog; SDK 13.0 blog, 2025-01-30) — i.e., UHQ is mostly an SDK-side tuning recipe, not new silicon.
- MiikaH (August 2026 test, Blackwell, SDK 13.1): "RTX 50 series using same settings usually produces **5% lower bitrate for same quality than RTX 40 series** encoder" [V].

UHQ tuning recipe (NVIDIA SDK 13.0 blog + rigaya/NVEnc issue #679, 2025-03) [V]: lookahead + temporal filtering combined; HEVC UHQ = **5 B-frames, middle B as reference, adaptive I/B disabled, fixed GOP**; AV1 UHQ = **7 B-frames**. rigaya's equivalent flag set: `-b 7 --tf-level 4 --lookahead 32 --lookahead-level 3 --no-i-adapt --no-b-adapt --bref-mode middle`.

### Does 4:2:2 matter for a 4:2:0 anime archival deliverable? — NO.

Honest technical analysis:
- 4:2:2 is an *encode-input/chroma-format* capability. It exists so cameras' native 4:2:2 (10-bit) footage can be archived/edited without chroma decimation mid-pipeline [V — NVIDIA creator blog, 2025-01-07: "The 4:2:2 format provides double the color information… ideal balance for video editing workflows"].
- The owner's deliverable is HEVC Main10 **4:2:0** (Plex/mpv, TV-consumable). Anime BD/Web sources are 4:2:0. Encoding 4:2:0→4:2:0 gains **nothing** from a 4:2:2-capable encoder; a 4:2:2 encode would only *cost* ~33% more bitrate for chroma that gets thrown away on playback.
- The 4:2:2 and H.264-High10 features are pro-camera-workflow features, zero relevance to this pipeline. The features that actually matter for anime from 9th-gen NVENC are: **AV1 lookahead, temporal filtering, UHQ tune, +5% BD-BR**.

---

## 2. Quality-at-equal-size: x265 10-bit vs NVENC HEVC for anime

### 2.1 The strongest controlled data: rigaya's 2025.12 quality comparison (VERIFIED numbers)

Source: rigaya (author of NVEncC) — rigaya.github.io/vq_results/, details page vq_results_details.html (test conditions), raw data downloaded from github.com/rigaya/vq_results `results/` and re-analyzed for this report (interpolation script preserved at /tmp/opencode/anime_research/rigaya/analyze.py).

Test conditions [V]:
- **Anime clip: sakuranotoki_op.mp4, 1920×1080, 30fps, 3194 frames, 16.3 Mbps** (also a scenery clip; anime data used here).
- x265 4.1+200: `--crf [x] --preset veryslow --output-depth 10` — **plain defaults, no anime tuning** (no psy-rd/no-sao overrides).
- NVEncC 9.07, **RTX 5050 (GB207 = 9th-gen NVENC)**, driver 581.80: "quality" preset = `--preset quality --tune uhq --multipass 2pass-full --lookahead 32 --weightp --bref-mode each`, HEVC `-b 8`, 10-bit, rate control `--qvbr` (NVIDIA quality-VBR).
- Metrics: VMAF, SSIM, SSIMULACRA2, butteraugli.

**Bitrate needed to reach x265-veryslow-10bit quality levels (SSIMULACRA2 anchor, interpolated from rigaya's published points):**

| x265 anchor | x265 bitrate | NVENC HEVC 10bit | NVENC AV1 10bit | SVT-AV1 p3 10bit |
|---|---|---|---|---|
| CRF16 (SSIMULACRA2 86.0) | 8056 kbps | **7818 kbps (−3%)** | **7099 kbps (−12%)** | out of range |
| CRF20 (82.7) | 5022 kbps | 5398 kbps (+7.5%) | 5050 kbps (+0.6%) | +22.5% |
| CRF24 (77.2) | 3171 kbps | 3440 kbps (+8.5%) | 3310 kbps (+4.4%) | +20.7% |
| CRF28 (72.3) | 1998 kbps | 2690 kbps (+35%) | 2493 kbps (+25%) | +23.3% |
| CRF32 (63.8) | 1246 kbps | 1811 kbps (+45%) | 1754 kbps (+41%) | +39% |

Butteraugli cross-check (detail/color-sensitive, lower = better) at the same anchors: CRF16 → NVENC HEVC +5.8%, **NVENC AV1 −6.7%**; CRF20 → HEVC +1.0%, AV1 **−15.7%**; CRF24 → HEVC −4.1%, AV1 **−20.3%**.

Same CQ/QP index ≠ same quality (mechanism of the owner's "CQ16 = huge files" complaint): at QP/CRF index 16, NVENC HEVC 10-bit produced **14,898 kbps vs x265 8,056 kbps (+85%)** for roughly comparable perceptual quality on this clip [V]. CQ and CRF scales are not interchangeable; NVENC needs ~CQ 20–22 to land at x265-CRF16 quality (interpolated).

**Mandatory caveats for honest reading:**
1. The x265 side is **untuned**. Real archival x265 uses anime-tuned params (psy-rd 2.0, no-sao, deblock −1:−1, aq-mode 1/3, qpoffs −2) — precisely the knobs that fix line-art softness and dark-gradient banding on anime (VCB-Studio tutorial 10 + guides.vcb-s.com §7 [V]). A tuned x265 at the same bitrate will look better than the default one rigaya measured, so real-world NVENC gap is somewhat larger than the table suggests, especially on line art and grain.
2. rigaya's NVENC "quality" config enables **temporal filtering (tf-level 4)** via UHQ. rigaya himself: "Temporal filtering is good for higher compression rate, but **in general it might be not good for grainy targets**" (rigaya/NVEnc #679, 2025-03-25) [C/V]. NVIDIA's own SDK 13.1 ffmpeg doc: temporal filtering is "**Recommended for: Natural content (real-world video, not synthetic/animated content)**" [V — docs.nvidia.com/video-technologies/video-codec-sdk/13.1/ffmpeg-with-nvidia-gpu/]. For anime (synthetic line art + possible dither/grain), tf can smear fine lines and texture — the metric tables above do not fully price this in.
3. One OP clip, 1080p30. OPs are cleaner than battle/gradient-heavy episodes.

### 2.2 What NVENC actually loses on anime (failure modes)

- **Dark-scene blocking/banding (gradients):** NVIDIA's own forum documented the class: NVENC HEVC dark regions "noticeably and distractingly blocky and jumpy"; bitrate had to be "nearly double that of libx264" to match black stability (forums.developer.nvidia.com #61116, 2018 — old generation, but this is the canonical statement of the failure mode) [V/C]. Same symptom reported on anime re-encodes: "few shows… heavily blocky, especially with darker scenes" at NVENC CQ24 (r/handbrake, 2022-08) [C].
- **Mechanism:** NVENC has **no equivalent of x265's psy-rd/psy-rdoq psychovisual RDO**, and its spatial AQ (aq-strength 1–15) is a different, cruder bit-allocation than x265's aq-mode 1/2/3 with tuned strength. x265 encoders use "adaptative quantization enabled by default. But not NVENC. So you may want to enable it with `-spatial-aq 1`" (u/vegansgetsick, r/ffmpeg 2025-05) [C]. aq-mode 3-style dark-bias (the anime banding fix) does not exist in NVENC [V — NVIDIA ffmpeg doc lists only spatial/temporal AQ].
- **Line art:** flat-region AQ is where NVENC under-allocates; thin dark lines on light flats are the anime killer. No 2025–2026 source claims Blackwell fixed this; the improvement is uniform ~5% BD-BR [V].
- **Grain/dither:** NVENC temporal filtering (auto-on in UHQ) averages across frames — explicitly contraindicated for "synthetic/animated" content by NVIDIA [V] and for grainy targets by rigaya [V]. If the owner's deband step removes grain, this risk shrinks but does not vanish (dither patterns are synthetic detail).
- **Detail "less noisy" look:** "In visual comparison NVENC video looks less noisy but still preserves all the important detail" (MiikaH, Aug 2026, on game footage) [V] — that "less noisy" look is the tf/AQ smoothing signature.

### 2.3 Community size/quality gap estimates 2025–2026 (mixed content)

- "Intel and Nvidia encoders do a decent job and will probably get **equivalent quality at roughly 25% larger file size**" for modern grain-free footage; "**GPU-based encoders choke on excessive film grain**" (u/agressiv, r/ffmpeg, 2025-01-25) [C].
- Apples-to-apples 2h 1080p movie: x265 slow CRF28 → **1.87 GB**, NVENC → **2.92 GB (+56%)** at equal CQ-ish targets (r/ffmpeg, 2025-02-24) [C].
- "At a given bitrate you'll always be able to get better quality with software encoding compared to HW, excluding… super high bitrates (>100 Mbps)" (u/nmkd, r/ffmpeg 2025-05-03); "nvenc at best quality will be 1 to 2 PSNR points below x265 preset medium" (u/vegansgetsick, same thread) [C].
- MiikaH (Aug 2026, gameplay, Blackwell): VMAF+SSIM favor libx265 slow; hevc_nvenc slightly ahead in PSNR; libx265 medium clearly weakest of the three [V].
- **Anime-specific 2025 datapoint (r/AV1, u/NebulaAccording8846, 2025-10-07):** equal-filesize SVT-AV1 vs NVENC AV1 on an anime episode: "Outside of that fight scene, there was absolutely no visible difference comparing the 3 versions (bloated original, AV1 NVENC, SVT-AV1)… for encoding anime with AV1… Software doesn't make any sense whatsoever" [C].

**Summary of the gap (anime, 2025–2026, Blackwell):** at archival quality levels (x265 CRF 14–18 territory) vs *untuned* x265, Blackwell NVENC UHQ is within roughly **−12%…+10% bitrate** (codec-dependent); at CRF 24+ (starved) territory the gap is **+25–45%**; vs *tuned* (VCB-style) x265 the gap widens by an unquantified margin (thin evidence; expect high single-digit to low double-digit % additionally, content-dependent) [U — no controlled tuned-x265-vs-Blackwell-NVENC benchmark found]. Pre-Blackwell generations sit at +25–56% [C].

---

## 3. NVENC AV1 (Blackwell) vs x265 10-bit for anime

- **Compression:** AV1 NVENC on Blackwell measured **equal or better than default x265 veryslow 10-bit at CRF16–24 anchors** on rigaya's anime clip (−12% to +4.4% bitrate at equal SSIMULACRA2; −6.7% to −20.3% at equal butteraugli) [V]. Against tuned x265 the advantage should shrink [U].
- **AV1 UHQ specifics:** 7 B-frames, tf-level 4, lookahead 32 (fixed GOP, no adaptive I/B) [V]. Same temporal-filtering anime risk applies.
- **Software AV1 still wins at the top end:** "software AV1 encoding remains undisputedly best quality… `libsvtav1 preset 2` vs `av1_nvenc p7 uhq`: software tops every metric and visual comparison. …with faster presets such as `libsvtav1 preset 8`, NVENC is ahead in nearly all metrics" (MiikaH, Aug 2026, gameplay) [V]. Counter-datapoint on a 1440p60 gameplay clip: NVENC AV1 slowest preset VMAF 85.53 vs SVT p5 84.07 (r/AV1, 2025-07-09) [C] — preset parity matters in every comparison.
- **Anime community adoption for archival 2025–2026:** VCB-Studio's public guide (guides.vcb-s.com, © 2026) teaches **only x264/x265** — no AV1/VTM chapter exists (basic chapters 1–12, advanced 30–31 Expr/Masktools) [V]. JET guide recommends x265 CRF 12–14 for anime archival [V]. r/AV1 anime testers compare SVT-AV1 vs NVENC AV1 but report no archival migration [C]. Nobody in the canonical anime-encoding sources recommends AV1-NVENC archival over x265 10-bit as of 2026-10.
- **Playback compatibility for the personal archive:** mpv (dav1d) — no concern. NVDEC decodes AV1 since Ampere; the RTX 5080 NVDEC handles AV1 fine [V]. **Plex is the friction point:** AV1 *encoding* output doesn't exist in Plex; AV1 direct-play is client-dependent — a 2025-04 Android Plex app update broke direct play of AV1 (and HEVC Main10) for some users (forums.plex.tv, 2025-04-29) [C]; Plex HEVC transcode itself is still "experimental" and Plex-Pass-gated (Plex announcement, 2025-01-21) [V]. **HEVC Main10 remains the max-compatibility choice for a Plex-served archive**; AV1 is safe for mpv-only archives.
- **Decode cost:** AV1 10-bit SW decode on weak clients is heavier than HEVC; on the owner's own mpv/NVDEC setup, irrelevant.

**Net:** NVENC AV1 is the *strongest* NVENC option for anime (best NVENC quality-per-byte measured), but it changes the archive codec for a still-fragile Plex ecosystem, and the top software AV1 tier (SVT p2) still beats it.

---

## 4. Tuning: can NVENC hit libx265-CRF16-like sizes at comparable quality?

**Answer: at the x265-CRF16 quality anchor — yes, approximately, on Blackwell with UHQ (metrics say ±10%, one clip, untuned x265 baseline). It was never achievable by keeping CQ16 — the file-size explosion the owner saw is the CQ-scale mismatch, not an NVENC defect per se.**

### x265 side (canonical anime archival settings)

VCB-Studio canonical line (tutorial 10, x265 v2.9-era, still the community reference) [V — vcb-s.nmm-hd.org tutorial-10 PDF]:
```
x265-10b --y4m -D 10 --preset slower --deblock -1:-1 --ctu 32 --qg-size 8 --crf 15.0
  --pbratio 1.2 --cbqpoffs -2 --crqpoffs -2 --no-sao --me 3 --subme 5 --merange 38
  --b-intra --limit-tu 4 --no-amp --ref 4 --weightb --keyint 240 --min-keyint 1
  --bframes 6 --aq-mode 1 --aq-strength 0.8 --rd 5 --psy-rd 2.0 --psy-rdoq 1.0
  --rdoq-level 2 --no-open-gop --rc-lookahead 80 --scenecut 40 --qcomp 0.65
  --no-strong-intra-smoothing
```
Current VCB-Studio guide rules (guides.vcb-s.com/basic-guide-07, © 2026) [V]:
- SAO: CRF ≥21 → `--sao`; 18–21 → `--limit-sao`; **≤17 → `--no-sao`** (SAO blurs at low CRF).
- deblock: BDRip `-1:-1`; CRF anchors: BDRip examples **CRF 14–16** (veryslow/slower), webrip 18–21.
- psy-rd 1.0–2.5 by texture; aq-mode 3 + strength 0.7 classic (or aq-mode 1 + 0.8 when CRF ≤15); qpoffs −2…−3; bframes 8–10; ref 4–5; qcomp 0.65–0.72; rc-lookahead 80; no-open-gop; keyint ≈ 10 s × fps.
- Grain: VCB workflows deband (f3kdb/GradFun3) and re-add synthesized grain in VapourSynth; x265 `--tune grain`/psy-rdoq governs encoder-side grain retention (kokomins: psy-rdoq is "the key to preserve grain"; `--tune grain` overdoes it on slow presets) [C/V].

Adapted command for THIS pipeline (deband→upscale→RIFE-60fps 10-bit source; ffmpeg):
```
ffmpeg -f rawvideo -pix_fmt p010le -s 3840x2160 -r 60 -i - \
  -c:v libx265 -preset slow -crf 16 -D 10 \
  -x265-params no-sao=true:deblock=-1,-1:psy-rd=2.0:psy-rdoq=1.0:rdoq-level=2:\
aq-mode=1:aq-strength=0.8:cbqpoffs=-2:crqpoffs=-2:bframes=8:ref=5:\
no-open-gop=true:keyint=600:min-keyint=1:rc-lookahead=80:qcomp=0.65:\
scenecut=40:no-strong-intra-smoothing=true:pbratio=1.2
```
(keyint 600 = 10 s at 60 fps; CRF 15–16 per owner's quality bar; preset slow ↔ slower/veryslow = speed/size tradeoff on a "capable CPU"; aq-mode 1 at CRF ≤16 per VCB, switch to aq-mode 3/0.7 if any CRF drift above 16.)

### NVENC side (if used)

Verified option semantics (NVIDIA SDK 13.1 ffmpeg doc) [V]:
- `-preset p7` = best quality; `-tune uhq` = UHQ (auto-enables lookahead + temporal filter); `-tune hq` = plain HQ (no tf).
- Rate control: **VBR-CQ** = `-rc vbr -cq N -b:v 0 -maxrate <cap>` — "closest match to CRF behavior and in my test produces clearly best results" (MiikaH, Aug 2026) [V]. constqp (`-rc constqp -qp N`) is worse for archival (no per-frame adaptation). Always set `-maxrate` — without it FFmpeg VBR-CQ "behaves unpredictably at lower CQ values" (MiikaH) [V].
- `-multipass fullres` = per-frame 2-pass, measurable quality bump, non-deterministic output possible [V].
- `-bf 4..5 -b_ref_mode each` (HEVC; middle for AV1 UHQ); `-spatial-aq 1 -aq-strength 8..10` (perceptual only; "may reduce PSNR"; never combine with `-temporal-aq`) [V].
- `-rc-lookahead` max = 31 − bf; `-lookahead_level 3` for max quality.
- **Anime adjustment:** UHQ's temporal filter is contraindicated for animated content [V]. Use `-tune hq` + explicit `-rc-lookahead 32 -lookahead_level 3 -b_ref_mode each -bf 4 -spatial-aq 1` instead of UHQ — or `-tune uhq` plus explicit `-tf_level 0` override (explicit option should win over tune defaults in FFmpeg nvenc; verify on the first batch — option-precedence ordering is an implementation detail [U]).

**Best-effort NVENC archival line (HEVC, for the "acceptable cases" column):**
```
-c:v hevc_nvenc -preset p7 -tune uhq -multipass fullres \
  -rc vbr -cq 21 -b:v 0 -maxrate 120M \
  -bf 4 -b_ref_mode each -spatial-aq 1 -aq-strength 10
```
CQ calibration: x265 CRF16 ↔ NVENC **CQ ≈ 20–22** on anime (interpolated from rigaya data [V]); calibrate per-library: encode one episode at CQ 20/22, compare SSIMULACRA2/VMAF against the x265 CRF16 reference, pick.

### Expected size deltas at same visual quality (headline numbers)

| Comparison | Size delta | Source class |
|---|---|---|
| NVENC HEVC CQ16 vs x265 CRF16 (same index!) | **≈ +85% NVENC** | [V] rigaya anime QP16: 14.9 vs 8.1 Mbps |
| NVENC Blackwell HEVC UHQ at x265-CRF16 quality vs default x265 CRF16 | ≈ −3%…+8% | [V] rigaya interpolation |
| Same, vs VCB-tuned x265 | larger than above, unquantified | [U] |
| NVENC Blackwell AV1 UHQ at x265-CRF16–24 quality | −20%…+5% (AV1 better) | [V] rigaya interpolation |
| Starved regime (CRF/CQ 28+) | +25…45% NVENC | [V] rigaya; [C] matches |
| Pre-Blackwell NVENC (Turing–Ada) vs x265 | +25% (clean), +56% observed, ~2× for dark blacks | [C]/[V] mixed 2018–2025 |

---

## 5. Community canonical 2025–2026

- **VCB-Studio** (the reference anime group; guides.vcb-s.com live/© 2026): x265 **10-bit** (`-D 10`), BDRip anchors **CRF 14–16**, presets slower/veryslow, `--no-sao` at CRF ≤17, `--deblock -1:-1`, `--psy-rd 2.0` (+ psy-rdoq 0.5–1.0), aq-mode 1/0.8 at CRF ≤15 (or aq-mode 3/0.7), qpoffs −2…−3, bframes 8–10, ref 4–5, no-open-gop, keyint 10 s [V]. Their BDRip workflow debands (f3kdb/masks, adaptive strength) then **adds synthesized grain** back (VCB Nyaruko-san case study, blog.chlc.cc 2023-01-17) [V/C]. Reference encoder in the guide = x265 Kyouko 3.5-AC (AmusementClub mod) [V].
- **No AV1/VTM chapter** anywhere in VCB-Studio's public guide as of 2026-10 [V]. No public VCB release switched to AV1 found.
- **JET Encoding Guide** (jaded-encoding-thaumaturgy): x265 CRF **12–14** for transparent anime, `--psy-rd 2.0 --psy-rdoq 1.5`, `--no-sao`, `--bframes 16`, deblock −1:−1 [V].
- **Kokomins guide** (2019, still widely quoted): preset slow, CRF 16–20 by content tier, aq-mode 3 "pretty much a must", 10-bit "to eliminate banding" [C/V].
- **guide.encode.moe**: tooling/descaling-focused (its encoder chapter predates Blackwell) [U].
- Hardware encoders are simply absent from all of these guides' archival workflows [V].

---

## 6. VERDICT

**Default for the owner's batch archival pipeline: libx265 10-bit. The owner's core instinct is confirmed by 2026 evidence; his size complaint has a precise, fixable cause (CQ-scale mismatch), and the RTX 50-series generation he forgot is "Ninth-generation NVENC (Blackwell)" — a real but modest (~5% BD-BR + UHQ) improvement that does NOT overturn the verdict for archival.**

1. **Default encoder + settings:** libx265 10-bit, preset slow (slow/slower/veryslow ladder by CPU budget — batch job, speed secondary), **CRF 15–16**, with VCB-style anime params: `no-sao, deblock=-1:-1, psy-rd=2.0, psy-rdoq=1.0, rdoq-level=2, aq-mode=1 (aq-strength 0.8) or aq-mode=3 (0.7), cbqpoffs/crqpoffs=-2, bframes=8, ref=5, no-open-gop, keyint=600 @60fps, rc-lookahead=80, qcomp=0.65`. Do NOT use `--tune animation` (US-cartoon oriented; VCB + kokomins both warn).
2. **Conditions under which NVENC (HEVC or AV1) is acceptable:** quick-view/scratch copies, remote-stream transcodes, deadline-bound batches where ×15–40 encode speed matters, or sources that are clean/debanded with no grain and moderate gradients — then: `hevc_nvenc -preset p7 -tune uhq -multipass fullres -rc vbr -cq 21 (calibrate 20–22) -b:v 0 -maxrate 120M -bf 4 -b_ref_mode each -spatial-aq 1 -aq-strength 10`, consider `-tf_level 0` for anime; AV1 variant (`av1_nvenc`, same skeleton) only if the archive is mpv-first and Plex direct-play friction is accepted.
3. **Is "equalize NVENC size to x265-CRF16 quality" achievable?** **Partially — not a total false hope, but not what the owner wants for archival.** On Blackwell with UHQ/VBR-CQ and CQ raised to ~21, measured quality-per-byte at the CRF16 anchor reaches parity with *default* x265 (−3…+8%) [V]. Three reasons it still fails the archival bar: (a) the fair baseline is **tuned** x265 (psy-rd/no-sao/dark-AQ), which NVENC cannot match structurally — no psy-RDO, no dark-scene aq-mode-3 analog; (b) UHQ's temporal filter is officially for "natural content, not synthetic/animated" — the exact content class being archived [V]; (c) at CRF 24-equivalent and below (harder episodes, fan-service gradients, heavy effects), the gap re-opens to +25–45% [V]. Community consensus (nmkd/vegansgetsick/agressiv, 2025) still holds: at a given bitrate, software wins [C].

**Bottom line for the pipeline:** keep NVENC where it is already used for what it's good at (nothing in this pipeline's archival leg), move the final pack to libx265 10-bit CRF 15–16 slow+VCB params; expect ~45–55% smaller files than current NVENC CQ16 at *equal-or-better* quality, and ~±10% vs default-x265 parity if NVENC must ever be used.

---

## Sources

**Primary/verified:** docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvenc-application-note/ (capability tables, per-engine fps); docs.nvidia.com/video-technologies/video-codec-sdk/13.1/ffmpeg-with-nvidia-gpu/ (presets, UHQ, VBR-CQ, AQ, tf semantics, archiving recipe); blogs.nvidia.com CES 2025 + SDK 13.0 blog (9th-gen, 4:2:2, 5% BD-BR, UHQ, 5080=2×NVENC); developer.nvidia.com SDK 13.0 blog (UHQ recipe); rigaya.github.io/vq_results/ + github.com/rigaya/vq_results (2025.12 data, RTX 5050, test conditions; this report's interpolation from published points); github.com/rigaya/NVEnc #679/#733 + GPUFeatures/rtx5090.txt (hw matrix, tf caveat); miikah.fi/utils/video_encoder_comparison (Aug 2026, Blackwell, ffmpeg 9.0/SDK 13.1); guides.vcb-s.com/basic-guide-07 (+ /advanced TOC) and vcb-s.nmm-hd.org tutorial-10 PDF (VCB canon); x265.readthedocs.io presets; support.plex.tv + forums.plex.tv (HEVC transcode experimental 2025-01-21; AV1 direct-play regression 2025-04).
**Community opinion:** r/ffmpeg 1i9oa9z (2025-01), 1ix45my (2025-02), 1ke5vmy (2025-05); r/ffmpeg 1mq9akm (RTX 5080 AV1 UHQ vs SVT-AV1, 2025-08-14 — full thread body unreachable from this network; scope/setting verified via search capture, conclusion [U]); r/AV1 1lvf2lg (2025-07), 1o02gmj (anime SVT vs NVENC AV1, 2025-10); r/handbrake wgk75w (2022); forums.developer.nvidia.com #61116 (2018); kokomins.wordpress.com (2019); JET guide; blog.chlc.cc (2023).
**Unverified:** tuned-x265-vs-Blackwell-NVENC controlled benchmark (does not publicly exist as of 2026-10); ffmpeg `-tf_level 0` override precedence over `-tune uhq`; RTX 5080-specific (vs 5050) anime benchmarks.

---

## Skills used
`brainstorming` (scope framing; subagent-bounded per workflow), `grilling` (spec-hole check against the brief's 6 questions), `build-with-exa` (Exa-first discipline; Exa 402 → documented fallback chain), `research` (primary-source doctrine, provenance labels), `planning-with-files` (plan/findings files in /tmp/opencode/anime_research/).

## Memory files read (full coverage)
`~/.config/opencode/MEMORY.md` (full) + `memory/topics/`: stack-and-tools.md, workflow.md, workflow-git-pr.md, communication-behavior.md, environment-infra.md, code-gotchas.md, tech-gotchas.md, remnawave.md, hh-ranking-ats-research.md, harness-server.md, job-search.md, 2026-10-09-telegram-serverless.md, 2026-10-11-novum-organum-report-plan.md, 2026-10-12-slides-novum-organum-plan.md, 2026-10-14-opencode-v2-storage.md, 2026-10-15-binary-re-landmines.md, 2026-10-15-vision-ocr-scans.md, 2026-10-21-stepin-book-contract.md, 2026-10-25-locke-pirogov-referaty.md + `memory/projects/`: anime_upscale.md, 4q4r-profile.md, WMS.md, anicli-go.md, anicli-py.md, bos-pro-bugs.md, cindex.md, ert.md, firefox-vpn-client.md, freeai-decompile.md, freepackages.md, habrparser.md, hhauto.md, home-global.md, interview-kb.md, iris-proposal.md, kolosyug.md, labboardvent.md, labs.md, labs-docker-stands.md, remnavawe_ansible.md, secureway_bot.md, shikimori-japanese-theme.md, stepik-whitehacker-course.md, teleparse.md, traffic_mimicry.md, tspu-research-2026.md, vibecoding-course.md, wauploader.md, wms.md, xray-raw.md. (No memory files written — facts returned to controller only.)
