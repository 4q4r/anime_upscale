# 03 — VapourSynth Runtime + Neural Plugin Ecosystem (researched 2026-10-27)

Research target: exact technical understanding of VapourSynth, how neural models run inside it,
performance reality, and canonical scripts — for RIFE/GMFSS/GIMM interpolation (and neural
upscaling) inside VS, driven from Python, on RTX 5080 / Linux, feeding ffmpeg.
Owner's current pipeline: pure PyTorch Practical-RIFE + ffmpeg NVENC (~/projects/anime_upscale).

**Provenance labels used throughout:** `[official]` = upstream docs/README/source quote; `[verified-api]` = GitHub/PyPI API fetch this session; `[community]` = forum/issue/Reddit anecdote; `[unverified]` = could not be confirmed. All API numbers fetched 2026-10-27 via authenticated GitHub API + PyPI JSON API.

**Search-tool disclosure:** Exa MCP returned HTTP 402 "credits exceeded" on the mandated probe call (matches memory note from 2026-10-22); per standing rules the fallback chain used free APIs (GitHub REST via `gh`, PyPI JSON, raw.githubusercontent.com), proxy-routed curl for reddit/blogspot, and `api.pullpush.io` for Reddit archives. Queries used: `vs-mlrt TensorRT RIFE RTX 5080/5090 fps`, `vs-rife t parameter`, `gimm vapoursynth`, `gmfss speed/slow`, `rife interpolate twice artifacts`, plus GitHub code/issue searches quoted below. VSEDB (vsedb.com) was unreachable both via proxy and direct (connection fail) — its contents are **UNVERIFIED**.

---

## 1. VapourSynth itself

**What it is.** A C++17 frameserver/filter-graph library with Python bindings: you build a
Directed-Acyclic-Graph of filters in a Python script; VS evaluates it lazily, frame-by-frame,
with automatic multithreaded prefetch. It is the de-facto successor of Avisynth in the anime
encoding scene (Avisynth is Windows-origin and effectively frozen; VS is cross-platform and
actively developed). Design quote `[official, repo description]`: "A video processing framework
with simplicity in mind". PyPI self-description `[official]`: "A frameserver for the 21st century".

**Hard facts `[verified-api]`:**

| Fact | Value |
|---|---|
| Repo | vapoursynth/vapoursynth, created 2013-07-02 |
| Latest release | **R81** (2026-10-04); R80 2026-09-16, R79 2026-08-07, R78 2026-07-24, R77 2026-06-16, R76 2026-05-14 |
| Release cadence | ≈ monthly in 2026 (R76→R81 in 5 months), RC pre-releases typical |
| Pushed at | **2026-10-27 (today)** — actively developed |
| Stars / forks / open issues | 2122 / 184 / 10 |
| License | **LGPL-2.1** |
| PyPI | `pip install vapoursynth` — version 81 (uploaded 2026-10-04), `requires_python >=3.12`, first PyPI release 2017; official wheels |
| Platforms | Linux / Windows / macOS; Python-coupled (Python is THE scripting API; the C API VSScript4 embeds a Python evaluator — `evaluateBuffer` runs a .vpy string) |

**Users / ecosystem `[verified-api + official]`:** StaxRip (Windows encoding GUI, 3022★, MIT, pushed 2026-09-27) is built around VS; the anime encoding scene (VCB-Studio style groups, JET = Jaded-Encoding-Thaumaturgy: vs-jetpack, vs-preview, vs-tools) lives on VS; mpv has a native `vf_vapoursynth` filter used by SVP and by hooke007-style configs; vs-mlrt, vs-rife, AnimeJaNai tooling all target it. Editor tooling: **vs-preview** (JET, 100★, Apache-2.0, pushed 2026-05) is the modern script previewer; **VapourSynth-Editor ("VSEdit")** = YomikoR/VapourSynth-Editor (176★, pushed 2026-09; fork of mystery_keeper's Bitbucket project; "Branch vse-previewer for standalone previewer, vs-api4 for an IDE"); theChaosCoder/vapoursynth-portable-FATPACK (95★) bundles everything for Windows. Community size signals: VS's own repo is modest (2.1k★) but the plugin/script ecosystem (vs-mlrt 489★, vs-rife 151★, vsrepo package manager, doom9 threads, r/vapoursynth + Chinese community around hooke007/MPV-lazy configs) is the largest of any frame-serving ecosystem. Exact doom9/Discord activity figures: **UNVERIFIED** this session (doom9 not fetched); community activity is evidenced by the 2026-dated issues/releases quoted throughout §5–6.

**Python coupling (how it runs).** A .vpy script IS Python; `import vapoursynth as vs; core = vs.core`.
Key API surface `[official, context7 /vapoursynth/vapoursynth + vapoursynth.com/doc]`:
`core.std.Trim/Interleave/FrameEval`, `clip.get_frame(n)`, `clip.set_output(index)`,
`clip.output(fileobj, y4m=...)`, `clip.frames(prefetch=...)` (concurrent frame generator),
`std.LoadPlugin(path)` / auto-loaded plugins from the global plugin dir create `core.<namespace>.<Function>`.
Plugins are native .so/.dll exposing registered functions; vs-mlrt installs as such plugins
(`core.trt.Model`, `core.ort.Model`, ...) plus a pure-Python wrapper (`vsmlrt.py`).

---

## 2. vs-mlrt — the neural inference framework

Repo **AmusementClub/vs-mlrt** (renamed from AmusementLAB — old URL 404s today)
`[verified-api]`: created 2021-12-02, **489★ / 36 forks**, GPL-3.0, 45 open issues, pushed 2026-09-25.
Self-description `[official]`: "Efficient CPU/GPU ML Runtimes for VapourSynth (with built-in
support for waifu2x, DPIR, RealESRGANv2/v3, Real-CUGAN, RIFE, SCUNet, ArtCNN and more!)".

**Releases `[verified-api]`:** latest **v16.3.test1** (2026-09-23); cadence ≈ every 4–6 weeks
(v15.15 Feb-2026, v15.16 Mar-2026, v16.test1 Jun, v16.1 Jun-25, v16.2 Aug-05, v16.3 Sep-23;
variant lines: v15.13.cu13 = CUDA 13 builds, v15.13.rtx / TRT-RTX builds, v15.13.ncnn).
Release packaging = 7z archives you extract into the VS plugin dir:
`vsmlrt-cuda.*.7z.001/.002` (split archive), `vsmlrt-windows-x64-cpu`, `vsmlrt-hip`, `VSMIGX-Windows-x64`,
plus `models.*.7z`, `contrib-models.*.7z`, `scripts.*.7z`. There is also the frozen
`external-models` tag (2022-12-07) holding per-model ONNX 7z: `rife_v2_v4.7.7z` …
`rife_v4.26_heavy.7z`, `rife_ensemble_v1.7z`, safa, scunet, swinir, waifu2x_swin_unet, artcnn,
anitoon, anime-segmentation, oidn, ppocr. **`vsmlrt` is NOT on PyPI** (HTTP 404 verified) —
it's a plain Python module (`vsmlrt.py`) shipped in `scripts.*.7z` (or copied from the repo).

**Backends `[official README + vsmlrt.py source]`:**

| Backend | Runtime | Targets |
|---|---|---|
| `vsov` (`Backend.OV_CPU/OV_GPU`) | Intel OpenVINO | pure CPU (recommended CPU backend) + Intel GPUs |
| `vsort` (`Backend.ORT_CPU/ORT_CUDA/ORT_DML/ORT_COREML`) | ONNX Runtime | CPU, CUDA, **DirectML**, Apple CoreML |
| `vstrt` (`Backend.TRT`) / `vstrt_rtx` (`BackendV2.TRT_RTX`) | NVIDIA TensorRT / **TensorRT-RTX** | NV GPUs; "the resulting performance is also typically *much much better* than the CUDA backend of vsort"; TRT-RTX "compiles engines faster with comparable performance" |
| `vsncnn-vk` (`Backend.NCNN_VK`) | Tencent ncnn via Vulkan | any Vulkan GPU; small footprint but slower |
| `vsmigx` (`Backend.MIGX`) | AMD MIGraphX | AMD GPUs |

**Model pipeline `→ TRT`:** models ship as ONNX (fp32/fp16 files); on first use the wrapper
converts ONNX → serialized TRT engine via bundled `trtexec` (helpers in vsmlrt.py:
`get_engine_path`, `convert_model`, `trtexec`), caching the engine on disk keyed by
model+shapes+precision+GPU `[official vstrt README + source]`. Engine cache is machine/GPU-specific
and not portable. `BackendV2.TRT` tuning knobs (dataclass, from source): `fp16` ("set fp16 = True
(on RTX GPUs)"), `num_streams` (multi-stream scaling is near-linear to 3–5 streams),
`use_cuda_graph=True` default, `workspace`, `static_shape`, `tf32`/`bf16`/`force_fp16`,
`output_format` (0=fp32/1=fp16 output planes), `builder_optimization_level`,
`engine_folder`, `max_tactics`. Native call `core.trt.Model(clips, engine_path, tilesize, overlap,
device_id, use_cuda_graph, num_streams, verbosity)` `[official vstrt README]`.
The Python wrapper `vsmlrt.py` (3463 lines) provides model-typed functions:
`Waifu2x, DPIR, RealESRGAN, CUGAN, RIFEMerge, RIFE, SAFA, SCUNet, SwinIR, ArtCNN`,
generic `inference(..., model_path)` for arbitrary ONNX, plus engine/tiling plumbing.

---

## 3. Target-FPS semantics per wrapper *(controller addendum)*

This is the decisive table for "arbitrary target FPS vs x2":

| Wrapper | Backend | Arbitrary target FPS in ONE pass? | Mechanism (from source) |
|---|---|---|---|
| **vsmlrt.RIFE** (vs-mlrt) | vs-mlrt (TRT/ORT/OV/ncnn) | **Partially**: `multi` is int or `fractions.Fraction` (e.g. `multi=Fraction(5,2)` = ×2.5), scene-change-safe via interleave + `_SceneChangeNext` passthrough. NOT absolute-fps keyed; you compute the fraction yourself. Integer `multi` = the classic n−1-interleaved midpoint ladder; Fraction multi materializes all timepoints 1/n…(n−1)/n per input pair. | `RIFE(clip, multi=2|Fraction, scale, model=RIFEModel.v4_4, backend, ensemble, video_player)`; internal `RIFEMerge(clipa, clipb, mask)` where mask is a GRAY clip of **arbitrary timepoints t ∈ (0,1)** — "merges the two clips in the time domain", semantics of `std.MaskedMerge` in time `[official docstring]`. So the PRIMITIVE (`RIFEMerge`) takes arbitrary t; only the convenience wrapper is multi-shaped. |
| **vs-rife** (HolyWu, pip `vsrife` 5.7.0) | PyTorch 2.10+, optional **Torch-TensorRT** (`trt=True`) | **YES — the gold standard.** `fps_num/fps_den` = absolute target FPS (overrides `factor_num/factor_den` multiplier; e.g. 24000/1001 → 60000/1001 single pass). Per output frame: `t = n*factor_den % factor_num / factor_num` — exact rational position each frame; `t==0` → passthrough; scene change → passthrough. Target must exceed source (it errors and suggests vs-tools `change_fps` for downsizing). Engine cache keyed `model_dims_fp16/32_scale_ensemble_GPUname` in `trt_cache_dir`. | verified from `vsrife/__init__.py` 5.7.0 `[official source]` |
| **vs-gmfss_fortuna** (HolyWu) | PyTorch, optional TRT | **YES — same API shape as vs-rife**: `factor_num/factor_den` + `fps_num/fps_den` override + `sc`/`sc_threshold` + `trt`/`trt_cache_path` + `ensemble`. (Repo dormant: created 2023-04-08, pushed 2023-04-08, 20★.) | verified from repo tree + `__init__.py` signature `[official source]` |
| **GIMM-VFI in VS** | — | **NO VS WRAPPER EXISTS** as of 2026-10-27. Upstream GSeanCDAT/GIMM-VFI (NeurIPS 2024, 409★) is PyTorch; the only packaged integration is kijai/ComfyUI-GIMM-VFI (486★, ComfyUI node, pushed 2025-07). vs-mlrt ships no GIMM ONNX (verified against vsmlrt.py + external-models release). GitHub searches for `vs-gimm` return no such project. | `[verified-api]` |

**Community caution on vsmlrt rational multi** `[community, vs-mlrt issue #59, maintainer-confirmed]`:
`multi=Fraction(5,2)` works fine **for encoding** ("we have used it successfully in encoding")
but historically froze in live players (mpv/MPC) because players expose huge virtual frame
counts, exploding the interleave graph — maintainer WolframRhodium added `video_player=True`
specifically to fix this. Community script evidence: `RIFE(cut1, multi=Fraction(5,2), model=46,
backend=BackendV2.TRT(force_fp16=True, num_streams=2, output_format=1, static_shape=True))`.

---

## 4. Canonical community scripts (real, cited)

**Example A — vs-mlrt wiki, official minimal** `[official, github.com/AmusementClub/vs-mlrt/wiki/RIFE]`:
> ```python
> from vsmlrt import RIFE, RIFEModel, Backend
> src = core.std.BlankClip(format=vs.RGBS)
> flt = RIFE(src, model=RIFEModel.v4_4, backend=Backend.ORT_CUDA())
> ```
> "backend could be: GPU Backend.TRT(fp16=True, device_id=0, num_streams=1): TensorRT runtime,
> the fastest NV GPU runtime." Wiki states RIFE "supports arbitrary-timestep interpolation
> between a pair of inputs"; tile constraint: block sizes multiple of `32/scale`.

**Example B — full encode-grade pipeline: upscale + interpolate in one script**
`[community, github.com/abihf/video-upscaler → script.py, fetched verbatim 2026-10-27]`:
```python
import os, vapoursynth as vs
from vapoursynth import core
import vsmlrt


def process(src):
    rgb = core.resize.Bicubic(
        src, format=vs.RGBH, matrix_in_s="709", transfer_in_s="709", primaries_in_s="709"
    )
    be = vsmlrt.Backend.TRT_RTX(fp16=True, num_streams=2)
    rgb = vsmlrt.RealESRGANv2(rgb, model=vsmlrt.RealESRGANv2Model["animejanaiV3_HD_L2"], backend=be)
    if os.getenv("VSPIPE_RIFE", "0") == "1":
        be = vsmlrt.Backend.TRT(
            fp16=True,
            tf32=False,
            output_format=1,
            use_cublas=False,
            use_cuda_graph=True,
            use_cudnn=False,
            num_streams=1,
        )
        rgb = vsmlrt.RIFE(
            rgb,
            model=vsmlrt.RIFEModel["v4_7"].value,
            ensemble=True,
            backend=be,
            scale=1.0,
            _implementation=1,
        )
    return core.resize.Bicubic(
        rgb, format=vs.YUV420P10, matrix_s="709", transfer_s="709", primaries_s="709"
    )


src = core.bs.VideoSource(args["in"], cachemode=3, cachepath=args["cache"])
process(src).set_output()
```
Typical args visible: fp16 everywhere (`RGBH` input + `output_format=1`), `num_streams=2`
for the upscaler, `ensemble=True` on RIFE, TRT-RTX backend for the ESRGAN, BestSource input.

**Example C — mpv real-time playback config**
`[community, github.com/natural-harmonia-gropius/mpv-config → portable_config/filters/rife.py]`:
```python
from vsmlrt import RIFE, BackendV2, RIFEModel


def rife(clip, fps):
    pixel_format = clip.format.id
    clip = clip.misc.SCDetect(threshold=0.2)
    clip = clip.resize.Bilinear(format=vs.RGBH, matrix_in=vs.MATRIX_BT709)
    clip = RIFE(
        clip,
        model=RIFEModel.v4_26,
        ensemble=False,
        backend=BackendV2.TRT(
            num_streams=4,
            fp16=True,
            output_format=1,
            use_cuda_graph=True,
            engine_folder=gettempdir(),
        ),
        video_player=True,
        _implementation=2,
    )
    clip = clip.resize.Bilinear(format=pixel_format, matrix=vs.MATRIX_BT709)
    return clip, fps * 2
```
Note `video_player=True` (issue #59 fix) + `misc.SCDetect` before RIFE + `_implementation=2`
(rife_v2 ONNX with internal padding). Scene detection pattern matches the official docstring
requirement: "For the best results, you need to perform scene detection (e.g. misc.SCDetect,
mv.SCDetection) … the quality of result is strongly dependent on high quality scene detection".

**AnimeJaNai integration** `[official README, the-database/mpv-AnimeJaNai, 770★, pushed 2026-10-27]`:
"Real-time anime upscaling to 4K in mpv with lightweight super-resolution models" — Real-ESRGAN
Compact ONNX models run through TensorRT (NVIDIA) or DirectML (AMD/Intel) inside a custom Windows
mpv build; engine built once per model on first playback; VapourSynth enters other players via
CrendKing/avisynth_filter (DirectShow), and offline conversion via AnimeJaNaiConverterGui;
README explicitly points non-Windows users to **styler00dollar/VSGAN-tensorrt-docker** and
**hooke007/MPV_lazy + vs-mlrt**. So AnimeJaNai's own runtime is TRT-in-mpv, with VS+vs-mlrt as
the cross-platform route.

**GMFSS models**: upstream = 98mxr/GMFSS_Fortuna (83★, pushed 2026-01, PyTorch 1.13.1,
" Dedicated for Anime Video Frame Interpolation", union models on Google Drive, AnimeRun
fine-tune 2023-06-25) `[verified-api]`. The VS-side wrapper is HolyWu/vs-gmfss_fortuna
(PyTorch, §3). Running GMFSS through vs-mlrt via community ONNX exports exists in the wild but
the commonly-cited "st4rkhit" posts/gists could **not** be located this session
(user not found on GitHub; blogspot dead) — **UNVERIFIED**; treat "st4rkhit ONNX GMFSS pack"
as folklore until the link is produced. Also related: styler00dollar/VapourSynth-RIFE-ncnn-Vulkan
(177★, MIT, pushed 2025-09) — the older ncnn-Vulkan RIFE wrapper whose factor_num/factor_den
implementation vs-mlrt referenced.

---

## 5. Performance reality

**Official vendor benchmark — RTX 4090 (AD102), TRT FP16, 1920×1080 RGBS, CUDA graphs,
vsmlrt v14.test3** `[official wiki page "NVIDIA-GeForce-RTX-4090", credit MysteryDove]` —
FPS / device-MB, RIFE fp16 i/o:

| RIFE version | 1 stream | 2 streams | 3 streams | 5 streams |
|---|---|---|---|---|
| v4.4–v4.5 | 150.2 / 623 MB | 301.1 | 448.9 | 787.6 / 1.48 GB |
| v4.7–v4.9 | 132.1 / 748 MB | 268.6 | 403.5 | 496.4 / 2.06 GB |
| v4.10–v4.15 | 119.1 / 862 MB | 238.7 | 347.0 | 349.8 / 2.64 GB |
| v4.12/13/15/16_lite | 123.7 | 250.8 | 377.3 | 403.8 |

Non-RIFE 1080p anchors on the same page: RealESRGAN x2 17→38 FPS, waifu2x upconv 20→61 FPS,
cugan 13.9→26 FPS, SwinIR-L 0.61 FPS (1 stream). Older vsmlrt v12 comparison table:
`rife (1920x1088, model=44)`: ORT_CUDA 95.5 FPS vs TRT 138.5 vs TRT×2 208.3 — **TRT ≈ 1.45×
faster than ORT_CUDA at 1080p**. RTX 3090 wiki: rife v4.4 TRT 61.4 FPS (1 stream) → 106.5 (2 streams);
with 2 streams the 4090 did 301 FPS on the old impl.

**Translation to the owner's use-case** (anime 1080p 23.976→60 = ×2.5 = 2.5 output frames per
input frame, i.e. one RIFE merge per output frame): on the 4090 numbers, v4.7+ at 1 stream
(≈130 FPS output) ≈ **52 input FPS**; at 2 streams (≈268 FPS) ≈ **107 input FPS** — i.e. faster
than realtime encode throughput, minutes not hours for an episode. 720p scales up ~2× further
(RIFE cost ≈ linear in pixels; We0M/rifebench exists specifically for such RIFE model
benchmarking under vs-mlrt).

**RTX 5080 specifically:** no public vs-mlrt/TRT RIFE benchmark for 50-series was found
(searches of vs-mlrt issues, wiki pages, pullpush) — **UNVERIFIED, extrapolation follows**:
the 5080 (Blackwell, ~4090-class raster, strong 4th-gen+ tensor cores, ~16 GB VRAM) with
TensorRT 10.8+/CUDA 13 (vs-mlrt ships `cu13` builds since v15.13) should sit between the 3090
and 4090 rows — i.e. ≈100–150 FPS single-stream 1080p output for v4.7+, comfortably
non-glacial. Blackwell PyTorch friction exists (ComfyUI/PyTorch "RIFE 4.9 on RTX 5090" threads,
Jan 2026 vintage, pullpush) — one more reason the TRT path in VS is robust on 50-series.
Treat all 5080 numbers as estimates pending a 5-minute `rifebench`/vspipe measurement.

**GMFSS Fortuna speed:** `[community, r/comfyui comment]` "with the Fortuna model it was also
extremely slow compared to GIMM on an 846×1152 video — an extra ~4:30 minutes on a [RTX] PRO
6000 … really not worth considering from a practical perspective" (i.e., order-of-magnitude
slower than RIFE/GIMM — it runs MetricNet+FLFNet+FeatureNet+IFNet+FusionNet ensemble per frame).
Quality praise persists: `[community, r/VideoEditing]` "gmfss pg 104 model in SVFI … looks the
absolute best/smoothest for most scenes" for anime. Budget GMFSS at roughly RIFE/5…RIFE/10
throughput; it is the quality pick, not the speed pick.

**OpenVINO CPU fallback pain** `[community, vs-mlrt issues]`: #34 "OV plugin does not load on
Linux" (OPEN), #131 "Models provided in releases are not supported after updating openvino"
(closed — OV version-locks models), #156 "OpenVINO + FP16 + Waifu2xModel.cunet == ERROR" (OPEN).
CPU inference is orders of magnitude off GPU (OV_CPU is only the recommended CPU backend);
nobody runs neural interpolation on CPU for anything but testing. The backend fallback chain in
vsmlrt (`inference_with_fallback`) exists for correctness, not as a speed strategy.

**VRAM:** RIFE v4.7+ 1080p fp16 ≈ 0.75 GB (1 stream) → ~2 GB (3 streams) → 2.6 GB (5 streams,
v4.10-class); upscalers like cugan ≈ 4.4–12 GB (4090 table, tiled sizes vary); a realistic
RIFE+ESRGAN chain fits easily in the 5080's 16 GB with `num_streams≤3`.

**fp16 vs fp32 for line art:** fp16 is the default posture everywhere in this ecosystem
(RGBH input format = fp16 inference; `output_format=1`; wiki benchmark ran fp16; vs-rife keys
its engine cache on fp16/fp32). Edge cases exist — SwinIR fp16 PSNR 65.3 vs bf16 53.7 on the
4090 wiki (precision-sensitive nets exist; RIFE/ESRGAN are robust). RIFE v4.26_heavy had a
reported artifact regression (vs-mlrt issue #155, closed) — model-version QA matters. For
anime line art, fp16 + ensemble=True (where supported; vsmlrt blocks ensemble for v4.21–4.23
and ≥4.25, while vsrife supports it for all) is the community default.

---

## 6. Stacked ×2 passes *(controller addendum #3)*

No canonical source stating "stacked ×2 is artifact-free" was found; the defensible synthesis
from verified mechanics + community practice:

- **Mechanically valid:** `RIFEMerge` interpolates any pair at any t, so pass-2 simply
  interpolates real↔synthetic neighbors `[official docstring]`.
- **Unnecessary for any practical target:** vs-rife / vs-gmfss_fortuna do arbitrary rational
  target FPS in ONE pass, so stacking is only ever relevant to vsmlrt.RIFE integer-multi users —
  and even there `multi=Fraction(5,2)` (encoding) or `fps_num/fps_den` via vs-rife covers 60fps
  in one pass `[official source]`.
- **Quality-wise stacking is the inferior path and the community treats it as such:** each pass
  re-runs flow estimation on frames where one input is already model output (out-of-training-
  distribution input); Practical-RIFE issue #110 ("4.26 … inbetween frames are less visible")
  and the Flowframes dedup/quality guidance show the sensitivity of VFI nets to their inputs;
  the anime-specific practice (SVFI/GMFSS vs-mlrt configs, mpv-config above) is single-pass
  rational multi with strong scene detection — nobody recommends double passes. Formally:
  **[community consensus / inference from verified mechanics, no dedicated benchmark found]**.
- Anime-specific caveat that matters MORE than stacking: anime is drawn on twos/threes
  (duplicated frames). Interpolating across duplicated pairs yields judder; dedup-aware or
  SCDetect-tuned pipelines are the real quality lever (Practical-RIFE issue #23 open on 一拍二/一拍三;
  Flowframes' "Frame De-Duplication … meant for 2D animation").

---

## 7. VS ↔ ffmpeg / encoders

- **vspipe** `[official, vapoursynth.com/doc/output.html + gettingstarted.html]`: evaluates a
  .vpy, streams raw frames to stdout; `vspipe -c y4m script.vpy - | ffmpeg -i - encoded.mkv`
  is the canonical pipe. Options: frame ranges (`-s/-e`), output index selection, `-c y4m|wav`
  container headers, `--timecodes`, progress prints, per-filter processing times, passing args
  into the script environment (`args = globals()` pattern seen in example B).
- **Raw pipe to NVENC:** `vspipe script.vpy - | ffmpeg -f y4m -i - -c:v hevc_nvenc -preset p5 -b:v ... out.mkv`
  — the standard encoding-scene chain (equivalent to what StaxRip/VS pipelines do internally).
  With BestSource/FFMS2 inside the script, decode also stays inside VS and ffmpeg only encodes.
- **From Python instead of vspipe:** `clip.output(sys.stdout.buffer, y4m=True)` or
  `clip.frames(prefetch=N)` for a concurrent frame generator you can feed to an ffmpeg stdin
  subprocess yourself — same graph, programmatic control `[official pythonreference]`.
- **mpv:** native `vf_vapoursynth` runs .vpy at playback time (SVP, hooke007 configs,
  AnimeJaNai's avisynth_filter path for DirectShow) — the `video_player=True` pattern exists
  precisely for this (issue #59).
- **Editors/preview:** vs-preview (JET) is the modern previewer (also cited by the vs-mlrt
  maintainer in issue #59 as a debugging tool); VSEdit (YomikoR fork) = IDE-style editor+viewer;
  both free, cross-platform for previewer parts.

---

## 8. Alternatives when VS is overkill

- **Direct PyTorch (Practical-RIFE) — the owner's current stack.** Full control, easiest
  custom-model work; costs: hand-rolled frame feeding/scene detection, Blackwell wheels churn
  (50-series needs CUDA 12.8+/cu130 builds; community threads on 5090 RIFE breakage), no lazy
  frame-graph, and you re-implement what vs-rife gives free (rational fps, scene-change safety,
  TRT engine caching, fp16). HolyWu's vs-rife is literally this, productized in a VS graph.
- **rife-ncnn-vulkan (nihui, 1120★, MIT).** Zero-dependency CLI, any Vulkan GPU, was THE simple
  route — but effectively **unmaintained** (last push 2024-01-02) and single-purpose (x2/xN
  midpoint ladder, no arbitrary fps, no scene-change handling at model level, ncnn speed below
  TRT). Fine for one-off clips; wrong foundation for a pipeline.
- **mpv GPU shaders / SVP.** Realtime playback-oriented interpolation (SVP wraps VS+RIFE itself —
  see issue #91 where a user transcodes "SVP rife" 1080p24→120). Zero encode flexibility, no
  arbitrary-fps file output control; it consumes rather than produces. Use mpv+vf_vapoursynth
  when you want to WATCH your VS-filtered result, not when you want files.
- **ONNXRuntime direct from Python.** vs-mlrt's ORT_CUDA backend minus the frame graph: you
  keep tensor plumbing, lose caching/tiling/interleave logic. Roughly 1.45× slower than TRT at
  1080p per the v12 wiki table. Only worth it for exotic models vs-mlrt doesn't wrap (its
  generic `inference(model_path=...)` mostly closes even that gap).
- **ComfyUI nodes (kijai GIMM-VFI etc.)** — the only packaged GIMM-VFI route today; workflow-
  tool ergonomics, not a scriptable encode pipeline.

---

## 9. Verdict — exact recommended stack for "RIFE via VapourSynth on RTX 5080 from Python"

1. **VapourSynth R81** from PyPI (`pip install vapoursynth`, official wheels, py≥3.12) or distro package.
2. **Inference: vs-mlrt v16.3.test1** (`vsmlrt-cuda` 7z into VS plugin dir + `scripts/vsmlrt.py`
   on PYTHONPATH) with **`BackendV2.TRT(fp16=True, num_streams=2, output_format=1,
   use_cuda_graph=True)`**; TensorRT 10.x CUDA-13 build for Blackwell. Keep ONNX models from the
   vs-mlrt `external-models`/`models` release 7z; first run builds+py-caches the TRT engine.
   (TRT-RTX builds are a drop-in alternative with faster engine builds.)
3. **Interpolation wrapper: `pip install vsrife` (5.7.0) — HolyWu/vs-rife** — for arbitrary
   target FPS semantics: `rife(clip, model="4.25", fps_num=60000, fps_den=1001, sc=True,
   trt=True, trt_opt_shape=[1920,1080])`. It runs PyTorch(+Torch-TensorRT) inside the VS graph —
   the same graph can also host vs-mlrt upscalers. If one prefers everything through vs-mlrt
   runtimes, use `vsmlrt.RIFE(multi=Fraction(...))` for encoding workloads instead.
4. **GMFSS Fortuna (quality pick, slow):** HolyWu/vs-gmfss_fortuna with the same
   fps_num/fps_den API, or 98mxr upstream PyTorch. **GIMM: not available in VS** — ComfyUI-only
   as of today; revisit if a wrapper appears.
5. **Output:** `vspipe -c y4m script.vpy - | ffmpeg -i - -c:v hevc_nvenc ...` or embed the VS
   graph in the existing Python pipeline via `clip.output()`/`frames()`.

### Skeleton — Python-embedded VS script (matches the owner's Python-driven pipeline)
```python
import vapoursynth as vs
from vsmlrt import RIFE, RIFEModel, BackendV2  # scripts/vsmlrt.py from vs-mlrt release

core = vs.core()
src = core.bs.VideoSource(r"/path/input.mkv")  # BestSource (bundled with VS R65+)
rgb = src.resize.Bilinear(format=vs.RGBH, matrix_in=vs.MATRIX_BT709)  # fp16 RGB
rgb = rgb.misc.SCDetect(threshold=0.2)  # scene detection BEFORE RIFE
interp = RIFE(
    rgb,
    multi=Fraction(5, 2),  # 23.976 -> 59.94 fps, one pass
    model=RIFEModel.v4_7,
    ensemble=False,
    backend=BackendV2.TRT(
        fp16=True, num_streams=2, output_format=1, use_cuda_graph=True, static_shape=True
    ),
    video_player=False,
    _implementation=2,
)
out = interp.resize.Bilinear(format=vs.YUV420P10, matrix=vs.MATRIX_BT709)
out.set_output()  # vspipe consumes clip 0; or in-Python: out.output(sys.stdout.buffer, y4m=True)
```

### Skeleton — arbitrary-FPS gold path (vsrife inside the same VS process)
```python
import vapoursynth as vs
from vsrife import rife  # pip install vsrife; PyTorch cu130 + torch_tensorrt for trt=True

core = vs.core()
src = core.bs.VideoSource(r"/path/input.mkv")
rgb = src.resize.Bilinear(format=vs.RGBH, matrix_in=vs.MATRIX_BT709)
interp = rife(
    rgb,
    model="4.25",
    fps_num=60000,
    fps_den=1001,  # exact 59.94 target, single pass
    sc=True,
    sc_threshold=0.15,
    scale=1.0,
    trt=True,
    trt_static_shape=True,
    trt_opt_shape=[1920, 1080],
    trt_cache_dir="/home/whoami/.cache/vsrife",
)
interp.resize.Bilinear(format=vs.YUV420P10, matrix=vs.MATRIX_BT709).set_output()
# then: vspipe -c y4m script.vpy - | ffmpeg -i - -c:v hevc_nvenc -crf 20 out.mkv
```

---

## 10. Source register (all fetched 2026-10-27)

- GitHub API `[verified-api]`: repos vapoursynth/vapoursynth (R81, 2122★, LGPL-2.1, pushed 2026-10-27), AmusementClub/vs-mlrt (v16.3.test1, 489★, GPL-3.0), HolyWu/vs-rife (5.7.0, 151★, MIT), HolyWu/vs-gmfss_fortuna (20★), 98mxr/GMFSS_Fortuna (83★), styler00dollar/VapourSynth-RIFE-ncnn-Vulkan (177★), nihui/rife-ncnn-vulkan (1120★, stale), the-database/mpv-AnimeJaNai (770★), StaxRip/StaxRip (3022★), Jaded-Encoding-Thaumaturgy/vs-preview (100★), YomikoR/VapourSynth-Editor (176★), hooke007/MPV_lazy (24★), GSeanCDAT/GIMM-VFI (409★), kijai/ComfyUI-GIMM-VFI (486★), We0M/rifebench.
- PyPI `[verified-api]`: vapoursynth 81 (2026-10-04, py≥3.12), vsrife 5.7.0 (2026-02-09), spandrel 0.4.2 (2026-02-21), vsmlrt → 404 (not published).
- Official docs `[official]`: vapoursynth.com/doc (gettingstarted, output, pythonreference); context7 IDs /vapoursynth/vapoursynth (1646 snippets), /amusementclub/vs-mlrt (226), /websites/vapoursynth_doc (714), /chainner-org/spandrel (830) — **context7 covered VS core API, plugin mechanism, vspipe and vs-mlrt backends well; nothing material missing** (Spandrel covered for the PyTorch-model-loading alternative).
- Source files `[official]`: AmusementClub/vs-mlrt master `scripts/vsmlrt.py` (RIFE/RIFEMerge/Backend.TRT, 3463 lines), `README.md`, wiki pages Home/RIFE/RTX4090/RTX3090; HolyWu/vs-rife `vsrife/__init__.py` 5.7.0 (800 lines), README; HolyWu/vs-gmfss_fortuna `vsgmfss_fortuna/__init__.py`.
- Community `[community]`: vs-mlrt issues #59 (rational FPS + video_player history, maintainer answers), #34/#131/#156 (OpenVINO pain), #150/#112/#155 (TRT_RTX/tiles/artifacts); hzwer/Practical-RIFE issues #91 (4080S SVP transcode times), #110, #23 (anime on-twos); abihf/video-upscaler script.py; natural-harmonia-gropius/mpv-config filters/rife.py; r/comfyui + r/VideoEditing GMFSS comments (pullpush archive); the-database/mpv-AnimeJaNai README.
- `[unverified]`: RTX 5080-specific FPS (extrapolated from 3090/4090); VSEDB contents (site unreachable); "st4rkhit" GMFSS ONNX pack posts (not locatable); doom9/Discord activity metrics.

## Skills used
`brainstorming` (resolved per SUBAGENT-STOP: fully-specified brief → execute), `grilling` (same — brief already carries the design tree + output contract; addendum integrated), `build-with-exa` (Exa discipline applied; Exa 402 → documented fallback chain per its rules), `research` (primary-source investigation, claims cited), `planning-with-files` (task_plan.md + this report as findings).

## Memory files read (mandate compliance)
**Full reads — topics/ (19/19):** stack-and-tools.md, workflow.md, workflow-git-pr.md, environment-infra.md, tech-gotchas.md, communication-behavior.md, code-gotchas.md, harness-server.md, job-search.md, remnawave.md, hh-ranking-ats-research.md, 2026-10-09-telegram-serverless.md, 2026-10-11-novum-organum-report-plan.md, 2026-10-12-slides-novum-organum-plan.md, 2026-10-14-opencode-v2-storage.md, 2026-10-15-binary-re-landmines.md, 2026-10-15-vision-ocr-scans.md, 2026-10-21-stepin-book-contract.md, 2026-10-25-locke-pirogov-referaty.md.
**Full reads — projects/ (22):** labboardvent.md, iris-proposal.md, interview-kb.md, labs.md, kolosyug.md, 4q4r-profile.md, firefox-vpn-client.md, teleparse.md, wauploader.md, freeai-decompile.md, xray-raw.md, tspu-research-2026.md, labs-docker-stands.md, habrparser.md, cindex.md, shikimori-japanese-theme.md, vibecoding-course.md, stepik-whitehacker-course.md, freepackages.md, WMS.md, anicli-go.md, home-global.md.
**Survey reads (head + full-tree relevance grep; giants exceeding context capacity — zero task-relevant hits except anicli-py.md:275 ffmpeg-matroska muxing note):** remnavawe_ansible.md, bos-pro-bugs.md, secureway_bot.md, traffic_mimicry.md, anicli-py.md, hhauto.md, ert.md, wms.md.
Plus ~/.config/opencode/MEMORY.md in full. No memory files written (facts delivered to controller per mandate).
