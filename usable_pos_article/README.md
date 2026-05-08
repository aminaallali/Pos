# Hybrid TS-CAN + article-style POS rPPG pipeline (with multi-algorithm consensus)

This is a modernized, usable replacement for the old POS pulse-extraction script.
It bundles **three heart-rate estimators** behind a single API and CLI:

- **TS-CAN neural rPPG** — the default. Uses the on-shelf
  ``UBFC-rPPG_TSCAN.pth`` checkpoint vendored from
  [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) (Liu et al.,
  NeurIPS 2020 — *"Multi-Task Temporal Shift Attention Networks for
  On-Device Contactless Vitals Measurement"*). No per-clip tuning required;
  the network already learnt to ignore harmonics that fooled the unsupervised
  POS path. Inference runs on CPU (``torch==2.4.1`` CPU build, ~3 s for a
  10 s clip).
- **Article-style POS** — the original Plane-Orthogonal-to-Skin pipeline,
  kept as the explicit-opt-in / fallback method. Used when a clip is
  shorter than ~4 s, when ``torch`` is not installed, or when a caller
  passes ``method=pos`` explicitly.
- **Consensus** — runs **seven** classical rPPG algorithms in parallel
  (POS, CHROM, GREEN, ICA, LGI, PBV, OMIT — all vendored from
  rPPG-Toolbox) on the same per-patch RGB trace, then applies the
  rPPG-VQA paper's consensus mechanism (RANSAC inlier selection on HR +
  Pearson spectral correlation weighting) to produce a single BPM and a
  scalar quality score `q_sig ∈ [0, 1]`. Use it when you want a built-in
  "is this measurement trustworthy?" signal — the verdict
  (`accepted`/`rejected`) tells you whether enough algorithms agreed
  within `5 BPM` to trust the reading. Selected with `method=consensus`.

The POS path follows the 2025 article's described rPPG estimation setup as
closely as possible in a standalone script:

- MediaPipe Face Mesh instead of dlib's old 68-point predictor.
- Facial ROI options from the article:
  - `face`: facial skin region excluding eyes and mouth.
  - `selected`: forehead + cheeks.
  - `full-frame`: whole frame.
- Default RGB extraction from the 100 MediaPipe landmarks used in the article's
  released code (`forehead`, `cheeks`, and `nose` patches).
- 8-second overlapping windows with 1-second stride.
- 6th-order Butterworth bandpass filter from `0.65` to `4.0` Hz.
- POS BVP extraction with the 1.6-second POS internal window.
- Welch PSD BPM estimation in the `0.65` to `4.0` Hz range with
  3-point parabolic peak interpolation.
- Variable-frame-rate aware: per-frame ``CAP_PROP_POS_MSEC`` timestamps are
  read from the video and the RGB trace is resampled onto a uniform grid
  before POS, so phone-recorded VFR clips don't pull the BPM off.
- Optional article-style lightweight video modifications.

## Vendored model & licensing

The TS-CAN model architecture (``tscan_inference.py``) and the
``UBFC-rPPG_TSCAN.pth`` checkpoint under ``models/`` are derived from
``rPPG-Toolbox`` and redistributed under that project's Responsible AI
License (RAIL). The full upstream license text is in
``THIRD_PARTY_LICENSES/rPPG-Toolbox-LICENSE.txt`` — please read it before
deploying this service in a commercial product. RAIL adds use restrictions
on top of a permissive base license; in particular it forbids a number of
medical / surveillance / discriminatory uses.

## Install

Use Python 3.10+ if possible.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

The first run downloads Google's public `face_landmarker.task` model to
`~/.cache/article_pos_pipeline/face_landmarker.task`. This is needed by the
newer MediaPipe Tasks API. If your installed MediaPipe still includes the older
`mp.solutions.face_mesh` API, the script will use that instead.

## Estimate heart rate with POS

```bash
python article_pos_pipeline.py extract \
  --video /path/to/input_video.mp4 \
  --output-prefix outputs/input_video_pos
```

Outputs:

- `outputs/input_video_pos.summary.json`: video-level summary and mean/median BPM.
- `outputs/input_video_pos.windows.csv`: per-window BPM estimates.
- `outputs/input_video_pos.rgb.npy`: extracted RGB trace.
- `outputs/input_video_pos.bvp.npy`: extracted POS BVP signal.

Recommended default:

```bash
python article_pos_pipeline.py extract \
  --video data/sample_video.avi \
  --output-prefix outputs/sample_pos \
  --rgb-mode patches \
  --roi face \
  --window-seconds 8 \
  --stride-seconds 1 \
  --min-hz 0.65 \
  --max-hz 4.0
```

`--rgb-mode patches` uses the article's 100 landmark patches. `--roi` is ignored
for patch RGB extraction but kept in the summary. Use `--rgb-mode mask` if you
want a single ROI average instead.

## Modify a video like the article

Apply the article's best-performing privacy method, 20-frame sliding time averaging,
to the face ROI:

```bash
python article_pos_pipeline.py modify \
  --input /path/to/input_video.mp4 \
  --output outputs/input_video_tas.mp4 \
  --method time-average-sliding \
  --roi face \
  --time-window-frames 20
```

Other methods:

- `median`
- `gaussian`
- `bilateral`
- `gaussian-noise`
- `saltpepper-noise`
- `poisson-noise`
- `pepper-noise`
- `speckle-noise`
- `time-average-cumulative`
- `time-average-sliding`

Article-style defaults are `--kernel-size 5`, bilateral sigma values of `75`, and
`--time-window-frames 20`.

## Host as an API on Modal

`modal_app.py` wraps the hybrid TS-CAN + POS pipeline (`extract_bpm`) in a
FastAPI app and serves it on [Modal](https://modal.com/) so any client
(Python, JavaScript, mobile, video-call apps, …) can submit a video and
receive a BPM estimate.

### One-time Modal setup

```bash
python3 -m pip install modal
modal token new                # interactive browser login, OR
modal token set --token-id <id> --token-secret <secret> --profile=<your-profile>
```

### Iterate locally with a temporary URL

```bash
cd usable_pos_article
modal serve modal_app.py
```

Modal prints a temporary URL such as
`https://<workspace>--pos-rppg-bpm-fastapi-app-dev.modal.run`. Live-reloads on
file changes. Stop with Ctrl-C.

### Deploy to a permanent URL

```bash
cd usable_pos_article
modal deploy modal_app.py
```

After deploy you'll get a permanent base URL. The app exposes:

| Method | Path              | Description                                                |
|-------:|-------------------|------------------------------------------------------------|
| GET    | `/`               | Service metadata and defaults                              |
| GET    | `/health`         | Liveness probe                                             |
| POST   | `/analyze`        | `multipart/form-data` upload (field name: `video`)         |
| POST   | `/analyze-url`    | JSON `{"video_url": "https://..."}`                        |
| POST   | `/analyze-base64` | JSON `{"video_base64": "<base64>"}`                        |

All `/analyze*` endpoints accept the same optional tuning fields (form fields
on `/analyze`, JSON keys on the others):

| Field            | Type    | Default   | Description                                                                                                       |
|------------------|---------|-----------|-------------------------------------------------------------------------------------------------------------------|
| `method`         | string  | `auto`    | `auto` (TS-CAN, fall back to POS), `tscan` (force neural), `pos` (force the POS path), or `consensus` (see below).|
| `roi_mode`       | string  | `face`    | POS / consensus only. `face`, `selected`, or `full-frame`.                                                        |
| `rgb_mode`       | string  | `patches` | POS / consensus only. `patches` (100 landmark patches) or `mask`.                                                 |
| `patch_size`     | int     | `28`      | POS / consensus only. Per-landmark patch side length in pixels.                                                   |
| `window_seconds` | float   | `8.0`     | Analysis window length (POS only — consensus runs on the full clip).                                              |
| `stride_seconds` | float   | `1.0`     | Analysis window stride (POS only).                                                                                |
| `min_hz`         | float   | `0.65` (POS) / `0.7` (consensus) | Lower band-pass / PSD bound. TS-CAN uses a fixed `[0.75, 2.5] Hz` band internally.             |
| `max_hz`         | float   | `4.0` (POS) / `2.5` (consensus)  | Upper band-pass / PSD bound. TS-CAN uses a fixed `[0.75, 2.5] Hz` band internally.             |
| `max_frames`     | int     | _none_    | Cap on processed frames (latency control).                                                                        |

> **Implicit method=pos**: if you leave `method` at the default but
> explicitly set any POS-only field (`roi_mode`, `rgb_mode`, `patch_size`,
> `min_hz`, or `max_hz`), the API runs the POS pipeline with your
> settings rather than silently routing the call through TS-CAN (which
> ignores those fields). When this happens, the response includes
> `method_inferred_from: ["rgb_mode", ...]` so you can tell what
> triggered the switch. To run TS-CAN with your `rgb_mode=patches`
> request a no-op (TS-CAN doesn't use it), pass `method=tscan`
> explicitly.

### Multi-algorithm consensus (`method=consensus`)

When a clip is short, motion-affected, or otherwise ambiguous, a single
algorithm can be confidently wrong (POS may peak-pick the second
harmonic, ICA may lock onto a rigid-motion component, etc.). Setting
`method=consensus` runs **seven independent classical rPPG algorithms**
(POS, CHROM, GREEN, ICA, LGI, PBV, OMIT) on the **same** per-patch RGB
trace and combines their estimates using the rPPG-VQA paper's
signal-level consensus (Aboul-Naga et al., 2024 — *"Towards a Holistic
Quality Assessment Method for Camera-Based Photoplethysmography"*).

Pipeline (all on the full clip):

1. Read 100 face-skin patches per frame (`rgb_mode=patches`,
   `roi_mode=face` are recommended). For each algorithm, the BVP is
   computed *per-patch*, bandpassed to the cardiac band, and the
   patch-median BVP is returned — same motion-robust aggregation as the
   POS-only path.
2. For each algorithm extract `(BPM, SNR_db, PSD)` via Welch +
   parabolic peak interpolation in the consensus band
   (default `0.7–2.5 Hz` — the rPPG-VQA recommendation).
3. **RANSAC consensus HR (Eq. 4 of the paper)**: pick the largest set of
   algorithms whose BPM falls within `±5 BPM` of a candidate; treat them
   as inliers; reject the rest. The `frequency_weight` is a Gaussian
   penalty `exp(-((bpm_i - bpm_consensus)^2) / (2σ²))` with `σ = 5 BPM`.
4. **Pearson spectral correlation (Eq. 8 of the paper)**: every
   algorithm's PSD (in the consensus band) is correlated with every
   other inlier's PSD; the `corr_weight` of algorithm `i` is the mean
   correlation with the other inliers.
5. **Combined per-algorithm weight** = `frequency_weight × corr_weight`
   (clamped to `[0, 1]`).
6. **`q_sig` (signal quality score) ∈ [0, 1]** = mean combined weight
   across the inlier algorithms. Higher = stronger cross-algorithm
   agreement on a clean cardiac peak.
7. **Final BPM** = inverse-variance weighted mean of inlier BPMs (using
   the combined weights). Falls back to the algorithm with the highest
   single weight when only one algorithm is an inlier.
8. **Verdict**: `accepted` when at least **2** algorithms agree within
   `5 BPM`, otherwise `rejected`. Reject typically means "ask the user
   to stabilize the camera and improve lighting" — the BPM number is
   still returned so a permissive client can use it, but `q_sig` will be
   low and `consensus_verdict == "rejected"`.

Recommended invocation:

```bash
curl -X POST "$BASE_URL/analyze" \
  -F "video=@/path/to/clip.mp4" \
  -F "method=consensus" \
  -F "max_frames=900"
```

Consensus uses POS-style ROI/RGB extraction, so the same `roi_mode`,
`rgb_mode`, `patch_size`, `min_hz`, `max_hz`, and `max_frames` knobs
apply (`window_seconds` / `stride_seconds` are ignored — consensus
operates on the whole clip). When the caller doesn't specify a band the
API automatically uses `[0.7, 2.5] Hz` for consensus (the rPPG-VQA
recommendation) instead of the wider POS-tuned `[0.65, 4.0] Hz`, because
classical algorithms have broader spectra than TS-CAN and would
otherwise pick up subharmonics or 2nd harmonics outside the cardiac
band, destroying cross-algorithm agreement.

Consensus-specific response fields. The same scalars are exposed both at
the **top level** of the JSON (for clients that don't want to dig into
`summary`) and inside `summary` (for parity with the other methods):

| Field                                                       | Type   | Description                                                                                                                |
|-------------------------------------------------------------|--------|----------------------------------------------------------------------------------------------------------------------------|
| `summary.method`                                            | string | `consensus`.                                                                                                               |
| `q_sig` / `summary.q_sig`                                   | float  | Signal-level quality score in `[0, 1]`. Mean combined weight across inliers. Higher = stronger consensus.                  |
| `consensus_verdict` / `summary.consensus_verdict`           | string | `accepted` (≥ 2 inliers agree within 5 BPM) or `rejected`.                                                                 |
| `consensus_inliers` / `summary.consensus_inliers`           | int    | Number of algorithms in the inlier set.                                                                                    |
| `inlier_std_bpm` / `summary.inlier_std_bpm`                 | float  | BPM standard deviation across the inlier set — useful as a finer-grained quality cutoff than the binary verdict.           |
| `consensus_rejection_reason` / `summary.consensus_rejection_reason` | string \| null | Human-readable reason when the verdict is `rejected` (e.g. `"only 1 inlier"`, `"all algorithms failed"`); `null` on accept. |
| `per_algorithm` (top-level) / `summary.consensus_methods`   | list   | Per-algorithm breakdown: `[{name, bpm, snr_db, weight_freq, weight_corr, weight, inlier, error}]` — one entry per algorithm. |

Live benchmark (Modal `/analyze`, `method=consensus`, defaults) on the
user-provided test clip `lv_0_20260508141911.mp4` (truth `92–94 BPM`,
hard for single-algorithm methods):

| `method`             | `bpm`  | verdict   | notes                                                                            |
|----------------------|-------:|-----------|----------------------------------------------------------------------------------|
| `tscan`              | 94.92  | n/a       | Default; no tuning required.                                                     |
| `pos` (`max_hz=2.5`) | 93.67  | n/a       | Best of the legacy classical paths, still needs band tuning.                     |
| `consensus`          | 92.97  | accepted  | 3/7 inliers (POS=93.17, CHROM=93.39, LGI=92.45); `inlier_std_bpm = 0.40`; `q_sig ≈ 0.49`. |

> **When to use which method?**
> - `auto` / `tscan` for the lowest tail latency and the most robust
>   single number on motion-affected clips.
> - `pos` if you need a deterministic classical-only path (no PyTorch).
> - `consensus` if you need the API to **tell you whether to trust the
>   reading**. The `q_sig` + `consensus_verdict` fields let your client
>   reject low-quality measurements without re-running the analysis.

All endpoints return the same JSON shape:

```json
{
  "bpm": 72.0,
  "summary": {
    "fps": 30.0,
    "frames_read": 600,
    "frames_with_face": 580,
    "roi_mode": "face",
    "rgb_mode": "patches",
    "patch_size": 28,
    "window_seconds": 8.0,
    "stride_seconds": 1.0,
    "min_hz": 0.65,
    "max_hz": 4.0,
    "mean_bpm": 72.5,
    "median_bpm": 72.0,
    "valid_windows": 12,
    "total_windows": 13
  },
  "windows": [
    {"start_s": 0.0, "center_s": 4.0, "end_s": 8.0, "bpm": 72.0, "peak_hz": 1.2, "valid": true}
  ]
}
```

The convenience field `bpm` is the median BPM across valid windows (it
matches `summary.median_bpm`).

### Calling the API

`curl` (multipart upload):

```bash
curl -X POST "$BASE_URL/analyze" \
  -F "video=@/path/to/clip.mp4" \
  -F "max_frames=900"
```

`curl` (URL pull, ideal for large clips on object storage):

```bash
curl -X POST "$BASE_URL/analyze-url" \
  -H "Content-Type: application/json" \
  -d '{"video_url": "https://example.com/clip.mp4", "window_seconds": 6}'
```

Browser / `fetch` (e.g. from a video-call frontend):

```javascript
const form = new FormData();
form.append("video", videoBlob, "clip.webm");
const res = await fetch(`${BASE_URL}/analyze`, { method: "POST", body: form });
const { bpm, summary, windows } = await res.json();
```

Python (using the bundled `client_example.py`):

```bash
python client_example.py upload --url "$BASE_URL" --video clip.mp4
python client_example.py from-url --url "$BASE_URL" --video-url https://example.com/clip.mp4
```

### Limits / production notes

* `MAX_UPLOAD_BYTES = 200 MB` for `multipart/form-data` and base64 uploads.
  Larger payloads should use `/analyze-url`.
* `REQUEST_TIMEOUT_SECONDS = 600`. To keep tail latency bounded, pass
  `max_frames` (e.g. analyze a rolling 30 s window for live calls).
* The container is configured with `cpu=1.0`, `memory=2048`,
  `min_containers=1`, `scaledown_window=600`, and
  `@modal.concurrent(max_inputs=4)` so a single replica handles bursts and
  one container stays warm to avoid cold starts; Modal autoscales replicas
  under load.
* The MediaPipe `face_landmarker.task` model is downloaded lazily on the
  first request inside each container (cached at
  `~/.cache/article_pos_pipeline/face_landmarker.task`), so the first
  request after a cold start adds a few seconds of one-time download
  latency. Subsequent requests on the same container reuse the cached
  model.

## Important scientific limitations

This script makes the old code usable and aligns it with the article's method
description, but exact reproduction of the paper's reported numbers still requires:

- The same LGI-PPGI videos and ground-truth finger PPG signals.
- The paper's full released repository/pyVHR configuration.
- Matching evaluation code for BPM error, MSE, and overall score.
- Same available subjects/activities and same versions of dependencies.

Do not treat a single video's BPM output as medically exact. rPPG is sensitive to
lighting, motion, compression, skin visibility, and camera frame rate.

The provided `data/sample_video.avi` in the article repository is only about 5
seconds long, so the default 8-second article window cannot run on it. Use a
longer video for the true article settings, or add `--window-seconds 3` only for
a smoke test.
