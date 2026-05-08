# Hybrid TS-CAN + article-style POS rPPG pipeline

This is a modernized, usable replacement for the old POS pulse-extraction script.
It bundles **two heart-rate estimators** behind a single API and CLI:

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

| Field            | Type    | Default   | Description                                                                                  |
|------------------|---------|-----------|----------------------------------------------------------------------------------------------|
| `method`         | string  | `auto`    | `auto` (TS-CAN, fall back to POS), `tscan` (force neural), or `pos` (force the POS path).    |
| `roi_mode`       | string  | `face`    | POS-only. `face`, `selected`, or `full-frame`.                                               |
| `rgb_mode`       | string  | `patches` | POS-only. `patches` (100 landmark patches) or `mask`.                                        |
| `patch_size`     | int     | `28`      | POS-only. Per-landmark patch side length in pixels.                                          |
| `window_seconds` | float   | `8.0`     | Analysis window length.                                                                      |
| `stride_seconds` | float   | `1.0`     | Analysis window stride.                                                                      |
| `min_hz`         | float   | `0.65`    | POS-only. Lower band-pass / PSD bound. TS-CAN uses a fixed `[0.75, 2.5] Hz` band internally. |
| `max_hz`         | float   | `4.0`     | POS-only. Upper band-pass / PSD bound. TS-CAN uses a fixed `[0.75, 2.5] Hz` band internally. |
| `max_frames`     | int     | _none_    | Cap on processed frames (latency control).                                                   |

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
