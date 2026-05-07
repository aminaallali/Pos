"""Modal-hosted FastAPI service for the rPPG POS BPM pipeline.

This wraps :func:`article_pos_pipeline.extract_article_style_pos` in a small
FastAPI app and exposes it as a public HTTPS endpoint on Modal so that any
client (Python, JavaScript, mobile, etc.) can submit a video and receive a
BPM estimate.

Local development:

    pip install modal
    modal token new          # one-time auth
    modal serve modal_app.py # ephemeral URL with live reload

Deploy:

    modal deploy modal_app.py

After ``modal deploy`` you'll get a permanent base URL like
``https://<workspace>--pos-rppg-bpm-fastapi-app.modal.run``. The endpoints are:

* ``GET  /``               - service metadata
* ``GET  /health``         - liveness probe
* ``POST /analyze``        - ``multipart/form-data`` with field ``video``
* ``POST /analyze-url``    - JSON ``{"video_url": "https://..."}``
* ``POST /analyze-base64`` - JSON ``{"video_base64": "<base64-encoded video>"}``

All ``/analyze*`` endpoints accept the same optional tuning parameters either
as form fields (``/analyze``) or as JSON keys (``/analyze-url``,
``/analyze-base64``):

* ``roi_mode``       - ``face`` | ``selected`` | ``full-frame`` (default ``face``)
* ``rgb_mode``       - ``patches`` | ``mask`` (default ``patches``)
* ``patch_size``     - int, default ``28``
* ``window_seconds`` - float, default ``8.0``
* ``stride_seconds`` - float, default ``1.0``
* ``min_hz``         - float, default ``0.65``
* ``max_hz``         - float, default ``4.0``
* ``max_frames``     - optional int cap on processed frames

They all return the same JSON schema::

    {
        "bpm": <float|null>,        # convenience: median across valid windows
        "summary": {                # ExtractionSummary minus the local path
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
            {"start_s": 0.0, "center_s": 4.0, "end_s": 8.0, "bpm": 72.0,
             "peak_hz": 1.2, "valid": true}
        ]
    }
"""

from __future__ import annotations

import modal


APP_NAME = "pos-rppg-bpm"

FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_CACHE_PATH = "/root/.cache/article_pos_pipeline/face_landmarker.task"

# The processing function caps the request body at this size. Larger uploads
# should use ``/analyze-url`` so Modal streams the bytes directly from object
# storage.
MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB

# Hard ceiling on how long a single request is allowed to run. Long videos can
# be expensive; clients should pass ``max_frames`` to keep latency bounded.
REQUEST_TIMEOUT_SECONDS = 600


def _prefetch_face_landmarker_model() -> None:
    """Download the MediaPipe face landmarker model into the image."""
    import urllib.request
    from pathlib import Path

    target = Path(MODEL_CACHE_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return
    urllib.request.urlretrieve(FACE_LANDMARKER_URL, target)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "numpy<2.0",
        "scipy==1.13.1",
        "opencv-python-headless==4.10.0.84",
        "mediapipe==0.10.14",
        "fastapi[standard]==0.115.4",
        "python-multipart==0.0.12",
        "requests==2.32.3",
    )
    .add_local_python_source("article_pos_pipeline")
    .run_function(_prefetch_face_landmarker_model)
)

app = modal.App(APP_NAME, image=image)


@app.function(timeout=REQUEST_TIMEOUT_SECONDS, memory=4096, cpu=2.0)
@modal.concurrent(max_inputs=4)
@modal.asgi_app()
def fastapi_app():
    """Build and return the FastAPI application served by Modal."""
    import base64
    import binascii
    import math
    import tempfile
    from dataclasses import asdict
    from pathlib import Path
    from typing import Annotated, Any

    import requests
    from fastapi import (
        FastAPI,
        File,
        Form,
        HTTPException,
        Request,
        UploadFile,
    )
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse

    from article_pos_pipeline import extract_article_style_pos

    web_app = FastAPI(
        title="POS rPPG BPM API",
        description=(
            "Estimate heart rate (BPM) from a face video using the "
            "Plane-Orthogonal-to-Skin (POS) rPPG algorithm "
            "(Wang et al., IEEE TBME 2017)."
        ),
        version="1.0.0",
    )

    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    DEFAULTS: dict[str, Any] = {
        "roi_mode": "face",
        "rgb_mode": "patches",
        "patch_size": 28,
        "window_seconds": 8.0,
        "stride_seconds": 1.0,
        "min_hz": 0.65,
        "max_hz": 4.0,
        "max_frames": None,
    }

    ALLOWED_ROI = {"face", "selected", "full-frame"}
    ALLOWED_RGB = {"patches", "mask"}

    def _validated_options(raw: dict[str, Any]) -> dict[str, Any]:
        opts = {**DEFAULTS}
        for key in DEFAULTS:
            if key in raw and raw[key] is not None:
                opts[key] = raw[key]

        if opts["roi_mode"] not in ALLOWED_ROI:
            raise HTTPException(
                status_code=400,
                detail=f"roi_mode must be one of {sorted(ALLOWED_ROI)}",
            )
        if opts["rgb_mode"] not in ALLOWED_RGB:
            raise HTTPException(
                status_code=400,
                detail=f"rgb_mode must be one of {sorted(ALLOWED_RGB)}",
            )

        try:
            opts["patch_size"] = int(opts["patch_size"])
            opts["window_seconds"] = float(opts["window_seconds"])
            opts["stride_seconds"] = float(opts["stride_seconds"])
            opts["min_hz"] = float(opts["min_hz"])
            opts["max_hz"] = float(opts["max_hz"])
            if opts["max_frames"] is not None:
                opts["max_frames"] = int(opts["max_frames"])
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid option type: {exc}")

        if opts["patch_size"] < 4 or opts["patch_size"] > 96:
            raise HTTPException(status_code=400, detail="patch_size must be in [4, 96]")
        if opts["window_seconds"] <= 0 or opts["stride_seconds"] <= 0:
            raise HTTPException(
                status_code=400,
                detail="window_seconds and stride_seconds must be positive",
            )
        if opts["min_hz"] <= 0 or opts["max_hz"] <= opts["min_hz"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid min_hz / max_hz (need 0 < min_hz < max_hz)",
            )
        if opts["max_frames"] is not None and opts["max_frames"] < 30:
            raise HTTPException(
                status_code=400,
                detail="max_frames must be >= 30 if provided",
            )

        return opts

    def _run_pipeline(video_path: Path, opts: dict[str, Any]) -> dict[str, Any]:
        try:
            summary, estimates, _, _ = extract_article_style_pos(
                video_path=video_path,
                roi_mode=opts["roi_mode"],
                rgb_mode=opts["rgb_mode"],
                patch_size=opts["patch_size"],
                window_seconds=opts["window_seconds"],
                stride_seconds=opts["stride_seconds"],
                min_hz=opts["min_hz"],
                max_hz=opts["max_hz"],
                max_frames=opts["max_frames"],
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        summary_dict = asdict(summary)
        summary_dict.pop("video", None)
        windows = [
            {
                k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                for k, v in asdict(estimate).items()
            }
            for estimate in estimates
        ]
        return {
            "bpm": summary_dict.get("median_bpm"),
            "summary": summary_dict,
            "windows": windows,
        }

    def _save_to_tempfile(data: bytes, suffix: str = ".mp4") -> Path:
        if not data:
            raise HTTPException(status_code=400, detail="Empty video payload.")
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Video too large ({len(data)} bytes); "
                    f"max is {MAX_UPLOAD_BYTES} bytes. "
                    "Use /analyze-url for larger files."
                ),
            )
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        try:
            tmp.write(data)
            tmp.flush()
            return Path(tmp.name)
        finally:
            tmp.close()

    @web_app.get("/")
    def root() -> dict[str, Any]:
        return {
            "service": "POS rPPG BPM API",
            "version": "1.0.0",
            "algorithm": (
                "Plane-Orthogonal-to-Skin (POS) rPPG, "
                "Wang et al., IEEE TBME 2017"
            ),
            "endpoints": {
                "GET /": "this metadata",
                "GET /health": "liveness probe",
                "POST /analyze": (
                    "multipart/form-data upload, field name 'video'"
                ),
                "POST /analyze-url": (
                    "JSON body with 'video_url' (https) and optional tuning fields"
                ),
                "POST /analyze-base64": (
                    "JSON body with 'video_base64' (base64-encoded bytes)"
                ),
            },
            "limits": {
                "max_upload_bytes": MAX_UPLOAD_BYTES,
                "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
            },
            "defaults": DEFAULTS,
        }

    @web_app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @web_app.post("/analyze")
    async def analyze(
        video: Annotated[UploadFile, File(...)],
        roi_mode: Annotated[str | None, Form()] = None,
        rgb_mode: Annotated[str | None, Form()] = None,
        patch_size: Annotated[int | None, Form()] = None,
        window_seconds: Annotated[float | None, Form()] = None,
        stride_seconds: Annotated[float | None, Form()] = None,
        min_hz: Annotated[float | None, Form()] = None,
        max_hz: Annotated[float | None, Form()] = None,
        max_frames: Annotated[int | None, Form()] = None,
    ) -> JSONResponse:
        if not video.filename:
            raise HTTPException(status_code=400, detail="Missing 'video' file.")

        opts = _validated_options(
            {
                "roi_mode": roi_mode,
                "rgb_mode": rgb_mode,
                "patch_size": patch_size,
                "window_seconds": window_seconds,
                "stride_seconds": stride_seconds,
                "min_hz": min_hz,
                "max_hz": max_hz,
                "max_frames": max_frames,
            }
        )

        suffix = Path(video.filename).suffix or ".mp4"
        data = await video.read()
        path = _save_to_tempfile(data, suffix=suffix)
        try:
            return JSONResponse(_run_pipeline(path, opts))
        finally:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    @web_app.post("/analyze-url")
    def analyze_url(payload: dict[str, Any]) -> JSONResponse:
        url = payload.get("video_url")
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            raise HTTPException(
                status_code=400,
                detail="Provide 'video_url' as an http(s) URL.",
            )

        opts = _validated_options(payload)

        suffix = Path(url.split("?")[0]).suffix or ".mp4"
        try:
            with requests.get(url, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                content_length = resp.headers.get("Content-Length")
                if content_length and int(content_length) > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"Remote video too large ({content_length} bytes); "
                            f"max is {MAX_UPLOAD_BYTES} bytes."
                        ),
                    )
                buf = bytearray()
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    if not chunk:
                        continue
                    buf.extend(chunk)
                    if len(buf) > MAX_UPLOAD_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail=(
                                f"Remote video too large; "
                                f"max is {MAX_UPLOAD_BYTES} bytes."
                            ),
                        )
                data = bytes(buf)
        except requests.RequestException as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to fetch video_url: {exc}",
            )

        path = _save_to_tempfile(data, suffix=suffix)
        try:
            return JSONResponse(_run_pipeline(path, opts))
        finally:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    @web_app.post("/analyze-base64")
    def analyze_base64(payload: dict[str, Any]) -> JSONResponse:
        b64 = payload.get("video_base64")
        if not isinstance(b64, str) or not b64:
            raise HTTPException(
                status_code=400,
                detail="Provide 'video_base64' as a base64-encoded video.",
            )

        opts = _validated_options(payload)

        # Strip data URL prefix if present (e.g. ``data:video/mp4;base64,...``).
        if b64.startswith("data:"):
            try:
                b64 = b64.split(",", 1)[1]
            except IndexError:
                raise HTTPException(
                    status_code=400,
                    detail="Malformed data URL in video_base64.",
                )

        try:
            data = base64.b64decode(b64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid base64 in video_base64: {exc}",
            )

        suffix = payload.get("filename")
        if isinstance(suffix, str) and suffix:
            ext = Path(suffix).suffix or ".mp4"
        else:
            ext = ".mp4"

        path = _save_to_tempfile(data, suffix=ext)
        try:
            return JSONResponse(_run_pipeline(path, opts))
        finally:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    @web_app.exception_handler(Exception)
    async def _unhandled_exception_handler(_request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error", "detail": str(exc)},
        )

    return web_app
