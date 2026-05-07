"""Reference client for the POS rPPG BPM Modal API.

Usage::

    python client_example.py upload   --url https://yyy.modal.run --video clip.mp4
    python client_example.py from-url --url https://yyy.modal.run --video-url https://example.com/clip.mp4
    python client_example.py base64   --url https://yyy.modal.run --video clip.mp4

Replace ``https://yyy.modal.run`` with the URL printed by ``modal deploy modal_app.py``.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import requests


def upload(api_url: str, video: Path, **opts: object) -> dict:
    with video.open("rb") as handle:
        files = {"video": (video.name, handle, "application/octet-stream")}
        data = {key: str(value) for key, value in opts.items() if value is not None}
        response = requests.post(f"{api_url.rstrip('/')}/analyze", files=files, data=data, timeout=600)
    response.raise_for_status()
    return response.json()


def from_url(api_url: str, video_url: str, **opts: object) -> dict:
    body: dict[str, object] = {"video_url": video_url}
    body.update({key: value for key, value in opts.items() if value is not None})
    response = requests.post(f"{api_url.rstrip('/')}/analyze-url", json=body, timeout=600)
    response.raise_for_status()
    return response.json()


def from_base64(api_url: str, video: Path, **opts: object) -> dict:
    encoded = base64.b64encode(video.read_bytes()).decode("ascii")
    body: dict[str, object] = {"video_base64": encoded, "filename": video.name}
    body.update({key: value for key, value in opts.items() if value is not None})
    response = requests.post(f"{api_url.rstrip('/')}/analyze-base64", json=body, timeout=600)
    response.raise_for_status()
    return response.json()


def _add_common_opts(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--url", required=True, help="Base URL of the deployed Modal app")
    parser.add_argument("--roi-mode", choices=["face", "selected", "full-frame"], default=None)
    parser.add_argument("--rgb-mode", choices=["patches", "mask"], default=None)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--window-seconds", type=float, default=None)
    parser.add_argument("--stride-seconds", type=float, default=None)
    parser.add_argument("--min-hz", type=float, default=None)
    parser.add_argument("--max-hz", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=None)


def _opts_from_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        "roi_mode": args.roi_mode,
        "rgb_mode": args.rgb_mode,
        "patch_size": args.patch_size,
        "window_seconds": args.window_seconds,
        "stride_seconds": args.stride_seconds,
        "min_hz": args.min_hz,
        "max_hz": args.max_hz,
        "max_frames": args.max_frames,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Reference client for the POS rPPG BPM Modal API.")
    sub = parser.add_subparsers(dest="command", required=True)

    upload_parser = sub.add_parser("upload", help="multipart upload via /analyze")
    _add_common_opts(upload_parser)
    upload_parser.add_argument("--video", required=True, type=Path)

    url_parser = sub.add_parser("from-url", help="POST /analyze-url with a public video URL")
    _add_common_opts(url_parser)
    url_parser.add_argument("--video-url", required=True)

    b64_parser = sub.add_parser("base64", help="base64 upload via /analyze-base64")
    _add_common_opts(b64_parser)
    b64_parser.add_argument("--video", required=True, type=Path)

    args = parser.parse_args()
    opts = _opts_from_args(args)

    if args.command == "upload":
        result = upload(args.url, args.video, **opts)
    elif args.command == "from-url":
        result = from_url(args.url, args.video_url, **opts)
    elif args.command == "base64":
        result = from_base64(args.url, args.video, **opts)
    else:
        parser.error(f"Unknown command: {args.command}")
        return 2

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
