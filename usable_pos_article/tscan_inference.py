"""TS-CAN (Liu et al., NeurIPS 2020) inference for rPPG heart-rate estimation.

The TSCAN model architecture in this file is adapted from
``ubicomplab/rPPG-Toolbox`` (forked at ``aminaallali/rPPG-Toolbox``), and the
pretrained weights distributed at ``models/UBFC-rPPG_TSCAN.pth`` are the
toolbox's UBFC-rPPG-trained checkpoint.

Both the architecture and the weights are subject to the toolbox's
"Responsible Artificial Intelligence Source Code License" (RAIL). A copy is
included at ``THIRD_PARTY_LICENSES/rPPG-Toolbox-LICENSE.txt``. In particular,
section 3.2 of that license restricts certain inferential uses; downstream
deployments must comply with those restrictions.

The pipeline below reproduces the toolbox's UBFC-rPPG inference exactly:

1. Per-frame face crop (1.5x bbox around the MediaPipe landmarks) resized to
   72x72.
2. Two-branch input: ``DiffNormalized`` (3 channels, scaled per-clip stddev)
   concatenated with ``Standardized`` (3 channels, z-scored) -> 6 channels.
3. Forward pass through the trained ``TSCAN(frame_depth=10, img_size=72)``,
   truncated to a multiple of ``frame_depth``.
4. Post-process: cumulative sum (the network outputs a *differentiated* PPG),
   smoothness-prior detrend (lambda=100, Tarvainen 2002), 1st-order
   Butterworth bandpass [0.75, 2.5] Hz.

Reference: Liu, Fromm, Patel, McDuff. "Multi-Task Temporal Shift Attention
Networks for On-Device Contactless Vitals Measurement." NeurIPS 2020.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np


_DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "models" / "UBFC-rPPG_TSCAN.pth"
_FRAME_DEPTH = 10
_IMG_SIZE = 72
_LARGE_BOX_COEF = 1.5  # matches rPPG-Toolbox UBFC-rPPG_TSCAN config


# ---------------------------------------------------------------------------
# Model architecture (vendored from rPPG-Toolbox neural_methods/model/TS_CAN.py)
# ---------------------------------------------------------------------------

def _build_tscan_module():
    """Return the TSCAN ``torch.nn.Module`` class.

    ``torch`` is imported lazily so importing this module on machines without
    PyTorch (e.g. the article-style POS-only path) doesn't fail.
    """

    import torch
    import torch.nn as nn

    class _AttentionMask(nn.Module):
        def forward(self, x):
            xsum = torch.sum(x, dim=2, keepdim=True)
            xsum = torch.sum(xsum, dim=3, keepdim=True)
            xshape = tuple(x.size())
            return x / xsum * xshape[2] * xshape[3] * 0.5

    class _TSM(nn.Module):
        def __init__(self, n_segment: int = 10, fold_div: int = 3) -> None:
            super().__init__()
            self.n_segment = n_segment
            self.fold_div = fold_div

        def forward(self, x):
            nt, c, h, w = x.size()
            n_batch = nt // self.n_segment
            x = x.reshape(n_batch, self.n_segment, c, h, w)
            fold = c // self.fold_div
            out = torch.zeros_like(x)
            out[:, :-1, :fold] = x[:, 1:, :fold]
            out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]
            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]
            return out.reshape(nt, c, h, w)

    class TSCAN(nn.Module):
        """Two-stream temporal-shift attention CNN for rPPG.

        Input layout: ``(T, 6, H, W)`` where channels [0:3] are
        ``DiffNormalized`` frames and channels [3:6] are ``Standardized``
        frames. Output: ``(T, 1)`` differential BVP.
        """

        def __init__(
            self,
            in_channels: int = 3,
            nb_filters1: int = 32,
            nb_filters2: int = 64,
            kernel_size: int = 3,
            dropout_rate1: float = 0.25,
            dropout_rate2: float = 0.5,
            pool_size: tuple[int, int] = (2, 2),
            nb_dense: int = 128,
            frame_depth: int = 10,
            img_size: int = 72,
        ) -> None:
            super().__init__()
            self.TSM_1 = _TSM(n_segment=frame_depth)
            self.TSM_2 = _TSM(n_segment=frame_depth)
            self.TSM_3 = _TSM(n_segment=frame_depth)
            self.TSM_4 = _TSM(n_segment=frame_depth)
            self.motion_conv1 = nn.Conv2d(in_channels, nb_filters1, kernel_size, padding=1, bias=True)
            self.motion_conv2 = nn.Conv2d(nb_filters1, nb_filters1, kernel_size, bias=True)
            self.motion_conv3 = nn.Conv2d(nb_filters1, nb_filters2, kernel_size, padding=1, bias=True)
            self.motion_conv4 = nn.Conv2d(nb_filters2, nb_filters2, kernel_size, bias=True)
            self.apperance_conv1 = nn.Conv2d(in_channels, nb_filters1, kernel_size, padding=1, bias=True)
            self.apperance_conv2 = nn.Conv2d(nb_filters1, nb_filters1, kernel_size, bias=True)
            self.apperance_conv3 = nn.Conv2d(nb_filters1, nb_filters2, kernel_size, padding=1, bias=True)
            self.apperance_conv4 = nn.Conv2d(nb_filters2, nb_filters2, kernel_size, bias=True)
            self.apperance_att_conv1 = nn.Conv2d(nb_filters1, 1, kernel_size=1, bias=True)
            self.attn_mask_1 = _AttentionMask()
            self.apperance_att_conv2 = nn.Conv2d(nb_filters2, 1, kernel_size=1, bias=True)
            self.attn_mask_2 = _AttentionMask()
            self.avg_pooling_1 = nn.AvgPool2d(pool_size)
            self.avg_pooling_2 = nn.AvgPool2d(pool_size)
            self.avg_pooling_3 = nn.AvgPool2d(pool_size)
            self.dropout_1 = nn.Dropout(dropout_rate1)
            self.dropout_2 = nn.Dropout(dropout_rate1)
            self.dropout_3 = nn.Dropout(dropout_rate1)
            self.dropout_4 = nn.Dropout(dropout_rate2)
            if img_size == 72:
                in_features = 16384
            elif img_size == 36:
                in_features = 3136
            elif img_size == 96:
                in_features = 30976
            else:
                raise ValueError(f"Unsupported img_size: {img_size}")
            self.final_dense_1 = nn.Linear(in_features, nb_dense, bias=True)
            self.final_dense_2 = nn.Linear(nb_dense, 1, bias=True)

        def forward(self, x):  # x: (T, 6, H, W)
            diff_input = x[:, :3, :, :]
            raw_input = x[:, 3:, :, :]

            d1 = torch.tanh(self.motion_conv1(self.TSM_1(diff_input)))
            d2 = torch.tanh(self.motion_conv2(self.TSM_2(d1)))

            r1 = torch.tanh(self.apperance_conv1(raw_input))
            r2 = torch.tanh(self.apperance_conv2(r1))

            g1 = self.attn_mask_1(torch.sigmoid(self.apperance_att_conv1(r2)))
            d4 = self.dropout_1(self.avg_pooling_1(d2 * g1))
            r4 = self.dropout_2(self.avg_pooling_2(r2))

            d5 = torch.tanh(self.motion_conv3(self.TSM_3(d4)))
            d6 = torch.tanh(self.motion_conv4(self.TSM_4(d5)))
            r5 = torch.tanh(self.apperance_conv3(r4))
            r6 = torch.tanh(self.apperance_conv4(r5))

            g2 = self.attn_mask_2(torch.sigmoid(self.apperance_att_conv2(r6)))
            d8 = self.dropout_3(self.avg_pooling_3(d6 * g2))
            d9 = d8.reshape(d8.size(0), -1)
            d10 = torch.tanh(self.final_dense_1(d9))
            return self.final_dense_2(self.dropout_4(d10))

    return TSCAN


_TSCAN_CACHE: dict[Path, object] = {}


def _load_model(weights_path: Path = _DEFAULT_WEIGHTS):
    """Lazy-build TSCAN, load the pretrained UBFC-rPPG weights, return it."""
    cached = _TSCAN_CACHE.get(weights_path)
    if cached is not None:
        return cached

    import torch  # noqa: F401

    if not weights_path.exists():
        raise FileNotFoundError(
            f"TS-CAN weights not found at {weights_path}. "
            "Bundle the UBFC-rPPG_TSCAN.pth file with the deployment."
        )

    TSCAN = _build_tscan_module()
    model = TSCAN(frame_depth=_FRAME_DEPTH, img_size=_IMG_SIZE)
    state_dict = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    # Toolbox checkpoints are saved through ``DataParallel``; strip the prefix.
    cleaned = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=True)
    model.eval()
    _TSCAN_CACHE[weights_path] = model
    return model


# ---------------------------------------------------------------------------
# Frame ingest: face crop + 1.5x bbox + 72x72 resize.
# ---------------------------------------------------------------------------

def _expand_square_bbox(
    xs: np.ndarray,
    ys: np.ndarray,
    width: int,
    height: int,
    large_coef: float = _LARGE_BOX_COEF,
) -> tuple[int, int, int, int]:
    cx = (xs.min() + xs.max()) / 2.0
    cy = (ys.min() + ys.max()) / 2.0
    half = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2.0 * large_coef
    x0 = max(0, int(math.floor(cx - half)))
    y0 = max(0, int(math.floor(cy - half)))
    x1 = min(width, int(math.ceil(cx + half)))
    y1 = min(height, int(math.ceil(cy + half)))
    return x0, y0, x1, y1


def read_face_crops_for_tscan(
    video_path: Path,
    face_mesh_processor,
    target_size: int = _IMG_SIZE,
    max_frames: int | None = None,
) -> tuple[np.ndarray, float, int, int, np.ndarray]:
    """Return (crops, fps, frames_read, frames_with_face, timestamps_s).

    ``crops`` has shape ``(T, target_size, target_size, 3)`` with values in
    [0, 1]. ``timestamps_s`` is the per-kept-frame container timestamp from
    ``CAP_PROP_POS_MSEC`` in seconds, useful for VFR resampling later if we
    ever want it.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

    crops: list[np.ndarray] = []
    timestamps: list[float] = []
    frames_read = 0
    frames_with_face = 0
    try:
        while True:
            if max_frames is not None and frames_read >= max_frames:
                break
            ok, bgr = cap.read()
            if not ok:
                break
            frames_read += 1
            t_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            lm = face_mesh_processor.landmarks(rgb)
            if lm is None:
                continue
            frames_with_face += 1
            h, w = rgb.shape[:2]
            x0, y0, x1, y1 = _expand_square_bbox(lm[:, 0], lm[:, 1], w, h)
            crop = rgb[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_AREA)
            crops.append(crop.astype(np.float32) / 255.0)
            timestamps.append(t_ms / 1000.0)
    finally:
        cap.release()

    if not crops:
        raise RuntimeError("No face frames detected; TS-CAN needs at least frame_depth=10.")

    return (
        np.stack(crops, axis=0),
        fps,
        frames_read,
        frames_with_face,
        np.asarray(timestamps, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Toolbox preprocessing (DiffNormalized + Standardized) and postprocessing
# (cumsum + detrend + bandpass).
# ---------------------------------------------------------------------------

def _diff_normalized(data: np.ndarray) -> np.ndarray:
    """Toolbox ``BaseLoader.diff_normalize_data``.

    First-difference along time, normalised by the difference's stddev,
    zero-padded back to length T at the tail.
    """
    n, h, w, c = data.shape
    out_len = n - 1
    out = np.zeros((out_len, h, w, c), dtype=np.float32)
    for j in range(out_len):
        out[j] = (data[j + 1] - data[j]) / (data[j + 1] + data[j] + 1e-7)
    std = float(np.std(out)) + 1e-12
    out /= std
    out = np.append(out, np.zeros((1, h, w, c), dtype=np.float32), axis=0)
    out[np.isnan(out)] = 0.0
    return out


def _standardized(data: np.ndarray) -> np.ndarray:
    """Toolbox ``BaseLoader.standardized_data`` (z-score over the whole tensor)."""
    out = (data - float(np.mean(data))) / (float(np.std(data)) + 1e-12)
    out[np.isnan(out)] = 0.0
    return out.astype(np.float32)


def _detrend_smoothness_prior(signal: np.ndarray, lambda_value: float = 100.0) -> np.ndarray:
    """Tarvainen (2002) smoothness-prior detrending.

    Mirrors ``rPPG-Toolbox/evaluation/post_process._detrend``. Equivalent to
    fitting a smoothness-penalised baseline and subtracting it.
    """
    from scipy import sparse

    n = signal.shape[0]
    H = np.identity(n)
    ones = np.ones(n)
    minus_twos = -2 * np.ones(n)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = sparse.spdiags(diags_data, diags_index, n - 2, n).toarray()
    detrended = (H - np.linalg.inv(H + (lambda_value ** 2) * D.T @ D)) @ signal
    return detrended


def post_process_tscan_bvp(
    diff_bvp: np.ndarray,
    fs: float,
    low_pass: float = 0.75,
    high_pass: float = 2.5,
) -> np.ndarray:
    """Reverse ``DiffNormalized`` (cumsum), detrend, bandpass.

    Matches ``calculate_metric_per_video(diff_flag=True, use_bandpass=True)``
    in the rPPG-Toolbox.
    """
    from scipy.signal import butter, filtfilt

    bvp = np.cumsum(np.asarray(diff_bvp, dtype=np.float64).reshape(-1))
    bvp = _detrend_smoothness_prior(bvp, lambda_value=100.0)
    b, a = butter(1, [low_pass / fs * 2.0, high_pass / fs * 2.0], btype="bandpass")
    return filtfilt(b, a, np.double(bvp))


def _next_power_of_2(x: int) -> int:
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def calculate_fft_hr(
    bvp: np.ndarray,
    fs: float,
    low_pass: float = 0.75,
    high_pass: float = 2.5,
) -> float:
    """FFT-periodogram HR estimate, matching ``rPPG-Toolbox _calculate_fft_hr``.

    Uses ``scipy.signal.periodogram`` with ``nfft = next_power_of_2(N)`` so the
    PSD bin width is ``fs / nfft`` (e.g. 0.029 Hz = 1.76 BPM at 30 fps with
    290 samples padded to 512), much finer than an unpadded welch / flattop
    window. This is what the toolbox papers use to report neural-method HR.
    """
    from scipy.signal import periodogram

    sig = np.asarray(bvp, dtype=np.float64).reshape(-1)
    if sig.size < 4 or not np.any(np.isfinite(sig)):
        return float("nan")
    n = _next_power_of_2(sig.size)
    freqs, psd = periodogram(sig[np.newaxis, :], fs=fs, nfft=n, detrend=False)
    freqs = np.asarray(freqs).ravel()
    psd = np.asarray(psd).ravel()
    mask = (freqs >= low_pass) & (freqs <= high_pass)
    if not np.any(mask):
        return float("nan")
    band_freqs = freqs[mask]
    band_psd = psd[mask]
    return float(band_freqs[int(np.argmax(band_psd))] * 60.0)


# ---------------------------------------------------------------------------
# End-to-end: face_crops -> BVP at fs ready for sliding-window HR estimation.
# ---------------------------------------------------------------------------

def tscan_bvp_from_crops(
    crops: np.ndarray,
    weights_path: Path = _DEFAULT_WEIGHTS,
    low_pass: float = 0.75,
    high_pass: float = 2.5,
    fs: float = 30.0,
) -> np.ndarray:
    """Run TS-CAN forward + post-process and return the cleaned BVP.

    Parameters
    ----------
    crops:
        ``(T, 72, 72, 3)`` ``float32`` array in ``[0, 1]``.
    weights_path:
        Pretrained checkpoint path (default: bundled UBFC-rPPG weights).
    low_pass, high_pass:
        Bandpass corners for the post-processed BVP, in Hz.
    fs:
        Sampling rate (frames per second) for the bandpass filter.

    Returns
    -------
    np.ndarray
        1-D BVP of length ``floor(T / 10) * 10``. Values are in arbitrary
        units; downstream code should run a spectral peak finder for HR.
    """
    import torch

    if crops.ndim != 4 or crops.shape[1] != _IMG_SIZE or crops.shape[2] != _IMG_SIZE or crops.shape[3] != 3:
        raise ValueError(f"crops must have shape (T, {_IMG_SIZE}, {_IMG_SIZE}, 3); got {crops.shape}")
    if crops.shape[0] < _FRAME_DEPTH:
        raise RuntimeError(
            f"TS-CAN needs at least {_FRAME_DEPTH} face frames; got {crops.shape[0]}."
        )

    diff = _diff_normalized(crops)
    std = _standardized(crops)
    network_input = np.concatenate([diff, std], axis=-1).transpose(0, 3, 1, 2)
    truncated = (network_input.shape[0] // _FRAME_DEPTH) * _FRAME_DEPTH
    network_input = network_input[:truncated]

    model = _load_model(weights_path)
    with torch.no_grad():
        out = model(torch.from_numpy(network_input).float())
    diff_bvp = out.squeeze(-1).cpu().numpy()
    return post_process_tscan_bvp(diff_bvp, fs=fs, low_pass=low_pass, high_pass=high_pass)


def default_weights_path() -> Path:
    """Where the bundled TS-CAN checkpoint lives in the deployed image."""
    return _DEFAULT_WEIGHTS


def frame_depth() -> int:
    return _FRAME_DEPTH


def img_size() -> int:
    return _IMG_SIZE
