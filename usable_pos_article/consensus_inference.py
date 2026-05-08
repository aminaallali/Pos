"""Consensus rPPG (rPPG-VQA) — multi-method signal-quality / heart-rate
estimator.

Runs seven classical rPPG algorithms in parallel on the same RGB trace,
combines them into a single heart-rate estimate, and reports a unified
signal-quality score ``q_sig`` plus a per-algorithm breakdown.

The seven algorithms — POS, CHROM, GREEN, ICA, LGI, PBV, OMIT — and the
classical-method bodies are adapted from the rPPG-Toolbox project
(github.com/ubicomplab/rPPG-Toolbox), under its Responsible AI License
(see ``THIRD_PARTY_LICENSES/rPPG-Toolbox-LICENSE.txt``).

Aggregation logic follows the rPPG-VQA paper:

  - **Frequency-consistency weight** (Eq. 4):
    ``w_freq_i = exp(-(hr_i - hr_consensus)^2 / (2 * sigma_freq^2))``,
    where ``hr_consensus`` is the largest cluster of HR estimates within
    ``RANSAC_TAU_BPM`` of any candidate (a small-N RANSAC).

  - **Spectral-correlation weight** (Eq. 8):
    ``w_corr_i = mean_{j != i} pearson(psd_i, psd_j)``,
    clipped to [0, 1].

  - **Combined weight** ``w_i = w_freq_i * w_corr_i``.
  - **Final HR** = weighted mean of ``hr_i`` with weights ``w_i``.
  - **q_sig**  = mean weight across inliers (in [0, 1]).
  - **Verdict** = ``"accepted"`` if ``q_sig >= QSIG_MIN_FOR_ACCEPT`` and
    ``inlier_std_bpm <= INLIER_STD_BPM_MAX``, else ``"rejected"``.

References:

  - Wang et al. 2017 (POS); De Haan & Jeanne 2013 (CHROM); Verkruysse 2008
    (GREEN); Pilz 2018 (LGI); De Haan & van Leest 2014 (PBV); Casado &
    Bordallo 2023 (OMIT); Poh, McDuff & Picard 2010 (ICA).
"""

import math
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np
from scipy import signal, sparse


CONSENSUS_METHODS: tuple[str, ...] = (
    "pos",
    "chrom",
    "green",
    "ica",
    "lgi",
    "pbv",
    "omit",
)

# Defaults tuned to the rPPG-VQA paper / classical-rPPG conventions.
DEFAULT_MIN_HZ = 0.7  # 42 BPM
DEFAULT_MAX_HZ = 2.5  # 150 BPM

# Consensus thresholds.
RANSAC_TAU_BPM = 5.0
GAUSS_SIGMA_BPM = 5.0
INLIER_STD_BPM_MAX = 5.0
MIN_INLIERS_FOR_ACCEPT = 2
MIN_SECONDS_FOR_CONSENSUS = 4.0


# ---------------------------------------------------------------------------
# Vendored classical rPPG algorithms (adapted from rPPG-Toolbox to operate on
# a pre-computed RGB trace of shape (N, 3) instead of raw video frames).
# ---------------------------------------------------------------------------


def _detrend(input_signal: np.ndarray, lambda_value: float) -> np.ndarray:
    """Smoothness-prior detrend (Tarvainen). Same recipe as rPPG-Toolbox."""
    n = input_signal.shape[0]
    H = np.identity(n)
    ones = np.ones(n)
    minus_twos = -2 * np.ones(n)
    diags_data = np.array([ones, minus_twos, ones])
    D = sparse.spdiags(diags_data, np.array([0, 1, 2]), n - 2, n).toarray()
    return np.dot(
        H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D)),
        input_signal,
    )


def _bandpass(bvp: np.ndarray, fs: float, lo_hz: float, hi_hz: float) -> np.ndarray:
    nyq = 0.5 * fs
    lo = max(lo_hz / nyq, 1e-3)
    hi = min(hi_hz / nyq, 0.999)
    b, a = signal.butter(3, [lo, hi], btype="bandpass")
    return signal.filtfilt(b, a, bvp.astype(np.float64))


def _pos_bvp(rgb: np.ndarray, fs: float) -> np.ndarray:
    """Wang et al. 2017 POS — adapted from rPPG-Toolbox's POS_WANG to operate
    on an already-extracted RGB trace.
    """
    win_sec = 1.6
    n = rgb.shape[0]
    H = np.zeros(n)
    L = max(2, math.ceil(win_sec * fs))
    for i in range(n):
        m = i - L
        if m < 0:
            continue
        Cn = rgb[m:i, :] / np.mean(rgb[m:i, :], axis=0)
        Cn = Cn.T  # (3, L)
        S = np.array([[0, 1, -1], [-2, 1, 1]]) @ Cn  # (2, L)
        s0_std = np.std(S[0]) or 1e-12
        s1_std = np.std(S[1]) or 1e-12
        h = S[0] + (s0_std / s1_std) * S[1]
        h = h - np.mean(h)
        H[m:i] = H[m:i] + h
    bvp = _detrend(H, 100.0)
    return bvp


def _chrom_bvp(rgb: np.ndarray, fs: float) -> np.ndarray:
    """De Haan & Jeanne 2013 CHROM — adapted from rPPG-Toolbox.
    Pre-bandpass [0.7, 2.5] Hz applied inside as in the upstream impl.
    """
    win_sec = 1.6
    n = rgb.shape[0]
    nyq = 0.5 * fs
    b, a = signal.butter(3, [0.7 / nyq, 2.5 / nyq], "bandpass")
    win_l = max(2, math.ceil(win_sec * fs))
    if win_l % 2:
        win_l += 1
    n_win = max(0, math.floor((n - win_l // 2) / (win_l // 2)))
    total = (win_l // 2) * (n_win + 1)
    if total < 2:
        return np.zeros(n)
    S = np.zeros(total)
    win_s = 0
    win_m = win_l // 2
    win_e = win_l
    for _ in range(n_win):
        seg = rgb[win_s:win_e]
        base = np.mean(seg, axis=0)
        if np.any(base == 0):
            base = np.where(base == 0, 1e-12, base)
        norm = seg / base
        Xs = 3 * norm[:, 0] - 2 * norm[:, 1]
        Ys = 1.5 * norm[:, 0] + norm[:, 1] - 1.5 * norm[:, 2]
        try:
            Xf = signal.filtfilt(b, a, Xs)
            Yf = signal.filtfilt(b, a, Ys)
        except ValueError:
            return np.zeros(n)
        x_std = np.std(Xf) or 1e-12
        y_std = np.std(Yf) or 1e-12
        alpha = x_std / y_std
        s_win = (Xf - alpha * Yf) * signal.windows.hann(win_l)
        S[win_s:win_m] += s_win[: win_l // 2]
        S[win_m:win_e] = s_win[win_l // 2:]
        win_s = win_m
        win_m = win_s + win_l // 2
        win_e = win_s + win_l
    # Pad/truncate to N for downstream consumption.
    if S.shape[0] < n:
        return np.concatenate([S, np.zeros(n - S.shape[0])])
    return S[:n]


def _green_bvp(rgb: np.ndarray, fs: float) -> np.ndarray:
    """Verkruysse et al. 2008 — green channel only."""
    return rgb[:, 1].astype(np.float64).copy()


def _lgi_bvp(rgb: np.ndarray, fs: float) -> np.ndarray:
    """Pilz et al. 2018 LGI — local-group-invariance projection."""
    rgb_t = rgb.T  # (3, N)
    rgb_3d = rgb_t[None, :, :]  # (1, 3, N) — rPPG-Toolbox expects this shape.
    U, _, _ = np.linalg.svd(rgb_3d)
    S = U[:, :, 0][:, :, None]  # (1, 3, 1)
    SST = np.matmul(S, np.swapaxes(S, 1, 2))  # (1, 3, 3)
    P = np.tile(np.identity(3), (1, 1, 1)) - SST
    Y = np.matmul(P, rgb_3d)  # (1, 3, N)
    return Y[0, 1, :].astype(np.float64).copy()


def _pbv_bvp(rgb: np.ndarray, fs: float) -> np.ndarray:
    """De Haan & van Leest 2014 PBV — blood-volume-pulse signature.

    Re-derived in 2-D to be agnostic to the numpy 1.x → 2.x change in how
    ``np.linalg.solve`` interprets right-hand-side shapes (gufunc signature
    became strictly ``(m,m),(m,n)->(m,n)`` in numpy 2.0; the upstream
    rPPG-Toolbox formulation relies on the older 1-D broadcast).
    """
    n = rgb.shape[0]
    rgb_t = rgb.T.astype(np.float64)  # (3, N)
    sig_mean = np.mean(rgb_t, axis=1, keepdims=True)  # (3, 1)
    sig_mean = np.where(sig_mean == 0, 1e-12, sig_mean)
    norm = rgb_t / sig_mean  # (3, N)
    pbv_n = np.std(norm, axis=1)  # (3,)
    pbv_d = math.sqrt(float(np.var(norm, axis=1).sum()))
    if pbv_d == 0:
        return np.zeros(n)
    pbv = pbv_n / pbv_d  # (3,)
    Q = norm @ norm.T  # (3, 3)
    try:
        W = np.linalg.solve(Q, pbv)  # (3,)
    except np.linalg.LinAlgError:
        return np.zeros(n)
    A = norm.T @ W  # (N,)
    denom = float(pbv @ W)
    if abs(denom) < 1e-12:
        return np.zeros(n)
    return (A / denom).astype(np.float64)


def _omit_bvp(rgb: np.ndarray, fs: float) -> np.ndarray:
    """Casado & Bordallo 2023 OMIT — orthogonal-matrix image transformation."""
    rgb_t = rgb.T  # (3, N)
    Q, _ = np.linalg.qr(rgb_t)
    S = Q[:, 0].reshape(1, -1)
    P = np.identity(3) - np.matmul(S.T, S)
    Y = np.dot(P, rgb_t)
    return Y[1, :].astype(np.float64).copy()


def _ica_bvp(rgb: np.ndarray, fs: float) -> np.ndarray:
    """Poh et al. 2010-style ICA — detrend + standardise the three channels,
    run blind-source separation (FastICA when scikit-learn is available,
    otherwise a PCA-on-whitened-signals fallback), pick the component whose
    spectrum has the highest peak in the cardiac band, and bandpass.

    Replaces rPPG-Toolbox's hand-written JADE because that implementation
    relies on ``np.matrix`` / ``np.linalg.solve`` semantics that broke in
    numpy 2.0; FastICA gives the same blind-source-separation behaviour
    without numpy-version-specific landmines.
    """
    n = rgb.shape[0]
    nyq = 0.5 * fs
    rgb_norm = np.zeros_like(rgb, dtype=np.float64)
    for c in range(3):
        d = _detrend(rgb[:, c], 100.0)
        s = np.std(d) or 1e-12
        rgb_norm[:, c] = (d - np.mean(d)) / s

    sources = _blind_source_separation(rgb_norm)

    max_pxs = np.zeros(sources.shape[1])
    for c in range(sources.shape[1]):
        FF = np.fft.fft(sources[:, c])
        Px = np.abs(FF[: n // 2]) ** 2
        if Px.sum() > 0:
            Px = Px / Px.sum()
        max_pxs[c] = Px.max()
    best = int(np.argmax(max_pxs))
    bvp_real = sources[:, best].astype(np.float64)
    b, a = signal.butter(3, [0.7 / nyq, 2.5 / nyq], "bandpass")
    try:
        return signal.filtfilt(b, a, bvp_real)
    except ValueError:
        return bvp_real


def _blind_source_separation(rgb_norm: np.ndarray) -> np.ndarray:
    """Return three source signals separated from the (N, 3) RGB matrix.

    Tries scikit-learn's FastICA (the numerically robust modern equivalent
    of JADE for our 3-channel use case); falls back to whitened PCA if
    FastICA isn't installed or fails to converge.
    """
    try:
        from sklearn.decomposition import FastICA  # type: ignore[import-not-found]
    except ImportError:
        return _pca_whitened(rgb_norm)
    try:
        ica = FastICA(n_components=3, random_state=0, max_iter=500, tol=1e-4)
        return ica.fit_transform(rgb_norm)
    except Exception:
        return _pca_whitened(rgb_norm)


def _pca_whitened(rgb_norm: np.ndarray) -> np.ndarray:
    """Whitened-PCA fallback when scikit-learn is unavailable. Returns the
    three principal components of the centred (N, 3) input as columns.
    """
    centred = rgb_norm - np.mean(rgb_norm, axis=0, keepdims=True)
    cov = (centred.T @ centred) / max(1, centred.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    eigvals = np.maximum(eigvals[order], 1e-12)
    whitening = eigvecs / np.sqrt(eigvals)
    return centred @ whitening


_ALGO_FNS: dict[str, Callable[[np.ndarray, float], np.ndarray]] = {
    "pos": _pos_bvp,
    "chrom": _chrom_bvp,
    "green": _green_bvp,
    "ica": _ica_bvp,
    "lgi": _lgi_bvp,
    "pbv": _pbv_bvp,
    "omit": _omit_bvp,
}


# ---------------------------------------------------------------------------
# PSD / HR / SNR helpers.
# ---------------------------------------------------------------------------


def _parabolic_peak_hz(freqs: np.ndarray, psd: np.ndarray) -> float:
    """3-point parabolic interpolation around the PSD peak, mirroring the
    refinement in ``article_pos_pipeline.estimate_window_bpm``.
    """
    if psd.size == 0:
        return 0.0
    k = int(np.argmax(psd))
    if k == 0 or k == psd.size - 1:
        return float(freqs[k])
    y0 = psd[k - 1]
    y1 = psd[k]
    y2 = psd[k + 1]
    denom = (y0 - 2.0 * y1 + y2)
    if abs(denom) < 1e-12:
        return float(freqs[k])
    delta = 0.5 * (y0 - y2) / denom
    df = freqs[1] - freqs[0]
    return float(freqs[k] + delta * df)


def _bvp_to_hr_psd_snr(
    bvp: np.ndarray,
    fs: float,
    lo_hz: float,
    hi_hz: float,
) -> tuple[float, np.ndarray, np.ndarray, float]:
    """Bandpass + Welch PSD + parabolic peak HR + SNR (dB).

    SNR = 10 log10( power within +/-0.1 Hz of the peak / power outside, both
    restricted to the [lo_hz, hi_hz] band ).
    """
    if bvp.size < int(2 * fs) or not np.isfinite(bvp).all():
        return 0.0, np.array([]), np.array([]), -np.inf
    try:
        bvp = _bandpass(bvp - np.mean(bvp), fs, lo_hz, hi_hz)
    except ValueError:
        return 0.0, np.array([]), np.array([]), -np.inf
    nperseg = min(bvp.size, max(int(fs * 8), 64))
    freqs, psd = signal.welch(
        bvp,
        fs=fs,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        window="hann",
        detrend="constant",
    )
    band = (freqs >= lo_hz) & (freqs <= hi_hz)
    if not np.any(band):
        return 0.0, freqs, psd, -np.inf
    band_psd = psd[band]
    band_freqs = freqs[band]
    peak_hz = _parabolic_peak_hz(band_freqs, band_psd)
    bpm = peak_hz * 60.0
    # SNR: signal power within +/-0.1 Hz of the peak vs noise (rest of band).
    sig_mask = np.abs(band_freqs - peak_hz) <= 0.1
    sig_power = float(band_psd[sig_mask].sum())
    noise_power = float(band_psd[~sig_mask].sum()) + 1e-12
    if sig_power <= 0:
        snr_db = -np.inf
    else:
        snr_db = 10.0 * math.log10(sig_power / noise_power)
    return bpm, band_freqs, band_psd, snr_db


# ---------------------------------------------------------------------------
# Consensus aggregation.
# ---------------------------------------------------------------------------


@dataclass
class AlgoEstimate:
    name: str
    bpm: float
    snr_db: float
    weight_freq: float = 0.0
    weight_corr: float = 0.0
    weight: float = 0.0
    inlier: bool = False
    error: str | None = None


@dataclass
class ConsensusResult:
    bpm: float
    q_sig: float
    verdict: str  # "accepted" | "rejected"
    rejection_reason: str | None
    inlier_std_bpm: float
    inlier_mean_snr_db: float
    consensus_inliers: int
    n_methods_succeeded: int
    fps: float
    n_frames: int
    methods: list[AlgoEstimate] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "bpm": self.bpm,
            "q_sig": self.q_sig,
            "verdict": self.verdict,
            "rejection_reason": self.rejection_reason,
            "inlier_std_bpm": self.inlier_std_bpm,
            "inlier_mean_snr_db": self.inlier_mean_snr_db,
            "consensus_inliers": self.consensus_inliers,
            "n_methods_succeeded": self.n_methods_succeeded,
            "fps": self.fps,
            "n_frames": self.n_frames,
            "methods": [asdict(m) for m in self.methods],
        }


def _ransac_consensus_hr(
    hrs: np.ndarray,
    weights: np.ndarray | None = None,
    tau_bpm: float = RANSAC_TAU_BPM,
) -> tuple[float, np.ndarray]:
    """For each candidate HR, count how many of the rest fall within
    ``+/-tau_bpm``. Return the (weighted-mean) consensus HR over the largest
    inlier set, and the boolean inlier mask.
    """
    n = hrs.size
    if n == 0:
        return 0.0, np.zeros(0, dtype=bool)
    if weights is None:
        weights = np.ones(n, dtype=np.float64)
    best_count = -1
    best_mask = np.zeros(n, dtype=bool)
    best_total_weight = -np.inf
    for i in range(n):
        mask = np.abs(hrs - hrs[i]) <= tau_bpm
        count = int(mask.sum())
        total_w = float(weights[mask].sum())
        # Tie-break by total weight, then by smaller spread.
        if count > best_count or (count == best_count and total_w > best_total_weight):
            best_count = count
            best_total_weight = total_w
            best_mask = mask
    inlier_w = weights[best_mask]
    inlier_hr = hrs[best_mask]
    if inlier_w.sum() <= 0:
        return float(np.median(inlier_hr)), best_mask
    return float(np.average(inlier_hr, weights=inlier_w)), best_mask


def _pairwise_corr_weights(psds: list[np.ndarray]) -> np.ndarray:
    """For each algo i, average Pearson correlation of psd_i with psd_j for
    j != i, clipped to [0, 1]. Algorithms with empty/zero PSD get weight 0.
    """
    n = len(psds)
    if n <= 1:
        return np.zeros(n)
    # Pad PSDs to the common length (min length wins; PSDs from different
    # algos have the same fs/nperseg by construction so this is a no-op).
    min_len = min((p.size for p in psds if p.size > 0), default=0)
    if min_len == 0:
        return np.zeros(n)
    M = np.zeros((n, min_len))
    for i, p in enumerate(psds):
        if p.size >= min_len and np.std(p[:min_len]) > 0:
            M[i] = p[:min_len]
    weights = np.zeros(n)
    for i in range(n):
        if np.std(M[i]) <= 0:
            continue
        corrs = []
        for j in range(n):
            if i == j or np.std(M[j]) <= 0:
                continue
            corr = float(np.corrcoef(M[i], M[j])[0, 1])
            if math.isfinite(corr):
                corrs.append(max(0.0, corr))
        if corrs:
            weights[i] = float(np.mean(corrs))
    return weights


def _safe_bandpass(
    bvp: np.ndarray, fs: float, min_hz: float, max_hz: float
) -> np.ndarray:
    """Bandpass-filter ``bvp`` to the cardiac band, returning the input
    unchanged on failure (e.g. trace too short for the IIR transient).
    """
    try:
        return _bandpass(bvp, fs, min_hz, max_hz)
    except (ValueError, RuntimeError):
        return bvp


def _per_patch_bvp(
    fn: Callable[[np.ndarray, float], np.ndarray],
    patch_trace: np.ndarray,
    fps: float,
    min_hz: float,
    max_hz: float,
) -> np.ndarray:
    """Run a single rPPG algorithm independently on every face-skin patch in
    ``patch_trace`` (shape ``(N, K, 3)``) and return the patch-median BVP.

    Each patch's BVP is bandpassed to the cardiac band before the median is
    taken. This is what gives the existing article-style POS pipeline its
    motion robustness: patch-local non-cardiac drift is removed first so the
    cross-patch median reinforces the (phase-coherent) cardiac peak instead
    of being washed out by per-patch motion artefacts. Without the per-patch
    bandpass the median across raw BVPs collapses the cardiac signal even
    when it is clearly visible in each individual patch.

    Patches with degenerate (zero-std) input or that raise inside ``fn`` are
    silently skipped, matching the article-style POS aggregator.
    """
    bvps: list[np.ndarray] = []
    eps = np.finfo(np.float64).eps
    for k in range(patch_trace.shape[1]):
        rgb_k = patch_trace[:, k, :].copy()
        # Replace per-patch NaNs from occluded landmarks before running the
        # algorithm; the median across patches still tolerates a few bad
        # patches but each algorithm needs finite input to produce finite BVP.
        finite = np.isfinite(rgb_k).all(axis=1)
        if finite.sum() < rgb_k.shape[0] * 0.5:
            continue
        if not finite.all():
            x = np.arange(rgb_k.shape[0])
            for c in range(3):
                col = rgb_k[:, c]
                col[~finite] = np.interp(x[~finite], x[finite], col[finite])
                rgb_k[:, c] = col
        if np.std(rgb_k) <= eps:
            continue
        try:
            bvp = fn(rgb_k, fps)
        except Exception:
            continue
        bvp = _safe_bandpass(bvp, fps, min_hz, max_hz)
        if np.any(np.isfinite(bvp)) and np.std(bvp) > eps:
            bvps.append(bvp)
    if not bvps:
        raise RuntimeError("algorithm failed for all patches")
    return np.median(np.vstack(bvps), axis=0)


def _run_algorithms(
    methods: tuple[str, ...],
    rgb: np.ndarray,
    fps: float,
    min_hz: float,
    max_hz: float,
    patch_trace: np.ndarray | None = None,
) -> tuple[list[AlgoEstimate], list[np.ndarray]]:
    """Apply each requested algorithm and collect ``AlgoEstimate``s plus
    Welch PSDs. When ``patch_trace`` is provided the algorithm is run
    independently on every patch and the BVP is the patch-median (matches
    the per-patch median aggregation that gives the article-style POS its
    motion robustness); otherwise we operate on the single mean RGB trace.
    Every BVP is bandpassed to the cardiac band before HR estimation so
    each algorithm contributes a comparable, drift-free spectrum to the
    Pearson correlation step.
    """
    estimates: list[AlgoEstimate] = []
    psds: list[np.ndarray] = []
    for name in methods:
        fn = _ALGO_FNS.get(name)
        if fn is None:
            estimates.append(AlgoEstimate(
                name=name, bpm=0.0, snr_db=-math.inf,
                error=f"unknown algorithm '{name}'",
            ))
            psds.append(np.array([]))
            continue
        try:
            if patch_trace is not None:
                bvp = _per_patch_bvp(fn, patch_trace, fps, min_hz, max_hz)
            else:
                bvp = fn(rgb, fps)
                bvp = _safe_bandpass(bvp, fps, min_hz, max_hz)
            bpm, _freqs, psd, snr_db = _bvp_to_hr_psd_snr(bvp, fps, min_hz, max_hz)
        except Exception as exc:  # pragma: no cover - defensive
            estimates.append(AlgoEstimate(
                name=name, bpm=0.0, snr_db=-math.inf,
                error=f"{type(exc).__name__}: {exc}",
            ))
            psds.append(np.array([]))
            continue
        estimates.append(AlgoEstimate(name=name, bpm=bpm, snr_db=snr_db))
        psds.append(psd)
    return estimates, psds


def consensus_bpm(
    rgb: np.ndarray,
    fps: float,
    min_hz: float = DEFAULT_MIN_HZ,
    max_hz: float = DEFAULT_MAX_HZ,
    methods: tuple[str, ...] = CONSENSUS_METHODS,
    patch_trace: np.ndarray | None = None,
) -> ConsensusResult:
    """Run all classical algorithms and combine them via the rPPG-VQA
    consensus weighting.

    When ``patch_trace`` is provided (shape ``(N, K, 3)``), each algorithm
    runs independently on every patch and its BVP is the patch-median
    across estimators. This is what gives the existing article-style POS
    aggregator its motion robustness, and it lifts the same robustness to
    every classical method in the consensus.
    """
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError(
            f"consensus_bpm: expected RGB array of shape (N, 3), got {rgb.shape}",
        )
    n = rgb.shape[0]
    duration_s = n / max(fps, 1e-6)
    if duration_s < MIN_SECONDS_FOR_CONSENSUS:
        raise RuntimeError(
            f"Consensus needs at least {MIN_SECONDS_FOR_CONSENSUS:.1f} s of "
            f"face-visible video; got {duration_s:.2f} s.",
        )

    estimates, psds = _run_algorithms(
        methods=methods,
        rgb=rgb,
        fps=fps,
        min_hz=min_hz,
        max_hz=max_hz,
        patch_trace=patch_trace,
    )

    succeeded = [e for e in estimates if e.error is None and e.bpm > 0]
    n_succeeded = len(succeeded)
    if n_succeeded == 0:
        return ConsensusResult(
            bpm=0.0, q_sig=0.0, verdict="rejected",
            rejection_reason="all algorithms failed; check video quality",
            inlier_std_bpm=0.0, inlier_mean_snr_db=-math.inf,
            consensus_inliers=0,
            n_methods_succeeded=0, fps=fps, n_frames=n, methods=estimates,
        )

    # (a) Frequency-consistency weights (Eq. 4).
    succ_indices = [i for i, e in enumerate(estimates) if e.error is None and e.bpm > 0]
    succ_hrs = np.array([estimates[i].bpm for i in succ_indices])
    consensus_hr, inlier_mask_succ = _ransac_consensus_hr(succ_hrs)
    for k, i in enumerate(succ_indices):
        delta = succ_hrs[k] - consensus_hr
        estimates[i].weight_freq = float(math.exp(-(delta ** 2) / (2 * GAUSS_SIGMA_BPM ** 2)))
        estimates[i].inlier = bool(inlier_mask_succ[k])

    # (b) Spectral-correlation weights (Eq. 8).
    corr_weights_all = _pairwise_corr_weights(psds)
    for i, e in enumerate(estimates):
        e.weight_corr = float(corr_weights_all[i])
        e.weight = float(e.weight_freq * e.weight_corr)

    # Final BPM = weighted mean over successes (corr weights also gate failures).
    weights = np.array([estimates[i].weight for i in succ_indices])
    if weights.sum() <= 0:
        # All correlations failed (degenerate PSDs). Fall back to frequency-only.
        weights = np.array([estimates[i].weight_freq for i in succ_indices])
    if weights.sum() <= 0:
        # Last-resort fallback: median of successes.
        final_bpm = float(np.median(succ_hrs))
    else:
        final_bpm = float(np.average(succ_hrs, weights=weights))

    # q_sig = mean weight across inliers.
    inlier_indices = [i for i in succ_indices if estimates[i].inlier]
    if inlier_indices:
        q_sig = float(np.mean([estimates[i].weight for i in inlier_indices]))
        inlier_hrs = np.array([estimates[i].bpm for i in inlier_indices])
        inlier_std = float(np.std(inlier_hrs)) if inlier_hrs.size > 1 else 0.0
        finite_snrs = [
            estimates[i].snr_db
            for i in inlier_indices
            if math.isfinite(estimates[i].snr_db)
        ]
        inlier_mean_snr = float(np.mean(finite_snrs)) if finite_snrs else -math.inf
    else:
        q_sig = 0.0
        inlier_std = float(np.std(succ_hrs)) if succ_hrs.size > 1 else 0.0
        inlier_mean_snr = -math.inf

    # Acceptance criteria, after the user's request: at least
    # ``MIN_INLIERS_FOR_ACCEPT`` algorithms must agree within
    # ``INLIER_STD_BPM_MAX`` BPM. ``q_sig`` and ``inlier_mean_snr_db`` are
    # surfaced for clients that want a finer-grained quality cutoff but they
    # do not gate the verdict — clients can apply their own thresholds.
    rejection_reason: str | None = None
    reasons: list[str] = []
    if len(inlier_indices) < MIN_INLIERS_FOR_ACCEPT:
        reasons.append(
            f"only {len(inlier_indices)} algorithm"
            f"{'s' if len(inlier_indices) != 1 else ''} agreed "
            f"(need >= {MIN_INLIERS_FOR_ACCEPT})",
        )
    if inlier_std > INLIER_STD_BPM_MAX:
        reasons.append(
            f"inlier spread is {inlier_std:.1f} BPM "
            f"(> {INLIER_STD_BPM_MAX})",
        )
    if reasons:
        verdict = "rejected"
        rejection_reason = (
            "; ".join(reasons)
            + " — improve lighting and stability and retry"
        )
    else:
        verdict = "accepted"

    return ConsensusResult(
        bpm=final_bpm,
        q_sig=q_sig,
        verdict=verdict,
        rejection_reason=rejection_reason,
        inlier_std_bpm=inlier_std,
        inlier_mean_snr_db=inlier_mean_snr,
        consensus_inliers=len(inlier_indices),
        n_methods_succeeded=n_succeeded,
        fps=fps,
        n_frames=n,
        methods=estimates,
    )
