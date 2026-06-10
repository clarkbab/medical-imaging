from dicomset.typing import *
from scipy.ndimage import gaussian_filter1d
import numpy as np

def create_amsterdam_shroud(
    images: BatchLabelImage2D,
    diaphragm_window: int = 20,
    log_thresh: float | None = None,
    grad_thresh: float | None = None,
    max_shift: int = 10,
    smooth_sigma: float | None = 2.0,
    ) -> Tuple[Image2D, np.ndarray]:
    """
    Create the Amsterdam Shroud and extract a breathing signal from a series
    of 2D kV images acquired during a radiotherapy arc.

    Implements the method of Zijp, Sonke & van Herk (NKI Amsterdam):
      1. Take log of each image (pixel values proportional to radiological thickness).
      2. Apply a CC (SI) derivative filter to enhance diaphragm transitions.
      3. Optionally threshold the log image (patient vs. air) and the CC gradient
         (isolate diaphragm-like transitions), and combine as a mask.
      4. Project each enhanced image onto the CC (SI) axis by summing along LR,
         producing a 1D profile per frame.
      5. Stack 1D profiles into the 'Amsterdam Shroud' (n_frames x n_si).
      6. Compute the temporal derivative of the shroud, project onto the CC axis,
         and find the SI position with the largest temporal variation (diaphragm).
      7. Extract a window around this SI position and align consecutive columns
         (frames) by minimising RMS pixel-value differences. The cumulative
         integer shifts are the breathing signal.

    Args:
        images: Array of shape (n_frames, n_lr, n_si). Should be raw or
            minimally processed transmission images (not intensity-inverted).
        diaphragm_window: Number of SI pixels either side of the detected
            diaphragm position to use for column alignment.
        log_thresh: Threshold applied to the log image to separate patient from
            air. Pixels below this threshold are masked out. If None, no
            patient/air masking is applied.
        grad_thresh: Threshold applied to the absolute CC gradient to isolate
            diaphragm-like transitions. Pixels below this threshold are masked
            out. If None, no gradient masking is applied.
        max_shift: Maximum pixel shift (in SI) to search when aligning
            consecutive columns.
        smooth_sigma: Sigma (in SI pixels) of the Gaussian derivative filter
            applied along the SI axis. Combines smoothing and differentiation
            in one step (order=1 Gaussian), which is the optimal edge detector
            in Gaussian noise. Set to None to fall back to np.gradient.

    Returns:
        shroud: 2D shroud image of shape (n_frames, n_si).
        signal: 1D breathing signal of shape (n_frames,) — cumulative SI pixel
            shifts needed to align each frame to the previous one.
    """
    n_frames, n_lr, n_si = images.shape

    # Step 1-4: Build shroud columns.
    columns = []
    for i in range(n_frames):
        img = images[i].astype(np.float32)          # (n_lr, n_si)

        # Take log of image; pixel values become proportional to radiological thickness.
        log_img = np.log(np.clip(img, 1e-6, None))  # (n_lr, n_si)

        # CC (SI) derivative via Gaussian derivative filter — combines smoothing
        # and differentiation in one step (optimal edge detection in noisy images).
        if smooth_sigma is not None:
            cc_grad = gaussian_filter1d(log_img, sigma=smooth_sigma, axis=1, order=1)
        else:
            cc_grad = np.gradient(log_img, axis=1)   # (n_lr, n_si)

        # Build mask: patient region AND diaphragm-like gradients.
        mask = np.ones((n_lr, n_si), dtype=bool)
        if log_thresh is not None:
            mask &= log_img > log_thresh
        if grad_thresh is not None:
            mask &= np.abs(cc_grad) > grad_thresh

        # Enhanced image: absolute CC gradient, masked.
        enhanced = np.abs(cc_grad) * mask            # (n_lr, n_si)

        # Project onto CC (SI) axis by summing along LR.
        col = enhanced.sum(axis=0)                   # (n_si,)
        columns.append(col)

    # Step 5: Stack into shroud — (n_frames, n_si).
    shroud = np.stack(columns, axis=0)

    # Step 6: Detect diaphragm region via temporal derivative of the shroud.
    # Temporal derivative: difference between successive frames.
    temporal_deriv = np.abs(np.diff(shroud, axis=0))  # (n_frames-1, n_si)

    # Project onto CC axis and find the SI position with maximum temporal variation.
    cc_profile = temporal_deriv.sum(axis=0)            # (n_si,)
    best_si = int(np.argmax(cc_profile))

    # Step 7: Extract window around the diaphragm position.
    si_lo = max(0, best_si - diaphragm_window)
    si_hi = min(n_si, best_si + diaphragm_window + 1)
    shroud_window = shroud[:, si_lo:si_hi]             # (n_frames, window_size)

    # Align consecutive columns (frames) by minimising RMS pixel-value differences.
    # The integer shift giving the minimum RMS is the incremental breathing displacement.
    shifts = [0]
    for i in range(n_frames - 1):
        col_a = shroud_window[i]
        col_b = shroud_window[i + 1]
        best_shift = 0
        best_rms = np.inf
        for s in range(-max_shift, max_shift + 1):
            if s >= 0:
                a = col_a[s:]
                b = col_b[:len(a)]
            else:
                b = col_b[-s:]
                a = col_a[:len(b)]
            if len(a) == 0:
                continue
            rms = np.sqrt(np.mean((a - b) ** 2))
            if rms < best_rms:
                best_rms = rms
                best_shift = s
        shifts.append(best_shift)

    # Cumulative sum of incremental shifts = breathing signal (SI pixel position).
    signal = np.cumsum(shifts).astype(np.float32)

    return shroud, signal


def shroud_to_signal(
    shroud: np.ndarray,
    bandwidth: int = 1,
) -> np.ndarray:
    """
    Extract a breathing signal from a 2D shroud image using FFT.

    Finds the SI column (temporal trace) with the strongest non-DC periodic
    content, identifies its dominant frequency, bandpass filters a narrow
    window around that frequency, and returns the IFFT as a 1D breathing
    signal.

    Args:
        shroud: 2D shroud image of shape (n_frames, n_si).
        bandwidth: Number of frequency bins either side of the dominant
            peak to retain in the bandpass filter. Default 1 keeps the
            peak bin and its immediate neighbours.

    Returns:
        1D breathing signal of shape (n_frames,), real-valued.
    """
    n_frames, n_si = shroud.shape

    # FFT along temporal axis (axis=0); rfft exploits real-valued input.
    F = np.fft.rfft(shroud, axis=0)   # (n_freqs, n_si)

    # Compute magnitude spectrum and zero out DC so we focus on oscillations.
    mag = np.abs(F).copy()
    mag[0, :] = 0.0

    # Find the SI column with the strongest periodic content (highest peak magnitude).
    best_si = int(np.argmax(mag.max(axis=0)))

    # Dominant frequency bin for that column.
    peak_freq = int(np.argmax(mag[:, best_si]))

    # Bandpass filter: retain only [peak_freq - bandwidth, peak_freq + bandwidth].
    filt = np.zeros_like(F)
    lo = max(1, peak_freq - bandwidth)
    hi = min(F.shape[0] - 1, peak_freq + bandwidth)
    filt[lo:hi + 1, best_si] = F[lo:hi + 1, best_si]

    # Inverse FFT → real-valued 1D breathing signal.
    signal = np.fft.irfft(filt[:, best_si], n=n_frames).astype(np.float32)

    return signal
