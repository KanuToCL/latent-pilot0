"""Core DSP + types for the degradation library.

Label decisions baked here (docs/DECISIONS.md L1–L4):
- **L1** Every function degrades at the sample rate it is GIVEN. The Phase-2
  pipeline WILL resample clean → each model's native rate (soxr) and degrade
  there — that caller does not exist yet; these functions only guarantee correct
  operation at whatever rate they receive, and refuse cells that are undefined at
  that rate (e.g. a low-pass cutoff ≥ Nyquist — see bandlimit / grid.is_applicable).
- **L3** Degradations are NEVER renormalised after application.
- **L4** Outputs are float32; true peak is measured, not clipped. Because we
  never renormalise, a loud degradation may push the true peak above 0 dBFS —
  preserved in float, so there is no silent second clipping stage. Cells with
  peak > 0 dBFS are flagged (`measured['over_0dbfs']`); a single global, logged
  headroom scalar is applied at the CORPUS level (Phase 2/3) before encoding so
  nothing overloads the codec, without per-clip renormalisation (relative
  energies preserved).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EPS = 1e-12


@dataclass(frozen=True)
class DegradationResult:
    wav: np.ndarray  # float32, same length as input
    sr: int
    family: str
    severity_index: int  # 1..5 (0 = clean)
    param: float  # physical parameter value at this severity
    measured: dict = field(default_factory=dict)


def to_float32(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(x, dtype=np.float64))) + EPS))


def signal_level(x: np.ndarray) -> float:
    """Reference signal level = full-file RMS.

    NOTE (Phase-2 upgrade): real speech with long silences should use ITU-T P.56
    active speech level; full-file RMS understates level by ~−10·log10(activity),
    so a fixed SNR label is optimistic *during* speech by that many dB. Within-
    family monotonicity survives; the absolute-SNR bias must be quantified in the
    Phase-2 manifest and P.56 prioritised for the speech arm.
    """
    return rms(x)


def band_power(x: np.ndarray, sr: int, lo: float, hi: float) -> float:
    """Un-normalised power of x within [lo, hi) Hz (∑|rFFT|²).

    Only ever used in RATIOS of same-length signals, so the missing global 1/N,
    window, and one-sided ×2 factors cancel — *provided* the band excludes DC and
    Nyquist (those bins are not doubled in a one-sided spectrum). Callers keep
    lo > 0 and hi ≤ Nyquist with a half-open mask; assert it so a future band that
    includes DC/Nyquist can't silently bias a ratio.
    """
    assert lo > 0, "band_power bands must exclude DC (lo > 0) for the ratio to be unbiased"
    x = np.asarray(x, dtype=np.float64)
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    mask = (freqs >= lo) & (freqs < hi)
    if not mask.any():
        return 0.0
    return float(np.sum(np.abs(spec[mask]) ** 2))


def add_noise_at_snr(
    sig: np.ndarray, noise: np.ndarray, snr_db: float, sr: int | None = None, band=None
) -> np.ndarray:
    """Return sig + g·noise, with g set so the (optionally band-limited) SNR of
    the result equals `snr_db`. Full-band by default; `band=(lo,hi)` for hiss."""
    sig = np.asarray(sig, dtype=np.float64)
    noise = np.asarray(noise, dtype=np.float64)
    if band is None:
        ps, pn = float(np.mean(sig**2)), float(np.mean(noise**2))
    else:
        ps, pn = band_power(sig, sr, *band), band_power(noise, sr, *band)
    g = np.sqrt(ps / (pn + EPS) / (10.0 ** (snr_db / 10.0)))
    return sig + g * noise


def true_peak_dbtp(x: np.ndarray, sr: int, oversample: int = 4) -> float:
    """Inter-sample peak estimate in dBTP (4× oversampled).

    Diagnostic, NOT a BS.1770-4-compliant true-peak meter: resample_poly's
    Kaiser-windowed sinc differs from the standard FIR and may under-read sharp
    inter-sample peaks by ~0.1 dB. Used only to flag over-0-dBFS cells."""
    from scipy.signal import resample_poly

    up = resample_poly(np.asarray(x, dtype=np.float64), oversample, 1)
    peak = float(np.max(np.abs(up))) + EPS
    return float(20.0 * np.log10(peak))
