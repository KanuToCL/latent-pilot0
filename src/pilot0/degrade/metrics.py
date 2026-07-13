"""Measurement of the *achieved* effect of a degradation, for the measured-vs-
target acceptance table and the analytic unit tests. Measures the truth, never
the requested parameter.

Honesty note: for the additive families (noise/hiss/hum) `measure_snr` recomputes
the same power ratio `add_noise_at_snr` used to set the gain, so measured==target
holds by construction — this validates the algebra + float32 storage, NOT that
the intended *physical* SNR is achieved. The independent checks live in the tests
(spectral slope of the injected noise, actual clipping, stopband attenuation)."""

from __future__ import annotations

import numpy as np

from .base import EPS, band_power, rms, true_peak_dbtp


def measure_snr(clean, degraded, sr=None, band=None) -> float:
    clean = np.asarray(clean, dtype=np.float64)
    noise = np.asarray(degraded, dtype=np.float64) - clean
    if band is None:
        ps, pn = float(np.mean(clean**2)), float(np.mean(noise**2))
    else:
        ps, pn = band_power(clean, sr, *band), band_power(noise, sr, *band)
    return float(10.0 * np.log10(ps / (pn + EPS) + EPS))


def measure_clip_fraction(degraded) -> float:
    x = np.abs(np.asarray(degraded, dtype=np.float64))
    thr = float(np.max(x))
    if thr <= 0:
        return 0.0
    return float(np.mean(x >= thr * (1 - 1e-4)))


def measure_dropout_fraction(degraded) -> float:
    return float(np.mean(np.abs(np.asarray(degraded, dtype=np.float64)) < 1e-6))


def measure_stopband_atten_db(clean, degraded, sr, cutoff_hz) -> float:
    lo = min(cutoff_hz * 1.15, sr / 2.0 * 0.99)
    hi = sr / 2.0
    pc = band_power(clean, sr, lo, hi)
    pd = band_power(degraded, sr, lo, hi)
    return float(10.0 * np.log10((pd + EPS) / (pc + EPS)))


def measure_top_hz(x, sr, frac: float = 0.99) -> float:
    """Frequency below which `frac` of the energy sits — surfaces a hidden
    low-pass (e.g. LAME's internal downsample at low bitrate)."""
    power = np.abs(np.fft.rfft(np.asarray(x, dtype=np.float64))) ** 2
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    cum = np.cumsum(power)
    if cum[-1] <= 0:
        return 0.0
    idx = int(np.searchsorted(cum, frac * cum[-1]))
    return float(freqs[min(idx, len(freqs) - 1)])


def measure(family: str, clean, degraded, sr: int, param: float) -> dict:
    clean = np.asarray(clean, dtype=np.float64)
    degraded = np.asarray(degraded, dtype=np.float64)
    m: dict = {}
    if family == "noise":
        m["snr_db"] = measure_snr(clean, degraded)
    elif family == "hiss":
        m["hf_snr_db"] = measure_snr(clean, degraded, sr=sr, band=(4000.0, sr / 2.0))
    elif family == "hum":
        m["hts_db"] = float(20.0 * np.log10(rms(degraded - clean) / (rms(clean) + EPS) + EPS))
    elif family == "clip":
        m["pct_clipped"] = 100.0 * measure_clip_fraction(degraded)
    elif family == "bandlimit":
        m["stopband_atten_db"] = measure_stopband_atten_db(clean, degraded, sr, param)
    elif family == "mp3":
        m["len_match"] = int(len(degraded) == len(clean))
        m["err_snr_db"] = measure_snr(clean, degraded)
        # Surfaces LAME's hidden internal low-pass at low bitrate (mp3↔bandlimit
        # confound): 24/16 kbps at 48 kHz collapse to the same top frequency.
        m["mp3_top_hz"] = measure_top_hz(degraded, sr)
    elif family == "dropout":
        m["pct_lost"] = 100.0 * measure_dropout_fraction(degraded)
    tp = true_peak_dbtp(degraded, sr)
    m["true_peak_dbtp"] = tp
    m["over_0dbfs"] = int(tp > 0.0)  # flag cells for the corpus-level headroom pass (L4)
    return m
