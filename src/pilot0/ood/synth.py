"""Out-of-grid audio for the Phase-7 OOD teaser: synthetic textures that are NOT
clean-voiced-speech-plus-a-classical-degradation, standing in for generative-model
output (TTS / music-gen) at bring-up. Deterministic, distinct from `audio.synth`
(harmonic voiced) and from every degradation family, so they land off the manifold
the probes and the quality head were trained on.

Four kinds cycle so the OOD set spans several timbres a codec has no clean reference
for: inharmonic FM (metallic), granular bursts, noise-band (vocoder-like), and a
swept chirp. Plumbing only — the real OOD set is generative-model renders.
"""

from __future__ import annotations

import numpy as np

OOD_KINDS = ("fm_inharmonic", "granular", "noise_band", "chirp")


def synth_ood_clip(seed: int, sr: int = 48000, seconds: float = 2.0) -> np.ndarray:
    """One deterministic OOD texture in [-0.5, 0.5], float64. `kind` cycles by seed."""
    rng = np.random.default_rng(seed)
    kind = OOD_KINDS[seed % len(OOD_KINDS)]
    n = int(sr * seconds)
    t = np.arange(n) / sr

    if kind == "fm_inharmonic":  # metallic FM with an irrational modulation ratio
        carrier = rng.uniform(300.0, 900.0)
        ratio = rng.uniform(1.3, 2.7)  # inharmonic on purpose
        index = rng.uniform(3.0, 8.0)
        sig = np.sin(2 * np.pi * carrier * t + index * np.sin(2 * np.pi * carrier * ratio * t))
    elif kind == "granular":  # sparse enveloped grains at random times/pitches
        sig = np.zeros(n)
        for _ in range(int(rng.integers(20, 40))):
            start = int(rng.uniform(0, max(1, n - sr // 10)))
            glen = int(sr * rng.uniform(0.01, 0.06))
            gt = np.arange(glen) / sr
            grain = np.sin(2 * np.pi * rng.uniform(200, 2000) * gt) * np.hanning(glen)
            sig[start:start + glen] += grain[: max(0, min(glen, n - start))]
    elif kind == "noise_band":  # band-passed noise sweeping centre freq (vocoder-like)
        noise = rng.standard_normal(n)
        centre = 500 + 2000 * (0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.5, 2.0) * t))
        sig = noise * np.sin(2 * np.pi * centre * t)  # crude ring-mod colouring
    else:  # chirp: log sweep across the band
        f0, f1 = rng.uniform(80, 200), rng.uniform(4000, 9000)
        k = (f1 / f0) ** (1.0 / seconds)
        sig = np.sin(2 * np.pi * f0 * (k**t - 1) / np.log(k))

    peak = float(np.max(np.abs(sig))) or 1.0
    return (0.5 * sig / peak).astype(np.float64)
