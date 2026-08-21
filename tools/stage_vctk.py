# Stage VCTK 0.92 for pilot0 preflight: flat dir of *.wav, mic1 only,
# ~5 clips/speaker preferring the plan's 4-10 s window (v0 = 500 sources).
# Preflight re-validates everything downstream; this is selection + format only.
import sys
from pathlib import Path

import soundfile as sf

RAW = Path(r"D:\Cosas\04_Coding\MONOREPOS\latent-pilot0\data\raw\vctk\wav48_silence_trimmed")
OUT = Path(r"D:\Cosas\04_Coding\MONOREPOS\latent-pilot0\data\staged\speech")
PER_SPEAKER = 5
TARGET_TOTAL = 510
PREF_LO, PREF_HI, FALLBACK_LO = 4.0, 10.0, 2.5

OUT.mkdir(parents=True, exist_ok=True)

speakers = sorted(d for d in RAW.iterdir() if d.is_dir())
total = 0
kept_per_spk = []
for spk in speakers:
    if total >= TARGET_TOTAL:
        break
    infos = []
    for f in sorted(spk.glob("*_mic1.flac")):
        try:
            i = sf.info(str(f))
        except Exception:
            continue
        infos.append((f, i.frames / i.samplerate))
    preferred = [x for x in infos if PREF_LO <= x[1] <= PREF_HI]
    fallback = sorted((x for x in infos if FALLBACK_LO <= x[1] < PREF_LO),
                      key=lambda x: -x[1])
    pick = preferred[:PER_SPEAKER]
    if len(pick) < PER_SPEAKER:
        pick += fallback[:PER_SPEAKER - len(pick)]
    for f, dur in pick:
        data, sr = sf.read(str(f))
        sf.write(str(OUT / (f.stem + ".wav")), data, sr, subtype="PCM_16")
    kept_per_spk.append((spk.name, len(pick)))
    total += len(pick)

n_spk = sum(1 for _, n in kept_per_spk if n)
short = [s for s, n in kept_per_spk if 0 < n < PER_SPEAKER]
print(f"staged {total} wav clips from {n_spk} speakers -> {OUT}")
print(f"speakers with fewer than {PER_SPEAKER} clips: {len(short)}"
      + (f" ({', '.join(short[:8])}{'...' if len(short) > 8 else ''})" if short else ""))
durs = [sf.info(str(p)).frames / sf.info(str(p)).samplerate for p in sorted(OUT.glob('*.wav'))]
import statistics
print(f"durations: min {min(durs):.1f}s  median {statistics.median(durs):.1f}s  max {max(durs):.1f}s")
sys.exit(0 if total >= 400 else 1)
