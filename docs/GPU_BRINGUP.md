# GPU-BOX BRING-UP RUNBOOK

The box (RTX-class GPU) is packed until Spain. Everything else is built and
tested on the Mac against the fake seam. This runbook makes the box turnkey: the
real backends in `seam/real.py` are written but **unverified** until each step
here passes. Do these in order; do not move the Phase-1 pre-registered gates.

## 0. Install
```bash
PYTHON=python3.11 make setup       # fresh venv on the box (core deps)
make setup-gpu                     # torch / torchaudio / transformers / dac
python -c "import torch; print(torch.cuda.is_available())"   # expect True
```

## 1. Real backends load & shape-check (all variants)
All four families are wired: `encodec24k` (`z` + RVQ depths `d1,d2,d4,d8`),
`wavlm` (layers `l1,l6,l12,l18,l24`), `dac44k` (`z` + `d1,d2,d4,d8`), `mimi`
(`semantic`, `acoustic`). Their encode paths are BRINGUP — written against the
documented APIs, first executed HERE.
```bash
python - <<'PY'
from pilot0.seam.registry import make_encoder
from pilot0.audio.synth import synth_clip
for name in ("encodec24k", "wavlm", "dac44k", "mimi"):
    out = make_encoder(name).encode(synth_clip(0, sr=48000), 48000)
    for v, r in out.items():
        print(name, v, r.frames.shape, f"{r.frame_rate_hz:.1f} fps", r.meta["device"])
PY
```
Confirm per variant: `frames` is `[T, D]`, `D` matches the checkpoint's real
latent dim, RVQ depths `d{k}` share their model's `z` dim, `device == "cuda"`, and
`frame_rate_hz` is sane (EnCodec-24k ≈ 75, WavLM ≈ 50, DAC-44k ≈ 86, Mimi ≈ 12.5).
**If real D ≠ REAL_SPECS latent_dim, update `REAL_SPECS` + `configs/models.yaml`
(D6) before any encoding** — `tests/test_config.py` guards the two staying in sync.
The Mimi semantic/acoustic split and the depth partial-decodes are the paths most
likely to need an API touch-up; fix them here, re-run parity, then encode.

## 2. Fake/real parity (contract, not values)
```bash
python -m pytest -q -k parity
```
Asserts fake and real return the same *contract* — same variant set, rank-2
frames, positive frame rate, matching latent dim, shared meta keys — not the same
numbers.

## 3. (Phase-5 backends are already wired — verify, don't implement)
`dac`, `mimi`, and the EnCodec/DAC RVQ-depth variants are implemented in `real.py`
(BRINGUP) and in `WIRED_FAMILIES`. Step 1 is where they first run: confirm the DAC
`from_codes` depth decode and the Mimi split-RVQ `semantic`/`acoustic` decoders
return `[T, D]`, then record real latent dims (D6). If an API signature has drifted
in the installed library version, fix it in the relevant `_encode_*` helper — the
`LatentResult` contract and everything downstream are unchanged.

## 4. Smoke on real encoders
Point `SMOKE_ENCODERS` at the bare names via the env override (no source edit) and:
```bash
PILOT0_SMOKE_ENCODERS=encodec24k,wavlm,dac44k,mimi make smoke
```
Expect all four available: True (each family's backend lib installed) and shapes
logged across variants.

**Frame-level dropout diagnostic (elder W3).** The frame-level dropout probe assumes
a zeroed audio burst maps to a LOW-NORM latent frame. That holds for EnCodec/DAC but
a per-frame-normalised SSL rep (WavLM) can flatten frame norm, silently yielding no
dropout signal. Before trusting `frame_dropout`'s frame-vs-pooled gap for a backend,
confirm the per-frame L2 norm actually drops on a dropout-degraded clip:
```python
import numpy as np
from pilot0.seam.registry import make_encoder
from pilot0.degrade.grid import apply_degradation   # dropout, severity index 5
enc = make_encoder("wavlm")
clean = enc.encode(wav, sr)["l12"].frames
dropped = enc.encode(apply_degradation(wav, sr, "dropout", 5).wav, sr)["l12"].frames
print(np.median(np.linalg.norm(dropped, axis=1)) / np.median(np.linalg.norm(clean, axis=1)))
# ≪ 1 means the signature survives; ≈ 1 means the probe is blind for this backend.
```

## 5. Build the real manifest — GROUPED (mandatory for VCTK/LibriSpeech)
Multi-clip corpora share a speaker/track across many clips. Preflight MUST get a
speaker/track `group_fn`, and `build_manifest` MUST be called with
`assert_grouped=True`, or the source-disjoint split leaks speaker identity into
both train and test and **invalidates Gate 1** (§2.5, elder W1). `assert_grouped`
fails closed if every clip ended up its own group (i.e. the `group_fn` was forgotten),
but it CANNOT catch a `group_fn` that buckets by the WRONG field (e.g. chapter instead
of speaker) — that still leaks. **Eyeball the clips-per-group distribution** after
building: a correct speaker grouping shows many clips per group; a near-1.0 mean or a
count matching the file count means the field is wrong.
```python
from pilot0.corpus.preflight import preflight
from pilot0.corpus.manifest import build_manifest, write_manifest

# VCTK: filenames like p225_003.wav → speaker = "p225"; LibriSpeech: <spk>-<chap>-<utt>.
pf = preflight(RAW_DIR, OUT_DIR, group_fn=lambda p: p.stem.split("_")[0])
man = build_manifest(pf, assert_grouped=True)     # raises if grouping was forgotten
write_manifest(man, OUT_DIR / "manifest.json")
```
**SNR-label caveat (physics):** additive-family SNR labels use full-file RMS, not
P.56 active-speech level, so absolute SNR reads ~`10·log10(activity)` dB optimistic on
speech (`degrade/base.py`). Within-family severity is monotone and unaffected — the
gates key on that — but quantify the absolute bias here if any absolute-SNR claim is made.

## 6. Encode the real corpus — codecs AND both baselines
Gate 1 needs the log-mel floor and the energy control in the cache alongside the
codecs (`run_gate1` reads `logmel` + `energy`), else the floor/energy comparison
has nothing to score against.
```python
from pilot0.encode.pipeline import encode_corpus
encoders = ["logmel", "energy", "encodec24k", "wavlm"]   # + dac44k, mimi after step 3
encode_corpus(pf.norm_dir, man, encoders, CACHE_DIR)      # resume-safe; real codecs here
```

## 7. Then, and only then — Gate 1
Run `probes.run.run_gate1(man, candidates, CACHE_DIR)` on the pre-registered
candidate list. Everything upstream (degradation grid, preflight, split logic) and
the probe/analysis code is already Mac-verified; the box only runs encode + the
gate. Confirm the report is **powered** (`n_test_groups ≥ MIN_TEST_GROUPS`, and in
practice ≫3) before reading any PASS/FAIL — an underpowered gate is not a result.
Per §2.5, the single selected winner must then independently re-clear on the
held-out confirmation split before Gate 1 is declared passed.

## 8. Phase 5 analysis + Phase 6 Gate 2 (reference metrics)
`make analyze` regenerates the readability heatmap, cosine matrix, and monotonicity
curves over the full matrix — Mac-verified, box just needs the real cache. For the
frame-level dropout probe, first run the §4 frame-norm-drop diagnostic per backend.

Gate 2 needs external score TABLES this repo cannot compute on the Mac. Build them
once, keyed by cell `"{source}|{family}|{severity}"`, and load with
`quality.scores.TableScores.from_json`. **`{source}` is the manifest row's `source`
token** (the opaque `content_token` preflight assigns, `manifest.py`), NOT the original
or degraded-wav filename — a filename-keyed table returns `None` for every cell, drops
every row, and fits the head on an empty design. `from_json` rejects any non-finite
(NaN/Inf) score cell (it would blank the §8 rank rig-guard):
- **ViSQOL** (full-reference training target; C++/bazel — §7 build risk, attempt
  early). PESQ on the 16 kHz arm is the fallback reference target.
- **NISQA / DNSMOS / UTMOS** — run on the degraded audio (no-reference).
- **Human MOS** on a speech subset — the G2b ground truth. WITHOUT it, `run_gate2`
  reports G2a only and marks G2b not-evaluable; do not substitute ViSQOL for MOS.
```python
from pilot0.quality.run import run_gate2
from pilot0.quality.scores import TableScores
report = run_gate2(man, candidates, CACHE_DIR, TableScores.from_json(SCORES_JSON))
```
Confirm powered before reading PASS/FAIL, same as Gate 1.
