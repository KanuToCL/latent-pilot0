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
Wired today: `encodec24k` (variant `z`) and `wavlm` (layers `l1,l6,l12,l18,l24`).
```bash
python - <<'PY'
from pilot0.seam.registry import make_encoder
from pilot0.audio.synth import synth_clip
for name in ("encodec24k", "wavlm"):
    out = make_encoder(name).encode(synth_clip(0, sr=48000), 48000)
    for v, r in out.items():
        print(name, v, r.frames.shape, f"{r.frame_rate_hz:.1f} fps", r.meta["device"])
PY
```
Confirm per variant: `frames` is `[T, D]`, `D` matches the checkpoint's real
latent dim, `device == "cuda"`, and `frame_rate_hz` is sane (EnCodec-24k ≈ 75,
WavLM ≈ 50). **If real D ≠ REAL_SPECS latent_dim, update `REAL_SPECS` +
`configs/models.yaml` (D6) before any encoding** — `tests/test_config.py` guards
the two staying in sync.

## 2. Fake/real parity (contract, not values)
```bash
python -m pytest -q -k parity
```
Asserts fake and real return the same *contract* — same variant set, rank-2
frames, positive frame rate, matching latent dim, shared meta keys — not the same
numbers.

## 3. Wire the Phase-5 backends
`dac` and `mimi` raise `NotImplementedError` in `_ensure` and report
`available == False`. Implement each against its library (descript-audio-codec;
transformers Mimi — surface dequantized embeddings, D8), add EnCodec RVQ-depth
variants `{d1,d2,d4,d8}`, extend `WIRED_FAMILIES`, re-run steps 1–2, and record
real latent dims (D6).

## 4. Smoke on real encoders
Point `SMOKE_ENCODERS` at the bare names and:
```bash
make smoke
```
Expect `encodec24k` + `wavlm` available: True (dac/mimi False until step 3) and
shapes logged across variants.

## 5. Build the real manifest — GROUPED (mandatory for VCTK/LibriSpeech)
Multi-clip corpora share a speaker/track across many clips. Preflight MUST get a
speaker/track `group_fn`, and `build_manifest` MUST be called with
`assert_grouped=True`, or the source-disjoint split leaks speaker identity into
both train and test and **invalidates Gate 1** (§2.5, elder W1). `assert_grouped`
fails closed if every clip ended up its own group (i.e. the `group_fn` was wrong).
```python
from pilot0.corpus.preflight import preflight
from pilot0.corpus.manifest import build_manifest, write_manifest

# VCTK: filenames like p225_003.wav → speaker = "p225"; LibriSpeech: <spk>-<chap>-<utt>.
pf = preflight(RAW_DIR, OUT_DIR, group_fn=lambda p: p.stem.split("_")[0])
man = build_manifest(pf, assert_grouped=True)     # raises if grouping was forgotten
write_manifest(man, OUT_DIR / "manifest.json")
```

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
