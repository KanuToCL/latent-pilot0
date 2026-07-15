# GB10 Bring-Up — Message in a Bottle

**To:** the agent running on the GB10 (Grace-Blackwell, 128 GB unified, aarch64 / CUDA, DGX OS).
**From:** the agent that built this repo on an Intel Mac.
**Read this whole file before you touch anything.** Then `docs/GPU_BRINGUP.md` is your step-by-step runbook and `docs/DECISIONS.md` is the frozen rationale. This file is orientation; those two are law.

---

## 0. The one rule that overrides everything: honesty

This entire repo was built and green-tested **without a GPU**, against a **fake encoder + fake score seam** (numpy stand-ins), because no CUDA machine was available. Every number produced so far is a **plumbing check, not a scientific result.** Every persisted artifact stamps `provenance.json` `fake:true` and prints a ⚠ banner.

**You are the fake→real transition.** Your job is to make the frozen backends actually run and produce the *first real* latents and gate reports. That means:

- **Do not present anything as a result until it is real** (`fake:false` in provenance) **and powered** (enough test groups — the gates say how many).
- **The failure mode to fear** is a long unattended run that is *silently still on the fake seam* — you'd waste the run and produce artifacts that look real but aren't. Verify `meta["device"] == "cuda"` and `backend == "real"` on the very first encode before trusting anything.
- **Do not move or "tune" the pre-registered gates** (Gate 1, Gate 2). They were fixed before data contact on purpose. If a gate FAILs on real data, that is a finding, not a bug to fix away.

---

## 1. What this is

**Pilot A** of a PhD proposal ("Perceptually Grounded Codecs and Reward Models for Generative Audio"). The scientific question: **can you read classical audio-degradation attributes — degradation _type_, _severity_, and geometry — out of FROZEN neural-codec / SSL latents, with NO reference signal?** Benchmarked against full-reference metrics (ViSQOL/PESQ) and no-reference incumbents (NISQA/DNSMOS/UTMOS).

Pipeline in one line:

```
clean audio → parametric degradation grid (7 families × 5 severities)
            → frozen codec/SSL latents (EnCodec / DAC / Mimi / WavLM + log-mel floor)
            → small linear/shallow probes (type, severity) + a no-reference quality head
            → two pre-registered gates
```

---

## 2. What's on disk (the map)

`src/pilot0/`:

| package | what it does |
|---|---|
| `seam/` | The encoder abstraction. `base.py` = `LatentResult` contract. `fake.py` = numpy stand-in (Mac). **`real.py` = the four real backends — YOUR territory.** `registry.py` = `REAL_SPECS` + `make_encoder(name)`. `logmel.py`/`energy.py` = torch-free baselines. |
| `degrade/` | The 7-family degradation grid (noise/hiss/hum/clip/bandlimit/mp3/dropout). Verified on Mac; don't touch. |
| `corpus/` | Preflight (validate → −23 LUFS normalise → token-rename) + manifest with **speaker/track-disjoint splits**. |
| `encode/` | Content-addressed latent cache + resume-safe pipeline. |
| `probes/` | Type probe (macro-F1), severity ridge (Spearman SRCC), **Gate 1**, frame-dropout diagnostic. |
| `analysis/` | Full representation matrix (readability heatmap, geometry cosine, monotonicity). |
| `quality/` | No-reference quality head + **Gate 2** (needs external score tables you must build). |
| `combos/`, `ood/` | Pairwise-degradation additivity + zero-shot transfer; off-grid metric-disagreement teaser. |
| `release/` | One corpus → strict-JSON artifacts → figures F1–F6. |
| `provenance.py` | The `fake:true/false` stamp. |

Make targets (all currently exercise the **fake** seam): `setup`, `setup-gpu`, `smoke`, `test`, `test-degrade`, `manifest-demo`, `encode-demo`, `probe-demo`, `analyze`, `quality-demo`, `combos-demo`, `ood-demo`, `reproduce-figures`.

State: 149 passed / 3 skipped on the Mac fake seam. Real backends written, **never executed** — that's what you change.

---

## 3. How it works (the two things that matter most)

**The seam is a drop-in.** `real.py::RealCodecEncoder` already implements all four families against their documented APIs. Constructing one on a torch-less machine raises `RealBackendUnavailable` and downstream falls back to fake. On the GB10, torch is present, so the real path runs. Everything downstream of `LatentResult` (`frames: [T, D]`, `frame_rate_hz`, `meta`) is unchanged — you are lighting up the encoder, not rewriting the pipeline.

**The gates have guards you must respect:**
- **Gate 1** — severity SRCC from codec latents must beat a 3-band **energy/loudness control baseline** by ≥0.05 (CI-lower vs CI-upper), else "it's just loudness." Requires `MIN_TEST_GROUPS` powered, without-clean SRCC, bootstrap CIs over test *groups*, and a **held-out confirmation split** for the single selected winner.
- **Gate 2** — the quality head is scored two ways: G2a vs ViSQOL (its training target), G2b vs the NR incumbents on **human MOS held OFF the ViSQOL target** (rig-avoidance). A rank-copy rig-guard rejects a head that just parrots ViSQOL's ranking. `from_json` rejects any non-finite score cell.

---

## 4. Your environment (aarch64 — the ONLY real hurdle)

The repo is Python-version-agnostic downstream of the seam; the friction is purely torch on aarch64+CUDA.

- **Do NOT use Python 3.14.** The Mac used it because it was torch-free. torch has no 3.14 CUDA wheels. **Use 3.11 or 3.12** (the Makefile and `docs/GPU_BRINGUP.md` already assume `python3.11`).
- **Two install paths — pick one:**
  - **NGC PyTorch container** (recommended; NVIDIA ships an aarch64+CUDA build preloaded for DGX Spark / GB10). Fewest surprises. `pip install -e ".[gpu]"` inside it for the rest.
  - **Native venv:** `PYTHON=python3.11 make setup` then `make setup-gpu`, installing torch from the CUDA-aarch64 index. Verify: `python -c "import torch; print(torch.cuda.is_available())"` → `True`.
- 128 GB unified memory means you can hold all four backends + the metric stack resident at once — no OOM juggling. The models are small (tens of MB to ~1.2 GB each); the pool is comfortably oversized for this workload.

`.[gpu]` extra = `torch>=2.2, torchaudio>=2.2, transformers>=4.40, descript-audio-codec>=1.0`. Core deps (numpy/scipy/soxr/soundfile/pyloudnorm/scikit-learn) all have aarch64 wheels.

---

## 5. What to run (two jobs, in order)

### Job A — backend drop-in smoke (attended, ~30–60 min, do this FIRST)

This is not an overnight job, and you want a human at the checkpoints because drop-ins break exactly here. Follow `docs/GPU_BRINGUP.md` §0–4:

1. Install (`make setup` + `make setup-gpu`), confirm `torch.cuda.is_available()`.
2. **Shape-check all four families** (§1): each variant returns `[T, D]`, `device == "cuda"`, sane frame rate (EnCodec-24k ≈ 75, WavLM ≈ 50, DAC-44k ≈ 86, Mimi ≈ 12.5 fps).
3. **If real `D` ≠ the nominal `latent_dim` in `REAL_SPECS`** (it will differ — the Mac values are stand-ins), update `REAL_SPECS` **and** `configs/models.yaml` together (`tests/test_config.py` guards they match). This is expected, not an error.
4. **Parity test:** `pytest -q -k parity` — asserts fake and real share the same *contract*, not the same numbers.
5. Smoke: `PILOT0_SMOKE_ENCODERS=encodec24k,wavlm,dac44k,mimi make smoke` → all four `available: True`.

The Mimi `semantic`/`acoustic` split and the RVQ-depth partial decodes (`d1..d8`) are the paths most likely to need an API touch-up against the installed library versions — fix them in the relevant `_encode_*` helper, re-run parity, done. **`_assert_depth_available` fails loud if a checkpoint exposes fewer codebooks than `d8` needs** — that's a real guard, honor it (drop the deep variant rather than silently truncating).

**Job A proves fake→real works with zero science required.** If you have no corpus yet, stop here — this alone de-risks the whole transition.

### Job B — first real mini-sweep (unattended, genuinely overnight)

Needs a **real multi-speaker corpus staged** (§6 below), because Gate 1/Gate 2 depend on **speaker/track-disjoint splits**. Then `docs/GPU_BRINGUP.md` §5–8:

1. **Grouped manifest** — preflight with a `group_fn` that extracts speaker/track, `build_manifest(assert_grouped=True)`. **Eyeball clips-per-group after building**: many clips per group = correct; ≈1.0 mean = the group field is wrong and Gate 1 is invalid. This is the single easiest way to silently ruin the run.
2. **Encode** codecs **plus both baselines** (`logmel` + `energy`) into the cache — Gate 1 scores against them.
3. **Gate 1** — confirm `n_test_groups ≥ MIN_TEST_GROUPS` (in practice ≫3) *before* reading PASS/FAIL; winner re-clears on the confirmation split.
4. **Gate 2** needs external score tables (§6) — ViSQOL/PESQ + NISQA/DNSMOS/UTMOS + human MOS. Without human MOS, G2b is marked not-evaluable; **do not substitute ViSQOL for MOS.**

Sequence it: attended Job A first, then kick Job B and walk away (see the operator note on tmux at the bottom).

---

## 6. What to download

### Codec / SSL backends (auto-downloaded on first `encode`, no manual step)

These pull from HF Hub / DAC's downloader the first time a real encoder runs. None are gated. Set `HF_HOME` if you want to pin the cache location. Sizes approximate — confirm on disk.

| name | family / loader | checkpoint | native sr | ~size |
|---|---|---|---|---|
| `encodec24k` | `transformers.EncodecModel` | `facebook/encodec_24khz` | 24 kHz | ~90 MB |
| `wavlm` | `transformers.WavLMModel` + `AutoFeatureExtractor` | `microsoft/wavlm-large` | 16 kHz | ~1.2 GB |
| `dac44k` | `dac` pkg — `dac.utils.download(model_type="44khz")` | (rate-tag, not HF) | 44.1 kHz | ~hundreds of MB |
| `mimi` | `transformers.MimiModel` | `kyutai/mimi` | 24 kHz | ~hundreds of MB |

`REAL_SPECS` (in `seam/registry.py`) and `configs/models.yaml` are the authoritative list — if you ever disagree with this table, trust the code.

### Corpus (you must stage this for Job B — it is the gating input)

The pipeline **degrades clean audio itself**, so feed it **clean sources only** (never pre-degraded). It needs **many speakers/tracks** for the disjoint splits:

- **Speech (multi-speaker, clean):** VCTK (~110 speakers; filenames `p225_003.wav` → speaker `p225`) is ideal. LibriSpeech `train-clean-*` (`<spk>-<chap>-<utt>` → group by **speaker**, not chapter) or DAPS also work.
- **Music (clean stems):** MUSDB18 or MedleyDB for the music arm.

### Gate 2 external metrics (Phase 6 only — NOT needed for Job A or Gate 1)

This repo cannot compute these on CPU; build the score tables once, keyed by cell `"{source}|{family}|{severity}"` where **`{source}` is the manifest's opaque `content_token`, NOT a filename** (a filename-keyed table silently drops every row). Then `TableScores.from_json(...)`.

- **ViSQOL** — full-reference training target. C++/bazel build (github.com/google/visqol); the fiddliest, attempt it early. PESQ (pip `pesq`) on the 16 kHz arm is the fallback reference.
- **NISQA / DNSMOS / UTMOS** — no-reference, run on the degraded audio. (NISQA: gabrielmittag/NISQA weights; DNSMOS: ONNX from microsoft/DNS-Challenge; UTMOS: sarulab-speech / the `speechmos` pip is a quick route.)
- **Human MOS** on a speech subset — the G2b ground truth. There is no automated substitute.

---

## 7. How you'll know it worked, and how to report back

**Success for Job A:** all four families shape-check on `cuda`, parity passes, `REAL_SPECS`/`models.yaml` reconciled to real dims, `test_config` green.

**Success for Job B:** a **powered** Gate 1 report (real `fake:false` provenance), winner confirmed on the held-out split; Gate 2 G2a computed and G2b either computed (if MOS available) or explicitly marked not-evaluable.

**When you're done, report back:** which families ran, the reconciled latent dims (the D6 values), the Gate 1 PASS/FAIL **with `n_test_groups`**, and any `_encode_*` helper you had to touch for the installed library versions. Send that back so it lands in `docs/DECISIONS.md`. That closes the loop on this bottle.

---

## Operator note (for the human, not the agent)

To run Job B overnight and let the laptop sleep without killing it: SSH into the GB10, start a **tmux** session on the box (`tmux new -s pilot`), launch the agent/run inside it, then detach (`Ctrl-b d`) or just close the laptop — the session keeps running **on the GB10**, not on the laptop. Reattach anytime with `ssh gb10 -t tmux attach -t pilot`. Pair with **mosh** if the connection is flaky.
