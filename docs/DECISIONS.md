# DECISIONS

Running log of choices that would otherwise be invisible in the code. Newest first.

## Phase 4 — probes & Gate 1, elder review applied (2026-07-13)

New `probes/` package: linear type + severity probes on the cached latents, scored
against two baselines under the pre-registered §8 gate. Physics + adversarial +
cartographer elders; APPROVE-WITH-FIXES (one SHOWSTOPPER), all findings landed.

- **Energy control baseline (finding B1 — the SHOWSTOPPER).** Because we never
  renormalise (L3) and the L4 scalar preserves relative energies, additive
  degradations raise RMS and subtractive ones lower it with severity, so a pure
  loudness meter tracks severity. `seam/energy.py` (`EnergyEncoder`: per-frame
  [log total, log LF, log HF] energy) is now a first-class baseline, and G1b
  requires the codec's severity SRCC to **beat the energy control by ≥ 0.05 per
  family** — otherwise "reads severity" is just "reads level". The demo confirms
  it bites: the fake codec passes 0/7 because it cannot beat energy.
- **Bootstrap CIs, CI-lower gating, underpowered guard (B2 / M3 / W2 / S2).**
  Every metric carries a 95% CI resampled over TEST GROUPS (`metrics.bootstrap_over_groups`,
  1000×, §2.5). Gate 1 keys on the CI **lower bound**, so a lucky point estimate
  can't pass. A split with < `MIN_TEST_GROUPS` (=3) test sources is UNDERPOWERED and
  cannot pass at all. This is the conservative core of the winner's-curse fix.
- **Fair floor comparison on common cells (M1).** Native rates differ (floor/energy
  16 kHz, EnCodec 24 kHz, DAC 44.1 kHz), so band-limit cells renderable per rate
  differ. `run.run_gate1` scores every representation on the **intersection** of
  renderable cells (`ProbeData.select`), so the floor margin is apples-to-apples.
- **Clean anchor reported both ways (M2).** Severity trains with clean as the
  severity-0 anchor but reports SRCC **with and without** clean; Gate 1 keys on the
  **without-clean** ordinal test (order severities 1..5 among themselves), since the
  clean jump is trivially separable and inflates/couples the families.
- **Δ-augmented floor (M4).** §2.3 pre-registers "log-mel (+Δ)"; `LogMelEncoder`
  now emits static + Δ + ΔΔ (192-d). A stronger floor makes the ≥0.05 margin
  conservative (a weak floor would bias the gate toward GO).
- **Group-disjoint guard (W1).** `build_manifest(..., assert_grouped=True)` fails
  closed when every source is its own group (operator forgot a speaker/track
  `group_fn` on a multi-clip corpus → speaker leakage). `GPU_BRINGUP.md` §5–6 is the
  operational wiring: it builds the real manifest with the speaker `group_fn` +
  `assert_grouped=True` and encodes the `logmel`/`energy` baselines alongside the
  codecs, so the guard actually runs on VCTK/LibriSpeech.
- **Conservative gate margins (N2).** Both the floor margin (type) and the energy
  control (severity) compare the codec's CI **lower** bound against the baseline's
  CI **upper** bound + 0.05, so a baseline that lands luckily low on a split cannot
  hand the codec a spurious pass.
- **Type probe excludes clean (S1, honesty).** The type result reads "GIVEN a
  degradation is present, its type is linearly decodable" — not clean-vs-degraded
  detection. The floor and energy baselines are excluded symmetrically, so G1c is fair.
- Determinism pinned (`random_state=0` on the estimators). macro-F1 passes explicit
  `labels` so an absent class can't silently drop from the mean.

**Pre-registered as bring-up PROCEDURE, not code (needs the real multi-speaker
corpus to be meaningful, so it cannot be exercised on the Mac synth corpus):** the
full winner's-curse protocol beyond CI-lower gating — a frozen candidate list, 3
probe seeds × 5-fold CV inside train, and a held-out **confirmation** split the
single selected winner must independently re-clear (§2.5). The held-out-severity
interpolation SRCC (train {1,2,4,5}, test {3}) is likewise reserved as a Phase-5
severity diagnostic. These are documented here so they are frozen before data
contact even though the estimator lands at bring-up.

## Phase 3 — encode & cache, elder review applied (2026-07-13)

New `encode/` package: render (L1) → headroom (L4) → `Encoder` seam → resume-safe
content-addressed cache. Adversarial + cartographer elders; APPROVE-WITH-FIXES,
all findings landed:

- **Render order (L1).** `render_cell` resamples the −23 LUFS master to the
  model's native rate (soxr, anti-aliased) BEFORE degrading, so a label like
  `bandlimit 6 kHz` is defined at the rate the codec ingests. `clean` = resampled
  master, untouched.
- **Corpus headroom (L4).** One global attenuation scalar **per native rate** =
  `min(1, ceiling / max_true_peak)` over all renderable cells, applied uniformly
  before encode. Attenuate-only (≤ 1); every cell keeps the same scale, so the
  severity↔loudness confound is not re-introduced. Shared across same-rate
  encoders (encodec24k & mimi at 24 kHz) — correct, since the render depends only
  on the rate.
- **Cache key = `cache_version / name / variant / cell_id`.** `cell_id` hashes
  (source, family, severity, native_sr); the key is locatable from the manifest
  without rendering, so resume is a file-existence check.
- **`cache_version` folds every byte-affecting input** (elder findings B, C):
  encode-pipeline tag + `grid_signature` (levels) + `degrade_semantics_version`
  (source hash of the family/DSP code — auto-invalidates on any algorithm/default
  change, no human bump) + headroom ceiling + `corpus_signature`. The corpus
  signature is essential: the L4 scalar is corpus-global, so adding/removing a
  source rescales every latent and MUST invalidate — the cache is valid only for
  the exact (code, grid, semantics, ceiling, source-set) tuple.
- **Atomic + durable writes** (findings A, E, F). Latents and the per-rate
  headroom JSON both publish via `mkstemp` → `fsync` → `os.replace`, so neither a
  process kill nor power loss can leave a partial file that `is_cached` accepts.
  Distinct temp names (mkstemp, not pid) survive NFS/container pid clashes.
- **Fail-closed on non-finite** (finding D). `measure_headroom` and `save_latent`
  refuse a NaN/Inf render or latent rather than silently caching it into the
  Phase-4 dataset — cheap insurance for the BLIND real-codec bring-up.
- **Sidecar meta** carries source/family/severity/param, the degradation
  `measured` dict (achieved SNR / over_0dbfs / mp3_top_hz), headroom scalar,
  degraded_sha, and cache_version — enough for Phase 4 to load latents without the
  manifest, and to run an external integrity audit against a re-render.
- Backend-agnostic throughout: `fake-*` on the Mac, bare names on the box, via the
  `Encoder` seam — no torch under `encode/`. `make encode-demo` proves the loop
  (1230 latents on the synth corpus, resume re-encodes 0).

## Phase 2 — corpus, elder review applied (2026-07-13)

Adversarial + cartographer elders; all findings fixed:

- **Splits are speaker/track-disjoint (§2.5), not per-clip.** Preflight attaches a
  `group_id` (speaker/track) and `arm` to each source via caller `group_fn`/`arm_fn`
  (defaults: per-file stem, single arm). `assign_splits` orders each arm's groups by
  a stable hash and assigns the lowest `⌊test_frac·N⌋` (≥1, ≤N−1) to test — so a
  group never crosses splits, the test fraction is exact, and both splits are
  non-empty for N≥2 (replaces the old per-clip hash-bucket that could come out
  all-train and leaked speaker identity).
- **Preflight fail-closed validations:** duration, non-finite, DC offset, already-
  clipped, sample-rate allowlist (`expected_sr`), loudness-undefined, post-normalise
  over-gain (`true_peak_dbtp` > ceiling), content dedup (logs the first file).
- **Reproducibility pinned in the manifest:** `test_frac`, `split_algo` tag,
  `grid_signature` (hash of the family/level ladder), `target_lufs`, plus tool
  versions (`provenance()`); content tokens use little-endian float32 for
  cross-platform stability.
- **`renderable_rows(manifest, sr)`** is the Phase-3 render entry point that drops
  band-limit cells ≥ Nyquist per model rate (L1) — no longer dead code.
- Tokens stay opaque and content-derived; the original filename is kept only in the
  preflight report, never in manifest rows.


## Phase 0 — elder review applied (2026-07-13)

Integration + Architecture elders approved-with-concerns; fixes landed:

**D7 — Variant axis is first-class.** `encode()` returns
`dict[variant -> LatentResult]` from a single forward (§2.3 sweeps RVQ depth /
WavLM layer / Mimi stream). `variant` is part of a latent's identity; the Phase-3
cache key is `content_hash x name x variant x code_version`. Fake implements all
declared variants; real implements the wired ones (EnCodec `z`, WavLM layers).
EnCodec RVQ depths, DAC and Mimi are wired at bring-up / Phase 5.

**D8 — Representations are surfaced as continuous float embeddings.** `[T, D]`
float + mean/std/max pooling is valid because RVQ/Mimi outputs are dequantized to
embeddings, not raw integer code indices. If raw tokens are ever needed, add a
`kind: "continuous"|"tokens"` tag (deferred).

**D9 — `frame_rate_hz` is required and measured** (`n_frames / duration`), so the
frame-level dropout probe can align bursts to frame indices. Real values are
recorded at bring-up.

**D10 — Pooling moved downstream.** `LatentResult` holds `frames` +
`frame_rate_hz` + `meta`; `pool_mean_std` / `pool_max` are standalone. Keeps both
the frame-level probe and the pooled probes possible; no pooling baked in.

**D11 — WavLM uses the model's own feature-extractor normalization**
(`do_normalize=True`); skipping it silently yields wrong-scale latents.

**D12 — Real models run on CUDA if available** (`.to(device)` on model + inputs);
availability reflects *encodability* (wired families only), never lies.

**Deps** — `soxr`/`soundfile`/`scipy` moved to core (Mac Phase-1 needs an
anti-aliased resampler); confirmed importable on Python 3.14. `[gpu]` keeps the
torch stack. `configs/models.yaml` mirrors `REAL_SPECS` and is drift-guarded by
`tests/test_config.py`.

## Phase 0 — environment & seam

**D0 — Two-machine, plug-and-play mandate.** The GPU box is disassembled and
packed until the move to Spain (~1 month). Every CPU-runnable stage is built and
**verified on the Mac** against the fake encoder seam; the real codec backends
are written now but **cannot run here** and are validated only at bring-up
(`GPU_BRINGUP.md`). Goal: drop the box in, run one runbook, no debugging.

**D1 — Python.** Mac dev uses Homebrew Python 3.14 (torch-free by design); the
GPU box uses its own 3.11 + `.[gpu]`. `requires-python = ">=3.11"` covers both.

**D2 — Encoder seam.** Pipeline written against `Encoder` + `LatentResult` only
(`seam/base.py`); no downstream code imports torch.

**D3 — FakeEncoder = seeded random projection of a numpy log-mag STFT.**
Deterministic, content-dependent stand-in. Carries real signal but makes **no
perceptual claim** and must never appear in a result table.

**D4 — Fake resampling is linear `np.interp` (not anti-aliased).** Fine for the
fake backend's shape/plumbing role. The real backend and the Phase-1 degradation
grid use **soxr** (anti-aliased) — the correctness point the physics elder
flagged: degrade at the rate each model consumes (see pending L1).

**D5 — Synthetic clips for the smoke path.** Real corpora
(VCTK / LibriSpeech / MUSDB / FSD50K) arrive in Phase 2 with the manifest.

**D6 — Nominal latent_dim** (`encodec24k=128, wavlm=1024, dac=1024, mimi=512`) —
shape stand-ins reconciled against the true checkpoints at bring-up.

## Label decisions (Phase 1 — pre-registered, do not move after data contact)

Frozen with the degradation grid. Status reflects the code as built + the
physics/cartographer review (2026-07-13).

- **L1** Degrade **at the rate each model consumes**. *Implemented:* every family
  function is sr-parameterised and refuses cells undefined at that rate (band-limit
  cutoff ≥ Nyquist → `is_applicable` False / `apply_degradation` raises). The
  caller now exists: `encode/render.py` resamples the master → each model's native
  rate (soxr) and degrades there, driven by `renderable_rows(manifest, sr)` so
  only defined cells are rendered (Phase 3).
- **L2** Clipping by **target %-samples-clipped** `{0.1, 0.5, 1, 3, 8}%` via the
  `(1−p)` quantile of |x| (`clip.py`). *(monotone, always clips.)* Supersedes the
  original §2.2 "gain over full-scale {3,6,9,12,15} dB".
- **L3** **Never renormalise** after degradation (verified: no family peak-scales
  its output). The **energy/LUFS baseline column** + energy-regressed severity is
  a Phase-4 analysis requirement, not a degradation-library concern.
- **L4** Degraded audio is **float32**; the true peak is **measured, not clipped**.
  Because we never renormalise, a loud cell (e.g. noise/hum sev 5) can exceed
  0 dBFS — preserved in float (float32 write/read is lossless, so no silent second
  clip). Such cells are flagged `measured['over_0dbfs']`. A single **global, logged
  headroom scalar** is applied at the **corpus level (Phase 2/3)** before encoding
  so nothing overloads the codec (esp. EnCodec, which does not self-normalise),
  without per-clip renormalisation. *(Replaces the earlier, contradictory
  "true-peak ≤ −1 dBTP assert" wording, which was never implementable under L3.)*

**Known confound (documented, not a bug): MP3 ↔ band-limit.** At 48 kHz, LAME
drops below ~32 kbps to MPEG-2/2.5 with internal downsampling, so 24 & 16 kbps
collapse to the same aggressive low-pass — the mp3 *type* label carries band-limit
structure at the bottom of the ladder. `measured['mp3_top_hz']` surfaces it; kbps
remains a valid *ordinal* severity. Expect some mp3↔bandlimit confusion in the
type-confusion matrix; degrading at the model native rate (L1) keeps LAME in a
consistent mode per arm.

**Validation honesty.** The measured-vs-target audit for additive families is
algebra + storage self-consistency (measured recomputes the ratio that set the
gain), not proof the physical SNR is achieved. Independent checks (injected-noise
spectral slope, real clipping, stopband attenuation) live in `tests/test_degrade.py`.
