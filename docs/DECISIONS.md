# DECISIONS

Running log of choices that would otherwise be invisible in the code. Newest first.

## Job B corpus + post-audit hardening (2026-08-21, evening)

Independent Opus audit of the bring-up caught real defects before the overnight
run; all applied:

- **Corpus (now reproducible via `tools/stage_vctk.py` + `tools/build_real_manifest.py`):**
  VCTK 0.92, mic1 only, 5 clips/speaker preferring 4–10 s, staged PCM_16 @48 kHz.
  510 sources / 102 speakers (the 510-cap tail-truncates the last 8 speakers by
  sort order: p360–p364, p374, p376, s5 — capacity, not quality). **Caveat
  (audit W7):** clips concentrate in utt 002–008, so train/test speakers read
  the SAME passages — Gate 1 is content-controlled; it carries no
  content-generalization claim. Follow-up arm should draw utt ≥025.
- **Disk (audit C2):** full-frame cache ≈ 300+ GB > D:'s 173 GB free. `data/cache`
  is now an NTFS junction → `C:\pilot0-cache` (785 GB free). Corpus/manifest
  untouched, so `cache_version` and all future latents stay valid.
- **Runner (audit C4/C5):** `Gate1Decision.passed`/`margin_over_floor` etc. are
  @properties and vanish under `dataclasses.asdict` — the report would have
  carried NO verdicts, and a `SeverityResult.pooled` typo would have crashed the
  summary. Decisions now serialized explicitly; caveats (unconfirmed winner —
  the §2.5 confirmation split remains unimplemented — and content-control) are
  embedded in `gate1_real.json` itself.
- **Monitor (audit C3):** the panel owned the job as a child and killed it on
  window close (`finally: stop()`), which had already produced a silent death
  loop — the headroom scan persists nothing until complete, so four attempted
  starts left zero state. Job is now spawned DETACHED with a pid-file;
  the panel is a reattachable cockpit, only ^P kills.

- **Parallel encode topology (operator, no pipeline edits):** the render path is
  single-threaded, so the six encoders now run as six concurrent worker
  processes (`tools/job_b_parallel.py` → `encode_worker.py` each), sharing the
  GPU. Safe by construction: cache paths are disjoint per encoder, headroom
  scans are deterministic + atomically written (concurrent duplicates converge),
  and row-sharding is explicitly NOT used — it would fragment the corpus-global
  headroom scalar. Gate 1 still runs through the canonical `job_b_run.main()`.

## Job A — fake→real bring-up complete (2026-08-21)

Ran on a Windows 11 box (RTX 5070 12 GB, Ryzen 9, 96 GB RAM), not the GB10 the
bottle anticipated — x86-64 + CUDA 12.8, no aarch64 friction. Per GB10_BRINGUP §7,
the report-back:

- **All four families ran**, first real execution of `seam/real.py`: encodec24k
  (z,d1,d2,d4,d8), wavlm (l1..l24), dac44k (z,d1..d8), mimi (semantic, acoustic).
  [erratum 2026-09-09 → §corrections: `dac44k:z` is the QUANTIZER output, not a
  pre-quant latent; DAC's pre-quant point is the new variant `enc`]
  Every variant `device == "cuda"`, shapes `[T, D]`, sane fps (75.0 / 49.8 / 86.2 / 12.5).
- **D6 reconciliation: no-op.** Real latent dims match the nominal `REAL_SPECS`
  stand-ins exactly (128 / 1024 / 1024 / 512). `REAL_SPECS` + `configs/models.yaml`
  untouched; `test_config` green by construction.
- **Zero `_encode_*` touch-ups.** The flagged risks (Mimi semantic/acoustic split,
  RVQ depth partial decodes) worked as written — including against
  **transformers 5.15.1**, a major version above the `>=4.40` pin.
- **Parity 4/4; full suite 153 passed / 2 skipped** (the 3 Mac-skipped parity tests
  now run); smoke: 170 latents (10 clips × 17 variants), all four `available: True`.
- **Windows install deviation from the Makefile:** venv created manually
  (`.venv/Scripts/`, not `bin/`); torch/torchaudio installed from the cu128 index
  **before** the `[gpu]` extra — PyPI torch wheels are CPU-only on Windows and
  would have silently produced a CUDA-less box. Landed: torch 2.11.0+cu128,
  torchaudio 2.11.0+cu128, descript-audio-codec 1.0.0.
- **Windows console quirk:** `make smoke`'s banner glyphs crash cp1252; run with
  `PYTHONIOENCODING=utf-8`. HF cache pinned via `HF_HOME=data/hf-cache`.

Gate 1 pending Job B (VCTK 0.92 staged for the grouped manifest).

## Final 5-elder ring — whole-repo completion pass (2026-07-13)

Physics, adversarial, architect, testing, integration reviewed the complete repo.
Physics/architect/integration APPROVED; adversarial APPROVED-with-concerns; testing
CHANGES (one false-green). All recommendations applied (with two reasoned exceptions):

- **Rig-guard NaN fail-OPEN closed (adversarial).** `TableScores.from_json` now rejects
  any non-finite score cell at parse, and `_reject_rig` masks to finite shared pairs
  before the rank check — a single `NaN` cell could previously blank `srcc→nan` and let
  a monotone `2·ViSQOL+1` MOS copy through, re-introducing the §8 rig. The one genuine
  fail-open in the integrity chain; regression-tested both ways.
- **Parity contract armed for dac/mimi (testing BLOCKER + integration, convergent).**
  `test_parity._RUNNABLE` now includes `dac44k`/`mimi` (torch-gated, box-only), so
  `pytest -k parity` actually guards their D6 latent-dim/variant reconciliation — the
  runbook told the operator it did, but it silently skipped the two families most likely
  to need an API touch-up.
- **Provenance is derived, not hand-set (adversarial).** New leaf `provenance.py`
  (`BANNER` + `is_fake` + `provenance`); the persisted `fake` flag is derived from the
  candidate encoders (`fake-*` prefix), so a byte-indistinguishable JSON artifact can't
  lie. `make analyze` now co-locates a `provenance.json` like the release bundle does.
- **Barrels + public promotion (architect).** The five newest packages (analysis/quality/
  combos/ood/release) gained `__all__` barrels matching the Phase 1-4 grain; `release`'s
  barrel exports only the matplotlib-free artifact API (figures/reproduce imported
  explicitly, keeping the extra off any JSON-only path). `probes.run._common_conditions`
  → public `common_conditions` and `type_probe._linear_classifier` → `linear_classifier`,
  since both are reused across module boundaries.
- **RQ5 interpolation fail-closed floor (physics).** `interpolation` now passes
  `min_groups=MIN_TEST_GROUPS` (and `bootstrap_fraction` gained the floor), so a
  single-source family can't report `interpolates=True` on one test group.
- **Bring-up ergonomics.** `SMOKE_ENCODERS` is env-overridable (`PILOT0_SMOKE_ENCODERS`)
  per the runbook; the DAC loader derives its rate tag from `native_sr` (the spec is
  authoritative, not a magic `"44khz"`); `_pack` records the ACTUAL latent dim; hiss
  gained a Nyquist guard; the gate1 docstring now matches the (more conservative) code.
  `docs/GPU_BRINGUP.md` clarified: `{source}` is the manifest token, group sizes need an
  eyeball, and the P.56 SNR-label caveat.
- **Deferred, with reason:** (1) retrofitting the six per-phase demos onto one shared
  synth-corpus helper (architect, demo-only) — the correct home is a new leaf (phase
  demos must not depend on `release`), and each demo's `n_sources`/encoder-set is tuned
  on purpose, so the regression risk on blessed code exceeds the demo-only DRY win.
  (2) flipping the global `bootstrap_over_groups` default to `MIN_TEST_GROUPS` (adversarial)
  — that constant lives in `gate1` and `metrics` can't import it without a cycle; the
  targeted interpolation fix covers the one flagged non-gate path.

## Phase 8 — write & release (2026-07-13)

Paper scaffold + `make reproduce-figures` (F1–F6) + release checklist. New `release/`
package with a strict JSON→PNG boundary; docs in `docs/PAPER.md` / `docs/RELEASE.md`.

- **One corpus, six figures.** `release.artifacts.write_artifacts` runs analyze +
  run_gate2 + analyze_combos + run_ood_teaser on a SINGLE synth corpus/cache, so the
  six figures share the same sources, splits, and headroom — no cross-figure drift.
  Each phase's report is serialised to `reports/{analysis,quality,combos,ood}/*.json`;
  `figures.render_figures` reads ONLY the JSON, so plotting never re-runs the pipeline
  and is testable on fixtures alone (fast) with one slow end-to-end integration test.
- **FAKE is stamped in three places, not one.** The Mac seam's numbers are plumbing:
  `reports/provenance.json` records `"fake": true` + git SHA, every PNG stamps the
  banner in-frame (a stray figure can't masquerade as a result), and the demo prints
  it. On the GPU box the real backends drop into `build_demo_corpus` unchanged.
- **Strict JSON, shared.** `serialize.to_jsonable` (numpy→list, non-finite→null) was
  promoted out of `analysis.audit` so the analysis demo and the release writer emit
  byte-identical schemas; the analysis report→dict mappers moved to public
  `analysis.serialize`. `ood.run.run_ood_teaser` was extracted from `ood.audit` so the
  demo and the release writer share the OOD orchestration (same run/audit split as
  every other phase).
- **F4 scatter rides on the Gate-2 report, not a second head fit.** The pooled
  head-vs-ViSQOL scatter needs the raw (ViSQOL, pred) pairs; `run_gate2` captures them
  from the SAME fitted `pred` it already scores G2a on (identical `(test)&(sev>0)` mask)
  and carries them on `Gate2Report.scatter`, so the F4 cloud is exactly the points its
  SRCC annotates — no redundant fit, no drift risk (integration review).
- **matplotlib is an optional `figures` extra**, pulled into `dev` (so tests render).
  Pure-Python + numpy, Mac-installable, no torch — it stays out of the core probe path.
  `figures.py` uses the Agg backend (headless) and errors clearly if the extra is absent.

## Phase 7 — combos & OOD teaser (2026-07-13)

Pairwise degradation combos for the RQ2 additivity test, plus a no-reference
metric-disagreement OOD teaser. Fake latents ⇒ plumbing; the real combo/OOD renders
come from the box. New `combos/` and `ood/` packages.

- **Combos share the singles' cache AND the L4 scalar.** `encode_combos` renders
  leg A then leg B (ordered — degradations do not commute) at a shared MID severity
  into the SAME Phase-3 cache, keyed by family `A+B`, reusing the corpus headroom
  scalar so z̄(a+b) is directly comparable to z̄(a)/z̄(b)/z̄(clean) — a different scalar
  would rotate the displacement vectors and corrupt the cosine. Runs after
  `encode_corpus` (headroom + singles already on disk); resume-safe. To reuse the
  scalar without duplicating the path convention, `encode/pipeline`'s headroom + master
  loaders were promoted to public (`resolve_headroom`/`headroom_path`/`memoized_master_loader`).
- **Additivity is a cosine, reported in BOTH bases.** cos(z̄(a+b)−z̄(clean), Δa+Δb),
  primary in the RAW pooled-latent space (the proposal's z̄ — a superposition test
  lives in the codec's own metric, and raw avoids the noise blow-up standardisation
  inflicts on near-constant latent channels), and `cosine_std` in the standardised
  basis (scaler on the degraded singles, equal-weighting dims so a few high-energy
  channels can't set the verdict) — the answer is basis-dependent, so both ship
  (physics review). The cosine constrains DIRECTION only: ≈1 ⇒ the combo is
  CODIRECTIONAL with Δa+Δb, not equal to it. Each cell is restricted to the sources
  present in all of a/b/ab so the role-centroids share one population. Source-level
  bootstrap CI (the `min_groups` floor reused from Gate 1 / Phase 6). Descriptive, not
  gated.
- **Zero-shot transfer.** The single-degradation linear type probe (train split)
  predicts held-out TEST combos; report the predicted single-family distribution and
  the fraction landing on a constituent leg (does noise×clip read as noise or clip?).
  Descriptive.
- **OOD teaser is real analysis on a fake data source.** Off-manifold textures (FM,
  granular, noise-band, chirp — deliberately not clean-speech-plus-degradation) are
  encoded and scored by the REAL grid-fit head (clipped to [1,5], a bounded quality
  scale — its off-manifold blow-up is itself the OOD-unreliability signal); the
  incumbent columns are FABRICATED (`ood.scores`, each with a different systematic OOD
  bias). All metrics share the 1–5 scale, so per-clip disagreement is the std across
  them in that RAW shared space after CENTERING each on its grid mean (removing only a
  calibration offset) — NOT dividing by each metric's own grid std, which would
  manufacture spread from a common off-grid drift and make "OOD > grid" a normalization
  artifact (physics review). Motivation only (F6), NO claims; real generative renders +
  real NISQA/DNSMOS/UTMOS replace the fabricated columns at bring-up, disagreement math
  unchanged.

## Phase 6 — quality head & Gate 2 (2026-07-13)

The no-reference quality head + the §8 Gate 2, built against the fake seam. Codec
latents are FAKE and the ViSQOL/MOS/baseline scores are SYNTHETIC (`FakeScores`) —
pure plumbing; the real tables come from the box. New `quality/` package.

- **The rig is designed out (§8, the proposal's own elder warning).** A head trained
  on ViSQOL must not be declared to "beat MOS-predictor incumbents" on ViSQOL — they
  predict MOS, so they'd lose by construction. So the training target and the G2b
  evaluation ground truth are DIFFERENT metrics, kept apart in `quality/scores.py`:
  the head regresses **ViSQOL** (full-reference, reference-free at inference); G2b
  scores the head AND the NR baselines (NISQA/DNSMOS/UTMOS) against **human MOS**,
  which none of them trained on. SRCC is rank-based, so the head emitting a ViSQOL
  scale rather than a MOS scale is fine. No MOS ⇒ G2b **NOT EVALUABLE** — it is never
  silently scored on ViSQOL (`Scores.has(mos)` fails closed).
- **G2a is scored on DEGRADED cells only (M2 discipline).** Clean is trivially
  top-quality; including it in the pooled ViSQOL SRCC would inflate the number
  without proving the head can rank degradations. Clean still anchors the head's
  TRAINING as a legitimate high-quality point. Pre-registered: G2a = pooled
  head-vs-ViSQOL SRCC CI-lower ≥ 0.85 over degraded test cells. Per-family SRCC is
  ALSO reported (§2.4) though the gate keys on pooled — pooling across families adds
  between-family rank spread that can flatter the pooled number on real ViSQOL, so the
  per-family view is what exposes a Simpson-type inflation (physics W2).
- **G2b is a PAIRED head-vs-baseline test on MOS.** Per family, over the same test
  groups, the head "beats" a baseline only if the bootstrap CI-lower of
  `srcc(head, MOS) − srcc(baseline, MOS)` is > 0 — a paired difference (shared seed →
  identical resamples), the same rigor Phase 5 uses for the MLP−linear gap, and far
  tighter than comparing two independent CIs. A family counts only if the head
  paired-beats **every** baseline (beat-the-best), on ≥ 5/7 families. A family is
  judged only if its MOS-covered test cells span ≥ `MIN_TEST_GROUPS` distinct SOURCES
  (not merely ≥ `MIN_MOS_CELLS` rows): a partial-MOS subset can cover a family on one
  speaker, and a cluster bootstrap over a single group collapses to a zero-width CI
  that would clear CI-lower > 0 on single-source evidence (elder blockers — physics
  B1 / adversarial B2). `bootstrap_over_groups` now also refuses a CI below that group
  floor (`min_groups`), belt-and-suspenders. So a MOS SUBSET (e.g. a speech-only human
  study) drops under-covered families instead of crashing OR passing on thin evidence;
  G2a still runs on the full corpus. CI-lower gating + the underpowered guard are
  reused from Gate 1, so a thin split can't pass on a point.
- **Two machines, one seam.** `Scores` is a Protocol; `FakeScores` (Mac) synthesises
  ViSQOL/MOS/baselines deterministically per cell — ViSQOL and MOS fall with severity
  around a shared per-source content offset (so they correlate as in reality) but MOS
  is family-weighted and independently noised, so recovering ViSQOL does NOT hand you
  MOS. `TableScores` (box, BRINGUP) reads precomputed metric tables keyed by cell and
  **rejects a MOS column that ranks ViSQOL-identically** — Spearman ≥ 0.999 over the
  shared cells, not just byte-equality, because the gate is rank-based and a monotone
  copy (`2·ViSQOL`, or one cell nudged) would otherwise reinstate the §8 rig it exists
  to stop (elder blocker — adversarial B1 / physics W1).
- **LCC added** (`metrics.lcc`, Pearson) as the §2.4 secondary next to SRCC; Gate 2
  keys on SRCC. `make quality-demo` runs the whole path on enough sources to be
  POWERED (past `MIN_TEST_GROUPS`), so the fake head's FAIL is a genuine clause
  decision — random latents recover neither ViSQOL (G2a) nor paired-beat the MOS
  predictors (G2b) — not the underpowered short-circuit masking as a result.
- **Deferred to bring-up (needs the real tables/annotations):** the ViSQOL C++/bazel
  build (§7 risk — attempt early; PESQ-16k arm is the fallback reference target), the
  actual NISQA/DNSMOS/UTMOS runs, and a MOS-annotated speech subset for G2b. Absent
  the last, Gate 2 reports G2a only and marks G2b not-evaluable — an honest partial
  result, not a fabricated pass.

## Phase 5 — full representation matrix, elder review applied (2026-07-13)

The §2.3 matrix is completed and the §2.4 analyses that don't need a reference
metric are built. All codec numbers on the Mac are still FAKE (plumbing); the
scientific matrix comes from the box. New `analysis/` package + probe extensions.
Physics (APPROVE-WITH-CONCERNS) + adversarial (NOT-APPROVED, 2 blockers) +
cartographer; all findings below landed and re-reviewed.

- **RVQ-depth variants are first-class (§2.3).** EnCodec-24k and DAC-44k now
  declare `z, d1, d2, d4, d8`; `d{k}` = the sum of the first k residual-codebook
  embeddings, in the same embedding space as the continuous `z`, so `latent_dim`
  is shared across a model's variants (fake auto-emits all; real sums codebooks at
  bring-up). [erratum 2026-09-09 → §corrections: "the continuous `z`" holds for
  EnCodec only — DAC's `z` is quantized, and `enc` is its continuous point]
  WavLM layer sweep and Mimi semantic/acoustic were already declared.
- **All four real families wired (`WIRED_FAMILIES` = encodec/wavlm/dac/mimi).**
  `real.py` now implements DAC (`dac.DAC.load` → `from_codes` for depths) and Mimi
  (the **semantic vs acoustic split** via the split-RVQ decoders — the §2.3 headline
  sub-experiment) plus EnCodec/DAC RVQ-depth partial decodes. These paths are
  BRINGUP: written against the documented APIs, validated on the box (shape +
  fake/real parity), never run on the Mac. `available_encoders()` still reports
  False for them on a torch-less machine, so nothing lies.
- **Nonlinearity gap (RQ1, §2.4 secondary).** `mlp_probe.py`: a 2-layer MLP-256
  type probe reported next to the linear probe; the MLP−linear macro-F1 gap is
  **paired-bootstrapped** (both probes scored on the same drawn test groups) so its
  CI-lower says whether nonlinear headroom is real. Reuses the linear probe's own
  pipeline (single source of truth).
- **Frame-level dropout probe (§2.4 "frame-level for dropouts only", uses D9).**
  mean+std pooling averages away the intermittent zeroed bursts that ARE the
  dropout signature. `frame_dropout.py` builds a **level-invariant** feature from
  the per-frame latent norm (normalised by the clip median, so the L4 scalar is
  irrelevant): low order-statistics + fraction of near-silent frames. The reported
  gap is precisely "dropout-tuned frame feature vs generic mean+std pooling" — NOT
  the broad "frame-level info helps" (mean+std's std already carries some burst
  variance) — findings W3/P2. The low-norm assumption is representation-dependent
  (a per-frame-normalised SSL rep like WavLM can flatten frame norm → honest
  false-negative); GPU_BRINGUP verifies frame norm actually drops on dropout per
  backend before the gap is trusted.
- **Interpolation / monotonicity (RQ5) — the single-level trap avoided, and made
  significance- and loudness-safe.** §2.4 pre-registered "train {1,2,4,5}, test
  {3}". Scoring SRCC on the held-out level 3 ALONE is a constant target →
  undefined. `interpolation.py` trains on {1,2,4,5} and scores SRCC over the FULL
  held-out-source ladder 1..5 (a real rank correlation). The interpolation claim is
  a **bootstrap probability** (`interp_frac` — fraction of test-group resamples in
  which mean-pred(2) < mean-pred(3) < mean-pred(4), STRICTLY increasing only; a bare
  three-point check fires ~1/3 on noise and a backwards-ordinal probe must not count
  — findings B1/P1). A family is `evaluable` only when both neighbours (2, 4) AND
  the held-out 3 are present in the common cells (band-limit at 16 kHz keeps only
  the widest cutoffs, so placing 3 would be extrapolation — finding W2). `interpolates`
  requires interp_frac ≥ 0.90 AND a positive SRCC CI-lower; whether the codec
  interpolates BEYOND loudness is judged against the **energy control's** own
  interpolation at the report level (`n_interpolate(energy)`: codec SRCC CI-lower ≥
  energy CI-upper + 0.05, the same buffer as G1b — findings W1). Clean is excluded.
  Supersedes the Phase-4 note that deferred the interpolation SRCC.
- **Geometry (RQ2).** `geometry.py`, all in the standardised feature space (one
  scaler on the degraded rows so no high-variance dim dominates a cosine): the
  linear probe's per-class weight directions (cosine matrix), the probe-free
  class-mean−clean directions (cosine matrix), and PCA of the degraded
  (family, severity) condition centroids (clean excluded; count is common-cell
  dependent — band-limit sheds cutoffs ≥ Nyquist per rate — so NOT a fixed 36,
  findings C1/P3/S6). These views are **in-sample / descriptive** (fit on all rows,
  they gate nothing) — `probe_cosine` is NOT the held-out Gate-1 probe (S1). Off-
  diagonal cosines near 0 ⇒ separable directions; feeds, but does NOT yet run, the
  Phase-7 additivity test.
- **Additivity deferred to Phase 7 (per §3).** The additivity cosine test needs the
  pairwise-combination subset (noise×clip, hiss×lowpass, hum×mp3), which §3 places
  in Phase 7 with the combo grid; Phase 5 geometry stops at the single-degradation
  direction structure so no grid/manifest change is dragged in here.
- **`make analyze` (acceptance).** `analysis/audit.py` writes the three artifacts —
  `reports/analysis/{heatmap,cosine,monotonicity}.json` — over the full matrix with
  the fake-⇒-plumbing banner. Figure rendering (PNG) stays in Phase 8
  `make reproduce-figures`; Phase 5 produces the numeric artifacts they draw from.
- **Hardening from the adversarial pass.** RVQ depth decode asserts the checkpoint
  exposes ≥ the deepest requested codebook before slicing, so `d8` can never be a
  silently truncated `d6` (B2). Artifact JSON serialises non-finite floats as
  `null` so strict parsers don't choke on bare NaN (S3). `available_encoders()` now
  also checks the family's backend lib via `find_spec`, so the box can't report a
  backend encodable then fail at encode (S4); the Mac still reports all real names
  False. `encode()` has an explicit `else: raise` after the family dispatch (S5).
  The analysis summary flags a candidate UNDERPOWERED when test sources <
  `MIN_TEST_GROUPS`, since a ≤1-group bootstrap CI collapses to a point (W4).

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

## Gate 1 — first real-corpus run: FAIL, and the failure has structure (2026-08-22)

> **Corrected 2026-09-09** — the G1b severity reading below is retracted and the
> "level-only control" framing with it. See §"Science spot check — corrections" at
> the end of this file.

Job B complete: cache 338,130/338,130 cells, all six encoders, zero worker
failures. Gate 1 ran per the frozen protocol (55.7 min): 34 common conditions,
20 test groups (>= 3, so NOT underpowered — this is a clean fail, not a power
fail). Floor = log-mel macro-F1 0.906 [0.885, 0.925]; energy control 0.638.
Full report: `reports/gate1_real.json` (local; reports/ is gitignored).

**All 17 candidates FAIL the pre-registered conjunction.** The decomposition
is the finding:

- **G1a type** (beat mel floor by >= 0.05, candidate CI-lower vs floor
  CI-upper): passed ONLY by wavlm l1 (0.988, margin +0.056) and l6 (0.985,
  +0.052). Every codec sits BELOW the mel floor on damage-type ID
  (encodec ~0.84, dac ~0.88-0.89, mimi acoustic 0.862).
- **G1b severity** (without-clean SRCC CI-lower >= 0.80 on >= 4/7 families
  AND >= 0.05 over the energy control): nobody reaches 4/7. encodec 3/7 at
  every depth; wavlm l1/l6 2/7; dac 1-2/7; mimi 0-1/7. Absolute SRCCs are
  often high (encodec z: noise .980, hum .977, hiss .952) — the margin over
  the level-only control is what's thin: severity is largely energy-explained.

**Predictions made before the run held** (ENCODER_ANATOMY §6, polyglot idea
doc): WavLM type-F1 decays monotonically with depth, 0.988 (l1) -> 0.906
(l24 = exactly the floor), severity passes 2 -> 0 — deeper layers abstract
toward phonetics and legally discard damage. Mimi's semantic path is
near-blind to damage type (0.402 vs 0.862 acoustic).

**One-line reading:** WavLM's early layers read WHAT (and beat spectral
statistics at it); EnCodec comes closest on HOW MUCH; nothing reads both.
Type identity is linearly accessible; the non-loudness severity residual is
thin everywhere.

**Bounds on the claim:** winner-confirmation split unimplemented;
content-controlled corpus (train/test share VCTK passages), which likely
flatters the mel floor — the content-general arm (utt >= 025) is the
scheduled follow-up before any strong negative is published.

Per pre-registration, FAIL is a finding (the G1c-style branch: publish the
negative with the geometry analysis). Next step: RQ2 geometry — descriptive
cosine/PCA views (`analysis/geometry.py`) plus the shift-vector experiment
(per-clip delta concentration vs the 1/sqrt(D) null, magnitude-vs-severity,
persistence under dimension reduction).

## RQ2 shift-vector geometry — directions are real everywhere probes struggled (2026-08-22)

> **Corrected 2026-09-09** — the "Nx null" concentration multipliers below are
> downgraded to descriptive, and the reduction-persistence claim is in-sample. See
> §"Science spot check — corrections" at the end of this file.

Full 18-candidate sweep (`tools/geometry_run.py`, 49 min, reports/geometry_real.json
+ figures/geometry/). Per-clip Δ = x(dirty) − x(clean) paired by source,
standardized basis, exact all-pairs concentration, isotropic 1/sqrt(D) null.

- **Per-family Δ directions concentrate at 9–30x the null in every candidate
  except mimi:semantic.** Median family mean-cosine: wavlm l1 0.65 (29.5x null,
  D=2048), dac d1 0.66 (29.7x), encodec z 0.63 (10.1x), logmel 0.60 (11.7x).
- **||Δ|| tracks severity everywhere**: SRCC medians 0.83–0.97; wavlm l1 the
  most consistent (min family 0.92).
- **Reduction persistence**: encodec z and wavlm l1 keep min-family
  concentration 0.89 / 0.91 after PCA to k=2; deeper wavlm layers collapse
  (l18 k=2 min 0.04) — the same early-layer story as the probes.
- **mimi:semantic is the negative control and behaves like one**: median
  concentration 0.069 (2.2x null), k=2 min 0.046 — the path distilled to
  discard acoustics has essentially no damage geometry, matching its 0.402
  type-F1.
- Caveats embedded in the artifact (descriptive/in-sample, standardized basis,
  isotropic null only, ||Δ|| never comparable across candidates, plus both
  Gate-1 corpus caveats).

Reading with Gate 1: the damage AXES exist and survive reduction; what Gate 1
punished was margins over strong baselines (mel floor on type, energy control
on severity), not the absence of geometry. logmel itself concentrates at 11.7x
null — spectral statistics have geometry too; the null-relative gap of wavlm l1
over the floor is the concrete positive to carry forward.

## Phase 7 combos — superposition holds for independent artifacts; the interacting pair is a severity-graded interaction readout (2026-08-22)

> **Corrected 2026-09-09** — the superposition reading below rests on a cosine that
> is not identifiable on its own; the reanalysis adds the dominance and recovery
> metrics. See §"Science spot check — corrections" at the end of this file.

Full run (`tools/job_c_parallel.py`, 141 min total: 40 min parallel encode of
4080–22950 combo cells/encoder into the same banked-headroom cache, 141 min
sequential 18-candidate analysis, reports/combos_real.json). Pairs
noise+clip / hiss+bandlimit / hum+mp3, severities {2,3,4}, order A-then-B
fixed; hiss+bandlimit sev2 dropped at 16 kHz (cutoff = Nyquist).
Additivity = cos(z̄(a+b) − z̄(clean), Δa + Δb), bootstrap CIs, raw +
standardized bases. Transfer = zero-shot read of combo cells by the
singles-trained Gate-1 type probe.

- **Non-interacting pairs superpose everywhere.** noise+clip and hum+mp3:
  additivity cosine +0.98..+1.00 in ALL 18 candidates at every severity —
  codec latents included. Independent artifacts add linearly even in curved
  latent spaces; this row of the harness does not discriminate candidates
  (logmel's +0.989 mean is the expected linear ceiling, and everyone matches
  it here).
- **hiss+bandlimit is physically interacting (bandlimit erases the hiss band)
  and the latents expose it, graded by severity.** Raw cosine sev2→3→4:
  encodec z 0.927/0.537/0.187, mimi acoustic 0.953/0.573/0.169,
  wavlm l1 n/a/0.807/0.359, dac z 0.974/0.893/0.761, wavlm l24
  n/a/0.920/0.895 — while logmel stays 0.965/0.948 (the killed band dominates
  the sum vector, linearizing over the interaction). Sub-additivity here is
  fidelity to signal physics, not representational failure; encodec z's clean
  monotone collapse is effectively an interaction-strength readout.
- **The single-label probe reads the dominant leg, never both.** noise+clip →
  "noise" ~1.00 everywhere (clip 0.00); hiss+bandlimit → "bandlimit"
  0.66–1.00 with hiss ≈ 0 in every candidate except logmel (0.37/0.63);
  leakage goes to "mp3" (itself band-limiting). hum+mp3 splits between the
  legs (either 0.89–1.00). Masking is consistent with the A-then-B physics —
  after severe bandlimiting the hiss evidence is largely gone, so
  "bandlimit" is arguably the perceptually correct single answer.
- **mimi:semantic fails transfer exactly as the negative control should**:
  frac_either 0.45 / 0.43 on the two spectral pairs (vs 0.68–1.00 for
  everyone else), predictions scattered — third independent instrument
  agreeing with its 0.402 F1 and 2.2x-null geometry.

Implications for the wheel: axis decomposition of independent artifacts is
safe in every candidate latent; simultaneous-artifact reporting needs
multi-label reads (argmax collapses to the dominant leg by construction);
and interacting pairs need either an interaction term or the encodec-style
sub-additivity signal itself as a feature. Caveats carried in the artifact:
descriptive (no pre-registered gate), A-then-B non-commuting order,
content-controlled corpus, winner-confirmation split still unimplemented.

## Science spot check — corrections to the Gate-1, RQ2 and Phase-7 readings (2026-09-09)

Independent DSP/ML spot check (`docs/DSP_ML_SCIENCE_SPOT_CHECK_2026-09-09.md`,
S1–S10) remediated on `fix/science-spot-check`, elder-blessed plan
`docs/plans/2026-09-09-science-spot-check-remediation.plan.md`. **Gate 1 is not
redefined and no historical text above is rewritten** — this section states, per
anchored claim, what is retracted, downgraded, or stands. New numbers come from
`reports/ceiling_real.json`, `reports/additivity_real.json` and
`reports/level_split_real.json` (all read the same banked latents; no GPU run).

### S1 — "nobody reaches 4/7" is not a fact about the codecs (RETRACTED as a verdict)

Anchor: the G1b bullet, "nobody reaches 4/7". With 100 test sources per level the
ladder is massively tied, so any Spearman against it is capped at **0.9798** (K=5)
and **0.9428** (band-limit, K=3). The energy control is already at that cap —
without-clean point 0.9797979 on noise (the K=5 ceiling to seven decimals), 99.93 %
on hiss, 99.91 % on band-limit, 95 % on hum — and G1b's bar is its CI-UPPER plus
0.05: 1.0298 / 1.0297 / 0.9928 / 0.9938. All four land **above the ceiling**. Only
clip, mp3 and dropout are feasible, G1b needs four, and an oracle probe pinned at
the ceiling with a zero-width CI passes **3/7** — the severity leg was unreachable
by any representation. One number in `reports/ceiling_real.json` reads oddly and is
not a contradiction: the energy control's bootstrap CI-upper can sit a hair *above*
the design ceiling (noise 0.9798137 vs 0.9797979, +1.6e-5; band-limit +3.1e-5),
because the ceiling is that of the full balanced ladder while each bootstrap resample
draws its own unbalanced one and is not bounded by it — an excursion of ~1e-5 against
a 0.05 margin, which moves no verdict here. What stands: the run was well powered
(20 test groups) and every per-candidate count reproduces exactly.

### S7 — "the level-only control" (RETRACTED)

Anchor: "the margin over the level-only control" and "severity is largely
energy-explained". The control's pooled vector is six log-energy statistics, and a
uniform gain moves exactly one direction of them: the three log-MEANS together,
`(1,1,1,0,0,0)/√3`. The same ridge on that 1-d coordinate and on its orthonormal
5-d complement (raw pooled space, pre-scaler) gives without-clean SRCC:

| family | 6-d | level 1-d | invariant 5-d |
|---|---|---|---|
| noise | 0.980 | 0.972 | 0.980 |
| hiss | 0.979 | 0.736 | 0.979 |
| hum | 0.932 | 0.753 | 0.932 |
| clip | 0.510 | −0.013 | 0.493 |
| bandlimit | 0.942 | 0.922 | 0.942 |
| mp3 | 0.550 | 0.419 | 0.557 |
| dropout | 0.687 | 0.248 | 0.572 |

The invariant complement matches or beats the full vector on every family, and on
clip the level coordinate carries **nothing** (−0.013). "Energy-explained" is
therefore not "loudness-explained": the control is a spectral-balance and
temporal-dispersion representation, so beating it is a harder, different claim than
the one published. A level-matched arm could not have shown this — no uniform gain
touches the invariant subspace (regression-tested in `tests/test_levelmatch.py`).
Validity bound: 1 frame in 10,582,738 sits below −80 dBFS, where `_EPS` would begin
to matter, so the split is not epsilon-limited.

### S2 — "pre-quant z" on DAC (RETRACTED)

Anchors: `:47`, `:250` above, `docs/PILOT_A_IMPLEMENTATION.md`,
`docs/GPU_BRINGUP.md`, `docs/TRENDS.html`, `docs/FIELD_GUIDE.html`.
`DAC.encode()` rebinds its `z` to the quantizer output before returning
(`dac/model/dac.py:243–247`), so every `dac44k:z` latent in the bank is quantized
over all of the model's codebooks. It is NOT `d9`: the codebook count is asserted
nowhere. The variant keeps its key (the bank stays valid); DAC's continuous point
ships as a new variant `enc = model.encoder(preprocess(x))`, same `[T, D]` space,
in `REAL_SPECS` but **not** in the frozen sweep — so no published number moves.
Every variant now carries a `semantics` string with a yaml drift guard, read
through a required `variant_semantics()` lookup that feeds each report's candidate
metadata. What stands: `encodec24k:z` really is the continuous encoder output.

### S4 — "9–30x the null" (DOWNGRADED to descriptive)

Anchor: the RQ2 concentration bullets. `1/√D` is the RMS cosine of ONE random pair
in D dimensions, so `C / (1/√D)` is a function of the NOMINAL dimension: duplicating
a latent channel inflates the ratio without changing the geometry, and a
`null_sd_mean` correction does not fix it (the duplicated-feature counterexample
survives it). The multiplier is therefore not a score and is no longer printed;
`tools/geometry_run.py` prints `C` itself and the figure line reads "single-pair RMS
null 1/√D (reference, not a score)". What stands: every raw concentration `C`
(pooled 0.16, per-family 0.40–0.88 for encodec z) and the ORDERING across families
and candidates, including mimi:semantic as the negative control — those are
dimension-free comparisons within a candidate.

### S5 — "the axes survive reduction" (DOWNGRADED, no new number)

Anchor: the reduction-persistence bullet. The PCA is fit on the same degraded rows
the persistence is then measured on, so "survives reduction to k=2" is an in-sample
statement, not a held-out one. Held-out projection evaluation is out of scope for
this remediation and no replacement number is offered here — the claim should be
read as descriptive until one exists.

### S6 — "superposition holds for independent artifacts" (DOWNGRADED, and split in two)

Anchor: the Phase-7 heading. One cosine cannot carry that claim. At severity 3
**noise+clip** reads cosine 0.999–1.000 across all seven candidates — yet cos_to_a
0.99–1.00 against cos_to_b 0.08–0.40, cos_legs 0.07–0.39, norm_ratio 9–31, and
(α, β) = (≈1.00, 0.04–0.34). The combo sits on the noise leg and clip is barely
present: dominance by a leg an order of magnitude larger, which no cosine can
distinguish from addition. **hum+mp3** at 3 is the real thing — cosine 0.988–1.000
with cos_to_a 0.15–0.78, cos_to_b 0.58–0.98, cos_legs −0.04 to 0.20, norm_ratio
0.18–1.26, and α 0.31–0.98, β 0.61–1.00: two comparable, near-orthogonal legs that
genuinely add. "Superposition holds" becomes "superposition holds where the legs are
comparable and near-orthogonal, and is *untestable* where one dominates".

Provenance: the numbers come from `reports/additivity_real.json`, computed at
`60f761c` before `combos/additivity.py` gained its duplicate-row guard. The guard is
proven inert on this bank: `additivity_run.py --scan-duplicates` finds 0 duplicate
role/source keys across all 18 candidates (400,350 keys), and recomputing the two
coverage-edge candidates with the guard in place reproduces all 18 of their cells
field-for-field. The D3 regression guard reproduced 156 evaluable cells bit-for-bit
against `combos_real.json`; the other 6 — `hiss+bandlimit@2` on the six 16 kHz
candidates, where band-limit 2 sits at or above Nyquist — are unevaluable in *both*
reports. That run's `n_cells_differing: 6` was a mis-binning of those six, not a
discrepancy; `combos/legacy_check.py` now counts them as `n_cells_both_unevaluable`.
