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
  bring-up). WavLM layer sweep and Mimi semantic/acoustic were already declared.
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
