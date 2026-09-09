# Science spot-check remediation — plan v2 (2026-09-09)

Ritual: Novem, 3-elder ring (Adversarial, Physics, Architect) by owner request.
Base: `main` @ `294d843`. Branch: `fix/science-spot-check`. Audit:
`docs/DSP_ML_SCIENCE_SPOT_CHECK_2026-09-09.md` (S1–S10). v1 → v2 folds the
Gate-3 verdicts (§8). Binding rules: `~/.claude/rules/modular-architecture.md`.

## 1. Goal

| Item | Audit | Deliverable |
|---|---|---|
| A | S1 | Ceiling-aware reanalysis of `reports/gate1_real.json` → `reports/gate1_ceiling.json` |
| B | S2 | DAC `z` labelled quantized at every site; new DAC variant `enc` = true encoder output |
| C | S6 | Source-paired dominance/recovery metrics in `additivity()`; cached-bank reanalysis → `reports/combos_additivity_v2.json` |
| D | S7 | Level / gain-invariant split of the cached energy control → `reports/energy_level_split.json` (free, no GPU) |
| E | S1/S2/S4/S5/S6/S7 | Dated correction section in `docs/DECISIONS.md`; pointer cards in the HTML pages; label fixes |
| F | S1/S7 | Level-matched arm **primitive + design note only**; pipeline threading deferred with every site listed |

Frozen Gate 1 is **not** redefined. Historical DECISIONS.md text is not rewritten
(inline `[erratum 2026-09-09 → §corrections]` brackets only). **No GPU run.**
**One builder, sequential phases** — v1's two-stream disjointness claim failed at
API granularity (§8).

## 2. Verified facts (cartographers A–D + elders, on `294d843`)

- F1 `Estimate(point, lo, hi)`, `clears()` ⇔ `lo ≥ threshold` — `probes/metrics.py:21–30`; `bootstrap_over_groups` unique groups, n=1000, seed 0 — `:59–99`; `srcc` = `scipy.stats.spearmanr` with nan guards — `:42–48`. File is 125 lines.
- F2 `reports/gate1_real.json` ← `tools/job_b_run.py:70–71`; `report.{floor, energy, decisions[], n_common_conditions}`; keys `encoder:variant`; energy `energy:energy`; decisions carry `severity`, `energy_severity`, `n_severity_pass`, `n_test_groups`.
- F3 `SEVERITIES=(1..5)` — `degrade/grid.py:64`; common grid — `probes/run.py:41–45`; bandlimit has K=3 in the 16 kHz-inclusive common grid. The gate reads the **without-clean** ladder (`severity.py:44–48`).
- F4 Suite baseline `173 passed, 2 skipped` (`.venv/Scripts/python.exe -B -m pytest -q tests`).
- F5 Tool conventions: `tools/<domain>_{run,figures,worker,parallel}.py`, `ROOT` from `__file__`, `to_jsonable()` before `json.dumps` — `tools/job_c_run.py:57–78`.
- F6 `_encode_dac` stores `model.encode(x)[0]` as `z`, comment says quantized — `seam/real.py:170–185`. Stale "pre-quant z" text: `seam/real.py:10–11, 24–26`; `seam/registry.py:20–22` (inside `REAL_SPECS`); `docs/PILOT_A_IMPLEMENTATION.md:46`; `docs/GPU_BRINGUP.md:17`; `docs/DECISIONS.md:47, 250`; `docs/TRENDS.html:68`; `docs/FIELD_GUIDE.html:93–94` (a `rowspan="2"` cell spanning encodec+dac rows).
- F7 Installed DAC overwrites encoder output with quantizer output — `dac/model/dac.py:243–247`; `latents` = low-dim projected per-codebook, **not** the encoder output. True pre-quant = `model.encoder(preprocess(x))`, same dim as `z`. `n_codebooks` for DAC-44k is asserted nowhere → do **not** label `z ≡ d9`.
- F8 Bank id `cache_version` — `encode/cache.py:51–62`; per-variant subdir `…/name/variant/cell.npz`; new variant = new subdir, same bank. `encode/pipeline.py:78` encodes **every registry variant**, so adding `enc` costs one full DAC corpus pass at the next encode run (accepted, stated in E and F).
- F9 `REAL_SPECS` is truth, yaml mirrors — `seam/registry.py:7–8`; `tests/test_config.py:19–25` compares family/sr/dim/checkpoint/variants only; the DAC tuple is pinned at `tests/test_analysis.py:64`. Bare `dac44k:z` literals: `tools/geometry_run.py:41`, `tools/job_b_run.py:24`, `FIRST_LIGHT.html:70,116`, `TRENDS.html:92`.
- F10 `additivity()` — `combos/additivity.py:86`; centroids per role `:72–76`; intersection by speaker **group** `:100–104` while the docstring `:19–20` and the `n_groups` comment `:47` claim *sources*; bootstrap over groups `:116–117`. Serialized twice: `tools/job_c_run.py:62–78` and `release/artifacts.py:78–86`; consumed by `release/figures.py:108,133–136`; schema pinned in `tests/test_release.py:94–100, 220`.
- F11 `ProbeData` has **no `source`** — `probes/dataset.py:35–42`; constructed at `:54, :73`, `tests/test_analysis.py:49`, `tests/test_combos.py:52`, `tests/test_probes.py:52`. `ComboData` has `source` — `combos/dataset.py:36–46`; `iter_combo_latents` silently skips missing cells `:23–32`.
- F12 `analyze_combos` is pure wiring — `combos/run.py:24–32`; combo bank `enc-v1+g2e02854749a2d8e2+dbfa8401da5db+c-1+kc4c74c4dc6a9` on disk for all six encoders.
- F13 `null_scale` consumers: `analysis/shift.py:115,145`; `tools/geometry_run.py:128,177–184,238`; `tools/geometry_figures.py:90,95`; `tests/test_shift.py:97,104,228`.
- F14 Render: −23 LUFS master (BS.1770-4 via `pyloudnorm`, `corpus/loudness.py:1–14`) → resample → degrade, never renormalized — `encode/render.py:40–46`. Headroom: one scalar per rate from `render_cell` **without any arm** — `encode/headroom.py:60–63, 79, 85–87`; `pipeline.py:119` keys `resolve_headroom` by bank id.
- F15 Energy control = per-frame {log mean(frames²), log Σ|rFFT|² <1 kHz, log Σ|rFFT|² ≥4 kHz} (not on a common Parseval normalisation), mean+std pooled → 6-d — `seam/energy.py:19,35–36,52–54`; docstring `:1` calls it "level-only". `_EPS=1e-10` breaks gain-equivariance below ≈ −80 dBFS (corpus floor ≈ −60 dBFS is unaffected).
- F16 `cell_id = blake2b(source_token|family|severity|native_sr)` — `encode/cache.py:55–57`. `cache_version` callers: `combos/dataset.py:23`, `combos/encode.py:62`, `probes/dataset.py:28`, `ood/run.py:62`, `pipeline.py:113`. `render_cell` callers: `pipeline.py:85`, `headroom.py:79`. `degrade/version.py:25,29` hashes the **source** of `_WAV_MODULES` into the `d…` segment; `tools/encode_monitor.py:68` sums counts across bank versions.
- F17 Claims to correct: `DECISIONS.md:562, 572–573` (S1/S7); `:592–598, 600–603` (S4); `:597–599` (S5); `:625–630` (S6); `FIRST_LIGHT.html:81, 91, 93, 107, 122, 145–146`; `TRENDS.html:101, 131`; plus F6 sites (S2).
- F18 DECISIONS.md sections `## Title (YYYY-MM-DD)`, appended chronologically. HTML pages are hand-authored (no generator); `FIRST_LIGHT.html:18` defines `.card`.
- F19 (Physics, measured) LUFS-matching a degraded clip to its clean partner leaves the control's mean-log-total feature offset by +4.6 nats at noise/1 and +6.3 at noise/5: the cue lives in pause frames that BS.1770's −70 LKFS / −10 LU gates exclude, and `std` of log-frame-energy is exactly gain-invariant. **No uniform gain removes the confound from the control.** The level coordinate of the 6-d pooled vector is `(1,1,1,0,0,0)/√3`; its orthogonal complement is gain-invariant.
- F20 (Physics, measured) pyloudnorm 0.2.0 raises `ValueError` for clips < 400 ms; silence returns −inf. Single-shot gain matches to ≤ 6e−7 LU. The gain is rate-dependent (bandlimit/5: +1.14/+0.56/+0.31 dB at 44.1/24/16 kHz).
- F21 (Physics, derived) General Spearman ceiling for level counts `m_j`, block mean rank `r_j`, `N=Σm_j`: `ρ_max = sqrt( 12/(N(N²−1)) · Σ_j m_j (r_j − (N+1)/2)² )`; equals the balanced closed form at equal `m`. Balanced-form error on `[3,100,100]` is +0.07 (**overstates** the ceiling). Fixtures: `[100]*5 → 0.9797958971`, `[100]*3 → 0.9428139…`, `[3,100,100] → 0.872316`, `[100,100,100,100,98] → 0.979794`.
- F22 (Physics, derived) `null_sd_mean = sqrt(2/(D·n(n−1)))` is correct but still a function of nominal `D`; the duplicated-feature counterexample survives it. A3 as proposed in v1 is therefore cut.
- F23 (Physics, derived) With `r = ‖Δ̄ab‖/‖Δ̄a+Δ̄b‖`: `rel_residual² = 1 + r² − 2r·cosine` exactly. Collinear legs (`cos(Δa,Δb)≈1`) make every cosine/ratio metric read "additive" even when one leg is absent; the identifiability condition is `cos_legs`, and per-source least squares `(α,β)` is the recovery test the audit asks for.

## 3. Decisions in force (v2)

- D1 DAC `z` **keeps its key** (rename orphans `dac44k/z/*.npz` and three archived reports). `REAL_SPECS` gains a per-variant `semantics` string; yaml mirrors it; `tests/test_config.py` compares it; a required `variant_semantics(name, variant)` lookup feeds every report/figure writer's candidate metadata. DAC `z` = "quantized, all codebooks"; EnCodec `z` = "continuous pre-quant". New DAC variant `enc`.
- D2 `enc` is **not** added to the frozen Gate-1 sweep (`job_b_run.py:21–26`). It enters `REAL_SPECS` (hence the next encode pass — accepted cost).
- D3 S6 metrics are computed on **source-paired** rows across all four roles. Per cell, `rows_identical` = per-role row-set equality between the source-paired and the legacy group-intersected selections; where true the legacy `cosine` must be bit-identical to `combos_real.json`; the reanalysis **fails if no cell is identical**. `n_groups` keeps its meaning; new `n_sources`.
- D4 v1-A3 (`null_sd_mean`) is **cut** (F22). Remaining S4 work is labels: the runner stops printing "×null"; the figure line is relabelled "single-pair RMS null 1/√D (reference, not a score)"; the correction text carries the argument.
- D5 The level-matched arm ships as `encode/levelmatch.py` (Architect: `degrade/` source is hashed into the bank id) with tests, plus a design note. **No change to `cache.py`, `render.py`, `headroom.py`, `pipeline.py` this round**; the note lists every threading site (F16, F14) and the pinned-literal test `dbfa8401da5db` for the follow-up.
- D6 Level matching = BS.1770-4 integrated loudness of the degraded clip set to its clean partner's; guards for −inf/NaN **and** pyloudnorm `ValueError`; tolerance 0.01 LU; `achieved_lufs` stored. Estimand wording: "removes the *gated broadband level* difference from the stimulus", never "removes the level confound from the control" (F19).
- D7 S7 gets a direct, free answer now (item D): severity SRCC per family for the 6-d control, its 1-d level coordinate, and its 5-d gain-invariant complement, from cached energy latents.
- D8 Corpus run for the arm and any Gate-1 protocol change are owner decisions, listed in the design note.

## 4. Phases (single builder, in this order, commit per phase)

**P1 — S2 DAC label + `enc` variant.**
`seam/real.py` (`enc = model.encoder(preprocess(x))`; fix `:10–11, :24–26`, DAC comment), `seam/registry.py` (`semantics`, `variant_semantics()`, fix `:20–22`), `configs/models.yaml`, `tests/test_config.py` (semantics compared), `tests/test_analysis.py:64`, `tests/test_seam.py` (fake DAC gains `enc`; new structural stub test: `enc` from `encoder(...)`, `z` from `encode(...)[0]`, equal shapes, values differ), `tools/geometry_run.py` + `tools/job_b_run.py` (candidate metadata carries `semantics` via the lookup; sweep list unchanged), `docs/PILOT_A_IMPLEMENTATION.md:46`, `docs/GPU_BRINGUP.md:17`.

**P2 — S1 ceiling.**
`probes/metrics.py`: `spearman_ceiling(counts: Sequence[int]) -> float` (general form F21; nan for <2 levels; section map if the file passes 150 lines). `probes/ceiling.py`: `family_ceilings(report, manifest, grid)` → per family `{K, counts_by_level, N, rho_max, rho_max_balanced, energy_hi, required_lo, feasible}` with K and counts derived from the common grid and the test-split manifest rows (without-clean ladder); `candidate_summary` → `absolute_clears`, `margin_passes`, `feasible_families`; `oracle_n_pass` through `SeverityResult.n_pass` with zero-width `Estimate(ρ,ρ,ρ)`. `tools/ceiling_run.py` → `reports/gate1_ceiling.json` + printed table. `tests/test_ceiling.py`: the four F21 fixtures against `scipy.stats.spearmanr` on constructed perfectly-ordered predictions **including the two unbalanced ones**; pinned literals `0.9797958971` (5×100), `0.9428139` (3×100); K map `{bandlimit:3, else 5}`; `oracle_n_pass == 3` on a fabricated report mirroring the real energy bounds.

**P3 — S6 dominance + recovery.**
`probes/dataset.py`: add `source` to `ProbeData` (+ the five construction sites F11). `combos/additivity.py` (section map; docstring `:19–20` made true): per cell add `cos_to_a`, `cos_to_b`, `cos_legs`, `r`, `norm_ratio` (median per-source, nan-guarded when ‖Δb‖<eps), `alpha`, `beta` (median per-source least squares, speaker-bootstrap CI), `rel_residual` (derived from `cosine`, `r`; documented as derived), `n_sources`, `rows_identical`. Both serializers (`tools/job_c_run.py:62–78`, `release/artifacts.py:78–86`) emit them; `tests/test_release.py:94–100, 220` updated; `release/figures.py` unchanged. `tools/additivity_run.py`: additivity-only pass over the cached bank for all 18 candidates + floor → `reports/combos_additivity_v2.json`; asserts D3; prints pair × severity × candidate table. Tests (`tests/test_combos.py`, split into a new `tests/test_additivity.py` if >500 lines): pinned `_cosine([100,0],[100,1]) == 0.9999500037496875`; dominance `Δab=Δa, Δb=(0,1)` → `cos_to_b≈0, α≈1, β≈0, norm_ratio=100`; collinear `Δa=(10,1), Δb=(5,0.4), Δab=Δa` → `cos_legs>0.999`, α/β flagged ill-conditioned; scaled orthogonal `Δab=0.1(Δa+Δb)` → `cosine=1, r=0.1, α=β=0.1, rel_residual=0.9`; missing-role source dropped; `rows_identical` true/false cases.

**P4 — S7 level split.**
`probes/level_split.py` (pure): project pooled energy `X[N,6]` onto `(1,1,1,0,0,0)/√3` and its complement; reuse `evaluate_severity_probes`-style ridge per family. `tools/level_split_run.py` → `reports/energy_level_split.json` (per family: SRCC without-clean CI for 6-d, level-1d, invariant-5d) + table. Test: synthetic data where severity is pure gain → level-1d ≈ 1, invariant ≈ 0; pure spectral tilt → the reverse.

**P5 — Level-match primitive + design note.**
`encode/levelmatch.py`: `match_loudness(degraded, clean, sr) -> (wav, LevelMatch{gain_db, clean_lufs, degraded_lufs, achieved_lufs, applied, reason})` (D6). Tests `tests/test_levelmatch.py`: gain identity ≤ 0.01 LU; short-clip `ValueError` guard; silence guard; **honest negative control**: noise/1-style cell (ΔLUFS≈0) → gain≈0 while `EnergyEncoder` mean-log-total still differs by >1 nat (documents F19). Design note `docs/plans/2026-09-09-level-matched-arm.design.md` (≤ 700 words): estimand (D6), what the arm does and does not remove (F19), the P4 split as the S7 instrument, threading sites (F16/F14: five `cache_version` callers, two `render_cell` callers, per-arm headroom, `encode_monitor.py:68`, arm segment minted in `encode/cache.py:40–52`, pinned literal `dbfa8401da5db`), rate-dependent gain (F20 → record per-rate spread; canonical-bandwidth arm S3 is the real fix), achieved-level audit spec, cost of the corpus run, proposed prospective criterion (absolute readability vs incremental value with feasible margin) — owner decisions.

**P6 — S4 labels + control name.** `tools/geometry_run.py:177–184` (print `C` only), `tools/geometry_figures.py:90,95` (relabel), `seam/energy.py:1` ("spectral-band energy control: level, balance, temporal dispersion").

**P7 — Corrections.** Append `## Science spot check — corrections to the Gate-1, RQ2 and Phase-7 readings (2026-09-09)` to `docs/DECISIONS.md`: per anchored claim (F17) → retracted / downgraded / stands, replacement reading, numbers from P2/P3/P4; ≤ 120 words per finding; inline `[erratum 2026-09-09 → §corrections]` at `:47, :250`; one pointer line under the Gate-1, RQ2 and Phase-7 headings. `FIRST_LIGHT.html`, `TRENDS.html`: one `.card` with ≤ 2 sentences pointing to the section; `TRENDS.html:68` and `FIELD_GUIDE.html:93–94` (split the rowspan) relabelled.

## 5. Verification (builder, before reporting)
Full suite (`… -m pytest -q tests 2>&1 | tail -3`, exact counts vs 173/2); the three tools run end-to-end on the real reports/bank with their tables in the report as numbers; red→green proof per new test (failing assertion first). No file over 500 lines; section maps where >150.

## 6. Out of scope
S3 canonical-bandwidth arm; S5 held-out projection evaluation; S8; S9 seeds; all S10 rows; PAPER.md; GPU runs; Gate-1 redefinition; paired-bootstrap probe re-runs; `D_eff`/participation-ratio geometry (follow-up named in P7); `geometry_run.py:84–100` switching to `ProbeData.source`.

## 7. Open questions (v2)
None blocking. Owner decisions live in the design note (D8).

## 8. Gate-3 synthesis (v1 → v2)

| Finding | Adv | Phy | Arch | Resolution |
|---|---|---|---|---|
| `ProbeData` lacks `source`; disjointness fails | ● | | ● | P3 adds it; single builder |
| Arm threading unreachable / overwrites native cells; headroom arm-blind | ● | ● | ● | D5: primitive + note only; all sites listed |
| `coverage_complete` malformed / vacuous | ● | | | D3 per-role row-set predicate, fail-if-zero |
| `n_groups` meaning drift | ● | | | `n_sources` new, `n_groups` unchanged |
| Ceiling tool can silently invert (7/7) | ● | | | pinned literals, K map, `oracle_n_pass==3` |
| Balanced closed form overstates unbalanced ceiling | | ● | | F21 general form always; unbalanced fixtures |
| `null_sd_mean` still nominal-D | | ● | | D4: v1-A3 cut; labels only |
| LUFS match leaves control's level cue | | ● | | D6 wording; P4 split; negative-control test |
| pyloudnorm `ValueError`, 0.1 LU too loose, rate-dependent gain | | ● | | D6; design note |
| Metrics fail on collinear legs; `rel_residual` redundant | | ● | | `cos_legs`, `r`, `(α,β)`; `rel_residual` derived |
| `degrade/levelmatch.py` hashed into bank id | | | ● | `encode/levelmatch.py` |
| `release/` second serializer + pinned tests | | | ● | P3 covers both serializers + `test_release.py` |
| Label inventory incomplete; no semantics drift test | ● | | ● | F6 list; `test_config.py` compares semantics |
| `enc` silently joins next encode pass | ● | | | D2 accepted, stated |
| Tool naming convention | | | ● | `ceiling_run.py`, `additivity_run.py`, `level_split_run.py` |
| HTML erratum shape | | | ● | `.card`, ≤ 2 sentences |
| `energy.py:1` "level-only" | | ● | | P6 |

## 9. Binding amendments (v2.1, from the Gate-4 re-verdicts) — override §4 where they differ

- AM1 (Adv) **Ceiling pins regenerated from scipy, never by hand.** `[100]*5 → 0.9797978567`, `[100]*3 → 0.9428142795`, `[3,100,100] → 0.8723164502`, `[100,100,100,100,98] → 0.9797939376`. Tests compare `spearman_ceiling(counts)` to `scipy.stats.spearmanr` on constructed perfectly-ordered untied predictions (abs 1e-10) and to these literals (abs 1e-9). v2's `0.9797958971` was the m→∞ limit — wrong.
- AM2 (Adv) Both additivity serializers (`tools/job_c_run.py::additivity_dict`, `release/artifacts.py::_additivity_json`) emit the full new field set including `n_sources` and `rows_identical`; a test asserts their key sets are equal.
- AM3 (Phy) P4 projects in **raw pooled space** (unstandardized log-energies) before any scaler; the ridge pipeline standardizes after projection. JSON records `projection_space: "raw"`.
- AM4 (Phy) P4 report flags cells with any frame below −80 dBFS (count per family) and states the `_EPS` validity bound (F15).
- AM5 (Phy) P4 synthetic test uses a non-degenerate noise floor (≈ −60 dBFS) so the level channel is measured above `_EPS`.
- AM6 (Phy) Key `rho_max_balanced` → `rho_max_balanced_crosscheck`, documented "cross-check only; invalid when level counts are unbalanced".
- AM7 (Phy) Design note records the measured per-rate gain spread: bandlimit/5 +1.14/+0.56/+0.31 dB at 44.1/24/16 kHz (0.83 dB spread), mp3/5 0.71 dB, hiss/5 0.19 dB; the same master's absolute LUFS across rates spans 1.0 LU.
- AM8 (Arch) `variant_semantics()` also feeds candidate metadata in `tools/job_c_run.py:177–178` and `release/artifacts.py:57`.
- AM9 (Arch) P4 was added after the Gate-3 review; Physics reviewed its physics at Gate 4. Gate-6 auditors give it full scrutiny; Gate 8 asks the Architect about it explicitly.
- AM10 (Arch) `seam/real.py` (214 lines) gets a section map when edited.
- AM11 (Arch) Report names follow the tool domain: `tools/ceiling_run.py → reports/ceiling_real.json`, `tools/additivity_run.py → reports/additivity_real.json`, `tools/level_split_run.py → reports/level_split_real.json` (mirrors `geometry_run.py → geometry_real.json`). §1's report names are superseded.
- AM12 (Arch) D2's cost (next full encode pass includes `dac44k:enc`, one DAC corpus pass) is an **owner decision** listed in the design note and named in the DECISIONS correction; the owner may drop `enc` from `REAL_SPECS` before that run.

## Ritual ledger
- G1 cartographers: haiku ×4 (A probes/gate, B seam/DAC, C combos/geometry, D stimulus/claims) — 2026-09-09, F1–F18.
- G2 plan v1 by orchestrator. Ring collapsed to 3 elders (Adversarial, Physics, Architect) by owner request.
- G3 elder ring on v1: NOT APPROVED ×3 (2026-09-09) — 17 distinct findings (§8). Parallel-stream mode abandoned (disjointness false at API level). v1-A3 cut on Physics' derivation.
- G4 re-verdict on v2 (7fed818): APPROVED WITH CONCERNS ×3 → 12 binding amendments AM1–AM12 (§9), plan v2.1. Adversarial caught two wrong ceiling literals in v2 (AM1).
