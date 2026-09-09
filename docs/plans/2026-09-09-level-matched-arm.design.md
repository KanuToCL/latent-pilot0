# Level-matched arm — design note (2026-09-09)

Ships with the S1/S7 remediation as decision D5: the primitive
(`src/pilot0/encode/levelmatch.py`) and this note, **no pipeline threading**. Every
item below marked **OWNER** is a decision this note does not make.

## 1. Estimand

`match_loudness(degraded, clean, sr)` applies one gain so the degraded clip's
BS.1770-4 **integrated** loudness equals its clean partner's. It removes the
**gated broadband level difference from the stimulus**. Tolerance 0.01 LU;
`achieved_lufs` is re-measured and stored, never assumed.

It does **not** "remove the level confound from the control". Measured (F19): after
matching, the energy control's mean-log-total feature is still offset by **+4.6
nats at noise/1 and +6.3 at noise/5**. The cue lives in the pause frames BS.1770's
−70 LKFS absolute and −10 LU relative gates exclude, and the pooled `std`
coordinates are exactly gain-invariant, so no uniform gain touches them at all.
`tests/test_levelmatch.py::test_level_matching_does_not_remove_the_controls_level_cue`
is that fact as a regression test.

## 2. The S7 instrument is the projection split, not this arm

`src/pilot0/probes/level_split.py` splits the control's 6-d pooled vector into its
1-d gain direction `(1,1,1,0,0,0)/√3` and an orthonormal 5-d gain-invariant
complement, re-scoring the same ridge probe on each — from the cached latents, for
free (`tools/level_split_run.py` → `reports/level_split_real.json`). That answers
"how much of the control's severity signal is loudness?" directly; the arm answers
a weaker question, about the stimulus.

## 3. Guards the corpus run will hit

- pyloudnorm 0.2.0 raises `ValueError` below 400 ms (its block size) — `too_short`.
- Digital silence, and anything wholly below the gate, measures `−inf` —
  `degraded_not_measurable` / `clean_not_measurable`.
- A required gain beyond ±`max_abs_gain_db` (default 20 dB) — `gain_out_of_range`.
  The measured per-cell gains are all under 1.2 dB (§4), so this is not a tuning
  knob: it is the assertion that the degraded clip and its clean partner are the
  same recording. A mispaired manifest row asking for 40 dB would otherwise be
  applied silently and the arm would look like it had worked.
- Every refusal returns the audio untouched with `applied=False` and a reason. A
  NaN gain never reaches a waveform, and `gain_db` stays `0.0` on every refusal —
  the refused magnitude is recoverable as `clean_lufs − degraded_lufs`.
- `peak_dbfs_after` is recorded on every path, refusals included, since it describes
  the waveform actually returned. The gain is chosen for loudness, not headroom, so a
  quiet clip with a high crest factor can land above 0 dBFS; the achieved-level audit
  must see that rather than infer it. Digital silence reports `−inf`, not a floor.

## 4. Rate-dependent gain (F20 / AM7)

The required gain depends on the render rate, because the degradation does.
**Measured on one −23 LUFS source (Physics elder, 2026-09-09)** — a single-clip
illustration of the direction and rough size of the effect, not a corpus statistic and
not a tolerance: bandlimit/5 needs **+1.14 / +0.56 / +0.31 dB** at 44.1 / 24 / 16 kHz
(0.83 dB spread); mp3/5 spreads 0.71 dB, hiss/5 0.19 dB; the same master's absolute
LUFS across rates spans 1.0 LU. So the arm is **per-rate**, and the spread must be
recorded per cell in its report — measured over the corpus, since these numbers are not. The real fix for cross-rate comparability is the
canonical-bandwidth arm (S3), out of scope here.

## 5. Threading sites for the follow-up (F14/F16)

Nothing below is changed this round.

- **Arm segment** must be minted into the bank id in `encode/cache.py`
  (`code_version` / `cache_version`, lines 40–52) — otherwise arm cells overwrite
  native cells at the same `cell_id`.
- **Five `cache_version` callers** must pass the arm: `combos/dataset.py:23`,
  `combos/encode.py:62`, `encode/pipeline.py:113`, `ood/run.py:62`,
  `probes/dataset.py:28`.
- **Two `render_cell` callers**: `encode/pipeline.py:85` (the encode path) and
  `encode/headroom.py:79`. Headroom is measured **without any arm** today
  (`headroom.py:60–63, 79, 85–87`), so the arm needs its own headroom scalar or the
  ceiling claim is wrong.
- `tools/encode_monitor.py:68` sums counts across bank versions and would merge the
  two arms into one number.
- The current bank's degrade-semantics segment is **`dbfa8401da5db`**
  (`data/cache/speech/enc-v1+g2e02854749a2d8e2+dbfa8401da5db+c-1+kc4c74c4dc6a9`).
  It is hashed from the source of `degrade/version.py::_WAV_MODULES` and is **not
  asserted by any test** — a change shows up only as a silent full cache miss.
  That is why the primitive lives in `encode/`, not `degrade/` (Architect, §8).
- **Achieved-level audit**: the arm's report must carry, per cell, `gain_db`,
  `achieved_lufs`, `within_tolerance` and the `reason` histogram, plus the per-rate
  gain spread of §4. An arm that cannot show its achieved levels has not matched.

## 6. Costs and open decisions

- **OWNER** Corpus run for the arm: a second full render+encode pass over
  510 sources × 34 cells × 6 encoders. Nothing reuses the existing bank.
- **OWNER** (AM12) `dac44k:enc` is now in `REAL_SPECS` but not in the Gate-1 sweep
  (D2). The next encode pass will therefore include one full DAC corpus pass for
  it. The owner may drop `enc` from `REAL_SPECS` before that run.
- **OWNER** Any Gate-1 protocol change. P2 shows G1b's severity leg is unreachable
  on this design (3 of 7 families feasible, 4 required). Two prospective criteria
  are on the table and neither is adopted here: **absolute readability** (drop the
  energy-margin clause, keep the CI-lower absolute bar, and report the margin
  descriptively) or **incremental value with a feasible margin** (keep the margin
  but set it from the per-family ceiling, so the bar is reachable by construction).
  Both redefine a frozen gate and are the owner's call, prospectively, never on
  this data.
