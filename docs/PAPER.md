# Pilot A — paper draft scaffold

Draft skeleton for the preprint (proposal §9). Prose is deferred; this fixes the
skeleton, the figures, and the pre-registered decision tree so writing is fill-in,
not design. **Numbers land only from the GPU box** (`docs/GPU_BRINGUP.md`); every
figure produced on the Mac carries the FAKE banner and must never enter the paper.

## Title candidates
1. *Reading the Old Wheel in New Coordinates: Classical Degradation Attributes in Neural Audio Codec Latents*
2. *Are Neural Codec Latents Perceptual? Probing Classical Degradations Without a Reference*

## Abstract (skeleton)
Frozen neural audio codecs (EnCodec/DAC/Mimi) and SSL encoders (WavLM) compress speech
into latents optimised for reconstruction, not perception. **Question:** do those
latents linearly expose the classical degradation attributes — type, severity, and
combination geometry — that decades of objective-quality work catalogued, *without a
reference signal*? We build a source-disjoint corpus, a 7×5 degradation grid with
free (measured, not assumed) labels, and read attributes off frozen latents with linear
probes under pre-registered gates. We benchmark a reference-free quality head against
ViSQOL (target) and the NR incumbents NISQA/DNSMOS/UTMOS (evaluated on human MOS, held
apart from the training target). **Contribution:** [readability result] · [codec-vs-SSL-vs-mel
ordering] · [additivity geometry] · a reproducible probe suite + degradation library.

## Sections
1. **Introduction & RQs** — the metrology gap; RQ1 readability, RQ2 geometry/additivity,
   RQ5 severity monotonicity; why reference-free matters for generative audio (F6 motivation).
2. **Related work** — objective quality (PESQ/ViSQOL/POLQA), NR metrics (NISQA/DNSMOS/UTMOS),
   codec/SSL representation probing, degradation taxonomies.
3. **Method**
   - *Corpus & splits* — source-disjoint speaker/track splits (test_frac 0.2); all CIs are
     95 % cluster bootstraps over TEST GROUPS; `MIN_TEST_GROUPS=3` underpowered guard.
   - *Degradation grid* — 7 families × 5 severities + clean; free labels (`degrade/`);
     documented MP3↔band-limit confound; global logged headroom scalar (no per-clip renorm).
   - *Representations* — EnCodec/DAC/Mimi/WavLM variants + log-mel floor + energy control,
     all scored on the SAME common cells (apples-to-apples).
   - *Probes & gates* — linear type/severity probes, MLP−linear nonlinearity gap, held-out
     severity interpolation; the pre-registered gates below.
   - *Quality head* — ridge on frozen pooled latents → ViSQOL, no reference; G2b scored on
     human MOS so a ViSQOL-trained head is never graded on ViSQOL (§8 rig avoidance).
4. **Results** — figures F1–F6 (below) + the gate decision tree outcome.
5. **Limitations** — pilot scope (speech arm can stand alone, §7); descriptive/in-sample
   geometry; fake-seam caveat (this repo's Mac numbers are plumbing, the GPU box is the
   science); MP3↔band-limit confound; NR incumbents are strong but imperfect MOS proxies.
6. **Bridge to the programme** (§10) — which paragraph of the doctoral proposal each gate
   outcome rewrites; the degradation library → Pilot B stimulus generator; the quality head
   → WP3 rater embryo.

## Figures (each rendered by `make reproduce-figures` from a JSON artifact)
| Fig | Shows | Artifact |
|-----|-------|----------|
| F1 | readability heatmap (representation × {type, per-family severity}) | `reports/analysis/heatmap.json` |
| F2 | held-out severity monotonicity — mean predicted vs true, per family | `reports/analysis/monotonicity.json` |
| F3 | attribute direction cosine matrix + pairwise additivity (combo vs Δa+Δb) | `reports/analysis/cosine.json` + `reports/combos/additivity.json` |
| F4 | quality head vs NR baselines (per-family MOS bars) + head-vs-ViSQOL scatter | `reports/quality/gate2.json` |
| F5 | codec vs SSL vs log-mel floor summary (type F1 + mean severity SRCC) | `reports/analysis/heatmap.json` |
| F6 | OOD teaser — NR-metric disagreement, grid vs generative material | `reports/ood/teaser.json` |

## Pre-registered gates (§8, set 2026-07 — do not move)
**Gate 1** (after Phase 4, speech arm, best single representation)
- G1a linear-probe type macro-F1 ≥ 0.85
- G1b severity SRCC ≥ 0.80 on ≥ 4/7 families
- G1c best codec beats the log-mel floor by ≥ 5 macro-F1 points (else NULL-INTERESTING)

**Gate 2** (after Phase 6)
- G2a quality head SRCC vs ViSQOL ≥ 0.85 pooled (CI-lower gated, degraded cells)
- G2b quality head ≥ best NR baseline (NISQA/DNSMOS/UTMOS) on ≥ 5/7 families — implemented
  as a per-family paired-difference test scored on human MOS, where "≥ best" ≡ "paired-beats every"

**Decision tree — every branch ends in a paper (that is the design):**
- G1 pass + G2 pass → **GO**: banker spoke feasible in codec latents; preprint claims the full result.
- G1 pass, codec ≪ WavLM (gap > 10 F1) → **PIVOT-SSL**: substrate changes; WP2 shifts to an SSL front-end.
- G1c fail (mel floor matches everything) → **NULL-INTERESTING**: publish the negative + geometry; WP2 redesigns around learned-from-scratch perceptual encoders.
- G2 fail only → readable-but-uncalibratable: publish readability + geometry; the quality head becomes a WP3 problem.

## Venue calendar
arXiv preprint (Feb 2027, **FIRST regardless of venue**) → Interspeech 2027 (≈ Mar) →
WASPAA 2027 (≈ Apr) → ICASSP 2028 (≈ Sep 2027) backup.

## Open release
Probe suite + manifests + degradation library. Audio is **not** redistributed —
regeneration scripts instead (`make` targets + the degradation library reproduce every cell).
