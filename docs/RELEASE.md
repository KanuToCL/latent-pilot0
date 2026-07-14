# Release checklist — Pilot A

Anonymized open release (proposal §9): probe suite + manifests + degradation library.
**Audio is never redistributed** — regeneration scripts stand in.

## What is real vs FAKE
The Mac dev tree runs a FAKE encoder + synthetic scores end to end so the whole
pipeline is verified torch-free before the GPU box exists. **Nothing produced on the
Mac is a scientific result.** Every demo prints a `⚠ FAKE` banner; every figure stamps
it; `reports/provenance.json` records `"fake": true`. The scientific run happens on the
RTX box after `make setup-gpu` (`docs/GPU_BRINGUP.md`) — the real backends drop into the
same seam and the same commands emit the real artifacts.

## Reproduce the figures
```
make setup              # Mac: torch-free, installs the figures extra
make reproduce-figures  # one synth corpus → F1–F6 JSON + PNG under reports/
```
Outputs: `reports/{analysis,quality,combos,ood}/*.json`, `reports/figures/F1–F6.png`,
`reports/provenance.json`. On the Mac these carry the FAKE banner; on the GPU box the
same target emits the real figures.

## Pre-release gate (must pass)
There is no CI runner yet, so this checklist IS the gate — run it by hand before tagging.
- [ ] `make test` green (full suite)
- [ ] `PILOT0_SLOW=1 make test` green (runs the automated end-to-end release smoke that the
      default suite skips — the only test that exercises the real orchestration)
- [ ] `make reproduce-figures` produces all six PNGs + provenance
- [ ] every figure carries the FAKE banner (Mac) / no banner only on real GPU-box runs
- [ ] `reports/provenance.json` records the git SHA of the tagged commit
- [ ] no audio in the tree or the release archive (manifests + regeneration scripts only)
- [ ] `docs/PAPER.md` gate decision tree matches `docs/PILOT_A_IMPLEMENTATION.md` §8

## Anonymized repo tag
- [ ] strip author identity from README/manifests/commit trailers for the review copy
- [ ] `git tag -a pilotA-preprint -m "Pilot A preprint artifacts"`
- [ ] archive the tag (source only, no audio) for the anonymized supplement

## arXiv preprint
- [ ] figures F1–F6 from the **GPU-box** run (real backends), FAKE banner absent
- [ ] gate outcome + branch (GO / PIVOT-SSL / NULL-INTERESTING / G2-fail) stated in the abstract
- [ ] limitations name the MP3↔band-limit confound and the pilot's speech-arm scope
- [ ] link the open release (probe suite + degradation library); confirm no audio redistributed
- [ ] preprint posted FIRST, before any venue submission (the race, §4)
