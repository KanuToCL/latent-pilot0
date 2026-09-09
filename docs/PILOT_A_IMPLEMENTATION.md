# PILOT A — IMPLEMENTATION PLAN
## The self-sufficient execution playbook

*Companion to `PILOT_A_ORIENTATION.md` (the constitution — its Failure Statements bind everything here). This file assumes the researcher may execute alone, without an AI assistant. Every phase states: goal, tasks, artifacts, acceptance check, and pitfalls. Gates are pre-registered in §8 and must not be moved after data contact.*

---

## 1. RESEARCH QUESTIONS

- **RQ1 Readability.** Which classical degradation attributes (type, severity) are decodable by small probes from frozen codec latents — and how does this vary by codec family, quantization depth, tokens vs continuous, and pooling?
- **RQ2 Geometry.** Do attributes correspond to approximately linear, near-orthogonal directions? Do directions add under combined degradations?
- **RQ3 Quality head.** Can a small head on frozen latents track full-reference quality (ViSQOL; PESQ on the 16 kHz speech arm) without the reference — and how does it compare to no-reference incumbents (NISQA, DNSMOS, UTMOS)?
- **RQ4 Control.** Codec latents vs general SSL encoders (WavLM layer sweep) vs the log-mel/statistics floor: who carries the information, and how cheaply?
- **RQ5 Monotonicity.** Is parametric severity preserved ordinally (held-out-severity interpolation)?

## 2. EXPERIMENTAL DESIGN

### 2.1 Source material (licenses logged in `configs/licenses.yaml` before download)
| Arm | Dataset | Rate | Role |
|---|---|---|---|
| Speech 48k | VCTK (CC BY 4.0) | 48 kHz | primary arm |
| Speech 16k | LibriSpeech test-clean/other (CC BY 4.0) | 16 kHz | PESQ-compatible arm |
| Music | MUSDB18-HQ (research) or FMA-small (CC) | 44.1 kHz | generality arm |
| Sound | FSD50K subset (per-clip CC) | 44.1 kHz | generality arm |

v0 scale: 500 sources per arm, 4–10 s clips. Full: 2000 (speech arm). Masters loudness-normalized to −23 LUFS (BS.1770) BEFORE degradation; never renormalized after (Failure Statement 3).

### 2.2 Degradation grid (each function unit-tested; parameters logged per clip)
| Family | Parameter | Severity levels (5) |
|---|---|---|
| Broadband noise (white + pink variants) | SNR dB | 30, 20, 10, 5, 0 |
| Hiss (HF-shaped noise; high-pass, not a literal shelf) | SNR dB above 4 kHz | 35, 25, 15, 10, 5 |
| Mains hum (50 Hz + 3 harmonics) | hum-to-signal dB | −40, −30, −20, −12, −6 |
| Clipping (hard; soft variant logged) | target %-samples-clipped (L2; supersedes "gain over full-scale") | 0.1, 0.5, 1, 3, 8 |
| Band-limiting (lowpass) | cutoff kHz | 12, 8, 6, 4, 3.4 |
| Low-bitrate transcode (MP3 via ffmpeg/lame) | kbps | 64, 48, 32, 24, 16 |
| Dropouts (20 ms zeroed bursts, 5 ms ramps) | % frames lost | 0.5, 1, 2, 5, 10 |

Plus: clean condition; and a **pairwise-combination subset** (noise×clipping, hiss×lowpass, hum×mp3; mid severities only; ~10% extra volume) for the additivity test (RQ2). Speech arm: 500 × 36 = 18k clips (v0). Rub-and-buzz-style harmonic rattle is explicitly deferred to v1.

### 2.3 Representations (frozen; checkpoint ids pinned in config)
| Model | Access | Variants probed |
|---|---|---|
| EnCodec 24k | transformers | continuous pre-quant z; RVQ embeddings summed at depths {1,2,4,8} |
| EnCodec 48k stereo | transformers/pip | music arm |
| DAC 44.1k | descript-audio-codec pip | `z` = QUANTIZED (`DAC.encode()` returns the quantizer output); `enc` = continuous pre-quant encoder output; depths {1,2,4,8} |
| Mimi | transformers | semantic tokens vs acoustic tokens separately — the split is a headline sub-experiment |
| WavLM-Large | transformers | layers {1, 6, 12, 18, 24} |
| log-mel (+Δ, mean/std) | torchaudio | the cheap floor (mandatory in every table) |

Pooling: mean+std over frames (primary); max (secondary); frame-level probes for dropouts only. Audio resampled per model's native rate AFTER degradation on the 48 kHz master. Cache policy: pooled stats for everything (small); full-frame fp16 for EnCodec-24k only; others encoded on the fly if disk-tight (log the decision in DECISIONS.md).

### 2.4 Probes & metrics
- **Type classification:** multinomial logistic regression (primary), 2-layer MLP-256 (secondary). Metrics: macro-F1, balanced accuracy, per-family one-vs-rest AUROC. The MLP−linear gap is reported as the nonlinearity measure.
- **Severity:** per-family ridge regression on severity index and on the physical parameter. Metrics: Spearman (primary), Pearson, RMSE. Interpolation split: train on severities {1,2,4,5}, test on {3}.
- **Quality head:** ridge/MLP on latents → ViSQOL score of (degraded vs clean); PESQ target on the 16 kHz arm only (Failure Statement 15). Compare against NISQA, DNSMOS, UTMOS run directly on the degraded audio. Metrics: LCC/SRCC per family and pooled; system-level correlation on per-condition means.
- **Geometry:** cosine matrix of linear-probe weight vectors; difference-of-class-means directions; PCA of condition centroids; additivity: cos( z̄(a+b)−z̄(clean), Δa+Δb ) on the pairwise subset, plus zero-shot probe transfer to combos.

### 2.5 Splits & statistics
- Source-disjoint everywhere (speaker/track never crosses train/test); split deterministic from manifest hash.
- 3 probe seeds × 5-fold CV inside train; single held-out test.
- Headline numbers: mean ± 95% bootstrap CI over sources (1000 resamples).
- Per-arm reporting (speech/music/sound); pooled only as a secondary view.

## 3. PHASED EXECUTION (evenings, ~8–10 h/week)

**Phase 0 — Environment & smoke (W1).** Repo skeleton per Orientation §4; pinned env (python 3.11, torch, torchaudio, transformers, descript-audio-codec, ffmpeg, pyloudnorm, scikit-learn); `make setup`. Acceptance: encode 10 clips through EnCodec and WavLM; shapes logged.
**Phase 1 — Degradation library (W1–2).** One module per family + unit tests asserting analytic effect (Orientation FS-1). Acceptance: `make test-degrade` prints measured-vs-target table, all within tolerance. Pitfall: clipping severity defined on the loudness-normalized master, not per-clip peaks.
**Phase 2 — Manifest & preflight (W2).** Build manifest (hashes, params, splits); loudness preflight. Acceptance: manifest row count = sources × conditions; split-leakage test green.
**Phase 3 — Encode & cache (W3).** Batch encoding, resume-safe, fp16 for continuous only (FS-11). Acceptance: cache-integrity test (random 20 clips re-encoded, max abs diff < 1e-3 fp16); storage within budget (§6).
**Phase 4 — First readability pass (W4).** Linear probes, EnCodec-24k + WavLM + mel floor, speech arm, type + severity. Produce heatmap v0. **GATE 1 evaluated here (§8).**
**Phase 5 — Full representation matrix (W5–6).** Add DAC, Mimi (semantic/acoustic split), EnCodec depths, layer sweep, MLP probes, frame-level dropout probe, geometry analyses. Acceptance: heatmap v1 + cosine matrix + monotonicity curves regenerate via `make analyze`.
**Phase 6 — Quality head & baselines (W7).** ViSQOL pipeline (see §7 build risk), PESQ 16 kHz arm, NISQA/DNSMOS/UTMOS baseline runs, correlation tables. **GATE 2 evaluated here.**
**Phase 7 — Combos & OOD teaser (W8).** Pairwise additivity; small out-of-grid set (one TTS system + one music-gen system outputs, ~200 clips): quality-head behavior where PESQ/ViSQOL are undefined/disagree — motivation figure only, no claims.
**Phase 8 — Write & release (W9–10).** Paper draft per §9; `make reproduce-figures` on clean checkout; anonymized repo tag; arXiv preprint.

## 4. RELATED WORK & DIFFERENTIATION (cite these; the race is real)

- 2509.18823 — codec-embedding distances vs FAD/MUSHRA: distance-level, no attribute taxonomy. *We add: type + severity + direction geometry.*
- 2605.21332 — degradation classification from quality-model embeddings: not codec latents. *We add: the generator-native substrate + severity + quality head.*
- 2606.31365 — probing room parameters from EnCodec latents: the probing pattern, different targets. *We adopt their protocol shape; new target family.*
- 2511.19734 — objective metrics evaluated on neural-codec artifacts: motivates our OOD framing.
- Codec-robustness line (Interspeech 2025): degradations *through* codecs measured downstream. *We read degradations *from* latents.*
- Context only: MOS-RMBench (2510.00743), PrefSQA (2606.19597) — the preference pivot this feeds later.

## 5. BASELINE MATRIX (mandatory columns in the headline table)
log-mel floor · WavLM best layer · best codec representation · NISQA · DNSMOS · UTMOS · (oracle row: ViSQOL/PESQ themselves).

## 6. COMPUTE & STORAGE BUDGET
Single 12–24 GB GPU. Encoding: ~18k × 6 s clips ≈ a few GPU-hours per model. Probes: CPU-minutes. Storage: audio masters+degraded ≈ 25 GB (flac); pooled caches < 5 GB; EnCodec-24k full-frame fp16 ≈ 15–20 GB. Hard ceiling 80 GB; beyond it, on-the-fly encoding (DECISIONS.md entry required).

## 7. RISK REGISTER & DEBUG PLAYBOOK
- **ViSQOL build pain (C++/bazel):** attempt early (Phase 0 stretch). Fallback: PESQ-16k arm as sole reference target + report STOI as auxiliary; state the limitation. Do NOT substitute an unofficial reimplementation without validation against published scores.
- **Probe at chance:** first suspect label misalignment — join manifest by content hash, spot-listen 10 clips per family.
- **Probe suspiciously perfect (>0.99):** leakage — check source-disjointness, and that degradation params aren't encoded in cache filenames consumed as features.
- **fp16 NaNs/overflow:** clamp pre-cast; keep code indices integer (FS-11).
- **Class imbalance after preflight drops:** stratified resampling of sources, never of conditions.
- **MP3 encoder unavailable:** ffmpeg libmp3lame check in `make setup`.
- **Time overrun:** the protected core is Phases 1–4 + Gate 1. Everything after Phase 5 can shrink to "EnCodec vs WavLM vs mel, speech arm only" and still publish.

## 8. PRE-REGISTERED GATES (set 2026-07; do not move — Orientation FS-6)

**Gate 1 (after Phase 4, speech arm, best single representation):**
- G1a: linear-probe type macro-F1 ≥ 0.85.
- G1b: severity SRCC ≥ 0.80 on ≥ 4 of 7 families.
- G1c: best codec representation beats the log-mel floor by ≥ 5 macro-F1 points (else branch NULL-INTERESTING).

**Gate 2 (after Phase 6):**
- G2a: quality head SRCC vs ViSQOL ≥ 0.85 pooled (speech arm).
- G2b: quality head ≥ best no-reference baseline (NISQA/DNSMOS/UTMOS) on ≥ 5 of 7 families.

**Decision tree:**
- G1 pass + G2 pass → **GO:** banker spoke feasible in codec latents; proceed to WP2 framing; preprint claims the full result.
- G1 pass, codec ≪ WavLM (gap > 10 F1) → **PIVOT-SSL:** transducer premise holds, substrate changes; preprint reports the comparison honestly; WP2 shifts to SSL front-end with codec-coupling as open question.
- G1c fail (mel floor matches everything) → **NULL-INTERESTING:** deep latents add nothing over spectral statistics for classical attributes; publish the negative with the geometry analysis; WP2 redesigns around learned-from-scratch perceptual encoders.
- G2 fail only → readable-but-uncalibratable; publish readability + geometry; quality-head becomes a WP3 problem with more data.
Every branch ends in a paper. That is the design.

## 9. PAPER PLAN
- **Title candidates:** "Reading the Old Wheel in New Coordinates: Classical Degradation Attributes in Neural Audio Codec Latents" / "Are Neural Codec Latents Perceptual? Probing Classical Degradations Without a Reference."
- **Figures:** F1 readability heatmap (representation × attribute); F2 severity monotonicity curves; F3 direction cosine matrix + additivity; F4 quality-head vs baselines (per-family bars + pooled scatter); F5 codec-vs-SSL-vs-mel summary; F6 OOD teaser (metric disagreement on generative material).
- **Venue calendar:** arXiv preprint target Feb 2027 → Interspeech 2027 (deadline ≈ March 2027) → WASPAA 2027 (≈ April) → ICASSP 2028 (≈ Sept 2027) as backup. Preprint FIRST regardless of venue (the race, §4).
- **Open release:** probe suite + manifests + degradation library (audio not redistributed; regeneration scripts instead).

## 10. BRIDGE TO THE PROGRAMME
Gate outcomes rewrite exactly one paragraph of the doctoral proposal (Pilot A row + Risks). Readability results select the codec family WP2 adapts. The degradation library becomes Pilot B's stimulus generator for the forced-coding probes. The quality head is the embryo of the WP3 rater. If all gates fail: the thesis's metrology legs (lexicon, corpus, equivalence protocol) are untouched — say so out loud and keep going.
