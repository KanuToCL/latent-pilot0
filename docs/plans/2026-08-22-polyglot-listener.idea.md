# Idea: The Polyglot Listener — confidence-aware routing metrics

**Status:** idea record, 2026-08-22. Born from the Monet-principle discussion
(see `docs/ENCODER_ANATOMY.html` §6). Candidate thesis anchors; Pilot A is
unaffected and keeps running.

## The insight chain (as it happened)

1. **No privileged basis.** A codec's latent axes are an accident of
   initialization — rotate the space, let the decoder absorb it, nothing
   changes. Meaning lives in geometry (directions, distances), never in
   coordinates. → justifies the shift-vector program.
2. **The vocabulary is the training world (Monet principle).** An encoder
   describes every input in the only language it has. On-manifold inputs get
   faithful shorthand; off-manifold inputs get rendered as *the nearest
   expressible natural sound* — confidently, with no error flag.
3. **Brains do the same — and it's documented.** Categorical perception;
   Kuhl's perceptual magnet (adult Japanese listeners cannot hear /r/–/l/ —
   the native prior collapsed that axis in infancy); Werker's narrowing
   (infants lose universal phoneme discrimination by ~12 months). "Hearing
   Swahili through English ears" is real science, not metaphor.
4. **But brains have three tricks codecs lack:**
   - **Routing** — a polyglot *switches priors* ("sounds Russian… no,
     Ukrainian") instead of diluting one model over all languages.
     ML echo: mixture-of-experts + a gating network.
   - **Evidence accumulation** — "let me hear more": a posterior over
     hypotheses updated online. Codecs are one-pass feedforward; the brain
     runs analysis-by-synthesis (predict → compare → re-encode the error;
     predictive coding).
   - **Metacognition** — hearing a language you can't even *place*, you know
     you don't know it. A codec never says "I don't speak this"; it always
     outputs its best Monet. There is no unfamiliarity channel.

## Four candidate thesis anchors (any one could carry the whole thesis)

### A. Damage-axis geometry (the current Pilot A path)
Frozen-latent shift vectors as distortion axes: direction = diagnosis,
magnitude = dose; decomposition under mixtures (RQ2) as the industrial
breaking point. Cheapest existence proof; already running.

### B. The routing committee (mixture of damage experts)
Per-family expert probes/heads + a gate. The diagnosis is the *routing
decision*; ambiguity surfaces as routing entropy instead of a wrong confident
label ("hum… wait, low-bitrate mp3"). Builds directly on A's axes.
Prior art to survey: MoE in speech (Switch/GShard lineage), ensemble
diagnostics, NISQA's dimension heads as a fixed 4-expert committee.

### C. The unfamiliarity channel (know-what-you-don't-know)
An explicit "distance from the natural-audio manifold" signal — router
entropy, reconstruction error, latent likelihood — as a first-class output
alongside quality. This is the missing organ for **generated audio**, whose
failure modes are precisely the sounds no expert claims. Turns the Monet
blind spot from a limitation into the measurand.
Prior art to survey: OOD detection in audio, uncertainty estimation for MOS
predictors, SpeechLMScore-style likelihood scoring.

### D. The iterative listener (analysis-by-synthesis)
Test-time refinement: encode → predict → compare → re-listen. A metric that
spends more compute when uncertain, like a human leaning closer. Freshest
scientifically, furthest from current tooling.

## Recommended starting point

**Stay on A** — it is running tonight and every other anchor consumes its
output: B routes over A's axes, C thresholds A's residuals, D iterates A's
encoder. **C is the natural second pilot**: it reuses Phase 7b's OOD textures
+ the additivity machinery, it is what the generative-audio endgame actually
requires, and it is the least occupied corner of the prior art (everyone
scores quality; almost nobody scores *unfamiliarity* as a product).
Decision point after Gate 1 + shift-vector geometry land: if axes are real,
B/C become concrete; if axes fail, C survives (unfamiliarity needs no axes)
and becomes the lead candidate.

## Guardrails

- **IP:** same caveat as `SPINOFF.md` — settle employment invention-assignment
  before public posts that claim these ideas (LinkedIn teaser ≠ spec).
- **Honesty:** all of this inherits the gates discipline — pre-register before
  data contact, baselines before claims, and the unfamiliarity channel itself
  must be validated against *known* OOD before it judges generated audio.
- **Perceptual-science companion:** `docs/PERCEPTION_LAB.html` (explainer +
  demos) and the perception track in `docs/MATH_BRUSHUP.html`.
