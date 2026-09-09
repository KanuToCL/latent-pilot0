# DSP and ML science spot check — latent-pilot0

Date: 2026-09-09. Reviewed commit: `294d84382d032b063e0ce1302d9d84f8dd76aa74`.

Scope: signal processing, statistical inference, representation geometry, and the connection to perceptual audio attributes. This is a research handoff, not a software architecture review. No experiment implementation, cached latents, existing reports, or historical decisions were changed.

**Assessment:** this is a useful experimental starting point for asking whether audio artifacts produce repeatable changes in learned representations. Several existing conclusions are stronger than their measurements support. Most urgently, the severity gate has a ceiling problem that defeats even perfectly ordered continuous predictions; DAC's supposed pre-quantization representation is actually quantized; and the dimension-normalized geometry and combination summaries do not establish distinct, additive artifact axes.

Evidence includes the three real JSON reports, source inspection, the installed DAC implementation, primary research, CPU DSP counterexamples, and a small read-only sample of real cached combinations. The full GPU experiments were not rerun. Human audibility, perceptual severity, and quality were not evaluated in this spot check.

## Findings to address before interpreting or extending the results

| ID | Priority | Finding | Main consequence |
| --- | --- | --- | --- |
| S1 | High | Severity margins exceed attainable correlation ceilings | The all-fail gate cannot support the stated negative inference |
| S2 | High | `dac44k:z` is post-quantization | The intended DAC pre/post-quantization comparison has not been run |
| S3 | High | Matching labels across rates does not match the stimuli | Cross-model differences mix representation and input differences |
| S4 | High | Mean concentration divided by `1/sqrt(D)` is not calibrated evidence | Wider or duplicated representations can appear stronger without adding information |
| S5 | High | PCA concentration can rise while artifact directions merge | Persistence in two dimensions does not establish preserved attribute geometry |
| S6 | High | Combination cosine does not establish superposition or identifiability | Near-perfect scores can reflect dominance or centroid averaging |
| S7 | Medium | The energy control is richer than loudness; gate failure is not an absence test | “Non-loudness residual is thin” needs a different experiment |
| S8 | Medium | Probe and pooled-feature directions do not establish causal or perceptual axes | Diagnosis, similarity, and correction need separate validation |
| S9 | Medium | Stochastic degradations reuse a single realization | Speaker holdout does not test artifact-realization generalization |
| S10 | Medium | Several DSP labels and the quality-score identity need tightening | Future training targets can describe a different effect or waveform |

### S1 — A perfectly ordered continuous severity predictor still fails the gate

Locations: [severity.py](../src/pilot0/probes/severity.py#L36), [gate1.py](../src/pilot0/probes/gate1.py#L25), `reports/gate1_real.json`; interpretation in [DECISIONS.md](DECISIONS.md#gate-1--first-real-corpus-run-fail-and-the-failure-has-structure-2026-08-22) and `FIRST_LIGHT.html`.

**Confirmed calculation using the existing report.** A family passes only if the candidate's lower confidence bound is at least 0.80 and exceeds the energy baseline's upper bound by 0.05. Four families must pass.

| Family | Energy SRCC upper bound | Required candidate lower bound | Maximum point SRCC for perfectly ordered, distinct continuous predictions on this test ladder |
| --- | ---: | ---: | ---: |
| Noise | 0.979814 | 1.029814 | 0.979798 |
| Hiss | 0.979726 | 1.029726 | 0.979798 |
| Hum | 0.943813 | 0.993813 | 0.979798 |
| Band-limit | 0.942846 | 0.992846 | 0.942814 |

Noise and hiss require a correlation above 1, which is impossible for any predictor. Hum and band-limit also exceed the ceiling for the continuous, untied predictions normally produced by ridge regression against this repeated ordinal target.

Why the ceiling is below 1: Spearman correlates ranks. The target has `K` severity levels repeated `m` times, while continuous predictions generally have distinct ranks within each level. Even when every between-level ordering is correct, within-level predicted rank variation remains. The maximum is

```text
rho_max = sqrt(((K^2 - 1) * m^2) / ((K*m)^2 - 1)).
```

Here `m=100` test clips per level, `K=5` for most families, and `K=3` for the common band-limit grid. This is an audit derivation, checked numerically with the same SciPy statistic. Spearman's definition and the use of average ranks for ties are documented in [SciPy spearmanr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html) and [rankdata](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rankdata.html).

Passing optimistic zero-width intervals at these perfectly ordered oracle scores into the actual `SeverityResult.n_pass(.8, energy, .05)` yields **3/7**, below the required four. Exact prediction ties matching the target can raise SRCC to 1, so the continuous ceiling is conditional on untied predictions, not a universal bound for every possible estimator. Bootstrap duplicate observations slightly alter rank ceilings within resamples; they do not make these large margins a sensible measure of incremental ordinal information.

This also changes the reading of the actual models: all five EnCodec variants clear the absolute without-clean SRCC lower-bound threshold of 0.80 on **all seven families**. Their recorded 3/7 count is the added baseline-margin result. WavLM l1 and l6 each clear the absolute threshold on five families.

**Handoff:** retain the historical gate outcome and report the ceiling defect beside it. Do not silently redefine the frozen gate and call the replacement preregistered. For a new protocol, separate absolute readability from incremental value, use paired speaker-bootstrap differences, explicitly account for ties, and consider pairwise ordering accuracy across *different* severity levels. Any superiority margin must be feasible near the baseline ceiling. Evaluate an alternative prospectively on new held-out data. The type/mel findings remain separate observations; this issue does not turn a codec into a Gate-1 winner.

### S2 — DAC `z` is the quantized reconstruction of its latent, not the encoder output

Locations: [real.py](../src/pilot0/seam/real.py#L170), `docs/PILOT_A_IMPLEMENTATION.md` §2.3, `configs/models.yaml`.

**Confirmed semantic mismatch.** `_encode_dac` assigns the first return value of `self._model.encode(x)` to variant `z`. The installed DAC source, `dac/model/dac.py:243–247`, computes the encoder output and immediately overwrites it with the quantizer result before returning it. The [official DAC implementation](https://github.com/descriptinc/descript-audio-codec/blob/main/dac/model/dac.py) documents that first output as quantized. In contrast, the EnCodec `z` path reads its encoder directly.

Consequently `dac44k:z` is a full-depth quantized embedding, not the pre-quantization endpoint described in the experimental design. Shape checks cannot distinguish these: both occupy the same channel dimension.

**Handoff:** relabel historical DAC `z` results accurately; add a separately named true encoder-output variant using the encoder output after the same waveform preprocessing. Validate it against that exact tensor, not just shape, and distinguish it from the full-codebook reconstruction. Give the corrected variant a fresh cache identity. Preserve the old bank for reproducing the published-in-repo numbers. No new neural forward pass was needed to establish this mismatch.

### S3 — The common condition grid does not supply equivalent audio across models

Locations: [render.py](../src/pilot0/encode/render.py#L45), [run.py](../src/pilot0/probes/run.py#L45), `degrade/grid.py`, `seam/real.py`.

**Confirmed stimulus confound.** The implementation resamples the clean master first and generates the degradation independently at each model's native rate. Intersecting `(family, severity)` labels removes undefined cells, but does not make the remaining waveforms the same experimental treatment.

For broadband white noise, a fixed full-band SNR distributes noise power over different bandwidths. A low-band clean test signal given nominal 10 dB noise at each rate produced these SNRs when subsequently inspected at 16 kHz:

| Rate where noise was generated | SNR after inspection at 16 kHz |
| --- | ---: |
| 16 kHz | 10.000 dB |
| 24 kHz | 12.002 dB |
| 44.1 kHz | 14.627 dB |

These are synthetic DSP checks, not VCTK model results; the exact offsets depend on the resampler and signal. They demonstrate the treatment mismatch. Hiss has a different above-4-kHz bandwidth at each rate; clipping thresholds change with resampled sample quantiles; MP3's bitrate/mode/bandwidth behavior also depends on rate. WavLM additionally applies its checkpoint's waveform normalization, while the other inputs retain level information differently.

**Handoff:** keep the current experiment as a comparison of complete native-rate pipelines. Add a controlled arm with one declared physical bandwidth and a canonical degraded waveform per source/condition/realization, then resample that same waveform for all models. Measure the achieved effects after model preprocessing and include DSP baselines at each relevant rate. Report within-model and cross-model questions separately. The older plan's degrade-then-resample statement and later L1 decision describe different experimental estimands; neither ordering makes comparability automatic.

### S4 — The dimensional null is for one random pair, not the mean concentration statistic

Locations: [shift.py](../src/pilot0/analysis/shift.py#L122), [geometry_run.py](../tools/geometry_run.py), `DECISIONS.md` RQ2 interpretation.

**Confirmed mathematical limitation.** The exact all-pairs mean cosine calculation itself is correct:

```text
u_i = delta_i / ||delta_i||
C = (||sum_i u_i||^2 - n) / (n*(n-1)).
```

For independent uniform directions in `D` dimensions, `E[u_i dot u_j]=0` and the RMS of one pair's cosine is `1/sqrt(D)`. For the *mean* over pairs, under that idealized independent null,

```text
E[C] = 0
SD[C] = sqrt(2 / (D*n*(n-1))).
```

Thus `C * sqrt(D)` is neither a significance score nor a calibrated model-ranking score. The code acknowledges an isotropic reference, but the report narrative promotes the ratio into evidence that WavLM's geometry is stronger than the mel floor.

Concrete counterexample: duplicate every feature four times. Every cosine is unchanged, and no information is added, but `D` quadruples and the reported “times null” doubles. The executed check preserved `C=0.198114` while the ratio rose from `0.792457` to `1.584913`.

The real median concentrations are `0.652285` for WavLM l1 and `0.599550` for log-mel; their displayed ratios are `29.519` and `11.749` because the dimensions are 2048 and 384. That ratio difference is not evidence of a 2.5-fold information advantage. Standardizing marginal variances does not whiten correlated features or make nominal dimensionality an effective dimension.

**Handoff:** report concentration as an effect size with speaker-level uncertainty and held-out replication. Construct separate, explicitly justified nulls for “any common displacement” and “family-specific displacement”; retain source/severity dependence in resampling. Include empirical covariance/eigenspectrum diagnostics and duplicated-feature controls. A small multiple of the single-pair RMS is also insufficient to declare Mimi's geometry absent: a positive mean across many dependent observations requires its own uncertainty analysis.

### S5 — Two-dimensional concentration can be high while distinct artifact directions collapse

Locations: [shift.py](../src/pilot0/analysis/shift.py#L216), `reports/geometry_real.json`, `FIRST_LIGHT.html` §3.

**Confirmed in the stored report.** The PCA is fit to all standardized degraded representations, not to held-out artifact directions. Within-family concentration after projection answers whether that family's projected shifts align with each other. It does not measure preservation of angles *between* families, displacement energy, or diagnosis accuracy.

WavLM l1's two-dimensional view has these between-family mean-direction cosines:

- Band-limit versus MP3: `0.999871`.
- Hiss versus noise: approximately `0.996`.
- Hum versus band-limit: approximately `0.998`.

Several different artifact directions have become almost parallel despite the high within-family concentration. PCA2 explains `42.19%` of WavLM l1's degraded-row variance, and `56.19%` for EnCodec z. These percentages concern total representation variance, not the fraction of artifact displacement energy preserved.

Seven independently adjustable linear coefficients cannot be uniquely recovered from a general two-dimensional linear displacement: the direction matrix has rank at most two. Seven classes can still occupy different rays in a plane, but this is a different property from seven independent additive axes.

**Handoff:** fit preprocessing and projection on training speakers; evaluate held-out between-family angle distortion, displacement-energy retention, singular values/conditioning of the direction matrix, and diagnostic/mixture recovery. Report the existing low-dimensional plot as visualization and within-family alignment, not preservation of essentially all structure. For comparing representations with different channel counts, use matched-observation similarity methods such as CKA or representational dissimilarity matrices, with controls for common severity and content. [Kornblith et al.](https://proceedings.mlr.press/v97/kornblith19a.html) explain the invariance choices and pitfalls; similarity alone does not establish perceptual equivalence.

### S6 — Near-one combination cosine is consistent with dominance and does not prove superposition

Locations: [additivity.py](../src/pilot0/combos/additivity.py#L67), [grid.py](../src/pilot0/combos/grid.py#L34), `DECISIONS.md` Phase-7 conclusions.

The module correctly notes that cosine measures direction only. The prose conclusion that independent artifacts superpose everywhere, and that axis decomposition is safe, goes beyond it.

**Executed counterexamples:**

```text
delta_a = (100, 0), delta_b = (0, 1), delta_ab = delta_a
cos(delta_ab, delta_a + delta_b) = 0.999950
```

The second artifact is entirely absent. Also, `delta_ab = 0.1*(delta_a+delta_b)` has cosine exactly 1 with the proposed sum, while its relative vector error is 90%.

**Read-only real-cache check:** first available source from each of the first 20 distinct speaker groups in manifest order; raw mean+std features; no model selection or hypothesis-test claim. At severity 3:

| Representation, pair | Centroid cosine to sum | Centroid cosine to first leg alone | Median per-source `||delta_a||/||delta_b||` |
| --- | ---: | ---: | ---: |
| Log-mel, noise+clip | 0.999725 | 0.999980 | 31.40 |
| EnCodec z, noise+clip | 0.999530 | 0.999680 | 18.57 |

Noise alone predicts the centroid direction at least as well in these examples. Conversely, EnCodec hum+MP3 at severity 2 has centroid cosine `0.999090`, but median per-source relative vector residual `0.167694`. Centroid alignment can conceal meaningful individual errors. These samples identify a concern to quantify across the full corpus; they do not establish failure of every combination.

There is also a physical interpretation problem: clipping a noisy signal is nonlinear. Here its percentile threshold is recomputed from the already-noisy input, so the second operation's physical threshold differs from clip-only. MP3 is also nonlinear and signal-adaptive. These are not established non-interacting control pairs. Log-power mel features are nonlinear in waveform addition, so log-mel is not an analytic “linear ceiling.” A low cosine for hiss+band-limit is evidence of directional interaction; the term *sub-additivity* additionally needs a magnitude comparison.

**Handoff:** add per-source norm ratios and normalized residuals, compare against each single-leg predictor, and test recovery of both artifact amounts. Sweep the two severities independently, include both orders, and add a genuinely additive waveform control. For example, generate two independent additive perturbations against the same clean signal with fixed gains. Separate waveform interaction from encoder interaction. Preserve exact source pairing: `additivity()` currently intersects speaker `group`, not source; complete balanced coverage makes the present centroids valid, but partial coverage can select different utterances for different roles. Intersect all four roles by source before speaker-bootstrap inference.

### S7 — The control is spectral and temporal, and a minimum group count is not a power analysis

Locations: [energy.py](../src/pilot0/seam/energy.py), [type_probe.py](../src/pilot0/probes/type_probe.py#L29), [severity.py](../src/pilot0/probes/severity.py#L54), `DECISIONS.md` Gate-1 interpretation.

**Confirmed features:** the control contains log total energy, log energy below 1 kHz, and log energy above 4 kHz, then means and standard deviations over frames. Ratios of its bands describe spectral balance; temporal dispersion describes activity. It is not a single loudness measurement. Matching it does not show that a representation reads only loudness.

Separate-model correlation differences also do not estimate a conditional “non-loudness residual.” Failure to beat a rich baseline by a chosen margin does not establish absence of additional information, especially with S1's ceiling.

The real experiment has 20 held-out speakers. Passing `MIN_TEST_GROUPS=3` establishes eligibility under a guard, not adequate statistical power. Pointwise percentile intervals and failure of a superiority criterion do not establish equivalence. The code uses fixed `C=1` and `alpha=1`; the design's grouped CV/hyperparameter protocol is not implemented in these paths, and the missing winner-confirmation split is already documented. A fixed regularization choice is a valid pilot convention, but does not support a broad claim of optimal probe performance across different dimensions and feature spectra.

**Handoff:** keep the strong spectral control, name it accurately, and add a scalar level control. Test incremental prediction from baseline-plus-latents using train-only fitting and matched held-out examples; inspect level-matched strata or prospectively randomized nuisance gain as a separate experimental arm. Use paired uncertainty for comparisons. State non-attainment of the gate separately from evidence of practical equivalence, and estimate prospective power for the effect that matters.

### S8 — Readability and paired shifts are useful measurements, but not established perceptual or corrective axes

Locations: [geometry.py](../src/pilot0/analysis/geometry.py#L51), [base.py](../src/pilot0/seam/base.py), `FIELD_GUIDE.html`, `PAPER.md`, `plans/2026-08-22-polyglot-listener.idea.md`.

Three distinctions matter for dimension similarity:

1. **Classifier weights are decision boundaries.** For multiclass softmax, adding the same vector to every class weight leaves predictions unchanged but alters weight cosines. Regularization picks a convention, not a physical artifact direction. Correlated features also make discriminative weights differ from generative change patterns. The general distinction is developed by [Haufe et al.](https://pubmed.ncbi.nlm.nih.gov/24239590/). Prefer independently estimated paired shifts for perturbation geometry, and do not infer additivity from orthogonal classifier weights.
2. **Mean+std pooling is a coordinate-dependent nonlinear summary.** Channel means transform linearly under a rotation of frame latents; channel standard deviations generally do not. They retain only the covariance diagonal. Per-channel standardization adds a chosen metric. Thus these results describe the specified pooled/scaled representation, not a coordinate-free property of the original decoder space. Report robustness to basis and pooling choices if making an invariance claim.
3. **A paired shift uses a clean reference.** `f(x_degraded)-f(x_clean)` removes a paired anchor, but content-dependent Jacobians and interactions remain. It does not supply a reference-free decomposition at inference. Furthermore, pooled `2D` statistics cannot simply be inserted into a decoder expecting a sequence of `D`-dimensional latents. Projection/subtraction needs a separate frame-level intervention and decoded-audio validation.

Positive type probing establishes predictability under the tested distribution; a failed small probe does not bound all recoverable information. [Hewitt and Liang](https://aclanthology.org/D19-1275/) motivate probe controls. Mimi's semantic stream is a meaningful comparison condition, but a known-zero negative control is not justified: its observed macro-F1 is 0.402, and the semantic objective does not guarantee independence from artifacts. The [Moshi/Mimi paper](https://arxiv.org/html/2410.00037v2#S3.SS3.SSS2) describes parallel semantic and acoustic quantizers whose outputs both contribute to reconstruction. It does not establish a damage-free semantic channel.

Finally, ordinal DSP settings are not perceptual interval scales. Equal index increments across clipping, cutoff, and SNR do not correspond to equal audible changes. A promising next question is whether within- and between-artifact representational distances predict held-out human similarity/discrimination judgments, with nuisance content and level controlled. A ViSQOL-trained head tests proxy prediction; perceptual claims require independent listening data. The planned separation between that training target and human evaluation is sound.

### S9 — Randomized families currently repeat one random realization

Locations: [noise.py](../src/pilot0/degrade/noise.py#L19), [hiss.py](../src/pilot0/degrade/hiss.py#L15), [dropout.py](../src/pilot0/degrade/dropout.py#L28), `degrade/grid.py`.

All use `seed=0`; the grid does not supply a source/realization seed. Equal-length clips receive the same noise waveform up to gain, and corresponding prefixes repeat across lengths. An executed check with two different clean signals produced injected-noise correlation `0.999999999999993`. Dropout placements likewise repeat for matching length and severity. Hiss and broadband noise also share the underlying seeded white sequence before shaping.

This is useful for controlled dose changes, but insufficient for generalization to the artifact family. It is not evidence that the current probes actually memorized these realizations; their exposure is the issue.

**Handoff:** use logged, independent source/realization seeds, preserve a realization across a source's severity ladder when testing dose, and reserve new realizations for testing. Include held-out generator variants: noise color/recording, filter slope, clipping characteristic, hum frequency/phase, and dropout schedule. The already-documented shared VCTK passages are an additional content restriction, not a substitute for this control.

### S10 — DSP and quality-target details that need explicit measurement

These are narrower issues to pick up before the next corpus or quality run.

| Issue and location | Evidence / implication | Follow-up |
| --- | --- | --- |
| MP3 decode writes default WAV, `degrade/mp3.py:56` | Executing the current FFmpeg command produced `PCM_16`; decoding the same MP3 explicitly to float gave max difference `1.525879e-5`, RMS `8.805652e-6`. Reading PCM16 into float32 does not restore lost precision. The family includes an extra PCM quantization stage; out-of-range decode values can also clip before later headroom attenuation. Actual clipping incidence in VCTK was not established. | Decode with an explicit float PCM format; compare old/new renders and audit overload before rebuilding affected caches. |
| Dropout “measured loss,” `degrade/metrics.py:40` | It counts near-zero output samples, including pre-existing silence. With 20% initial silence and target 5% dropout, the executed check reported 24.0167% loss; newly zeroed non-silent samples were 4.0167%. | Return the injected gain mask; distinguish scheduled loss, newly erased active samples, ramps, and total output silence. Retain requested and achieved labels separately. |
| Forward/backward Butterworth, `degrade/bandlimit.py:27` | `filtfilt` squares the magnitude response and doubles order. The named cutoff has gain −6.0206 dB in the executed check, rather than the one-pass −3 dB corner. Zero phase does not mean absence of temporal smearing or ringing. [SciPy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html) confirms the two-pass operation. | Document the implemented response convention; measure passband/stopband and impulse behavior. Adjust design only if a different effective cutoff was intended. |
| Requested versus achieved severity, `probes/dataset.py` and `probes/severity.py` | The fitted target is the manifest's integer severity, not the cached measured DSP quantity. Therefore measured metadata does not by itself make the current severity experiment a measurement of achieved physical dose. | Retain ordinal results and add physical-target analyses where well-defined; validate within-source ordering across new random realizations. |
| Score identity, `quality/dataset.py:52` and `quality/scores.py` | Scores are keyed only by source/family/severity, while S3 shows that rate-dependent renders differ. One table reused across candidates can attach the same MOS or ViSQOL number to different audio. No real quality table was present in the reviewed reports. | Bind scores to exact waveform identity, processing/rate, and metric configuration, or use one canonical scored stimulus for every candidate. Collect human ratings for the waveform actually evaluated. |
| MOS provenance, `quality/scores.py:137` | Rejecting very high MOS/ViSQOL rank agreement is a heuristic, not proof of copied targets; eight legitimately ordered examples can agree perfectly. Conversely, imperfect agreement does not authenticate human ratings. | Retain human-study provenance and raw ratings; treat rank similarity as a review signal. Use matched coverage for baseline comparisons and account for listener uncertainty. |

## Research basis and a sensible next experiment

The scientific premise has relevant prior art. The cited papers in the plan resolve to real studies, but their demonstrated tasks should bound the claims:

- [Biswas and Villemoes, codec embedding distances](https://arxiv.org/abs/2509.18823): compares distribution-level FAD/MMD relationships with MUSHRA. It supports investigating codec features for evaluation, not an inference that individual latent directions are named perceptual attributes.
- [Kuhlmann et al., speech quality embeddings](https://arxiv.org/abs/2605.21332): directly relevant degradation discrimination/localization using quality-trained representations. A quality-model embedding baseline would help distinguish generator-native convenience from attribute-readout quality.
- [Grundhuber and Habets, acoustic teleportation probing](https://arxiv.org/html/2606.31365v1): studies partitions of an EnCodec-based model trained for acoustic teleportation, not simply stock EnCodec. Its discussion explicitly treats simple-probe leakage as a lower bound on information. This supports the probing methodology while cautioning against interpreting weak probes as absence.
- [Lanzendörfer and Grötschla, objective metrics for neural codecs](https://arxiv.org/abs/2511.19734): evaluates metrics against listening judgments and motivates independent perceptual validation. NISQA also already predicts multiple perceptual quality dimensions; see [Mittag et al.](https://arxiv.org/abs/2104.09494).

For the user's intended “dimension similitude associated with audio artifacts,” the next experiment should distinguish **repeatability within an artifact**, **separation between artifacts**, **correspondence between representations**, and **agreement with perception**. They are four different measurements.

A practical sequence for the later team:

1. Preserve the existing reports and append the S1/S2 corrections. Reanalyze existing cached shifts with per-speaker uncertainty, direction-matrix conditioning, and combination norm/residual/single-leg comparisons.
2. Define a new held-out stimulus arm with canonical bandwidth, source-disjoint content, new artifact realizations, and separately varied combination strengths. Correct DAC extraction and the affected DSP labels before rendering it.
3. Fit scalers, directions, projections, and probe hyperparameters on training data only. On held-out data, compare within-family versus between-family geometry and evaluate whether mixture coefficients can actually be recovered. Benchmark matched DSP and quality-model features.
4. Add a small blinded listening study designed around the target construct: artifact identification/dose ordering or pairwise perceptual similarity, rather than assuming overall MOS measures every attribute. Keep reference-free prediction and paired-reference explanatory analysis separately labeled.

## Reproduction notes

No full training/test suite was needed for this documentation-only audit. The checks were direct counterexamples and read-only report/cache analyses, run with the repository's existing `.venv/Scripts/python.exe -B`, NumPy/SciPy/scikit-learn/soundfile and installed FFmpeg. Supporting scripts and output from this session are retained locally under `C:\Users\KanuTo\latent-pilot0-science-audit`; the essential checks below are self-contained so the handoff does not depend on that folder.

Run the following as Python from the repository root to reproduce the gate ceiling and geometry counterexamples:

```python
import json
import numpy as np
from scipy.stats import spearmanr
from pilot0.analysis.shift import concentration
from pilot0.combos.additivity import _cosine
from pilot0.probes.metrics import Estimate
from pilot0.probes.severity import FamilySeverity, SeverityResult

r = json.load(open('reports/gate1_real.json', encoding='utf-8'))['report']
energy = SeverityResult({
    f: FamilySeverity(**{k: Estimate(**v) for k, v in fs.items()})
    for f, fs in r['energy']['severity']['by_family'].items()
})
oracle = {}
for f, baseline in energy.by_family.items():
    k = 3 if f == 'bandlimit' else 5
    y = np.repeat(np.arange(k), 100)
    perfect_order = np.arange(len(y))
    rho = float(spearmanr(y, perfect_order).statistic)
    e = Estimate(rho, rho, rho)  # deliberately optimistic zero uncertainty
    oracle[f] = FamilySeverity(e, e)
    print(f, rho, baseline.without_clean.hi + .05)
assert SeverityResult(oracle).n_pass(.8, energy, .05) == 3

d = np.random.default_rng(13).normal(size=(100, 16)) + .5
a, b = concentration(d), concentration(np.tile(d, (1, 4)))
assert np.isclose(a.mean_cosine, b.mean_cosine)
assert np.isclose(b.mean_cosine / b.null_scale,
                  2 * a.mean_cosine / a.null_scale)
print(_cosine(np.array([100., 0.]), np.array([100., 1.])))
# 0.9999500037496875, even though the second leg is absent.

g = json.load(open('reports/geometry_real.json', encoding='utf-8'))
w = g['by_candidate']['wavlm:l1']
labels = w['geometry']['labels']
cm = np.asarray(w['shift']['reduction']['by_k']['2']['mean_cosine'])
print(cm[labels.index('bandlimit'), labels.index('mp3')])
# 0.999870680344019: these projected family directions nearly coincide.
```

The archived report files are gitignored; preserve them with any research release. SHA-256 fingerprints at review time:

```text
reports/gate1_real.json
749d7a5d6a62382d04b24c63a61c98dae90dceb6a5d9ae709348509ede42d9a5
reports/geometry_real.json
27e5125200684ce1c66f46c59a4065a79f1be5142cf8b1a3b15b3faaa476cf7d
reports/combos_real.json
633a2cbf16df11652648cd8d79aa562b22da33cb2e7145b8305e8573593d2d60
```

The sampled combo bank was `enc-v1+g2e02854749a2d8e2+dbfa8401da5db+c-1+kc4c74c4dc6a9`. For each sampled source and pair/severity, load clean, each single, and their ordered combination; mean+std pool the cached frame arrays; subtract that source's clean vector; then compute the per-source and centroid quantities defined above. Bootstrap units for a follow-up inference should be speakers, not individual pairwise cosines or the repeated severity rows.
