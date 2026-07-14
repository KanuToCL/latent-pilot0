# pilot0 — Pilot A

Reading classical audio-degradation attributes (type + severity) out of **frozen
neural-codec latents** with small probes, without a reference — the feasibility
pilot for the doctoral programme. Full design in
[`docs/PILOT_A_IMPLEMENTATION.md`](docs/PILOT_A_IMPLEMENTATION.md).

## The two-machine split

The GPU box (real codecs) is packed until the move to Spain, so **everything is
built and tested here on the Mac against a fake encoder seam**; the real
backends are written to be **drop-in** at bring-up. See
[`docs/GPU_BRINGUP.md`](docs/GPU_BRINGUP.md).

- **Mac dev** (this machine, no torch): `Encoder` seam + `FakeEncoder` (numpy).
- **GPU box** (Spain): the same code, `pip install -e ".[gpu]"`, real backends.

Everything on the Mac runs a **FAKE encoder + synthetic scores** so the pipeline is
verified torch-free. Nothing produced here is a scientific result — every demo prints a
`⚠ FAKE` banner, every figure stamps it, `reports/provenance.json` records `"fake": true`.
The real numbers come from the GPU box.

## Quickstart (Mac)

```bash
make setup              # .venv + editable install (torch-free, incl. figure extra)
make test               # full suite
make reproduce-figures  # one synth corpus → F1–F6 JSON + PNG under reports/
```

## Per-phase demos (all fake-seam plumbing checks)

```bash
make test-degrade    # P1  measured-vs-target degradation grid
make manifest-demo   # P2  preflight → manifest → source-disjoint splits
make encode-demo     # P3  render → headroom → resume-safe latent cache
make probe-demo      # P4  Gate-1 table (type/severity probes vs log-mel floor)
make analyze         # P5  readability heatmap + cosine matrix + monotonicity
make quality-demo    # P6  quality head + pre-registered Gate 2
make combos-demo     # P7  pairwise additivity + zero-shot type-probe transfer
make ood-demo        # P7  OOD teaser (NR-metric disagreement)
```

## Layout

```
src/pilot0/
  seam/        encoder contract + fake (Mac) + real (GPU box) + registry
  audio/       synthetic clips + off-manifold (OOD) textures
  corpus/      preflight, manifest, source-disjoint splits
  degrade/     7×5 degradation grid with free (measured) labels
  encode/      render → headroom → content-addressed latent cache
  probes/      linear type/severity probes, interpolation, Gate 1
  analysis/    full representation matrix → readability/geometry/monotonicity
  quality/     no-reference quality head + pre-registered Gate 2
  combos/      pairwise combos → additivity + transfer
  ood/         off-manifold teaser (metric disagreement)
  release/     Phase-8 artifacts + F1–F6 figures + reproduce
docs/          implementation plan, decisions, paper scaffold, release + GPU runbooks
tests/
```

Status: **Phases 0–8 complete on the fake seam.** Awaiting the GPU box for the
scientific run (`docs/GPU_BRINGUP.md`); paper scaffold in `docs/PAPER.md`.
