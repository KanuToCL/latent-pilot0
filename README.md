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

## Quickstart (Mac)

```bash
make setup     # .venv + editable install (torch-free)
make test      # seam + smoke contract tests
make smoke     # encode 10 synthetic clips through the fake stand-ins → reports/smoke.json
```

## Layout

```
src/pilot0/
  seam/        encoder contract (base) + fake (Mac) + real (GPU box) + registry
  audio/       synthetic clips (Phase 0); real corpora arrive Phase 2
  smoke.py     Phase-0 acceptance: encode 10 clips, log shapes
configs/       pinned model checkpoints
docs/          implementation plan, decisions, spinoff note, GPU bring-up runbook
tests/
```

Status: **Phase 0** (environment & seam).
