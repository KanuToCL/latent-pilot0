"""Phase-8 figures F1–F6, rendered from the JSON artifacts (`release.artifacts`) — a
clean JSON→PNG boundary so the plotting never re-runs the pipeline and is testable on
fixtures alone. Every figure stamps the FAKE banner in the frame: on the plumbing seam
these are shapes-and-wiring, not results, and a PNG that escapes the repo must say so.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..provenance import BANNER
from .artifacts import CANDIDATES

try:
    import matplotlib
    matplotlib.use("Agg")  # headless: no display on the Mac dev box or CI
    import matplotlib.pyplot as plt
except ImportError as e:  # pragma: no cover - exercised only without the extra
    raise ImportError("Figure rendering needs matplotlib — `pip install 'pilot0[figures]'`") from e

_PRIMARY = f"{CANDIDATES[0][0]}/{CANDIDATES[0][1]}"
FIGURES = ("F1", "F2", "F3", "F4", "F5", "F6")


def _f(x) -> float:
    return float("nan") if x is None else float(x)


def _load(reports_dir: Path) -> dict:
    def read(p):
        return json.loads((reports_dir / p).read_text())
    return {
        "heatmap": read("analysis/heatmap.json"), "cosine": read("analysis/cosine.json"),
        "monotonicity": read("analysis/monotonicity.json"), "gate2": read("quality/gate2.json"),
        "additivity": read("combos/additivity.json"), "ood": read("ood/teaser.json"),
    }


def _finish(fig, title: str) -> None:
    fig.suptitle(title, fontsize=12, y=0.99)
    fig.text(0.5, 0.005, BANNER, ha="center", va="bottom", fontsize=8, color="crimson")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))


def _row_label(r: dict) -> str:
    return f"{r['name']}/{r['variant']}"


def f1_readability(data: dict):
    rows = data["heatmap"]
    fams = list(rows[0]["severity_srcc"].keys())
    cols = ["type", *fams]
    M = np.full((len(rows), len(cols)), np.nan)
    for i, r in enumerate(rows):
        M[i, 0] = _f(r["type_macro_f1"][0])
        for j, fam in enumerate(fams):
            M[i, 1 + j] = _f(r["severity_srcc"][fam][0])

    fig, ax = plt.subplots(figsize=(1.4 * len(cols) + 2, 0.7 * len(rows) + 2))
    # vmin=-1: a negative severity SRCC (anti-correlated, i.e. broken) lands at the dark
    # end, distinct from a ~0 null-signal cell, rather than clamping to the same floor
    im = ax.imshow(M, cmap="viridis", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(rows)), [_row_label(r) for r in rows])
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = M[i, j]
            # white on the dark (viridis low) half, grey on the transparent NaN cell
            txt, col = ("—", "grey") if np.isnan(v) else (f"{v:.2f}", "white" if v < 0.0 else "black")
            ax.text(j, i, txt, ha="center", va="center", color=col, fontsize=8)
    fig.colorbar(im, ax=ax, label="macro-F1 / severity SRCC")
    _finish(fig, "F1 — readability heatmap (representation × attribute)")
    return fig


def f2_monotonicity(data: dict):
    mono = data["monotonicity"]
    keys = list(mono.keys())
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4.5), squeeze=False)
    for ax, key in zip(axes[0], keys):
        dropped = []
        for fam, fi in mono[key].items():
            curve = fi["curve"]
            xs = [lvl for lvl in range(1, 6) if curve.get(str(lvl)) is not None]
            if not xs:  # not evaluable — nothing honest to draw, don't imply a curve
                dropped.append(fam)
                continue
            ys = [_f(curve[str(lvl)]) for lvl in xs]
            interp = bool(fi.get("interpolates"))  # solid = passes the gate, dashed = does not
            ax.plot(xs, ys, marker="o", label=fam if interp else f"{fam} (n/i)",
                    linestyle="-" if interp else "--", alpha=1.0 if interp else 0.45)
        ax.plot([1, 5], [1, 5], "k:", alpha=0.4, label="identity")
        if dropped:
            ax.text(0.02, 0.98, "n/a: " + ", ".join(dropped), transform=ax.transAxes,
                    fontsize=6, va="top", color="grey")
        ax.set_xlabel("true severity"); ax.set_ylabel("mean predicted severity")
        ax.set_title(key, fontsize=10); ax.set_xticks(range(1, 6))
        ax.legend(fontsize=7, ncol=2)
    _finish(fig, "F2 — held-out severity monotonicity (mean predicted vs true)")
    return fig


def f3_geometry(data: dict):
    key = _PRIMARY if _PRIMARY in data["cosine"] else next(iter(data["cosine"]))
    geo, add = data["cosine"][key], data["additivity"][key]
    labels = geo["labels"]
    M = np.array(geo["mean_cosine"], dtype=float)

    fig, (axm, axb) = plt.subplots(1, 2, figsize=(13, 5.5),
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    im = axm.imshow(M, cmap="coolwarm", vmin=-1, vmax=1)
    axm.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axm.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            axm.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                     color="black" if abs(M[i, j]) < 0.6 else "white")
    axm.set_title(f"class-mean direction cosine — {key}", fontsize=10)
    fig.colorbar(im, ax=axm, label="cosine", fraction=0.046)

    cells = sorted(add["by_cell"].values(), key=lambda c: (c["pair"], c["severity"]))
    names = [f"{c['pair']}\n@sev{c['severity']}" for c in cells]
    pts = np.array([_f(c["raw"][0]) for c in cells])
    lo = np.array([_f(c["raw"][1]) for c in cells])
    hi = np.array([_f(c["raw"][2]) for c in cells])
    err = np.abs(np.vstack([pts - lo, hi - pts]))
    x = np.arange(len(cells))
    axb.bar(x, pts, color="steelblue")
    axb.errorbar(x, pts, yerr=np.where(np.isfinite(err), err, 0), fmt="none", ecolor="black", capsize=3)
    axb.axhline(1.0, color="green", ls="--", alpha=0.5, label="perfect additivity")
    axb.set_xticks(x, names, rotation=45, ha="right", fontsize=7)
    axb.set_ylabel("cos(Δab, Δa+Δb)  (raw basis)"); axb.set_ylim(-1.05, 1.05)
    axb.set_title("additivity — combo vs sum-of-parts", fontsize=10); axb.legend(fontsize=8)
    _finish(fig, "F3 — attribute geometry: direction cosines + additivity")
    return fig


def f4_quality(data: dict):
    key = _PRIMARY if _PRIMARY in data["gate2"] else next(iter(data["gate2"]))
    g = data["gate2"][key]
    fig, (axb, axs) = plt.subplots(1, 2, figsize=(13, 5))

    if g["g2b"] is not None:
        fams = list(g["g2b"]["by_family"].keys())
        base_names = list(next(iter(g["g2b"]["by_family"].values()))["baselines_mos"].keys())
        series = ["head", *base_names]
        x = np.arange(len(fams)); w = 0.8 / len(series)
        for s, name in enumerate(series):
            vals = [_f(g["g2b"]["by_family"][fam]["head_mos"][0]) if name == "head"
                    else _f(g["g2b"]["by_family"][fam]["baselines_mos"][name][0]) for fam in fams]
            axb.bar(x + s * w, vals, w, label=name)
        axb.set_xticks(x + w * (len(series) - 1) / 2, fams, rotation=45, ha="right")
        axb.set_ylabel("SRCC vs human MOS"); axb.legend(fontsize=8)
        axb.set_title(f"G2b per-family: head vs NR baselines  ({g['g2b']['n_beats']}/7 beaten)", fontsize=10)
    else:
        axb.text(0.5, 0.5, "G2b NOT EVALUABLE\n(no human MOS)", ha="center", va="center", fontsize=12)
        axb.set_axis_off()

    scatter = np.array(g["scatter"], dtype=float).reshape(-1, 2)
    if scatter.size:
        axs.scatter(scatter[:, 0], scatter[:, 1], s=14, alpha=0.6, color="darkorange")
    srcc = _f(g["g2a"]["ref_srcc"][0])  # null on a degenerate split → renders "nan", never a TypeError
    axs.set_xlabel("ViSQOL (full-reference target)"); axs.set_ylabel("head prediction (no reference)")
    axs.set_title(f"G2a pooled: head vs ViSQOL  (SRCC {srcc:.2f} — {'PASS' if g['g2a']['pass'] else 'fail'})",
                  fontsize=10)
    # surface power/overall so a spuriously-tight CI on thin data can't read as a clause PASS
    status = "UNDERPOWERED" if g.get("underpowered") else ("GATE 2 PASS" if g.get("passed") else "GATE 2 fail")
    _finish(fig, f"F4 — quality head vs incumbents  ({key})  —  {status}")
    return fig


def f5_summary(data: dict):
    rows = data["heatmap"]
    fams = list(rows[0]["severity_srcc"].keys())
    labels = [_row_label(r) for r in rows]
    type_f1 = [_f(r["type_macro_f1"][0]) for r in rows]
    mean_sev = [float(np.nanmean([_f(r["severity_srcc"][f][0]) for f in fams])) for r in rows]

    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(2 + 1.3 * len(labels), 5))
    ax.bar(x - w / 2, type_f1, w, label="type macro-F1", color="teal")
    ax.bar(x + w / 2, mean_sev, w, label="mean severity SRCC", color="indianred")
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylabel("score"); ax.set_ylim(0, 1); ax.legend()
    ax.set_title("codec vs SSL vs log-mel floor (mel + energy are real)", fontsize=10)
    _finish(fig, "F5 — representation summary")
    return fig


def f6_ood(data: dict):
    ood = data["ood"]
    keys = list(ood.keys())
    x = np.arange(len(keys)); w = 0.38
    fig, ax = plt.subplots(figsize=(2 + 2.2 * len(keys), 5))
    ax.bar(x - w / 2, [_f(ood[k]["grid_disagreement"]) for k in keys], w, label="in-grid", color="slategray")
    ax.bar(x + w / 2, [_f(ood[k]["ood_disagreement"]) for k in keys], w, label="out-of-grid", color="crimson")
    for i, k in enumerate(keys):
        floor = _f(ood[k]["calibration_floor"])
        ax.hlines(floor, i - 0.45, i + 0.45, color="black", ls=":",
                  label="calibration floor" if i == 0 else None)
    ax.set_xticks(x, keys, rotation=15, ha="right")
    ax.set_ylabel("metric disagreement (score units)"); ax.legend(fontsize=8)
    ax.set_title("NR-metric disagreement: grid vs generative (by construction — FAKE incumbents)", fontsize=10)
    _finish(fig, "F6 — OOD teaser")
    return fig


_RENDERERS = {"F1": f1_readability, "F2": f2_monotonicity, "F3": f3_geometry,
              "F4": f4_quality, "F5": f5_summary, "F6": f6_ood}


def render_figures(reports_dir: Path, fig_dir: Path) -> dict[str, Path]:
    data = _load(reports_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for name in FIGURES:
        fig = _RENDERERS[name](data)
        path = fig_dir / f"{name}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        out[name] = path
    return out
