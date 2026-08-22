# RQ2 geometry figures. Rendered from ONE candidate's serialised JSON block (the
# same dict that lands in reports/geometry_real.json) - a JSON -> PNG boundary, so
# plotting never re-runs the analysis and never re-reads the cache. Keys of the
# serialised block are strings (json), hence the int() coercions.
#
# No FAKE banner is stamped here (unlike release/figures.py): this runner aborts on
# a fake encoder, so a banner would be a false claim on a real-latent figure.
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
except ImportError as e:  # pragma: no cover - exercised only without the extra
    raise ImportError("Figure rendering needs matplotlib - pip install 'pilot0[figures]'") from e

DPI = 130


def safe_name(key: str) -> str:
    """'encodec24k:z' -> 'encodec24k-z'. ':' is illegal in a Windows filename."""
    return key.replace(":", "-").replace("/", "-")


def _f(x) -> float:
    """to_jsonable maps a degenerate nan statistic to null; plot it as a gap."""
    return float("nan") if x is None else float(x)


def _family_colors(labels):
    cmap = plt.get_cmap("tab10")
    return {f: cmap(i % 10) for i, f in enumerate(labels)}


def _save(fig, path: Path) -> Path:
    fig.savefig(path, dpi=DPI, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def fig_cosine_matrix(block: dict, path: Path) -> Path:
    """(a) class-mean direction cosines, family-labelled."""
    labels = block["geometry"]["labels"]
    M = np.array(block["geometry"]["mean_cosine"], dtype=float)
    fig, ax = plt.subplots(figsize=(1.0 * len(labels) + 3.0, 1.0 * len(labels) + 2.2))
    im = ax.imshow(M, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="black" if abs(M[i, j]) < 0.6 else "white")
    fig.colorbar(im, ax=ax, label="cosine", fraction=0.046)
    ax.set_title(f"class-mean direction cosines (standardised) - {block['key']}", fontsize=10)
    fig.tight_layout()
    return _save(fig, path)


def fig_centroid_pca(block: dict, path: Path) -> Path:
    """(b) condition centroids on PC1 x PC2; colour = family, size ~ severity."""
    geo = block["geometry"]
    coords = np.array(geo["centroid_coords"], dtype=float)
    fams = [lab.split("/")[0] for lab in geo["centroid_labels"]]
    sevs = [int(lab.split("/")[1]) for lab in geo["centroid_labels"]]
    explained = np.array(geo["centroid_pca_explained"], dtype=float)
    colors = _family_colors(geo["labels"])

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    for f in geo["labels"]:
        idx = [i for i, fam in enumerate(fams) if fam == f]
        ax.scatter(coords[idx, 0], coords[idx, 1], s=[25 + 35 * sevs[i] for i in idx],
                   color=colors[f], alpha=0.85, edgecolors="none", label=f)
        order = sorted(idx, key=lambda i: sevs[i])  # severity ladder, low -> high
        ax.plot(coords[order, 0], coords[order, 1], color=colors[f], alpha=0.35, lw=1.0)
    ax.set_xlabel(f"PC1 ({100 * explained[0]:.0f}% of centroid variance)")
    ax.set_ylabel(f"PC2 ({100 * explained[1]:.0f}%)")
    ax.set_title(f"condition centroids (clean excluded), size ~ severity - {block['key']}", fontsize=10)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return _save(fig, path)


def fig_concentration(block: dict, path: Path) -> Path:
    """(c) per-family delta concentration vs the 1/sqrt(D) isotropic null."""
    by_fam = block["shift"]["concentration"]["by_family"]
    fams = list(by_fam)
    vals = [_f(by_fam[f]["mean_cosine"]) for f in fams]
    null = float(block["null_scale"])
    colors = _family_colors(fams)

    fig, ax = plt.subplots(figsize=(1.1 * len(fams) + 3.0, 5.0))
    ax.bar(range(len(fams)), vals, color=[colors[f] for f in fams])
    ax.axhline(null, color="white", ls="--", lw=1.2, label=f"null 1/sqrt(D) = {null:.3f}")
    ax.axhline(0.0, color="grey", lw=0.8)
    for i, v in enumerate(vals):
        if np.isfinite(v):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    ax.set_xticks(range(len(fams)), fams, rotation=45, ha="right")
    ax.set_ylabel("mean pairwise cos(delta_i, delta_j)")
    overall = _f(block["shift"]["concentration"]["overall"]["mean_cosine"])
    ax.set_title(f"delta concentration - {block['key']}  (all families pooled: {overall:.3f})", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, path)


def fig_magnitude(block: dict, path: Path) -> Path:
    """(d) median ||delta|| vs severity, one curve per family."""
    med = block["shift"]["magnitude"]["median_norm"]
    rho = block["shift"]["magnitude"]["srcc_by_family"]
    colors = _family_colors(list(med))
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    for f, curve in med.items():
        xs = sorted(int(s) for s in curve)
        ys = [curve[str(s)] for s in xs]
        r = rho.get(f)
        tag = f"{f} (rho={r:.2f})" if r is not None else f"{f} (rho=n/a)"
        ax.plot(xs, ys, marker="o", color=colors[f], label=tag)
    ax.set_xlabel("severity"); ax.set_ylabel("median ||delta||  (standardised units)")
    ax.set_xticks(range(1, 6))
    ax.set_title(f"shift magnitude vs severity - {block['key']}", fontsize=10)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return _save(fig, path)


_PANELS = (("a_cosine", fig_cosine_matrix), ("b_centroid_pca", fig_centroid_pca),
           ("c_concentration", fig_concentration), ("d_magnitude", fig_magnitude))


def render_candidate_figures(block: dict, fig_dir: Path) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_name(block["key"])
    with plt.style.context("dark_background"):
        return [render(block, fig_dir / f"{stem}_{tag}.png") for tag, render in _PANELS]
