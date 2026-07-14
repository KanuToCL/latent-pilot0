"""AnalysisReport → JSON-able dicts for the three Phase-5 acceptance artifacts
(F1 heatmap, F3 cosine matrices, F2 monotonicity curves). Shared by the `analyze`
demo and the Phase-8 release writer so both emit byte-identical schemas.
"""

from __future__ import annotations

from .run import AnalysisReport


def heatmap_rows(report: AnalysisReport) -> list[dict]:
    rows = [report.floor, report.energy, *(c.readability for c in report.candidates.values())]
    return [
        {"name": r.name, "variant": r.variant, "type_macro_f1": r.type_macro_f1,
         "severity_srcc": r.severity_srcc, "n_test_groups": r.n_test_groups}
        for r in rows
    ]


def cosine_matrices(report: AnalysisReport) -> dict:
    return {
        key: {"labels": c.geometry.labels, "probe_cosine": c.geometry.probe_cosine,
              "mean_cosine": c.geometry.mean_cosine,
              "centroid_pca_explained": c.geometry.centroid_pca_explained}
        for key, c in report.candidates.items()
    }


def monotonicity_curves(report: AnalysisReport) -> dict:
    return {
        key: {fam: {"srcc": (fi.srcc.point, fi.srcc.lo, fi.srcc.hi), "curve": fi.curve,
                    "interp_frac": fi.interp_frac, "evaluable": fi.evaluable,
                    "interpolates": fi.interpolates}
              for fam, fi in c.interpolation.by_family.items()}
        for key, c in report.candidates.items()
    }
