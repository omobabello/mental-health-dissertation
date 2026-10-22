"""fig-roc-pr: ROC and precision-recall curves per model, D1 (top) and D2 (bottom).

Ported 1:1 from 05_figures.fig_curves (same scores, sklearn curves, AUC/AP legend values, chance
diagonal, base-rate line, panel titles and ordering); only the styling changed.

Styling notes (deliberate deviations from the chart design spec section 6):
- Up to 3 models the legends sit inside each panel (one style for all four: opaque paper fill, no edge),
  because they carry the per-panel AUC / AP values; ROC legends use the empty region below the chance
  diagonal, PR legends sit just above the base-rate line, inset to clear RF's drop near recall 0.
  With more than 3 models each panel's legend moves BELOW its panel as a frameless 2-column block
  (classical column, transformer column) so no curve is hidden.
- Legend handles are ordered by family, then by ORDER.
- Every curve carries its family marker, as section 6 requires: markers are placed by display arc length
  with a per-model phase, and markers within ~6 px of the plot edges (origin, (1, 1), the FPR = 0 rise,
  recall = 1 / base-rate end) are dropped so they read as curve identifiers, not axis clutter.
- Line weight encodes nothing: every curve is 1.3 pt; coincident LR / SVM curves stay distinguishable
  through their phase-shifted markers.
- No accent (no focal threshold exists); the neutral base-rate line is labelled on the line
  ("base rate 0.30", slate, halo); legends are frameless so gridlines stay unbroken.
Alt text (for the caption when inserted): four panels; ROC (a, c) with a dotted chance diagonal and
precision-recall (b, d) with a dotted base-rate line (0.30 in both D1 and D2); each model is a coloured
line with its family marker; legend values give AUC / AP per model.
Standalone:  cd experiments && .venv/bin/python -m figs.fig_roc_pr
"""
import json

import numpy as np
from matplotlib.lines import Line2D
from sklearn.metrics import precision_recall_curve, roc_curve

from fig_style import (COLORS, axis_lines, FAMILY, FS_ANNOT, FS_LABEL, FS_TITLE, HALO, MARK, ORDER, T, W, panel_title, plt,
                       reference, save)


def load_scores(an, cond):
    """Same as 05_figures.load_scores, but takes the already-loaded analysis dict."""
    import pandas as pd
    from common import DATA, RESULTS, load_json

    te = pd.read_parquet(DATA / f"{cond}_test.parquet")
    cp = load_json(f"{cond}_classical_preds.json")
    sc = {m: np.array(cp[m]["y_score"]) for m in ["LR", "SVM", "RF"]}
    for m, d in an["conditions"][cond]["models"].items():
        if d["family"] == "transformer":
            p = RESULTS / "transformers" / f"{cond}_{m}_{d['rep_seed']}.json"
            sc[m] = np.array(json.load(open(p))["y_score"])
    return te.label.values, sc


# (lw, zorder) per model: one width for every curve (width encodes nothing); zorder keeps LR above SVM.
_LINE = {"SVM": (1.3, 3), "LR": (1.3, 4), "RF": (1.3, 2)}


def _line(m):
    return _LINE.get(m, (1.3, 3))


def _handle(m, label):
    """Legend handle: model colour line + the same family marker as the curve, at the curve's own width."""
    return Line2D([], [], color=COLORS[m], lw=_line(m)[0], marker=MARK[FAMILY[m]], ms=3.6, mec="white", mew=0.5,
                  label=label)


def _legend_order(models):
    """Family first (classical, then transformer), then fixed ORDER."""
    fam = {"classical": 0, "transformer": 1}
    return sorted(models, key=lambda m: (fam[FAMILY[m]], ORDER.index(m)))


def _legend(ax, handles, loc, **kw):
    """In-panel legend (<= 3 models): one style for all panels - opaque paper fill, no edge, above the grid."""
    leg = ax.legend(handles=handles, loc=loc, fontsize=FS_LABEL, frameon=False, borderpad=0.3, borderaxespad=0.3,
                    handlelength=1.8, handletextpad=0.5, labelspacing=0.3, **kw)
    leg.set_zorder(5)
    return leg


def _legend_below(ax, handles):
    """> 3 models: frameless 2-column block under the panel (column 1 classical, column 2 transformers)."""
    return ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.12, -0.2), ncol=2,
                     fontsize=FS_ANNOT, frameon=False, borderaxespad=0, handlelength=1.3, handletextpad=0.3,
                     labelspacing=0.25, columnspacing=0.6)


def _mark_pts(x, y, k, ylo, yhi, aspect, edge):
    """Marker positions evenly spaced by display arc length (per-model phase k), interpolated along the
    drawn segments (so long straight runs still get markers), skipping points near the plot edges.
    x, y in data units; y normalised by (ylo, yhi); aspect = panel height / width; edge(x, y) -> bool."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    yn = (y - ylo) / (yhi - ylo) * aspect
    s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(yn)))])
    s = s / s[-1]
    t = np.arange(0.10 + 0.06 * k, 0.93, 0.19)
    xs, ys = np.interp(t, s, x), np.interp(t, s, y)
    keep = [i for i in range(len(t)) if not edge(xs[i], ys[i])]
    return xs[keep], ys[keep]


def draw(an, ds=None, name="fig-roc-pr", scores=None):
    """scores: optional {cond: (y, {model: y_score})}; loaded from results/ when omitted."""
    n_models = max(len([m for m in ORDER if m in an["conditions"][c]["models"]]) for c in ["D1", "D2"])
    below = n_models > 3   # 6 models: legends move under each panel so no curve is hidden
    fig, axes = plt.subplots(2, 2, figsize=(W, 4.75 if not below else 6.4))
    ROC_LIM, PR_HI = (-0.01, 1.02), 1.02
    asp = 0.72   # approx. panel height / width (display), for arc-length marker spacing
    for r, cond in enumerate(["D1", "D2"]):
        y, sc = scores[cond] if scores else load_scores(an, cond)
        roc_h, pr_h = {}, {}
        lo = min(0.4, y.mean() - 0.04) - 0.02
        for m in [m for m in ORDER if m in sc]:
            fpr, tpr, _ = roc_curve(y, sc[m])
            pr, rc, _ = precision_recall_curve(y, sc[m])
            d = an["conditions"][cond]["models"][m]["test"]
            lw, z = _line(m)
            kw = dict(color=COLORS[m], lw=lw, zorder=z, solid_joinstyle="round", solid_capstyle="round")
            mkw = dict(color=COLORS[m], zorder=z, ls="none", marker=MARK[FAMILY[m]], ms=3.6, mec="white",
                       mew=0.5)   # family marker, always

            # markers evenly spaced along the drawn curve (display distance, not data index), with a
            # per-model phase so coincident curves (LR / SVM) do not hide each other's markers;
            # nothing on the plot edges (origin, (1, 1), FPR = 0 rise, recall = 1 / base-rate end)
            k = [x for x in ORDER if x in sc].index(m)
            roc_mk = _mark_pts(fpr, tpr, k, *ROC_LIM, asp,
                               lambda a, b: a < 0.02 or a > 0.98 or b < 0.02 or b > 0.96)
            pr_mk = _mark_pts(rc, pr, k, lo, PR_HI, asp,
                              lambda a, b: a < 0.02 or a > 0.98 or b < y.mean() + 0.03 or b > 0.96)

            axes[r, 0].plot(fpr, tpr, **kw)
            axes[r, 0].plot(*roc_mk, **mkw)
            axes[r, 1].plot(rc, pr, **kw)
            axes[r, 1].plot(*pr_mk, **mkw)
            roc_h[m] = _handle(m, f"{m} (AUC {d['roc_auc']:.3f})")
            pr_h[m] = _handle(m, f"{m} (AP {d['pr_auc']:.3f})")
        roc_h = [roc_h[m] for m in _legend_order(roc_h)]
        pr_h = [pr_h[m] for m in _legend_order(pr_h)]
        a0, a1 = axes[r]
        a0.plot([0, 1], [0, 1], color=T.LINE, lw=0.8, ls=(0, (2, 2)), zorder=1)   # chance diagonal
        reference(a1, y=y.mean())                                                 # base rate
        a0.set(xlabel="False positive rate", ylabel="True positive rate")
        a1.set(xlabel="Recall", ylabel="Precision")
        panel_title(a0, f"({'ac'[r]}) ROC, {cond}")
        panel_title(a1, f"({'bd'[r]}) Precision-recall, {cond}")
        # base rate labelled on its own line (slate, halo), mid-panel where no curve reaches it
        a1.annotate(f"base rate {y.mean():.2f}", (0.5, y.mean()), xytext=(0, 1.5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=FS_ANNOT, color=T.SLATE, bbox=HALO, zorder=4)
        for ax in (a0, a1):
            ax.set_xlim(-0.01, 1.01)
            ax.set_xticks(np.linspace(0, 1, 6))
            axis_lines(ax, grid="both")
        a0.set_ylim(*ROC_LIM)
        a0.set_yticks(np.linspace(0, 1, 6))
        hi = PR_HI
        a1.set_ylim(lo, hi)
        a1.set_yticks(list(np.round(np.arange(0.4, 1.01, 0.2), 1)))   # base rate is labelled on its line
        if below:
            _legend_below(a0, roc_h)
            _legend_below(a1, pr_h)
        else:
            _legend(a0, roc_h, "lower right", bbox_to_anchor=(1.0, -0.005))
            # lift the PR legend so its lower edge clears the base-rate label (+0.10 in data units)
            # and inset it from the left so it clears RF's vertical drop near recall 0
            _legend(a1, pr_h, "lower left", bbox_to_anchor=(0.04, (y.mean() + 0.111 - lo) / (hi - lo)))
    save(fig, name)
    return fig


if __name__ == "__main__":
    import sys

    from common import load_json

    an = load_json("analysis.json")
    if "--mock6" in sys.argv:   # the chart design spec section 6: 6-model stress test (writes a _mock6 file)
        import copy

        an = copy.deepcopy(an)
        rng = np.random.default_rng(0)
        scores = {}
        for cond in ["D1", "D2"]:
            y, sc = load_scores(an, cond)
            for k, m in enumerate(["DistilBERT", "MentalBERT", "BERT"]):
                base = an["conditions"][cond]["models"]["LR"]
                an["conditions"][cond]["models"][m] = {**base, "family": "transformer", "rep_seed": 42,
                                                       "test": {**base["test"], "roc_auc": 0.9 + k / 100,
                                                                "pr_auc": 0.8 + k / 100}}
                sc[m] = y + rng.normal(0, 0.55 - 0.08 * k, len(y))
            scores[cond] = (y, sc)
        import os
        from pathlib import Path

        import fig_style
        fig_style.OUT = Path(os.environ.get("MOCK_OUT", "/tmp/fig-mock"))   # keep mocks out of figures/
        fig_style.PNG = fig_style.OUT
        draw(an, None, name="fig-roc-pr_mock6", scores=scores)
    else:
        draw(an, load_json("dataset_summary.json"))
