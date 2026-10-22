"""fig-top-features: top-k depression-indicative features (D2) - LR coefficients and, when a
transformer run exists, mean Integrated Gradients attributions of the first of MentalBERT/DistilBERT.

Ported 1:1 from 05_figures.fig_features (same fields: classical_results[cond]["LR"]
["interpretability"]["top_pos"][:k] and results/transformers/<cond>_<m>_<rep_seed>.json
["interpretability"]["top_pos"][:k]; same order: rank 1 at the top; same titles and x-labels).
Only the styling changed. Standalone:  cd experiments && .venv/bin/python -m figs.fig_top_features
"""
import json

import numpy as np

from common import RESULTS, load_json
from fig_style import COLORS, axis_lines, FS_LABEL, FS_TICK, FS_TITLE, T, W, panel_title, plt, save


def _panels(an, cond, k):
    d = an["conditions"][cond]
    lr = load_json("classical_results.json")[cond]["LR"]["interpretability"]["top_pos"][:k]
    tm = [m for m in ["MentalBERT", "DistilBERT"] if m in d["models"]]
    panels = [("LR", lr, "Coefficient (TF-IDF, stemmed)")]
    for m in tm[:1]:
        rep = json.load(open(RESULTS / "transformers" / f"{cond}_{m}_{d['models'][m]['rep_seed']}.json"))
        panels.append((m, rep["interpretability"]["top_pos"][:k], "Mean Integrated Gradients attribution"))
    return panels


def draw(an, ds=None, cond="D2", k=15, name="fig-top-features"):
    panels = _panels(an, cond, k)
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(W, 3.3))
    for ax, (m, items, xl) in zip(np.atleast_1d(axes), panels):
        names = [t for t, _ in items][::-1]
        vals = [v for _, v in items][::-1]
        # thin bars; 0.68 of the slot leaves a clear white gap between neighbours
        ax.barh(range(len(names)), vals, color=COLORS[m], height=0.68, zorder=2)
        ax.set_yticks(range(len(names)), names)
        axis_lines(ax, grid="x")   # both axis lines; value axis is horizontal, so vertical grid only
        ax.tick_params(axis="y", labelcolor=T.INK, labelsize=FS_TICK, pad=3)
        ax.set_ylim(-0.6, len(names) - 0.4)
        ax.set_xlim(0, max(vals) * 1.04)
        ax.set_xlabel(xl, fontsize=FS_LABEL)
    # one shared figure title (the v1 wording) + short matching one-line panel titles, so both
    # panel titles sit on one baseline and nothing wraps
    for i, (ax, (m, _, _)) in enumerate(zip(np.atleast_1d(axes), panels)):
        panel_title(ax, f"({'ab'[i]}) {m}" if n > 1 else m, size=FS_LABEL)
    sup = fig.suptitle(f"Top {k} depression-indicative features ({cond})", x=0.0, ha="left",
                       fontsize=FS_TITLE, fontweight="bold", color=T.INK)
    # no y-label here, so align each title (and the shared title, to the LR panel) with the left
    # edge of the feature names (house rule: titles flush with the outer edge of the panel block)
    for _ in range(3):
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        for j, ax in enumerate(np.atleast_1d(axes)):
            ab = ax.get_window_extent(r)
            x0 = min(t.get_window_extent(r).x0 for t in ax.get_yticklabels() if t.get_text())
            ax._left_title.set_x((x0 - ab.x0) / ab.width)
            if j == 0:
                sup.set_x(x0 / fig.bbox.width)
    save(fig, name)
    return fig


if __name__ == "__main__":
    draw(load_json("analysis.json"), load_json("dataset_summary.json"))
