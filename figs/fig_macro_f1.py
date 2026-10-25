"""fig-macro-f1: test macro-F1 per model with 95% bootstrap CIs, per condition.

Ported 1:1 from 05_figures.fig_macro_f1 (same data, CIs, threshold, value labels, seed sd);
only the styling changed. Standalone:  cd experiments && .venv/bin/python -m figs.fig_macro_f1
"""
import numpy as np

from fig_style import (COLORS, axis_lines, COND_LABEL, FAMILY, FS_ANNOT, MARK, T, W, models_by_family, panel_title, plt, save,
                       threshold, value_label)


def draw(an, ds=None, name="fig-macro-f1"):
    conds = [c for c in ["D1", "D2"] if c in an["conditions"]]
    n_models = max(len(models_by_family(an["conditions"][c])) for c in conds)
    wide = n_models > 3          # 6 models: stack the conditions so each panel gets the full width
    if wide:
        fig, axes = plt.subplots(len(conds), 1, figsize=(W, 2.45 * len(conds)), sharey=False)
    else:
        fig, axes = plt.subplots(1, len(conds), figsize=(W, 2.9), sharey=False)
    for ax, cond in zip(np.atleast_1d(axes), conds):
        d = an["conditions"][cond]
        ms = models_by_family(d)
        for i, m in enumerate(ms):
            v = d["models"][m]["test"]["macro_f1"]
            lo, hi = d["models"][m]["macro_f1_ci95"]
            ax.errorbar(i, v, yerr=[[v - lo], [hi - v]], fmt=MARK[FAMILY[m]], color=COLORS[m], ms=7,
                        mec="white", mew=1.0, capsize=2.5, capthick=1.1, elinewidth=1.1, zorder=3)
            value_label(ax, i, v, f"{v:.3f}", dx=7)
            if "test_seed_agg" in d["models"][m]:
                sd = d["models"][m]["test_seed_agg"]["macro_f1"]["std"]
                value_label(ax, i, lo, f"seed SD {sd:.4f}", dx=0, dy=-4, ha="center", va="top",
                            color=T.MUTED)
        best = d["best_classical"]
        thr = d["models"][best]["test"]["macro_f1"] + an["meta"]["min_delta_f1"]
        ax.set_xticks(range(len(ms)), ms)
        axis_lines(ax, grid="y")
        ax.tick_params(axis="x", labelcolor=T.INK)
        ax.set_xlim(-0.5, len(ms) - 0.3)
        if wide:   # label in the right margin so it never sits on a CI
            threshold(ax, thr)
            ax.annotate(f"{best} + 0.02\nthreshold", (1.0, thr), xycoords=("axes fraction", "data"),
                        xytext=(6, 0), textcoords="offset points", ha="left", va="center",
                        fontsize=FS_ANNOT, color=T.ACCENT, linespacing=1.1)
        else:
            threshold(ax, thr, f"{best} + 0.02 threshold", x_text=len(ms) - 0.35)
        lo_all = min(d["models"][m]["macro_f1_ci95"][0] for m in ms)
        if any("test_seed_agg" in d["models"][m] for m in ms):   # room for the seed-sd notes
            ax.set_ylim(bottom=lo_all - 0.012)
        ax.margins(y=0.08)
        panel_title(ax, COND_LABEL[cond], size=FS_ANNOT + 0.5)
        ax.set_ylabel("Macro-F1 (test, 95% bootstrap CI)")
    save(fig, name)
    return fig


if __name__ == "__main__":
    from common import load_json

    draw(load_json("analysis.json"), load_json("dataset_summary.json"))
