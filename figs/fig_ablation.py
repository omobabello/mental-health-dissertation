"""fig-ablation: LR test macro-F1 under the protocol vs two preprocessing ablations, D1 and D2.

Ported 1:1 from 05_figures.fig_ablation (same fields, variants, order, value labels; y-range now from 0);
only the styling changed. Standalone:  cd experiments && .venv/bin/python -m figs.fig_ablation

Colour: all bars are the same model (LR), so the variants are not model identities. Following
the chart design spec (non-model categories use accent/tint or slate, never a model colour) the protocol bar
is the focal accent (the darkest mark); the near-identical "stop-words retained" variant is a mid teal
(>= 3:1 on paper) and the "target lexicon masked" variant is the lighter slate token.
"""
import numpy as np
from matplotlib.patches import Patch

from fig_style import FS_ANNOT, axis_lines, FS_LABEL, T, W, plt, save, value_label

VARIANTS = [("baseline", "Protocol (stop words removed)"), ("stopwords_retained", "Stop words retained"),
            ("target_lexicon_masked", "Target lexicon masked")]
# accent (8.8:1) is the darkest, focal mark; the two ablations are a mid teal (3.1:1) and the lighter
# slate token MUTED (4.8:1). All three clear 3:1 on paper and stay distinct in greyscale (L 0.07/0.29/0.17).
SHADES = [T.ACCENT, "#6f97ab", T.MUTED]
CONDS = ["D1", "D2"]
BW = 0.16          # thin bars (the chart design spec s6)
GAP_PT = 2.0       # white gap between adjacent bars, in printed points; no outlines
XLIM = (-0.45, len(CONDS) - 0.55)


def _vals(an, k):
    return [an["conditions"][c]["models"]["LR"]["test"]["macro_f1"] if k == "baseline"
            else an["conditions"][c]["ablations"][k]["test"]["macro_f1"] for c in CONDS]


def draw(an, ds=None, name="fig-ablation"):
    fig, ax = plt.subplots(figsize=(W, 2.6))
    x = np.arange(len(CONDS))
    axis_lines(ax, grid="y")  # both axis lines; value-axis grid under the bars
    ax.set_xticks(x, CONDS)
    ax.tick_params(axis="x", labelcolor=T.INK, labelsize=FS_LABEL)
    ax.set_xlim(*XLIM)
    ax.set_ylim(0, 1.05)  # bars start at zero so bar length is proportional to macro-F1
    ax.set_yticks(np.arange(0, 1.01, 0.2), [f"{t:.1f}" for t in np.arange(0, 1.01, 0.2)])
    ax.set_ylabel("LR macro-F1 (test)")
    handles = [Patch(color=s, label=lab) for s, (_, lab) in zip(SHADES, VARIANTS)]
    leg = fig.legend(handles=handles, ncol=3, loc="outside upper left", frameon=False, fontsize=FS_LABEL,
                     handlelength=1.0, handleheight=0.8, columnspacing=1.6, handletextpad=0.5)

    # settle the layout once, then size the bar gap in data units from the real axes width
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    ab = ax.get_window_extent(r)
    pt_per_unit = ab.width / (XLIM[1] - XLIM[0]) * 72.0 / fig.dpi
    step = BW + GAP_PT / pt_per_unit

    for i, (k, lab) in enumerate(VARIANTS):
        vals = _vals(an, k)
        xs = x + (i - 1) * step
        ax.bar(xs, vals, BW, color=SHADES[i], label=lab, linewidth=0, zorder=2)
        for xx, v in zip(xs, vals):
            value_label(ax, xx, v, f"{v:.3f}", dx=0, dy=3, ha="center", va="bottom", halo=False)

    # freeze the layout and put the legend row flush with the y-label edge (house left alignment)
    fig.canvas.draw()
    fig.set_layout_engine("none")
    r = fig.canvas.get_renderer()
    # rotated y-label: its extent starts at the glyph descent, so add the descent to land on the ink edge
    yl = ax.yaxis.label
    _, _, desc = r.get_text_width_height_descent(yl.get_text(), yl.get_fontproperties(), ismath=False)
    y_left = yl.get_window_extent(r).x0 + desc
    lb = leg.get_window_extent(r)
    x0, y1 = lb.x0 / fig.bbox.width, lb.y1 / fig.bbox.height
    leg.set_loc("upper left")
    leg.borderaxespad = 0
    for _ in range(3):  # converge: first swatch's left edge onto the y-label's left edge
        leg.set_bbox_to_anchor((x0, y1), transform=fig.transFigure)
        fig.canvas.draw()
        dx = y_left - leg.get_patches()[0].get_window_extent(fig.canvas.get_renderer()).x0
        x0 += dx / fig.bbox.width
    save(fig, name)
    return fig


if __name__ == "__main__":
    from common import load_json

    draw(load_json("analysis.json"), load_json("dataset_summary.json"))
