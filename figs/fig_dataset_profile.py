"""fig-dataset-profile: (a) class and source composition per condition, (b) post-length distributions.

Ported from 05_figures.fig_dataset_profile (same fields: source_counts, length_hist_pos/neg; same
ordering: depression first, negatives by descending count; same 40-token bins, density x 100 x 40,
medians in the legend). Segment names are the exact strings printed in figures-v1 (rotated D1 names
were cut to 6 characters there; parity requires them verbatim). Only the styling changed.
Layout: panels stacked so every segment name fits horizontally at >= 8 pt.
Standalone:  cd experiments && .venv/bin/python -m figs.fig_dataset_profile
"""
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import offset_copy

from fig_style import FS_ANNOT, FS_LABEL, FS_TITLE, SEM, T, W, axis_lines, panel_title, plt, save

def _px(ax, pt):
    """Width in x-data units of `pt` printed points (axes limits must already be set)."""
    x0 = ax.transData.transform((0, 0))[0]
    return ax.transData.inverted().transform((x0 + pt * ax.figure.dpi / 72, 0))[0]


# Semantic colour (the chart design spec 2b): the positive class is warm everywhere (solid segment in (a), solid
# line in (b)); negative sources are cool. D1 everyday-life controls take the navy ramp, D2 mental-health
# "hard negatives" a bluer indigo ramp, so the two kinds of negative read differently at a glance.
# Each ramp alternates dark (white label) / light (ink label) so adjacent segments separate.
POS, POS_TXT = SEM["positive"], T.PAPER
NEG_RAMP = {"D1": [(SEM["navy"], T.PAPER), ("#a8c3d1", T.INK)],
            "D2": [("#3c4f8c", T.PAPER), ("#bcc6e3", T.INK)]}
NEG_LINE = {"D1": SEM["navy"], "D2": "#3c4f8c"}


# Parity keeps the v1 6-character D1 tokens. If a parity exception is granted, set True to print the
# full source names (two lines where needed) instead.
FULL_D1_NAMES = False
# (~1,000-post segments are ~30 pt wide: longer names wrap; parity re-joins line-end hyphens)
FULL_NAME = {"personalfinance": "personal\nfinance", "parenting": "parent-\ning",
             "meditation": "medi-\ntation", "teaching": "teach-\ning"}
H_IN = 3.9          # trimmed again after layout so the inter-panel gap matches the (a) title gap
LEG_AXES_PT = 5      # (b) key row -> plot top
LEG_TITLE_PT = 4     # (b) title -> key row
RIGHT_PAD_PT = 2     # last x tick label -> figure right edge
THOUSANDS = FuncFormatter(lambda v, _: f"{int(v):,}")


def _seg_name(cond, k):
    """Segment label as printed in figures-v1 (D1 names cut to 6 chars, D2 names raw keys).

    A cut name gets an abbreviation point ("person." not "person") so it reads as a deliberate
    abbreviation rather than a typo; the parity checker strips edge punctuation, so the token is unchanged.
    """
    if cond != "D1":
        return k
    if FULL_D1_NAMES:
        return FULL_NAME.get(k, k)
    return k[:6] + "." if len(k) > 6 else k


def draw(an=None, ds=None, name="fig-dataset-profile"):
    fig, axes = plt.subplots(2, 1, figsize=(W, H_IN), gridspec_kw={"height_ratios": [1, 1.55]})

    # ---------------------------------------------------------------- (a) composition
    ax = axes[0]
    conds = ["D1", "D2"]
    ax.set_xlim(0, 10000)  # set first: _px() converts points to data units
    for y, cond in enumerate(conds):
        sc = ds["conditions"][cond]["source_counts"]
        neg = sorted([k for k in sc if k != "depression"], key=lambda k: -sc[k])
        # true segment widths; the 2 px white gap comes from a 1 pt paper edge on each segment
        seg = dict(height=0.62, edgecolor=T.PAPER, linewidth=1, zorder=2)
        # depression is the same (positive) class in both rows -> identical solid warm segment, white label
        ax.barh(y, sc["depression"], color=POS, **seg)
        ax.text(sc["depression"] / 2, y, f"depression\n{sc['depression']:,}", ha="center", va="center",
                color=POS_TXT, fontsize=FS_ANNOT, fontweight="bold", linespacing=1.15, zorder=3)
        left = sc["depression"]
        for i, k in enumerate(neg):
            fill, txt = NEG_RAMP[cond][i % 2]
            ax.barh(y, sc[k], left=left, color=fill, **seg)
            ax.text(left + sc[k] / 2, y, _seg_name(cond, k), ha="center", fontsize=FS_ANNOT,
                    # single lines sit on a shared baseline (cap height centred), so descenders and the
                    # abbreviation point do not shift labels up or down
                    **({"va": "center"} if "\n" in _seg_name(cond, k) else
                       {"va": "baseline", "transform": offset_copy(ax.transData, fig=fig, y=-0.36 * FS_ANNOT,
                                                                   units="points")}), color=txt, linespacing=1.1, zorder=3)
            left += sc[k]
    ax.set_yticks(range(len(conds)), conds)
    ax.tick_params(axis="y", labelcolor=T.INK, labelsize=FS_LABEL)
    ax.invert_yaxis()
    ax.set_ylim(len(conds) - 0.62, -0.38)
    ax.set_xticks(range(0, 10001, 2000))
    ax.xaxis.set_major_formatter(THOUSANDS)
    ax.set_xlabel("Posts (n)")
    axis_lines(ax, grid="x")  # both axis lines; value-axis (x) grid behind the bars
    panel_title(ax, "(a) Class and source composition (N = 10,000 each)")

    # ---------------------------------------------------------------- (b) length distribution
    ax = axes[1]
    bins = np.arange(0, 1001, 40)
    pos = np.array(ds["conditions"]["D1"]["length_hist_pos"])
    neg1 = np.array(ds["conditions"]["D1"]["length_hist_neg"])
    neg2 = np.array(ds["conditions"]["D2"]["length_hist_neg"])
    # same hues as (a): warm positive class (focal, thicker), navy D1 negatives, indigo dashed D2 negatives
    series = [(pos, "Depression (positive)", POS, "-", 3, 1.7),
              (neg1, "D1 negatives", NEG_LINE["D1"], "-", 2, 1.2),
              (neg2, "D2 negatives", NEG_LINE["D2"], (0, (4, 2)), 2.5, 1.3)]
    handles = []
    for arr, lab, c, ls, z, lw in series:
        h, e = np.histogram(arr, bins=bins, density=True)
        ax.stairs(h * 100 * 40, e, baseline=None, color=c, lw=lw, ls=ls, zorder=z)  # draws every bin incl. 960-1000
        handles.append(Line2D([], [], color=c, lw=lw, ls=ls, label=f"{lab}, median {int(np.median(arr))}"))
    ax.set_xlim(0, 1000)
    ax.xaxis.set_major_formatter(THOUSANDS)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Post length (whitespace tokens, capped at 1,000)")
    ax.set_ylabel("Share of posts per\n40-token bin (%)", linespacing=1.1)
    axis_lines(ax, grid="both")
    # key: one frameless row directly under the (b) title (house rule: legends only as a single row)
    leg = ax.legend(handles=handles, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False,
                    fontsize=FS_ANNOT, handlelength=1.8, columnspacing=1.2, handletextpad=0.5,
                    borderpad=0, borderaxespad=LEG_AXES_PT / FS_ANNOT)
    fig.canvas.draw()
    leg_h = leg.get_window_extent(fig.canvas.get_renderer()).height * 72 / fig.dpi
    ax.set_title("(b) Length distribution", loc="left", fontsize=FS_TITLE, fontweight="bold", color=T.INK,
                 pad=LEG_AXES_PT + leg_h + LEG_TITLE_PT)

    # ---------------------------------------------------------------- layout
    fig.canvas.draw()
    fig.set_layout_engine("none")  # freeze, then fix the vertical rhythm by hand
    r = fig.canvas.get_renderer()
    inch = lambda px: px / fig.dpi
    a, b = axes
    # gap (a) title -> top of the D1 bar: the rhythm the (a)-xlabel -> (b)-title gap must match
    bar_top = a.transData.transform((0, -0.31))[1]
    g_target = inch(a._left_title.get_window_extent(r).y0 - bar_top)
    g_now = inch(a.xaxis.label.get_window_extent(r).y0 - b._left_title.get_window_extent(r).y1)
    delta = g_now - g_target          # inches to remove between the panels
    h_old = fig.get_figheight()
    pa, pb = a.get_position(), b.get_position()
    h_new = h_old - delta
    fig.set_size_inches(W, h_new)
    # (b) keeps its distance from the bottom, (a) keeps its distance from the top
    b.set_position([pb.x0, pb.y0 * h_old / h_new, pb.width, pb.height * h_old / h_new])
    a.set_position([pa.x0, (pa.y0 * h_old - delta) / h_new, pa.width, pa.height * h_old / h_new])

    # widen the plot areas: right edge so the "10,000" / "1,000" tick label ends at the right margin
    # (the pre-alignment key row had made constrained layout reserve extra room on the right)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    over = max(t.get_window_extent(r).x1 - axx.get_window_extent(r).x1
               for axx in axes for t in axx.get_xticklabels() if t.get_text())
    x1 = (fig.bbox.width - RIGHT_PAD_PT * fig.dpi / 72 - over) / fig.bbox.width
    yl = b.yaxis.label
    _, _, desc = r.get_text_width_height_descent(yl.get_text(), yl.get_fontproperties(), ismath=False)
    x_lab = yl.get_window_extent(r).x0 + desc
    for axx in axes:
        p = axx.get_position()
        x0 = p.x0
        if axx is a:  # (a) has only the short D1/D2 row labels on the left: start right after them
            tick_w = max(t.get_window_extent(r).width for t in a.get_yticklabels())
            tk = a.yaxis.get_major_ticks()[0]
            x0 = (x_lab + tick_w + (tk.get_pad() + tk.get_tick_padding()) * fig.dpi / 72) / fig.bbox.width
        axx.set_position([x0, p.y0, x1 - x0, p.height])

    # both titles and the (b) key share one left edge: the y-label's ink edge
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for axx in axes:
        ab = axx.get_window_extent(r)
        axx._left_title.set_x((x_lab - ab.x0) / ab.width)
        axx._left_title._house_outer = True
    bb = b.get_window_extent(r)
    for _ in range(3):  # converge the first key line's left end onto the same edge
        fig.canvas.draw()
        dx = x_lab - leg.get_lines()[0].get_window_extent(fig.canvas.get_renderer()).x0
        anc = leg.get_bbox_to_anchor().transformed(b.transAxes.inverted())
        leg.set_bbox_to_anchor((anc.x0 + dx / bb.width, 1.0), transform=b.transAxes)

    save(fig, name)
    return fig


if __name__ == "__main__":
    from common import load_json

    draw(load_json("analysis.json"), load_json("dataset_summary.json"))
