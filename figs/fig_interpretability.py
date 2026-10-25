"""fig-interpretability: top-20 explanation stability (mean pairwise Jaccard) and lexicon
plausibility (share of top-20) per model, for D1 and D2.

Ported 1:1 from 05_figures.fig_interp (same fields, `.get(key) or 0` fallback, D1/D2 order,
model set from D1, 0-1 axis, panel titles, model legend); only the styling changed.
Pass 2 (the chart design spec 2b): both axis lines, value labels on each bar, a light band behind each
condition group. A missing value (e.g. a transformer without interpretability output) draws no bar
and a small "n/a" at the baseline instead of a misleading zero-height bar.
Standalone:  cd experiments && .venv/bin/python -m figs.fig_interpretability  [--mock6]
"""
import re
import sys

import numpy as np
from matplotlib.patches import Patch

from fig_style import (FAMILY, FS_ANNOT, FS_LABEL, FS_TITLE, ORDER, T, W, COLORS, axis_lines, models_in,
                       panel_title, plt, save)

PANELS = [("stability_jaccard", "(a) Top-20 stability (mean pairwise Jaccard)"),
          ("plausibility_top20", "(b) Lexicon plausibility (share of top-20)")]
YLABELS = ["Mean pairwise Jaccard", "Share of top-20 in lexicon"]  # axis titles (round-10 marking)
CONDS = ["D1", "D2"]
GAP_PT = 1.5          # constant white gap between adjacent bars (~2 diagram px)
BAND_PAD = 0.05       # condition band extends this far (data units) beyond the outer bars
LEG_GAP_PT = 6        # legend row -> panel-title gap (figure-level key, not part of the (a) title)
HATCH = "////"        # transformer bars only (family identity on the data marks, no text)
plt.rcParams["hatch.linewidth"] = 0.7

ALT = ("Two panels of grouped bars sharing a y axis from 0 to 1. Panel (a): top-20 explanation "
       "stability (mean pairwise Jaccard); panel (b): lexicon plausibility (share of top-20). Each "
       "panel has a group for D1 and a group for D2, with one bar per model ({models}).")


def _ordered(cond_d):
    """models_in() order, but classical models first, then transformers (colours unchanged)."""
    ms = models_in(cond_d)
    return sorted(ms, key=lambda m: (FAMILY[m] != "classical", ORDER.index(m)))


def _bar_style(m):
    if FAMILY[m] == "transformer":
        return dict(facecolor=COLORS[m], hatch=HATCH, edgecolor=(1, 1, 1, 0.55), linewidth=0)
    return dict(facecolor=COLORS[m], linewidth=0)


def _title_to_ticks(fig, axes):
    """Panel title flush with the left edge of the y-axis title (or the tick labels when it has none)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for ax in axes:
        ab = ax.get_window_extent(r)
        x0 = min(t.get_window_extent(r).x0 for t in ax.get_yticklabels() if t.get_visible() and t.get_text())
        if ax.yaxis.label.get_text():
            x0 = min(x0, ax.yaxis.label.get_window_extent(r).x0)
        ax._left_title.set_x((x0 - ab.x0) / ab.width)


def _const_gap(fig, axes, bars, n, group):
    """Bar width = slot - GAP_PT (converted to data units through the axes width)."""
    fig.canvas.draw()
    ax = axes[0]
    w_pt = ax.get_window_extent().width * 72 / fig.dpi
    gap = GAP_PT * (ax.get_xlim()[1] - ax.get_xlim()[0]) / w_pt
    slot = group / n
    ax._slot_pt = (slot - gap) * w_pt / (ax.get_xlim()[1] - ax.get_xlim()[0])
    for rects in bars:
        for rc in rects:
            cx = rc.get_x() + rc.get_width() / 2
            rc.set_width(slot - gap)
            rc.set_x(cx - (slot - gap) / 2)


def _fit_labels(fig, axes, labels):
    """Value labels sit horizontally when they fit the bar; with many models (narrow bars) they turn
    vertical so neighbours never collide (size stays FS_ANNOT, the 8 pt print minimum)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    slot_px = axes[0]._slot_pt * fig.dpi / 72
    widest = max(t.get_window_extent(r).width for t in labels)
    if widest > slot_px + (GAP_PT - 2.5) * fig.dpi / 72:          # keep >= 2.5 pt between neighbours
        for t in labels:
            t.set_rotation(90)


def draw(an, ds=None, name="fig-interpretability"):
    ms = _ordered(an["conditions"]["D1"])           # same model set as v1 (taken from D1)
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.95), sharey=True)
    x = np.arange(len(CONDS))
    group = 0.56 if len(ms) <= 3 else 0.82          # grouped-bar block width per condition
    wdt = group / len(ms)
    bars, labels = [], []
    for ax, (key, title) in zip(axes, PANELS):
        for xc in x:                                 # light band = one condition group
            ax.axvspan(xc - group / 2 - BAND_PAD, xc + group / 2 + BAND_PAD, color=T.PANEL, lw=0, zorder=0)
        for i, m in enumerate(ms):
            raw = [an["conditions"][c]["models"].get(m, {}).get(key) for c in CONDS]
            xs = x + i * wdt - group / 2 + wdt / 2
            bars.append(ax.bar(xs, [v or 0 for v in raw], wdt, label=m, zorder=2, **_bar_style(m)))
            for xi, v in zip(xs, raw):
                txt = "n/a" if v is None else f"{v:.2f}"
                labels.append(ax.annotate(txt, (xi, v or 0), xytext=(0, 2.5), textcoords="offset points",
                                          ha="center", va="bottom", fontsize=FS_ANNOT,
                                          color=T.INK if v is not None else T.SLATE, zorder=4))
        axis_lines(ax, grid="y")
        ax.set_xticks(x, CONDS)
        ax.tick_params(axis="x", length=0, labelcolor=T.INK, pad=5)
        ax.set_xlim(-0.5, len(CONDS) - 0.5)
        ax.set_ylim(0, 1)
        ax.set_yticks(np.linspace(0, 1, 6), [f"{v:.1f}" for v in np.linspace(0, 1, 6)])
        ax.yaxis.set_tick_params(labelleft=True)
        ax.set_ylabel(YLABELS[PANELS.index((key, title))], fontsize=FS_LABEL)
        ax.set_xlabel("Condition", fontsize=FS_LABEL)
        panel_title(ax, title, size=FS_TITLE - 0.5)

    # legend: bar-shaped keys (8 x 8 pt), one frameless row centred over both panels
    hl = 8 / FS_LABEL
    handles = [Patch(label=m, **_bar_style(m)) for m in ms]
    leg = fig.legend(handles=handles, ncol=len(ms), loc="lower left", frameon=False,
                     handlelength=hl, handleheight=hl, handletextpad=0.45, columnspacing=1.2,
                     borderpad=0, borderaxespad=0, bbox_to_anchor=(0, 1))
    fig.canvas.draw()
    leg_h = leg.get_window_extent().height                       # px
    top_px = fig.bbox.height - leg_h - (LEG_GAP_PT + 2) * fig.dpi / 72
    fig.get_layout_engine().set(rect=(0, 0, 1, top_px / fig.bbox.height))
    _title_to_ticks(fig, axes)
    _const_gap(fig, axes, bars, len(ms), group)
    _title_to_ticks(fig, axes)
    _fit_labels(fig, axes, labels)
    fig.set_layout_engine("none")                                # freeze; place legend exactly
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    tb = axes[0]._left_title.get_window_extent(r)
    leg.set_bbox_to_anchor((tb.x0 / fig.bbox.width, (tb.y1 + LEG_GAP_PT * fig.dpi / 72) / fig.bbox.height),
                           transform=fig.transFigure)
    fig.canvas.draw()
    # centre the visible key row (first key -> last label) over the span (a) title left .. (b) axes right
    lb = leg.get_window_extent(r)
    kx0 = leg.legend_handles[0].get_window_extent(r).x0
    kx1 = max(t.get_window_extent(r).x1 for t in leg.get_texts())
    span_mid = (tb.x0 + axes[1].get_window_extent(r).x1) / 2
    shift = span_mid - (kx0 + kx1) / 2
    leg.set_bbox_to_anchor(((lb.x0 + shift) / fig.bbox.width,
                            (tb.y1 + LEG_GAP_PT * fig.dpi / 72) / fig.bbox.height), transform=fig.transFigure)
    svg = save(fig, name)
    _add_title_desc(svg, ms)
    return fig


def _add_title_desc(svg, ms):
    """Accessible <title>/<desc> (ignored by check_parity; does not change the rendering)."""
    s = svg.read_text()
    alt = ALT.format(models=", ".join(ms))
    s = re.sub(r"(<svg [^>]*>)", r"\1\n <title>Interpretability: stability and plausibility</title>\n <desc>"
               + alt.replace("\\", r"\\") + "</desc>", s, count=1)
    svg.write_text(s)


if __name__ == "__main__":
    from common import load_json

    an = load_json("analysis.json")
    if "--mock6" in sys.argv:   # the chart design spec section 6: 6-model stress test (writes a _mock6 file)
        import copy

        an = copy.deepcopy(an)
        for cond in CONDS:
            base = an["conditions"][cond]["models"]["LR"]
            for k, m in enumerate(["DistilBERT", "MentalBERT", "BERT"]):
                an["conditions"][cond]["models"][m] = {
                    **base, "family": "transformer",
                    "stability_jaccard": 0.5 + k / 10, "plausibility_top20": 0.35 + k / 20}
        an["conditions"]["D2"]["models"]["BERT"].pop("plausibility_top20")   # exercises the `or 0` path
        draw(an, None, name="fig-interpretability_mock6")
    else:
        draw(an, load_json("dataset_summary.json"))
