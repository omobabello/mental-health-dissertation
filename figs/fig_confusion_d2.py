"""fig-confusion-d2: D2 test-set confusion matrices (counts + row-normalised %) for the best classical
model and every transformer present.

Ported 1:1 from 05_figures.fig_confusion (same models, matrix layout, counts, row normalisation,
0.55 white-text switch, labels and note); only the styling changed.
Standalone:  cd experiments && .venv/bin/python -m figs.fig_confusion_d2
"""
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, PowerNorm

import fig_style
from fig_style import FAMILY, FS_ANNOT, FS_LABEL, ORDER, T, W, panel_title, plt, save

# the chart design spec s6 ramp exactly (single hue, #f4f6f9 -> accent #17506b, same CMAP as fig_errors_d2).
# A square-root value mapping (PowerNorm 0.5, monotone: darker = higher row share) darkens the upper
# half so every white-text cell (> 0.55, the v1/spec switch) is >= 4.59:1 against white; ink text on
# cells below 0.55 stays >= 3.2:1 at the boundary (real ink cells 18.5 % / 28.7 % are far lighter).
CMAP = LinearSegmentedColormap.from_list("house_seq", ["#f4f6f9", T.ACCENT])
NORM = PowerNorm(0.5, vmin=0, vmax=1)
CLASSES = ["Control", "Depression"]


def _ink_x0(fig, text):
    """Display x (px) of the first inked column of `text` (bbox x0 includes glyph-free line space)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bb = text.get_window_extent(r)
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    H = buf.shape[0]
    y0, y1 = int(max(0, H - bb.y1)), int(min(H, H - bb.y0 + 1))
    x0, x1 = int(max(0, bb.x0)), int(min(buf.shape[1], bb.x1 + 1))
    cols = np.where((buf[y0:y1, x0:x1].min(axis=2) < 200).any(axis=0))[0]
    return bb.x0 if cols.size == 0 else x0 + cols[0]


def draw(an, ds=None, name="fig-confusion-d2"):
    cond = "D2"
    d = an["conditions"][cond]
    ms = [d["best_classical"]] + [m for m in ORDER if m in d["models"] and FAMILY[m] == "transformer"]
    n = len(ms)

    # Square panels sized to fit the 15.5 cm width (1 panel now, up to 4 once transformers land),
    # left-aligned at the house margin like the sibling charts; only the first panel carries the
    # row ticks and "Actual". Row ticks are horizontal; rotated only as a width fallback at n >= 4.
    rot_y = n >= 4
    left_m = 0.44 if rot_y else 0.87          # room for "Actual" + row ticks
    gap, right_m = 0.28, 0.16                 # gap between panels | >= 8 pt clearance at right edge
    side = min(2.2 if n == 1 else 1.6, (W - left_m - right_m - (n - 1) * gap) / n)  # inches per matrix
    fs_cell = FS_ANNOT + (1.5 if n == 1 else 0.5)
    bottom, top = 0.52, 0.60                  # x ticks + "Predicted" | panel title + note line (>= 8 pt apart)
    h = bottom + side + top
    fig = plt.figure(figsize=(W, h), layout="none")
    # n == 1: centre the block ("Actual" .. matrix right edge) on the page instead of leaving the right
    # half empty; n > 1 rows already fill the width and stay at the house left margin.
    x0 = (W - side - left_m) / 2 + left_m if n == 1 else left_m
    axes = []
    for k in range(n):
        ax = fig.add_axes([(x0 + k * (side + gap)) / W, bottom / h, side / W, side / h])
        axes.append(ax)

    for k, (ax, m) in enumerate(zip(axes, ms)):
        c = d["models"][m]["test"]["confusion"]
        mat = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
        norm = mat / mat.sum(1, keepdims=True)
        ax.imshow(norm, cmap=CMAP, norm=NORM)
        # 2 px white gutters between cells (house rule: no outlines, white gaps)
        for v in (0.5,):
            ax.axhline(v, color=T.PAPER, lw=2)
            ax.axvline(v, color=T.PAPER, lw=2)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{mat[i, j]}\n({norm[i, j]*100:.1f}%)", ha="center", va="center",
                        fontsize=fs_cell, linespacing=1.25,
                        color="white" if norm[i, j] > 0.55 else T.INK)
        ax.set_xticks([0, 1], CLASSES)
        if rot_y:
            ax.set_yticks([0, 1], CLASSES if k == 0 else [], rotation=90, va="center")
        else:
            ax.set_yticks([0, 1], CLASSES if k == 0 else [], rotation=0, ha="right", va="center")
        ax.tick_params(length=0, pad=3)       # theme defaults: tick labels slate 8 pt
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xlabel("Predicted", labelpad=4)   # theme default: axis labels in ink
        panel_title(ax, m)
        ax.title.set_position((0.0, 1.0))
        if n > 1:
            # One rule for a row of panels: every title starts at its own matrix's left edge
            # (stops save() moving panel 1's title out to the "Actual" label).
            ax._left_title._house_outer = True
    # n == 1: the single title is flush with the "Actual" edge (house rule, done by save()),
    # i.e. the same left margin as the note and the sibling charts.
    axes[0].set_ylabel("Actual", labelpad=4)

    # One visual left edge for note, first title and "Actual": the rotated label's bbox includes empty
    # line-height space before its first glyph, so measure where the ink actually starts.
    fig_style._fit_titles(fig)                # same alignment save() applies (idempotent afterwards)
    r = fig.canvas.get_renderer()
    x_edge = _ink_x0(fig, axes[0].yaxis.label)            # px
    ab = axes[0].get_window_extent(r)
    t0 = axes[0]._left_title
    t0.set_x((x_edge - ab.x0) / ab.width)
    if n > 1:
        for ax in axes[1:]:
            ax._left_title.set_x(0.0)
    # v1 suptitle kept verbatim as a quiet slate note above the block, >= 8 pt clear of the panel title.
    fig.text(x_edge / fig.bbox.width, 1 - 0.06 / h, "D2 test set (n = 1,501); row-normalised percentages",
             ha="left", va="top", fontsize=FS_LABEL, color=T.SLATE)
    save(fig, name)
    return fig


if __name__ == "__main__":
    import sys

    from common import load_json

    an = load_json("analysis.json")
    if "--mock6" in sys.argv:          # layout test: fake the three transformers (never saved over the real fig)
        import copy

        an = copy.deepcopy(an)
        dd = an["conditions"]["D2"]["models"]
        base = dd[an["conditions"]["D2"]["best_classical"]]
        for t in ["DistilBERT", "MentalBERT", "BERT"]:
            dd[t] = copy.deepcopy(base)
        import os
        from pathlib import Path

        import fig_style
        out = Path(os.environ.get("MOCK_OUT", "/tmp/fig-mock"))
        fig_style.OUT, fig_style.PNG = out, out / "_png"
        draw(an, None, name="mock-confusion-d2")
    else:
        draw(an, load_json("dataset_summary.json"))
