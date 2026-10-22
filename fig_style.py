"""fig_style.py - shared matplotlib theme for every results chart.

Usage inside experiments/figs/<fig>.py:

    from fig_style import plt, ORDER, COLORS, FAMILY, MARK, COND_LABEL, W, models_in, \
        panel_title, value_label, threshold, model_legend, save, T
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.9))
    ...
    save(fig, "fig-xyz")     # figures/fig-xyz.svg + figures/fig-xyz.png

Identity encoding (used across chapters - do not change the mapping):
  colour = model (fixed ORDER, never cycled); marker = family (o classical, s transformer).
"""
import os
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# ------------------------------------------------------------------ paths
EXP = Path(__file__).resolve().parent
# Output dirs: figures/ inside the repository by default; set FIG_OUT / FIG_PNG to write elsewhere.
OUT = Path(os.environ.get("FIG_OUT", EXP / "figures"))
PNG = Path(os.environ.get("FIG_PNG", EXP / "figures"))


# ------------------------------------------------------------------ tokens (same as the dissertation diagram kit T)
class T:
    INK = "#1c1f24"      # titles, value labels, axis labels
    SLATE = "#5b6472"    # tick labels, secondary annotations
    MUTED = "#6b7380"    # tertiary annotations (seed sd, notes)
    LINE = "#8a929e"     # spines, ticks, reference lines (neutral)
    HAIR = "#cfd4dc"     # rules / dividers
    GRID = "#e6e9ee"     # y-grid only
    PAPER = "#ffffff"
    PANEL = "#f6f7f9"
    ACCENT = "#17506b"   # ONE focal reference per chart: decision threshold, alpha line, etc.
    ACCENT_TINT = "#eaf1f4"
    WARM = "#b4532a"     # sparing secondary (e.g. a second reference); never a series colour
    WARM_TINT = "#fbf0ea"
    TEAL = "#2a7a72"     # semantic: classical family (diagrams + non-model categories), the chart design spec 2b
    TEAL_TINT = "#e6f2f0"


# Semantic (non-model) categories - same hexes as the dissertation diagram kit HUES. Use for things that are NOT
# a single model: classes, families, sources, lenses. Model identity always stays COLORS[model].
SEM = {"positive": T.WARM, "positive_tint": T.WARM_TINT, "navy": T.ACCENT, "navy_tint": T.ACCENT_TINT,
       "classical": T.TEAL, "classical_tint": T.TEAL_TINT, "transformer": T.WARM, "transformer_tint": T.WARM_TINT}


# ------------------------------------------------------------------ identity encoding (from 05_figures.py)
ORDER = ["LR", "DistilBERT", "SVM", "RF", "MentalBERT", "BERT"]
# Retuned for harmony + print: same hue identity as v1 (blue, orange, green, amber, pink, purple),
# every slot >= 3:1 on white; dataviz validator: adjacent CVD dE >= 9.9, all-pairs normal-vision
# dE 13.8 (v1: 10.5). All-pairs CVD is weak for RF/DistilBERT (4.6) exactly as in v1 - which is
# why marker shape (family) + direct labels are mandatory secondary encodings.
COLORS = {"LR": "#2272b8", "DistilBERT": "#d65028", "SVM": "#0f9070", "RF": "#b08c00",
          "MentalBERT": "#d0609a", "BERT": "#7442c8"}
COLORS_V1 = {"LR": "#2a78d6", "DistilBERT": "#eb6834", "SVM": "#1baf7a", "RF": "#eda100",
             "MentalBERT": "#e87ba4", "BERT": "#6250d6"}
FAMILY = {"LR": "classical", "SVM": "classical", "RF": "classical",
          "DistilBERT": "transformer", "MentalBERT": "transformer", "BERT": "transformer"}
MARK = {"classical": "o", "transformer": "s"}
COND_LABEL = {"D1": "D1: depression vs non-mental-health", "D2": "D2: depression vs mental-health (hard negatives)",
              "D1T": "D1T: temporal hold-out (2020)"}
# Two-tone neutrals for non-model categories (e.g. class composition): use ACCENT + its tint
# or SLATE/HAIR - never borrow a model colour for something that is not that model.

W = 6.1  # inches = 15.5 cm = the printed width, so pt sizes below are TRUE printed sizes

# ------------------------------------------------------------------ type scale (pt, printed)
FS_TICK = 8        # tick labels
FS_LABEL = 8.5     # axis labels, legend
FS_TITLE = 9       # panel titles (semibold, left)
FS_ANNOT = 8       # value labels, direct labels, threshold labels  (8 pt = hard minimum)

FONT_STACK = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]


def _check_font():
    names = {f.name for f in font_manager.fontManager.ttflist}
    found = [f for f in FONT_STACK if f in names]
    if not found or found[0] not in ("Helvetica Neue", "Helvetica", "Arial"):
        raise RuntimeError(f"fig_style: none of {FONT_STACK[:3]} found by matplotlib ({found})")
    return found[0]


FONT = _check_font()

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": FONT_STACK,
    "font.size": FS_LABEL,
    "text.color": T.INK,
    "axes.titlesize": FS_TITLE, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlepad": 8, "axes.titlecolor": T.INK,
    "axes.labelsize": FS_LABEL, "axes.labelcolor": T.INK, "axes.labelpad": 4,
    "axes.edgecolor": T.LINE, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
    "axes.facecolor": T.PAPER, "figure.facecolor": T.PAPER,
    "axes.prop_cycle": matplotlib.cycler(color=[COLORS[m] for m in ORDER]),
    "grid.color": T.GRID, "grid.linewidth": 0.6,
    "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
    "xtick.color": T.LINE, "ytick.color": T.LINE,
    "xtick.labelcolor": T.SLATE, "ytick.labelcolor": T.SLATE,
    "xtick.major.size": 3, "ytick.major.size": 3, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "ytick.major.pad": 4, "xtick.major.pad": 3,
    "lines.linewidth": 1.5, "lines.markersize": 6, "patch.linewidth": 0,
    "legend.frameon": False, "legend.fontsize": FS_LABEL, "legend.handlelength": 1.4,
    "legend.borderaxespad": 0.2, "legend.labelcolor": T.INK,
    "svg.fonttype": "none", "svg.hashsalt": "dissertation",
    "figure.dpi": 150, "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.04, "figure.constrained_layout.w_pad": 0.06,
    "savefig.bbox": None, "savefig.facecolor": T.PAPER,
})


# ------------------------------------------------------------------ data helpers
def models_in(cond_d):
    """Models present in a condition, in fixed identity ORDER (works for 3 or 6 models)."""
    return [m for m in ORDER if m in cond_d["models"]]


def models_by_family(cond_d):
    """models_in, grouped as a classical block then a transformer block (ORDER kept within each block)."""
    ms = models_in(cond_d)
    rank = {"classical": 0, "transformer": 1}
    return sorted(ms, key=lambda m: (rank.get(FAMILY.get(m), 9), ms.index(m)))


# ------------------------------------------------------------------ drawing helpers
def panel_title(ax, text, size=FS_TITLE):
    """Left-aligned semibold panel title (sentence case; '(a) ...' prefixes allowed)."""
    ax.set_title(text, loc="left", fontsize=size, fontweight="bold", color=T.INK)


HALO = dict(boxstyle="square,pad=0.12", fc=T.PAPER, ec="none")  # paper backing, NOT a stroke effect
# (path-effect halos duplicate <text> in the SVG and break check_parity counts)


def value_label(ax, x, y, text, dx=5, dy=0, ha="left", va="center", color=T.INK, size=FS_ANNOT, halo=True, **kw):
    """Direct value label offset in points from a data point (text stays ink, never series colour).
    halo=True paints a tight paper box behind it so grid/threshold lines never cut through digits."""
    if halo:
        kw.setdefault("bbox", HALO)
    kw.setdefault("zorder", 4)
    return ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                       fontsize=size, color=color, **kw)


def direct_label(ax, x, y, model, dx=5, dy=0, ha="left", va="center", size=FS_ANNOT, **kw):
    """Series name next to its line/point end, replacing a legend (name in ink, swatch = the mark)."""
    return value_label(ax, x, y, model, dx=dx, dy=dy, ha=ha, va=va, size=size, **kw)


def threshold(ax, y, label=None, color=T.ACCENT, x_text=None, ha="right", va="bottom", ls=(0, (4, 3)), lw=1):
    """The chart's ONE focal reference line (accent, dashed) with its label sitting on it."""
    ax.axhline(y, color=color, ls=ls, lw=lw, zorder=1.5)
    if label:
        xt = ax.get_xlim()[1] if x_text is None else x_text
        ax.annotate(label, (xt, y), xytext=(0, 2), textcoords="offset points", ha=ha, va=va,
                    fontsize=FS_ANNOT, color=color)


def reference(ax, y=None, x=None, color=T.LINE, ls=(0, (2, 2)), lw=0.8):
    """Neutral (non-focal) reference: chance diagonal, base rate, zero line."""
    if y is not None:
        ax.axhline(y, color=color, ls=ls, lw=lw, zorder=1)
    if x is not None:
        ax.axvline(x, color=color, ls=ls, lw=lw, zorder=1)


def model_handles(models, lw=None):
    """Legend handles carrying colour + family marker."""
    return [Line2D([], [], color=COLORS[m], marker=MARK[FAMILY[m]], ms=5.5, mec="white", mew=0.8,
                   lw=lw if lw is not None else 0, label=m) for m in models]


def model_legend(target, models, lw=None, ncol=None, loc="upper center", **kw):
    """Single frameless legend row (use on the FIGURE, above the panels, when direct labels are not
    possible). target = fig or ax."""
    ncol = ncol or len(models)
    kw.setdefault("bbox_to_anchor", (0.5, 1.0) if hasattr(target, "savefig") else None)
    return target.legend(handles=model_handles(models, lw), ncol=ncol, loc=loc, frameon=False,
                         columnspacing=1.2, handletextpad=0.4, **{k: v for k, v in kw.items() if v is not None})


def style_axes(ax, grid="y"):
    """Re-apply the house axes look (use after imshow / special axes)."""
    ax.grid(False)
    if grid:
        ax.grid(axis=grid, color=T.GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def axis_lines(ax, grid="both"):
    """House rule (the chart design spec 6): every non-heatmap chart shows BOTH axis lines - left (y) and bottom
    (x) spines in LINE grey with outward tick marks - plus a light grid: 'both' for continuous x
    continuous charts (scatter, line, ROC/PR), 'y' or 'x' (the value axis) for bar charts."""
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_color(T.LINE)
        ax.spines[s].set_linewidth(0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="both", which="major", direction="out", length=3, width=0.6, color=T.LINE)
    ax.grid(False)
    if grid:
        ax.grid(axis=grid, which="major", color=T.GRID, lw=0.6)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------ output
def _fit_titles(fig, gap_px=10, outer=True):
    """House rule for panel titles: flush with the LEFT EDGE OF THE PANEL BLOCK (the y-label),
    not the plot area, and wrapped at a word boundary if they would run into the next panel or
    off the figure (constrained layout ignores horizontal title overflow). Text is unchanged."""
    for _ in range(4):
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        changed = False
        tight = {ax: ax.get_tightbbox(r, for_layout_only=True) for ax in fig.axes}
        for ax in fig.axes:
            t = ax._left_title
            txt = t.get_text()
            if not txt:
                continue
            ab = ax.get_window_extent(r)
            if outer and getattr(t, "_house_outer", False) is False and ax.yaxis.label.get_text():
                x_frac = (ax.yaxis.label.get_window_extent(r).x0 - ab.x0) / ab.width
                t.set_x(x_frac)
                t._house_outer = True
                changed = True
                continue
            if "\n" in txt:
                continue
            bb = t.get_window_extent(r)
            right = fig.bbox.x1 - 2
            for other, ob in tight.items():
                if other is not ax and ob.x0 > ab.x1 - 1 and ob.y1 > bb.y0 - 200:
                    right = min(right, ob.x0 - gap_px)
            if bb.x1 <= right:
                continue
            cut = max(1, int(len(txt) * (right - bb.x0) / bb.width))
            head = txt[:cut]
            line1 = head.rsplit(" ", 1)[0] if " " in head else txt.split(" ")[0]
            t.set_text(line1 + "\n" + txt[len(line1):].lstrip())
            changed = True
        if not changed:
            return


def save(fig, name):
    """Write OUT/<name>.svg (deterministic, no date) and a 1800 px PNG via rsvg-convert
    (the same renderer as the diagrams, so review PNGs match the report's SVG rendering)."""
    OUT.mkdir(parents=True, exist_ok=True)
    PNG.mkdir(parents=True, exist_ok=True)
    svg = OUT / f"{name}.svg"
    _fit_titles(fig)
    fig.savefig(svg, metadata={"Date": None})
    png = PNG / f"{name}.png"
    if shutil.which("rsvg-convert"):
        subprocess.run(["rsvg-convert", "-w", "1800", "-b", "white", str(svg), "-o", str(png)], check=True)
    else:
        fig.savefig(png, dpi=300)
    plt.close(fig)
    print("saved", svg)
    return svg
