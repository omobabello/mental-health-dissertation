"""fig-errors-d2: D2 error analysis - (a) false-positive rate by negative source, (b) error rate by
post length, one row per model.

Ported 1:1 from 05_figures.fig_errors (same fields, % transforms, sorted sources, band order, n= labels,
1-decimal cell values); only the styling changed. Colour is ONE shared scale for both panels (both are %),
with the ramp top chosen so every cell label clears WCAG AA 4.5:1 in ink or white (label colour picked
per cell by computed contrast, not a fixed % switch point).
Standalone:  cd experiments && .venv/bin/python -m figs.fig_errors_d2   [--mock6]
"""
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb

from fig_style import FS_ANNOT, FS_TICK, T, W, models_by_family, panel_title, plt, save

CMAP = LinearSegmentedColormap.from_list("house_seq", ["#f4f6f9", T.ACCENT])


def _matrices(d):
    ms = models_by_family(d)
    e0 = d["errors"][ms[0]]
    srcs = sorted(e0["neg_count_by_source"])
    mat = np.array([[d["errors"][m]["fp_by_source"].get(s, 0) / d["errors"][m]["neg_count_by_source"][s] * 100
                     for s in srcs] for m in ms])
    bands = list(e0["error_rate_by_length"])
    mat2 = np.array([[d["errors"][m]["error_rate_by_length"][b] * 100 for b in bands] for m in ms])
    cols2 = [f"{b.replace('-', '–')}\n(n={e0['n_by_length'][b]})" for b in bands]   # en dash, as in the text
    # "mentalhealth" is wider than its cell and collided with "lonely": wrap it like Fig. 7's long names
    cols1 = [{"mentalhealth": "mental-\nhealth"}.get(s, s) for s in srcs]
    # v1 titles verbatim; pre-wrapped at a word boundary so both panels share the same two-line head
    return ms, [(mat, cols1, "(a) False-positive rate by\nnegative source (%)"),
                (mat2, cols2, "(b) Error rate by post length\n(tokens, %)")]


def _lum(rgb):
    f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (f(c) for c in rgb[:3])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(a, b):
    la, lb = sorted((_lum(a), _lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


_WHITE, _INK = (1.0, 1.0, 1.0), to_rgb(T.INK)


def _label_color(t):
    """Ink or white, whichever contrasts more with the cell colour CMAP(t)."""
    c = CMAP(t)
    cw, ci = _contrast(_WHITE, c), _contrast(_INK, c)
    return ("white", cw) if cw > ci else (T.INK, ci)


def _shared_vmax(values, target=4.5):
    """Smallest ramp top (>= data max) at which every label clears `target` contrast; keeps cells out
    of the mid-ramp dead zone where neither ink nor white reaches AA. Falls back to the best found."""
    vals = np.asarray(values, float).ravel()
    top = max(1.0, vals.max())
    best = (0, top)
    for f in np.linspace(1.0, 1.6, 121):
        vmax = top * f
        worst = min(_label_color(v / vmax)[1] for v in vals)
        if worst >= target:
            return vmax
        best = max(best, (worst, vmax))
    return best[1]


def _first_to_last_baseline(fig, texts, linespacing):
    """Max distance (pt) from first-line to last-line baseline over `texts`, from matplotlib's own layout
    (glyph-dependent, so it is measured on the real labels rather than estimated)."""
    r = fig.canvas.get_renderer()
    best = 0.0
    for txt in texts:
        probe = fig.text(0, 0, txt, fontsize=FS_TICK, linespacing=linespacing)
        _, lines, _ = probe._get_layout(r)
        probe.remove()
        best = max(best, lines[0][-1][1] - lines[-1][-1][1])
    return best * 72 / fig.dpi


def _flush_titles(fig):
    """Panel titles flush with the y tick-label edge (no y-axis label here, so save() cannot do it)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for ax in fig.axes:
        ab = ax.get_window_extent(r)
        x0 = min(t.get_window_extent(r).x0 for t in ax.get_yticklabels() if t.get_text())
        ax._left_title.set_x((x0 - ab.x0) / ab.width)
        ax._left_title._house_outer = True


def draw(an, ds=None, name="fig-errors-d2"):
    d = an["conditions"]["D2"]
    ms, panels = _matrices(d)
    n = len(ms)
    fig, axes = plt.subplots(1, 2, figsize=(W, 1.35 + 0.34 * n))
    vmax = _shared_vmax(np.concatenate([M.ravel() for M, _, _ in panels]))   # one scale: both panels are %
    for ax, (M, cols, title) in zip(axes, panels):
        im = ax.imshow(M, cmap=CMAP, aspect="auto", vmin=0, vmax=vmax)
        # 2 px white gutters between cells (house rule: no outlines, white gaps)
        for v in np.arange(0.5, M.shape[1] - 1, 1.0):
            ax.axvline(v, color=T.PAPER, lw=2)
        for v in np.arange(0.5, M.shape[0] - 1, 1.0):
            ax.axhline(v, color=T.PAPER, lw=2)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=FS_ANNOT + 0.5,
                        color=_label_color(M[i, j] / vmax)[0])
        ax.set_xticks(range(len(cols)), cols, linespacing=1.15, va="baseline")
        ax.set_yticks(range(n), ms)
        # categorical labels in ink on both axes (matches fig-macro-f1). va="baseline" pins the LAST line's
        # baseline, so push multi-line panels down by their extra lines: every panel's FIRST line then sits on
        # one shared baseline, a cap height + 3 pt below the cells (glyph-independent, unlike va="top").
        extra = _first_to_last_baseline(fig, cols, 1.15)
        ax.tick_params(length=0, pad=3 + 0.72 * FS_TICK + extra, labelcolor=T.INK)
        ax.tick_params(axis="y", pad=3)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        panel_title(ax, title)
    # colour key: one shared percentage scale for both panels
    cb = fig.colorbar(im, ax=list(axes), fraction=0.025, pad=0.02, aspect=18)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=FS_TICK, length=2, color=T.LINE, labelcolor=T.MUTED)
    cb.set_label("Rate (%)", fontsize=FS_TICK, color=T.MUTED)
    _flush_titles(fig)
    save(fig, name)
    return fig


if __name__ == "__main__":
    import sys

    from common import load_json

    an = load_json("analysis.json")
    if "--mock6" in sys.argv:          # layout test: fake the three transformers (never saved over the real fig)
        import copy
        import os
        from pathlib import Path

        import fig_style

        an = copy.deepcopy(an)
        dd = an["conditions"]["D2"]
        base = dd["best_classical"]
        for t in ["DistilBERT", "MentalBERT", "BERT"]:
            dd["models"][t] = copy.deepcopy(dd["models"][base])
            dd["errors"][t] = copy.deepcopy(dd["errors"][base])
        out = Path(os.environ.get("MOCK_OUT", "/tmp/fig-mock"))
        fig_style.OUT, fig_style.PNG = out, out / "_png"
        draw(an, None, name="mock-errors-d2")
    else:
        draw(an, load_json("dataset_summary.json"))
