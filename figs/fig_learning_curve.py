"""fig-learning-curve: test macro-F1 vs number of training posts (log x), D1 and D2.

Ported 1:1 from 05_figures.fig_learning_curve (same learning_curve records, mean per n_train,
log x-axis, axis labels, '(a) D1' / '(b) D2' titles, fixed model ORDER); only the styling changed.
The per-panel legend is replaced by per-panel direct labels at the line ends (same model names,
same count), nudged apart so close curves never collide.
Standalone:  cd experiments && .venv/bin/python -m figs.fig_learning_curve   [--mock6]
"""
import pandas as pd
import numpy as np
from matplotlib.ticker import FixedLocator, FormatStrFormatter, MaxNLocator, NullLocator
from matplotlib.transforms import ScaledTranslation

from fig_style import (COLORS, FAMILY, FS_ANNOT, FS_TICK, MARK, ORDER, T, W, axis_lines, direct_label,
                       panel_title, plt, save)


def _spread(ys, min_gap):
    """Nudge label y-positions apart by at least min_gap (data units), keeping their mean."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    pos = [ys[i] for i in order]
    for _ in range(50):
        moved = False
        for k in range(1, len(pos)):
            if pos[k] - pos[k - 1] < min_gap:
                mid = (pos[k] + pos[k - 1]) / 2
                pos[k - 1], pos[k] = mid - min_gap / 2, mid + min_gap / 2
                moved = True
        if not moved:
            break
    out = [0.0] * len(ys)
    for k, i in enumerate(order):
        out[i] = pos[k]
    return out


def _declutter_ticks(fig, axes, pad_pt=None):
    """Safety net: if neighbouring x tick labels still come closer than ~0.3 em (a real collision) (pad_pt, in POINTS,
    converted at the canvas dpi) -- e.g. narrow panels when many long direct labels take the right
    margin -- drop the LAST label one line lower (staggered) instead of blanking one, so every
    sampled size stays labelled under its own tick."""
    pad_pt = 0.3 * FS_TICK if pad_pt is None else pad_pt
    for _ in range(2):
        fig.canvas.draw()
        fig.canvas.draw()                  # second pass: constrained layout has settled
        r = fig.canvas.get_renderer()
        pad_px = pad_pt * fig.dpi / 72
        done = True
        for ax in axes:
            labs = ax.xaxis.get_ticklabels()
            boxes = [t.get_window_extent(r) for t in labs]
            if any(a.x1 + pad_px > b.x0 for a, b in zip(boxes, boxes[1:])) and not getattr(ax, "_lc_stag", 0):
                lab = labs[-1]
                lab.set_transform(lab.get_transform()
                                  + ScaledTranslation(0, -FS_TICK * 1.15 / 72, fig.dpi_scale_trans))
                ax._lc_stag = 1
                done = False
        if done:
            return


def _freeze_yticks(fig, ax, ymax, min_pt=4):
    """Freeze the y ticks. Make sure one tick sits at or above the data maximum (so a peak never
    floats above the last gridline), and drop the lowest one if its label would crowd the x spine."""
    fig.canvas.draw()
    y0, y1 = ax.get_ylim()
    loc = ax.yaxis.get_major_locator()
    all_ticks = list(loc.tick_values(y0, y1))
    step = all_ticks[1] - all_ticks[0]
    tol = 0.1 * step                                      # a peak within 0.1 step of a gridline sits on it
    while max(all_ticks) < ymax - tol:
        all_ticks.append(all_ticks[-1] + step)
    top = min(t for t in all_ticks if t >= ymax - tol)
    if top > y1:
        y1 = top + 0.15 * step
        ax.set_ylim(y0, y1)
    ticks = [t for t in all_ticks if y0 <= t <= y1]
    pt_per = ax.get_window_extent().height * 72 / fig.dpi / (y1 - y0)
    if ticks and (ticks[0] - y0) * pt_per < min_pt:
        ticks = ticks[1:]
    ax.yaxis.set_major_locator(FixedLocator(ticks))


def draw(an, ds=None, name="fig-learning-curve"):
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.6), sharex=True)
    ends_by = {}                                          # cond -> [(model, x_end, y_end)]
    ymax_by = {}
    for ax, cond in zip(axes, ["D1", "D2"]):
        lc = an["conditions"][cond]["learning_curve"]
        models = [m for m in ORDER if m in lc]
        n = len(models)
        ends, xs_all, ymax = [], set(), -np.inf
        series = {m: pd.DataFrame(lc[m]).groupby("n_train").macro_f1.mean().sort_index() for m in models}
        curves = {m: v for m, v in series.items() if len(v) >= 2}   # single-size models are points
        if len(curves) >= 2:
            # light envelope between the lowest and highest curve at each shared size: makes the
            # spread between models (how much the choice of model matters) readable at a glance
            env = pd.concat(curves, axis=1).dropna()
            if len(env) >= 2:
                ax.fill_between(env.index, env.min(axis=1), env.max(axis=1), color="#dde8ee",
                                lw=0, zorder=1.2)
        for i, m in enumerate(models):
            pts = series[m]
            # later series sit on top with very slightly smaller markers, so a coincident earlier
            # series (e.g. LR under SVM) still shows a thin coloured ring -- no visual emphasis
            ms = 6.2 - 0.7 * i / max(n - 1, 1)
            ax.plot(pts.index, pts.values, color=COLORS[m], lw=2.1, solid_capstyle="round",
                    solid_joinstyle="round", zorder=3 + i * 0.1, label=m)
            ax.plot(pts.index, pts.values, ls="none", marker=MARK[FAMILY[m]], color=COLORS[m], ms=ms,
                    mec="white", mew=1.1, zorder=3.5 + i * 0.1)
            ends.append((m, pts.index[-1], pts.values[-1], len(pts)))
            xs_all.update(pts.index)
            ymax = max(ymax, pts.values.max())
        ends_by[cond], ymax_by[cond] = ends, ymax
        ax.set_xscale("log")
        xs = sorted(xs_all)
        ax.xaxis.set_major_locator(FixedLocator(xs))       # ticks at the sampled sizes, not 10^k
        ax.xaxis.set_minor_locator(NullLocator())
        # compact thousands (1k, 2k, 4k) so the full-set label "6,999" no longer collides with 4,000
        ax.set_xticklabels([f"{x:,}" if x < 1000 or x == xs[-1] else f"{x / 1000:g}k" for x in xs])
        ax.set_xlim(xs[0] / 1.2, xs[-1] * 1.12)          # direct labels hang in the right margin
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))   # same step rule both panels
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.margins(y=0.06)
        ax.set_xlabel("Training posts (log scale)")
        ax.set_ylabel("Macro-F1 (test)")
        axis_lines(ax, grid="both")                      # both axis lines + light x/y grid (design spec 2b)
        panel_title(ax, f"({'ab'[cond == 'D2']}) {cond}")
    for ax, cond in zip(axes, ["D1", "D2"]):
        _freeze_yticks(fig, ax, ymax_by[cond])
    fig.canvas.draw()
    fig.canvas.draw()                                     # layout settled before sizing labels
    for ax, cond in zip(axes, ["D1", "D2"]):
        ends = ends_by[cond]
        # direct labels at the line ends (replace v1's per-panel legend), de-collided; a label
        # moved more than half a line-height gets a thin leader back to its own end point
        y0, y1 = ax.get_ylim()
        h_pt = ax.get_window_extent().height * 72 / fig.dpi
        gap = (y1 - y0) * (FS_ANNOT * 1.25) / h_pt
        singles = [e for e in ends if e[3] == 1]
        if len(singles) == 1:
            # a model scored at one size only (e.g. the transformer at the full set) is a lone
            # point, not a curve: name it right above its marker (or left of it if the panel top is
            # too close) so it does not widen the right-hand label margin
            m, x, y, _ = singles[0]
            room_pt = (y1 - y) * h_pt / (y1 - y0)
            if room_pt > 2.2 * FS_ANNOT:
                direct_label(ax, x, y, m, dx=0, dy=7, ha="center", va="bottom", annotation_clip=False,
                             fontweight="medium", color=COLORS[m])
            else:
                direct_label(ax, x, y, m, dx=-8, ha="right", annotation_clip=False,
                             fontweight="medium", color=COLORS[m])
            ends = [e for e in ends if e[3] != 1]
        for (m, x, y, _), yl in zip(ends, _spread([e[2] for e in ends], gap)):
            dy = (yl - y) * h_pt / (y1 - y0)
            kw = {}
            if abs(dy) > 3:                               # > ~0.3 line: tie it back
                kw["arrowprops"] = dict(arrowstyle="-", lw=0.8, color=COLORS[m], shrinkA=2, shrinkB=5,
                                        relpos=(0, 0.5))
            direct_label(ax, x, y, m, dx=10, dy=dy, annotation_clip=False, fontweight="medium",
                         color=COLORS[m], **kw)
    _declutter_ticks(fig, axes)
    save(fig, name)
    return fig


if __name__ == "__main__":
    import sys

    from common import load_json

    an = load_json("analysis.json")
    if "--mock6" in sys.argv:                     # layout test only: fake transformers, never the report file
        import copy
        import os
        from pathlib import Path

        import fig_style

        an = copy.deepcopy(an)
        fake = {"DistilBERT": 0.012, "MentalBERT": 0.025, "BERT": 0.018, "RF": -0.03}
        for c in ("D1", "D2"):
            lc = an["conditions"][c]["learning_curve"]
            for m, dv in fake.items():
                lc[m] = [{"n_train": r["n_train"], "macro_f1": r["macro_f1"] + dv * (0.5 + i / 10)}
                         for i, r in enumerate(lc["SVM"])]
        fig_style.OUT = Path(os.environ.get("MOCK_OUT", "/tmp/fig-mock"))   # keep mocks out of figures/
        fig_style.PNG = fig_style.OUT
        draw(an, None, name="mock6-fig-learning-curve")
    else:
        draw(an, load_json("dataset_summary.json"))
