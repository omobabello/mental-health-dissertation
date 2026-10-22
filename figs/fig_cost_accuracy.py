"""fig-cost-accuracy: training time (log) vs test macro-F1, one panel per condition (D1, D2).

Ported 1:1 from 05_figures.fig_cost (same fields: efficiency.cpu_single_ms (benchmark), test.macro_f1; log x;
same axis labels, panel titles "(a) D1" / "(b) D2", model-name labels); only the styling changed.
Standalone:  cd experiments && .venv/bin/python -m figs.fig_cost_accuracy
"""
import numpy as np
from matplotlib import patheffects as pe
from matplotlib.ticker import FuncFormatter, LogLocator, MultipleLocator, NullFormatter

from fig_style import (COLORS, COND_LABEL, FAMILY, FS_ANNOT, MARK, SEM, T, W, axis_lines, direct_label, models_in,
                       panel_title, plt, save)

# uniform 2-decimal y ticks at 0.01 steps in both panels, matching the sibling fig-macro-f1
_YFMT = FuncFormatter(lambda v, _: f"{v:.2f}")
_YSTEP = 0.01

# candidate label offsets (points): right, left, above, below, side labels nudged 3 pt off a gridline,
# then diagonals; then wider rings
# (12-16 pt) so a crowded point never falls back to a label detached far from its marker
_CANDIDATES = [(7, 0, "left", "center"), (-7, 0, "right", "center"), (0, 7, "center", "bottom"),
               (0, -7, "center", "top"), (-7, -3, "right", "center"), (7, -3, "left", "center"),
               (-7, 3, "right", "center"), (7, 3, "left", "center"), (6, 6, "left", "bottom"),
               (-6, 6, "right", "bottom"), (6, -6, "left", "top"), (-6, -6, "right", "top")]
_CANDIDATES += [(sx * r, sy * r, ha, va) for r in (12, 16)
                for sx, sy, ha, va in [(1, 0, "left", "center"), (-1, 0, "right", "center"),
                                       (0.7, 0.7, "left", "bottom"), (-0.7, 0.7, "right", "bottom"),
                                       (0.7, -0.7, "left", "top"), (-0.7, -0.7, "right", "top"),
                                       (0, 1, "center", "bottom"), (0, -1, "center", "top")]]
_NEAR = 12                                        # the first 12 candidates are the close (6-7 pt) ring
_MS, _EDGE_LW = 38, 1.0                           # ONE marker area (pt^2) and white edge for every model, both
                                                  # panels: size never reads as an encoding; a near-coincident pair
                                                  # overlaps partially, the lower-F1 disc on top as a clean crescent
# paper halo as a path effect (not a bbox): gridlines and band edge rules stop at the glyphs without
# cutting rectangular notches out of the grid
_HALO = [pe.withStroke(linewidth=2.2, foreground="white", alpha=0.9)]
_BAND_ALPHA, _EDGE_ALPHA = 0.11, 0.45             # family band tint / edge rule, equal for teal and warm
_DEBUG = False
_LABEL_Z = 2                                      # labels below the markers (scatter zorder 3): never notch a disc


def _crowding(ax, pts):
    """Smallest pixel distance between two markers (smaller = more crowded panel)."""
    xy = [ax.transData.transform((x, y)) for _, x, y in pts]
    return min((np.hypot(*(a - b)) for i, a in enumerate(xy) for b in xy[i + 1:]), default=np.inf)


def _grid_px(ax):
    """Display y (px) of the major y gridlines inside the current view."""
    lo, hi = ax.get_ylim()
    return [ax.transData.transform((1, t))[1] for t in ax.yaxis.get_majorticklocs() if lo < t < hi]


def _place_labels(ax, pts, prefer=None, ms=40, focal=None):
    """Direct-label each point (highest y first), choosing the first offset whose box does not overlap a
    previously placed label, another marker, or the axes edge (works for 3 or 6 models). `prefer` maps
    model -> candidate index chosen in the other panel, tried first so a model keeps its side.
    Labels carry a thin paper path-effect halo, so any gridline or band rule stops at the glyphs; a first
    pass still prefers offsets whose box clears every horizontal gridline, a second pass relaxes that rule.
    Returns model -> chosen candidate index."""
    prefer = prefer or {}
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    axbb = ax.get_window_extent(r)
    disp = {m: ax.transData.transform((x, y)) for m, x, y in pts}
    ms = ms if isinstance(ms, dict) else {m: ms for m, _, _ in pts}
    rad = (np.sqrt(max(ms.values())) / 2 + 1) * fig.dpi / 72     # marker half-size (+ white edge) in px
    placed, chosen = [], {}
    grid = _grid_px(ax)
    # focal cue (model hues are fixed identity): best-F1 label bold ink, rest slate (+ accent ring on the disc)
    style = lambda m: (dict(color=T.INK, fontweight="bold") if m == focal else dict(color=T.INK)) | {
        "halo": False}
    for m, x, y in sorted(pts, key=lambda p: -p[2]):
        order = list(range(len(_CANDIDATES)))
        paired = False
        # a near-coincident neighbour (< 24 pt): point the label AWAY from it so two labels never cross
        here = disp[m]
        nb = min((disp[o] - here for o in disp if o != m), key=lambda v: np.hypot(*v), default=None)
        if nb is not None and np.hypot(*nb) * 72 / fig.dpi < 24:
            u = nb / np.hypot(*nb)
            paired = True
            away = lambda i: np.dot(np.array(_CANDIDATES[i][:2]) / np.hypot(*_CANDIDATES[i][:2]), u)
            # a label must never sit on the far side of its neighbour's disc: keep only offsets that point
            # away from the neighbour (combined vector) AND do not face it along x (dx = 0 is allowed)
            ok = lambda i: away(i) < 0 and (abs(nb[0]) < 0.5 or _CANDIDATES[i][0] * nb[0] <= 0)
            # rank by the SIGN of the neighbour offset, so a diagonal facing away on both axes comes first
            # (lower-right for the lower-right member of a pair, upper-left for the other), then the sides
            sgn = np.sign(u) / max(np.hypot(*np.sign(u)), 1e-9)
            diag = lambda i: np.dot(np.array(_CANDIDATES[i][:2]) / np.hypot(*_CANDIDATES[i][:2]), sgn)
            # straight sides / below first (a label straight right of or below its disc reads as attached),
            # diagonals only after; within each group the most 'away' offset first
            axial = lambda i: _CANDIDATES[i][0] == 0 or _CANDIDATES[i][1] == 0
            order = sorted([i for i in order[:_NEAR] if ok(i)], key=lambda i: (not axial(i), round(away(i), 3), diag(i))) + [i for i in order[_NEAR:] if ok(i)]
        elif m in prefer:
            order.remove(prefer[m])
            order.insert(0, prefer[m])
        best = None
        near, wide = [i for i in order if i < _NEAR], [i for i in order if i >= _NEAR]
        if prefer.get(m) in wide and prefer[m] in order:
            near, wide = [prefer[m]] + near, [i for i in wide if i != prefer[m]]
        # close offsets first (strict, then relaxed about gridlines), only then the wider rings; for a
        # near-coincident pair the side of the marker matters more than a gridline, so skip the strict pass
        # close ring always first (strict about gridlines, then relaxed): a label stays within ~0.6 of a marker
        # diameter of its disc; the wider rings are only a last resort
        passes = [(True,), (False,)]
        for strict, i in [(p[0], i) for p in passes for i in near] + [(p[0], i) for p in passes for i in wide]:
            dx, dy, ha, va = _CANDIDATES[i]
            t = direct_label(ax, x, y, m, dx=dx, dy=dy, ha=ha, va=va, zorder=_LABEL_Z, **style(m))
            bb = t.get_window_extent(r).expanded(1.06, 1.15)
            hit = any(bb.overlaps(p) for p in placed)
            hit |= any(o != m and bb.x0 - rad < px < bb.x1 + rad and bb.y0 - rad < py < bb.y1 + rad
                       for o, (px, py) in disp.items())
            inside = bb.x0 >= axbb.x0 and bb.x1 <= axbb.x1 + 2 and bb.y1 <= axbb.y1 + 2 and bb.y0 >= axbb.y0
            if strict:
                hit |= any(bb.y0 - 1 < g < bb.y1 + 1 for g in grid)
            if _DEBUG:
                print(m, i, strict, "hit" if hit else "", "" if inside else "outside")
            if not hit and inside:
                best, chosen[m] = (t, bb, (dx, dy, ha, va)), i
                break
            t.remove()
        if best is None:                          # fall back to the first allowed offset (house default: right)
            dx, dy, ha, va = _CANDIDATES[order[0]] if order else _CANDIDATES[0]
            t = direct_label(ax, x, y, m, dx=dx, dy=dy, ha=ha, va=va, zorder=_LABEL_Z, **style(m))
            best = (t, t.get_window_extent(r), (dx, dy, ha, va))
        # halo twin just under the label: a white stroked copy (outlines in the SVG) while the visible label
        # stays a real <text> element; gridlines and band edge rules stop at the glyphs
        dx, dy, ha, va = best[2]
        direct_label(ax, x, y, m, dx=dx, dy=dy, ha=ha, va=va, zorder=_LABEL_Z - 0.05, path_effects=_HALO,
                     **(style(m) | {"color": "white"}))
        placed.append(best[1])
    return chosen


def _snap_ylim(ax, ys, span):
    """Lower y-limit that never lands just under a major tick (a gridline hugging the x-spine
    reads as a doubled baseline)."""
    lo, hi = ys.min() - 0.11 * span, ys.max() + 0.12 * span
    ax.set_ylim(lo, hi)
    ticks = ax.yaxis.get_majorticklocs()
    step = np.diff(ticks).min() if len(ticks) > 1 else span
    # a tick within 0.35 step above the floor would hug the x-spine: drop the floor so it clears by half a step
    near = [t for t in ticks if lo < t < lo + 0.35 * step]
    if near:
        lo = min(lo, near[0] - 0.5 * step)
    ax.set_ylim(lo, hi)


# x = controlled single-post CPU latency (benchmark.json, same machine for every model); training time is
# not comparable across models because DistilBERT and MentalBERT were trained on different hardware
XKEY = "cpu_single_ms"


def draw(an, ds=None, name="fig-cost-accuracy"):
    conds = ["D1", "D2"]
    # 6 models span ~3.5 log decades: stack the condition panels so each gets the full width
    wide = max(len(models_in(an["conditions"][c])) for c in conds) > 3
    fig, axes = plt.subplots(2, 1, figsize=(W, 4.1)) if wide else plt.subplots(1, 2, figsize=(W, 2.7))
    # one shared x-range so 0.7 / 1 / 1.5 / 2 s sit at the same positions in both panels
    allx = np.array([an["conditions"][c]["models"][m]["efficiency"][XKEY]
                     for c in conds for m in models_in(an["conditions"][c])])
    narrow = np.log10(allx.max() / allx.min()) <= 1.5
    # breathing room on the log axis so labels fit inside the plot area (tight when narrow, so the
    # 0.7-2 s tick range fills the axis; slightly wider on the left so a left label clears the tick column)
    xlim = (allx.min() / 1.25, allx.max() * 1.2) if narrow else (allx.min() / 1.7, allx.max() * 4.2)
    for ax, cond in zip(axes, conds):
        d = an["conditions"][cond]
        # semantic family bands (the chart design spec 2b: teal = classical, warm = transformer): a tint of the family hue
        # behind each family's training-time range, edged by thin hue rules, so the ~1000x cost gap between
        # the families reads at a glance. Both bands use the SAME alpha of their own hue so neither fades in print.
        # bands only when the families' ranges are disjoint (RF's CPU latency reaches the transformer range)
        rng = {f: [d["models"][m]["efficiency"][XKEY] for m in models_in(d) if FAMILY[m] == f]
               for f in ("classical", "transformer")}
        disjoint = all(rng.values()) and max(rng["classical"]) * 1.3 < min(rng["transformer"]) / 1.3
        if not narrow and disjoint:
            for fam in ("classical", "transformer"):
                fx = [d["models"][m]["efficiency"][XKEY] for m in models_in(d) if FAMILY[m] == fam]
                if fx:
                    x0, x1 = min(fx) / 1.3, max(fx) * 1.3
                    ax.axvspan(x0, x1, color=SEM[fam], alpha=_BAND_ALPHA, lw=0, zorder=0.2)
                    for xe in (x0, x1):
                        ax.axvline(xe, color=SEM[fam], alpha=_EDGE_ALPHA, lw=0.6, zorder=0.3)
        ax.set_xscale("log")
        ys = np.array([d["models"][m]["test"]["macro_f1"] for m in models_in(d)])
        # plain-number log ticks (0.7, 1, 1.5 ... or 1, 10, 100 ...) instead of mathtext 6x10^-1
        ax.set_xlim(*xlim)
        span = max(ys.max() - ys.min(), 0.005)
        ax.yaxis.set_major_locator(MultipleLocator(_YSTEP))
        ax.yaxis.set_major_formatter(_YFMT)
        _snap_ylim(ax, ys, span)
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 1.5, 2, 3, 5, 7) if narrow else (1.0,)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=() if narrow else np.arange(2, 10)))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("Single-post inference latency, Apple M1 CPU (ms, log scale)")
        ax.set_ylabel("Macro-F1 (test)")
        axis_lines(ax, grid="both")
        ax.tick_params(axis="x", which="minor", direction="out", length=1.8, width=0.5, color=T.LINE)
        # '(a) D1' prefix kept from v1, descriptor shared with fig-macro-f1 (COND_LABEL)
        panel_title(ax, f"({'ab'[cond == 'D2']}) {COND_LABEL[cond]}", size=FS_ANNOT + 0.5)
    fig.canvas.draw()                             # settle constrained layout before measuring marker gaps
    sizes = {}
    for ax, cond in zip(axes, conds):
        d = an["conditions"][cond]
        ms = _MS
        sizes[cond] = {}
        focal = max(models_in(d), key=lambda m: d["models"][m]["test"]["macro_f1"])
        # z-order: higher F1 underneath, lower F1 on top (a near-coincident lower point is otherwise eaten by
        # its upper neighbour); the focal model and its ring stay topmost
        order = sorted(models_in(d), key=lambda m: (m == focal, -d["models"][m]["test"]["macro_f1"]))
        for k, m in enumerate(order):
            e = d["models"][m]["efficiency"]
            v = d["models"][m]["test"]["macro_f1"]
            s, lw = ms, _EDGE_LW
            sizes[cond][m] = s
            ax.scatter(e[XKEY], v, marker=MARK[FAMILY[m]], s=s, color=COLORS[m], edgecolor="white",
                       lw=lw, zorder=3 + 0.01 * k)
            if m == focal:                        # the one accent element: a thin ring around the best model
                ax.scatter(e[XKEY], v, marker=MARK[FAMILY[m]], s=s * 1.9, facecolor="none",
                           edgecolor=T.ACCENT, lw=0.9, zorder=3.3)
    fig.canvas.draw()                             # settle constrained layout before measuring labels
    panels = []
    for ax, cond in zip(axes, conds):
        d = an["conditions"][cond]
        panels.append((ax, cond, [(m, d["models"][m]["efficiency"][XKEY], d["models"][m]["test"]["macro_f1"])
                                  for m in models_in(d)]))
    # label the more crowded panel first, then reuse each model's side in the other panel
    prefer = {}
    for ax, cond, pts in sorted(panels, key=lambda p: _crowding(p[0], p[2])):
        focal = max(pts, key=lambda p: p[2])[0]
        prefer = {**_place_labels(ax, pts, prefer, ms=sizes[cond], focal=focal), **prefer}
    save(fig, name)
    return fig


if __name__ == "__main__":
    import sys

    from common import load_json

    an = load_json("analysis.json")
    if "--mock6" in sys.argv:                     # layout test only: fake transformers, never the report file
        import copy

        an = copy.deepcopy(an)
        fake = {"DistilBERT": (900, 0.010), "MentalBERT": (2100, 0.016), "BERT": (1800, 0.014)}
        for c in ("D1", "D2"):
            base = an["conditions"][c]["models"]["SVM"]
            for m, (t, dv) in fake.items():
                rec = copy.deepcopy(base)
                rec["efficiency"][XKEY] = t
                rec["test"]["macro_f1"] = base["test"]["macro_f1"] + dv
                an["conditions"][c]["models"][m] = rec
        import os
        from pathlib import Path

        import fig_style
        fig_style.OUT = Path(os.environ.get("MOCK_OUT", "/tmp/fig-mock"))   # keep mocks out of figures/
        fig_style.PNG = fig_style.OUT
        draw(an, None, name="mock6-fig-cost-accuracy")
    else:
        draw(an, load_json("dataset_summary.json"))
