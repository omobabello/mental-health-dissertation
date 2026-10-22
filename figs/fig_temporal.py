"""fig-temporal: D1 test (2019) vs D1T temporal hold-out (2020) macro-F1, one slope per model.

Ported 1:1 from 05_figures.fig_temporal (same fields test.macro_f1 / temporal.macro_f1 on D1, models
with a "temporal" block in ORDER via models_in, same x tick labels, y-label and end labels
"{m} {b:.3f} ({b - a:+.3f})"); only the styling changed. End labels are spread vertically (in points)
so near-equal scores never overprint; a leader joins only labels that had to move. Labels sit on a paper
halo (design spec s6); deltas are plain ink so the slopes stay the primary read. The right x-limit is fitted to the
measured label extent + 4 pt.
Standalone:  cd experiments && .venv/bin/python -m figs.fig_temporal
"""
import matplotlib.transforms as mtrans

from fig_style import COLORS, axis_lines, FAMILY, FS_ANNOT, HALO, MARK, T, W, models_in, plt, save


def _spread(ys_pt, gap):
    """Minimal vertical separation (points) for labels, keeping order; centred on the originals."""
    order = sorted(range(len(ys_pt)), key=lambda i: ys_pt[i])
    pos = [ys_pt[i] for i in order]
    for _ in range(50):
        moved = False
        for k in range(1, len(pos)):
            if pos[k] - pos[k - 1] < gap:
                mid = (pos[k] + pos[k - 1]) / 2
                pos[k - 1], pos[k] = mid - gap / 2, mid + gap / 2
                moved = True
        if not moved:
            break
    out = [0.0] * len(ys_pt)
    for k, i in enumerate(order):
        out[i] = pos[k]
    return out


def draw(an, ds=None, name="fig-temporal"):
    d1 = an["conditions"]["D1"]
    ms = [m for m in models_in(d1) if "temporal" in d1["models"][m]]
    if not ms:
        return None
    fig, ax = plt.subplots(figsize=(W, 1.7 + 0.22 * len(ms)))
    pairs = []
    for m in ms:
        a, b = d1["models"][m]["test"]["macro_f1"], d1["models"][m]["temporal"]["macro_f1"]
        pairs.append((m, a, b))
        ax.plot([0, 1], [a, b], color=COLORS[m], lw=1.5, zorder=2, solid_capstyle="round")
    ax.set_xticks([0, 1], ["D1 test (2019)", "D1T temporal hold-out (2020)"])
    axis_lines(ax, grid="y")
    ax.tick_params(axis="x", pad=3, labelcolor=T.INK, labelsize=FS_ANNOT + 0.5)
    ax.set_xlim(-0.08, 1.45)
    ax.margins(y=0.08)
    ax.set_ylabel("Macro-F1")
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.3f"))

    # markers: where two scores at the same end would overprint (< one marker apart), dodge them
    # horizontally by a few points, so every marker stays visible (lines still meet the true x)
    fig.canvas.draw()
    to_pt = 72 / fig.dpi
    ms_pt = 6.5
    dodge_r = 0.0   # largest rightward marker shift at x=1 (labels start beyond it)
    for xi in (0, 1):
        ys = [(ax.transData.transform((xi, p[1 + xi]))[1] * to_pt, i) for i, p in enumerate(pairs)]
        ys.sort()
        groups, cur = [], [ys[0]]
        for y, i in ys[1:]:
            (cur.append((y, i)) if y - cur[-1][0] < ms_pt else (groups.append(cur), cur := [(y, i)]))
        groups.append(cur)
        for g in groups:
            for k, (_, i) in enumerate(g):
                off = (k - (len(g) - 1) / 2) * ms_pt * 0.75
                m, a, b = pairs[i]
                if xi == 1:
                    dodge_r = max(dodge_r, off)
                tr = ax.transData + mtrans.ScaledTranslation(off / 72, 0, fig.dpi_scale_trans)
                ax.plot([xi], [(a, b)[xi]], MARK[FAMILY[m]], color=COLORS[m], ms=ms_pt, mec="white",
                        mew=1.0, zorder=3 + k * 0.01, transform=tr)

    # direct end labels, de-collided in display space: "{m} {b:.3f}" + "({delta:+.3f})", both in ink
    fig.canvas.draw()
    to_pt = 72 / fig.dpi
    y_pt = [ax.transData.transform((1, b))[1] * to_pt for _, _, b in pairs]
    tgt = _spread(y_pt, gap=FS_ANNOT * 1.35)
    dx = 8 + dodge_r
    deltas = []
    for (m, a, b), y0, y1 in zip(pairs, y_pt, tgt):
        dy = y1 - y0
        kw = {}
        if abs(dy) > 4:   # leader only for labels visibly displaced (~half a line)
            kw["arrowprops"] = dict(arrowstyle="-", color=T.LINE, lw=0.75, shrinkA=1, shrinkB=2)
        t1 = ax.annotate(f"{m} {b:.3f}", (1, b), xytext=(dx, dy), textcoords="offset points",
                         ha="left", va="center", fontsize=FS_ANNOT, color=T.INK, zorder=4, bbox=HALO, **kw)
        deltas.append(ax.annotate(f"({b - a:+.3f})", (1, 0.5), xycoords=t1, xytext=(3, 0),
                                  textcoords="offset points", ha="left", va="center", fontsize=FS_ANNOT,
                                  color=T.INK, zorder=4, bbox=HALO))
    # fit the right x-limit to the widest label (+4 pt) so no empty gridded strip is left on the right
    lo = ax.get_xlim()[0]
    for _ in range(4):
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        bb = ax.get_window_extent(r)
        right = max(t.get_window_extent(r).x1 for t in deltas) + 4 / to_pt
        x1_px = ax.transData.transform((1, 0))[0]
        need = right - x1_px                                 # px needed right of x=1
        frac = 1 - need / bb.width                          # where x=1 must sit in axes fraction
        ax.set_xlim(lo, lo + (1 - lo) / max(frac, 0.3))
    save(fig, name)
    return fig


if __name__ == "__main__":
    import sys

    from common import load_json

    an = load_json("analysis.json")
    if "--mock6" in sys.argv:   # the chart design spec s6: verify layout with 6 models (not a real figure)
        import copy

        an = copy.deepcopy(an)
        ms = an["conditions"]["D1"]["models"]
        for k, (m, src, dt, dd) in enumerate([("DistilBERT", "LR", 0.012, -0.004), ("MentalBERT", "SVM", 0.018, 0.001),
                                               ("BERT", "LR", 0.015, -0.006)]):
            ms[m] = copy.deepcopy(ms[src])
            ms[m]["test"]["macro_f1"] += dt
            ms[m]["temporal"]["macro_f1"] = ms[m]["test"]["macro_f1"] + dd
        import os
        from pathlib import Path

        import fig_style
        fig_style.OUT = Path(os.environ.get("MOCK_OUT", "/tmp/fig-mock"))   # keep mocks out of figures/
        fig_style.PNG = fig_style.OUT
        draw(an, name="_mock6-fig-temporal")
    else:
        draw(an, load_json("dataset_summary.json"))
