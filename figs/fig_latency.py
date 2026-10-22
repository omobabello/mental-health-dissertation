"""fig-latency: single-post CPU inference latency per model (D1, Apple M1, benchmark.json), log x-axis.

Ported 1:1 from 05_figures.fig_latency (same field infer_ms_per_sample_single, D1, ORDER via
models_in, inverted y, log scale, "{v:.2f} ms" labels, same x-label); only the styling changed.
Horizontal bars in model colours (as v1, per user feedback), anchored at the left axis limit, which
sits on a 1-3-10 rung below the fastest model; both axis lines (axis_lines, value-axis grid) with
readable 0.3 / 1 / 3 / 10 / 30 tick labels and short unlabelled minor ticks. The slowest model's
value label is the single accent (focal) element.
Standalone:  cd experiments && .venv/bin/python -m figs.fig_latency
Mock (6 models, written outside figures/):  .venv/bin/python -m figs.fig_latency --mock6
"""
import sys

import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from fig_style import COLORS, FAMILY, FS_ANNOT, T, W, axis_lines, models_by_family, plt, save, value_label

# every bar is CPU latency from the controlled benchmark (one device for all models); the transformers'
# M1 GPU (MPS) latency is added to their value labels so the device never mixes within the bars
XKEY = "cpu_single_ms"


def draw(an, ds=None, name="fig-latency"):
    d = an["conditions"]["D1"]
    ms = models_by_family(d)
    n = len(ms)
    fig, ax = plt.subplots(figsize=(W, 0.75 + 0.36 * n))
    y = np.arange(n)
    vals = [d["models"][m]["efficiency"][XKEY] for m in ms]
    gpu = {m: d["models"][m]["efficiency"]["infer_ms_per_sample_single"] for m in ms if FAMILY[m] == "transformer"}
    lo, hi = min(vals), max(vals)
    ax.set_xscale("log")
    # 1-3-10 ladder trimmed to the data (left: nearest rung below the fastest; right: room for the label)
    ladder = [b * 10.0 ** k for k in range(-3, 6) for b in (1, 3)]
    xmin = max(t for t in ladder if t <= lo / 1.1)
    xmax = hi * 1.8                                   # provisional; refined from the label extent below
    ax.set_xlim(xmin, xmax)
    axis_lines(ax, grid="x")
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", which="minor", direction="out", length=1.8, width=0.5, color=T.LINE)
    ax.tick_params(axis="x", which="major", labelsize=FS_ANNOT, pad=3)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))   # 0.3, 1, 3, 10, 30 instead of 10^k
    # bars start at the left axis limit; white edge gives a crisp 2 px gap between neighbours
    ax.barh(y, [v - xmin for v in vals], left=xmin, height=0.56, color=[COLORS[m] for m in ms],
            linewidth=0, zorder=2)
    focal = int(np.argmax(vals))                      # slowest model = the one finding, in accent
    labels = []
    for i, (m, v) in enumerate(zip(ms, vals)):
        txt = f"{v:.2f} ms" + (f"  (GPU/MPS {gpu[m]:.2f} ms)" if m in gpu else "")
        if i == focal:
            labels.append(value_label(ax, v, i, txt, dx=5, size=FS_ANNOT, color=T.ACCENT, fontweight=600))
        else:
            labels.append(value_label(ax, v, i, txt, dx=5, size=FS_ANNOT))
    # end the axis just past the longest label (log axis: pad ~4% of the decade span); iterate since
    # the label's data-space extent depends on the limits themselves
    for _ in range(4):
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        end_px = max(t.get_window_extent(r).x1 for t in labels)
        end = ax.transData.inverted().transform((end_px, 0))[0]
        xmax = end * 10 ** (0.04 * np.log10(end / xmin))
        ax.set_xlim(xmin, xmax)
    ax.set_xticks([t for t in ladder if xmin <= t <= xmax])
    ax.set_xlim(xmin, xmax)
    ax.set_yticks(y, ms)
    ax.tick_params(axis="y", labelcolor=T.INK, labelsize=FS_ANNOT, pad=3)
    for t in ax.get_yticklabels():
        t.set_va("center")
    ax.set_ylim(n - 0.45, -0.55)                      # inverted: first model on top, bars clear of the x-axis
    ax.set_xlabel("Single-post inference latency, Apple M1 CPU (ms, log scale; D1)")
    save(fig, name)
    return fig


if __name__ == "__main__":
    from common import load_json

    an = load_json("analysis.json")
    if "--mock6" in sys.argv:
        import copy
        import os
        from pathlib import Path

        import fig_style

        an = copy.deepcopy(an)
        dd = an["conditions"]["D1"]["models"]
        for t, ms_ in {"DistilBERT": 6.4, "MentalBERT": 11.9, "BERT": 12.3}.items():
            rec = copy.deepcopy(dd["SVM"])
            rec["efficiency"][XKEY] = rec["efficiency"]["infer_ms_per_sample_single"] = ms_
            dd[t] = rec
        out = Path(os.environ.get("MOCK_OUT", "/tmp/fig-mock"))
        fig_style.OUT, fig_style.PNG = out, out / "_png"
        draw(an, None, name="mock6-fig-latency")
    else:
        draw(an, load_json("dataset_summary.json"))
