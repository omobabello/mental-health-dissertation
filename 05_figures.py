"""Generate every results chart (SVG, print-ready at 15.5 cm text width) from results/*.json.

Each chart lives in figs/<name>.py with draw(an, ds); the shared theme, palette and save() are in
fig_style.py. Output goes to figures/ unless FIG_OUT / FIG_PNG are set.
Colour = model identity in fixed categorical order; secondary encoding = marker shape by family
(circle classical, square transformer) and direct labels, so identity never depends on colour alone.
"""
import importlib
import os
from pathlib import Path

EXP = Path(__file__).resolve().parent
os.environ.setdefault("FIG_OUT", str(EXP / "figures"))
os.environ.setdefault("FIG_PNG", str(EXP / "figures"))

from common import load_json  # noqa: E402

CHARTS = ["fig_dataset_profile", "fig_macro_f1", "fig_roc_pr", "fig_confusion_d2", "fig_cost_accuracy",
          "fig_latency", "fig_learning_curve", "fig_top_features", "fig_interpretability", "fig_errors_d2",
          "fig_ablation", "fig_temporal"]


def main():
    an = load_json("analysis.json")
    ds = load_json("dataset_summary.json")
    for name in CHARTS:
        importlib.import_module(f"figs.{name}").draw(an, ds)


if __name__ == "__main__":
    main()
