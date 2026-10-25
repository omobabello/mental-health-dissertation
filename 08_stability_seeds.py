"""Like-for-like explanation stability: classical models across the transformer seeds.

02_classical.py measured classical stability across the 5 CV folds (data perturbation), whereas the transformers were
measured across 3 seeds on the full training split (initialisation perturbation). This script removes the perturbation
mismatch: each classical model is refit on the full training split with its selected hyper-parameters under the
transformer seeds (42, 1337, 2026), and the top-20 Jaccard is computed over the same 3 pairs with the same stem
mapping used for the transformers in 04_analysis.py. The seed-42 refit must reproduce the stored test macro-F1 and
top-20. The attribution methods still differ by family (coefficients / impurity importance vs Integrated Gradients).
Output: results/stability_seeds.json
"""
import importlib
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from common import DATA, SEEDS, load_json, save_json

cls = importlib.import_module("02_classical")
ana = importlib.import_module("04_analysis")


def main():
    res = load_json("classical_results.json")
    an = load_json("analysis.json")
    out = {"meta": {"seeds": SEEDS, "pairs": 3, "k": 20, "mapping": "04_analysis.stems (as for the transformers)"}}
    for cond in ["D1", "D2"]:
        tr = pd.read_parquet(DATA / f"{cond}_train.parquet")
        te = pd.read_parquet(DATA / f"{cond}_test.parquet")
        xtr, xte = tr.text.map(cls.preprocess), te.text.map(cls.preprocess)
        out[cond] = {}
        for name, (clf, _) in cls.MODELS.items():
            params = {k.removeprefix("clf__"): v for k, v in res[cond][name]["best_params"].items()}
            tops = []
            for seed in SEEDS:
                p = Pipeline([("tfidf", cls.vectorizer()), ("clf", clf.__class__(**{**clf.get_params(), **params,
                                                                                    "random_state": seed}))])
                p.fit(xtr, tr.label.values)
                top = cls.top_k(p, 20)
                if seed == 42:          # the refit reproduces the stored run
                    f1 = f1_score(te.label.values, p.predict(xte), average="macro")
                    assert abs(f1 - res[cond][name]["test"]["macro_f1"]) < 1e-9, (cond, name, f1)
                    assert top == [w for w, _ in res[cond][name]["interpretability"]["top_pos"][:20]], (cond, name)
                tops.append(top)
            jac = [ana.jaccard(ana.stems(a), ana.stems(b)) for a, b in combinations(tops, 2)]
            out[cond][name] = {"seed_top20_jaccard_mean": float(np.mean(jac)), "seed_top20_jaccard_std": float(np.std(jac)),
                               "seed_top20_jaccard_pairs": [float(j) for j in jac],
                               "fold_top20_jaccard_mean": res[cond][name]["interpretability"]["fold_top20_jaccard_mean"],
                               "seed_top20": tops}
            print(cond, name, round(np.mean(jac), 3), flush=True)
        for m in ["DistilBERT", "MentalBERT"]:
            d = an["conditions"][cond]["models"][m]
            out[cond][m] = {"seed_top20_jaccard_mean": d["stability_jaccard"], "seed_top20_jaccard_std": d["stability_jaccard_std"]}
    save_json(out, "stability_seeds.json")


if __name__ == "__main__":
    main()
