"""Significance of the two LR ablations against the full LR model (same test set, paired).

02_classical.py stored ablation metrics but not per-post predictions. This script refits each ablation with the
identical pipeline and selected C, asserts that it reproduces the stored test macro-F1, and then compares its
predictions with the full LR model (results/{cond}_classical_preds.json) by McNemar's exact test and a paired
bootstrap of the macro-F1 difference (same procedure as 04_analysis.py).
Output: results/ablation_tests.json
"""
import importlib

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from common import DATA, SEED, load_json, save_json

cls = importlib.import_module("02_classical")
B = 2000


def mcnemar_exact(y, a, b):
    ra, rb = a == y, b == y
    n01, n10 = int(np.sum(ra & ~rb)), int(np.sum(~ra & rb))
    p = binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return {"full_right_ablation_wrong": n01, "full_wrong_ablation_right": n10, "p_exact": float(p)}


def paired_bootstrap(y, a, b, rng):
    n = len(y)
    d = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        d.append(f1_score(y[i], b[i], average="macro") - f1_score(y[i], a[i], average="macro"))
    return {"delta_mean": float(np.mean(d)), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


def main():
    res = load_json("classical_results.json")
    out = {}
    rng = np.random.default_rng(SEED)
    for cond in ["D1", "D2"]:
        tr = pd.read_parquet(DATA / f"{cond}_train.parquet")
        te = pd.read_parquet(DATA / f"{cond}_test.parquet")
        y = te.label.values
        full = np.array(load_json(f"{cond}_classical_preds.json")["LR"]["y_pred"])
        C = res[cond]["LR"]["best_params"]["clf__C"]
        variants = {
            "stopwords_retained": (tr.text.map(lambda t: cls.preprocess(t, stop=False)),
                                   te.text.map(lambda t: cls.preprocess(t, stop=False))),
            "target_lexicon_masked": (tr.text.map(lambda t: cls.preprocess(cls.TARGET_LEXICON.sub(" ", t))),
                                      te.text.map(lambda t: cls.preprocess(cls.TARGET_LEXICON.sub(" ", t)))),
        }
        out[cond] = {}
        for label, (xtr, xte) in variants.items():
            p = Pipeline([("tfidf", cls.vectorizer()),
                          ("clf", LogisticRegression(C=C, max_iter=2000, class_weight="balanced",
                                                     solver="liblinear", random_state=SEED))])
            p.fit(xtr, tr.label.values)
            pred = p.predict(xte)
            f1 = f1_score(y, pred, average="macro")
            stored = res[cond]["LR_ablations"][label]["test"]["macro_f1"]
            assert abs(f1 - stored) < 1e-9, (cond, label, f1, stored)      # refit reproduces the stored run
            out[cond][label] = {"macro_f1": float(f1), "delta_vs_full": float(f1 - f1_score(y, full, average="macro")),
                                "mcnemar": mcnemar_exact(y, full, pred),
                                "paired_bootstrap_ablation_minus_full": paired_bootstrap(y, full, pred, rng)}
            print(cond, label, out[cond][label], flush=True)
    save_json(out, "ablation_tests.json")


if __name__ == "__main__":
    main()
