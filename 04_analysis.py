"""Aggregate results: seed statistics, significance tests, cost ratios, interpretability metrics,
error analysis. Writes results/analysis.json (the single source for every number in the report).
"""
import json
from itertools import combinations

import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from scipy.stats import binomtest, chi2
from sklearn.metrics import f1_score

from common import DATA, MIN_DELTA_F1, RESULTS, SEED, load_json, save_json

STEM = PorterStemmer()
B = 2000
# Psycholinguistic depression-indicator lexicon (Rude et al., 2004; Al-Mosaiwi & Johnstone, 2018;
# Tausczik & Pennebaker, 2010): first-person singular, absolutist, negative-affect / hopelessness terms.
LEXICON = {
    "first_person": ["i", "me", "my", "myself", "mine", "im", "i'm", "i've", "ive"],
    "absolutist": ["always", "never", "nothing", "completely", "everything", "constantly", "totally", "entire",
                   "anymore", "every", "all", "whole", "absolutely", "definitely"],
    "negative_affect": ["sad", "depressed", "depression", "hopeless", "worthless", "empty", "lonely", "alone", "tired",
                        "hate", "pain", "cry", "crying", "numb", "suicidal", "die", "dead", "kill", "hurt", "miserable",
                        "anxious", "anxiety", "fail", "failure", "useless", "exhausted", "sleep", "therapy", "therapist",
                        "medication", "meds", "life", "feel", "feeling", "want", "help", "bad", "worse", "struggle"],
}
LEX_STEMS = {STEM.stem(w) for ws in LEXICON.values() for w in ws}
# Compute provenance. MentalBERT x6 and the DistilBERT learning curves ran on a private VPS, but
# 03_transformers.py hard-coded an "Apple M1" label, so those runs are relabelled from the VPS
# queue log (run_all.log, one "DONE <run>" line per finished job). Raw run JSONs are left untouched.
VPS_HW = "AMD EPYC 9354P VPS (cpu, 4 vCPU, 15 GB)"
VPS_LOG = RESULTS.parent / "run_all.log"


def load_runs(tdir):
    vps = set()
    if VPS_LOG.exists():
        vps = {l.split(" DONE ")[1].split()[0] for l in open(VPS_LOG) if " DONE " in l}
    runs = []
    for p in sorted(tdir.glob("*.json")):
        t = json.load(open(p))
        e = t["efficiency"]
        e["hardware_in_run"] = e["hardware"]
        if p.stem in vps:
            e["hardware"] = VPS_HW
            e["hardware_source"] = "run_all.log (VPS queue)"
        else:
            e["hardware_source"] = "in-run label"
        runs.append(t)
    return runs, sorted(vps)


def mcnemar(y, a, b):
    a_ok, b_ok = a == y, b == y
    n01 = int(np.sum(a_ok & ~b_ok))
    n10 = int(np.sum(~a_ok & b_ok))
    n = n01 + n10
    p_exact = binomtest(min(n01, n10), n, 0.5).pvalue if n else 1.0
    stat = (abs(n01 - n10) - 1) ** 2 / n if n else 0.0
    return {"a_right_b_wrong": n01, "a_wrong_b_right": n10, "chi2_cc": stat,
            "p_chi2": float(1 - chi2.cdf(stat, 1)), "p_exact": float(p_exact)}


def bootstrap_ci(y, pred, rng):
    idx = rng.integers(0, len(y), (B, len(y)))
    vals = np.array([f1_score(y[i], pred[i], average="macro") for i in idx])
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def paired_bootstrap(y, a, b, rng):
    idx = rng.integers(0, len(y), (B, len(y)))
    d = np.array([f1_score(y[i], b[i], average="macro") - f1_score(y[i], a[i], average="macro") for i in idx])
    return {"delta_mean": float(d.mean()), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_one_sided_b_le_a": float(np.mean(d <= 0))}


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 0.0


def stems(tokens):
    return [STEM.stem(t.split()[0]) if " " in t else STEM.stem(t) for t in tokens]


def plausibility(tokens, k=20):
    toks = tokens[:k]
    return float(np.mean([any(STEM.stem(p) in LEX_STEMS for p in t.split()) for t in toks]))


def truncation_rates(max_len=256):
    """Share of posts over max_len word pieces per condition (all splits), DistilBERT and MentalBERT tokenisers.

    Computed once and cached in results/truncation.json (the tokenisers are the only heavy dependency)."""
    path = RESULTS / "truncation.json"
    if path.exists():
        return load_json("truncation.json")
    from transformers import AutoTokenizer
    files = {"D1": ["D1_train", "D1_val", "D1_test"], "D2": ["D2_train", "D2_val", "D2_test"], "D1T": ["D1T_test"]}
    out = {"max_len": max_len}
    for model, mid in {"DistilBERT": "distilbert-base-uncased", "MentalBERT": "mental/mental-bert-base-uncased"}.items():
        tok = AutoTokenizer.from_pretrained(mid)
        for cond, parts in files.items():
            texts = pd.concat([pd.read_parquet(DATA / f"{f}.parquet") for f in parts]).text.tolist()
            ids = tok(texts, truncation=False)["input_ids"]
            out.setdefault(cond, {})[model] = float(np.mean([len(x) > max_len for x in ids]))
    save_json(out, "truncation.json")
    return out


def main():
    cls = load_json("classical_results.json")
    tdir = RESULTS / "transformers"
    trans, vps_runs = load_runs(tdir)
    out = {"meta": {"bootstrap_resamples": B, "min_delta_f1": MIN_DELTA_F1, "lexicon": LEXICON,
                    "compute": {"vps_hardware": VPS_HW, "vps_runs": vps_runs,
                                "note": "Training times are hardware-specific: classical on M1 CPU, DistilBERT on M1 MPS, "
                                        "MentalBERT and DistilBERT learning curves on VPS CPU. Training time is the median "
                                        "across seeds; per-seed values and spread are kept because shared-VPS throughput "
                                        "varied between runs. Cross-model cost is compared on benchmark.json latency "
                                        "(one machine)."},
                    "truncation": truncation_rates()},
           "conditions": {}}
    rng = np.random.default_rng(SEED)

    for cond in cls:
        te = pd.read_parquet(DATA / f"{cond}_test.parquet")
        y = te.label.values
        cp = load_json(f"{cond}_classical_preds.json")
        models, preds, rows, tpreds = {}, {}, [], {}
        for m in ["LR", "SVM", "RF"]:
            r = cls[cond][m]
            preds[m] = np.array(cp[m]["y_pred"])
            models[m] = {"family": "classical", "test": r["test"], "seeds": 1, "cv": r["cv"],
                         "efficiency": r["efficiency"],
                         "top_pos": [t for t, _ in r["interpretability"]["top_pos"]],
                         "stability_jaccard": r["interpretability"]["fold_top20_jaccard_mean"],
                         "stability_jaccard_std": r["interpretability"]["fold_top20_jaccard_std"],
                         "stability_basis": "5 CV folds (10 pairs)"}
            if "temporal" in r:
                models[m]["temporal"] = r["temporal"]
                tpreds[m] = np.array(cp[m]["temporal_y_pred"])
        full = [t for t in trans if t["cond"] == cond and t["n_train"] >= len(pd.read_parquet(DATA / f"{cond}_train.parquet"))]
        for m in sorted({t["model"] for t in full}):
            runs = [t for t in full if t["model"] == m]
            f1s = [t["test"]["macro_f1"] for t in runs]
            # representative run = median macro-F1 seed (used for paired tests)
            rep = runs[int(np.argsort(f1s)[len(f1s) // 2])]
            preds[m] = np.array(rep["y_pred"])
            agg = {k: {"mean": float(np.mean([t["test"][k] for t in runs])), "std": float(np.std([t["test"][k] for t in runs]))}
                   for k in ["macro_f1", "roc_auc", "pr_auc", "precision_pos", "recall_pos", "f1_pos", "accuracy"]}
            tops = [[w for w, _ in t["interpretability"]["top_pos"][:20]] for t in runs if "interpretability" in t]
            jac = [jaccard(stems(a), stems(b)) for a, b in combinations(tops, 2)]
            eff = {k: float(np.median([t["efficiency"][k] for t in runs])) for k in
                   ["train_time_s", "infer_ms_per_sample_batch", "infer_ms_per_sample_single", "n_params", "model_size_mb"]}
            tts = [t["efficiency"]["train_time_s"] for t in runs]
            hws = sorted({t["efficiency"]["hardware"] for t in runs})
            eff.update({"train_time_by_seed_s": {str(t["seed"]): t["efficiency"]["train_time_s"] for t in runs},
                        "train_time_min_s": float(min(tts)), "train_time_max_s": float(max(tts)),
                        "train_time_spread_ratio": float(max(tts) / min(tts)),
                        "hardware": hws[0] if len(hws) == 1 else hws,
                        "hardware_source": runs[0]["efficiency"]["hardware_source"]})
            models[m] = {"family": "transformer", "test": rep["test"], "test_seed_agg": agg, "seeds": len(runs),
                         "seed_list": [t["seed"] for t in runs], "rep_seed": rep["seed"],
                         "efficiency": eff, "history": rep["history"], "best_epoch": rep["best_epoch"],
                         "truncation_rate": rep["test_truncation_rate"],
                         "top_pos": [w for w, _ in rep["interpretability"]["top_pos"]] if "interpretability" in rep else [],
                         "stability_jaccard": float(np.mean(jac)) if jac else None,
                         "stability_jaccard_std": float(np.std(jac)) if jac else None,
                         "stability_basis": f"{len(tops)} seeds ({len(jac)} pairs)"}
            if "temporal" in rep:
                models[m]["temporal"] = rep["temporal"]
                models[m]["temporal_seed_agg"] = {"macro_f1": {"mean": float(np.mean([t["temporal"]["macro_f1"] for t in runs])),
                                                               "std": float(np.std([t["temporal"]["macro_f1"] for t in runs]))}}
                tpreds[m] = np.array(rep["temporal_y_pred"])

        bpath = RESULTS / "benchmark.json"
        if bpath.exists():  # controlled, idle-machine inference benchmark overrides in-run timings
            bm = json.load(open(bpath))
            for m, d in models.items():
                e = d["efficiency"]
                e["train_time_runs_s"] = e["train_time_s"]
                if d["family"] == "classical":
                    b = bm["classical"][cond][m]
                    e.update({"train_time_s": b["train_time_s"], "infer_ms_per_sample_single": b["cpu_single_ms"],
                              "infer_ms_per_sample_batch": b["cpu_batch_ms"], "cpu_single_ms": b["cpu_single_ms"],
                              "cpu_batch_ms": b["cpu_batch_ms"], "model_size_mb": b["size_mb"]})
                elif m in bm["transformers"]:
                    b = bm["transformers"][m]
                    e.update({"infer_ms_per_sample_single": b["mps"]["single_ms"], "infer_ms_per_sample_batch": b["mps"]["batch_ms"],
                              "cpu_single_ms": b["cpu"]["single_ms"], "cpu_batch_ms": b["cpu"]["batch_ms"],
                              "train_time_basis": "median of seeds (in-run)"})
                e["timing_source"] = "benchmark.json"
        for m, d in models.items():
            d["macro_f1_ci95"] = bootstrap_ci(y, preds[m], rng)
            d["plausibility_top20"] = plausibility(d["top_pos"], 20) if d["top_pos"] else None
            rows.append(m)

        best_cls = max(["LR", "SVM", "RF"], key=lambda m: models[m]["cv"]["f1_macro"]["mean"])  # chosen on CV, not test
        tests = {}
        for a, b in combinations(rows, 2):
            tests[f"{a}_vs_{b}"] = {"mcnemar": mcnemar(y, preds[a], preds[b]),
                                    "paired_bootstrap_b_minus_a": paired_bootstrap(y, preds[a], preds[b], rng)}
        # cost ratios & meaningful-improvement decision vs best classical
        decisions = {}
        for m in rows:
            if models[m]["family"] != "transformer":
                continue
            key = f"{best_cls}_vs_{m}"
            t = tests[key]
            delta = models[m]["test"]["macro_f1"] - models[best_cls]["test"]["macro_f1"]
            sig = t["mcnemar"]["p_exact"] < 0.05 and t["paired_bootstrap_b_minus_a"]["ci95"][0] > 0
            decisions[m] = {
                "vs": best_cls, "delta_macro_f1": float(delta), "significant": bool(sig),
                "meets_threshold": bool(sig and delta >= MIN_DELTA_F1),
                "train_time_ratio": models[m]["efficiency"]["train_time_s"] / models[best_cls]["efficiency"]["train_time_s"],
                "train_time_hardware": [models[best_cls]["efficiency"]["hardware"], models[m]["efficiency"]["hardware"]],
                "train_time_ratio_note": "cross-hardware; descriptive only",
                "latency_ratio_single": models[m]["efficiency"]["infer_ms_per_sample_single"] / models[best_cls]["efficiency"]["infer_ms_per_sample_single"],
                "latency_ratio_cpu": (models[m]["efficiency"].get("cpu_single_ms", float("nan")) /
                                      models[best_cls]["efficiency"].get("cpu_single_ms", float("nan"))),
                "param_ratio": models[m]["efficiency"]["n_params"] / models[best_cls]["efficiency"]["n_params"],
            }
        temporal_tests = {}
        if tpreds:
            yt = pd.read_parquet(DATA / f"{cond}T_test.parquet").label.values
            for m in tpreds:
                models[m]["temporal_ci95"] = bootstrap_ci(yt, tpreds[m], rng)
            for m in [m for m in tpreds if models[m]["family"] == "transformer"]:
                temporal_tests[f"{best_cls}_vs_{m}"] = {"mcnemar": mcnemar(yt, tpreds[best_cls], tpreds[m]),
                                                        "paired_bootstrap_b_minus_a": paired_bootstrap(yt, tpreds[best_cls], tpreds[m], rng)}
        # error analysis by negative source and length band
        lens = te.text.str.split().str.len()
        bands = pd.cut(lens, [0, 50, 150, 300, 100000], labels=["1-50", "51-150", "151-300", ">300"])
        err = {}
        for m in rows:
            wrong = preds[m] != y
            err[m] = {
                "fp_by_source": te[(preds[m] == 1) & (y == 0)].source.value_counts().to_dict(),
                "neg_count_by_source": te[y == 0].source.value_counts().to_dict(),
                "fn": int(np.sum((preds[m] == 0) & (y == 1))), "fp": int(np.sum((preds[m] == 1) & (y == 0))),
                "error_rate_by_length": {str(b): float(wrong[bands == b].mean()) for b in bands.cat.categories},
                "n_by_length": {str(b): int(np.sum(bands == b)) for b in bands.cat.categories},
            }
        both_wrong = {f"{best_cls}&{m}": int(np.sum((preds[best_cls] != y) & (preds[m] != y)))
                      for m in rows if models[m]["family"] == "transformer"}
        cross = {f"{a}~{b}": jaccard(stems(models[a]["top_pos"][:20]), stems(models[b]["top_pos"][:20]))
                 for a, b in combinations(rows, 2) if models[a]["top_pos"] and models[b]["top_pos"]}
        # learning curves (classical + transformer subsets)
        lc = cls[cond]["learning_curve"]
        for t in trans:
            if t["cond"] == cond:
                lc.setdefault(t["model"], []).append({"n_train": t["n_train"], "macro_f1": t["test"]["macro_f1"], "seed": t["seed"],
                                                      "train_time_s": t["efficiency"]["train_time_s"],
                                                      "hardware": t["efficiency"]["hardware"]})
        out["conditions"][cond] = {"models": models, "best_classical": best_cls, "tests": tests, "temporal_tests": temporal_tests, "decisions": decisions,
                                   "errors": err, "both_wrong": both_wrong, "cross_model_top20_jaccard": cross,
                                   "learning_curve": lc, "ablations": cls[cond]["LR_ablations"], "n_test": int(len(y))}
    save_json(out, "analysis.json")
    for c, d in out["conditions"].items():
        print(c, {m: round(v["test"]["macro_f1"], 4) for m, v in d["models"].items()}, d["decisions"])


if __name__ == "__main__":
    main()
