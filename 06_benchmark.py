"""Controlled inference benchmark, run after all training on an otherwise idle machine.

Inference cost depends on architecture, not on fine-tuned weights, so transformer latency is
measured with each checkpoint plus a fresh classification head. Classical models are refit with
their selected hyper-parameters. Each measurement: warm-up, then 3 repeats, median reported.
Devices: transformers on MPS (Apple M1 GPU) and CPU; classical on CPU.
Output: results/benchmark.json
"""
import os
import pickle
import platform
import statistics
import time

import numpy as np
import pandas as pd
import torch
from sklearn.pipeline import Pipeline

from common import DATA, SEED, load_json, save_json, set_seed

import importlib
cls_mod = importlib.import_module("02_classical")

N_SINGLE = 200
REPEATS = 3
TRANSFORMERS = {"DistilBERT": "distilbert-base-uncased", "MentalBERT": "mental/mental-bert-base-uncased"}


def timed(fn):
    fn()  # warm-up
    vals = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn()
        vals.append(time.perf_counter() - t0)
    return statistics.median(vals)


def bench_classical(cond, res):
    tr = pd.read_parquet(DATA / f"{cond}_train.parquet")
    te = pd.read_parquet(DATA / f"{cond}_test.parquet")
    Xtr = tr.text.map(cls_mod.preprocess)
    out = {}
    for name, (clf, _) in cls_mod.MODELS.items():
        params = {k.replace("clf__", ""): v for k, v in res[cond][name]["best_params"].items()}
        pipe = Pipeline([("tfidf", cls_mod.vectorizer()), ("clf", clf.set_params(**params))])
        set_seed()
        train_s = timed(lambda: pipe.fit(Xtr, tr.label.values))
        texts = list(te.text)
        single = timed(lambda: [pipe.predict([cls_mod.preprocess(t)]) for t in texts[:N_SINGLE]]) / N_SINGLE
        batch = timed(lambda: pipe.predict(te.text.map(cls_mod.preprocess))) / len(te)
        out[name] = {"train_time_s": train_s, "cpu_single_ms": single * 1000, "cpu_batch_ms": batch * 1000,
                     "size_mb": len(pickle.dumps(pipe)) / 1e6, "n_params": cls_mod.n_params(pipe)}
        print(cond, name, out[name], flush=True)
    return out


@torch.no_grad()
def bench_transformer(mid, texts, device):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(mid)
    model = AutoModelForSequenceClassification.from_pretrained(mid, num_labels=2).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())

    def single():
        for t in texts[:N_SINGLE]:
            e = tok(t, truncation=True, max_length=256, return_tensors="pt").to(device)
            model(**e).logits.cpu()

    def batch():
        order = np.argsort([len(t) for t in texts])
        for k in range(0, len(texts), 32):
            e = tok([texts[i] for i in order[k:k + 32]], truncation=True, max_length=256, padding=True,
                    return_tensors="pt").to(device)
            model(**e).logits.cpu()

    s = timed(single) / N_SINGLE
    b = timed(batch) / len(texts)
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    return {"single_ms": s * 1000, "batch_ms": b * 1000, "n_params": n_params, "size_mb": n_params * 4 / 1e6}


def main():
    torch.set_num_threads(os.cpu_count())
    res = load_json("classical_results.json")
    out = {"meta": {"machine": f"{platform.machine()} {platform.platform()}", "torch": torch.__version__,
                    "repeats": REPEATS, "n_single": N_SINGLE, "note": "idle machine; median of repeats"},
           "classical": {}, "transformers": {}}
    for cond in ["D1", "D2"]:
        out["classical"][cond] = bench_classical(cond, res)
    texts = list(pd.read_parquet(DATA / "D1_test.parquet").text)
    for name, mid in TRANSFORMERS.items():
        out["transformers"][name] = {
            "mps": bench_transformer(mid, texts, torch.device("mps")),
            "cpu": bench_transformer(mid, texts[:500], torch.device("cpu")),
        }
        print(name, out["transformers"][name], flush=True)
    save_json(out, "benchmark.json")


if __name__ == "__main__":
    main()
