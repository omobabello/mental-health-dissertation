"""Fine-tune transformer models (DistilBERT, MentalBERT, BERT-base) on the shared splits.

Usage: python 03_transformers.py <COND> <MODEL_KEY> <SEED> [--n_train N] [--no-ig]
Writes results/transformers/<COND>_<MODEL>_<SEED>[_nN].json (one file per run, so long
experiment queues can be resumed). Protocol: AdamW, linear warm-up/decay, class-weighted
cross-entropy (parity with class_weight="balanced"), best epoch selected on validation
macro-F1, single evaluation on the held-out test split.
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from common import DATA, RESULTS, metrics, set_seed, stratified_subsample

MODEL_IDS = {
    "DistilBERT": "distilbert-base-uncased",
    "MentalBERT": "mental/mental-bert-base-uncased",
    "BERT": "bert-base-uncased",
}
MAX_LEN = 256
EPOCHS = 2
LR = 2e-5
BATCH = 8
ACCUM = 2   # effective batch 16
IG_SAMPLES = 200
IG_STEPS = 16
OUT = RESULTS / "transformers"
OUT.mkdir(exist_ok=True)
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def encode(tok, texts):
    """Tokenise without padding; returns a list of id lists (dynamic padding per batch)."""
    return tok(list(texts), truncation=True, max_length=MAX_LEN)["input_ids"], None


def collate(seqs, pad_id):
    L = min(MAX_LEN, -(-max(len(s) for s in seqs) // 32) * 32)  # round up to a multiple of 32 (few MPS shapes)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for j, s in enumerate(seqs):
        ids[j, :len(s)] = torch.tensor(s)
        mask[j, :len(s)] = 1
    return ids, mask


def bucket_batches(lengths, bs, rng, chunk=50):
    """Length-bucketed batching: shuffle, sort within chunks of `chunk` batches, shuffle batch order."""
    idx = rng.permutation(len(lengths))
    batches = []
    for c in range(0, len(idx), bs * chunk):
        part = sorted(idx[c:c + bs * chunk], key=lambda i: lengths[i])
        batches += [part[k:k + bs] for k in range(0, len(part), bs)]
    rng.shuffle(batches)
    return batches


@torch.no_grad()
def predict(model, seqs, pad_id, bs=32):
    model.eval()
    order = np.argsort([len(s) for s in seqs])
    p = np.zeros(len(seqs), dtype=np.float32)
    for k in range(0, len(order), bs):
        b = order[k:k + bs]
        ids, mask = collate([seqs[i] for i in b], pad_id)
        logits = model(input_ids=ids.to(DEV), attention_mask=mask.to(DEV)).logits
        p[b] = torch.softmax(logits, -1)[:, 1].float().cpu().numpy()
    return (p >= 0.5).astype(int), p


def integrated_gradients(model, tok, seqs):
    """Word-level IG attributions toward the positive class (zero-embedding baseline)."""
    model.eval()
    emb_layer = model.get_input_embeddings()
    token_scores = {}
    per_instance = []
    alphas = torch.linspace(1 / IG_STEPS, 1, IG_STEPS, device=DEV)
    for i in range(len(seqs)):
        n_real = len(seqs[i])
        ids_p, mask_p = collate([seqs[i]], tok.pad_token_id)
        x_ids, x_mask = ids_p.to(DEV), mask_p.to(DEV)
        emb = emb_layer(x_ids).detach()
        scaled = (alphas.view(-1, 1, 1) * emb).requires_grad_(True)
        logits = model(inputs_embeds=scaled, attention_mask=x_mask.expand(IG_STEPS, -1)).logits
        logits[:, 1].sum().backward()
        attr = (scaled.grad.mean(0) * emb[0]).sum(-1).detach().cpu().numpy()[:n_real]
        toks = tok.convert_ids_to_tokens(seqs[i])
        # merge word-pieces into words
        words, scores = [], []
        for t, a in zip(toks, attr):
            if t in tok.all_special_tokens:
                continue
            if t.startswith("##") and words:
                words[-1] += t[2:]
                scores[-1] += a
            else:
                words.append(t)
                scores.append(a)
        per_instance.append({"words": words[:60], "attr": [float(s) for s in scores[:60]]})
        for w, s in zip(words, scores):
            d = token_scores.setdefault(w, [0.0, 0])
            d[0] += float(s)
            d[1] += 1
    glob = {w: v[0] / v[1] for w, v in token_scores.items() if v[1] >= 5 and any(c.isalpha() for c in w)}
    ranked = sorted(glob.items(), key=lambda kv: -kv[1])
    return ranked, per_instance


def main():
    global BATCH, ACCUM
    ap = argparse.ArgumentParser()
    ap.add_argument("cond")
    ap.add_argument("model")
    ap.add_argument("seed", type=int)
    ap.add_argument("--n_train", type=int, default=0)
    ap.add_argument("--no-ig", action="store_true")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--micro_batch", type=int, default=BATCH, help="micro-batch; accumulation keeps effective batch 16")
    ap.add_argument("--grad_ckpt", action="store_true", help="gradient checkpointing (memory saving, same maths)")
    a = ap.parse_args()
    BATCH, ACCUM = a.micro_batch, (BATCH * ACCUM) // a.micro_batch
    tag = f"{a.cond}_{a.model}_{a.seed}" + (f"_n{a.n_train}" if a.n_train else "")
    out_path = OUT / f"{tag}.json"
    if out_path.exists():
        print("exists, skipping", out_path)
        return
    set_seed(a.seed)
    tr = pd.read_parquet(DATA / f"{a.cond}_train.parquet")
    if a.n_train:
        tr = stratified_subsample(tr, a.n_train, a.seed)
    va = pd.read_parquet(DATA / f"{a.cond}_val.parquet")
    te = pd.read_parquet(DATA / f"{a.cond}_test.parquet")

    mid = MODEL_IDS[a.model]
    tok = AutoTokenizer.from_pretrained(mid, token=os.environ.get("HF_TOKEN"))
    model = AutoModelForSequenceClassification.from_pretrained(mid, num_labels=2, token=os.environ.get("HF_TOKEN")).to(DEV)
    n_params = sum(p.numel() for p in model.parameters())
    if a.grad_ckpt:
        model.gradient_checkpointing_enable()

    pad = tok.pad_token_id
    tr_ids, _ = encode(tok, tr.text)
    va_ids, _ = encode(tok, va.text)
    te_ids, _ = encode(tok, te.text)
    trunc_rate = float(np.mean([len(x) > MAX_LEN for x in tok(list(te.text[:500]), truncation=False)["input_ids"]]))
    ytr = torch.tensor(tr.label.values)
    lengths = [len(s) for s in tr_ids]
    rng = np.random.default_rng(a.seed)
    n_batches = math.ceil(len(tr_ids) / BATCH)
    w = torch.tensor(len(ytr) / (2 * np.bincount(ytr.numpy())), dtype=torch.float32, device=DEV)
    loss_fn = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = math.ceil(n_batches / ACCUM) * a.epochs
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)

    history, best_f1, best_state = [], -1, None
    t_start = time.perf_counter()
    for ep in range(a.epochs):
        model.train()
        t_ep, tot = time.perf_counter(), 0.0
        dl = bucket_batches(lengths, BATCH, rng)
        for i, b in enumerate(dl):
            ids, mask = collate([tr_ids[j] for j in b], pad)
            y = ytr[b]
            logits = model(input_ids=ids.to(DEV), attention_mask=mask.to(DEV)).logits
            loss = loss_fn(logits, y.to(DEV)) / ACCUM
            loss.backward()
            tot += loss.item() * ACCUM
            if (i + 1) % ACCUM == 0 or i + 1 == len(dl):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sch.step(); opt.zero_grad()
            if i % 100 == 0:
                print(f"{tag} ep{ep} step{i}/{len(dl)} loss {loss.item()*ACCUM:.4f}", flush=True)
        yv, pv = predict(model, va_ids, pad)
        vm = metrics(va.label.values, yv, pv)
        history.append({"epoch": ep + 1, "train_loss": tot / len(dl), "val_macro_f1": vm["macro_f1"],
                        "val_roc_auc": vm["roc_auc"], "epoch_time_s": time.perf_counter() - t_ep})
        print(history[-1], flush=True)
        if vm["macro_f1"] > best_f1:
            best_f1 = vm["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    train_time = time.perf_counter() - t_start
    model.load_state_dict(best_state)

    t0 = time.perf_counter()
    yhat, ps = predict(model, te_ids, pad)
    infer_batch = (time.perf_counter() - t0) / len(te)
    t0 = time.perf_counter()
    with torch.no_grad():
        for t in te.text.iloc[:100]:
            e = tok(t, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(DEV)
            model(**e)
    infer_single = (time.perf_counter() - t0) / 100
    res = {
        "cond": a.cond, "model": a.model, "model_id": mid, "seed": a.seed, "n_train": len(tr),
        "hyperparams": {"max_len": MAX_LEN, "epochs": a.epochs, "lr": LR, "batch": BATCH * ACCUM, "warmup": 0.1,
                        "weight_decay": 0.01, "loss": "class-weighted CE",
                        "batching": "length-bucketed dynamic padding (chunk=50 batches)",
                        "micro_batch": BATCH, "grad_accum": ACCUM, "gradient_checkpointing": bool(a.grad_ckpt)},
        "history": history, "best_epoch": int(np.argmax([h["val_macro_f1"] for h in history]) + 1),
        "test": metrics(te.label.values, yhat, ps),
        "test_truncation_rate": trunc_rate,
        "efficiency": {"train_time_s": train_time, "infer_ms_per_sample_batch": infer_batch * 1000,
                       "infer_ms_per_sample_single": infer_single * 1000, "n_params": n_params,
                       "model_size_mb": n_params * 4 / 1e6, "hardware": f"Apple M1 ({DEV.type}, 8 GB)"},
        "y_pred": yhat.tolist(), "y_score": [float(p) for p in ps],
    }
    tpath = DATA / f"{a.cond}T_test.parquet"
    if tpath.exists() and not a.n_train:
        tt = pd.read_parquet(tpath)
        t_ids, _ = encode(tok, tt.text)
        yt, pt = predict(model, t_ids, pad)
        res["temporal"] = metrics(tt.label.values, yt, pt)
        res["temporal_y_pred"] = yt.tolist()
    if not a.no_ig and not a.n_train:
        ranked, per_inst = integrated_gradients(model, tok, te_ids[:IG_SAMPLES])
        res["interpretability"] = {"top_pos": ranked[:30], "top_neg": ranked[::-1][:30],
                                   "ig_samples": IG_SAMPLES, "ig_steps": IG_STEPS}
        res["ig_instances"] = per_inst
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)
    print("DONE", tag, res["test"]["macro_f1"], train_time, flush=True)


if __name__ == "__main__":
    main()
