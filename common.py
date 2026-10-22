"""Shared configuration and helpers for the dissertation experiments."""
import json
import os
import random
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
for _p in (DATA, RESULTS, FIGURES):
    _p.mkdir(exist_ok=True)

SEED = 42
SEEDS = [42, 1337, 2026]

# Condition definitions (Low et al., 2020, 2019 snapshot)
POSITIVE = "depression"
CONDITIONS = {
    "D1": ["fitness", "jokes", "meditation", "parenting", "teaching", "divorce", "personalfinance"],
    "D2": ["anxiety", "lonely", "mentalhealth", "bpd"],
}
N_TOTAL = 10_000          # posts per condition
POS_RATE = 0.30           # realistic minority positive class
SPLIT = (0.70, 0.15, 0.15)
MIN_TOKENS = 10

# Pre-specified meaningful-improvement threshold (dissertation Section 3.4)
MIN_DELTA_F1 = 0.02

URL_RE = re.compile(r"https?://\S+|www\.\S+")
USER_RE = re.compile(r"(?:^|\s)/?u/[A-Za-z0-9_-]+")
SUB_RE = re.compile(r"(?:^|\s)/?r/[A-Za-z0-9_]+")
WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Privacy- and leakage-oriented cleaning applied to every model family."""
    text = str(text)
    text = URL_RE.sub(" [URL] ", text)
    text = USER_RE.sub(" [USER] ", text)
    text = SUB_RE.sub(" [SUB] ", text)
    return WS_RE.sub(" ", text).strip()


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def save_json(obj, name: str) -> Path:
    path = RESULTS / name
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return path


def load_json(name: str):
    with open(RESULTS / name) as f:
        return json.load(f)


def metrics(y_true, y_pred, y_score) -> dict:
    from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                                 precision_score, recall_score, roc_auc_score, confusion_matrix)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_pos": precision_score(y_true, y_pred, zero_division=0),
        "recall_pos": recall_score(y_true, y_pred, zero_division=0),
        "f1_pos": f1_score(y_true, y_pred, zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def stratified_subsample(df, n, seed):
    """Stratified subsample of n rows (pandas-3-safe; keeps all columns)."""
    from sklearn.model_selection import train_test_split
    if n >= len(df):
        return df
    sub, _ = train_test_split(df, train_size=n, stratify=df["label"], random_state=seed)
    return sub
