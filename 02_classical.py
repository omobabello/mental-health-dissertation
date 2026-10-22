"""Classical ML baselines: TF-IDF + Logistic Regression / Linear SVM / Random Forest.

Protocol (identical across families): hyper-parameters chosen by stratified 5-fold CV on the
training split (macro-F1), refit on train, evaluated once on the shared held-out test split.
Also produces: CV fold scores, efficiency measurements, interpretability (top features,
cross-fold top-k stability), test predictions (for McNemar / bootstrap), learning curves,
and two LR ablations (stop-words retained; target-lexicon masked).
"""
import pickle
import re
import sys
import time

import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from common import DATA, RESULTS, SEED, metrics, save_json, set_seed, stratified_subsample

TOKEN_RE = re.compile(r"[a-z']+")
STEM = PorterStemmer()
TARGET_LEXICON = re.compile(r"\b(depress\w*|anxi\w*|lonel\w*|bpd|borderline|suicid\w*|mental\w*)\b", re.I)


def preprocess(text: str, stop=True) -> str:
    toks = TOKEN_RE.findall(text.lower())
    if stop:
        toks = [t for t in toks if t not in ENGLISH_STOP_WORDS]
    return " ".join(STEM.stem(t) for t in toks)


def vectorizer():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, max_features=50_000, sublinear_tf=True)


MODELS = {
    "LR": (LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=SEED),
           {"clf__C": [0.1, 1, 10]}),
    "SVM": (LinearSVC(class_weight="balanced", random_state=SEED),
            {"clf__C": [0.01, 0.1, 1]}),
    "RF": (RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
           {"clf__n_estimators": [300], "clf__max_depth": [None, 50], "clf__min_samples_leaf": [1, 2]}),
}


def score_of(pipe, X):
    clf = pipe.named_steps["clf"]
    return pipe.predict_proba(X)[:, 1] if hasattr(clf, "predict_proba") else pipe.decision_function(X)


def n_params(pipe):
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "coef_"):
        return int(clf.coef_.size + clf.intercept_.size)
    return int(sum(est.tree_.node_count for est in clf.estimators_))  # RF: total tree nodes


def feature_weights(pipe):
    vocab = np.array(pipe.named_steps["tfidf"].get_feature_names_out())
    clf = pipe.named_steps["clf"]
    w = clf.coef_.ravel() if hasattr(clf, "coef_") else clf.feature_importances_
    return vocab, w


def top_k(pipe, k=20):
    vocab, w = feature_weights(pipe)
    return list(vocab[np.argsort(-w)[:k]])


def run_condition(cond: str, out: dict):
    tr = pd.read_parquet(DATA / f"{cond}_train.parquet")
    te = pd.read_parquet(DATA / f"{cond}_test.parquet")
    Xtr = tr.text.map(preprocess)
    Xte = te.text.map(preprocess)
    ytr, yte = tr.label.values, te.label.values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    res = {}
    preds = {}
    for name, (clf, grid) in MODELS.items():
        set_seed()
        pipe = Pipeline([("tfidf", vectorizer()), ("clf", clf)])
        gs = GridSearchCV(pipe, grid, scoring="f1_macro", cv=skf, n_jobs=1 if name == "RF" else -1)
        gs.fit(Xtr, ytr)
        best = gs.best_estimator_
        # CV fold scores for the chosen configuration
        cv = cross_validate(best, Xtr, ytr, cv=skf, scoring=["f1_macro", "roc_auc", "precision", "recall"],
                            return_estimator=True, n_jobs=1)
        fold_topk = [top_k(e, 20) for e in cv["estimator"]]
        jacc = [len(set(a) & set(b)) / len(set(a) | set(b))
                for i, a in enumerate(fold_topk) for b in fold_topk[i + 1:]]
        # timed refit on full train split
        t0 = time.perf_counter()
        best.fit(Xtr, ytr)
        train_time = time.perf_counter() - t0
        # timed inference (raw text incl. preprocessing, batch)
        t0 = time.perf_counter()
        Xte_p = te.text.map(preprocess)
        yhat = best.predict(Xte_p)
        ys = score_of(best, Xte_p)
        infer_batch = (time.perf_counter() - t0) / len(te)
        t0 = time.perf_counter()
        for t in te.text.iloc[:200]:
            best.predict([preprocess(t)])
        infer_single = (time.perf_counter() - t0) / 200
        vocab, w = feature_weights(best)
        order = np.argsort(-w)
        size_mb = len(pickle.dumps(best)) / 1e6
        res[name] = {
            "best_params": gs.best_params_,
            "grid_cv_macro_f1": float(gs.best_score_),
            "cv": {m: {"mean": float(np.mean(cv[f"test_{m}"])), "std": float(np.std(cv[f"test_{m}"])),
                       "folds": [float(x) for x in cv[f"test_{m}"]]} for m in ["f1_macro", "roc_auc", "precision", "recall"]},
            "test": metrics(yte, yhat, ys),
            "efficiency": {"train_time_s": train_time, "infer_ms_per_sample_batch": infer_batch * 1000,
                           "infer_ms_per_sample_single": infer_single * 1000, "n_params": n_params(best),
                           "model_size_mb": size_mb, "hardware": "Apple M1 CPU (8 GB)"},
            "interpretability": {
                "top_pos": [(vocab[i], float(w[i])) for i in order[:30]],
                "top_neg": [(vocab[i], float(w[i])) for i in order[::-1][:30]] if name != "RF" else [],
                "fold_top20_jaccard_mean": float(np.mean(jacc)), "fold_top20_jaccard_std": float(np.std(jacc)),
                "fold_top20": fold_topk,
            },
        }
        preds[name] = {"y_pred": yhat.tolist(), "y_score": [float(s) for s in ys]}
        tpath = DATA / f"{cond}T_test.parquet"
        if tpath.exists():
            tt = pd.read_parquet(tpath)
            Xt = tt.text.map(preprocess)
            yt_hat, yt_s = best.predict(Xt), score_of(best, Xt)
            res[name]["temporal"] = metrics(tt.label.values, yt_hat, yt_s)
            preds[name]["temporal_y_pred"] = yt_hat.tolist()
        print(cond, name, res[name]["test"]["macro_f1"], res[name]["efficiency"]["train_time_s"], flush=True)
        if name == "LR":
            with open(RESULTS / f"{cond}_LR.pkl", "wb") as f:
                pickle.dump(best, f)

    # Ablations on LR (best C reused)
    C = res["LR"]["best_params"]["clf__C"]
    abl = {}
    for label, prep_tr, prep_te in [
        ("stopwords_retained", tr.text.map(lambda t: preprocess(t, stop=False)), te.text.map(lambda t: preprocess(t, stop=False))),
        ("target_lexicon_masked", tr.text.map(lambda t: preprocess(TARGET_LEXICON.sub(" ", t))),
         te.text.map(lambda t: preprocess(TARGET_LEXICON.sub(" ", t)))),
    ]:
        p = Pipeline([("tfidf", vectorizer()), ("clf", LogisticRegression(C=C, max_iter=2000, class_weight="balanced",
                                                                          solver="liblinear", random_state=SEED))])
        p.fit(prep_tr, ytr)
        abl[label] = {"test": metrics(yte, p.predict(prep_te), p.predict_proba(prep_te)[:, 1]),
                      "top_pos": top_k(p, 20)}
    res["LR_ablations"] = abl

    # Learning curve (LR, SVM) on stratified subsets of train
    lc = {}
    for n in [250, 500, 1000, 2000, 4000, len(tr)]:
        sub = stratified_subsample(tr, n, SEED)
        Xs = sub.text.map(preprocess)
        for name in ["LR", "SVM"]:
            clf = MODELS[name][0].set_params(C=res[name]["best_params"]["clf__C"])
            p = Pipeline([("tfidf", vectorizer()), ("clf", clf)]).fit(Xs, sub.label.values)
            lc.setdefault(name, []).append({"n_train": len(sub), "macro_f1": metrics(yte, p.predict(Xte), score_of(p, Xte))["macro_f1"]})
    res["learning_curve"] = lc
    out[cond] = res
    save_json(preds, f"{cond}_classical_preds.json")


def main():
    conds = sys.argv[1:] or ["D1", "D2"]
    out = {}
    for c in conds:
        run_condition(c, out)
        save_json(out, "classical_results.json")


if __name__ == "__main__":
    main()
