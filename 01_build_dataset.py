"""Build the two experimental conditions (D1, D2) from the Reddit Mental Health Dataset.

Ethical safeguards applied at ingest: the `author` column is never loaded,
URLs / usernames / subreddit mentions are masked, duplicates are removed.
Output: data/{D1,D2}_{train,val,test}.parquet with columns [text, label, source]
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from common import CONDITIONS, DATA, MIN_TOKENS, N_TOTAL, POS_RATE, POSITIVE, SEED, SPLIT, clean_text, save_json

USECOLS = ["subreddit", "date", "post"]  # author deliberately excluded


def load_sub(name: str, snap: str = "2019") -> pd.DataFrame:
    df = pd.read_csv(DATA / f"{name}_{snap}_features_tfidf_256.csv", usecols=USECOLS)
    df = df.rename(columns={"post": "text"})
    df["source"] = name
    return df


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    log = {"raw": len(df)}
    df = df.dropna(subset=["text"])
    df = df[~df["text"].str.strip().isin(["[removed]", "[deleted]", ""])]
    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].str.split().str.len() >= MIN_TOKENS]
    log["after_filter"] = len(df)
    df = df.drop_duplicates(subset=["text"])
    log["after_dedup"] = len(df)
    return df, log


def main():
    pos_raw = load_sub(POSITIVE)
    pos, pos_log = prepare(pos_raw)
    summary = {"positive_source": POSITIVE, "positive_log": pos_log, "conditions": {}}
    n_pos = int(N_TOTAL * POS_RATE)
    n_neg = N_TOTAL - n_pos

    for cond, subs in CONDITIONS.items():
        negs, neg_logs = [], {}
        for s in subs:
            d, lg = prepare(load_sub(s))
            negs.append(d)
            neg_logs[s] = lg
        neg = pd.concat(negs, ignore_index=True)
        # remove any text appearing in both classes (cross-posting)
        neg = neg[~neg["text"].isin(set(pos["text"]))]
        # balanced draw across control sources where possible
        per = n_neg // len(subs)
        parts = []
        for s in subs:
            g = neg[neg["source"] == s]
            parts.append(g.sample(n=min(per, len(g)), random_state=SEED))
        neg_s = pd.concat(parts)
        if len(neg_s) < n_neg:  # top up from remaining pool
            rest = neg.drop(neg_s.index)
            neg_s = pd.concat([neg_s, rest.sample(n=n_neg - len(neg_s), random_state=SEED)])
        pos_s = pos.sample(n=n_pos, random_state=SEED)
        df = pd.concat([pos_s.assign(label=1), neg_s.assign(label=0)], ignore_index=True)
        df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)[["text", "label", "source"]]

        tr, tmp = train_test_split(df, test_size=1 - SPLIT[0], stratify=df["label"], random_state=SEED)
        va, te = train_test_split(tmp, test_size=SPLIT[2] / (SPLIT[1] + SPLIT[2]),
                                  stratify=tmp["label"], random_state=SEED)
        for name, part in (("train", tr), ("val", va), ("test", te)):
            part.reset_index(drop=True).to_parquet(DATA / f"{cond}_{name}.parquet")

        lens = df["text"].str.split().str.len()
        summary["conditions"][cond] = {
            "negative_sources": subs,
            "negative_logs": neg_logs,
            "n_total": len(df),
            "n_pos": int(df.label.sum()),
            "n_neg": int((df.label == 0).sum()),
            "source_counts": df["source"].value_counts().to_dict(),
            "split_sizes": {"train": len(tr), "val": len(va), "test": len(te)},
            "split_pos": {"train": int(tr.label.sum()), "val": int(va.label.sum()), "test": int(te.label.sum())},
            "tokens_whitespace": {
                "mean": float(lens.mean()), "median": float(lens.median()),
                "p90": float(lens.quantile(0.9)), "max": int(lens.max()),
                "mean_pos": float(lens[df.label == 1].mean()), "mean_neg": float(lens[df.label == 0].mean()),
            },
            "length_hist_pos": lens[df.label == 1].clip(upper=1000).tolist(),
            "length_hist_neg": lens[df.label == 0].clip(upper=1000).tolist(),
        }
        print(cond, summary["conditions"][cond]["split_sizes"], summary["conditions"][cond]["source_counts"])
    # Temporal hold-out (D1T): same classes, 2020 "post" snapshot (Low et al., 2020), test-only
    seen = set(pd.concat([pd.read_parquet(DATA / f"D1_{p}.parquet") for p in ("train", "val", "test")]).text)
    n_t, n_tp = 1500, 450
    tp, _ = prepare(load_sub(POSITIVE, "post"))
    tp = tp[~tp.text.isin(seen)].sample(n=n_tp, random_state=SEED)
    tn = pd.concat([prepare(load_sub(s, "post"))[0] for s in CONDITIONS["D1"]])
    tn = tn[~tn.text.isin(seen) & ~tn.text.isin(set(tp.text))]
    per = (n_t - n_tp) // len(CONDITIONS["D1"])
    pool = tn
    tn = pd.concat([g.sample(n=min(per, len(g)), random_state=SEED) for _, g in pool.groupby("source")])
    if len(tn) < n_t - n_tp:  # top up from the remaining pool
        tn = pd.concat([tn, pool.drop(tn.index).sample(n=n_t - n_tp - len(tn), random_state=SEED)])
    dt = pd.concat([tp.assign(label=1), tn.assign(label=0)]).sample(frac=1, random_state=SEED)[["text", "label", "source", "date"]]
    dt.reset_index(drop=True).to_parquet(DATA / "D1T_test.parquet")
    summary["temporal_D1T"] = {"snapshot": "post (2020)", "n": len(dt), "n_pos": int(dt.label.sum()),
                               "source_counts": dt.source.value_counts().to_dict(),
                               "date_min": str(dt.date.min()), "date_max": str(dt.date.max())}
    print("D1T", summary["temporal_D1T"])
    save_json(summary, "dataset_summary.json")


if __name__ == "__main__":
    main()
