"""Supplementary analyses, part 1: re-run with per-contract scores saved.

The main runs (run.py) store only aggregate metrics. Cluster-bootstrap
intervals, contract-level (de-duplicated) metrics, recall at a fixed review
budget and the similarity dose-response curve need per-contract test scores. This script repeats the main runs with the same
splits, seeds, hyper-parameters and threshold rule (it imports them from
run.py) and saves, per run, results/scores/<ds>__<rg>__<model>__<seed>.npz:

  h, cluster, sol_minor   identifiers of the test contracts
  y, s                    test labels and scores (n x labels)
  thr                     validation-selected thresholds (labels)
  sim                     max TF-IDF cosine similarity of each test contract to
                          the training part (same vectoriser as TF-IDF + LR)

The main-run JSON files are not changed. Deterministic models reproduce the
main-run MCC exactly; TextCNN is retrained on CPU and can differ in the third
decimal. analyze_r1.py reports the agreement.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import config as C
from common import PROC, RESULTS
from run import EXTERNAL_MAP, LABELS, best_threshold, fit_predict, splits

OUT = RESULTS / "scores"
OUT.mkdir(exist_ok=True)


def max_sim(tr_df, te_df):
    vec = TfidfVectorizer(tokenizer=str.split, **C.TFIDF)
    Xtr = vec.fit_transform(tr_df.text)
    Xte = vec.transform(te_df.text)
    out = np.zeros(Xte.shape[0], dtype=np.float32)
    for i in range(0, Xte.shape[0], 500):
        out[i:i + 500] = (Xte[i:i + 500] @ Xtr.T).max(axis=1).toarray().ravel()
    return out


def cached_sim(ds, rg, seed, tr_df, te_df):
    """max_sim depends only on the split, so it is computed once per split."""
    p = PROC.parent / "sim_cache" / f"{ds}__{rg}__{seed}.npy"
    p.parent.mkdir(exist_ok=True)
    if p.exists():
        return np.load(p)
    x = max_sim(tr_df, te_df)
    np.save(p, x)
    return x


def one(ds, rg, m, seed):
    out = OUT / f"{ds}__{rg}__{m}__{seed}.npz"
    if out.exists():
        return
    df = pd.read_parquet(PROC / f"pool_{ds}.parquet")
    labels = LABELS[ds]
    tr, va, te = splits(df, ds, rg, seed)
    tr_df, va_df = df.iloc[tr].reset_index(drop=True), df.iloc[va].reset_index(drop=True)
    if rg == "external":
        sb = pd.read_parquet(PROC / "pool_smartbugs.parquet")
        te_df = sb.rename(columns={EXTERNAL_MAP[l]: l for l in labels if l in EXTERNAL_MAP})
        labels = [l for l in labels if l in EXTERNAL_MAP]
        cl = np.full(len(te_df), -1)
    else:
        te_df = df.iloc[te].reset_index(drop=True)
        cl = te_df.cluster.values
    res = fit_predict(m, ds, labels, tr_df, va_df, te_df, seed)
    thr = np.array([best_threshold(va_df[l].values, res[l][0]) if m != "rule" else 0.5
                    for l in labels])
    seen = te_df.h_l1.isin(set(tr_df.h_l1)).values if rg == "external" else \
        te_df.cluster.isin(set(tr_df.cluster)).values
    np.savez_compressed(
        out, labels=np.array(labels), h=np.array(te_df.h_l1.tolist(), dtype="U64"), cluster=np.asarray(cl, dtype=np.int64),
        sol_minor=pd.to_numeric(te_df.sol_minor, errors="coerce").fillna(-1).values.astype(np.int16)
        if "sol_minor" in te_df else np.full(len(te_df), -1, dtype=np.int16),
        y=te_df[labels].values.astype(np.int8),
        s=np.stack([res[l][1] for l in labels], 1).astype(np.float64),
        thr=thr, seen=np.asarray(seen, dtype=bool), sim=cached_sim(ds, rg, seed, tr_df, te_df))
    print(ds, rg, m, seed, "saved", flush=True)


if __name__ == "__main__":
    ds, rg, m = sys.argv[1:4]
    for s in [int(x) for x in sys.argv[4:]] or range(10):
        one(ds, rg, m, s)
