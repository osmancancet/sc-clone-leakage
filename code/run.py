"""Train and evaluate one (dataset, regime, model, seed) cell; write a JSON.

usage: python run.py <dataset> <regime> <model> [seeds...]
dataset: slither | messi_re | messi_ts
regime : random | cluster | official | version | dedup | external
model  : rule | knn | lr | cnn | cb
"""
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score,
                             matthews_corrcoef, roc_auc_score)

from common import PROC, RESULTS, SEEDS
import config as C

OUT = RESULTS / "runs"
OUT.mkdir(exist_ok=True)

LABELS = {"slither": C.SLITHER_LABELS, "messi_re": ["y"], "messi_ts": ["y"]}
# SmartBugs-curated categories used as manually labelled external test sets
EXTERNAL_MAP = {"y_reentrancy": "y_reentrancy", "y": "y_reentrancy",
                "y_unchecked_calls": "y_unchecked_low_level_calls",
                "y_access_control": "y_access_control",
                "y_arithmetic": "y_arithmetic"}

RULES = {
    "y_reentrancy": r"\bcall \. value\b|\bcall \{ value\b",
    "y_unchecked_calls": r"\. (?:call|send|delegatecall) [\(\{.]",
    "y_arithmetic": r" / ",
    "y_access_control": r"\btx \. origin\b|\bselfdestruct\b|\bsuicide\b|\bdelegatecall\b",
    "y_other": r"\bbalance ==|== address \( this \) \. balance",
}


def rule_for(dataset, label):
    if dataset == "messi_re":
        return RULES["y_reentrancy"]
    if dataset == "messi_ts":
        return r"\bblock \. timestamp\b|\bnow\b"
    return RULES[label]


# ----------------------------------------------------------------- splits
def take(rng, idx, n):
    idx = np.asarray(idx)
    return np.sort(rng.choice(idx, min(n, len(idx)), replace=False))


def group_partition(rng, groups, fracs):
    """Assign whole groups to parts until each part reaches its row share."""
    uniq = rng.permutation(np.unique(groups))
    sizes = pd.Series(groups).value_counts()
    n = len(groups)
    targets = [f * n for f in fracs]
    parts, acc, k = [[] for _ in fracs], [0.0] * len(fracs), 0
    for g in uniq:
        while k < len(fracs) - 1 and acc[k] >= targets[k]:
            k += 1
        parts[k].append(g)
        acc[k] += sizes[g]
    return [np.flatnonzero(np.isin(groups, p)) for p in parts]


def splits(df, dataset, regime, seed):
    rng = np.random.default_rng(seed)
    n = len(df)
    if dataset == "slither":
        if regime == "random":
            perm = rng.permutation(n)
            te, va, tr = np.split(perm, [C.N_TEST, C.N_TEST + C.N_VAL])
            return take(rng, tr, C.N_TRAIN), np.sort(va), np.sort(te)
        if regime == "cluster":
            fr = [C.N_TEST / 11500, C.N_VAL / 11500]
            te, va, tr = group_partition(rng, df.cluster.values, fr + [1 - sum(fr)])
        elif regime == "official":
            s = df.split_official.values
            tr, va, te = (np.flatnonzero(s == x) for x in ("train", "val", "test"))
        elif regime == "version":
            v = df.sol_minor.values
            tr, va, te = (np.flatnonzero(np.isin(v, x)) for x in
                          (C.VERSION_TRAIN, C.VERSION_VAL, C.VERSION_TEST))
        elif regime == "external":
            perm = rng.permutation(n)
            return take(rng, perm[C.N_VAL:], C.N_TRAIN), np.sort(perm[:C.N_VAL]), None
        return take(rng, tr, C.N_TRAIN), take(rng, va, C.N_VAL), take(rng, te, C.N_TEST)
    # Messi-Q: 80/20 with 10% of the training part for threshold selection
    tf, vf = C.MESSI_TEST_FRAC, C.MESSI_VAL_FRAC
    if regime == "random":
        perm = rng.permutation(n)
        nte, nva = int(round(tf * n)), int(round(vf * (1 - tf) * n))
        te, va, tr = np.split(perm, [nte, nte + nva])
    elif regime == "dedup":
        first = np.flatnonzero(~df.h_l1.duplicated().values)
        perm = rng.permutation(first)
        m = len(perm)
        nte, nva = int(round(tf * m)), int(round(vf * (1 - tf) * m))
        te, va, tr = np.split(perm, [nte, nte + nva])
    elif regime == "cluster":
        te, va, tr = group_partition(rng, df.cluster.values,
                                     [tf, vf * (1 - tf), 1 - tf - vf * (1 - tf)])
    elif regime == "external":
        # train on all Messi-Q rows; near-duplicate groups kept whole between train/val
        va, tr = group_partition(rng, df.cluster.values, [0.15, 0.85])
        return np.sort(tr), np.sort(va), None
    return np.sort(tr), np.sort(va), np.sort(te)


# ---------------------------------------------------------------- metrics
def best_threshold(y, s):
    qs = np.unique(np.quantile(s, np.linspace(0, 1, 201)))
    best, thr = -2, 0.5
    for t in qs:
        p = (s >= t).astype(int)
        m = matthews_corrcoef(y, p) if p.min() != p.max() else 0.0
        if m > best:
            best, thr = m, t
    return float(thr)


def metrics(y, s, thr):
    p = (s >= thr).astype(int)
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    tpr = tp / max(tp + fn, 1); fpr = fp / max(fp + tn, 1)
    out = {"mcc": float(matthews_corrcoef(y, p)), "f1": float(f1_score(y, p, zero_division=0)),
           "tpr": tpr, "fpr": fpr, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
           "pos": int(y.sum()), "n": int(len(y))}
    both = 0 < y.sum() < len(y)
    out["pr_auc"] = float(average_precision_score(y, s)) if both else None
    out["roc_auc"] = float(roc_auc_score(y, s)) if both else None
    for pi in C.PREVALENCES:
        den = tpr * pi + fpr * (1 - pi)
        out[f"prec_at_{pi}"] = float(tpr * pi / den) if den > 0 else None
    return out


# ----------------------------------------------------------------- models
def fit_predict(model, dataset, labels, tr_df, va_df, te_df, seed):
    """Return dict label -> (val_scores, test_scores)."""
    Ytr = tr_df[labels].values
    if model == "rule":
        res = {}
        for j, l in enumerate(labels):
            pat = re.compile(rule_for(dataset, l))
            f = lambda d: d.text.map(lambda t: float(bool(pat.search(t)))).values
            res[l] = (f(va_df), f(te_df))
        return res
    if model in ("lr", "knn"):
        vec = TfidfVectorizer(tokenizer=str.split, **C.TFIDF)
        Xtr = vec.fit_transform(tr_df.text)
        Xva, Xte = vec.transform(va_df.text), vec.transform(te_df.text)
        if model == "lr":
            res = {}
            for j, l in enumerate(labels):
                clf = LogisticRegression(random_state=seed, **C.LR).fit(Xtr, Ytr[:, j])
                res[l] = (clf.decision_function(Xva), clf.decision_function(Xte))
            return res
        def knn(X):
            S = (X @ Xtr.T).toarray() if sparse.issparse(X) else X @ Xtr.T
            idx = np.argpartition(-S, C.KNN_K - 1, axis=1)[:, :C.KNN_K]
            w = np.take_along_axis(S, idx, axis=1) + 1e-9
            return {l: (w * Ytr[idx, j]).sum(1) / w.sum(1) for j, l in enumerate(labels)}
        kv, kt = knn(Xva), knn(Xte)
        return {l: (kv[l], kt[l]) for l in labels}
    if model == "cb":
        z = {}
        for name in ("slither", "messi_re", "messi_ts", "smartbugs"):
            p = PROC / f"emb_{name}.npz"
            if p.exists():
                d = np.load(p)
                z.update(dict(zip(d["h"], d["x"])))
        E = lambda d: np.stack([z[h] for h in d.h_l1])
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(E(tr_df))
        Xtr, Xva, Xte = sc.transform(E(tr_df)), sc.transform(E(va_df)), sc.transform(E(te_df))
        res = {}
        for j, l in enumerate(labels):
            clf = LogisticRegression(random_state=seed, **C.EMB_LR).fit(Xtr, Ytr[:, j])
            res[l] = (clf.decision_function(Xva), clf.decision_function(Xte))
        return res
    if model == "cnn":
        from cnn import train_cnn
        sv, st = train_cnn(tr_df.text.tolist(), Ytr, va_df.text.tolist(),
                           va_df[labels].values, te_df.text.tolist(), seed, C.CNN)
        return {l: (sv[:, j], st[:, j]) for j, l in enumerate(labels)}
    raise ValueError(model)


def run(dataset, regime, model, seed):
    out = OUT / f"{dataset}__{regime}__{model}__{seed}.json"
    if out.exists():
        return
    df = pd.read_parquet(PROC / f"pool_{dataset}.parquet")
    labels = LABELS[dataset]
    tr, va, te = splits(df, dataset, regime, seed)
    tr_df, va_df = df.iloc[tr].reset_index(drop=True), df.iloc[va].reset_index(drop=True)
    if regime == "external":
        sb = pd.read_parquet(PROC / "pool_smartbugs.parquet")
        te_df = sb.rename(columns={EXTERNAL_MAP[l]: l for l in labels if l in EXTERNAL_MAP})
        labels = [l for l in labels if l in EXTERNAL_MAP]
        seen = set(tr_df.h_l1)
        te_df = te_df.assign(_seen=te_df.h_l1.isin(seen))
    else:
        te_df = df.iloc[te].reset_index(drop=True)
    t0 = time.time()
    res = fit_predict(model, dataset, labels, tr_df, va_df, te_df, seed)
    elapsed = time.time() - t0
    rec = {"dataset": dataset, "regime": regime, "model": model, "seed": seed,
           "n_train": len(tr_df), "n_val": len(va_df), "n_test": len(te_df),
           "seconds": elapsed, "threads": int(os.environ.get("THREADS", "1")),
           "leak_l1": float(te_df.h_l1.isin(set(tr_df.h_l1)).mean()),
           "labels": {}}
    if regime != "external":
        rec["leak_cluster"] = float(te_df.cluster.isin(set(tr_df.cluster)).mean())
        seen_mask = te_df.cluster.isin(set(tr_df.cluster)).values
    else:
        seen_mask = te_df._seen.values
    for l in labels:
        sv, st = res[l]
        thr = best_threshold(va_df[l].values, sv) if model != "rule" else 0.5
        y = te_df[l].values.astype(int)
        r = {"thr": thr, "all": metrics(y, st, thr)}
        # split the test set by whether a near-duplicate exists in training
        for name, m in (("seen", seen_mask), ("unseen", ~seen_mask)):
            if m.sum() >= 30 and 0 < y[m].sum() < m.sum():
                r[name] = metrics(y[m], st[m], thr)
        rec["labels"][l] = r
    json.dump(rec, open(out, "w"), indent=1)
    macro = np.mean([rec["labels"][l]["all"]["mcc"] for l in labels])
    print(f"{dataset} {regime} {model} {seed} MCC={macro:.3f} leak={rec['leak_l1']:.2f} "
          f"{elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    ds, regime, model = sys.argv[1:4]
    seeds = [int(s) for s in sys.argv[4:]] or SEEDS
    for s in seeds:
        run(ds, regime, model, s)
