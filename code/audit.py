"""Clone audit: exact, whitespace/comment-insensitive, type-2 and near duplicates.

Near duplicates: MinHash (128 permutations) over 5-gram shingles of the type-2
token stream, LSH with 16 bands x 8 rows, candidate pairs kept when the
estimated Jaccard similarity is >= JACCARD. Clusters are connected components.
Writes $SC_WORK/processed/<name>_clusters.parquet and results/audit.json.
"""
import glob
import json
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from common import PROC, RESULTS, SEEDS

NPERM, BANDS, ROWS, K = 128, 16, 8, 5
JACCARD = 0.9
P = np.uint64((1 << 31) - 1)
_rng = np.random.default_rng(12345)
A = _rng.integers(1, int(P), NPERM, dtype=np.uint64)
B = _rng.integers(0, int(P), NPERM, dtype=np.uint64)
VOCAB = {}


def signature(l2: str):
    ids = np.array([VOCAB.setdefault(t, len(VOCAB) + 1) for t in l2.split()],
                   dtype=np.uint64)
    if len(ids) < K:
        ids = np.pad(ids, (0, K - len(ids)))
    # polynomial rolling hash of each K-gram, reduced mod P
    h = np.zeros(len(ids) - K + 1, dtype=np.uint64)
    for j in range(K):
        h = (h * np.uint64(1000003) + ids[j:len(ids) - K + 1 + j]) % P
    h = np.unique(h)
    return ((A[:, None] * h[None, :] + B[:, None]) % P).min(axis=1).astype(np.uint32)


class DSU:
    def __init__(self, n):
        self.p = np.arange(n)

    def find(self, x):
        p = self.p
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)


def near_dup_clusters(df):
    """Cluster ids per row; rows with the same type-2 hash start merged."""
    rep = df.drop_duplicates("h_l2")
    rep_idx = {h: i for i, h in enumerate(rep.h_l2)}
    sigs = np.stack([signature(s) for s in rep.l2])
    n = len(rep)
    dsu = DSU(n)
    n_pairs = 0
    for b in range(BANDS):
        band = sigs[:, b * ROWS:(b + 1) * ROWS]
        buckets = defaultdict(list)
        for i, key in enumerate(map(bytes, band)):
            buckets[key].append(i)
        for members in buckets.values():
            if len(members) < 2:
                continue
            m = np.array(members)
            # Large buckets: compare every member to up to 64 pivots.
            pivots = m if len(m) <= 400 else m[np.linspace(0, len(m) - 1, 64).astype(int)]
            sim = (sigs[pivots][:, None, :] == sigs[m][None, :, :]).mean(axis=2)
            pi, mi = np.nonzero(sim >= JACCARD)
            n_pairs += len(pi)
            for a, c in zip(pivots[pi], m[mi]):
                if a != c:
                    dsu.union(a, c)
    roots = np.array([dsu.find(i) for i in range(n)])
    cid = df.h_l2.map(lambda h: int(roots[rep_idx[h]])).to_numpy()
    return cid, n_pairs


def load(name):
    if name == "slither":
        files = sorted(glob.glob(str(PROC / "slither_*.parquet")))
        files = [f for f in files if "clusters" not in f]
        cols = ["id", "split_official", "h_raw", "h_l1", "h_l2", "l2", "n_tokens",
                "sol_minor", "y_access_control", "y_arithmetic", "y_other",
                "y_reentrancy", "y_safe", "y_unchecked_calls"]
        return pd.concat([pd.read_parquet(f, columns=cols) for f in files],
                         ignore_index=True)
    df = pd.read_parquet(PROC / f"{name}.parquet").drop(columns=["source"])
    if "bug" in df:
        df["y_bug"] = df.bug
    return df


def label_cols(df):
    return [c for c in df.columns if c.startswith("y")]


def conflict_rate(df, key):
    """Share of duplicate groups (size>1) whose members disagree on any label."""
    ys = label_cols(df)
    g = df.groupby(key)
    sizes = g.size()
    multi = sizes[sizes > 1].index
    if len(multi) == 0:
        return 0.0, 0
    sub = df[df[key].isin(multi)]
    nun = sub.groupby(key)[ys].nunique().max(axis=1)
    return float((nun > 1).mean()), int(len(multi))


def cross_contam(train, test, key):
    s = set(train[key])
    return float(test[key].isin(s).mean())


def random_split_contam(df, key, test_frac=0.2):
    vals = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        mask = rng.random(len(df)) < test_frac
        vals.append(cross_contam(df[~mask], df[mask], key))
    return float(np.mean(vals)), float(np.std(vals, ddof=1))


def audit(name, df):
    cid, n_pairs = near_dup_clusters(df)
    df = df.assign(cluster=cid)
    out = {"n": len(df)}
    for key in ("h_raw", "h_l1", "h_l2", "cluster"):
        u = df[key].nunique()
        sizes = df.groupby(key).size()
        out[key] = {
            "unique": int(u),
            "unique_frac": u / len(df),
            "rows_in_multi": float(sizes[sizes > 1].sum() / len(df)),
            "largest": int(sizes.max()),
            "largest_frac": float(sizes.max() / len(df)),
        }
        out[key]["conflict_groups_frac"], out[key]["multi_groups"] = conflict_rate(df, key)
        out[key]["random_split_test_in_train_mean"], out[key]["random_split_test_in_train_sd"] = \
            random_split_contam(df, key)
    out["lsh_pairs_checked"] = n_pairs
    ys = label_cols(df)
    out["label_rates"] = {c: float(df[c].mean()) for c in ys if pd.api.types.is_numeric_dtype(df[c])}
    if "file" in df:
        out["n_files"] = int(df.file.nunique())
    for c in ys:
        if pd.api.types.is_numeric_dtype(df[c]) and set(df[c].unique()) <= {0, 1}:
            out.setdefault("unique_pos_l1", {})[c] = int(df[df[c] == 1].h_l1.nunique())
    out["tokens_median"] = float(df.n_tokens.median())
    if "split_official" in df:
        tr = df[df.split_official == "train"]
        for sp in ("val", "test"):
            te = df[df.split_official == sp]
            out[f"official_{sp}_n"] = len(te)
            for key in ("h_raw", "h_l1", "h_l2", "cluster"):
                out[f"official_{sp}_in_train_{key}"] = cross_contam(tr, te, key)
    df[["id", "cluster"]].to_parquet(PROC / f"{name}_clusters.parquet", index=False)
    return out


def main(names):
    path = RESULTS / "audit.json"
    res = json.load(open(path)) if path.exists() else {}
    for name in names:
        df = load(name)
        res[name] = audit(name, df)
        print(name, json.dumps({k: res[name][k] for k in ("n", "h_l1", "cluster")}), flush=True)
        json.dump(res, open(path, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:] or ["messi_resnip", "messi_re", "messi_ts", "smartbugs",
                          "solidifi", "slither"])
