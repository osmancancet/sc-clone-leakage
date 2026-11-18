"""Supplementary analyses, part 2: audit-only checks (no model training).

1. Near-duplicate threshold sensitivity: re-cluster at Jaccard 0.8 and 0.95 (0.9
   is the main setting) and report cluster counts and the share of the official
   Slither-Audited test (and validation) set with a near-duplicate in training.
2. Pragma-version buckets: size and label prevalence per Solidity minor version,
   and the share of 0.7/0.8 contracts that have a near-duplicate written for an
   older pragma (a proxy for how much of a new stream is "seen").
3. Tool versus expert labels: Slither-Audited labels against SmartBugs-Curated
   labels on contracts present verbatim (L1) in both datasets.
4. Messi-Q reentrancy provenance counts (rows, files, positives).
Writes results/audit_r1.json.
"""
import json

import numpy as np
import pandas as pd

import audit as A
from common import PROC, RESULTS

LAB = ["y_access_control", "y_arithmetic", "y_other", "y_reentrancy", "y_unchecked_calls"]
SB_MAP = {"y_reentrancy": "y_reentrancy", "y_unchecked_calls": "y_unchecked_low_level_calls",
          "y_access_control": "y_access_control", "y_arithmetic": "y_arithmetic"}


def leak(df, key):
    tr = df[df.split_official == "train"]
    s = set(tr[key])
    return {sp: float(df[df.split_official == sp][key].isin(s).mean()) for sp in ("val", "test")}


def main():
    out = {}
    sl = A.load("slither")
    sl["cluster"] = pd.read_parquet(PROC / "slither_clusters.parquet").cluster.values

    # 1. threshold sensitivity
    sens = {"0.9": {"clusters": int(sl.cluster.nunique()), **leak(sl, "cluster")}}
    for j in (0.8, 0.95):
        A.JACCARD = j
        cid, _ = A.near_dup_clusters(sl)
        tmp = sl.assign(c=cid)
        sens[str(j)] = {"clusters": int(tmp.c.nunique()), **leak(tmp, "c")}
        for name in ("messi_re", "messi_ts"):
            d = A.load(name)
            c2, _ = A.near_dup_clusters(d)
            sens[str(j)][f"{name}_clusters"] = int(len(set(c2)))
        print("J", j, sens[str(j)], flush=True)
    A.JACCARD = 0.9
    for name in ("messi_re", "messi_ts"):
        sens["0.9"][f"{name}_clusters"] = int(pd.read_parquet(PROC / f"{name}_clusters.parquet").cluster.nunique())
    out["jaccard_sensitivity"] = sens

    # 2. pragma-version buckets
    v = pd.to_numeric(sl.sol_minor, errors="coerce")
    out["version_buckets"] = {str(int(k)): {"n": int((v == k).sum()),
                                            **{l: float(sl.loc[v == k, l].mean()) for l in LAB}}
                              for k in sorted(v.dropna().unique())}
    old_cl = set(sl.cluster[v.isin([4, 5, 6])])
    new = sl[v.isin([7, 8])]
    out["new_with_old_neighbour"] = float(new.cluster.isin(old_cl).mean())
    out["new_n"] = int(len(new))

    # 3. tool vs expert labels on shared contracts
    sb = pd.read_parquet(PROC / "smartbugs.parquet")
    shared = sb.merge(sl.drop_duplicates("h_l1")[["h_l1"] + LAB], on="h_l1", suffixes=("_sb", ""))
    agree = {"n_shared": int(len(shared))}
    for l, s in SB_MAP.items():
        sbl = s + "_sb" if s in LAB else s
        a, b = shared[sbl].astype(int), shared[l].astype(int)
        agree[l] = {"sb_pos": int(a.sum()), "slither_pos": int(b.sum()),
                    "both": int((a & b).sum()), "agree": float((a == b).mean())}
    out["label_agreement"] = agree

    # 4. Messi-Q provenance
    for name in ("messi_re", "messi_ts"):
        d = pd.read_parquet(PROC / f"{name}.parquet")
        out[f"{name}_provenance"] = {"rows": int(len(d)), "files": int(d.file.nunique()) if "file" in d else None,
                                     "pos_rows": int(d.y.sum()), "unique_l1": int(d.h_l1.nunique()),
                                     "unique_pos_l1": int(d[d.y == 1].h_l1.nunique())}
    json.dump(out, open(RESULTS / "audit_r1.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
