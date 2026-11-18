"""Build modelling tables: the Slither pool plus Messi-Q and SmartBugs.

Each table has: id, text (comment-stripped token stream), h_l1, cluster,
labels, and for Slither split_official and sol_minor.
"""
import glob

import numpy as np
import pandas as pd

from common import PROC, tokens
from config import POOL_SEED, SLITHER_POOL


def with_clusters(df, name):
    cl = pd.read_parquet(PROC / f"{name}_clusters.parquet")
    assert (cl.id.values == df.id.values).all()
    return df.assign(cluster=cl.cluster.values)


def main():
    files = sorted(f for f in glob.glob(str(PROC / "slither_[0-9].parquet")))
    cols = ["id", "source", "split_official", "sol_minor", "h_l1",
            "y_access_control", "y_arithmetic", "y_other", "y_reentrancy",
            "y_safe", "y_unchecked_calls"]
    ids = pd.concat([pd.read_parquet(f, columns=["id"]) for f in files], ignore_index=True)
    cl = pd.read_parquet(PROC / "slither_clusters.parquet")
    assert (cl.id.values == ids.id.values).all()
    rng = np.random.default_rng(POOL_SEED)
    keep = np.zeros(len(ids), bool)
    keep[rng.choice(len(ids), SLITHER_POOL, replace=False)] = True
    parts, off = [], 0
    for f in files:
        d = pd.read_parquet(f, columns=cols)
        k = keep[off:off + len(d)]
        c = cl.cluster.values[off:off + len(d)]
        assert (cl.id.values[off:off + len(d)] == d.id.values).all()
        parts.append(d[k].assign(cluster=c[k]))
        off += len(k)
    pool = pd.concat(parts, ignore_index=True)
    pool["text"] = pool.source.map(lambda s: " ".join(tokens(s)))
    pool.drop(columns=["source"]).to_parquet(PROC / "pool_slither.parquet", index=False)
    print("slither pool", len(pool), pool.cluster.nunique())

    for name in ("messi_re", "messi_ts", "smartbugs"):
        d = with_clusters(pd.read_parquet(PROC / f"{name}.parquet"), name)
        d["text"] = d.source.map(lambda s: " ".join(tokens(s)))
        d.drop(columns=["source", "l2"]).to_parquet(PROC / f"pool_{name}.parquet", index=False)
        print(name, len(d))


if __name__ == "__main__":
    main()
