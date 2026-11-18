"""Over-fitting diagnostic (added after the main runs, on request).

For TF-IDF+LR and TextCNN, report MCC on the training, validation and test
parts of the same run, at the validation-selected threshold. Same splits,
seeds and hyper-parameters as run.py; the main runs are not changed.
Writes results/fit/<dataset>__<regime>__<model>__<seed>.json.
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

from common import PROC, RESULTS, SEEDS
from run import LABELS, best_threshold, fit_predict, splits

OUT = RESULTS / "fit"
OUT.mkdir(exist_ok=True)


def one(ds, rg, m, seed):
    out = OUT / f"{ds}__{rg}__{m}__{seed}.json"
    if out.exists():
        return
    df = pd.read_parquet(PROC / f"pool_{ds}.parquet")
    labels = LABELS[ds]
    tr, va, te = splits(df, ds, rg, seed)
    tr_df, va_df, te_df = (df.iloc[i].reset_index(drop=True) for i in (tr, va, te))
    both = pd.concat([tr_df, te_df], ignore_index=True)
    res = fit_predict(m, ds, labels, tr_df, va_df, both, seed)
    rec = {"dataset": ds, "regime": rg, "model": m, "seed": seed, "labels": {}}
    for l in labels:
        sv, sb = res[l]
        thr = best_threshold(va_df[l].values, sv)
        st_tr, st_te = sb[:len(tr_df)], sb[len(tr_df):]
        rec["labels"][l] = {
            "train": float(matthews_corrcoef(tr_df[l].values, (st_tr >= thr).astype(int))),
            "val": float(matthews_corrcoef(va_df[l].values, (sv >= thr).astype(int))),
            "test": float(matthews_corrcoef(te_df[l].values, (st_te >= thr).astype(int)))}
    json.dump(rec, open(out, "w"), indent=1)
    mean = {p: np.mean([v[p] for v in rec["labels"].values()]) for p in ("train", "val", "test")}
    print(ds, rg, m, seed, {k: round(v, 3) for k, v in mean.items()}, flush=True)


if __name__ == "__main__":
    ds, rg, m = sys.argv[1:4]
    for s in [int(x) for x in sys.argv[4:]] or SEEDS:
        one(ds, rg, m, s)
