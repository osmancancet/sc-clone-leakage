"""Aggregate per-run JSON into results/numbers.json.

Statistics: mean and 95% t-interval over seeds (clipped to [-1, 1]); regime
gaps as an unpaired Welch comparison of Nadeau-Bengio corrected means (nb_welch)
with Holm correction over the family; operating points from confusion counts
pooled over seeds (pooled_op). Contract-level and bootstrap analyses are in
analyze_r1.py.
"""
import glob
import json
from collections import defaultdict

import numpy as np
from scipy import stats

from common import RESULTS

METRICS = ["mcc", "f1", "pr_auc", "fpr", "tpr", "prec_at_0.01", "prec_at_0.001"]
MODELS = ["rule", "knn", "lr", "cb", "cnn"]


def ci(x):
    x = np.asarray([v for v in x if v is not None and not np.isnan(v)], float)
    if len(x) == 0:
        return None
    m = float(x.mean())
    if len(x) < 2:
        return {"mean": m, "sd": 0.0, "lo": m, "hi": m, "n": 1}
    sd = float(x.std(ddof=1))
    h = float(stats.t.ppf(0.975, len(x) - 1) * sd / np.sqrt(len(x)))
    # t-interval clipped to the metric range [-1, 1]
    return {"mean": m, "sd": sd, "lo": max(m - h, -1.0), "hi": min(m + h, 1.0), "n": int(len(x))}


def macro(rec, part, metric):
    vals = []
    for l, r in rec["labels"].items():
        if part in r and r[part].get(metric) is not None:
            vals.append(r[part][metric])
    return float(np.mean(vals)) if vals else None


def nb_welch(a, b, ra, rb):
    """Difference of two regime means, each from J resampled splits.

    The regimes draw unrelated partitions for the same seed, so runs are not
    paired. Each mean gets the Nadeau-Bengio corrected variance
    (1/J + n_test/n_train) * s^2 and the two are compared with a Welch-type
    statistic (Satterthwaite degrees of freedom)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    va = (1 / len(a) + ra) * a.var(ddof=1)
    vb = (1 / len(b) + rb) * b.var(ddof=1)
    d = float(a.mean() - b.mean())
    se = np.sqrt(va + vb)
    if se == 0:
        return {"diff": d, "lo": d, "hi": d, "t": None, "df": None, "p": 0.0 if d else 1.0}
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    q = stats.t.ppf(0.975, df)
    t = d / se
    return {"diff": d, "lo": d - q * se, "hi": d + q * se, "t": float(t), "df": float(df),
            "p": float(2 * stats.t.sf(abs(t), df))}


def pooled_op(rs, prevalences=(0.01, 0.001)):
    """Operating point from confusion counts pooled over seeds, per label, then
    macro-averaged. Precision at prevalence pi follows Eq. 1 on the pooled
    TPR/FPR; the lower bound uses the one-sided 95% Clopper-Pearson upper bound
    of FPR and lower bound of TPR."""
    labs = list(rs[0]["labels"])
    per = []
    for l in labs:
        c = {k: sum(r["labels"][l]["all"][k] for r in rs) for k in ("tp", "fp", "fn", "tn")}
        P, N = c["tp"] + c["fn"], c["fp"] + c["tn"]
        tpr, fpr = c["tp"] / P, c["fp"] / N
        fpr_hi = 1.0 if c["fp"] == N else float(stats.beta.ppf(0.95, c["fp"] + 1, N - c["fp"]))
        tpr_lo = 0.0 if c["tp"] == 0 else float(stats.beta.ppf(0.05, c["tp"], P - c["tp"] + 1))
        e = {"tpr": tpr, "fpr": fpr, "fpr_hi": fpr_hi, "neg_per_seed": N / len(rs)}
        for pi in prevalences:
            den = tpr * pi + fpr * (1 - pi)
            e[f"prec_{pi}"] = tpr * pi / den if den > 0 else None
            den = tpr_lo * pi + fpr_hi * (1 - pi)
            e[f"prec_{pi}_lo"] = tpr_lo * pi / den if den > 0 else 0.0
        per.append(e)
    out = {k: float(np.mean([e[k] for e in per if e[k] is not None]))
           for k in per[0] if any(e[k] is not None for e in per)}
    # degenerate seeds: all-positive or all-negative predictions on some label
    out["degenerate_seeds"] = int(sum(any(r["labels"][l]["all"]["tp"] + r["labels"][l]["all"]["fp"] in
                                          (0, r["labels"][l]["all"]["n"]) for l in labs) for r in rs))
    return out


def holm(pvals):
    keys = sorted(pvals, key=pvals.get)
    m, out, prev = len(keys), {}, 0.0
    for i, k in enumerate(keys):
        adj = min(1.0, max(prev, (m - i) * pvals[k]))
        out[k] = adj
        prev = adj
    return out


def main():
    runs = [json.load(open(f)) for f in sorted(glob.glob(str(RESULTS / "runs" / "*.json")))]
    by = defaultdict(list)
    for r in runs:
        by[(r["dataset"], r["regime"], r["model"])].append(r)
    num = {"n_runs": len(runs), "cells": {}, "gaps": {}, "seen_unseen": {},
           "runtime": {}}

    for (ds, rg, m), rs in sorted(by.items()):
        rs = sorted(rs, key=lambda r: r["seed"])
        cell = {"n_seeds": len(rs), "n_train": rs[0]["n_train"], "n_val": rs[0]["n_val"],
                "n_test": rs[0]["n_test"], "leak_l1": ci([r["leak_l1"] for r in rs])}
        if "leak_cluster" in rs[0]:
            cell["leak_cluster"] = ci([r["leak_cluster"] for r in rs])
        for met in METRICS:
            cell[met] = ci([macro(r, "all", met) for r in rs])
        if ds == "slither":
            cell["per_label_mcc"] = {l: ci([r["labels"][l]["all"]["mcc"] for r in rs])
                                     for l in rs[0]["labels"]}
        for part in ("seen", "unseen"):
            v = [macro(r, part, "mcc") for r in rs]
            if all(x is not None for x in v):
                cell[f"mcc_{part}"] = ci(v)
                cell[f"n_{part}"] = ci([np.mean([lab[part]["n"] for lab in r["labels"].values()
                                                  if part in lab]) for r in rs])
        cell["seconds"] = ci([r["seconds"] for r in rs])
        cell["op"] = pooled_op(rs)
        cell["constant"] = all(r["labels"][l]["all"]["tp"] + r["labels"][l]["all"]["fp"] in
                               (0, r["labels"][l]["all"]["n"]) for r in rs for l in r["labels"])
        num["cells"][f"{ds}|{rg}|{m}"] = cell

    # regime gaps: optimistic regime minus clone-aware regime (unpaired, see nb_welch)
    pairs = [("messi_re", "random", "cluster"), ("messi_ts", "random", "cluster"),
             ("messi_re", "random", "dedup"), ("messi_ts", "random", "dedup"),
             ("slither", "random", "cluster"), ("slither", "official", "cluster"),
             ("slither", "random", "version")]
    pv = {}
    for ds, a, b in pairs:
        for m in MODELS:
            ra, rb = by.get((ds, a, m)), by.get((ds, b, m))
            if not ra or not rb or m == "rule":
                continue
            ratio = lambda rr: np.mean([r["n_test"] / r["n_train"] for r in rr])
            key = f"{ds}|{a}-{b}|{m}"
            num["gaps"][key] = nb_welch([macro(r, "all", "mcc") for r in ra],
                                        [macro(r, "all", "mcc") for r in rb], ratio(ra), ratio(rb))
            pv[key] = num["gaps"][key]["p"]
    for k, p in holm(pv).items():
        num["gaps"][k]["p_holm"] = p

    # seen vs unseen test rows inside optimistic regimes (same model, same threshold)
    pv2 = {}
    for ds, rg in (("slither", "official"), ("slither", "random"), ("messi_re", "random"),
                   ("messi_ts", "random")):
        for m in MODELS:
            rs = by.get((ds, rg, m))
            if not rs or m == "rule":
                continue
            d = [macro(r, "seen", "mcc") - macro(r, "unseen", "mcc") for r in rs
                 if macro(r, "seen", "mcc") is not None and macro(r, "unseen", "mcc") is not None]
            if len(d) >= 3:
                key = f"{ds}|{rg}|{m}"
                num["seen_unseen"][key] = {"diff": ci(d), "t_p": float(stats.ttest_1samp(d, 0).pvalue)}
                pv2[key] = num["seen_unseen"][key]["t_p"]
    for k, p in holm(pv2).items():
        num["seen_unseen"][k]["p_holm"] = p

    # over-fitting diagnostic (exp_fit.py): train / val / test MCC per run
    fits = defaultdict(list)
    for f in sorted(glob.glob(str(RESULTS / "fit" / "*.json"))):
        r = json.load(open(f))
        fits[(r["dataset"], r["regime"], r["model"])].append(r)
    num["fit"] = {}
    for (ds, rg, m), rs in sorted(fits.items()):
        e = {p: ci([np.mean([v[p] for v in r["labels"].values()]) for r in rs])
             for p in ("train", "val", "test")}
        e["gap_train_test"] = ci([np.mean([v["train"] - v["test"] for v in r["labels"].values()]) for r in rs])
        num["fit"][f"{ds}|{rg}|{m}"] = e

    num["runtime"] = {"main_runs_cpu_h": sum(r["seconds"] for r in runs) / 3600,
                      "by_model_cpu_h": {m: sum(r["seconds"] for r in runs if r["model"] == m) / 3600
                                         for m in MODELS}}
    emb = RESULTS / "runtime_embed.txt"
    if emb.exists():
        num["runtime"]["embed_note"] = emb.read_text().strip()
    json.dump(num, open(RESULTS / "numbers.json", "w"), indent=1)
    print("runs", len(runs), "cells", len(num["cells"]))


if __name__ == "__main__":
    main()
