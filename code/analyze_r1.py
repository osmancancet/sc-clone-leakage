"""Supplementary analyses, part 3: aggregate the saved per-contract scores.

Input : results/scores/*.npz (exp_r1.py), results/runs/*.json, processed pools.
Output: results/numbers_r1.json with
  boot        hierarchical bootstrap 95% interval of macro MCC per cell
              (resample seeds; resample near-duplicate clusters of the test
              part once per draw and apply the same weights in every seed,
              because seeds share test contracts; external test: contracts),
              B = 2000. Regime gaps use the NB-corrected Welch test in
              analyze.py only (a bootstrap over seeds ignores the overlap of
              training sets, so it is not reported).
  contract    Messi-Q contract-level metrics (one row per distinct L1 contract)
  seen_unseen paired cluster bootstrap (shared cluster weights) of MCC(seen) - MCC(unseen), plus
              prevalence, ROC-AUC and median length per subset
  dose        macro MCC and ROC-AUC binned by max TF-IDF similarity to training,
              with MCC of seen and unseen contracts inside each bin
  oracle      TPR at FPR = 1% and precision in the top 5% (chosen on test,
              labelled oracle in the paper)
  external    Messi-Q -> SmartBugs: bootstrap intervals over contracts, paired
              rule-vs-model differences, MCC of each model at the rule's alert
              rate, subset without a type-2 neighbour
  version     per-stratum (0.7 / 0.8) results of the pragma-version split
  agreement   max |MCC(npz) - MCC(run json)| per model (reproduction check)
"""
import glob
import json
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from common import PROC, RESULTS

B = int(__import__("os").environ.get("BOOT", 2000))
RNG = np.random.default_rng(2026)
BINS = [0.0, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01]


def load_scores():
    by = defaultdict(dict)
    for f in sorted(glob.glob(str(RESULTS / "scores" / "*.npz"))):
        ds, rg, m, seed = f.split("/")[-1][:-4].split("__")
        d = np.load(f)
        by[(ds, rg, m)][int(seed)] = {k: d[k] for k in d.files}
    return by


def mcc_counts(c):
    """c[..., 4] = tp, fp, fn, tn -> MCC (0 when undefined)."""
    tp, fp, fn, tn = (c[..., i].astype(float) for i in range(4))
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (tp * tn - fp * fn) / den, 0.0)


def unit_counts(d, mask=None):
    """Per-unit confusion counts: array (U, L, 4) and the unit ids."""
    units = d["cluster"] if d["cluster"].min() >= 0 else np.arange(len(d["y"]))
    if mask is not None:
        units = units[mask]
    y = d["y"] if mask is None else d["y"][mask]
    p = (d["s"] >= d["thr"][None, :]) if mask is None else (d["s"][mask] >= d["thr"][None, :])
    u, inv = np.unique(units, return_inverse=True)
    C = np.zeros((len(u), y.shape[1], 4))
    for j in range(y.shape[1]):
        for k, (pp, yy) in enumerate(((1, 1), (1, 0), (0, 1), (0, 0))):
            np.add.at(C[:, j, k], inv, ((p[:, j] == pp) & (y[:, j] == yy)).astype(float))
    return C, u


def shared_units(runs, masks=None):
    """Per-seed unit counts mapped to one global unit index, so that a bootstrap
    draw gives the same weight to a test unit in every seed that contains it
    (seeds share most test contracts; independent per-seed resampling would
    understate the variance)."""
    seeds = sorted(runs)
    per = []
    for s in seeds:
        d = runs[s]
        C, u = unit_counts(d, None if masks is None else masks[s])
        per.append((C, u))
    allu = np.unique(np.concatenate([u for _, u in per]))
    return [(C, np.searchsorted(allu, u)) for C, u in per], len(allu)


def draw_weights(n_units, rng):
    return rng.multinomial(n_units, np.full(n_units, 1 / n_units)).astype(float)


def boot_cell(runs, b=B, rng=RNG):
    """Hierarchical bootstrap of the seed-mean macro MCC: seeds resampled with
    replacement; test units (near-duplicate clusters, or contracts on the
    external test) resampled once per draw and shared across seeds."""
    per, U = shared_units(runs)
    out = np.empty(b)
    for i in range(b):
        w = draw_weights(U, rng)
        pick = rng.integers(0, len(per), len(per))
        out[i] = np.mean([mcc_counts(np.tensordot(w[per[k][1]], per[k][0], axes=1)).mean() for k in pick])
    return out


def summary(x):
    return {"lo": float(np.percentile(x, 2.5)), "hi": float(np.percentile(x, 97.5)),
            "boot_mean": float(x.mean())}


def point_mcc(d):
    C, _ = unit_counts(d)
    return float(mcc_counts(C.sum(0)).mean())


def contract_level(d):
    """Collapse rows to distinct L1 contracts (mean score, shared label)."""
    df = pd.DataFrame({"h": d["h"], "s": d["s"][:, 0], "y": d["y"][:, 0]})
    g = df.groupby("h").agg(s=("s", "mean"), y=("y", "max"))
    p = (g.s >= d["thr"][0]).astype(int)
    c = np.array([((p == 1) & (g.y == 1)).sum(), ((p == 1) & (g.y == 0)).sum(),
                  ((p == 0) & (g.y == 1)).sum(), ((p == 0) & (g.y == 0)).sum()])
    both = 0 < g.y.sum() < len(g)
    return {"mcc": float(mcc_counts(c)), "n": int(len(g)), "pos": int(g.y.sum()),
            "pr_auc": float(average_precision_score(g.y, g.s)) if both else None,
            "roc_auc": float(roc_auc_score(g.y, g.s)) if both else None,
            "largest_cluster_share": float(pd.Series(d["cluster"]).value_counts(normalize=True).iloc[0])}


def mean_ci(v):
    v = np.array([x for x in v if x is not None], float)
    from scipy import stats
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0
    return {"mean": float(v.mean()), "lo": float(max(v.mean() - h, -1)), "hi": float(min(v.mean() + h, 1)),
            "n": int(len(v))}


def macro_auc(y, s):
    v = [roc_auc_score(y[:, j], s[:, j]) for j in range(y.shape[1]) if 0 < y[:, j].sum() < len(y)]
    return float(np.mean(v)) if v else None


def seen_unseen(runs, b=B, rng=RNG):
    seeds = sorted(runs)
    # one weight per cluster id for both subsets: map both onto a common index
    allu = np.unique(np.concatenate([unit_counts(runs[s])[1] for s in seeds]))
    def remap(mask_fn):
        out = []
        for s in seeds:
            C, u = unit_counts(runs[s], mask_fn(runs[s]))
            out.append((C, np.searchsorted(allu, u)))
        return out
    ps, pu = remap(lambda d: d["seen"]), remap(lambda d: ~d["seen"])
    prep = [(a[0], b_[0]) for a, b_ in zip(ps, pu)]
    diff = np.empty(b)
    for i in range(b):
        w = draw_weights(len(allu), rng)
        pick = rng.integers(0, len(seeds), len(seeds))
        diff[i] = np.mean([mcc_counts(np.tensordot(w[ps[k][1]], ps[k][0], 1)).mean()
                           - mcc_counts(np.tensordot(w[pu[k][1]], pu[k][0], 1)).mean() for k in pick])
    pt = np.mean([mcc_counts(p[0].sum(0)).mean() - mcc_counts(p[1].sum(0)).mean() for p in prep])
    sub = {}
    for name, sel in (("seen", lambda d: d["seen"]), ("unseen", lambda d: ~d["seen"])):
        sub[name] = {"prevalence": float(np.mean([runs[s]["y"][sel(runs[s])].mean() for s in seeds])),
                     "roc_auc": float(np.mean([macro_auc(runs[s]["y"][sel(runs[s])], runs[s]["s"][sel(runs[s])])
                                               for s in seeds])),
                     "n": float(np.mean([sel(runs[s]).sum() for s in seeds]))}
    return {"diff": float(pt), **summary(diff),
            "p": float(max(2 * min((diff <= 0).mean(), (diff >= 0).mean()), 1 / b)), **sub}


def dose(runs):
    rows = []
    for s, d in runs.items():
        b = np.digitize(d["sim"], BINS) - 1
        for k in range(len(BINS) - 1):
            m = b == k
            if m.sum() == 0:
                continue
            C, _ = unit_counts(d, m)
            Cs = unit_counts(d, m & d["seen"])[0].sum(0) if (m & d["seen"]).any() else 0
            Cu = unit_counts(d, m & ~d["seen"])[0].sum(0) if (m & ~d["seen"]).any() else 0
            rows.append((k, s, C.sum(0), d["y"][m], d["s"][m], Cs, Cu, (m & d["seen"]).sum()))
    out = {}
    for k in range(len(BINS) - 1):
        sel = [r for r in rows if r[0] == k]
        if not sel:
            continue
        C = sum(r[2] for r in sel)
        y = np.concatenate([r[3] for r in sel]); s = np.concatenate([r[4] for r in sel])
        out[f"{BINS[k]:.2f}-{min(BINS[k + 1], 1):.2f}"] = {
            "n_per_seed": float(len(y) / len(runs)), "mcc": float(mcc_counts(C).mean()),
            "roc_auc": macro_auc(y, s),
            "seen_share": float(sum(r[7] for r in sel) / len(y)),
            "mcc_seen": float(mcc_counts(sum(r[5] for r in sel)).mean()) if any(np.ndim(r[5]) for r in sel) else None,
            "mcc_unseen": float(mcc_counts(sum(r[6] for r in sel)).mean()) if any(np.ndim(r[6]) for r in sel) else None}
    return out


def oracle(runs):
    tpr1, top5 = [], []
    for d in runs.values():
        for j in range(d["y"].shape[1]):
            y, s = d["y"][:, j], d["s"][:, j]
            if not 0 < y.sum() < len(y):
                continue
            neg = np.sort(s[y == 0])[::-1]
            t = neg[max(int(np.floor(0.01 * len(neg))) - 1, 0)] if len(neg) else np.inf
            tpr1.append(float((s[y == 1] > t).mean()))
            k = max(int(round(0.05 * len(y))), 1)
            top5.append(float(y[np.argsort(-s)[:k]].mean()))
    return {"tpr_at_fpr1": float(np.mean(tpr1)), "prec_top5": float(np.mean(top5))}


def ext_boot(by, models, b=B, rng=RNG):
    """Messi-Q -> SmartBugs reentrancy: contract bootstrap, paired across models."""
    runs = {m: by[("messi_re", "external", m)] for m in models}
    n = len(next(iter(runs["lr"].values()))["y"])
    stats_m = {m: {"mcc": np.empty(b), "pr_auc": np.empty(b)} for m in models}
    for i in range(b):
        pick = rng.integers(0, 10, 10)
        ix0 = rng.integers(0, n, n)
        idx = [ix0 for _ in pick]
        for m in models:
            mc, pa = [], []
            for k, ix in zip(pick, idx):
                d = runs[m][int(k)]
                y, s = d["y"][ix, 0], d["s"][ix, 0]
                p = (s >= d["thr"][0]).astype(int)
                c = np.array([((p == 1) & (y == 1)).sum(), ((p == 1) & (y == 0)).sum(),
                              ((p == 0) & (y == 1)).sum(), ((p == 0) & (y == 0)).sum()])
                mc.append(mcc_counts(c))
                pa.append(average_precision_score(y, s) if 0 < y.sum() < len(y) else np.nan)
            stats_m[m]["mcc"][i] = np.mean(mc)
            stats_m[m]["pr_auc"][i] = np.nanmean(pa)
    out = {m: {k: summary(v) for k, v in stats_m[m].items()} for m in models}
    for m in models:
        if m == "rule":
            continue
        for k in ("mcc", "pr_auc"):
            dd = stats_m["rule"][k] - stats_m[m][k]
            out[f"rule-{m}"] = out.get(f"rule-{m}", {})
            out[f"rule-{m}"][k] = {**summary(dd), "p": float(max(2 * min((dd <= 0).mean(), (dd >= 0).mean()), 1 / b))}
    # each model at the rule's alert rate (same number of flagged contracts per seed)
    for m in models:
        mc = []
        for k, d in runs[m].items():
            r = runs["rule"][k]
            kk = int((r["s"][:, 0] >= r["thr"][0]).sum())
            y, s_ = d["y"][:, 0], d["s"][:, 0]
            p = np.zeros(len(y), int); p[np.argsort(-s_, kind="stable")[:kk]] = 1
            c = np.array([((p == 1) & (y == 1)).sum(), ((p == 1) & (y == 0)).sum(),
                          ((p == 0) & (y == 1)).sum(), ((p == 0) & (y == 0)).sum()])
            mc.append(float(mcc_counts(c)))
        out[m]["matched_rate_mcc"] = float(np.mean(mc))
    # subset of SmartBugs contracts without a type-2 (L2) match anywhere in Messi-Q reentrancy
    sb = pd.read_parquet(PROC / "smartbugs.parquet", columns=["h_l1", "h_l2"])
    mr = set(pd.read_parquet(PROC / "messi_re.parquet", columns=["h_l2"]).h_l2)
    l2 = dict(zip(sb.h_l1, sb.h_l2))
    for m in models:
        mc, pa = [], []
        for d in runs[m].values():
            keep = np.array([l2[h] not in mr for h in d["h"]])
            y, s = d["y"][keep, 0], d["s"][keep, 0]
            p = (s >= d["thr"][0]).astype(int)
            c = np.array([((p == 1) & (y == 1)).sum(), ((p == 1) & (y == 0)).sum(),
                          ((p == 0) & (y == 1)).sum(), ((p == 0) & (y == 0)).sum()])
            mc.append(float(mcc_counts(c))); pa.append(float(average_precision_score(y, s)))
        out[m]["no_l2_match"] = {"mcc": float(np.mean(mc)), "pr_auc": float(np.mean(pa)),
                                 "n": int(keep.sum()), "pos": int(y.sum())}
    return out


def main():
    by = load_scores()
    res = {"B": B, "boot": {}, "contract": {}, "seen_unseen": {}, "dose": {},
           "oracle": {}, "agreement": {}, "version": {}, "residual_sim": {}}
    draws = {}
    # reproduction check against the main runs
    for (ds, rg, m), runs in by.items():
        for s, d in runs.items():
            rj = json.load(open(RESULTS / "runs" / f"{ds}__{rg}__{m}__{s}.json"))
            ref = np.mean([v["all"]["mcc"] for v in rj["labels"].values()])
            dd = abs(point_mcc(d) - ref)
            res["agreement"][m] = max(res["agreement"].get(m, 0.0), float(dd))
    for (ds, rg, m), runs in sorted(by.items()):
        if len(runs) < 10:
            continue
        key = f"{ds}|{rg}|{m}"
        x = boot_cell(runs)
        draws[key] = x
        res["boot"][key] = {"point": float(np.mean([point_mcc(d) for d in runs.values()])), **summary(x)}
        if rg != "external":
            res["oracle"][key] = oracle(runs)
            res["residual_sim"][key] = {t: float(np.mean([(d["sim"] >= t).mean() for d in runs.values()]))
                                        for t in (0.8, 0.9, 0.95)}
        if ds.startswith("messi") and rg != "external":
            per = [contract_level(d) for d in runs.values()]
            res["contract"][key] = {k: mean_ci([p[k] for p in per]) for k in per[0]}
            res["contract"][key]["n_min"] = min(p["n"] for p in per)
            res["contract"][key]["n_max"] = max(p["n"] for p in per)
            res["contract"][key]["pos_min"] = min(p["pos"] for p in per)
            res["contract"][key]["pos_max"] = max(p["pos"] for p in per)
        if ds == "slither" and rg in ("official", "random", "cluster"):
            res["dose"][key] = dose(runs)
        if ds == "slither" and rg in ("official", "random"):
            res["seen_unseen"][key] = seen_unseen(runs)
        if ds == "slither" and rg == "version":
            res["version"][key] = {}
            for v in (7, 8):
                C = sum(unit_counts(d, d["sol_minor"] == v)[0].sum(0) for d in runs.values())
                y = np.concatenate([d["y"][d["sol_minor"] == v] for d in runs.values()])
                s = np.concatenate([d["s"][d["sol_minor"] == v] for d in runs.values()])
                res["version"][key][str(v)] = {"mcc": float(mcc_counts(C).mean()), "roc_auc": macro_auc(y, s),
                                               "n_per_seed": float(len(y) / len(runs)),
                                               "prevalence": float(y.mean())}
        if ds == "slither" and rg == "cluster":
            res.setdefault("largest_test_cluster_share", {})[m] = [
                float(pd.Series(d["cluster"]).value_counts(normalize=True).iloc[0]) for d in runs.values()]
        print(key, res["boot"][key], flush=True)
    from analyze import holm
    for k, p in holm({k: v["p"] for k, v in res["seen_unseen"].items()}).items():
        res["seen_unseen"][k]["p_holm"] = p
    models = [m for m in ("rule", "knn", "lr", "cb", "cnn") if len(by.get(("messi_re", "external", m), {})) == 10]
    res["external"] = ext_boot(by, models)
    json.dump(res, open(RESULTS / "numbers_r1.json", "w"), indent=1)
    print("written", len(res["boot"]), "cells")


if __name__ == "__main__":
    main()
