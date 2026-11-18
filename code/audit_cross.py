"""Cross-dataset overlap: share of each small dataset's rows whose L1 or type-2
(L2) hash also occurs in Slither-Audited, and SmartBugs-Curated vs Messi-Q.
Writes results/audit_cross.json."""
import glob
import json

import pandas as pd

from common import PROC, RESULTS


def main():
    files = [f for f in sorted(glob.glob(str(PROC / "slither_[0-9].parquet")))]
    sl = pd.concat([pd.read_parquet(f, columns=["h_l1", "h_l2"]) for f in files])
    S1, S2 = set(sl.h_l1), set(sl.h_l2)
    out = {}
    for name in ("messi_re", "messi_ts", "smartbugs", "solidifi"):
        d = pd.read_parquet(PROC / f"{name}.parquet", columns=["h_l1", "h_l2"])
        out[f"{name}_in_slither_l1"] = float(d.h_l1.isin(S1).mean())
        out[f"{name}_in_slither_l2"] = float(d.h_l2.isin(S2).mean())
    sb = pd.read_parquet(PROC / "smartbugs.parquet", columns=["h_l1", "h_l2", "y_reentrancy"])
    mr = pd.read_parquet(PROC / "messi_re.parquet", columns=["h_l1", "h_l2"])
    out["smartbugs_in_messi_re_l1"] = float(sb.h_l1.isin(set(mr.h_l1)).mean())
    out["smartbugs_in_messi_re_l2"] = float(sb.h_l2.isin(set(mr.h_l2)).mean())
    out["smartbugs_n_reentrancy"] = int(sb.y_reentrancy.sum())
    json.dump(out, open(RESULTS / "audit_cross.json", "w"), indent=1)
    print(out)


if __name__ == "__main__":
    main()
