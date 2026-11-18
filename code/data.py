"""Build one parquet file per dataset with source, labels and hashes.

Outputs (in $SC_WORK/processed): slither.parquet, messi_re.parquet,
messi_ts.parquet, messi_resnip.parquet, smartbugs.parquet, solidifi.parquet,
and results/data_manifest.json with commits and SHA-256 digests.
"""
import glob
import json
import re
import subprocess

import pandas as pd

from common import (PROC, RAW, RESULTS, sha256_file, sha256_text, solidity_minor,
                    tokens, type2)

SLITHER_BIG = ['access-control', 'arithmetic', 'other', 'reentrancy',
               'safe', 'unchecked-calls']
SB_CATS = ['access_control', 'arithmetic', 'bad_randomness', 'denial_of_service',
           'front_running', 'other', 'reentrancy', 'short_addresses',
           'time_manipulation', 'unchecked_low_level_calls']


def add_hashes(df):
    toks = df.source.map(tokens)
    df["n_tokens"] = toks.map(len)
    df["h_raw"] = df.source.map(sha256_text)
    df["h_l1"] = toks.map(lambda t: sha256_text(" ".join(t)))
    df["l2"] = toks.map(lambda t: " ".join(type2(t)))
    df["h_l2"] = df.l2.map(sha256_text)
    df["sol_minor"] = df.source.map(solidity_minor)
    return df


def slither_shards():
    """Yield one processed frame per raw parquet shard (keeps memory low)."""
    d = RAW / "hf_slither" / "data"
    for f in sorted(glob.glob(str(d / "raw" / "*.parquet"))):
        yield slither(pd.read_parquet(f, columns=["contracts", "source_code", "results"]))


def slither(df):
    d = RAW / "hf_slither" / "data"
    maps = json.load(open(d / "label_mappings.json"))
    splits = pd.read_csv(d / "big-splits.csv")
    df = df.merge(splits, on="contracts", how="inner")
    rows = []
    for r in df.itertuples(index=False):
        res = json.loads(r.results)
        if not res["results"]:
            cls = {"safe"}
        else:
            cls = {maps[x["check"]] for x in res["results"]["detectors"]} - {"ignore"}
            cls = cls or {"safe"}
        rows.append(cls)
    out = pd.DataFrame({"id": df.contracts, "source": df.source_code,
                        "split_official": df.split})
    for c in SLITHER_BIG:
        out["y_" + c.replace("-", "_")] = [int(c in s) for s in rows]
    # contracts whose labels fall outside the six "big" classes
    out["has_other_class"] = [int(bool(s - set(SLITHER_BIG))) for s in rows]
    return out


_SEP = re.compile(r'^-{10,}\s*$', re.M)


def messi(fname, snippet=False):
    text = open(RAW / "Messi-Q_ReChecker" / "train_data" / fname,
                encoding="utf-8", errors="replace").read()
    rows = []
    for block in _SEP.split(text):
        lines = [l for l in block.strip("\n").split("\n")]
        while lines and not lines[-1].strip():
            lines.pop()
        if len(lines) < 3:
            continue
        head = lines[0].split()
        label = int(lines[-1].strip())
        body = lines[2:-1] if snippet else lines[1:-1]
        rows.append({"id": f"{head[0]}_{head[1]}", "file": head[1],
                     "source": "\n".join(body), "y": label})
    return pd.DataFrame(rows)


def smartbugs():
    base = RAW / "smartbugs_smartbugs-curated"
    meta = json.load(open(base / "vulnerabilities.json"))
    rows = []
    for m in meta:
        src = open(base / m["path"], encoding="utf-8", errors="replace").read()
        cats = {v["category"] for v in m["vulnerabilities"]}
        row = {"id": m["path"], "source": src}
        for c in SB_CATS:
            row["y_" + c] = int(c in cats)
        rows.append(row)
    return pd.DataFrame(rows)


def solidifi():
    base = RAW / "DependableSystemsLab_SolidiFI-benchmark" / "buggy_contracts"
    rows = []
    for f in sorted(glob.glob(str(base / "*" / "*.sol"))):
        bug = f.split("/")[-2]
        base_id = int(re.search(r'(\d+)\.sol$', f).group(1))
        rows.append({"id": f"{bug}/{base_id}", "bug": bug, "base_id": base_id,
                     "source": open(f, encoding="utf-8", errors="replace").read()})
    return pd.DataFrame(rows)


def git_head(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"],
                                   text=True).strip()


def main():
    parts = []
    for i, df in enumerate(slither_shards()):
        df = add_hashes(df)
        df.to_parquet(PROC / f"slither_{i}.parquet", index=False)
        parts.append(len(df))
        print("slither shard", i, len(df), flush=True)
        del df
    summary = {"slither": sum(parts)}
    builders = {
        "messi_re": lambda: messi("reentrancy_1671.txt"),
        "messi_ts": lambda: messi("timestamp.txt"),
        "messi_resnip": lambda: messi("reentrancy_code_snippets_2000.txt", snippet=True),
        "smartbugs": smartbugs,
        "solidifi": solidifi,
    }
    for name, fn in builders.items():
        df = add_hashes(fn())
        df.to_parquet(PROC / f"{name}.parquet", index=False)
        summary[name] = len(df)
        print(name, len(df), flush=True)

    manifest = {"n_rows": summary, "sources": {}}
    repos = {
        "Messi-Q_ReChecker": "https://github.com/Messi-Q/ReChecker",
        "smartbugs_smartbugs-curated": "https://github.com/smartbugs/smartbugs-curated",
        "DependableSystemsLab_SolidiFI-benchmark": "https://github.com/DependableSystemsLab/SolidiFI-benchmark",
    }
    for d, url in repos.items():
        manifest["sources"][d] = {"url": url, "commit": git_head(RAW / d)}
    manifest["sources"]["hf_slither"] = {
        "url": "https://huggingface.co/datasets/mwritescode/slither-audited-smart-contracts",
        "revision": "13594107c7afa216cb0c126f38b8ff6548112dcf"}
    files = sorted(glob.glob(str(RAW / "hf_slither" / "data" / "**" / "*.*"), recursive=True))
    files += [str(RAW / "Messi-Q_ReChecker" / "train_data" / f) for f in
              ("reentrancy_1671.txt", "timestamp.txt", "reentrancy_code_snippets_2000.txt")]
    files += [str(RAW / "smartbugs_smartbugs-curated" / "vulnerabilities.json")]
    manifest["sha256"] = {str(p).split("/data/raw/")[1]: sha256_file(p) for p in files}
    json.dump(manifest, open(RESULTS / "data_manifest.json", "w"), indent=1)


if __name__ == "__main__":
    main()
