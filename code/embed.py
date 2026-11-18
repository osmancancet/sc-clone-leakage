"""Frozen CodeBERT embeddings (mean of the last hidden layer over the last
512 sub-word tokens; flattened sources put the main contract last).
Embeds each distinct h_l1 once. Output: $SC_WORK/processed/emb_<name>.npz
"""
import os
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from common import PROC, RESULTS
from config import CODEBERT


def main(names):
    torch.manual_seed(0)
    torch.set_num_threads(int(os.environ.get("THREADS", "4")))
    tok = AutoTokenizer.from_pretrained(CODEBERT["name"])
    tok.truncation_side = "left" if CODEBERT["side"] == "tail" else "right"
    model = AutoModel.from_pretrained(CODEBERT["name"]).eval()
    for name in names:
        out = PROC / f"emb_{name}.npz"
        if out.exists():
            continue
        df = pd.read_parquet(PROC / f"pool_{name}.parquet", columns=["h_l1", "text"])
        df = df.drop_duplicates("h_l1")
        # sort by length so batches pad little
        df = df.assign(n=df.text.str.len()).sort_values(["n", "h_l1"], kind="stable")
        texts, t0, CH = df.text.tolist(), time.time(), 1600
        # resumable: each chunk of CH rows is saved as soon as it is done
        parts = []
        for c0 in range(0, len(texts), CH):
            part = PROC / f"emb_{name}.part{c0 // CH:03d}.npy"
            if not part.exists():
                vecs = []
                for i in range(c0, min(c0 + CH, len(texts)), 16):
                    x = tok(texts[i:min(i + 16, c0 + CH)], truncation=True,
                            max_length=CODEBERT["max_len"], padding=True, return_tensors="pt")
                    with torch.inference_mode():
                        h = model(**x).last_hidden_state
                    m = x["attention_mask"].unsqueeze(-1).float()
                    vecs.append(((h * m).sum(1) / m.sum(1)).numpy().astype(np.float32))
                np.save(part, np.concatenate(vecs))
                print(name, c0 + CH, len(texts), f"{time.time() - t0:.0f}s", flush=True)
            parts.append(np.load(part))
        np.savez(out, h=np.array(df.h_l1.tolist()), x=np.concatenate(parts))
        for p in PROC.glob(f"emb_{name}.part*.npy"):
            p.unlink()
        with open(RESULTS / "runtime_embed.txt", "a") as f:
            f.write(f"{name}\t{len(texts)}\t{time.time() - t0:.1f}s\n")


if __name__ == "__main__":
    main(sys.argv[1:] or ["messi_re", "messi_ts", "smartbugs", "slither"])
