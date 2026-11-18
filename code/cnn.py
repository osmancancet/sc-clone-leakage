"""Token-level TextCNN (Kim, 2014) trained from scratch, multi-label head.

Input: the last max_len tokens of the comment-stripped token stream
(flattened sources put the main contract last). Early stopping on the
validation loss. Returns sigmoid scores for validation and test rows.
"""
import os
from collections import Counter

import numpy as np
import torch
import torch.nn as nn


class TextCNN(nn.Module):
    def __init__(self, vocab, emb, filters, widths, dropout, n_out):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.convs = nn.ModuleList(nn.Conv1d(emb, filters, w) for w in widths)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(filters * len(widths), n_out)

    def forward(self, x):
        e = self.emb(x).transpose(1, 2)
        h = torch.cat([torch.relu(c(e)).amax(dim=2) for c in self.convs], dim=1)
        return self.out(self.drop(h))


def encode(texts, vocab, max_len):
    X = np.zeros((len(texts), max_len), dtype=np.int64)
    for i, t in enumerate(texts):
        ids = [vocab.get(w, 1) for w in t.split()[-max_len:]]
        X[i, :len(ids)] = ids
    return torch.from_numpy(X)


def scores(model, X, batch):
    model.eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(X), batch * 4):
            out.append(torch.sigmoid(model(X[i:i + batch * 4])).numpy())
    return np.concatenate(out)


def train_cnn(tr, ytr, va, yva, te, seed, cfg):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(int(os.environ.get("THREADS", "1")))
    cnt = Counter(w for t in tr for w in t.split()[-cfg["max_len"]:])
    vocab = {w: i + 2 for i, (w, _) in enumerate(cnt.most_common(cfg["vocab"] - 2))}
    Xtr, Xva, Xte = (encode(t, vocab, cfg["max_len"]) for t in (tr, va, te))
    Ytr = torch.tensor(ytr, dtype=torch.float32)
    Yva = torch.tensor(yva, dtype=torch.float32)
    pos = Ytr.mean(0).clamp(1e-3, 1 - 1e-3)
    lossf = nn.BCEWithLogitsLoss(pos_weight=(1 - pos) / pos)
    model = TextCNN(cfg["vocab"], cfg["emb"], cfg["filters"], cfg["widths"],
                    cfg["dropout"], Ytr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    g = torch.Generator().manual_seed(seed)
    best, best_state, bad = float("inf"), None, 0
    for _ in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(perm), cfg["batch"]):
            b = perm[i:i + cfg["batch"]]
            opt.zero_grad()
            lossf(model(Xtr[b]), Ytr[b]).backward()
            opt.step()
        model.eval()
        with torch.inference_mode():
            vl = float(lossf(model(Xva), Yva))
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    return scores(model, Xva, cfg["batch"]), scores(model, Xte, cfg["batch"])
