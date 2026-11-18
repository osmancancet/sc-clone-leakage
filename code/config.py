"""Experiment settings. Fixed on 2026-11-14 before any model was trained."""

POOL_SEED = 2026          # draws the Slither modelling pool
SLITHER_POOL = 24000      # contracts sampled from the 106,474 big-split rows
SLITHER_LABELS = ["y_access_control", "y_arithmetic", "y_other",
                  "y_reentrancy", "y_unchecked_calls"]
# equal sizes in every split regime (rows drawn per seed)
N_TRAIN, N_VAL, N_TEST = 7000, 1500, 3000
# version regime: Solidity 0.x minor of the first pragma
VERSION_TRAIN, VERSION_VAL, VERSION_TEST = (4, 5), (6,), (7, 8)

MESSI_TEST_FRAC, MESSI_VAL_FRAC = 0.2, 0.1   # 80/20, val carved from train

TFIDF = dict(token_pattern=None, lowercase=False, ngram_range=(1, 2),
             max_features=50000, min_df=2, sublinear_tf=True, dtype=__import__("numpy").float32)
LR = dict(C=4.0, class_weight="balanced", max_iter=2000, solver="liblinear")
KNN_K = 5
CNN = dict(max_len=1024, vocab=30000, emb=64, filters=128, widths=(3, 5, 7),
           dropout=0.3, epochs=6, batch=64, lr=2e-3, weight_decay=1e-4,
           patience=2)
CODEBERT = dict(name="microsoft/codebert-base", max_len=512, side="tail")
EMB_LR = dict(C=1.0, class_weight="balanced", max_iter=3000)
PREVALENCES = (0.01, 0.001)
