# Clone leakage in smart contract vulnerability detection datasets

Code, per-run results and manuscript source for the paper
"Akıllı Sözleşme Zafiyet Tespiti Veri Setlerinde Klon Sızıntısı: Açık Kıyaslama Setlerinin Klona Duyarlı Yeniden Değerlendirilmesi"
(Clone leakage in smart contract vulnerability detection datasets: a clone-aware re-evaluation of public benchmarks).

Author: Osman Can Çetlenbik, Manisa Celal Bayar University. Licence: MIT (code and results). The raw datasets keep their own licences and are not redistributed here.

## Data (download, pinned versions)

| Source | Version | Used as |
|---|---|---|
| Messi-Q/ReChecker `train_data/` (reentrancy_1671, timestamp, reentrancy_code_snippets_2000) | commit 05d722f | training/test |
| mwritescode/slither-audited-smart-contracts (Hugging Face, `big` splits) | revision 13594107 | training/test |
| smartbugs/smartbugs-curated | commit 230e649 | external test |
| DependableSystemsLab/SolidiFI-benchmark | commit 4b0573e | audit only |

```bash
mkdir -p data/raw && cd data/raw
git clone https://github.com/Messi-Q/ReChecker Messi-Q_ReChecker && git -C Messi-Q_ReChecker checkout 05d722f
git clone https://github.com/smartbugs/smartbugs-curated smartbugs_smartbugs-curated && git -C smartbugs_smartbugs-curated checkout 230e649
git clone https://github.com/DependableSystemsLab/SolidiFI-benchmark DependableSystemsLab_SolidiFI-benchmark && git -C DependableSystemsLab_SolidiFI-benchmark checkout 4b0573e
# Slither-Audited: download data/*.csv, data/label_mappings.json and data/raw/contracts{0..8}.parquet
# from revision 13594107c7afa216cb0c126f38b8ff6548112dcf into data/raw/hf_slither/data/
```

SHA-256 digests of every raw file are in `results/data_manifest.json`.

## Run order

```bash
pip install -r requirements.txt
bash code/run_all.sh      # data -> audit -> pool -> embed -> runs -> fit -> r1 -> analyze -> tables -> figures
bash code/finalize.sh     # re-aggregate, rebuild manuscript, record code SHA-256
```

| Step | Script | Output |
|---|---|---|
| Parse, hash, normalise | `data.py` | `$SC_WORK/processed/*.parquet`, `results/data_manifest.json` |
| Clone audit (raw, L1, type-2, MinHash J ≥ 0.9) | `audit.py` | `results/audit.json`, cluster ids |
| Modelling tables | `pool.py` | Slither pool (24,000, seed 2026), Messi-Q, SmartBugs |
| Frozen CodeBERT embeddings | `embed.py` | `emb_*.npz` |
| One run = dataset × regime × model × seed | `run.py` | `results/runs/*.json` |
| Over-fitting diagnostic (same splits and seeds) | `exp_fit.py` via `fit_jobs.sh` | `results/fit/*.json` |
| Per-contract test scores for the supplementary analyses (same splits, seeds, models and thresholds) | `exp_r1.py` via `r1_jobs.sh` | `results/scores/*.npz` |
| Jaccard sensitivity, pragma buckets, label agreement, Messi-Q provenance | `audit_r1.py` | `results/audit_r1.json` |
| Statistics: NB-corrected Welch gaps, Holm, pooled operating points | `analyze.py` | `results/numbers.json` |
| Hierarchical bootstrap, contract-level metrics, similarity curve, external test detail | `analyze_r1.py` (`BOOT=2000`) | `results/numbers_r1.json` |
| Manuscript numbers and tables | `make_tables.py`, `make_figures.py` | `manuscript/macros.json`, `manuscript/tables/`, `manuscript/figures/` |
| Word manuscript (journal layout) | `manuscript/build_docx.py` | `manuscript/main.docx` |

All hyper-parameters are fixed in `code/config.py`. Seeds 0-9. CPU only (4 cores). Run times are in `results/logs/` and in each run JSON (`seconds`).

## Notes on the logs

- The main runs were interrupted once by a container restart and finished with `code/resume_runs.sh`, which skips completed runs. Its jobs write to the per-dataset `runs_*.log` files, so `results/logs/resume.log` is empty.
- The first attempt of the over-fitting jobs failed with `ModuleNotFoundError: No module named 'common'` (PYTHONPATH not set on a network mount). Those tracebacks are in the first part of `results/logs/fit.log`, which ends with the first attempt's own `FIT_DONE` marker (line 438); the rerun follows and ends with a second `FIT_DONE`; the jobs were rerun with PYTHONPATH set, and every `results/fit/*.json` file comes from the rerun. `fit_chain.log` holds only `shell-init: getcwd` warnings from the same restart.
- TextCNN is retrained in `exp_fit.py` and `exp_r1.py`; CPU training can differ from the main runs in the third decimal. `analyze_r1.py` reports the largest per-model MCC difference (`agreement` in `numbers_r1.json`).
- During `exp_r1.py`, the 5-NN and CodeBERT score files were first written as float32. The 5-NN scores lie within 1e-8 of the validation threshold, so the rounding changed some predictions; these files were rewritten as float64. After that, 5-NN, TF-IDF + LR, the keyword rule and TextCNN reproduce the main-run MCC exactly, and CodeBERT + LR differs by at most 0.007 (`agreement` in `results/numbers_r1.json`).

