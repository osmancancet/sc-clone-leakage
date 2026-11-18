"""numbers.json -> manuscript/figures/*.png (300 dpi, print).

Figure 1: MCC (mean over 10 seeds, 95% hierarchical bootstrap interval) per
model and split regime, one panel per dataset. Regimes carry both a colour
(fixed categorical order) and a marker shape, so the figure stays readable in
grayscale.
Figure 2: Slither-Audited official split, MCC and ROC-AUC of test contracts
binned by their maximum TF-IDF cosine similarity to the training part.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import RESULTS, ROOT

FIG = ROOT / "manuscript" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 8, "axes.linewidth": 0.6})

MODELS = [("knn", "5-EYK"), ("lr", "TF-IDF+LR"), ("cb", "CodeBERT+LR"), ("cnn", "TextCNN")]
REG = {"random": ("Rastgele", "#2a78d6", "o"), "cluster": ("Klona duyarlı", "#eb6834", "^"),
       "version": ("Pragma sürümü", "#1baf7a", "s"), "official": ("Resmî", "#eda100", "D"),
       "dedup": ("Tekilleştirilmiş", "#e87ba4", "v"), "external": ("Harici", "#008300", "x")}
PANELS = [("messi_re", "Messi-Q yeniden giriş", ["random", "dedup", "cluster", "external"]),
          ("messi_ts", "Messi-Q zaman damgası", ["random", "dedup", "cluster"]),
          ("slither", "Slither-Audited", ["official", "random", "version", "cluster", "external"])]


def main():
    C = json.load(open(RESULTS / "numbers.json"))["cells"]
    BT = json.load(open(RESULTS / "numbers_r1.json"))["boot"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), sharey=True)
    for ax, (ds, title, regs) in zip(axes, PANELS):
        w = 0.8 / len(regs)
        for j, rg in enumerate(regs):
            lab, col, mk = REG[rg]
            for i, (m, _) in enumerate(MODELS):
                c = C.get(f"{ds}|{rg}|{m}")
                if not c:
                    continue
                x = i - 0.4 + w * (j + 0.5)
                mu = c["mcc"]["mean"]
                b = BT.get(f"{ds}|{rg}|{m}", c["mcc"])
                ax.errorbar(x, mu, yerr=[[max(mu - b["lo"], 0)], [max(b["hi"] - mu, 0)]], fmt=mk,
                            ms=4, color=col, mfc=col if mk != "x" else None, mec=col,
                            elinewidth=0.8, capsize=1.5, label=lab if i == 0 else None)
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels([t for _, t in MODELS], rotation=30, ha="right")
        ax.set_title(title, fontsize=8)
        ax.axhline(0, color="#999999", lw=0.5)
        ax.set_ylim(-0.25, 1.05)
        ax.grid(axis="y", color="#dddddd", lw=0.4)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("MCC")
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h); labels.append(l)
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "fig_mcc_regimes.png", dpi=300)
    plt.close(fig)
    dose_figure()
    print("figures saved")


MCOL = {"knn": ("#2a78d6", "o"), "lr": ("#eb6834", "^"), "cb": ("#1baf7a", "s"), "cnn": ("#eda100", "D")}


def dose_figure():
    D = json.load(open(RESULTS / "numbers_r1.json"))["dose"]
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 2.0))
    for ax, met, yl in zip(axes, ("mcc", "roc_auc"), ("MCC", "ROC-AUC")):
        for m, lab in MODELS:
            d = D.get(f"slither|official|{m}")
            if not d:
                continue
            bins = list(d)
            ys = [d[b][met] for b in bins]
            col, mk = MCOL[m]
            ax.plot(range(len(bins)), ys, marker=mk, ms=3, lw=0.9, color=col, label=lab)
        ax.set_xticks(range(len(bins)))
        ax.set_xticklabels([b.split("-")[0].replace(".", ",") for b in bins], rotation=45, fontsize=6)
        ax.set_xlabel("En yüksek benzerlik (alt sınır)", fontsize=6)
        ax.set_ylabel(yl, fontsize=7)
        ax.grid(axis="y", color="#dddddd", lw=0.4)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=6, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(FIG / "fig_dose.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
