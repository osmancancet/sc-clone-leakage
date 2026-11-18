"""numbers.json + audit.json -> manuscript/macros.json and manuscript/tables/*.json.

Every number printed in the manuscript comes from here. Decimal separator is a
comma (journal rule). Nothing is typed by hand.
"""
import json
import re
import numpy as np

from common import RESULTS, ROOT

MAN = ROOT / "manuscript"
TAB = MAN / "tables"
TAB.mkdir(parents=True, exist_ok=True)

MODEL_TR = {"rule": "Anahtar kelime kuralı", "knn": "5-EYK (bellek)", "lr": "TF-IDF + LR",
            "cb": "CodeBERT + LR", "cnn": "TextCNN"}
REGIME_TR = {"random": "Rastgele", "dedup": "Tekilleştirilmiş rastgele", "cluster": "Klona duyarlı",
             "official": "Resmî bölme", "version": "Pragma sürümü", "external": "Harici (SmartBugs)"}
DS_EN = {"messi_re": "Messi-Q reentrancy", "messi_ts": "Messi-Q timestamp", "slither": "Slither-Audited"}
DS_TR = {"messi_re": "Messi-Q yeniden giriş", "messi_ts": "Messi-Q zaman damgası",
         "messi_resnip": "Messi-Q kod parçası", "slither": "Slither-Audited",
         "smartbugs": "SmartBugs-Curated", "solidifi": "SolidiFI"}


def f(x, d=3):
    if x is None:
        return "-"
    return f"{x:.{d}f}".replace(".", ",").replace("-", "−")


def pct(x, d=1):
    return "%" + f(100 * x, d) if x is not None else "-"


def n(x):
    return f"{int(round(x)):,}".replace(",", ".")


def pval(p):
    if p is None:
        return "-"
    return "<0,001" if p < 0.001 else f(p, 3)


def ci_txt(c, d=3):
    if not c:
        return "-"
    return f"{f(c['mean'], d)} [{f(c['lo'], d)}; {f(c['hi'], d)}]"


def main():
    A = json.load(open(RESULTS / "audit.json"))
    N = json.load(open(RESULTS / "numbers.json"))
    M = json.load(open(RESULTS / "data_manifest.json"))
    R = json.load(open(RESULTS / "numbers_r1.json"))
    AR = json.load(open(RESULTS / "audit_r1.json"))
    mac = {}

    # ------------------------------------------------ audit macros + table
    rows = []
    for ds in ("messi_re", "messi_ts", "messi_resnip", "slither", "smartbugs", "solidifi"):
        a = A[ds]
        key = ds.replace("_", "")
        mac[f"{key}N"] = n(a["n"])
        mac[f"{key}UniqLone"] = n(a["h_l1"]["unique"])
        mac[f"{key}UniqLtwo"] = n(a["h_l2"]["unique"])
        mac[f"{key}Clusters"] = n(a["cluster"]["unique"])
        mac[f"{key}UniqLoneFrac"] = pct(a["h_l1"]["unique_frac"])
        mac[f"{key}MultiCluster"] = pct(a["cluster"]["rows_in_multi"])
        mac[f"{key}Largest"] = n(a["cluster"]["largest"])
        mac[f"{key}LargestFrac"] = pct(a["cluster"]["largest_frac"])
        mac[f"{key}Conflict"] = pct(a["cluster"]["conflict_groups_frac"])
        mac[f"{key}ConflictLone"] = pct(a["h_l1"]["conflict_groups_frac"])
        mac[f"{key}RandLeakLone"] = pct(a["h_l1"]["random_split_test_in_train_mean"])
        mac[f"{key}RandLeakCl"] = pct(a["cluster"]["random_split_test_in_train_mean"])
        if "n_files" in a:
            mac[f"{key}Files"] = n(a["n_files"])
        for lab, v in a.get("unique_pos_l1", {}).items():
            mac[f"{key}UniqPos{lab.replace('y_', '').replace('_', '').capitalize()}"] = n(v)
        rows.append([DS_TR[ds], n(a["n"]), pct(a["h_raw"]["unique_frac"]), pct(a["h_l1"]["unique_frac"]),
                     pct(a["h_l2"]["unique_frac"]), n(a["cluster"]["unique"]), pct(a["cluster"]["rows_in_multi"]),
                     n(a["cluster"]["largest"]), pct(a["cluster"]["conflict_groups_frac"]),
                     pct(a["h_l1"]["random_split_test_in_train_mean"]),
                     pct(a["cluster"]["random_split_test_in_train_mean"])])
    s = A["slither"]
    for sp in ("val", "test"):
        for lv, tag in (("h_raw", "Raw"), ("h_l1", "Lone"), ("h_l2", "Ltwo"), ("cluster", "Cl")):
            mac[f"slitherOff{sp.capitalize()}{tag}"] = pct(s[f"official_{sp}_in_train_{lv}"])
        mac[f"slitherOff{sp.capitalize()}N"] = n(s[f"official_{sp}_n"])
    mac["slitherTokMedian"] = n(s["tokens_median"])
    json.dump({"caption_tr": "Veri setlerinde tekrar ve klon denetimi",
               "caption_en": "Duplicate and clone audit of the datasets",
               "wide": True, "font": 7,
               "header": ["Veri seti", "N", "Tekil ham", "Tekil L1", "Tekil L2", "Yakın kopya kümesi",
                          "Kümelenmiş satır", "En büyük küme", "Etiket çatışması",
                          "Test∈eğitim (L1)", "Test∈eğitim (küme)"],
               "rows": rows,
               "note": "L1: yorum ve boşluktan arındırılmış belirteç dizisi; L2: tanımlayıcı ve sabitleri "
                       "soyutlanmış (tip-2) dizi; küme: L2 5-gramları üzerinde MinHash Jaccard ≥ 0,9. "
                       "Etiket çatışması: üyeleri farklı etiket taşıyan çok üyeli kümelerin payı. "
                       "Son iki sütun, doğrulama payı ayrılmadan yapılan 10 tohumlu %80/%20 rastgele bölmede eğitimde eşi bulunan test satırı oranıdır. "
                       "/ L1: comment- and whitespace-free tokens; L2: type-2 normalised tokens; cluster: MinHash Jaccard ≥ 0.9; "
                       "label conflict: share of multi-member clusters with disagreeing labels; last two columns: test rows with a copy in training under a 10-seed 80/20 random split."},
              open(TAB / "audit.json", "w"), ensure_ascii=False, indent=1)

    # ------------------------------------------------ main results
    C = N["cells"]

    def cell(ds, rg, m):
        return C.get(f"{ds}|{rg}|{m}")

    for ds, regs in (("messi_re", ["random", "dedup", "cluster", "external"]),
                     ("messi_ts", ["random", "dedup", "cluster"]),
                     ("slither", ["official", "random", "version", "cluster", "external"])):
        hdr = ["Model"] + [REGIME_TR[r] for r in regs]
        rows = []
        for m in ("rule", "knn", "lr", "cb", "cnn"):
            row = [MODEL_TR[m]]
            for rg in regs:
                c = cell(ds, rg, m)
                b = R["boot"].get(f"{ds}|{rg}|{m}")
                if not c:
                    row.append("-")
                elif c.get("constant"):
                    row.append("t.d. (sabit tahmin)")
                elif b:
                    row.append(f"{f(c['mcc']['mean'])} [{f(b['lo'])}; {f(b['hi'])}]")
                else:
                    row.append(ci_txt(c["mcc"]) + "†")
                if c:
                    k = f"{ds.replace('_', '')}{rg.capitalize()}{m.capitalize()}"
                    mac[k + "Mcc"] = f(c["mcc"]["mean"])
                    mac[k + "MccLo"] = f(b["lo"]) if b else f(c["mcc"]["lo"])
                    mac[k + "MccHi"] = f(b["hi"]) if b else f(c["mcc"]["hi"])
                    for met in ("f1", "pr_auc", "fpr", "roc_auc"):
                        if c.get(met):
                            mac[k + met.replace("_", "").capitalize()] = f(c[met]["mean"])
                    op = c["op"]
                    mac[k + "OpTpr"] = f(op["tpr"]); mac[k + "OpFpr"] = f(op["fpr"]); mac[k + "OpFprHi"] = f(op["fpr_hi"])
                    mac[k + "PrecOne"] = f(op["prec_0.01"]); mac[k + "PrecOneLo"] = f(op["prec_0.01_lo"])
                    mac[k + "PrecTenth"] = f(op["prec_0.001"], 4); mac[k + "PrecTenthLo"] = f(op["prec_0.001_lo"], 4)
                    mac[k + "Degenerate"] = n(op["degenerate_seeds"])
                    if c.get("leak_cluster"):
                        mac[k + "LeakCl"] = pct(c["leak_cluster"]["mean"])
                    mac[k + "LeakLone"] = pct(c["leak_l1"]["mean"])
                    mac[k + "Ntrain"] = n(c["n_train"]); mac[k + "Ntest"] = n(c["n_test"])
                    mac[k + "Sec"] = f(c["seconds"]["mean"], 1)
                    o = R["oracle"].get(f"{ds}|{rg}|{m}")
                    if o:
                        mac[k + "OrTpr"] = f(o["tpr_at_fpr1"]); mac[k + "OrTop"] = f(o["prec_top5"])
                    rs = R["residual_sim"].get(f"{ds}|{rg}|{m}")
                    if rs:
                        mac[k + "SimNinety"] = pct(rs["0.9"]); mac[k + "SimEighty"] = pct(rs["0.8"])
            rows.append(row)
        leak_cl, leak_l1 = ["Test∈eğitim (küme)"], ["Test∈eğitim (L1)"]
        for rg in regs:
            c = cell(ds, rg, "lr")
            leak_cl.append(pct(c["leak_cluster"]["mean"]) if c and c.get("leak_cluster") else "-")
            leak_l1.append(pct(c["leak_l1"]["mean"]) if c and c.get("leak_l1") else "-")
        rows.append(leak_cl); rows.append(leak_l1)
        note = ("GA: tohumlar ve test kümesindeki yakın kopya kümeleri (harici testte sözleşmeler) üzerinde hiyerarşik bootstrap. "
                "t.d.: tüm tohumlarda sabit tahmin, MCC tanımsız. Son iki satır: eğitimde yakın kopyası (küme) ve birebir kopyası (L1) bulunan test satırı oranı (TF-IDF + LR koşuları). "
                "/ CI: hierarchical bootstrap over seeds and test near-duplicate clusters (contracts on the external test), cluster weights shared across seeds; "
                "t.d.: constant prediction in every seed, MCC undefined. Last two rows: share of test rows with a near-duplicate (cluster) or exact copy (L1) in training (TF-IDF + LR runs).")
        if ds == "slither":
            note += " Harici sütun dört etiketin makro ortalamasıdır (Tablo 12) / The external column averages four labels (Table 12)."
        if ds == "messi_re":
            note += (" Rastgele bölmede eğitimde kopyası olan test satırı oranı Tablo 3'tekinden (denetim bölmesi, doğrulama payı yok) biraz düşüktür."
                     " / The random-split leakage is slightly lower than in Table 3, whose audit split has no validation part.")
        json.dump({"caption_tr": f"{DS_TR[ds]}: bölme rejimine göre MCC (10 tohum, ortalama [%95 GA])",
                   "caption_en": f"{DS_EN[ds]}: MCC by split regime (10 seeds, mean [95% CI])",
                   "wide": True, "font": 7, "header": hdr, "rows": rows, "note": note},
                  open(TAB / f"res_{ds}.json", "w"), ensure_ascii=False, indent=1)

    # ------------------------------------------------ gaps
    rows = []
    SHOW = {("messi_re", "random", "cluster"), ("messi_ts", "random", "cluster"),
            ("slither", "official", "cluster"), ("slither", "random", "version")}
    for k, g in N["gaps"].items():
        ds, pair, m = k.split("|")
        a, b = pair.split("-")
        mk = f"gap{ds.replace('_', '')}{a.capitalize()}{b.capitalize()}{m.capitalize()}"
        mac[mk] = f(g["diff"]); mac[mk + "Lo"] = f(g["lo"]); mac[mk + "Hi"] = f(g["hi"])
        mac[mk + "P"] = pval(g["p_holm"])
        if (ds, a, b) in SHOW:
            rows.append([DS_TR[ds], f"{REGIME_TR[a]} − {REGIME_TR[b]}", MODEL_TR[m],
                         f"{f(g['diff'])} [{f(g['lo'])}; {f(g['hi'])}]", pval(g["p_holm"])])
    json.dump({"caption_tr": "Bölme rejimleri arasındaki MCC farkı (eşlenmemiş; Nadeau-Bengio düzeltilmiş Welch; Holm)",
               "caption_en": "MCC gap between split regimes (unpaired; Nadeau-Bengio corrected Welch test; Holm)",
               "wide": True, "font": 7,
               "header": ["Veri seti", "Karşılaştırma", "Model", "ΔMCC [%95 GA, NB-Welch]", "p (Holm)"],
               "rows": rows,
               "note": "Holm düzeltmesi, tekilleştirilmiş rejim ve rastgele − klona duyarlı Slither karşılaştırmaları dahil 28 testlik aile üzerinden yapılmıştır; bu satırlar depodadır. "
                       "/ Holm is applied over the full family of 28 tests, including rows kept in the repository."},
              open(TAB / "gaps.json", "w"), ensure_ascii=False, indent=1)

    rows = []
    for k, g in N["seen_unseen"].items():
        ds, rg, m = k.split("|")
        c = cell(ds, rg, m)
        mk = f"su{ds.replace('_', '')}{rg.capitalize()}{m.capitalize()}"
        mac[mk + "Seen"] = f(c["mcc_seen"]["mean"]); mac[mk + "Unseen"] = f(c["mcc_unseen"]["mean"])
        mac[mk + "NUnseen"] = n(c["n_unseen"]["mean"]); mac[mk + "NSeen"] = n(c["n_seen"]["mean"])
        b = R["seen_unseen"].get(k)
        if not b:
            continue
        mac[mk + "Diff"] = f(b["diff"]); mac[mk + "DiffLo"] = f(b["lo"]); mac[mk + "DiffHi"] = f(b["hi"])
        mac[mk + "SeenAuc"] = f(b["seen"]["roc_auc"]); mac[mk + "UnseenAuc"] = f(b["unseen"]["roc_auc"])
        mac[mk + "SeenPrev"] = pct(b["seen"]["prevalence"]); mac[mk + "UnseenPrev"] = pct(b["unseen"]["prevalence"])
        if rg == "official":
            rows.append([MODEL_TR[m], f(c["mcc_seen"]["mean"]), f(c["mcc_unseen"]["mean"]),
                         f"{f(b['diff'])} [{f(b['lo'])}; {f(b['hi'])}]",
                         f(b["seen"]["roc_auc"]), f(b["unseen"]["roc_auc"])])
    json.dump({"caption_tr": "Slither-Audited resmî bölme: eğitimde yakın kopyası olan (görülmüş) ve olmayan (görülmemiş) test sözleşmeleri",
               "caption_en": "Slither-Audited official split: test contracts with (seen) and without (unseen) a near-duplicate in training",
               "wide": True, "font": 7,
               "header": ["Model", "MCC görülmüş", "MCC görülmemiş", "Fark [%95 GA]", "ROC-AUC görülmüş", "ROC-AUC görülmemiş"],
               "rows": rows,
               "note": "GA: kümeler düzeyinde eşli hiyerarşik bootstrap (küme ağırlıkları tohumlar arasında ortak); 5-EYK ve öğrenen modellerde hiçbir bootstrap yinelemesinde fark sıfırın altına düşmemiştir. Görülmüş/görülmemiş pozitif oranı TF-IDF + LR için "
                       + mac.get("suslitherOfficialLrSeenPrev", "-") + " / " + mac.get("suslitherOfficialLrUnseenPrev", "-")
                       + ". / CI: paired hierarchical cluster bootstrap with cluster weights shared across seeds; for 5-NN and the learned models no bootstrap draw gave a difference below zero. ROC-AUC is prevalence-invariant."},
              open(TAB / "seen_unseen.json", "w"), ensure_ascii=False, indent=1)

    # ------------------------------------------------ operating point (clone-aware, pooled counts)
    rows = []
    for ds in ("messi_re", "messi_ts", "slither"):
        for m in ("knn", "lr", "cb", "cnn"):
            c = cell(ds, "cluster", m)
            if not c:
                continue
            op = c["op"]; o = R["oracle"].get(f"{ds}|cluster|{m}", {})
            sl = ds == "slither"
            one = f"{f(op['prec_0.01'])} [≥{f(op['prec_0.01_lo'])}]" if sl else f(op["prec_0.01"])
            tenth = f"{f(op['prec_0.001'], 4)} [≥{f(op['prec_0.001_lo'], 4)}]" if sl else f(op["prec_0.001"], 4)
            rows.append([DS_TR[ds], MODEL_TR[m], f(c["pr_auc"]["mean"]), f(op["tpr"]), f(op["fpr"]),
                         f(op["fpr_hi"]) if sl else "-", one, tenth,
                         f(o.get("tpr_at_fpr1")), f(o.get("prec_top5"))])
    json.dump({"caption_tr": "Klona duyarlı bölmede çalışma noktası: doğrulamada seçilen eşik ve test verisinde seçilen (kâhin) ölçütler",
               "caption_en": "Operating point, clone-aware split: validation-selected threshold and measures chosen on test data (oracle)",
               "font": 7, "wide": True,
               "header": ["Veri seti", "Model", "PR-AUC", "TPR", "FPR", "FPR üst sınır", "Kesinlik π=%1", "Kesinlik π=%0,1",
                          "TPR @ FPR=%1 (kâhin)", "Kesinlik ilk %5 (kâhin)"],
               "rows": rows,
               "note": "TPR ve FPR tohumlar üzerinde toplanan karışıklık sayılarından etiket başına hesaplanıp etiketler üzerinden ortalanmıştır; "
                       "kesinlik de etiket başına Eşitlik 1 ile hesaplanıp ortalandığı için tablodaki ortalama TPR ve FPR değerlerinden yeniden hesaplanan değerle birebir aynı değildir. "
                       "Köşeli parantez: satırları bağımsız sayan tek yönlü %95 Clopper-Pearson sınırlarıyla alt sınır; Messi-Q'da satırlar tekrar içerdiği için bu sınır verilmemiştir. "
                       "Kâhin sütunları eşiği test verisinde seçer ve Messi-Q'da tekrarlı satırlar üzerinde hesaplanmıştır; ilk %5 kesinliği test kümesinin kendi yaygınlığında (Slither yaklaşık %27) ölçülür. "
                       "/ TPR and FPR from confusion counts pooled over seeds, per label, then averaged; precision is computed per label with Eq. 1 and then averaged, "
                       "so it differs slightly from a plug-in of the averaged rates. Brackets: one-sided 95% Clopper-Pearson lower bound treating rows as independent; "
                       "omitted for Messi-Q, whose rows contain duplicates. Oracle columns choose the threshold on test data (Messi-Q: on duplicated rows); top-5% precision is at the test prevalence."},
              open(TAB / "oppoint.json", "w"), ensure_ascii=False, indent=1)

    # ------------------------------------------------ contract-level Messi-Q
    rows = []
    for ds in ("messi_re", "messi_ts"):
        for rg in ("random", "dedup", "cluster"):
            first = R["contract"].get(f"{ds}|{rg}|lr")
            if not first:
                continue
            row = [DS_TR[ds], REGIME_TR[rg], (f"{first['n_min']}" if first['n_min'] == first['n_max'] else f"{first['n_min']}-{first['n_max']}")
                   + (f" ({first['pos_min']})" if first['pos_min'] == first['pos_max'] else f" ({first['pos_min']}-{first['pos_max']})")]
            for m in ("knn", "lr", "cb", "cnn"):
                e = R["contract"].get(f"{ds}|{rg}|{m}")
                if e:
                    k = f"ct{ds.replace('_', '')}{rg.capitalize()}{m.capitalize()}"
                    mac[k + "Mcc"] = f(e["mcc"]["mean"]); mac[k + "Prauc"] = f(e["pr_auc"]["mean"])
                    mac[k + "Rocauc"] = f(e["roc_auc"]["mean"])
                    row.append(f"{f(e['mcc']['mean'])} / {f(e['roc_auc']['mean'])}")
                else:
                    row.append("-")
            k = f"ct{ds.replace('_', '')}{rg.capitalize()}"
            mac[k + "NMin"] = n(first["n_min"]); mac[k + "NMax"] = n(first["n_max"])
            mac[k + "PosMin"] = n(first["pos_min"]); mac[k + "PosMax"] = n(first["pos_max"])
            mac[k + "LargestShare"] = pct(first["largest_cluster_share"]["mean"])
            rows.append(row)
    json.dump({"caption_tr": "Messi-Q sözleşme düzeyi sonuçlar: her tekil L1 sözleşmesi bir kez sayılır (MCC / ROC-AUC, 10 tohum ortalaması)",
               "caption_en": "Messi-Q contract-level results: each distinct L1 contract counted once (MCC / ROC-AUC, mean of 10 seeds)",
               "font": 7, "wide": True,
               "header": ["Veri seti", "Rejim", "Tekil test sözleşmesi (pozitif)"] + [MODEL_TR[m] for m in ("knn", "lr", "cb", "cnn")],
               "rows": rows,
               "note": "Üçüncü sütun tohumlar arasındaki en küçük ve en büyük değerdir. / Third column: min-max over seeds."},
              open(TAB / "contract.json", "w"), ensure_ascii=False, indent=1)

    # ------------------------------------------------ external per label (Slither -> SmartBugs)
    ALIGN = {"y_reentrancy": "uyumlu", "y_unchecked_calls": "kısmen", "y_access_control": "kısmen",
             "y_arithmetic": "uyumsuz"}
    LAB_TR = {"y_reentrancy": "Yeniden giriş", "y_unchecked_calls": "Denetlenmemiş çağrı",
              "y_access_control": "Erişim denetimi", "y_arithmetic": "Aritmetik"}
    rows = []
    import glob as _g
    for l in ("y_reentrancy", "y_unchecked_calls", "y_access_control", "y_arithmetic"):
        row = [LAB_TR[l], ALIGN[l]]
        pos = None
        for m in ("rule", "lr", "cnn"):
            rs = [json.load(open(p)) for p in sorted(_g.glob(str(RESULTS / "runs" / f"slither__external__{m}__*.json")))]
            mc = sum(r["labels"][l]["all"]["mcc"] for r in rs) / len(rs)
            pr = sum(r["labels"][l]["all"]["pr_auc"] for r in rs) / len(rs)
            pos = rs[0]["labels"][l]["all"]["pos"]
            mac[f"extSl{l.replace('y_', '').replace('_', '').capitalize()}{m.capitalize()}Mcc"] = f(mc)
            mac[f"extSl{l.replace('y_', '').replace('_', '').capitalize()}{m.capitalize()}Prauc"] = f(pr)
            row.append(f"{f(mc)} / {f(pr)}")
        row.insert(2, n(pos))
        rows.append(row)
    # macro over the three labels that are at least partly aligned (arithmetic excluded)
    for m in ("rule", "lr", "cnn"):
        rs = [json.load(open(p)) for p in sorted(_g.glob(str(RESULTS / "runs" / f"slither__external__{m}__*.json")))]
        three = ("y_reentrancy", "y_unchecked_calls", "y_access_control")
        mac[f"extSlThree{m.capitalize()}Mcc"] = f(np.mean([r["labels"][l]["all"]["mcc"] for r in rs for l in three]))
        mac[f"extSlThree{m.capitalize()}Prauc"] = f(np.mean([r["labels"][l]["all"]["pr_auc"] for r in rs for l in three]))
    ex = R["external"]
    for m, e in ex.items():
        if "-" in m:
            for met, v in e.items():
                mac[f"extMessi{m.replace('-', '').capitalize()}{met.replace('_', '').capitalize()}P"] = pval(v["p"])
                mac[f"extMessi{m.replace('-', '').capitalize()}{met.replace('_', '').capitalize()}Lo"] = f(v["lo"])
                mac[f"extMessi{m.replace('-', '').capitalize()}{met.replace('_', '').capitalize()}Hi"] = f(v["hi"])
        else:
            for met in ("mcc", "pr_auc"):
                mac[f"extMessi{m.capitalize()}{met.replace('_', '').capitalize()}Lo"] = f(e[met]["lo"])
                mac[f"extMessi{m.capitalize()}{met.replace('_', '').capitalize()}Hi"] = f(e[met]["hi"])
            nl = e["no_l2_match"]
            mac[f"extMessi{m.capitalize()}NoltwoMcc"] = f(nl["mcc"]); mac[f"extMessi{m.capitalize()}NoltwoPrauc"] = f(nl["pr_auc"])
            mac["extMessiNoltwoN"] = n(nl["n"]); mac["extMessiNoltwoPos"] = n(nl["pos"])
            if "matched_rate_mcc" in e:
                mac[f"extMessi{m.capitalize()}Matched"] = f(e["matched_rate_mcc"])
    json.dump({"caption_tr": "Slither-Audited ile eğitilen modellerin SmartBugs-Curated üzerinde etiket bazında sonucu (MCC / PR-AUC, 10 tohum)",
               "caption_en": "Per-label results of Slither-Audited models on SmartBugs-Curated (MCC / PR-AUC, 10 seeds)",
               "font": 7, "wide": False,
               "header": ["Etiket", "Sınıf uyumu", "Pozitif", MODEL_TR["rule"], MODEL_TR["lr"], MODEL_TR["cnn"]],
               "rows": rows,
               "note": "Sınıf uyumu: Slither-Audited sınıfını üreten dedektörlerin SmartBugs kategorisiyle örtüşmesi; aritmetik sınıfı yalnızca bölmeden önce çarpma "
                       "dedektöründen gelir, SmartBugs'ta ise taşma anlamındadır. / Alignment between the Slither detectors behind each class and the SmartBugs category."},
              open(TAB / "external.json", "w"), ensure_ascii=False, indent=1)

    # data provenance (commit prefixes: 7 characters; Hugging Face revision: 8, as in the reference list)
    for d, v in M["sources"].items():
        mac["commit" + d.split("_")[0].replace("-", "").capitalize()] = \
            v["commit"][:7] if v.get("commit") else v["revision"][:8]
    # ------------------------------------------------ provenance
    prov = [["Messi-Q (ReChecker)", "GitHub, commit " + (M["sources"].get("Messi-Q_ReChecker", {}).get("commit") or "")[:7], "MIT",
             "Yazar örüntüleri; yazarların yazdığı örnekler", "Sözleşme", n(A["messi_re"]["n"]) + " + " + n(A["messi_ts"]["n"])],
            ["Slither-Audited", "Hugging Face, revizyon " + mac.get("commitHf", ""), "MIT (veri kartı)", "Slither çıktısı (sürüm belirtilmemiş)",
             "Düzleştirilmiş dosya", n(A["slither"]["n"])],
            ["SmartBugs-Curated", "GitHub, commit " + mac.get("commitSmartbugs", ""), "Apache-2.0", "Uzman", "Sözleşme", n(A["smartbugs"]["n"])],
            ["SolidiFI", "GitHub, commit " + mac.get("commitDependablesystemslab", ""), "MIT; sözleşmeler özgün lisanslarında",
             "Hata enjeksiyonu", "Sözleşme", n(A["solidifi"]["n"])]]
    json.dump({"caption_tr": "Veri kaynakları, lisanslar ve etiket kaynakları",
               "caption_en": "Data sources, licences and label sources",
               "font": 7, "wide": True,
               "header": ["Veri seti", "Kaynak ve sürüm", "Lisans", "Etiket kaynağı", "Birim", "N"], "rows": prov},
              open(TAB / "provenance.json", "w"), ensure_ascii=False, indent=1)

    # ------------------------------------------------ audit r1 macros
    js = AR["jaccard_sensitivity"]
    for j, e in js.items():
        t = {"0.8": "Eighty", "0.9": "Ninety", "0.95": "Ninetyfive"}[j]
        mac[f"js{t}Clusters"] = n(e["clusters"]); mac[f"js{t}OffTest"] = pct(e["test"]); mac[f"js{t}OffVal"] = pct(e["val"])
        mac[f"js{t}MessireClusters"] = n(e["messi_re_clusters"]); mac[f"js{t}MessitsClusters"] = n(e["messi_ts_clusters"])
    for v, e in AR["version_buckets"].items():
        mac[f"verBucket{v}N"] = n(e["n"])
        for l in ("y_reentrancy", "y_unchecked_calls"):
            mac[f"verBucket{v}{l.replace('y_', '').replace('_', '').capitalize()}"] = pct(e[l])
    mac["verNewOldNeighbour"] = pct(AR["new_with_old_neighbour"]); mac["verNewN"] = n(AR["new_n"])
    ag = AR["label_agreement"]
    mac["agreeShared"] = n(ag["n_shared"])
    for l, e in ag.items():
        if isinstance(e, dict):
            k = "agree" + l.replace("y_", "").replace("_", "").capitalize()
            mac[k + "SbPos"] = n(e["sb_pos"]); mac[k + "SlPos"] = n(e["slither_pos"])
            mac[k + "Both"] = n(e["both"]); mac[k + "Rate"] = pct(e["agree"])
    mac["messireProvPos"] = n(AR["messi_re_provenance"]["pos_rows"])
    for key, e in R["version"].items():
        m = key.split("|")[2]
        for v, x in e.items():
            k = f"ver{m.capitalize()}{v}"
            mac[k + "Mcc"] = f(x["mcc"]); mac[k + "Auc"] = f(x["roc_auc"]); mac[k + "N"] = n(x["n_per_seed"])
    for key, d in R["dose"].items():
        ds, rg, m = key.split("|")
        bins = list(d)
        lo, hi = d[bins[0]], d[bins[-1]]
        k = f"dose{rg.capitalize()}{m.capitalize()}"
        mac[k + "LowMcc"] = f(lo["mcc"]); mac[k + "HighMcc"] = f(hi["mcc"])
        mac[k + "LowAuc"] = f(lo["roc_auc"]); mac[k + "HighAuc"] = f(hi["roc_auc"])
        mac[k + "LowBin"] = bins[0].replace(".", ","); mac[k + "HighBin"] = bins[-1].replace(".", ",")
    ls = R.get("largest_test_cluster_share", {}).get("lr")
    if ls:
        mac["slClusterLargestMax"] = pct(max(ls)); mac["slClusterLargestMed"] = pct(sorted(ls)[len(ls) // 2])
    mac["agreeCnn"] = f(R["agreement"].get("cnn", 0)); mac["agreeLr"] = f(R["agreement"].get("lr", 0))
    mac["agreeCb"] = f(R["agreement"].get("cb", 0)); mac["agreeKnn"] = f(R["agreement"].get("knn", 0))
    fitd = [abs(N["fit"][k]["test"]["mean"] - N["cells"][k]["mcc"]["mean"]) for k in N.get("fit", {})
            if k.endswith("|cnn") and k in N["cells"] and "test" in N["fit"][k]]
    if fitd:
        mac["agreeFitCnn"] = f(max(fitd), 4)
    # dose: seen / unseen inside each similarity bin (Slither official)
    for key, bins_ in R["dose"].items():
        ds_, rg_, m_ = key.split("|")
        if rg_ != "official":
            continue
        for i, (bn, v) in enumerate(bins_.items()):
            k = f"dose{rg_.capitalize()}{m_.capitalize()}Bin{i}"
            mac[k + "Seen"] = f(v.get("mcc_seen")) if v.get("mcc_seen") is not None else "-"
            mac[k + "Unseen"] = f(v.get("mcc_unseen")) if v.get("mcc_unseen") is not None else "-"
            mac[k + "SeenShare"] = pct(v.get("seen_share", 0))
            mac[k + "Range"] = bn.replace(".", ",")
    # summary bounds used in the abstract, discussion and conclusion
    sl_leak = [N["gaps"][f"slither|official-cluster|{m}"]["diff"] for m in ("knn", "lr", "cnn", "cb")]
    mac["slLeakMin"] = f(min(sl_leak), 2); mac["slLeakMax"] = f(max(sl_leak), 2)
    C_ = N["cells"]
    mac["messiRandMin"] = f(min(C_[f"{d}|random|{m}"]["mcc"]["mean"] for d in ("messi_re", "messi_ts")
                                for m in ("lr", "cb", "cnn")))
    mac["messiClusterMax"] = f(max(C_[f"{d}|cluster|{m}"]["mcc"]["mean"] for d in ("messi_re", "messi_ts")
                                   for m in ("knn", "lr", "cb", "cnn")))
    mac["bootB"] = n(R["B"])
    rt = N["runtime"]
    mac["cpuMainH"] = f(rt["main_runs_cpu_h"], 1); mac["cpuCnnH"] = f(rt["by_model_cpu_h"]["cnn"], 1)
    emb = {l.split("\t")[0]: (int(l.split("\t")[1]), float(l.split("\t")[2].rstrip("s")))
           for l in rt["embed_note"].strip().splitlines()}
    mac["cpuEmbH"] = f(sum(v[1] for v in emb.values()) / 3600, 1)
    mac["cpuEmbN"] = n(sum(v[0] for v in emb.values()))

    rows = []
    for k, e in N.get("fit", {}).items():
        ds, rg, m = k.split("|")
        mk = f"fit{ds.replace('_', '')}{rg.capitalize()}{m.capitalize()}"
        for p in ("train", "val", "test"):
            mac[mk + p.capitalize()] = f(e[p]["mean"])
        mac[mk + "Gap"] = f(e["gap_train_test"]["mean"])
        rows.append([DS_TR[ds], REGIME_TR[rg], MODEL_TR[m], ci_txt(e["train"]), ci_txt(e["val"]),
                     ci_txt(e["test"]), ci_txt(e["gap_train_test"])])
    if rows:
        json.dump({"caption_tr": "Aşırı uyum tanısı: aynı koşuda eğitim, doğrulama ve test MCC (doğrulamada seçilen eşik)",
                   "caption_en": "Over-fitting diagnostic: train, validation and test MCC in the same run (validation-selected threshold)",
                   "wide": True, "font": 7,
                   "note": "Değerler Tablo 4, 5 ve 7 ile aynı bölme ve tohumlardan elde edilmiştir; TextCNN yeniden eğitildiği için test değeri en fazla üçüncü ondalıkta farklılık gösterebilir. Köşeli parantezler 10 tohum üzerinden %95 güven aralığıdır (t dağılımı). "
                           "/ Same splits and seeds as Tables 4, 5 and 7; TextCNN test values may differ in the third decimal because the model was retrained. Brackets: 95% t interval over 10 seeds.",
                   "header": ["Veri seti", "Rejim", "Model", "Eğitim", "Doğrulama", "Test", "Eğitim − test"],
                   "rows": rows}, open(TAB / "fit.json", "w"), ensure_ascii=False, indent=1)

    X = json.load(open(RESULTS / "audit_cross.json"))
    for k, v in X.items():
        parts = k.split("_")
        mac["x" + "".join(p.capitalize() for p in parts)] = pct(v) if isinstance(v, float) else n(v)
    mac["nRuns"] = n(N["n_runs"])
    # English-formatted twins ({key}En) for the English abstract
    for k, v in list(mac.items()):
        if isinstance(v, str):
            e = v.replace("−", "-")
            e = e.translate(str.maketrans(".,", ",."))  # TR 1.671 / 0,986 -> EN 1,671 / 0.986
            e = re.sub(r"^%(.*)$", r"\1%", e)
            mac[k + "En"] = e
    json.dump(mac, open(MAN / "macros.json", "w"), ensure_ascii=False, indent=1, sort_keys=True)
    print("macros", len(mac))


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------- journal style
# Short captions and short Turkish notes, in the style of recent issues of the
# target journal (Bilişim Teknolojileri Dergisi). Method details live in the text.
STYLE = {
 "lit": ("Literatürdeki yakın çalışmaların karşılaştırılması", "Comparison with related studies", None),
 "provenance": ("Veri kaynakları ve lisanslar", "Data sources and licences", None),
 "audit": ("Veri setlerindeki tekrar ve klon oranları", "Duplicate and clone rates in the datasets",
           "Not: L1: birebir kopya; L2: sözdizimsel (tip-2) kopya; küme: yakın kopya kümesi (MinHash Jaccard ≥ 0,9). Son iki sütun, rastgele bölmede eğitimde kopyası bulunan test örneklerinin oranıdır."),
 "res_messi_re": ("Messi-Q yeniden giriş kümesinde bölme rejimine göre MCC değerleri", "MCC values by split regime on the Messi-Q reentrancy set", "RES"),
 "res_messi_ts": ("Messi-Q zaman damgası kümesinde bölme rejimine göre MCC değerleri", "MCC values by split regime on the Messi-Q timestamp set", "RES"),
 "res_slither": ("Slither-Audited veri setinde bölme rejimine göre MCC değerleri", "MCC values by split regime on Slither-Audited", "RES+"),
 "contract": ("Messi-Q kümelerinde sözleşme düzeyinde MCC / ROC-AUC değerleri", "Contract-level MCC / ROC-AUC values on Messi-Q",
              "Not: Her tekil sözleşme bir kez sayılmıştır. Değerler MCC / ROC-AUC biçiminde, 10 tohumun ortalaması olarak verilmiştir."),
 "gaps": ("Bölme rejimleri arasındaki MCC farkları", "MCC differences between split regimes",
          "Not: p değerleri Nadeau-Bengio düzeltmeli Welch testiyle hesaplanmış ve 28 testlik aile üzerinden Holm yöntemiyle düzeltilmiştir."),
 "seen_unseen": ("Görülmüş ve görülmemiş test sözleşmelerinde başarı (Slither-Audited, resmî bölme)",
                 "Performance on seen and unseen test contracts (Slither-Audited, official split)",
                 "Not: Görülmüş sözleşme, eğitim kümesinde yakın kopyası bulunan sözleşmedir. Köşeli parantezler %95 güven aralığıdır."),
 "fit": ("Eğitim, doğrulama ve test MCC değerleri", "Training, validation and test MCC values",
         "Not: Değerler 10 tohumun ortalamasıdır; köşeli parantezler %95 güven aralığıdır."),
 "oppoint": ("Klona duyarlı bölmede çalışma noktası ölçütleri", "Operating point measures under the clone-aware split",
             "Not: Kâhin sütunlarında eşik test verisi üzerinde seçilmiştir. Köşeli parantezler %95 Clopper-Pearson alt sınırıdır."),
 "external": ("SmartBugs-Curated harici testinde etiket bazında MCC / PR-AUC değerleri", "Per-label MCC / PR-AUC values on the SmartBugs-Curated external test",
              "Not: Değerler MCC / PR-AUC biçimindedir. Sınıf uyumu, Slither-Audited etiketinin SmartBugs kategorisiyle örtüşme düzeyini gösterir."),
}
RES_NOTE = ("Not: Değerler 10 tohumun ortalamasıdır; köşeli parantezler %95 güven aralığıdır. t.d.: tanımsız (model tüm tohumlarda tek sınıf tahmin etmiştir). "
            "Son iki satır, eğitimde yakın kopyası (küme) ya da birebir kopyası (L1) bulunan test örneklerinin oranıdır.")
for name, (ctr, cen, note) in STYLE.items():
    p = TAB / f"{name}.json"
    if not p.exists():
        continue
    d = json.load(open(p))
    d["caption_tr"], d["caption_en"] = ctr, cen
    if note in ("RES", "RES+"):
        note = RES_NOTE + (" Harici sütun dört etiketin ortalamasıdır (Tablo 12)." if note == "RES+" else "")
    d.pop("note", None)  # printed articles of the journal carry no table notes
    json.dump(d, open(p, "w"), ensure_ascii=False, indent=1)
