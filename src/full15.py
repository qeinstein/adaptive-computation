"""Full 15-language clean gate.

All AfriXNLI languages that do not intersect XNLI. Excluded: eng, fra, swa
(verified 99.7% verbatim overlap on eng; XNLI includes fr and sw; both cheap
rungs' model cards state XNLI/MNLI training).

Answers the question the six-language run raised: is the within-language
association of effective rank with Delta (rho ~ -0.21, 5/6 languages) genuine
and consistent, concentrated in a subset, or a small-sample artifact?

Logits and geometry are cached to .npz so re-analysis costs nothing.
"""

import csv, json, os, sys, urllib.request

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
CACHE = os.path.join(ROOT, "data", "cache")
OUT = os.path.join(ROOT, "results", "full15")
for d in (RAW, CACHE, OUT):
    os.makedirs(d, exist_ok=True)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CANON = ["entailment", "neutral", "contradiction"]
CHANCE = 1 / 3

CLEAN = ["amh", "ewe", "hau", "ibo", "kin", "lin", "lug", "orm",
         "sna", "sot", "twi", "wol", "xho", "yor", "zul"]
CONTAMINATED = ["eng", "fra", "swa"]

RUNGS = [
    ("minilm_l6", "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"),
    ("mdeberta_base", "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"),
    ("xlmr_large", "joeddav/xlm-roberta-large-xnli"),
]
FEATURE_MODELS = [
    ("mdeberta_base", "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"),
    ("afroxlmr_base", "Davlan/afro-xlmr-base"),
]
LAYERS = [4, 8, 12]
URL = "https://huggingface.co/datasets/masakhane/afrixnli/resolve/main/data/{lang}/{split}.tsv"


def fetch(lang, split):
    p = os.path.join(RAW, f"{lang}_{split}.tsv")
    if not os.path.exists(p):
        urllib.request.urlretrieve(URL.format(lang=lang, split=split), p)
    with open(p, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    return [r for r in rows if r.get("premise") and r.get("hypothesis") and r.get("label") in "012"]


def perm_to_canonical(cfg):
    m = {int(k): v.lower() for k, v in cfg.id2label.items()}
    return [next(i for i, lb in m.items() if lb.startswith(c[:4])) for c in CANON]


@torch.no_grad()
def nli_logits(tag, repo, rows, split, batch=32):
    cp = os.path.join(CACHE, f"logits_{tag}_{split}.npy")
    if os.path.exists(cp):
        z = np.load(cp)
        if len(z) == len(rows):
            print(f"  [{tag}/{split}] cached", flush=True)
            return z
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForSequenceClassification.from_pretrained(repo).to(DEVICE).eval()
    perm = perm_to_canonical(model.config)
    out = []
    for i in range(0, len(rows), batch):
        c = rows[i:i + batch]
        enc = tok([r["premise"] for r in c], [r["hypothesis"] for r in c], truncation=True,
                  max_length=128, padding=True, return_tensors="pt").to(DEVICE)
        out.append(model(**enc).logits.float().cpu().numpy()[:, perm])
        if i % (batch * 40) == 0:
            print(f"  [{tag}/{split}] {i}/{len(rows)}", flush=True)
    del model
    z = np.concatenate(out, 0)
    np.save(cp, z)
    return z


def geometry(X):
    if X.shape[0] < 2:
        return (np.nan, np.nan, np.nan)
    Xc = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    p = s ** 2
    if p.sum() <= 0:
        return (np.nan, np.nan, np.nan)
    p = p / p.sum()
    nz = p[p > 1e-12]
    U = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    return (float(np.exp(-(nz * np.log(nz)).sum())), float(p[0]),
            float(1.0 - np.linalg.norm(U.mean(0))))


@torch.no_grad()
def geom_feats(tag, repo, rows, batch=32):
    cp = os.path.join(CACHE, f"geom_{tag}_test.npz")
    names = [f"{tag}_L{l}_{k}" for l in LAYERS for k in ("eff_rank", "spec_conc", "ang_disp")]
    if os.path.exists(cp):
        z = np.load(cp)
        if z["M"].shape[0] == len(rows):
            print(f"  [{tag}] geometry cached", flush=True)
            return z["M"], names
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModel.from_pretrained(repo).to(DEVICE).eval()
    M = np.full((len(rows), len(names)), np.nan)
    for i in range(0, len(rows), batch):
        c = rows[i:i + batch]
        enc = tok([r["premise"] for r in c], [r["hypothesis"] for r in c], truncation=True,
                  max_length=128, padding=True, return_tensors="pt").to(DEVICE)
        hs = model(**enc, output_hidden_states=True).hidden_states
        m = enc["attention_mask"].cpu().numpy().astype(bool)
        for b in range(len(c)):
            vals = []
            for l in LAYERS:
                vals.extend(geometry(hs[l][b].float().cpu().numpy()[m[b]]))
            M[i + b] = vals
        if i % (batch * 40) == 0:
            print(f"  [{tag}] geom {i}/{len(rows)}", flush=True)
    del model
    np.savez(cp, M=M)
    return M, names


def fit_temperature(logits, labels):
    lg = torch.tensor(logits, dtype=torch.float64)
    y = torch.tensor(labels, dtype=torch.long)
    logT = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(lg / logT.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(logT.exp().item())


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def rank(v):
    return np.argsort(np.argsort(v)).astype(float)


def partial_spearman(y, x, ctrls):
    m = np.isfinite(y) & np.isfinite(x) & np.all(np.isfinite(ctrls), axis=0)
    n = int(m.sum())
    if n < 30:
        return np.nan, n
    C = np.column_stack([rank(c[m]) for c in ctrls] + [np.ones(n)])
    res = lambda v: v - C @ np.linalg.lstsq(C, v, rcond=None)[0]
    ry, rx = res(rank(y[m])), res(rank(x[m]))
    den = np.sqrt((ry ** 2).sum() * (rx ** 2).sum())
    return (float((ry * rx).sum() / den) if den > 0 else np.nan), n


def fisher_meta(rhos, ns, k_ctrl=3):
    """Random-effects-free fixed-effect meta-analysis of Spearman rhos via Fisher z."""
    r = np.array(rhos, float)
    n = np.array(ns, float)
    m = np.isfinite(r) & (n > k_ctrl + 4)
    if m.sum() < 3:
        return dict(z=np.nan, rho=np.nan, se=np.nan, ci=[np.nan, np.nan], p=np.nan, k=int(m.sum()))
    r = np.clip(r[m], -0.999, 0.999)
    w = n[m] - k_ctrl - 4
    z = np.arctanh(r)
    zbar = float((w * z).sum() / w.sum())
    se = float(1 / np.sqrt(w.sum()))
    lo, hi = zbar - 1.96 * se, zbar + 1.96 * se
    from math import erfc, sqrt
    p = float(erfc(abs(zbar / se) / sqrt(2)))
    return dict(z=zbar, rho=float(np.tanh(zbar)), se=se,
                ci=[float(np.tanh(lo)), float(np.tanh(hi))], p=p, k=int(m.sum()))


def holm(pvals):
    idx = np.argsort(pvals)
    n = len(pvals)
    adj = np.empty(n)
    prev = 0.0
    for rank_i, i in enumerate(idx):
        v = min(1.0, (n - rank_i) * pvals[i])
        prev = max(prev, v)
        adj[i] = prev
    return adj


def main():
    print(f"device={DEVICE}\nclean langs ({len(CLEAN)}): {CLEAN}\nexcluded: {CONTAMINATED}\n", flush=True)

    dev, dev_lang, test, test_lang = [], [], [], []
    for lg in CLEAN:
        d, t = fetch(lg, "dev"), fetch(lg, "test")
        dev += d; dev_lang += [lg] * len(d)
        test += t; test_lang += [lg] * len(t)
    test_lang = np.array(test_lang)
    y_dev = np.array([int(r["label"]) for r in dev])
    y = np.array([int(r["label"]) for r in test])
    print(f"calibration (dev): {len(dev)}   evaluation (test): {len(test)}\n", flush=True)

    report = {"clean_langs": CLEAN, "excluded": CONTAMINATED, "n_dev": len(dev), "n_test": len(test)}
    P = {}
    print("=" * 100)
    print("RUNG VIABILITY (chance=0.333)")
    print("=" * 100)
    for tag, repo in RUNGS:
        T = fit_temperature(nli_logits(tag, repo, dev, "dev"), y_dev)
        P[tag] = softmax(nli_logits(tag, repo, test, "test") / T)
        acc = float((P[tag].argmax(1) == y).mean())
        per = {}
        viable = []
        for lg in CLEAN:
            m = test_lang == lg
            k = int((P[tag].argmax(1)[m] == y[m]).sum())
            lo, hi = wilson(k, int(m.sum()))
            per[lg] = dict(acc=k / int(m.sum()), lo=lo, hi=hi, above_chance=bool(lo > CHANCE))
            if lo > CHANCE:
                viable.append(lg)
        report[f"rung::{tag}"] = dict(T=T, acc=acc, per_lang=per, viable=viable)
        print(f"\n{tag}  T={T:.3f}  overall acc={acc:.3f}  above-chance in {len(viable)}/{len(CLEAN)}")
        print("   " + "  ".join(f"{lg}:{per[lg]['acc']:.2f}{'' if per[lg]['above_chance'] else '*'}"
                                for lg in CLEAN) + "    (* = at chance)")

    tok = AutoTokenizer.from_pretrained(RUNGS[1][1])
    ntok = np.array([len(tok(r["premise"], r["hypothesis"], truncation=True,
                             max_length=128)["input_ids"]) for r in test], float)
    frag = ntok / np.array([max(1, len((r["premise"] + " " + r["hypothesis"]).split())) for r in test], float)

    print("\nextracting geometry", flush=True)
    F, FN = [], []
    for tag, repo in FEATURE_MODELS:
        M, names = geom_feats(tag, repo, test)
        F.append(M); FN += names
    F = np.concatenate(F, 1)

    print("\n" + "=" * 100)
    print("DELTA + WITHIN-LANGUAGE SIGNAL TEST")
    print("=" * 100)
    for s, e in [("minilm_l6", "mdeberta_base"), ("mdeberta_base", "xlmr_large")]:
        D = P[e][np.arange(len(test)), y] - P[s][np.arange(len(test)), y]
        conf = P[s].max(1)
        print(f"\n### {s} -> {e}")
        print(f"  Delta: mean {D.mean():+.3f}  sd {D.std():.3f}  "
              f"helps {(D>.05).mean():.1%}  hurts {(D<-.05).mean():.1%}  flat {(np.abs(D)<=.05).mean():.1%}")
        pooled_all, _ = partial_spearman(D, F[:, 0] * 0 + rank(np.array([CLEAN.index(l) for l in test_lang], float)),
                                         [conf, ntok, frag])
        print(f"  language-identity leakage: partial rho(Delta, lang_index) = {pooled_all:+.3f}"
              f"   (why pooled correlations mislead)")

        rows = []
        cb, nb = zip(*[partial_spearman(D[test_lang == lg], conf[test_lang == lg],
                                        [ntok[test_lang == lg], frag[test_lang == lg]]) for lg in CLEAN])
        mb = fisher_meta(cb, nb, k_ctrl=2)
        for j, name in enumerate(FN):
            v = F[:, j]
            pooled, _ = partial_spearman(D, v, [conf, ntok, frag])
            rr, nn = zip(*[partial_spearman(D[test_lang == lg], v[test_lang == lg],
                                            [conf[test_lang == lg], ntok[test_lang == lg],
                                             frag[test_lang == lg]]) for lg in CLEAN])
            meta = fisher_meta(rr, nn)
            sign_ok = int(np.nansum(np.sign(np.array(rr, float)) == np.sign(meta["rho"])))
            rows.append(dict(feat=name, pooled=pooled, meta=meta, sign_ok=sign_ok,
                             per_lang={lg: (None if not np.isfinite(x) else float(x))
                                       for lg, x in zip(CLEAN, rr)},
                             shrink=(abs(pooled) / abs(meta["rho"])) if meta["rho"] else np.nan))
        pv = np.array([r["meta"]["p"] if np.isfinite(r["meta"]["p"]) else 1.0 for r in rows])
        for r, a in zip(rows, holm(pv)):
            r["p_holm"] = float(a)
        rows.sort(key=lambda r: -abs(r["meta"]["rho"] if np.isfinite(r["meta"]["rho"]) else 0))
        report[f"signals::{s}->{e}"] = dict(rows=rows, conf_baseline=mb,
                                            delta=dict(mean=float(D.mean()), sd=float(D.std())))
        print(f"\n  {'feature':30s} {'pooled':>8s} {'within':>8s} {'95% CI':>18s} "
              f"{'p_holm':>9s} {'sign':>6s} {'infl':>6s}")
        for r in rows:
            m = r["meta"]
            print(f"  {r['feat']:30s} {r['pooled']:+8.3f} {m['rho']:+8.3f} "
                  f"[{m['ci'][0]:+.3f},{m['ci'][1]:+.3f}] {r['p_holm']:9.2e} "
                  f"{r['sign_ok']:>3d}/15 {r['shrink']:6.1f}x")
        print(f"  {'-- CONFIDENCE baseline':30s} {'':>8s} {mb['rho']:+8.3f} "
              f"[{mb['ci'][0]:+.3f},{mb['ci'][1]:+.3f}]")

    with open(os.path.join(OUT, "full15_report.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    np.savez(os.path.join(OUT, "full15_arrays.npz"), lang=test_lang, y=y, ntok=ntok, frag=frag,
             F=F, feat_names=np.array(FN), **{f"P_{k}": v for k, v in P.items()})
    print(f"\nwrote {OUT}/full15_report.json + full15_arrays.npz")


if __name__ == "__main__":
    main()
