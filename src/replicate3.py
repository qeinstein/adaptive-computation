"""Third representation source: XLM-R-base.

PRE-REGISTRATION NOTE (post-hoc, recorded honestly):
This analysis was NOT part of the original design. It was added after Finding 3
(language identity confounds pooled representation correlations) was observed on
two feature sources -- mDeBERTa-base (task-tuned) and AfroXLMR-base (African-
adapted MLM). It is a robustness replication, motivated solely by the fact that
a two-source result cannot distinguish "a general property of multilingual
representation spaces" from "a coincidence of two particular models".

Nothing else changes. Same 15 clean languages, same test examples, same Delta
target (minilm_l6 -> mdeberta_base, calibrated), same three geometry statistics
at the same layers {4,8,12}, same controls (confidence + n_tokens + fragmentation),
same partial-Spearman + Fisher-z within-language procedure, same eta^2 definition.
Only the feature-extracting model is new.

Hypothesis under test: eta^2 (between-language variance share of a feature)
predicts how badly the pooled correlation misstates the within-language one.
"""

import json, os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "cache")
OUT = os.path.join(ROOT, "results", "full15")
LAYERS = [4, 8, 12]
NEW = ("xlmr_base", "FacebookAI/xlm-roberta-base")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

import csv, urllib.request
RAW = os.path.join(ROOT, "data", "raw")
CLEAN = ["amh", "ewe", "hau", "ibo", "kin", "lin", "lug", "orm",
         "sna", "sot", "twi", "wol", "xho", "yor", "zul"]


def fetch(lang, split):
    p = os.path.join(RAW, f"{lang}_{split}.tsv")
    with open(p, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    return [r for r in rows if r.get("premise") and r.get("hypothesis") and r.get("label") in "012"]


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
def extract(tag, repo, rows, batch=32):
    cp = os.path.join(CACHE, f"geom_{tag}_test.npz")
    names = [f"{tag}_L{l}_{k}" for l in LAYERS for k in ("eff_rank", "spec_conc", "ang_disp")]
    if os.path.exists(cp):
        z = np.load(cp)
        if z["M"].shape[0] == len(rows):
            print(f"  [{tag}] cached")
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
            M[i + b] = [v for l in LAYERS for v in geometry(hs[l][b].float().cpu().numpy()[m[b]])]
        if i % (batch * 40) == 0:
            print(f"  [{tag}] {i}/{len(rows)}", flush=True)
    del model
    np.savez(cp, M=M)
    return M, names


def rank(v):
    return np.argsort(np.argsort(v)).astype(float)


def pspear(yv, xv, ctrls):
    m = np.isfinite(yv) & np.isfinite(xv) & np.all(np.isfinite(ctrls), 0)
    n = int(m.sum())
    if n < 30:
        return np.nan, n
    C = np.column_stack([rank(c[m]) for c in ctrls] + [np.ones(n)])
    r = lambda v: v - C @ np.linalg.lstsq(C, v, rcond=None)[0]
    a, b = r(rank(yv[m])), r(rank(xv[m]))
    d = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return (float((a * b).sum() / d) if d > 0 else np.nan), n


def fisher(rs, ns, k=3):
    r = np.array(rs, float); N = np.array(ns, float)
    m = np.isfinite(r) & (N > k + 4)
    if m.sum() < 3:
        return np.nan
    w = N[m] - k - 4
    return float(np.tanh((w * np.arctanh(np.clip(r[m], -.999, .999))).sum() / w.sum()))


def eta2(v, g):
    v = np.asarray(v, float); m = np.isfinite(v); v, g = v[m], g[m]
    gm = v.mean()
    ssb = sum((g == l).sum() * (v[g == l].mean() - gm) ** 2 for l in set(g.tolist()))
    return float(ssb / ((v - gm) ** 2).sum())


def main():
    z = np.load(os.path.join(OUT, "full15_arrays.npz"), allow_pickle=True)
    lang, y, ntok, frag = z["lang"], z["y"], z["ntok"], z["frag"]
    F_old, FN_old = z["F"], list(z["feat_names"])
    Ps, Pe = z["P_minilm_l6"], z["P_mdeberta_base"]
    idx = np.arange(len(y))
    D = Pe[idx, y] - Ps[idx, y]
    conf = Ps.max(1)

    rows = []
    for lg in CLEAN:
        rows += fetch(lg, "test")
    assert len(rows) == len(y), (len(rows), len(y))

    print(f"third source: {NEW[1]}")
    M, names = extract(NEW[0], NEW[1], rows)
    F = np.concatenate([F_old, M], 1)
    FN = FN_old + names

    print("\n" + "=" * 92)
    print("ETA^2 vs POOLED-CORRELATION BIAS, across THREE representation sources")
    print("=" * 92)
    print(f"{'feature':32s} {'eta2':>7s} {'pooled':>8s} {'within':>8s} {'|bias|':>8s}")
    recs = []
    for j, nm in enumerate(FN):
        v = F[:, j]
        e2 = eta2(v, lang)
        pooled, _ = pspear(D, v, [conf, ntok, frag])
        rr, nn = zip(*[pspear(D[lang == l], v[lang == l],
                              [conf[lang == l], ntok[lang == l], frag[lang == l]]) for l in CLEAN])
        w = fisher(rr, nn)
        recs.append(dict(feat=nm, source=nm.split("_L")[0], eta2=e2, pooled=pooled,
                         within=w, bias=abs(pooled) - abs(w)))
    for r in sorted(recs, key=lambda r: -r["eta2"]):
        print(f"{r['feat']:32s} {r['eta2']:7.3f} {r['pooled']:+8.3f} {r['within']:+8.3f} "
              f"{r['bias']:+8.3f}")

    e2 = np.array([r["eta2"] for r in recs])
    bias = np.array([r["bias"] for r in recs])
    rho = pspear(bias, e2, [np.ones(len(e2))])[0]
    pear = float(np.corrcoef(e2, bias)[0, 1])
    print(f"\nACROSS ALL {len(recs)} FEATURES (3 sources x 3 layers x 3 statistics):")
    print(f"  Spearman(eta^2, |pooled|-|within|) = {rho:+.3f}")
    print(f"  Pearson                            = {pear:+.3f}")
    for src in sorted({r["source"] for r in recs}):
        s = [r for r in recs if r["source"] == src]
        se2 = np.array([r["eta2"] for r in s]); sb = np.array([r["bias"] for r in s])
        print(f"  {src:16s} n={len(s):2d}  mean eta2={se2.mean():.3f}  "
              f"mean |bias|={sb.mean():+.3f}  rho={np.corrcoef(se2, sb)[0,1]:+.3f}")

    json.dump(dict(records=recs, spearman=rho, pearson=pear),
              open(os.path.join(OUT, "three_source_eta2.json"), "w"), indent=2, default=float)
    np.savez(os.path.join(OUT, "full15_arrays_3src.npz"), lang=lang, y=y, ntok=ntok, frag=frag,
             F=F, feat_names=np.array(FN), P_minilm_l6=Ps, P_mdeberta_base=Pe)
    print(f"\nwrote {OUT}/three_source_eta2.json + full15_arrays_3src.npz")


if __name__ == "__main__":
    main()
