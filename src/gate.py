"""Clean six-language feasibility gate.

Supersedes spike.py, whose design was invalidated by the contamination finding:
AfriXNLI is a translation of XNLI (99.7% verbatim overlap on the English split),
and every off-the-shelf XNLI checkpoint has trained on it. XNLI's languages
intersect AfriXNLI at {eng, fra, swa}, so those three are excluded here and the
temperature is no longer fitted on English.

Protocol:
  calibrate  -> pooled CLEAN dev  (never test)
  evaluate   -> CLEAN test
  target     -> Delta_{s->e}(x) = p_e(gold|x) - p_s(gold|x), calibrated probs
"""

import csv, json, os, urllib.request

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "results", "gate")
os.makedirs(RAW, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CANON = ["entailment", "neutral", "contradiction"]  # == AfriXNLI label ids 0,1,2
CHANCE = 1 / 3

# Clean = AfriXNLI languages absent from XNLI. Excluded: eng, fra, swa (contaminated).
LANGS = ["amh", "hau", "ibo", "yor", "zul", "lug"]
N_CAL, N_EVAL = 150, 300

RUNGS = [
    ("minilm_l6", "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"),
    ("mdeberta_base", "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"),
    ("xlmr_large", "joeddav/xlm-roberta-large-xnli"),
]
FEATURE_MODELS = [
    ("mdeberta_base", "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"),
    ("afroxlmr_base", "Davlan/afro-xlmr-base"),
]
URL = "https://huggingface.co/datasets/masakhane/afrixnli/resolve/main/data/{lang}/{split}.tsv"


def fetch(lang, split):
    path = os.path.join(RAW, f"{lang}_{split}.tsv")
    if not os.path.exists(path):
        urllib.request.urlretrieve(URL.format(lang=lang, split=split), path)
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    return [r for r in rows if r.get("premise") and r.get("hypothesis") and r.get("label") in "012"]


def perm_to_canonical(cfg):
    id2label = {int(k): v.lower() for k, v in cfg.id2label.items()}
    return [next(i for i, lb in id2label.items() if lb.startswith(c[:4])) for c in CANON]


@torch.no_grad()
def nli_logits(repo, rows, batch=32):
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForSequenceClassification.from_pretrained(repo).to(DEVICE).eval()
    perm = perm_to_canonical(model.config)
    out = []
    for i in range(0, len(rows), batch):
        c = rows[i:i + batch]
        enc = tok([r["premise"] for r in c], [r["hypothesis"] for r in c], truncation=True,
                  max_length=128, padding=True, return_tensors="pt").to(DEVICE)
        out.append(model(**enc).logits.float().cpu().numpy()[:, perm])
    info = dict(layers=model.config.num_hidden_layers, hidden=model.config.hidden_size, perm=perm)
    del model
    return np.concatenate(out, 0), info


def geometry(X):
    if X.shape[0] < 2:
        return dict(eff_rank=np.nan, spec_conc=np.nan, ang_disp=np.nan)
    Xc = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    p = s ** 2
    if p.sum() <= 0:
        return dict(eff_rank=np.nan, spec_conc=np.nan, ang_disp=np.nan)
    p = p / p.sum()
    nz = p[p > 1e-12]
    U = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    return dict(eff_rank=float(np.exp(-(nz * np.log(nz)).sum())),
                spec_conc=float(p[0]),
                ang_disp=float(1.0 - np.linalg.norm(U.mean(0))))


@torch.no_grad()
def geom_feats(tag, repo, rows, layers, batch=32):
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModel.from_pretrained(repo).to(DEVICE).eval()
    L = model.config.num_hidden_layers
    layers = [l for l in layers if l <= L]
    feats = []
    for i in range(0, len(rows), batch):
        c = rows[i:i + batch]
        enc = tok([r["premise"] for r in c], [r["hypothesis"] for r in c], truncation=True,
                  max_length=128, padding=True, return_tensors="pt").to(DEVICE)
        hs = model(**enc, output_hidden_states=True).hidden_states
        m = enc["attention_mask"].cpu().numpy().astype(bool)
        for b in range(len(c)):
            row = {}
            for l in layers:
                H = hs[l][b].float().cpu().numpy()[m[b]]
                for k, v in geometry(H).items():
                    row[f"{tag}_L{l}_{k}"] = v
            feats.append(row)
    del model
    return feats, layers


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


def ece(probs, labels, bins=10):
    conf, pred = probs.max(1), probs.argmax(1)
    acc = (pred == labels).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    return float(sum(((conf > lo) & (conf <= hi)).mean() *
                     abs(acc[(conf > lo) & (conf <= hi)].mean() - conf[(conf > lo) & (conf <= hi)].mean())
                     for lo, hi in zip(edges[:-1], edges[1:]) if ((conf > lo) & (conf <= hi)).sum()))


def spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return np.nan
    ra = np.argsort(np.argsort(a[m])).astype(float)
    rb = np.argsort(np.argsort(b[m])).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else np.nan


def partial_spearman(y, x, controls):
    """Spearman(y, x) after linearly removing `controls` from both, on ranks."""
    m = np.isfinite(y) & np.isfinite(x) & np.all(np.isfinite(controls), axis=0)
    if m.sum() < 20:
        return np.nan
    r = lambda v: np.argsort(np.argsort(v[m])).astype(float)
    C = np.column_stack([r(c) for c in controls] + [np.ones(m.sum())])
    resid = lambda v: v - C @ np.linalg.lstsq(C, v, rcond=None)[0]
    ry, rx = resid(r(y)), resid(r(x))
    d = np.sqrt((ry ** 2).sum() * (rx ** 2).sum())
    return float((ry * rx).sum() / d) if d > 0 else np.nan


def boot_ci(v, fn=np.mean, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    v = v[np.isfinite(v)]
    if len(v) < 5:
        return (np.nan, np.nan)
    s = [fn(rng.choice(v, len(v), replace=True)) for _ in range(n)]
    return float(np.quantile(s, .025)), float(np.quantile(s, .975))


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def main():
    rng = np.random.default_rng(0)
    print(f"device={DEVICE}  clean langs={LANGS}  (excluded eng/fra/swa: XNLI overlap)\n", flush=True)

    cal, cal_lang, ev, ev_lang = [], [], [], []
    for lg in LANGS:
        d = fetch(lg, "dev"); t = fetch(lg, "test")
        for i in sorted(rng.choice(len(d), min(N_CAL, len(d)), replace=False)):
            cal.append(d[i]); cal_lang.append(lg)
        for i in sorted(rng.choice(len(t), min(N_EVAL, len(t)), replace=False)):
            ev.append(t[i]); ev_lang.append(lg)
    ev_lang = np.array(ev_lang)
    y_cal = np.array([int(r["label"]) for r in cal])
    y = np.array([int(r["label"]) for r in ev])
    print(f"calibration: {len(cal)} (clean dev)   evaluation: {len(ev)} (clean test)\n", flush=True)

    report = {"langs": LANGS, "n_cal": len(cal), "n_eval": len(ev),
              "excluded": ["eng", "fra", "swa"], "overlap_note": "AfriXNLI-eng 99.7% verbatim XNLI"}
    P, store = {}, {"lang": ev_lang.tolist(), "label": y.tolist()}

    print("=" * 78)
    print("RUNG VIABILITY  (chance = 0.333; a rung at chance cannot serve as a cascade stage)")
    print("=" * 78)
    for name, repo in RUNGS:
        lc, info = nli_logits(repo, cal)
        T = fit_temperature(lc, y_cal)
        le, _ = nli_logits(repo, ev)
        P[name] = softmax(le / T)
        raw = softmax(le)
        acc = float((P[name].argmax(1) == y).mean())
        report[f"rung::{name}"] = dict(T=T, acc=acc, ece_raw=ece(raw, y), ece_cal=ece(P[name], y),
                                       **{k: int(v) if isinstance(v, (int,)) else v for k, v in info.items()})
        print(f"\n{name}  ({info['layers']}L/{info['hidden']}d)  T={T:.3f}  "
              f"ECE {ece(raw, y):.3f}->{ece(P[name], y):.3f}  overall acc={acc:.3f}")
        per = {}
        for lg in LANGS:
            m = ev_lang == lg
            k = int((P[name].argmax(1)[m] == y[m]).sum())
            lo, hi = wilson(k, int(m.sum()))
            per[lg] = dict(acc=k / int(m.sum()), lo=lo, hi=hi, above_chance=bool(lo > CHANCE))
            flag = "" if lo > CHANCE else "  <-- AT CHANCE"
            print(f"    {lg}: {k/int(m.sum()):.3f} [{lo:.3f},{hi:.3f}]{flag}")
        report[f"acc_by_lang::{name}"] = per
        store[f"{name}_pgold"] = P[name][np.arange(len(ev)), y].tolist()
        store[f"{name}_maxp"] = P[name].max(1).tolist()

    # surface controls
    tok = AutoTokenizer.from_pretrained(RUNGS[1][1])
    n_tok, frag = [], []
    for r in ev:
        ids = tok(r["premise"], r["hypothesis"], truncation=True, max_length=128)["input_ids"]
        n_tok.append(len(ids))
        frag.append(len(ids) / max(1, len((r["premise"] + " " + r["hypothesis"]).split())))
    n_tok = np.array(n_tok, float); frag = np.array(frag, float)
    store["n_tokens"] = n_tok.tolist(); store["fragmentation"] = frag.tolist()

    # geometry
    geo = {}
    for tag, repo in FEATURE_MODELS:
        print(f"\nextracting geometry: {tag}", flush=True)
        feats, used = geom_feats(tag, repo, ev, layers=[4, 8, 12])
        for k in feats[0]:
            geo[k] = np.array([f[k] for f in feats], float)
            store[k] = geo[k].tolist()
        print(f"  layers {used}", flush=True)

    print("\n" + "=" * 78)
    print("GATE: Delta distribution on CLEAN test")
    print("=" * 78)
    pairs = [("minilm_l6", "xlmr_large"), ("mdeberta_base", "xlmr_large"), ("minilm_l6", "mdeberta_base")]
    for s, e in pairs:
        d = P[e][np.arange(len(ev)), y] - P[s][np.arange(len(ev)), y]
        store[f"delta_{s}_{e}"] = d.tolist()
        lo, hi = boot_ci(d)
        print(f"\n{s} -> {e}")
        print(f"  mean {d.mean():+.3f} (95% CI [{lo:+.3f},{hi:+.3f}])  sd {d.std():.3f}  "
              f"[q10 {np.quantile(d,.1):+.3f} | med {np.median(d):+.3f} | q90 {np.quantile(d,.9):+.3f}]")
        print(f"  helps(>+.05) {(d>.05).mean():.1%}   hurts(<-.05) {(d<-.05).mean():.1%}   "
              f"flat {(np.abs(d)<=.05).mean():.1%}")
        print("  by lang: " + "  ".join(f"{lg}={d[ev_lang==lg].mean():+.3f}" for lg in LANGS))
        report[f"delta::{s}->{e}"] = dict(
            mean=float(d.mean()), ci=[lo, hi], sd=float(d.std()),
            frac_pos=float((d > .05).mean()), frac_neg=float((d < -.05).mean()),
            frac_flat=float((np.abs(d) <= .05).mean()),
            by_lang={lg: float(d[ev_lang == lg].mean()) for lg in LANGS})

        # confounder screen on the principal pair
        conf = P[s].max(1)
        ctrl = [conf, n_tok, frag]
        print(f"  corr(D, cheap_conf)={spearman(d, conf):+.3f}   "
              f"corr(D, n_tokens)={spearman(d, n_tok):+.3f}   corr(D, frag)={spearman(d, frag):+.3f}")
        rows = []
        for k, v in sorted(geo.items()):
            rows.append((k, spearman(d, v), spearman(v, n_tok), partial_spearman(d, v, ctrl)))
        report[f"signals::{s}->{e}"] = [dict(feat=k, rho=a, rho_len=b, partial=c) for k, a, b, c in rows]
        print(f"  {'feature':32s} {'rho(D,f)':>9s} {'rho(f,len)':>11s} {'partial':>9s}")
        for k, a, b, c in rows:
            print(f"  {k:32s} {a:+9.3f} {b:+11.3f} {c:+9.3f}")

    with open(os.path.join(OUT, "gate_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    with open(os.path.join(OUT, "gate_data.json"), "w") as fh:
        json.dump(store, fh)
    print(f"\nwrote {OUT}/gate_report.json, gate_data.json")


if __name__ == "__main__":
    main()
