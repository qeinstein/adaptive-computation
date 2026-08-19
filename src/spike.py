"""Feasibility spike: does Delta(x) have usable structure?

Gate for the whole project. Runs a frozen NLI cascade over a sample of AfriXNLI,
computes the seed-free continuous escalation target

    Delta_{s->e}(x) = p_e(gold|x) - p_s(gold|x)

on temperature-calibrated probabilities, and reports its distribution.
No fine-tuning. No African data used for calibration.
"""

import csv, json, math, os, sys, urllib.request

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # xet backend is flaky on this network
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "results", "spike")
os.makedirs(RAW, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# AfriXNLI label ids: 0=entailment 1=neutral 2=contradiction. This is canonical.
CANON = ["entailment", "neutral", "contradiction"]

# eng = English control, swa = comparatively higher-resource African,
# amh = low-resource + non-Latin script, yor = low-resource + tonal diacritics.
LANGS = ["eng", "swa", "amh", "yor"]
N_PER_LANG = 200

RUNGS = [
    ("minilm_l6", "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"),
    ("mdeberta_base", "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"),
    ("xlmr_large", "joeddav/xlm-roberta-large-xnli"),
]
FEATURE_MODELS = [("afroxlmr_base", "Davlan/afro-xlmr-base")]

URL = "https://huggingface.co/datasets/masakhane/afrixnli/resolve/main/data/{lang}/{split}.tsv"


def fetch(lang, split):
    path = os.path.join(RAW, f"{lang}_{split}.tsv")
    if not os.path.exists(path):
        urllib.request.urlretrieve(URL.format(lang=lang, split=split), path)
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    return [r for r in rows if r.get("premise") and r.get("hypothesis") and r.get("label") in "012"]


def perm_to_canonical(model):
    """Map a checkpoint's own logit order onto CANON. joeddav is reversed."""
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    return [next(i for i, lb in id2label.items() if lb.startswith(c[:4])) for c in CANON]


@torch.no_grad()
def run_nli(name, repo, rows, batch=16):
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForSequenceClassification.from_pretrained(repo).to(DEVICE).eval()
    perm = perm_to_canonical(model)
    print(f"  [{name}] layers={model.config.num_hidden_layers} "
          f"hidden={model.config.hidden_size} id2label={model.config.id2label} perm={perm}",
          flush=True)
    logits = []
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        enc = tok([r["premise"] for r in chunk], [r["hypothesis"] for r in chunk],
                  truncation=True, max_length=128, padding=True, return_tensors="pt").to(DEVICE)
        out = model(**enc).logits.float().cpu().numpy()
        logits.append(out[:, perm])  # reorder into canonical entail/neutral/contra
    del model
    return np.concatenate(logits, 0)


def geometry(H, mask):
    """Per-example representation statistics over the token dimension.

    H: (T, d) hidden states for one example, mask: (T,) valid-token mask.
    Returns effective rank, spectral concentration, angular dispersion.
    """
    X = H[mask.astype(bool)]
    if X.shape[0] < 2:
        return dict(eff_rank=float("nan"), spec_conc=float("nan"), ang_disp=float("nan"))
    Xc = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    p = s ** 2
    tot = p.sum()
    if tot <= 0:
        return dict(eff_rank=float("nan"), spec_conc=float("nan"), ang_disp=float("nan"))
    p = p / tot
    nz = p[p > 1e-12]
    eff_rank = float(np.exp(-(nz * np.log(nz)).sum()))       # entropy-based effective rank
    spec_conc = float(p[0])                                   # top-eigenvalue share
    U = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    ang_disp = float(1.0 - np.linalg.norm(U.mean(0)))         # 1 - resultant length
    return dict(eff_rank=eff_rank, spec_conc=spec_conc, ang_disp=ang_disp)


@torch.no_grad()
def run_features(name, repo, rows, layers, batch=16):
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModel.from_pretrained(repo).to(DEVICE).eval()
    L = model.config.num_hidden_layers
    layers = [l for l in layers if l <= L]
    print(f"  [{name}] feature model layers={L}, extracting at {layers}", flush=True)
    feats = []
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        enc = tok([r["premise"] for r in chunk], [r["hypothesis"] for r in chunk],
                  truncation=True, max_length=128, padding=True, return_tensors="pt").to(DEVICE)
        hs = model(**enc, output_hidden_states=True).hidden_states
        m = enc["attention_mask"].cpu().numpy()
        for b in range(len(chunk)):
            row = {}
            for l in layers:
                for k, v in geometry(hs[l][b].float().cpu().numpy(), m[b]).items():
                    row[f"{name}_L{l}_{k}"] = v
            feats.append(row)
    del model
    return feats


def fit_temperature(logits, labels):
    """Scalar temperature minimising NLL. Fitted on English only."""
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
    conf = probs.max(1)
    pred = probs.argmax(1)
    acc = (pred == labels).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            out += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(out)


def main():
    rng = np.random.default_rng(0)

    # --- calibration set: English dev only. No African example is ever used to fit anything.
    cal_rows = fetch("eng", "dev")
    print(f"calibration set: eng/dev n={len(cal_rows)}", flush=True)

    # --- spike sample: dev splits, test splits held pristine.
    sample, meta = [], []
    for lang in LANGS:
        rows = fetch(lang, "dev")
        idx = rng.choice(len(rows), size=min(N_PER_LANG, len(rows)), replace=False)
        for i in sorted(idx):
            sample.append(rows[i])
            meta.append(lang)
    print(f"spike sample: n={len(sample)} across {LANGS}", flush=True)

    labels = np.array([int(r["label"]) for r in sample])
    cal_labels = np.array([int(r["label"]) for r in cal_rows])

    store = {"lang": meta, "label": labels.tolist()}
    temps, probs = {}, {}

    only = os.environ.get("RUNGS")
    rungs = [(n, r) for n, r in RUNGS if not only or n in only.split(",")]
    for name, repo in rungs:
        print(f"running rung {name}", flush=True)
        cal_logits = run_nli(name, repo, cal_rows)
        T = fit_temperature(cal_logits, cal_labels)
        cal_p = softmax(cal_logits / T)
        print(f"  [{name}] T={T:.3f} eng-dev acc={float((cal_p.argmax(1)==cal_labels).mean()):.3f} "
              f"ECE_raw={ece(softmax(cal_logits), cal_labels):.3f} -> ECE_cal={ece(cal_p, cal_labels):.3f}",
              flush=True)
        lg = run_nli(name, repo, sample)
        temps[name] = T
        probs[name] = softmax(lg / T)
        probs[name + "_uncal"] = softmax(lg)
        store[f"{name}_p_gold"] = probs[name][np.arange(len(sample)), labels].tolist()
        store[f"{name}_pred"] = probs[name].argmax(1).tolist()
        store[f"{name}_maxp"] = probs[name].max(1).tolist()
        ent = -(probs[name] * np.log(probs[name] + 1e-12)).sum(1)
        store[f"{name}_entropy"] = ent.tolist()

    for name, repo in FEATURE_MODELS:
        print(f"running feature model {name}", flush=True)
        try:
            rows_f = run_features(name, repo, sample, layers=[4, 8, 12])
        except Exception as exc:
            print(f"  [{name}] SKIPPED: {type(exc).__name__}: {str(exc)[:100]}", flush=True)
            continue
        for i, row in enumerate(rows_f):
            for k, v in row.items():
                store.setdefault(k, [None] * len(sample))[i] = v

    # surface controls, tokenised with the cheap rung's tokenizer
    tok = AutoTokenizer.from_pretrained(RUNGS[0][1])
    n_tok, frag = [], []
    for r in sample:
        text = r["premise"] + " " + r["hypothesis"]
        ids = tok(r["premise"], r["hypothesis"], truncation=True, max_length=128)["input_ids"]
        n_tok.append(len(ids))
        frag.append(len(ids) / max(1, len(text.split())))
    store["n_tokens"] = n_tok
    store["fragmentation"] = frag

    # ---- the gate
    cheap, mid, exp = "minilm_l6", "mdeberta_base", "xlmr_large"
    report = {"temps": temps, "n": len(sample), "langs": LANGS}
    print("\n" + "=" * 74)
    print("GATE: distribution of Delta(x)")
    print("=" * 74)

    pairs = [(cheap, exp, "minilm->xlmr_large (distilled pair)"),
             (mid, exp, "mdeberta->xlmr_large (non-distilled)"),
             (cheap, mid, "minilm->mdeberta (both cheap)")]
    for s, e, tag in [p for p in pairs if p[0] in probs and p[1] in probs]:
        d = probs[e][np.arange(len(sample)), labels] - probs[s][np.arange(len(sample)), labels]
        r = dict(mean=float(d.mean()), sd=float(d.std()), q10=float(np.quantile(d, .1)),
                 med=float(np.median(d)), q90=float(np.quantile(d, .9)),
                 frac_pos=float((d > 0.05).mean()), frac_neg=float((d < -0.05).mean()),
                 frac_flat=float((np.abs(d) <= 0.05).mean()))
        report[f"delta::{s}->{e}"] = r
        print(f"\n{tag}")
        print(f"  mean {r['mean']:+.3f}  sd {r['sd']:.3f}  "
              f"[q10 {r['q10']:+.3f} | med {r['med']:+.3f} | q90 {r['q90']:+.3f}]")
        print(f"  helps(>+.05) {r['frac_pos']:.1%}   hurts(<-.05) {r['frac_neg']:.1%}   "
              f"flat {r['frac_flat']:.1%}")
        store[f"delta_{s}_{e}"] = d.tolist()
        by = {}
        for lang in LANGS:
            m = np.array([x == lang for x in meta])
            by[lang] = dict(mean=float(d[m].mean()), sd=float(d[m].std()),
                            frac_pos=float((d[m] > 0.05).mean()))
        report[f"delta_by_lang::{s}->{e}"] = by
        print("  by language: " + "  ".join(
            f"{l}={by[l]['mean']:+.3f}(sd {by[l]['sd']:.2f})" for l in LANGS))

    print("\naccuracy by rung and language")
    for name, _ in RUNGS:
        pred = np.array(store[f"{name}_pred"])
        accs = {l: float((pred[[i for i, m in enumerate(meta) if m == l]] ==
                          labels[[i for i, m in enumerate(meta) if m == l]]).mean()) for l in LANGS}
        report[f"acc::{name}"] = accs
        print(f"  {name:14s} " + "  ".join(f"{l}={accs[l]:.3f}" for l in LANGS))

    with open(os.path.join(OUT, "spike_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    with open(os.path.join(OUT, "spike_data.json"), "w") as fh:
        json.dump(store, fh)
    print(f"\nwrote {OUT}/spike_report.json and spike_data.json")


if __name__ == "__main__":
    main()
