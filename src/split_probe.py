"""Contamination probe: accuracy on the XNLI-overlapping configurations
(eng/fra/swa) on BOTH dev and test, full splits.

Caches logits to data/cache/ so Table 4 can be re-derived without inference.
Accuracy is temperature-invariant, so no calibration is applied here.
"""
import csv, json, os, numpy as np, torch
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW, CACHE = os.path.join(ROOT, "data", "raw"), os.path.join(ROOT, "data", "cache")
OUT = os.path.join(ROOT, "results", "contamination")
for d in (RAW, CACHE, OUT): os.makedirs(d, exist_ok=True)
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
CANON = ["entailment", "neutral", "contradiction"]
MODELS = [("MiniLM-L6", "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"),
          ("mDeBERTa-base", "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"),
          ("XLM-R-large", "joeddav/xlm-roberta-large-xnli")]
LANGS, SPLITS = ["eng", "fra", "swa"], ["dev", "test"]

def fetch(l, s):
    p = os.path.join(RAW, f"{l}_{s}.tsv")
    if not os.path.exists(p):
        import urllib.request
        urllib.request.urlretrieve(
            f"https://huggingface.co/datasets/masakhane/afrixnli/resolve/main/data/{l}/{s}.tsv", p)
    return [r for r in csv.DictReader(open(p, newline="", encoding="utf-8"), delimiter="\t")
            if r.get("premise") and r.get("hypothesis") and r.get("label") in "012"]

@torch.no_grad()
def logits(tag, repo, rows, key, batch=32):
    cp = os.path.join(CACHE, f"contam_logits_{tag}_{key}.npy")
    if os.path.exists(cp):
        z = np.load(cp)
        if len(z) == len(rows): return z, True
    tk = AutoTokenizer.from_pretrained(repo)
    m = AutoModelForSequenceClassification.from_pretrained(repo).to(DEV).eval()
    i2l = {int(k): v.lower() for k, v in m.config.id2label.items()}
    perm = [next(i for i, lb in i2l.items() if lb.startswith(c[:4])) for c in CANON]
    o = []
    for i in range(0, len(rows), batch):
        c = rows[i:i + batch]
        e = tk([r["premise"] for r in c], [r["hypothesis"] for r in c], truncation=True,
               max_length=128, padding=True, return_tensors="pt").to(DEV)
        o.append(m(**e).logits.float().cpu().numpy()[:, perm])
    del m
    z = np.concatenate(o, 0); np.save(cp, z)
    return z, False

def main():
    table, cached_all = {}, True
    for tag, repo in MODELS:
        table[tag] = {}
        for l in LANGS:
            for s in SPLITS:
                rows = fetch(l, s)
                y = np.array([int(r["label"]) for r in rows])
                z, was_cached = logits(tag.replace("-", "_"), repo, rows, f"{l}_{s}")
                cached_all &= was_cached
                table[tag][f"{l}/{s}"] = dict(acc=float((z.argmax(1) == y).mean()), n=len(rows))
        print(f"{tag:14s} " + "  ".join(
            f"{l}/{s}={table[tag][f'{l}/{s}']['acc']:.3f}" for l in LANGS for s in SPLITS), flush=True)
    json.dump(table, open(os.path.join(OUT, "contamination_table.json"), "w"), indent=2)
    print(f"\n(all logits {'served from cache' if cached_all else 'computed and cached'})")
    print(f"wrote {OUT}/contamination_table.json")

if __name__ == "__main__":
    main()
