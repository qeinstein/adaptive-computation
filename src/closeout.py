"""Closes two ledger open items:
(1) eng/fra/swa accuracy on the TEST split under the full15 protocol, so the
    contamination contrast is like-for-like with the 9,000-example clean figure.
(2) Paired bootstrap on the mDeBERTa - XLM-R-large accuracy difference."""
import csv, os, numpy as np, torch
os.environ.setdefault("HF_HUB_DISABLE_XET","1")
from transformers import AutoTokenizer, AutoModelForSequenceClassification
DEV="mps" if torch.backends.mps.is_available() else "cpu"
CANON=["entailment","neutral","contradiction"]
RUNGS=[("minilm_l6","MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"),
       ("mdeberta_base","MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"),
       ("xlmr_large","joeddav/xlm-roberta-large-xnli")]
CONT=["eng","fra","swa"]
def fetch(l,s):
    p=f"data/raw/{l}_{s}.tsv"
    if not os.path.exists(p):
        import urllib.request; urllib.request.urlretrieve(
            f"https://huggingface.co/datasets/masakhane/afrixnli/resolve/main/data/{l}/{s}.tsv",p)
    return [r for r in csv.DictReader(open(p,newline="",encoding="utf-8"),delimiter="\t")
            if r.get("premise") and r.get("hypothesis") and r.get("label") in "012"]
@torch.no_grad()
def logits(repo,rows,b=32):
    tk=AutoTokenizer.from_pretrained(repo); m=AutoModelForSequenceClassification.from_pretrained(repo).to(DEV).eval()
    i2l={int(k):v.lower() for k,v in m.config.id2label.items()}
    perm=[next(i for i,lb in i2l.items() if lb.startswith(c[:4])) for c in CANON]
    o=[]
    for i in range(0,len(rows),b):
        c=rows[i:i+b]
        e=tk([r["premise"] for r in c],[r["hypothesis"] for r in c],truncation=True,max_length=128,
             padding=True,return_tensors="pt").to(DEV)
        o.append(m(**e).logits.float().cpu().numpy()[:,perm])
    del m; return np.concatenate(o,0)

print("="*78); print("(1) CONTAMINATED LANGUAGES ON TEST (like-for-like with clean n=9,000)"); print("="*78)
rows=[]; lg=[]
for l in CONT:
    r=fetch(l,"test"); rows+=r; lg+=[l]*len(r)
lg=np.array(lg); y=np.array([int(r["label"]) for r in rows])
print(f"n = {len(rows)}  ({', '.join(f'{l}:{(lg==l).sum()}' for l in CONT)})\n")
for tag,repo in RUNGS:
    pred=logits(repo,rows).argmax(1)
    per={l: float((pred[lg==l]==y[lg==l]).mean()) for l in CONT}
    print(f"  {tag:15s} " + "  ".join(f"{l}={per[l]:.3f}" for l in CONT))

print("\n"+"="*78); print("(2) PAIRED BOOTSTRAP: mdeberta_base - xlmr_large on 15 clean languages"); print("="*78)
z=np.load("results/full15/full15_arrays.npz",allow_pickle=True)
yc=z["y"]; lang=z["lang"]
cm=(z["P_mdeberta_base"].argmax(1)==yc).astype(float)
cx=(z["P_xlmr_large"].argmax(1)==yc).astype(float)
d=cm-cx
print(f"  mDeBERTa {cm.mean():.4f}   XLM-R-large {cx.mean():.4f}   diff {d.mean():+.4f}")
rng=np.random.default_rng(0)
# cluster bootstrap: resample languages, then examples within
bs=[]
for _ in range(5000):
    ls=rng.choice(sorted(set(lang.tolist())),15,replace=True); acc=[]
    for l in ls:
        ii=np.where(lang==l)[0]; jj=rng.choice(ii,len(ii),replace=True); acc.append(d[jj].mean())
    bs.append(np.mean(acc))
bs=np.array(bs)
print(f"  cluster bootstrap (resample languages+examples, 5000x): "
      f"95% CI [{np.quantile(bs,.025):+.4f}, {np.quantile(bs,.975):+.4f}]  P(diff>0)={float((bs>0).mean()):.3f}")
bs2=np.array([d[rng.choice(len(d),len(d),replace=True)].mean() for _ in range(5000)])
print(f"  example-level bootstrap (ignores language clustering): "
      f"95% CI [{np.quantile(bs2,.025):+.4f}, {np.quantile(bs2,.975):+.4f}]")
print("\n  per-language diff (mdeberta - xlmr):")
print("   " + "  ".join(f"{l}={d[lang==l].mean():+.3f}" for l in sorted(set(lang.tolist()))))
