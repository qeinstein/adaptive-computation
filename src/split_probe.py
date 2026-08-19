"""Complete the dev-vs-test contamination contrast on full splits (no sampling).
Accuracy is temperature-invariant, so no calibration is needed here."""
import csv,os,numpy as np,torch
os.environ.setdefault("HF_HUB_DISABLE_XET","1")
from transformers import AutoTokenizer, AutoModelForSequenceClassification
DEV="mps" if torch.backends.mps.is_available() else "cpu"
CANON=["entailment","neutral","contradiction"]
M=[("MiniLM-L6","MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"),
   ("mDeBERTa-base","MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"),
   ("XLM-R-large","joeddav/xlm-roberta-large-xnli")]
def fetch(l,s):
    p=f"data/raw/{l}_{s}.tsv"
    if not os.path.exists(p):
        import urllib.request; urllib.request.urlretrieve(
          f"https://huggingface.co/datasets/masakhane/afrixnli/resolve/main/data/{l}/{s}.tsv",p)
    return [r for r in csv.DictReader(open(p,newline="",encoding="utf-8"),delimiter="\t")
            if r.get("premise") and r.get("hypothesis") and r.get("label") in "012"]
@torch.no_grad()
def acc(repo,rows,b=32):
    tk=AutoTokenizer.from_pretrained(repo); m=AutoModelForSequenceClassification.from_pretrained(repo).to(DEV).eval()
    i2l={int(k):v.lower() for k,v in m.config.id2label.items()}
    perm=[next(i for i,lb in i2l.items() if lb.startswith(c[:4])) for c in CANON]
    y=np.array([int(r["label"]) for r in rows]); pr=[]
    for i in range(0,len(rows),b):
        c=rows[i:i+b]
        e=tk([r["premise"] for r in c],[r["hypothesis"] for r in c],truncation=True,max_length=128,
             padding=True,return_tensors="pt").to(DEV)
        pr.append(m(**e).logits.float().cpu().numpy()[:,perm])
    del m
    return float((np.concatenate(pr,0).argmax(1)==y).mean())
print(f"{'model':15s} " + "  ".join(f"{l}/{s}" for l in ("eng","fra","swa") for s in ("dev","test")))
for name,repo in M:
    vals=[acc(repo,fetch(l,s)) for l in ("eng","fra","swa") for s in ("dev","test")]
    print(f"{name:15s} " + "  ".join(f"{v:8.3f}" for v in vals))
