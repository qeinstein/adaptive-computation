"""Language identity is confounder #3. The pooled correlations in gate.py could be
Simpson's paradox: geometry may simply encode which language an input is, and
languages differ hugely in mean Delta. Recompute every correlation WITHIN language."""
import json, numpy as np
d = json.load(open("results/gate/gate_data.json"))
lang = np.array(d["lang"]); LANGS = ["amh","hau","ibo","yor","zul","lug"]
ntok = np.array(d["n_tokens"], float); frag = np.array(d["fragmentation"], float)

def rank(v): return np.argsort(np.argsort(v)).astype(float)
def pspear(y, x, ctrls):
    m = np.isfinite(y)&np.isfinite(x)&np.all(np.isfinite(ctrls),axis=0)
    if m.sum() < 25: return np.nan
    C = np.column_stack([rank(c[m]) for c in ctrls]+[np.ones(m.sum())])
    res = lambda v: v - C@np.linalg.lstsq(C, v, rcond=None)[0]
    ry, rx = res(rank(y[m])), res(rank(x[m]))
    den = np.sqrt((ry**2).sum()*(rx**2).sum())
    return float((ry*rx).sum()/den) if den>0 else np.nan

FEATS = [k for k in d if ("_L" in k and any(s in k for s in ("ang_disp","eff_rank","spec_conc")))]
for pair, cheap in [("delta_minilm_l6_mdeberta_base","minilm_l6"),
                    ("delta_mdeberta_base_xlmr_large","mdeberta_base")]:
    D = np.array(d[pair], float); conf = np.array(d[f"{cheap}_maxp"], float)
    print("\n"+"="*96); print(f"{pair}   (partial rho, controlling conf+len+frag)"); print("="*96)
    print(f"{'feature':30s} {'POOLED':>8s} " + " ".join(f"{l:>7s}" for l in LANGS) + f" {'mean|w/in|':>10s} {'sign_ok':>8s}")
    rows=[]
    for f in sorted(FEATS):
        v = np.array(d[f], float)
        pooled = pspear(D, v, [conf, ntok, frag])
        per = [pspear(D[lang==l], v[lang==l], [conf[lang==l], ntok[lang==l], frag[lang==l]]) for l in LANGS]
        per = np.array(per, float)
        same = int(np.nansum(np.sign(per)==np.sign(pooled)))
        rows.append((abs(np.nanmean(per)), f, pooled, per, same))
    for _, f, pooled, per, same in sorted(rows, reverse=True):
        print(f"{f:30s} {pooled:+8.3f} " + " ".join(f"{p:+7.3f}" for p in per)
              + f" {abs(np.nanmean(per)):10.3f} {same:>5d}/6")
    # baselines for reference
    print(f"{'-- confidence alone':30s} {pspear(D, conf, [ntok, frag]):+8.3f} "
          + " ".join(f"{pspear(D[lang==l], conf[lang==l], [ntok[lang==l], frag[lang==l]]):+7.3f}" for l in LANGS))
