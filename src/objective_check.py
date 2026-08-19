"""Is the Stage IV negative result an artifact of target misalignment?

Delta = p_e(gold) - p_s(gold) is a probability gain. The routing metric is the
correctness flip, correct_e - correct_s. Train the router on the ACTUAL routing
objective and re-run the comparison."""
import numpy as np
z=np.load("results/full15/full15_arrays_3src.npz",allow_pickle=True)
lang,y,ntok,frag,F=z["lang"],z["y"],z["ntok"],z["frag"],z["F"]
Ps,Pe=z["P_minilm_l6"],z["P_mdeberta_base"]
n=len(y); i=np.arange(n)
D   = Pe[i,y]-Ps[i,y]                      # probability-gain target (what we used)
cs  = (Ps.argmax(1)==y).astype(float); ce=(Pe.argmax(1)==y).astype(float)
FLIP= ce-cs                                # correctness-flip target (the real objective)
conf= Ps.max(1)
LANGS=sorted(set(lang.tolist()))
viable=[]
for l in LANGS:
    m=lang==l; k=(Ps[m].argmax(1)==y[m]).sum(); N=m.sum(); p=k/N
    lo=(p+1.92/(2*N)-1.96*np.sqrt(p*(1-p)/N+0.96/(N*N)))/(1+3.84/N)
    if lo>1/3: viable.append(l)
vm=np.isin(lang,viable)
print(f"targets: corr(Delta, FLIP) = {np.corrcoef(D,FLIP)[0,1]:+.3f}")
print(f"  FLIP: +1 on {(FLIP>0).mean():.1%}, -1 on {(FLIP<0).mean():.1%}, 0 on {(FLIP==0).mean():.1%}")
print(f"  corr(conf, FLIP) = {np.corrcoef(conf,FLIP)[0,1]:+.3f}   corr(conf, Delta) = {np.corrcoef(conf,D)[0,1]:+.3f}")

def ridge(X,t,lam=10.0):
    X=np.column_stack([X,np.ones(len(X))]); return np.linalg.solve(X.T@X+lam*np.eye(X.shape[1]),X.T@t)
def run(target,use_conf):
    out={b:[] for b in (0.2,0.4,0.6,0.8)}
    for held in viable:
        tr=vm&(lang!=held); te=lang==held
        Xtr=F[tr]; Xte=F[te]
        if use_conf:
            Xtr=np.column_stack([Xtr,conf[tr],ntok[tr],frag[tr]]); Xte=np.column_stack([Xte,conf[te],ntok[te],frag[te]])
        ok=np.all(np.isfinite(Xtr),1); Xtr,t=Xtr[ok],target[tr][ok]
        mu,sd=Xtr.mean(0),np.where(Xtr.std(0)>0,Xtr.std(0),1)
        w=ridge((Xtr-mu)/sd,t)
        s=np.column_stack([(np.nan_to_num(Xte)-mu)/sd,np.ones(te.sum())])@w
        for b in out:
            k=int(round(b*te.sum())); sel=np.zeros(te.sum(),bool); sel[np.argsort(-s)[:k]]=True
            out[b].append(float(np.where(sel,ce[te],cs[te]).mean()))
    return {b:np.mean(v) for b,v in out.items()}
def baseline(kind):
    out={b:[] for b in (0.2,0.4,0.6,0.8)}
    rng=np.random.default_rng(0)
    for held in viable:
        te=lang==held
        for b in out:
            k=int(round(b*te.sum())); sel=np.zeros(te.sum(),bool)
            if kind=="conf": sel[np.argsort(conf[te])[:k]]=True
            elif kind=="oracle": sel[np.argsort(-FLIP[te])[:k]]=True
            else: sel[rng.choice(te.sum(),k,replace=False)]=True
            out[b].append(float(np.where(sel,ce[te],cs[te]).mean()))
    return {b:np.mean(v) for b,v in out.items()}

gD=run(D,False); gF=run(FLIP,False); gFc=run(FLIP,True); cB=baseline("conf"); rB=baseline("rand"); oB=baseline("oracle")
print(f"\n{'budget':>7s} {'random':>7s} {'conf':>7s} {'geom(D)':>8s} {'geom(FLIP)':>11s} {'geom+conf(FLIP)':>16s} {'oracle':>7s}")
for b in (0.2,0.4,0.6,0.8):
    print(f"{b:7.0%} {rB[b]:7.3f} {cB[b]:7.3f} {gD[b]:8.3f} {gF[b]:11.3f} {gFc[b]:16.3f} {oB[b]:7.3f}")
print("\ndoes retargeting close the gap to confidence?")
for b in (0.2,0.4,0.6,0.8):
    print(f"  budget {b:.0%}: conf-geom(D) = {cB[b]-gD[b]:+.4f}   conf-geom(FLIP) = {cB[b]-gF[b]:+.4f}")

print("\n"+"="*84)
print("STABILITY OF THE RETARGETED COMPARISON (target = correctness flip)")
print("="*84)
def gaps(target,use_conf,lam):
    out={b:[] for b in (0.2,0.4,0.6,0.8)}
    for held in viable:
        tr=vm&(lang!=held); te=lang==held
        Xtr=F[tr]; Xte=F[te]
        if use_conf:
            Xtr=np.column_stack([Xtr,conf[tr],ntok[tr],frag[tr]]); Xte=np.column_stack([Xte,conf[te],ntok[te],frag[te]])
        ok=np.all(np.isfinite(Xtr),1); Xtr,t=Xtr[ok],target[tr][ok]
        mu,sd=Xtr.mean(0),np.where(Xtr.std(0)>0,Xtr.std(0),1)
        w=ridge((Xtr-mu)/sd,t,lam)
        s=np.column_stack([(np.nan_to_num(Xte)-mu)/sd,np.ones(te.sum())])@w
        for b in out:
            k=int(round(b*te.sum()))
            g=np.zeros(te.sum(),bool); g[np.argsort(-s)[:k]]=True
            c=np.zeros(te.sum(),bool); c[np.argsort(conf[te])[:k]]=True
            out[b].append(float(np.where(c,ce[te],cs[te]).mean()-np.where(g,ce[te],cs[te]).mean()))
    return out
for lam in (1.0,10.0,100.0,1000.0):
    g=gaps(FLIP,False,lam)
    print(f"  lambda={lam:6g}  " + "  ".join(
        f"{b:.0%}: {np.mean(g[b]):+.4f}+/-{np.std(g[b]):.4f} (conf wins {int((np.array(g[b])>0).sum())}/11)"
        for b in (0.2,0.6)))
print("\n  geometry+confidence vs confidence alone (lambda=10):")
g=gaps(FLIP,True,10.0)
for b in (0.2,0.4,0.6,0.8):
    a=np.array(g[b])
    print(f"    budget {b:.0%}: conf - (geom+conf) = {a.mean():+.4f} +/- {a.std():.4f}   "
          f"combined wins {int((a<0).sum())}/11 langs")
