"""Confirmatory: is the negative conclusion itself a statistical accident?
(a) eff_rank effect under bootstrap, subsampling, leave-one-language-out
(b) router gap (confidence - geometry) under seeds, ridge lambda, budget"""
import numpy as np
z=np.load("results/full15/full15_arrays.npz",allow_pickle=True)
lang,y,ntok,frag,F=z["lang"],z["y"],z["ntok"],z["frag"],z["F"]
FN=list(z["feat_names"]); Ps,Pe=z["P_minilm_l6"],z["P_mdeberta_base"]
n=len(y); idx=np.arange(n); D=Pe[idx,y]-Ps[idx,y]; conf=Ps.max(1)
LANGS=sorted(set(lang.tolist()))
ER=F[:,FN.index("mdeberta_base_L12_eff_rank")]

def rank(v): return np.argsort(np.argsort(v)).astype(float)
def pspear(yv,xv,C_):
    m=np.isfinite(yv)&np.isfinite(xv)&np.all(np.isfinite(C_),0)
    if m.sum()<30: return np.nan,int(m.sum())
    C=np.column_stack([rank(c[m]) for c in C_]+[np.ones(m.sum())])
    r=lambda v: v-C@np.linalg.lstsq(C,v,rcond=None)[0]
    a,b=r(rank(yv[m])),r(rank(xv[m])); d=np.sqrt((a**2).sum()*(b**2).sum())
    return (float((a*b).sum()/d) if d>0 else np.nan),int(m.sum())
def meta(rs,ns,k=3):
    r=np.array(rs,float); N=np.array(ns,float); m=np.isfinite(r)&(N>k+4)
    if m.sum()<3: return np.nan
    w=N[m]-k-4; return float(np.tanh((w*np.arctanh(np.clip(r[m],-.999,.999))).sum()/w.sum()))
def within(sel_mask=None, langs=LANGS, D_=None, ER_=None):
    D2=D if D_ is None else D_; E2=ER if ER_ is None else ER_
    rs,ns=[],[]
    for l in langs:
        m=(lang==l) if sel_mask is None else ((lang==l)&sel_mask)
        r,nn=pspear(D2[m],E2[m],[conf[m],ntok[m],frag[m]]); rs.append(r); ns.append(nn)
    return meta(rs,ns)

print("="*78); print("(a) STABILITY OF mdeberta_L12_eff_rank  (full-data within-lang rho = -0.127)"); print("="*78)
rng=np.random.default_rng(0)
bs=[]
for _ in range(300):
    sel=np.zeros(n,bool)
    for l in LANGS:
        ii=np.where(lang==l)[0]; sel[rng.choice(ii,len(ii),replace=True)]=True
    bs.append(within(sel))
bs=np.array(bs,float); print(f"  bootstrap (300x, within-language resample): mean {np.nanmean(bs):+.4f}  "
      f"95% [{np.nanquantile(bs,.025):+.4f},{np.nanquantile(bs,.975):+.4f}]  all-negative: {np.all(bs<0)}")
for frac in [0.25,0.5]:
    sub=[]
    for s in range(40):
        r2=np.random.default_rng(100+s); sel=np.zeros(n,bool)
        for l in LANGS:
            ii=np.where(lang==l)[0]; sel[r2.choice(ii,int(frac*len(ii)),replace=False)]=True
        sub.append(within(sel))
    sub=np.array(sub,float); print(f"  subsample {frac:.0%} (40 seeds): mean {np.nanmean(sub):+.4f} sd {np.nanstd(sub):.4f}  all-negative: {np.all(sub<0)}")
loo=[within(langs=[l for l in LANGS if l!=h]) for h in LANGS]
print(f"  leave-one-language-out: min {np.nanmin(loo):+.4f}  max {np.nanmax(loo):+.4f}  all-negative: {np.all(np.array(loo)<0)}")
print(f"  label-permutation null (20x): ", end="")
nulls=[within(D_=np.random.default_rng(s).permutation(D)) for s in range(20)]
print(f"mean {np.nanmean(nulls):+.4f}  max|rho| {np.nanmax(np.abs(nulls)):.4f}")

print("\n"+"="*78); print("(b) STABILITY OF ROUTER GAP  (confidence - geometry), leave-one-language-out"); print("="*78)
viable=[]
for l in LANGS:
    m=lang==l; k=(Ps[m].argmax(1)==y[m]).sum(); N=m.sum(); p=k/N
    lo=(p+1.92/(2*N)-1.96*np.sqrt(p*(1-p)/N+0.96/(N*N)))/(1+3.84/N)
    if lo>1/3: viable.append(l)
vm=np.isin(lang,viable); cs=(Ps.argmax(1)==y).astype(float); ce=(Pe.argmax(1)==y).astype(float)
def ridge(X,t,lam):
    X=np.column_stack([X,np.ones(len(X))]); return np.linalg.solve(X.T@X+lam*np.eye(X.shape[1]),X.T@t)
for lam in [1.0,10.0,100.0,1000.0]:
    print(f"\n  ridge lambda={lam:g}")
    for b in [0.2,0.4,0.6,0.8]:
        gaps=[]
        for held in viable:
            tr=vm&(lang!=held); te=lang==held
            Xtr=F[tr]; ok=np.all(np.isfinite(Xtr),1); Xtr,t=Xtr[ok],D[tr][ok]
            mu,sd=Xtr.mean(0),np.where(Xtr.std(0)>0,Xtr.std(0),1)
            w=ridge((Xtr-mu)/sd,t,lam)
            s=np.column_stack([(np.nan_to_num(F[te])-mu)/sd,np.ones(te.sum())])@w
            k=int(round(b*te.sum()))
            g=np.zeros(te.sum(),bool); g[np.argsort(-s)[:k]]=True
            c=np.zeros(te.sum(),bool); c[np.argsort(conf[te])[:k]]=True
            gaps.append(float(np.where(c,ce[te],cs[te]).mean()-np.where(g,ce[te],cs[te]).mean()))
        gaps=np.array(gaps)
        print(f"    budget {b:.0%}: conf-geom = {gaps.mean():+.4f} +/- {gaps.std():.4f}  "
              f"conf wins {int((gaps>0).sum())}/{len(gaps)} langs")
