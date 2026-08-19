"""Stage III(b) with BOTH notions of marginal benefit.

  Delta_prob(x)    = p_e(y*|x) - p_s(y*|x)              continuous confidence gain
  Delta_correct(x) = 1[yhat_e = y*] - 1[yhat_s = y*]    discrete correctness flip

Same procedure for both: partial Spearman per language controlling for cheap-model
confidence, token count and fragmentation; Fisher-z meta-analysis; Holm across the
27 features. Reported jointly to test whether the two notions of "benefit from more
computation" are interchangeable."""
import numpy as np
z=np.load("results/full15/full15_arrays_3src.npz",allow_pickle=True)
lang,y,ntok,frag,F=z["lang"],z["y"],z["ntok"],z["frag"],z["F"]
FN=list(z["feat_names"]); Ps,Pe=z["P_minilm_l6"],z["P_mdeberta_base"]
i=np.arange(len(y)); LANGS=sorted(set(lang.tolist()))
DP = Pe[i,y]-Ps[i,y]
DC = (Pe.argmax(1)==y).astype(float)-(Ps.argmax(1)==y).astype(float)
conf=Ps.max(1)

def rank(v): return np.argsort(np.argsort(v)).astype(float)
def pspear(yv,xv,C_):
    m=np.isfinite(yv)&np.isfinite(xv)&np.all(np.isfinite(C_),0); n=int(m.sum())
    if n<30: return np.nan,n
    C=np.column_stack([rank(c[m]) for c in C_]+[np.ones(n)])
    r=lambda v: v-C@np.linalg.lstsq(C,v,rcond=None)[0]
    a,b=r(rank(yv[m])),r(rank(xv[m])); d=np.sqrt((a**2).sum()*(b**2).sum())
    return (float((a*b).sum()/d) if d>0 else np.nan),n
def meta(rs,ns,k=3):
    r=np.array(rs,float); N=np.array(ns,float); m=np.isfinite(r)&(N>k+4)
    if m.sum()<3: return np.nan,np.nan,(np.nan,np.nan)
    w=N[m]-k-4; zb=(w*np.arctanh(np.clip(r[m],-.999,.999))).sum()/w.sum()
    se=1/np.sqrt(w.sum())
    from math import erfc,sqrt
    return float(np.tanh(zb)), float(erfc(abs(zb/se)/sqrt(2))), (float(np.tanh(zb-1.96*se)),float(np.tanh(zb+1.96*se)))
def holm(p):
    idx=np.argsort(p); n=len(p); adj=np.empty(n); prev=0
    for k,j in enumerate(idx):
        prev=max(prev,min(1,(n-k)*p[j])); adj[j]=prev
    return adj
def within(target,v):
    rr,nn=zip(*[pspear(target[lang==l],v[lang==l],
               [conf[lang==l],ntok[lang==l],frag[lang==l]]) for l in LANGS])
    return meta(rr,nn)+(int(np.nansum(np.sign(np.array(rr,float))==np.sign(meta(rr,nn)[0]))),)

print(f"corr(Delta_prob, Delta_correct) = {np.corrcoef(DP,DC)[0,1]:+.3f}")
print(f"Delta_correct: +1 {(DC>0).mean():.1%}   -1 {(DC<0).mean():.1%}   0 {(DC==0).mean():.1%}")
print(f"eta^2(Delta_prob)={((lambda v,g: sum((g==l).sum()*(v[g==l].mean()-v.mean())**2 for l in set(g.tolist()))/((v-v.mean())**2).sum())(DP,lang)):.3f}"
      f"   eta^2(Delta_correct)={((lambda v,g: sum((g==l).sum()*(v[g==l].mean()-v.mean())**2 for l in set(g.tolist()))/((v-v.mean())**2).sum())(DC,lang)):.3f}")

rows=[]
for j,nm in enumerate(FN):
    v=F[:,j]
    rp,pp,cip,sp = within(DP,v)
    rc,pc,cic,sc = within(DC,v)
    poolp,_=pspear(DP,v,[conf,ntok,frag]); poolc,_=pspear(DC,v,[conf,ntok,frag])
    rows.append(dict(f=nm,rp=rp,pp=pp,cip=cip,sp=sp,rc=rc,pc=pc,cic=cic,sc=sc,
                     poolp=poolp,poolc=poolc))
hp=holm(np.array([r["pp"] if np.isfinite(r["pp"]) else 1 for r in rows]))
hc=holm(np.array([r["pc"] if np.isfinite(r["pc"]) else 1 for r in rows]))
for r,a,b in zip(rows,hp,hc): r["hp"],r["hc"]=a,b
rows.sort(key=lambda r:-abs(r["rp"]))
print(f"\n{'feature':30s} | {'D_prob: within [CI]':>28s} {'holm':>9s} {'sgn':>5s} | "
      f"{'D_corr: within [CI]':>28s} {'holm':>9s} {'sgn':>5s}")
for r in rows[:10]:
    print(f"{r['f']:30s} | {r['rp']:+7.3f} [{r['cip'][0]:+.3f},{r['cip'][1]:+.3f}] {r['hp']:9.1e} {r['sp']:3d}/15 | "
          f"{r['rc']:+7.3f} [{r['cic'][0]:+.3f},{r['cic'][1]:+.3f}] {r['hc']:9.1e} {r['sc']:3d}/15")
cp,_,cip2=meta(*zip(*[pspear(DP[lang==l],conf[lang==l],[ntok[lang==l],frag[lang==l]]) for l in LANGS]),k=2)[:3]
cc,_,cic2=meta(*zip(*[pspear(DC[lang==l],conf[lang==l],[ntok[lang==l],frag[lang==l]]) for l in LANGS]),k=2)[:3]
print(f"\n{'-- CONFIDENCE baseline':30s} | {cp:+7.3f} [{cip2[0]:+.3f},{cip2[1]:+.3f}] {'':>9s} {'':>5s} | "
      f"{cc:+7.3f} [{cic2[0]:+.3f},{cic2[1]:+.3f}]")
er=[r for r in rows if r["f"]=="mdeberta_base_L12_eff_rank"][0]
print(f"\nheadline feature mdeberta_base_L12_eff_rank:")
print(f"  vs Delta_prob    : within {er['rp']:+.3f} {er['cip']}  pooled {er['poolp']:+.3f}  holm {er['hp']:.1e}  sign {er['sp']}/15")
print(f"  vs Delta_correct : within {er['rc']:+.3f} {er['cic']}  pooled {er['poolc']:+.3f}  holm {er['hc']:.1e}  sign {er['sc']}/15")
ad=[r for r in rows if r["f"]=="mdeberta_base_L8_ang_disp"][0]
print(f"comparison feature mdeberta_base_L8_ang_disp:")
print(f"  vs Delta_prob    : within {ad['rp']:+.3f}  pooled {ad['poolp']:+.3f}")
print(f"  vs Delta_correct : within {ad['rc']:+.3f}  pooled {ad['poolc']:+.3f}")
