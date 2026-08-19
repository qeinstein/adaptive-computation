"""(1) Proper eta^2 quantification of the Simpson's effect.
   (2) Does rho=-0.127 buy anything? Leave-one-language-out router at matched budget."""
import json, numpy as np
z = np.load("results/full15/full15_arrays.npz", allow_pickle=True)
lang, y, ntok, frag, F = z["lang"], z["y"], z["ntok"], z["frag"], z["F"]
FN = list(z["feat_names"]); LANGS = sorted(set(lang.tolist()))
Ps, Pe = z["P_minilm_l6"], z["P_mdeberta_base"]
n = len(y); idx = np.arange(n)
D = Pe[idx, y] - Ps[idx, y]
conf = Ps.max(1)

def eta2(v, g):
    v = np.asarray(v, float); m = np.isfinite(v); v, g = v[m], g[m]
    gm = v.mean(); ssb = sum(((g==l).sum())*((v[g==l].mean()-gm)**2) for l in set(g.tolist()))
    return float(ssb/((v-gm)**2).sum())

print("="*84); print("(1) BETWEEN-LANGUAGE VARIANCE SHARE (eta^2) -- the Simpson's driver"); print("="*84)
print(f"  Delta itself              eta^2 = {eta2(D, lang):.3f}")
print(f"  cheap-model confidence    eta^2 = {eta2(conf, lang):.3f}")
print(f"  n_tokens                  eta^2 = {eta2(ntok, lang):.3f}")
for nm in ["mdeberta_base_L8_ang_disp","mdeberta_base_L12_eff_rank","mdeberta_base_L12_spec_conc"]:
    print(f"  {nm:26s}eta^2 = {eta2(F[:,FN.index(nm)], lang):.3f}")

# viability: languages where minilm beats chance
viable=[]
for l in LANGS:
    m=lang==l; k=(Ps[m].argmax(1)==y[m]).sum(); N=m.sum(); p=k/N
    lo=(p+1.96**2/(2*N)-1.96*np.sqrt(p*(1-p)/N+1.96**2/(4*N*N)))/(1+1.96**2/N)
    if lo>1/3: viable.append(l)
print(f"\nviable ladder languages ({len(viable)}/15): {viable}")

print("\n"+"="*84)
print("(2) LEAVE-ONE-LANGUAGE-OUT ROUTER, MATCHED BUDGET (viable languages only)")
print("="*84)
GEO=[i for i,f in enumerate(FN)]
def ridge_fit(X,t,lam=1.0):
    X=np.column_stack([X,np.ones(len(X))]); A=X.T@X+lam*np.eye(X.shape[1]); return np.linalg.solve(A,X.T@t)
def zs(X,mu,sd): return (X-mu)/np.where(sd>0,sd,1)

vm = np.isin(lang, viable)
correct_s=(Ps.argmax(1)==y).astype(float); correct_e=(Pe.argmax(1)==y).astype(float)
budgets=[0.2,0.4,0.6,0.8]
FEATSETS={"geometry(18)":GEO, "surface+conf":None, "geometry+conf":GEO}
res={b:{k:[] for k in list(FEATSETS)+["random","confidence","oracle","all_cheap","all_exp"]} for b in budgets}
for held in viable:
    tr=vm&(lang!=held); te=lang==held
    for name,cols in FEATSETS.items():
        if name=="surface+conf": Xtr=np.column_stack([conf[tr],ntok[tr],frag[tr]]); Xte=np.column_stack([conf[te],ntok[te],frag[te]])
        elif name=="geometry+conf": Xtr=np.column_stack([F[tr][:,cols],conf[tr],ntok[tr],frag[tr]]); Xte=np.column_stack([F[te][:,cols],conf[te],ntok[te],frag[te]])
        else: Xtr=F[tr][:,cols]; Xte=F[te][:,cols]
        ok=np.all(np.isfinite(Xtr),1); Xtr,ttr=Xtr[ok],D[tr][ok]
        mu,sd=Xtr.mean(0),Xtr.std(0); w=ridge_fit(zs(Xtr,mu,sd),ttr,lam=10.0)
        Xte=np.nan_to_num(Xte,nan=0.0); s=np.column_stack([zs(Xte,mu,sd),np.ones(te.sum())])@w
        for b in budgets:
            k=int(round(b*te.sum())); order=np.argsort(-s); sel=np.zeros(te.sum(),bool); sel[order[:k]]=True
            res[b][name].append(float(np.where(sel,correct_e[te],correct_s[te]).mean()))
    rng=np.random.default_rng(0)
    for b in budgets:
        k=int(round(b*te.sum()))
        sel=np.zeros(te.sum(),bool); sel[rng.choice(te.sum(),k,replace=False)]=True
        res[b]["random"].append(float(np.where(sel,correct_e[te],correct_s[te]).mean()))
        o=np.argsort(conf[te]); sel=np.zeros(te.sum(),bool); sel[o[:k]]=True   # escalate least-confident
        res[b]["confidence"].append(float(np.where(sel,correct_e[te],correct_s[te]).mean()))
        o=np.argsort(-D[te]); sel=np.zeros(te.sum(),bool); sel[o[:k]]=True
        res[b]["oracle"].append(float(np.where(sel,correct_e[te],correct_s[te]).mean()))
        res[b]["all_cheap"].append(float(correct_s[te].mean())); res[b]["all_exp"].append(float(correct_e[te].mean()))
print(f"\n{'budget':>7s} {'cheap':>7s} {'random':>7s} {'conf':>7s} {'surf+conf':>10s} {'geom':>7s} {'geom+conf':>10s} {'oracle':>7s} {'all_exp':>8s}")
for b in budgets:
    r=res[b]; f=lambda k: np.mean(r[k])
    print(f"{b:7.0%} {f('all_cheap'):7.3f} {f('random'):7.3f} {f('confidence'):7.3f} "
          f"{f('surface+conf'):10.3f} {f('geometry(18)'):7.3f} {f('geometry+conf'):10.3f} {f('oracle'):7.3f} {f('all_exp'):8.3f}")
print("\ngeometry - random (mean over held-out languages, +/- sd):")
for b in budgets:
    d=np.array(res[b]["geometry(18)"])-np.array(res[b]["random"])
    print(f"  budget {b:.0%}: {d.mean():+.4f} +/- {d.std():.4f}   (wins {int((d>0).sum())}/{len(d)} languages)")
