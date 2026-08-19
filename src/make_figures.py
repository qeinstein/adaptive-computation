"""Generate the three paper figures from cached numerical artifacts.

No value is typed by hand: every number plotted is recomputed here from
results/full15/*.npz, the same arrays the manuscript tables are audited against.

Palette: #2166ac, #b2182b, #762a83, #4d9221 -- validated with the dataviz
validator (lightness band, chroma floor, CVD separation, normal-vision floor,
contrast all PASS in this order). Grey #8c8c8c and near-black #1a1a1a are neutral
reference marks, not categorical identities.

Grayscale: the worst categorical pair differs by only 0.025 relative luminance,
so colour alone is not sufficient in print. Every series therefore carries
redundant marker and linestyle encoding, and key series are directly labelled.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Emit LaTeX-native PGF alongside vector PDF. In the PGF version all text is
# typeset by LaTeX in the document's own font, so figure labels and math match
# the body text exactly. PDF is kept as a fallback that needs no extra packages.
EMIT_PGF = os.environ.get("EMIT_PGF", "1") == "1"
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,          # let the document's own font rule
    "pgf.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    if EMIT_PGF:
        try:
            fig.savefig(os.path.join(FIG, name + ".pgf"))
        except Exception as exc:            # pgf needs a working latex on PATH
            print(f"  (pgf skipped for {name}: {type(exc).__name__})")
    plt.close(fig)
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARR = os.path.join(ROOT, "results", "full15")
FIG = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG, exist_ok=True)

BLUE, RED, PURPLE, GREEN = "#2166ac", "#b2182b", "#762a83", "#4d9221"
GREY, INK = "#8c8c8c", "#1a1a1a"
MUTED = "#5a5a5a"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9,
    "axes.titlesize": 9.5, "legend.fontsize": 8, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5, "axes.linewidth": 0.6, "axes.edgecolor": "#666666",
    "xtick.color": "#666666", "ytick.color": "#666666",
    "axes.labelcolor": INK, "text.color": INK,
    "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "#dddddd", "grid.linewidth": 0.5,
})

# ---------------------------------------------------------------- shared data
z = np.load(os.path.join(ARR, "full15_arrays_3src.npz"), allow_pickle=True)
lang, y, ntok, frag, F = z["lang"], z["y"], z["ntok"], z["frag"], z["F"]
FN = list(z["feat_names"])
Ps, Pe = z["P_minilm_l6"], z["P_mdeberta_base"]
idx = np.arange(len(y))
LANGS = sorted(set(lang.tolist()))
conf = Ps.max(1)
DPROB = Pe[idx, y] - Ps[idx, y]
cs = (Ps.argmax(1) == y).astype(float)
ce = (Pe.argmax(1) == y).astype(float)
DCORR = ce - cs

viable = []
for l in LANGS:
    m = lang == l
    k = (Ps[m].argmax(1) == y[m]).sum(); N = int(m.sum()); p = k / N
    lo = (p + 1.92 / (2 * N) - 1.96 * np.sqrt(p * (1 - p) / N + 0.96 / (N * N))) / (1 + 3.84 / N)
    if lo > 1 / 3:
        viable.append(l)
vm = np.isin(lang, viable)
BUDGETS = [0.2, 0.4, 0.6, 0.8]


def ridge(X, t, lam=10.0):
    X = np.column_stack([X, np.ones(len(X))])
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ t)


def route(target, use_conf):
    out = {b: [] for b in BUDGETS}
    for held in viable:
        tr = vm & (lang != held); te = lang == held
        Xtr, Xte = F[tr], F[te]
        if use_conf:
            Xtr = np.column_stack([Xtr, conf[tr], ntok[tr], frag[tr]])
            Xte = np.column_stack([Xte, conf[te], ntok[te], frag[te]])
        ok = np.all(np.isfinite(Xtr), 1)
        Xtr, t = Xtr[ok], target[tr][ok]
        mu, sd = Xtr.mean(0), np.where(Xtr.std(0) > 0, Xtr.std(0), 1)
        w = ridge((Xtr - mu) / sd, t)
        sc = np.column_stack([(np.nan_to_num(Xte) - mu) / sd, np.ones(int(te.sum()))]) @ w
        for b in out:
            k = int(round(b * te.sum()))
            sel = np.zeros(int(te.sum()), bool); sel[np.argsort(-sc)[:k]] = True
            out[b].append(float(np.where(sel, ce[te], cs[te]).mean()))
    return np.array([np.mean(out[b]) for b in BUDGETS])


def baseline(kind):
    out = {b: [] for b in BUDGETS}
    rng = np.random.default_rng(0)
    for held in viable:
        te = lang == held; n = int(te.sum())
        for b in out:
            k = int(round(b * n)); sel = np.zeros(n, bool)
            if kind == "conf":
                sel[np.argsort(conf[te])[:k]] = True
            elif kind == "oracle":
                sel[np.argsort(-DCORR[te])[:k]] = True
            else:
                sel[rng.choice(n, k, replace=False)] = True
            out[b].append(float(np.where(sel, ce[te], cs[te]).mean()))
    return np.array([np.mean(out[b]) for b in BUDGETS])


# =============================================================== FIGURE 1
def figure_frontier():
    oracle = baseline("oracle"); confb = baseline("conf"); rand = baseline("rand")
    geom_p = route(DPROB, False); geom_c = route(DCORR, False); geom_pc = route(DCORR, True)
    all_cheap = float(cs[vm].mean()); all_exp = float(ce[vm].mean())
    x = [b * 100 for b in BUDGETS]

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.set_axisbelow(True); ax.grid(axis="y")

    # the region no practical method reaches -- the paper's central negative result
    ax.axhspan(all_exp, 0.71, color="#f2f2f2", zorder=0)
    ax.axhline(all_exp, color=INK, lw=1.1, ls=(0, (6, 3)), zorder=2)
    ax.text(20.4, all_exp + 0.006, "always-expensive (100% compute)",
            fontsize=7.6, color=INK, va="bottom")
    ax.axhline(all_cheap, color=GREY, lw=1.0, ls=(0, (2, 2)), zorder=2)
    ax.text(20.4, all_cheap + 0.005, "always-cheap", fontsize=7.6, color=MUTED, va="bottom")

    series = [
        (oracle,  "oracle",                    INK,    "o", "-",           2.0),
        (geom_pc, "geometry + confidence",     BLUE,   "s", "-",           1.6),
        (confb,   "confidence",                RED,    "^", "-",           1.6),
        (geom_c,  r"geometry ($\Delta_{corr}$)", PURPLE, "D", (0, (5, 2)),  1.6),
        (geom_p,  r"geometry ($\Delta_{prob}$)", GREEN,  "v", (0, (1, 1.6)), 1.6),
        (rand,    "random",                    GREY,   "x", (0, (3, 2, 1, 2)), 1.3),
    ]
    for vals, label, c, mk, ls, lw in series:
        ax.plot(x, vals, color=c, marker=mk, linestyle=ls, linewidth=lw,
                markersize=4.6, markerfacecolor="white", markeredgewidth=1.2,
                markeredgecolor=c, label=label, zorder=4, clip_on=False)

    ax.annotate("", xy=(60, oracle[2]), xytext=(60, all_exp),
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=0.8, shrinkA=0, shrinkB=0))
    ax.text(61.2, (oracle[2] + all_exp) / 2, f"{(oracle[2]-all_exp)*100:.0f} pts\nheadroom",
            fontsize=7.4, color=MUTED, va="center")

    ax.set_xlabel("compute budget (share of inputs escalated)")
    ax.set_ylabel("accuracy")
    ax.set_xticks(x); ax.set_xticklabels([f"{v:.0f}%" for v in x])
    ax.set_xlim(19, 84); ax.set_ylim(0.40, 0.71)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False,
              ncol=3, columnspacing=1.5, handlelength=2.6, borderpad=0.2,
              handletextpad=0.5)
    save(fig, "frontier")
    return dict(oracle=oracle, conf=confb, rand=rand, geom_p=geom_p,
                geom_c=geom_c, geom_pc=geom_pc, cheap=all_cheap, exp=all_exp)


# =============================================================== FIGURE 2
def eta2(v, g):
    v = np.asarray(v, float); m = np.isfinite(v); v, g = v[m], g[m]
    gm = v.mean()
    return float(sum((g == l).sum() * (v[g == l].mean() - gm) ** 2
                     for l in set(g.tolist())) / ((v - gm) ** 2).sum())


def figure_eta2():
    SRC = [("mdeberta_base", "mDeBERTa-base\n(task-tuned)", BLUE, "o"),
           ("afroxlmr_base", "AfroXLMR-base\n(African MLM)", RED, "s"),
           ("xlmr_base", "XLM-R-base\n(generic MLM)", PURPLE, "^")]
    STATS = [("ang_disp", "angular dispersion"),
             ("spec_conc", "spectral concentration"),
             ("eff_rank", "effective rank")]
    vals = {(s, st): np.mean([eta2(F[:, FN.index(f"{s}_L{l}_{st}")], lang) for l in (4, 8, 12)])
            for s, _, _, _ in SRC for st, _ in STATS}

    fig, ax = plt.subplots(figsize=(5.4, 2.7))
    ax.set_axisbelow(True); ax.grid(axis="x")
    ypos = np.arange(len(STATS))[::-1]
    dodge = {SRC[0][0]: 0.13, SRC[1][0]: 0.0, SRC[2][0]: -0.13}
    for yi, (st, stlab) in zip(ypos, STATS):
        xs = [vals[(s, st)] for s, _, _, _ in SRC]
        ax.plot([min(xs), max(xs)], [yi, yi], color="#cccccc", lw=1.4, zorder=1)
        for (s, slab, c, mk) in SRC:
            ax.plot(vals[(s, st)], yi + dodge[s], marker=mk, color=c, markersize=6,
                    markerfacecolor="white", markeredgewidth=1.5, markeredgecolor=c, zorder=3)
    ax.set_yticks(ypos); ax.set_yticklabels([s[1] for s in STATS])
    ax.set_xlabel(r"$\eta^2$   (share of variance lying between languages)")
    ax.set_xlim(0, 0.68); ax.set_ylim(-0.6, len(STATS) - 0.3)
    ax.legend(handles=[Line2D([], [], marker=mk, color=c, linestyle="none", markersize=6,
                              markerfacecolor="white", markeredgewidth=1.5, label=lab.replace("\n", " "))
                       for _, lab, c, mk in SRC],
              loc="lower right", frameon=False, handletextpad=0.4, borderpad=0.2)
    ax.text(0.02, len(STATS) - 0.55, "more example-driven $\\longrightarrow$",
            fontsize=7.2, color=MUTED)
    ax.text(0.66, len(STATS) - 0.55, "$\\longleftarrow$ more language-driven",
            fontsize=7.2, color=MUTED, ha="right")
    save(fig, "eta2")
    return vals


# =============================================================== FIGURE 3
def rank(v):
    return np.argsort(np.argsort(v)).astype(float)


def pspear(yv, xv, C_):
    m = np.isfinite(yv) & np.isfinite(xv) & np.all(np.isfinite(C_), 0)
    n = int(m.sum())
    if n < 30:
        return np.nan, n
    C = np.column_stack([rank(c[m]) for c in C_] + [np.ones(n)])
    r = lambda v: v - C @ np.linalg.lstsq(C, v, rcond=None)[0]
    a, b = r(rank(yv[m])), r(rank(xv[m]))
    d = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return (float((a * b).sum() / d) if d > 0 else np.nan), n


def meta(rs, ns, k=3):
    r = np.array(rs, float); N = np.array(ns, float)
    m = np.isfinite(r) & (N > k + 4)
    w = N[m] - k - 4
    return float(np.tanh((w * np.arctanh(np.clip(r[m], -.999, .999))).sum() / w.sum()))


def figure_pooled_vs_within():
    feats = [("mdeberta_base_L8_ang_disp", "angular dispersion (L8)", RED, "s"),
             ("mdeberta_base_L12_spec_conc", "spectral concentration (L12)", PURPLE, "^"),
             ("mdeberta_base_L12_eff_rank", "effective rank (L12)", BLUE, "o")]
    rows = []
    for key, lab, c, mk in feats:
        v = F[:, FN.index(key)]
        pooled, _ = pspear(DPROB, v, [conf, ntok, frag])
        rr, nn = zip(*[pspear(DPROB[lang == l], v[lang == l],
                              [conf[lang == l], ntok[lang == l], frag[lang == l]]) for l in LANGS])
        e2 = eta2(v, lang)
        rows.append((lab, pooled, meta(rr, nn), c, mk, e2))

    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.axhline(0, color="#bbbbbb", lw=0.8, zorder=1)
    # stagger annotation anchors so near-equal within-values do not collide
    order = sorted(range(len(rows)), key=lambda i: rows[i][2])
    anchors = {}
    prev = -1e9
    for i in order:
        a = max(rows[i][2], prev + 0.075)
        anchors[i] = a
        prev = a
    for i, (lab, p, w, c, mk, e2) in enumerate(rows):
        shrinks = abs(w) < abs(p)
        ax.plot([0, 1], [p, w], color=c, lw=1.7,
                ls="-" if shrinks else (0, (5, 2)), zorder=2)
        ax.plot([0, 1], [p, w], marker=mk, color=c, markersize=6, linestyle="none",
                markerfacecolor="white", markeredgewidth=1.5, markeredgecolor=c, zorder=3)
        ax.text(-0.035, p, f"{p:+.3f}", ha="right", va="center", fontsize=8, color=c)
        a = anchors[i]
        if abs(a - w) > 1e-6:
            ax.plot([1.05, 1.14], [w, a], color=c, lw=0.7, zorder=2)
        ax.text(1.17, a, f"{w:+.3f}   {lab}\n$\\eta^2$={e2:.3f}", ha="left", va="center",
                fontsize=7.6, color=INK)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["pooled\nacross languages", "within language\n(Fisher-$z$ combined)"])
    ax.set_ylabel(r"partial Spearman $\rho$ with $\Delta_{prob}$")
    ax.set_xlim(-0.22, 2.6); ax.set_ylim(-0.20, 0.36)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.legend(handles=[Line2D([], [], color=MUTED, lw=1.7, ls="-", label="inflated by pooling"),
                       Line2D([], [], color=MUTED, lw=1.7, ls=(0, (5, 2)), label="masked by pooling")],
              loc="lower left", frameon=False, borderpad=0.2)
    save(fig, "pooled_vs_within")
    return rows


if __name__ == "__main__":
    f1 = figure_frontier()
    print("figure 1 (frontier) values, for cross-check against Table 7:")
    for k in ("rand", "conf", "geom_p", "geom_c", "geom_pc", "oracle"):
        print(f"  {k:8s} " + "  ".join(f"{v:.3f}" for v in f1[k]))
    print(f"  all-cheap {f1['cheap']:.3f}   all-expensive {f1['exp']:.3f}")
    f2 = figure_eta2()
    print("\nfigure 2 (eta^2) values, for cross-check against Table 5:")
    for (s, st), v in sorted(f2.items()):
        print(f"  {s:15s} {st:10s} {v:.3f}")
    f3 = figure_pooled_vs_within()
    print("\nfigure 3 (pooled vs within), for cross-check against Table 6:")
    for lab, p, w, _, _, e2 in f3:
        print(f"  {lab:32s} pooled {p:+.3f}  within {w:+.3f}  eta2 {e2:.3f}")
    print(f"\nwrote {FIG}/frontier.pdf, eta2.pdf, pooled_vs_within.pdf")
