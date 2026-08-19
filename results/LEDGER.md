# Results Ledger

Exact numbers intended for publication, with sample, statistic, uncertainty, and
limitation recorded per finding. Purpose: prevent the write-up from drifting
stronger than the evidence. **Nothing here may be restated more strongly in prose
than it is stated here.**

Commit anchor: `c675d74`. Findings 1 and 2 corrected 2026-08-19 after closing open items. All numbers reproducible from
`src/` against `data/cache/`.

---

## Shared experimental setup

| item | value |
|---|---|
| Dataset | `masakhane/afrixnli`, 18 configs, 450 dev / 600 test per language |
| Clean languages (15) | amh ewe hau ibo kin lin lug orm sna sot twi wol xho yor zul |
| Excluded (3) | eng, fra, swa — intersect XNLI |
| Evaluation split | **test**, n = 9,000 (600 × 15) |
| Calibration split | **dev**, n = 6,750 (450 × 15), pooled, one scalar temperature per model |
| Label order | canonical `0=entailment, 1=neutral, 2=contradiction`; per-checkpoint permutation applied |
| Rungs | `multilingual-MiniLMv2-L6-mnli-xnli` (6L/384d), `mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` (12L/768d), `xlm-roberta-large-xnli` (24L/1024d) |
| Feature sources | mDeBERTa-base (task-tuned), AfroXLMR-base (African-adapted MLM), XLM-R-base (generic MLM, post-hoc) |
| Geometry | effective rank, spectral concentration, angular dispersion; layers {4,8,12}; computed over the token dimension of a single sequence |
| Δ target | `p_expensive(gold|x) − p_cheap(gold|x)` on temperature-calibrated probabilities |
| Controls | cheap-model max-prob confidence, n_tokens, fragmentation |
| Within-language statistic | partial Spearman per language → Fisher-z meta-analysis weighted by `n−k−3` |
| Multiple comparisons | Holm across the 18 features per pair |

**No fine-tuning was performed.** All checkpoints frozen.

---

## Finding 1 — Benchmark contamination

**Claim.** AfriXNLI is a translation of XNLI, and XNLI-tuned checkpoints therefore
cannot be evaluated cleanly on its eng/fra/swa configs.

| evidence | value |
|---|---|
| AfriXNLI-eng unique pairs (dev+test) | 1,050 |
| Verbatim overlap with XNLI English (validation+test) | **1,047 (99.7%)** |
| AfriXNLI-eng/dev vs XNLI validation | 450/450 — exact |
| XNLI ∩ AfriXNLI languages | {eng, fra, swa} |

Accuracy on the contaminated configs, **full dev and test splits**
($n=450$ dev, $n=600$ test; accuracy is temperature-invariant so no calibration
is involved):

| model | eng/dev | eng/test | fra/dev | fra/test | swa/dev | swa/test | 15 clean (test) |
|---|---|---|---|---|---|---|---|
| MiniLM-L6 | 0.887 | 0.760 | 0.927 | 0.720 | 0.891 | 0.622 | 0.410 |
| mDeBERTa-base | 1.000 | 0.890 | 0.998 | 0.867 | 0.993 | 0.742 | 0.545 |
| XLM-R-large | 0.998 | **1.000** | 0.987 | **0.995** | 0.973 | **0.978** | 0.523 |

**The dev–test gap is a contamination fingerprint.** MiniLM and mDeBERTa — the two
whose cards report training on XNLI *development* data — drop 11–27 points from dev
to test. XLM-R-large drops nothing; test is marginally higher than dev on all three.

**The contamination is split-specific, and this is the sharper claim.** MiniLM and
mDeBERTa were trained on XNLI *dev*, so they memorise dev (mDeBERTa scores 1.000 on
eng/dev but only 0.890 on eng/test). `joeddav/xlm-roberta-large-xnli` scores **1.000
on the eng test split** and 0.978 on swa test, indicating it trained on XNLI *test*.
That checkpoint has ~91k downloads.

Documentary support (model/dataset cards, retrieved 2026-08-19):
- AfriXNLI card: *"translations of a subset of the XNLI dataset into 16 African languages … maintaining the English and French subsets from the original XNLI dataset"*; *"dev and test [are] a subset of the original dev and test splits of the XNLI dataset."*
- MiniLM card: *"This model was trained on the XNLI development dataset and the MNLI train dataset."*
- joeddav card: *"fine-tunes it on a combination of NLI data in 15 languages … fine-tuned on XNLI."* (does not specify split)

**Limitations.**
- The joeddav training-split claim is inferred from 1.000 test accuracy, not stated on the card. Word it as *consistent with* training on XNLI test, not as documented fact.
- For the 15 African languages this is **benchmark-lineage / underlying-example contamination**, NOT verbatim string leakage. The African strings are translations; the *examples* were seen in other languages. Do not conflate these.
- CORRECTED 2026-08-19: an earlier version of this ledger reported mDeBERTa at 1.000/0.995 on eng/swa and contrasted it with a test-split clean figure. Those were **dev** numbers. The like-for-like test numbers are above and they reassign the dramatic memorisation to XLM-R-large.

---

## Finding 2 — Parameter count does not reliably order capability

**Claim.** Among off-the-shelf XNLI checkpoints, model size does not predict which
model wins on a given African language; the ordering is language-dependent and
unstable. Cascade design therefore cannot be read off parameter count.

n = 9,000 (test, 15 clean languages). Chance = 0.333, Wilson 95% intervals.

| rung | params | temperature | accuracy | above chance |
|---|---|---|---|---|
| MiniLMv2-L6 | ~118M | 4.116 | 0.410 | 11/15 |
| mDeBERTa-v3-base | ~278M | 1.704 | 0.545 | 14/15 |
| XLM-R-large | ~560M | 2.723 | 0.523 | 14/15 |

**The aggregate mDeBERTa − XLM-R gap of +0.022 is NOT significant.** Cluster
bootstrap resampling languages and then examples within language (5,000×):
**95% CI [−0.035, +0.083]**, P(diff>0) = 0.772. (An example-level bootstrap gives
[+0.009, +0.035], but it ignores language clustering and is the wrong analysis.)

What *is* substantial is the per-language instability — mDeBERTa minus XLM-R-large:

```
amh -0.145  ewe -0.037  hau -0.080  ibo +0.220  kin +0.147  lin +0.008
lug +0.033  orm -0.127  sna +0.200  sot +0.197  twi +0.015  wol -0.002
xho -0.087  yor -0.010  zul -0.003
```

Swings exceed ±0.2 with no consistent winner. **Do not write "the larger model is
worse."** Write that the ordering is language-dependent and not predictable from size.

Consequence for Δ: `mDeBERTa → XLM-R-large` has mean Δ = **−0.011** (helps 37.5%,
hurts 39.7%) — no capacity ladder. Only `MiniLM → mDeBERTa` is a usable ladder:
mean Δ = **+0.099**, sd 0.221, helps 51.2% / hurts 26.0% / flat 22.7%, ~8× FLOPs.

**Limitations.** Three checkpoints from two families, all XNLI-tuned: a statement
about the available off-the-shelf ecosystem, not about scaling in general. XLM-R-large's
clean-language accuracy may itself be depressed relative to its contaminated-language
accuracy in ways that confound the comparison.

---

## Finding 3 — Pooled correlations mislead in both directions

**Claim.** A feature's between-language variance share (η²) governs how badly its
pooled correlation with Δ misstates the within-language relationship — inflating
some features and masking others.

Pair: MiniLM → mDeBERTa. n = 9,000.

| quantity | η² | pooled ρ | within ρ | effect |
|---|---|---|---|---|
| mDeBERTa L8 angular dispersion | **0.757** | +0.296 | +0.061 | inflated **4.9×** |
| mDeBERTa L12 effective rank | **0.040** | −0.009 | −0.127 | **masked ~10×** |
| mDeBERTa L12 spectral concentration | 0.031 | +0.044 | +0.099 | masked |
| Δ itself | 0.126 | — | — | — |
| cheap-model confidence | 0.212 | — | — | — |
| n_tokens | 0.196 | — | — | — |

Angular dispersion is ~76% language identity. Effective rank is ~4%. Naïve pooling
therefore manufactured a relationship that was language identity and erased one that
was genuine — errors in *opposite directions*.

**Three-source replication (post-hoc, `src/replicate3.py`).** Registered explicitly as
post-hoc robustness, motivated by the two-source result being unable to distinguish a
general property from a coincidence of two models. Protocol frozen; only the
feature-extracting model differs. Third source: XLM-R-base (generic multilingual MLM).

*The proposed generalisation FAILED.* Across all 27 features (3 sources x 3 layers x
3 statistics), Spearman(eta^2, |pooled| - |within|) = +0.415 (Pearson +0.566),
permutation p = 0.032, but **bootstrap 95% CI = [-0.031, +0.737]** -- straddling zero.
Per source: mDeBERTa +0.917, AfroXLMR +0.133, XLM-R-base +0.133. The relationship is
carried almost entirely by one model. The 27 features are not independent (9 per source,
sharing layers and statistics), which is why the permutation p and bootstrap CI disagree.
**Do not claim that eta^2 predicts pooling bias as a general quantitative law.**

*What DID replicate.* The rank ordering of statistics by eta^2 is identical in all three
representation spaces:

| statistic | mDeBERTa (task-tuned) | AfroXLMR (African MLM) | XLM-R-base (generic MLM) |
|---|---|---|---|
| angular dispersion | 0.583 | 0.484 | 0.542 |
| spectral concentration | 0.112 | 0.249 | 0.241 |
| effective rank | 0.075 | 0.133 | 0.151 |

Angular dispersion is the most language-determined statistic and effective rank the least,
in a task-tuned model, an African-adapted MLM, and a generic multilingual MLM alike. This
is the defensible three-source claim, and it supports the same practical warning:
angular-dispersion-type statistics are the ones that must not be pooled across languages.

**Limitations.** η² is computed on the same data as the correlations. The
relationship is descriptive, not causal. Geometry is computed over one sequence's
tokens, so all three statistics are partly length-sensitive by construction.

---

## Finding 4 — A robust but operationally weak representation signal

**Claim.** One geometry statistic has a statistically unimpeachable association with
Δ that confers no practical routing advantage over a confidence threshold.

### 4a-pre. THE TWO NOTIONS OF MARGINAL BENEFIT ARE NOT INTERCHANGEABLE

Post-hoc (`src/dual_target.py`), prompted by the Stage IV target-misalignment
correction. Same procedure for both targets: partial Spearman per language
controlling for confidence, token count and fragmentation; Fisher-z meta;
Holm across all 27 features.

| | Δ_prob = p_e(y*) − p_s(y*) | Δ_correct = 1[ŷ_e=y*] − 1[ŷ_s=y*] |
|---|---|---|
| corr between targets | \multicolumn{2}{c}{+0.655} | |
| η² (between-language) | 0.126 | **0.033** |
| distribution | continuous, sd 0.221 | +1: 24.7%, 0: 64.1%, −1: 11.2% |

| feature | vs Δ_prob | Holm | vs Δ_correct | Holm |
|---|---|---|---|---|
| mdeberta L12 eff_rank | **−0.127** [−0.147,−0.106] | 5.8e−32 | −0.027 [−0.048,−0.006] | **0.28 (n.s.)** |
| mdeberta L12 spec_conc | +0.099 | 2.9e−19 | +0.003 | 1.0 |
| mdeberta L8 ang_disp | +0.061 | 2.2e−07 | +0.007 | 1.0 |
| mdeberta L12 ang_disp | +0.029 | 0.13 (n.s.) | **−0.066** [−0.086,−0.045] | **1.6e−08** |
| **confidence baseline** | **+0.003** [−0.018,+0.023] | — | **−0.099** [−0.120,−0.079] | — |

**KEY CLAIM (candidate paper centrepiece).**

> The representation signal is real but attaches to the wrong quantity. Effective
> rank predicts the continuous probability gain from escalation (ρ=−0.127,
> Holm p=5.8e−32) but not whether escalation changes the prediction (ρ=−0.027,
> Holm p=0.28, not significant). Confidence shows the reverse pattern (+0.003 vs
> −0.099). The two notions of "benefit from more computation" are correlated at
> only +0.655 and are predicted by different features — for one feature
> (L12 angular dispersion) with opposite signs.

This explains the Stage IV routing outcome mechanistically rather than leaving it
as an unexplained baseline loss, and it supersedes "the effect is real but tiny"
as the characterisation of Finding 4. The effect is real, and it is *targeted at a
quantity that is not the decision variable*.

Also note η²(Δ_correct)=0.033 vs η²(Δ_prob)=0.126: target choice determines how
much language confounding is inherited. And pooling inverts the eff_rank/Δ_correct
sign too (pooled +0.050 vs within −0.027).

### 4a. The association is real

Feature: `mdeberta_base_L12_eff_rank`. Pair: MiniLM → mDeBERTa. n = 9,000.

| test | result |
|---|---|
| Within-language ρ (Fisher-z meta) | **−0.127**, 95% CI [−0.147, −0.106] |
| Holm-adjusted p (18 features) | **3.88 × 10⁻³²** |
| Sign consistency | 12/15 languages |
| Bootstrap, 300× within-language resample | −0.1272, 95% [−0.1420, −0.1107], **all negative** |
| Subsample 25%, 40 seeds | −0.1215 ± 0.0187, all negative |
| Subsample 50%, 40 seeds | −0.1292 ± 0.0107, all negative |
| Leave-one-language-out | min −0.1381, max −0.1167, all negative |
| Label-permutation null, 20× | mean −0.0005, max\|ρ\| 0.0263 |

Variance explained ≈ **1.6%**.

### 4b. Target misalignment (post-hoc correction, 2026-08-19)

**An error in the original Stage IV design was found while drafting.** The router
was trained to predict Δ (a gold-probability gain) but scored on routing accuracy
(a correctness flip, `correct_e − correct_s`). These are not the same objective:
corr(Δ, flip) = +0.655. Confidence relates to the flip (ρ = −0.085) far more than
to Δ (ρ = +0.021), so the original comparison handicapped geometry by training it
on a misaligned proxy.

Registered as post-hoc: this check was not preplanned. It was prompted by noticing
that confidence has ~zero within-language correlation with Δ (+0.003) yet wins the
routing comparison — two facts only consistent if Δ is not the routing objective.

Retrained on the correctness flip (`src/objective_check.py`), leave-one-language-out,
11 viable languages:

| budget | random | confidence | geom(Δ) | **geom(flip)** | **geom+conf(flip)** | oracle |
|---|---|---|---|---|---|---|
| 20% | 0.457 | 0.466 | 0.461 | **0.468** | **0.472** | 0.613 |
| 40% | 0.486 | 0.506 | 0.498 | 0.504 | **0.508** | 0.688 |
| 60% | 0.519 | 0.543 | 0.524 | 0.535 | **0.545** | 0.688 |
| 80% | 0.548 | 0.564 | 0.553 | 0.559 | **0.565** | 0.688 |

Confidence − geometry(flip) gap, across λ ∈ {1,10,100,1000}:

| budget | gap | sd | conf wins |
|---|---|---|---|
| 20% | −0.0015 to −0.0024 | ~0.019 | 5/11 |
| 60% | +0.0059 to +0.0147 | ~0.019 | 6–8/11 |

Confidence − (geometry+confidence): −0.0005 to −0.0062, combined wins 4–6/11.

**REVISED CLAIM.** Every gap is far smaller than the per-language sd (~0.018), and
win counts sit at chance (4–8 of 11). The correct statement is:

> *With a correctly aligned target, representation geometry and confidence are
> statistically indistinguishable as routing signals at every compute budget, and
> neither approaches the oracle (0.688 vs ~0.545 at the 60% budget).*

**SUPERSEDED WORDING — DO NOT USE.** The previously frozen constraint
*"geometry never beats confidence at any budget or penalty, and is clearly worse at
moderate budgets"* is now known to be an artifact of target misalignment. It must
not appear in the paper.

**The negative conclusion survives, with a changed basis.** Representation geometry
still provides no useful routing signal — but because it is no better than a trivial
confidence baseline, not because it is worse. The unexploited oracle headroom
(+0.14 over the best method at 60%) is the substantive point.

**New methodological finding (candidate 5th).** Evaluating a routing method against
a target that differs from the scoring metric inflates apparent baseline superiority.
Here it manufactured a ~0.019 advantage for confidence at the 60% budget that
disappears under correct targeting.

## Cross-cutting limitations (must appear in the paper)

1. **Δ is not seed-averaged.** The design called for Δ averaged over ≥3 fine-tuning
   seeds to avoid a noise-dominated target. Moving to frozen off-the-shelf checkpoints
   made that impossible and it was never revisited. Δ here comes from a single
   checkpoint pair, so part of its variance is checkpoint idiosyncrasy rather than
   example difficulty. This weakens Finding 4's negative claim: a cleaner target might
   be more predictable. **This is the most significant methodological gap.**
2. **Benchmark-lineage contamination affects the "clean" languages too.** They are
   translations of XNLI examples the checkpoints trained on in other languages.
3. **Temperature was calibrated on African dev splits**, which share that lineage. All
   reported results are on test, and temperature is one scalar per model, but the
   pipeline is not "contamination-free."
4. **Encoder-only.** No decoder/instruction-tuned rung was tested; conclusions may not
   transfer to LLM test-time-compute settings.
5. **Router coverage.** Restricted to the 11/15 languages where the cheap rung clears
   chance; claims cannot extend to the other 4.
6. **Geometry definition.** Per-example statistics over one sequence's tokens are
   partly length-determined by construction; length is controlled statistically, not
   by design.
7. **Single Δ pair for Findings 3–4.** Both rest on MiniLM → mDeBERTa.
