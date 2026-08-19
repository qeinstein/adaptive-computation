# Results Ledger

Exact numbers intended for publication, with sample, statistic, uncertainty, and
limitation recorded per finding. Purpose: prevent the write-up from drifting
stronger than the evidence. **Nothing here may be restated more strongly in prose
than it is stated here.**

Commit anchor: `03803471` (pre-third-source). All numbers reproducible from
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
| mDeBERTa accuracy, eng | **1.000** (n=200, dev sample) |
| mDeBERTa accuracy, swa | **0.995** (n=200, dev sample) |
| mDeBERTa accuracy, 15 clean languages | **0.545** (n=9,000, test) |

Documentary support (model/dataset cards, retrieved 2026-08-19):
- AfriXNLI card: *"translations of a subset of the XNLI dataset into 16 African languages … maintaining the English and French subsets from the original XNLI dataset"*; *"dev and test [are] a subset of the original dev and test splits of the XNLI dataset."*
- MiniLM card: *"This model was trained on the XNLI development dataset and the MNLI train dataset."*
- joeddav card: *"fine-tunes it on a combination of NLI data in 15 languages … fine-tuned on XNLI."*

**Limitations.**
- The eng/swa accuracies come from a **200-example dev sample**, not the test split, and were produced by the superseded `spike.py` run. Accuracy is temperature-invariant so the numbers stand, but they are not on the same footing as the 9,000-example clean figure. *Open item: re-run eng/fra/swa on test under the `full15` protocol before publication.*
- Wording constraint: for the 15 African languages this is **benchmark-lineage / underlying-example contamination**, NOT verbatim string leakage. The African strings are translations; the *examples* were seen in other languages. Do not conflate these.

---

## Finding 2 — Capability is not monotone in parameter count

**Claim.** Among off-the-shelf XNLI checkpoints, model size does not order accuracy
on African languages, so cascade design cannot be read off parameter count.

n = 9,000 (test, 15 clean languages). Chance = 0.333, Wilson 95% intervals.

| rung | params | temperature | accuracy | above chance |
|---|---|---|---|---|
| MiniLMv2-L6 | ~118M | 4.116 | 0.410 | 11/15 |
| mDeBERTa-v3-base | ~278M | 1.704 | **0.545** | 14/15 |
| XLM-R-large | ~560M | 2.723 | **0.523** | 14/15 |

XLM-R-large has ~2× the parameters and 2× the depth of mDeBERTa-base and is
**0.022 lower**. On the earlier 6-language subset the same inversion appeared
(0.606 vs 0.613).

Consequence for Δ: `mDeBERTa → XLM-R-large` has mean Δ = **−0.011** (helps 37.5%,
hurts 39.7%) — no capacity ladder. Only `MiniLM → mDeBERTa` is a usable ladder:
mean Δ = **+0.099**, sd 0.221, helps 51.2% / hurts 26.0% / flat 22.7%, ~8× FLOPs.

**Limitations.** Three checkpoints from two families; all XNLI-tuned, so this is a
statement about the available off-the-shelf ecosystem, not about scaling in general.
No significance test on the 0.022 gap is recorded yet. *Open item: paired bootstrap
on the mDeBERTa−XLM-R accuracy difference.*

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

### 4b. It does not survive contact with a baseline

Leave-one-language-out, 11 viable ladder languages, matched compute budget.
Ridge on all 18 geometry features (λ=10) vs. escalate-least-confident.

| budget | all-cheap | random | **confidence** | geometry | geom+conf | oracle | all-expensive |
|---|---|---|---|---|---|---|---|
| 20% | 0.426 | 0.461 | **0.466** | 0.460 | 0.463 | 0.555 | 0.577 |
| 40% | 0.426 | 0.493 | **0.506** | 0.498 | 0.498 | 0.632 | 0.577 |
| 60% | 0.426 | 0.518 | **0.543** | 0.524 | 0.526 | 0.678 | 0.577 |
| 80% | 0.426 | 0.546 | **0.564** | 0.556 | 0.554 | 0.661 | 0.577 |

Confidence − geometry gap, across ridge λ ∈ {1, 10, 100, 1000}:

| budget | gap | conf wins |
|---|---|---|
| 20% | +0.006 to +0.007 | 7–8 / 11 |
| 40% | +0.008 to +0.012 | 7 / 11 |
| 60% | **+0.017 to +0.019** | 9–10 / 11 |
| 80% | +0.008 to +0.018 | 7–10 / 11 |

Geometry beats *random* by only +0.5 to +1.0 points (wins 5–9 / 11). Oracle reaches
0.678 at 60% budget, so the headroom is real (+0.10 over all-expensive) and unexploited.

**Required qualification.** At the 20% and 40% budgets the per-language sd (~0.015)
exceeds the mean gap (~0.007). The correct statement is: *geometry never beats
confidence at any budget or penalty, and is clearly worse at moderate budgets.* Do
**not** write "confidence substantially outperforms geometry at all budgets."

---

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
