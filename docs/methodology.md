# RFS-SCP v16.1 — Extended Methodology

## Overview

This document provides a detailed description of the methods used in the RFS-SCP v16.1 pipeline, supplementing the paper's Methods section. It covers the Circumplex proxy operationalisation, statistical procedures, and design decisions not fully described in the main text.

---

## 1. Circumplex Proxy Operationalisation

### 1.1 Cohesion Proxy

The Cohesion proxy is a weighted sum of seven sub-features derived from MI session annotations. Sub-feature weights are theory-driven, sourced from Olson (2011) and Miller & Rollnick (2012); they are not fitted to AnnoMI labels.

| Sub-feature | Computation | Weight | Source |
|------------|-------------|--------|--------|
| `empathy_rate` | Proportion of therapist utterances labelled as reflections | 0.24 | Olson (2011) "supportive communication" |
| `agreement_rate` | Proportion of client utterances labelled as change talk | 0.18 | Olson (2011) "togetherness" |
| `sent_congruence` | `1 − |VADER(therapist) − VADER(client)|` | 0.15 | Olson (2011) "emotional congruence" |
| `sent_pos` | `(VADER compound mean + 1) / 2` | 0.12 | Gottman (1994) positivity ratio |
| `neg_absence` | `1 − proportion of negation tokens` | 0.12 | Gottman (1994) negative affect |
| `wc_balance` | `min(N_th, N_cl) / max(N_th, N_cl)` | 0.11 | Olson (2011) "turn balance" |
| `sent_div_absence` | `1 − |VADER diff| / 0.60` (clipped) | 0.08 | Olson (2011) "shared reality" |

Cohesion = 100 × Σ w_k × sub-feature_k, clipped to [0, 100].

### 1.2 Flexibility Proxy

| Sub-feature | Computation | Weight | Source |
|------------|-------------|--------|--------|
| `question_rate` | Proportion of therapist question utterances | 0.25 | Olson (2011) role flexibility |
| `oscillation_rate` | Proportion of consecutive client talk-type transitions | 0.22 | Olson (2011) adaptability |
| `sent_variance` | VADER std clipped to [0,1] | 0.20 | Olson (2011) affective range |
| `anti_rigidity` | `1 / (1 + exp(3 × lag1_autocorr))` | 0.20 | Inverse emotional inertia |
| `novelty` | Mean type-token ratio across utterances | 0.13 | Cognitive flexibility proxy |

Flexibility = 100 × Σ w_k × sub-feature_k, clipped to [0, 100].

### 1.3 Communication Proxy (Future Work)

The Communication proxy is computed but not used in primary claims. Coverage of the clarification-act regex is 1.3% of AnnoMI utterances, which is insufficient for confirmatory analysis. See `docs/concern_responses.md` §Concern 4 for full justification.

---

## 2. VADER Sentiment Analysis

VADER (Valence Aware Dictionary and sEntiment Reasoner; Hutto & Gilbert, 2014) is used for utterance-level sentiment extraction. The `compound` score ranges from −1 (maximally negative) to +1 (maximally positive).

VADER is **mandatory**: the pipeline raises `RuntimeError` on import failure. Silent fallback is disabled for reproducibility. The VADER backend is recorded in `results/reproducibility_manifest_v16.json`.

---

## 3. Bayesian MCMC Weight Optimisation

### 3.1 Model Specification

```
y_i ~ Bernoulli(σ(α·ĉ_i(w) + β))
w   ~ Dirichlet(α = 2.0)
```

where ĉ_i(w) is the Cohesion proxy value for session i under weight vector w, and α, β are fitted via logistic regression in each MCMC step.

### 3.2 MCMC Sampler

- **Algorithm**: Metropolis-Hastings in ALR (Additive Log-Ratio) transformed space
- **Proposal**: isotropic Gaussian, step size adapted for acceptance rate ∈ [0.20, 0.40]
- **Iterations**: N = 5,000 with burn-in = 1,250 (25%)
- **Acceptance rate (v16.1)**: 36.0%
- **Temperature constant**: None — the acceptance ratio uses exact log-posterior differences

### 3.3 ΔAUC Reporting

Two versions are reported:

1. **In-sample ΔAUC** (biased, supplementary only): MCMC posterior evaluated on all N=133 sessions — the same data used for likelihood computation. Expected to inflate ΔAUC.
2. **Hold-out ΔAUC** (primary): MCMC posterior evaluated on N_test=40 held-out sessions only.

The primary claim is based on the hold-out ΔAUC CI = [+0.015, +0.126], p = 0.024.

---

## 4. Hold-Out Weight Learning

A 70/30 stratified split (N_train=93, N_test=40) is used to estimate the generalisation of both theory-driven and data-driven weights.

**Objective**: Minimise negative log-loss subject to L2 regularisation towards theory weights:

```
L(w) = −Σ [y_i log(ĉ_i) + (1−y_i) log(1−ĉ_i)] + λ × ||w_normalised − w_theory||²
```

with λ = 0.5 (L2 penalty).

**Results (v16.1)**:

| Weight set | Train AUC | Test AUC | Gap |
|-----------|-----------|----------|-----|
| Theory (Olson 2011) | 0.897 | 0.606 | +0.291 ⚠ |
| Learned (L2 λ=0.5) | 0.908 | 0.753 | +0.155 ⚠ |

Both gaps exceed 0.10, indicating weight-space under-identification at N=133. Theory weights are used in the main analysis on grounds of parsimony. N ≥ 300 is estimated for stable weight learning.

---

## 5. Cross-Validation Design

### Primary CV (RQ1, RQ5, SHAP)

- 5-fold Stratified K-Fold × 5 repeats = 25 folds
- Single-class validation folds are skipped (counted in `n_folds_skipped`)
- Primary metrics: MCC and Balanced Accuracy (AUC reported as secondary)

### GroupKFold (RQ5 sensitivity)

- 10-fold GroupKFold by `transcript_id`
- Prevents transcript-level leakage across folds

### LOFO

- 5-fold × 3 repeats per feature
- Drop CI: ±1.96 × √(σ²_full + σ²_leave-one-out) / n_folds

---

## 6. Multiple Comparisons

Benjamini-Hochberg FDR correction (Benjamini & Hochberg, 1995) is applied across all primary hypothesis tests. Tests are classified as:

- **PRIMARY**: main confirmatory hypothesis tests (RQ1–RQ4, RQ6 urgency)
- **FUTURE WORK**: Communication axis tests (explicitly excluded from primary count)
- **SECONDARY**: temporal dynamics, exploratory analyses

---

## 7. Decision Curve Analysis

Net benefit is computed as:

```
NB(t) = TP/n − FP/n × t/(1−t)
```

Three strategies are compared: Circumplex proxy, full logistic model, treat-all baseline. Treat-none = 0 by definition.

---

## 8. Temporal Dynamics Features

| Feature | Computation |
|---------|-------------|
| `transition_entropy` | Shannon entropy of sentiment sign-change rate |
| `emotional_inertia` | Absolute lag-1 autocorrelation of VADER sequence |
| `cohesion_volatility` | VADER standard deviation across session |
| `empathy_recovery_rate` | Proportion of positive VADER transitions in therapist turns |

These are secondary features, not used in the primary Cohesion proxy, but analysed separately for their association with MI quality labels.

---

## 9. References

- Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery rate. *Journal of the Royal Statistical Society B, 57*(1), 289–300.
- Gottman, J. M. (1994). *What Predicts Divorce?* Lawrence Erlbaum Associates.
- Hutto, C. J. & Gilbert, E. (2014). VADER: A parsimonious rule-based model for sentiment analysis of social media text. *ICWSM*.
- Miller, W. R. & Rollnick, S. (2012). *Motivational Interviewing* (3rd ed.). Guilford Press.
- Olson, D. (2011). FACES IV and the Circumplex Model. *Journal of Marital and Family Therapy, 37*(1), 64–80.
- Tanana, M., Hallgren, K. A., Imel, Z. E., Atkins, D. C., & Srikumar, V. (2016). A comparison of NLP methods for automated coding of motivational interviewing. *Journal of Substance Abuse Treatment, 65*, 43–50.
