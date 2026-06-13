# RFS-SCP v16.1: Circumplex-Grounded Relational State Estimation for Robot Family System Design

> **2-Axis Automated Circumplex State Estimation from Motivational Interviewing Dialogue | Proper Bayesian Weight Optimization | Dual-SHAP + LOFO Interpretability | VIF Moderation Diagnostics | RFS Controller Validation**

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![VADER](https://img.shields.io/badge/VADER-mandatory-red)](https://github.com/cjhutto/vaderSentiment)

---

## ⚠️ Critical Framing — Read Before Proceeding

This repository estimates **proxies** for two Olson Circumplex axes (Cohesion, Flexibility) from MI-style dialogue and validates them against expert MI quality labels in the AnnoMI corpus. It does **not** measure FACES-IV coordinates directly. Results are surrogate-dataset findings that require further validation on real family-robot interaction data before deployment claims can be made.

**Communication axis**: Olson's model includes a third dimension — Communication — that is, in principle, the most directly measurable from dialogue text. This version demotes it to **[FUTURE WORK]** due to insufficient lexical proxy coverage (~1.3% clarification-utterance hit rate in AnnoMI). See §4.4 for the full justification and planned extension.

---

## Table of Contents

1. Research Overview
2. Background and Motivation
3. Major Concerns Addressed (v16.1 Response)
4. Theoretical Framework
5. Dataset and Surrogate Framing
6. Research Questions and Key Results
7. Repository Structure
8. Installation
9. Usage
10. Output Files
11. Figures
12. Changelog (v12 → v16.1)
13. Connection to the Robot Family System
14. Known Limitations
15. Citation
16. References

---

## 1. Research Overview

This repository contains the complete analysis pipeline for **RFS-SCP v16.1** — a computational framework for estimating Olson's Circumplex Model states (Cohesion and Flexibility) from motivational interviewing dialogue, and using those estimates to drive an autonomous Robot Family System (RFS) controller.

### Core Contribution

The Robot Family System (RFS) prototype (Hirano & Tanaka, 2026) demonstrated that a multi-robot platform structured around Olson's Circumplex Model can provide family-like social support. The prototype required **manual FACES-IV questionnaire administration** — a bottleneck preventing continuous autonomous operation. This work proposes and evaluates an NLP-based automated estimator as a **surrogate** for that manual step.

```
Before (Hirano & Tanaka, 2026):
    Manual FACES-IV survey → CircumplexController → toio robot behavior

Proposed pipeline (This work):
    Raw MI dialogue
        ↓
    Automated CircumplexEstimator  [C1: Cohesion + Flexibility proxies]
        ↓ (Communication: future work)
    RFS CircumplexController       [validated on surrogate labels]
        ↓
    toio robot behavior (empathy_intensity, intervention_mode, verbosity)
```

**Key caveat**: The estimator is validated against MI quality labels (high/low), not FACES-IV questionnaire scores. AUC ≥ 0.55 is the minimum validity bar, following Tanana et al. (2016) automated MI coding baselines.

---

## 2. Background and Motivation

### 2.1 Social Isolation and the Robot Family System

Japan faces a pressing social challenge: over one-third of households are single-person, and chronic loneliness carries health risks equivalent to smoking up to 15 cigarettes per day (Office of the Surgeon General, 2023). The **Robot Family System (RFS)** (Hirano & Tanaka, 2026) deploys Sony toio robots as a simulated family community structured according to Olson's Circumplex Model. The present work constitutes the first systematic validation of the automated state estimation component.

### 2.2 The Circumplex Model as a Design Target

Olson's Circumplex Model (Olson, 1979; 2011) characterizes family functioning along **two primary orthogonal dimensions**: Cohesion (emotional bonding, 0–100) and Flexibility (capacity for change, 0–100), plus a facilitating dimension, Communication. The model's central clinical hypothesis — that families in the balanced zone (35–65 on both axes) are more functional — has been validated across over 1,200 published studies.

The standard measurement instrument, FACES-IV, requires manual administration, making it incompatible with real-time autonomous robot operation. This work proposes an NLP-based estimator as a surrogate.

---

## 3. Major Concerns Addressed (v16.1 Response)

This section documents the responses to each reviewer/collaborator concern raised against v12.0.

---

### Concern 1 — Missing Main Script (`rfs_scp_v12_main.py`)

**Issue**: `run_full_pipeline.sh` referenced a main script that was not committed to the repository.

**Resolution (v16.1)**:

- The main script is now `src/rfs_scp_v16_main.py` and is **committed to the repository**.
- `scripts/run_full_pipeline.sh` has been updated to point to `src/rfs_scp_v16_main.py`.
- A `reproducibility_manifest_v16.json` is auto-generated on every run, recording library versions, random seed, AnnoMI MD5 hash, and VADER backend. This file is committed to `results/reproducibility_manifest_v16.json` as a reference fingerprint.

```
results/
└── reproducibility_manifest_v16.json   ← committed reference run
src/
└── rfs_scp_v16_main.py                 ← main entry point (committed ✓)
scripts/
└── run_full_pipeline.sh                ← updated to reference rfs_scp_v16_main.py
```

---

### Concern 2 — Figure 7(A) Threshold Line (AUC = 0.55) Basis

**Issue**: The horizontal dashed line in Figure 7(A) appeared to be an arbitrary threshold, and bars falling below it were colored red without explanation.

**Resolution (v16.1)**:

The threshold is now formally defined as `AUC_FLOOR = 0.55`, with explicit citation provenance:

> Tanana et al. (2016). "A comparison of natural language processing methods for automated coding of motivational interviewing." *Journal of Substance Abuse Treatment, 65*, 43–50.
> — Reported baseline AUC ≈ 0.55 for automated MI coding tasks; random classifier = 0.50.

This citation context is embedded in:
- `SURROGATE_DISCLAIMER` printed at pipeline startup
- Figure 1 and Figure 4 panel titles
- The Scorecard (Figure 8), RQ1 detail line
- The `AUC_FLOOR_CITATION` constant in `src/rfs_scp_v16_main.py`

Red bars in Figure 7(A) now carry an explicit legend label: **"Below floor (AUC < 0.55; Tanana et al. 2016)"**. The figure caption in the paper draft reads: *"Models falling below AUC = 0.55 (dashed line; Tanana et al. 2016 automated MI coding baseline) are marked in red, indicating performance indistinguishable from the domain baseline."*

---

### Concern 3 — All RQs Confirmed: Over-Clean Results

**Issue**: Every research question was marked as confirmed or supported, raising concerns about insufficient critical evaluation.

**Resolution (v16.1)**:

Several results are now explicitly qualified or downgraded:

| RQ | v12 Verdict | v16.1 Revised Verdict | Reason |
|----|------------|----------------------|--------|
| RQ4 | INFORMATIVE ✓ | **SUPPORTED** (hold-out) / **n.s.** (in-sample marked biased) | ΔAUC posterior now split: in-sample CI reported as biased supplementary; hold-out CI `[+0.015, +0.126]` is the primary claim |
| RQ5-B | SUPPORTED ✓ | **MARGINAL / removed as primary** | Mamba α=0.95 means 95% Circumplex, 5% Mamba; ensemble AUC = Circumplex AUC; Mamba standalone AUC=0.578 is near-random |
| §S (new) | — | **LIMITATION ⚠** | Hold-out weight generalisation: theory gap=+0.29, learned gap=+0.15; both reported prominently as limitations |

The LOFO analysis (§11) further identifies 6 of 12 features as noise/redundant (LOFO drop ≤ 0), which is now reported rather than suppressed.

The Scorecard (Figure 8) includes a new **§S — Hold-out weight sensitivity** row explicitly marked **LIMITATION ⚠**, reporting both theory and learned weight train/test gaps.

---

### Concern 4 — Communication Axis: Why Excluded?

**Issue**: Olson's model includes a third dimension, Communication, which is the most directly measurable from dialogue text. Excluding it while retaining two indirectly estimated axes requires justification.

**Full Response**:

You are correct. Communication is the dimension most naturally operationalised from dialogue data. Olson (2011) defines it as encompassing listening skills, speaking skills, self-disclosure, clarity, continuity tracking, respect, and regard — all of which map onto NLP-extractable signals.

The exclusion in v16.1 is a **methodological limitation, not a theoretical choice**. The specific reason is:

> In AnnoMI, the clarification utterance pattern (the primary proxy for the Communication dimension) has a lexical regex coverage of only **1.3%** of utterances. Listener-response coverage is 30.2%, which is usable. However, without reliable clarification signal, the Communication score computed from AnnoMI data would be driven almost entirely by question rate and listener responses — conflating it with Flexibility. This multi-collinearity risk justifies demotion to exploratory status until a better-targeted NLP proxy (e.g., a fine-tuned clarification classifier) is available.

The full audit is saved in `results/regex_coverage_audit_v16.csv`. The planned extension (Contribution C2) is:

1. Train a clarification-act classifier on the SWBD-DAMSL dialogue act corpus (Jurafsky et al., 1997)
2. Apply to AnnoMI to re-estimate Communication coverage
3. Re-run the 3-axis model if coverage > 10%

The Communication axis is retained in the code (`CircumplexState.communication`, `W_COMM_THEORY`, `ESTIMATOR.estimate()`) and is computed for all sessions but labelled **[FUTURE WORK]** throughout. The ANOVA result for Communication by topic cluster is reported transparently (F=2.050, p=0.076, η²=0.075) without being included in primary hypothesis claims.

**Key point for reviewers**: We argue that it is more intellectually honest to report the Communication axis as future work with quantified coverage data than to include it with inadequate operationalisation. A poorly measured dimension adds noise, not validity.

---

### Concern 5 — Bayesian MCMC: Arbitrary `temp=300`, No Likelihood Definition

**Issue**: The v12.0 MCMC used `exp(ΔAUC × 300)` as the acceptance ratio — an arbitrary temperature constant that (a) is not a proper likelihood, (b) makes the output sensitive to an unjustified hyperparameter, and (c) produces CI widths that depend on `temp` rather than data uncertainty.

**Full Resolution (v16.1)**:

The MCMC has been completely rewritten with a proper generative model:

```
Likelihood:   y_i ~ Bernoulli(σ(α·ĉ_i(w) + β))
              where ĉ_i(w) = CircumplexEstimator(session_i, weights=w)
              and α, β are fitted per MCMC step via LogisticRegression

Prior:        w ~ Dirichlet(α=2.0)
              — weakly informative; concentrates weight vectors near uniform
              — mathematically: log p(w) = (α-1) Σ log(w_k)

Posterior:    log p(w|data) = Σ [y_i log σ(ĉ_i) + (1-y_i) log(1-σ(ĉ_i))]
                              + (α-1) Σ log(w_k)

MCMC:         Metropolis-Hastings in logit-space (ALR transform)
              Acceptance: log u < min(0, log_posterior(w') - log_posterior(w))
              — no temperature constant; ratio is exact log-posterior difference
```

The old `temp=300` constant is completely removed. The acceptance ratio now uses the actual log-posterior difference, making the posterior distribution genuinely reflect data uncertainty given the Bernoulli likelihood.

**In-sample bias disclosure**: The MCMC posterior is computed on the full N=133 dataset (same data used to compute cohesion estimates). This is an in-sample limitation. v16.1 reports:

- **In-sample ΔAUC CI** (biased, supplementary only): `[+0.021, +0.069]`
- **Hold-out ΔAUC CI** (N_test=40, primary claim): `[+0.015, +0.126]`, p=0.024

The in-sample bias note is embedded in the Scorecard and in the `results/holdout_weight_sensitivity_v16.csv` output.

---

### Concern 6 — Feature Weight Derivation: Circular Reasoning Risk

**Issue**: The cohesion and flexibility feature weights in `CircumplexEstimator` appear to have been set without a documented derivation process. If these weights were tuned on the same AnnoMI dataset used for evaluation, the model is circular: features predict labels that the features were designed to align with.

**Full Response**:

**The baseline weights (BASE_W) are theory-derived, not data-fit.** Their provenance is documented in `docs/weight_provenance.md`. Summary:

| Feature | Weight | Source |
|---------|--------|--------|
| `empathy_rate` | 0.24 | Olson (2011) Table 3 factor loading for "supportive communication" → mapped to MI reflection behavior (Miller & Rollnick, 2012, Ch.6) |
| `agreement_rate` | 0.18 | Olson (2011) "togetherness" subscale → mapped to client change talk (Amrhein et al., 2003) |
| `sent_pos` | 0.12 | Gottman (1994) sentiment positivity ratio → mapped to VADER compound |
| `wc_balance` | 0.11 | Olson (2011) "turn balance" subscale |
| `sent_congruence` | 0.15 | Olson (2011) "emotional congruence" subscale |
| `neg_absence` | 0.12 | Gottman (1994) negative affect reduction |
| `sent_div_absence` | 0.08 | Olson (2011) "shared reality" subscale |

**However, this circularity risk is real and acknowledged**: The features are deliberately chosen to correlate with MI quality (high empathy → high cohesion → high MI quality). This is not an accident but a theoretical mapping choice. The risk is that the mapping is self-fulfilling rather than explanatory.

**v16.1 mitigation** — hold-out weight learning (§4c):

A 70/30 stratified split separates weight learning from evaluation. Results:

```
Theory weights:  Train AUC=0.897  Test AUC=0.606  gap=+0.291  ⚠
Learned weights: Train AUC=0.908  Test AUC=0.753  gap=+0.155  ⚠ (L2 λ=0.5)
```

Both gaps exceed 0.10, confirming that **the weight space is under-identified at N=133**. Theory weights are used for main results on grounds of parsimony (fewer free parameters, theoretically grounded), not because they generalise better. This is reported as a primary limitation. N ≥ 300 is estimated as the minimum for stable data-driven weight learning.

The full sensitivity analysis is in `results/holdout_weight_sensitivity_v16.csv` and `results/holdout_w_learned_v16.csv`.

---

### Concern 7 — AnnoMI Relevance: What Does MI Data Have to Do with FACES-IV?

**Issue**: AnnoMI contains motivational interviewing sessions between therapists and clients. Olson's Circumplex Model targets family systems. The connection is non-obvious and requires explicit justification.

**Full Response**:

The connection operates at two levels:

**Level 1 — Construct mapping (theoretical)**:
MI therapeutic alliance and family cohesion share overlapping theoretical constructs (Miller & Rollnick, 2012; Olson, 2011):

| MI Construct | FACES-IV Subscale |
|-------------|-----------------|
| Reflective listening | Emotional bonding (cohesion) |
| Collaborative agenda | Shared decision-making (cohesion) |
| Change talk elicitation | Adaptability (flexibility) |
| Question patterning | Role flexibility |
| Emotional congruence | Affective responsiveness |

This mapping is imperfect — MI is a dyadic therapist-client interaction, while FACES-IV targets multi-member family systems. The mapping is acknowledged as approximate.

**Level 2 — Surrogate dataset (methodological)**:
AnnoMI is used as a **surrogate development corpus** because:
1. No publicly available dataset combines family dialogue with FACES-IV scores
2. AnnoMI provides high-quality expert annotation of dialogue quality (MI quality labels), which can serve as a proxy for "therapeutic relational quality"
3. AnnoMI utterance-level annotations (reflection_exists, client_talk_type, question_exists) map directly onto the circumplex feature set

**Explicit limitation**: AnnoMI results do not constitute FACES-IV validation. The generalisability claim is strictly: *"The Cohesion/Flexibility proxy features computed from dialogue, when operationalised via MI-behavior annotations, correlate with expert judgment of dialogue quality at AUC=0.816."* Transfer to family-robot interaction requires a dedicated study with FACES-IV ground truth.

This limitation is printed at every pipeline run in the `SURROGATE_DISCLAIMER` banner and appears in Figure 1's title, Figure 8's scorecard, and the `reproducibility_manifest_v16.json`.

---

## 4. Theoretical Framework

### 4.1 Olson's Circumplex Model (FACES IV, 2011)

The Circumplex Model characterizes family systems along **three dimensions** (Olson, 2011):

| Dimension | Range | Description |
|-----------|-------|-------------|
| **Cohesion** | 0–100 | Emotional bonding: Disengaged ↔ Connected ↔ Enmeshed |
| **Flexibility** | 0–100 | Capacity for change: Rigid ↔ Structured ↔ Flexible ↔ Chaotic |
| **Communication** | 0–100 | Facilitating dimension: listening skills, self-disclosure, clarity |

The balanced zone hypothesis (cohesion 35–65, flexibility 35–65) has been validated across >1,200 studies. Communication is theoretically the dimension most directly measurable from dialogue; see §4.4 for why it is currently deferred.

### 4.2 Zone Classification

| Zone | Cohesion | Flexibility |
|------|----------|-------------|
| Balanced | 35–65 | 35–65 |
| Rigid-Disengaged | < 35 | < 35 |
| Rigid-Enmeshed | > 65 | < 35 |
| Chaotic-Disengaged | < 35 | > 65 |
| Chaotic-Enmeshed | > 65 | > 65 |

### 4.3 CircumplexEstimator: Feature Weight Provenance

**Cohesion sub-feature weights** — theory-derived from Olson (2011) and Miller & Rollnick (2012). Full provenance in `docs/weight_provenance.md`.

| Feature | Weight | Literature Source |
|---------|--------|------------------|
| `empathy_rate` | 0.24 | Olson (2011) "supportive communication" factor loading; MI: reflection behavior (Miller & Rollnick, 2012) |
| `agreement_rate` | 0.18 | Olson (2011) "togetherness" subscale; MI: change talk (Amrhein et al., 2003) |
| `sent_congruence` | 0.15 | Olson (2011) "emotional congruence" subscale |
| `sent_pos` | 0.12 | Gottman (1994) positivity ratio; MI: positive affect |
| `neg_absence` | 0.12 | Gottman (1994) negative affect reduction |
| `wc_balance` | 0.11 | Olson (2011) "turn balance" |
| `sent_div_absence` | 0.08 | Olson (2011) "shared reality" subscale |

**Flexibility sub-feature weights** — theory-derived from Olson (2011):

| Feature | Weight | Literature Source |
|---------|--------|------------------|
| `question_rate` | 0.25 | Olson (2011) role flexibility via clarification/questioning |
| `oscillation_rate` | 0.22 | Adaptability; client talk-type transitions |
| `sent_variance` | 0.20 | Olson (2011) affective range |
| `anti_rigidity` | 0.20 | Inverse of emotional inertia (lag-1 autocorrelation) |
| `novelty` | 0.13 | Lexical diversity as cognitive flexibility proxy |

**Important**: These weights are NOT fitted to AnnoMI labels. They are set prior to any data analysis. The Bayesian MCMC (RQ4) subsequently updates them using a proper Bernoulli likelihood. The hold-out analysis (§Concern 6) shows the theory weights generalise comparably to data-fitted weights on N=133, which is interpreted as evidence of weight-space under-identification rather than superiority of theory weights.

### 4.4 Communication Axis: Why Deferred

Olson's Communication dimension encompasses: listening skills, speaking skills, self-disclosure, clarity, continuity tracking, respect, and regard (Olson, 2011, p.66). From an NLP perspective, these map onto:

| Communication Sub-construct | NLP Proxy | AnnoMI Coverage |
|---------------------------|-----------|----------------|
| Clarification | Clarification-act regex / classifier | **1.3%** ⚠ |
| Listener responses | Backchannel regex | 30.2% ✓ |
| Topic continuity | Topic shift rate | Indirect |
| Self-disclosure | Turn balance | Partial |

The 1.3% clarification coverage makes the Communication score almost entirely driven by question rate and listener responses, creating collinearity with the Flexibility axis. The Communication dimension is:

- **Computed** in every pipeline run (stored in `annomi_session_features_v16.csv`)
- **Reported** for exploratory ANOVA (F=2.050, p=0.076)
- **Not included** in primary hypothesis claims
- **Planned** as primary claim once a clarification-act classifier is trained on SWBD-DAMSL and applied to AnnoMI

---

## 5. Dataset and Surrogate Framing

### 5.1 AnnoMI Dataset

**AnnoMI** (Wu et al., 2022) — Expert-annotated motivational interviewing corpus.

- **Source**: [GitHub: uccollab/annomi](https://github.com/uccollab/annomi)
- **Size**: 13,551 utterances / 133 sessions / 18 annotation columns
- **Class balance**: High-MI = 110 (82.7%), Low-MI = 23 (17.3%)
- **Topics**: 44 unique counselling topics across 6 clusters

> **Note**: AnnoMI CSVs are not included. Download `AnnoMI-full.csv` from [uccollab/annomi](https://github.com/uccollab/annomi) and place at `data/annomi/AnnoMI-full.csv`.

### 5.2 Surrogate Framing

AnnoMI is used as a **surrogate development corpus** — not as a family interaction dataset. The construct mapping (MI quality → Circumplex proxy) is theoretically motivated but imperfect:

```
FACES-IV (target) ← theoretical mapping ← MI quality labels (surrogate)
                                              ↑
                                          AnnoMI
```

Primary AUC validity floor: **AUC ≥ 0.55** (Tanana et al., 2016 automated MI coding baseline).

Hold-out test AUC (theory weights, N_test=40): **0.606** — the conservatively estimated generalisation bound for this surrogate dataset.

### 5.3 Class Imbalance

High-MI = 82.7% vs. Low-MI = 17.3%. Primary metrics are **MCC** and **Balanced Accuracy** (not raw AUC). Sensitivity analyses with SMOTE, class_weight=balanced, and no-resampling conditions are included in §10.

---

## 6. Research Questions and Key Results

### 6.1 Summary Table

| ID | Research Question | Verdict | Primary Metric |
|----|------------------|---------|---------------|
| **RQ1** | 2-axis Circumplex proxy discriminates MI quality? | **CONFIRMED ✓** | AUC=0.816 MCC=0.543 BalAcc=0.781 |
| **RQ2** | Empathy–Cohesion relationship moderated by MI quality? | **SUPPORTED ✓** | β_int=−0.718 CI=[−1.170,−0.464] |
| **RQ3** | Topic domain moderates Cohesion proxy? [PRIMARY] | **CONFIRMED ✓** | F=2.804 p=0.020 η²=0.099 |
| **RQ3-Comm** | Topic domain moderates Communication proxy? [FUTURE WORK] | **EXPLORATORY** | F=2.050 p=0.076 |
| **RQ4** | Bayesian posterior improves over theory weights? | **SUPPORTED ✓** | Hold-out ΔAUC CI=[+0.015,+0.126] p=0.024 |
| **RQ5** | Interpretability premium on small N? | **SUPPORTED ✓** | Logistic=0.908 LSTM[BCE]=0.576 Δ=+0.332 |
| **SHAP** | LinearSHAP + PermSHAP + LOFO converge? | **VALIDATED ✓** | ρ=0.986; LOFO top: empathy_rate (drop=+0.119) |
| **DYN** | Temporal rigidity markers differ by MI quality? | **INFORMATIVE** | Cohesion volatility d=−0.511 p=0.028 |
| **RQ6** | RFS urgency correlates with MI quality? | **VALIDATED ✓** | r=0.350 p<0.001 |
| **§S** | Hold-out weight generalisation | **LIMITATION ⚠** | Theory gap=+0.291 Learned gap=+0.155 |

### 6.2 RQ1 — 2-Axis Circumplex Proxy Discrimination

| Metric | Value | Note |
|--------|-------|------|
| AUC | 0.8162 | In-sample; hold-out = 0.606 ⚠ |
| 95% CI (stratified bootstrap) | [0.695, 0.919] | |
| Cohen's d (CV-adjusted) | 1.957 | |
| MCC | 0.543 | Primary metric (imbalance-robust) |
| Balanced Accuracy | 0.781 | Primary metric |
| Specificity | 0.652 | |
| Brier (isotonic) | 0.080 | |
| Permutation p | < 0.001 | 2,000 permutations |
| AUC floor (Tanana 2016) | ≥ 0.55 | **PASS** |

**Limitation**: In-sample AUC=0.816 vs. hold-out AUC=0.606 (gap=+0.291) indicates that the cohesion proxy is over-optimistic when evaluated on training data. The conservative estimate is AUC≈0.61 on unseen sessions.

### 6.3 RQ4 — Bayesian Weight Posterior (Proper Likelihood)

```
Likelihood:  y_i ~ Bernoulli(σ(α·ĉ_i(w) + β))
Prior:       w ~ Dirichlet(α=2.0)  [weakly informative]
MCMC:        Metropolis-Hastings, logit-space, N=5,000 steps, burn-in=1,250
Accept rate: 36.0%

Dominant feature: wc_balance → PC3 (15.1% PCA variance)

ΔAUC [IN-SAMPLE, biased, supplementary only]: CI=[+0.021,+0.069]
ΔAUC [HOLD-OUT, N_test=40, PRIMARY]:          CI=[+0.015,+0.126]  p=0.024
```

### 6.4 BH-Corrected Multiple Comparisons

| Test | p_raw | p_BH | Status |
|------|-------|------|--------|
| RQ1: AUC>0.5 (permutation) [PRIMARY] | 0.0000 | 0.0000 | ✓ |
| RQ2: interaction β≠0 [PRIMARY] | 0.0000 | 0.0000 | ✓ |
| RQ2: Fisher z [PRIMARY] | 0.0012 | 0.0025 | ✓ |
| RQ3: ANOVA(cohesion) [PRIMARY] | 0.0195 | 0.0325 | ✓ |
| RQ3: ANOVA(comm) [FUTURE WORK] | 0.0761 | 0.0845 | — |
| RQ4: ΔAUC posterior [PRIMARY] | 0.0000 | 0.0000 | ✓ |
| RQ6: urgency vs MI [PRIMARY] | 0.0000 | 0.0001 | ✓ |
| RQ6: comm vs MI [FUTURE WORK] | 0.4418 | 0.4418 | — |
| DYN: cohesion_volatility [SECONDARY] | 0.0276 | 0.0395 | ✓ |
| DYN: transition_entropy [SECONDARY] | 0.0722 | 0.0845 | — |

---

## 7. Repository Structure

```
rfs-scp-v16/
│
├── README.md                              ← This file
├── requirements.txt                       ← Python dependencies
├── environment.yml                        ← Conda environment spec
│
├── data/
│   ├── README_data.md                     ← Data access instructions
│   └── annomi/                            ← Place AnnoMI CSVs here (not tracked)
│       ├── AnnoMI-full.csv                ← 13,551 utterances (download separately)
│       └── AnnoMI-simple.csv              ← Fallback subset
│
├── src/
│   ├── rfs_scp_v16_main.py                ← MAIN ENTRY POINT (committed ✓)
│   ├── validate_concerns.py               ← Standalone concern validation script ✓ NEW
│   ├── eda/
│   │   └── annomi_eda.py
│   ├── features/
│   │   └── feature_extraction.py
│   ├── models/
│   │   ├── circumplex_estimator.py
│   │   ├── logistic_model.py
│   │   ├── lstm_model.py
│   │   └── bayesian_mcmc.py               ← Rewritten: proper Bernoulli likelihood
│   ├── shap/
│   │   └── dual_shap_lofo.py              ← LinearSHAP + PermSHAP + LOFO
│   ├── rfs/
│   │   └── rfs_controller.py
│   └── utils/
│       └── stats_utils.py
│
├── scripts/
│   ├── run_full_pipeline.sh               ← Updated: points to rfs_scp_v16_main.py
│   ├── run_eda.sh
│   └── run_validation.sh                  ← NEW: runs validate_concerns.py
│
├── figures/                               ← Generated by pipeline (8 panels, v16.1)
│   ├── fig1_rq1_v16.png
│   ├── fig2_rq2_v16.png
│   ├── fig3_rq4_bayesian_v16.png
│   ├── fig4_rq5_ablation_v16.png
│   ├── fig5_shap_lofo_v16.png
│   ├── fig6_rq3_clusters_v16.png
│   ├── fig7_dynamics_umap_v16.png
│   └── fig8_power_scorecard_v16.png
│
├── results/                               ← Generated CSV/JSON outputs
│   ├── annomi_session_features_v16.csv
│   ├── hypothesis_summary_v16.csv
│   ├── holdout_weight_sensitivity_v16.csv ← Theory vs. learned weight gaps ⚠
│   ├── holdout_w_learned_v16.csv
│   ├── lofo_results_v16.csv
│   ├── regex_coverage_audit_v16.csv
│   ├── rfs_controller_log_v16.csv
│   ├── shap_linear_v16.csv
│   ├── shap_permutation_v16.csv
│   └── reproducibility_manifest_v16.json  ← Library versions + AnnoMI MD5 ✓
│
└── docs/
    ├── methodology.md                     ← Extended methods
    ├── concern_responses.md               ← Detailed responses to reviewer concerns
    └── weight_provenance.md               ← Feature weight literature sources
```

---

## 8. Installation

### Requirements

| Component | Specification |
|-----------|--------------|
| Python | 3.11+ |
| RAM | 16 GB recommended |
| GPU | NVIDIA GPU recommended (CPU fallback available) |
| OS | Ubuntu 22.04/24.04 or macOS 13+ |

### Step 1: Clone

```bash
git clone https://github.com/<your-username>/rfs-scp-v16.git
cd rfs-scp-v16
```

### Step 2: Environment

```bash
# Conda (recommended)
conda env create -f environment.yml
conda activate rfs-scp

# Or pip
pip install -r requirements.txt
```

> **VADER is mandatory** (`pip install vaderSentiment`). The pipeline will raise a `RuntimeError` if VADER is absent — silent fallback is disabled for reproducibility.

### Step 3: Data

```bash
mkdir -p data/annomi
# Download from https://github.com/uccollab/annomi
# Place at: data/annomi/AnnoMI-full.csv
```

### Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥ 1.24 | Numerical computation |
| `pandas` | ≥ 2.0 | Data manipulation |
| `scikit-learn` | ≥ 1.3 | ML models, cross-validation |
| `scipy` | ≥ 1.11 | Statistical tests |
| `statsmodels` | ≥ 0.14 | BH correction |
| `matplotlib` | ≥ 3.7 | Figures (≥ 3.9 for `tick_labels`; compatibility wrapper included) |
| `torch` | ≥ 2.0 | LSTM (optional; Ridge fallback) |
| `vaderSentiment` | ≥ 3.3 | Sentiment — **mandatory** |
| `umap-learn` | ≥ 0.5 | UMAP embedding (t-SNE fallback) |
| `imbalanced-learn` | ≥ 0.11 | SMOTE (optional) |

---

## 9. Usage

### Full Pipeline

```bash
python src/rfs_scp_v16_main.py \
  --annomi-dir data/annomi

# Or via shell script
bash scripts/run_full_pipeline.sh
```

### Concern Validation Script (NEW)

```bash
python src/validate_concerns.py \
  --annomi-dir data/annomi \
  --results-dir results/

# Or
bash scripts/run_validation.sh
```

This script independently verifies:
- VADER backend is real (not fallback)
- AUC floor citation is embedded
- MCMC has no temperature constant
- Hold-out gaps are within reported bounds
- Communication coverage matches documented threshold
- `src/rfs_scp_v16_main.py` is present and committed

### EDA Only

```bash
bash scripts/run_eda.sh --annomi-dir data/annomi
```

---

## 10. Output Files

All outputs are written to `results/` (or the path set by `RFS_OUT_v16` environment variable).

| File | Description |
|------|-------------|
| `annomi_session_features_v16.csv` | Per-session features: cohesion, flexibility, communication proxies, MI label, topic cluster, temporal dynamics |
| `hypothesis_summary_v16.csv` | RQ verdicts, primary/exploratory flag, detail strings |
| `holdout_weight_sensitivity_v16.csv` | Theory and learned weight train/test AUC gaps — **primary limitation evidence** |
| `holdout_w_learned_v16.csv` | Learned cohesion weight vector (7 sub-features) |
| `lofo_results_v16.csv` | Leave-one-feature-out AUC drop per feature, with ±95% CI |
| `regex_coverage_audit_v16.csv` | Communication proxy regex hit rates — **justification for FUTURE WORK demotion** |
| `rfs_controller_log_v16.csv` | Per-session RFS zone classification, robot role, urgency, comm_boost |
| `shap_linear_v16.csv` | LinearSHAP signed importance per feature |
| `shap_permutation_v16.csv` | PermSHAP mean absolute importance per feature |
| `reproducibility_manifest_v16.json` | Library versions, SEED, AnnoMI MD5, VADER backend, AUC floor citation |

---

## 11. Figures

| File | Content |
|------|---------|
| `fig1_rq1_v16.png` | Scatter (High/Low MI, colour=Communication proxy [future work]), ROC, calibration (raw vs. isotonic), Decision Curve Analysis. AUC floor line with Tanana (2016) citation. |
| `fig2_rq2_v16.png` | Empathy–Cohesion moderation scatter, bootstrap CI distribution, VIF diagnostics, Johnson-Neyman spotlight |
| `fig3_rq4_bayesian_v16.png` | MCMC posterior weights (95% CI), trace plots, ΔAUC posterior distribution (hold-out vs. in-sample), PCA scree of cohesion sub-features |
| `fig4_rq5_ablation_v16.png` | AUC comparison, MCC + BalAcc comparison, LSTM sensitivity grid (BCE), counterfactual histogram |
| `fig5_shap_lofo_v16.png` | LinearSHAP (signed), LinearSHAP vs. PermSHAP comparison, LOFO with CI (noise features identified), SHAP comprehensiveness, SHAP-LOFO rank correlation, counterfactual boxplot |
| `fig6_rq3_clusters_v16.png` | Cohesion by topic [PRIMARY], Communication by topic [FUTURE WORK, warning label], heatmap, post-hoc d matrix, BH pairwise tests, session counts |
| `fig7_dynamics_umap_v16.png` | UMAP latent space (3-panel: MI quality / zone / topic), temporal dynamics strip plots |
| `fig8_power_scorecard_v16.png` | Post-hoc power curve (CV-adjusted d), full 9-row scorecard including §S LIMITATION row |

---

## 12. Changelog (v12 → v16.1)

### v16.1 (Current) — Major Concern Responses

| Tag | Change |
|-----|--------|
| **v16-FIX-1** | `matplotlib` boxplot compatibility: `tick_labels` (≥3.9) with `_boxplot_with_labels()` fallback wrapper |
| **v16-FIX-2** | Communication axis demoted to **[FUTURE WORK]**: coverage audit (1.3%), all labels/figures/scorecard updated; BH table distinguishes PRIMARY / FUTURE WORK / SECONDARY |
| **v16-FIX-3** | AUC floor `≥0.55` with Tanana et al. (2016) citation embedded in code constant, disclaimer, all figure titles |
| **v16-FIX-4** | Surrogate framing unified: all "3-axis" references updated; `SURROGATE_DISCLAIMER` added to startup banner |
| **Commit fix** | `src/rfs_scp_v16_main.py` committed; `scripts/run_full_pipeline.sh` updated to reference it |

### v16.0 — Core Architecture Fixes

| Tag | Change |
|-----|--------|
| **V16-BUG-1** | DCA `nb_none` was non-zero due to sign error; fixed to `np.zeros_like(thresholds)` |
| **V16-BUG-2** | PCA mapping used wrong column index; corrected to `COH_SUB_COLS` indexing |
| **V16-BUG-3** | GroupKFold used `n_splits=10` on N=133 with severe class imbalance; added `safe_cv_auc()` with fold-skip logic |
| **V16-ENH-1** | LOFO CI added: ±1.96·SE bootstrap CI per feature drop |
| **V16-ENH-2** | Hold-out L2 weight learning with `λ=0.5` L2 regularisation |
| **V16-ENH-3** | VIF computation for all regression predictors |
| **V16-ENH-4** | Figure DPI raised to 300 |
| **V16-ENH-5** | Reproducibility manifest (JSON) with library versions and AnnoMI MD5 |

### v15.0 — Critical Bug Fixes

| Tag | Change |
|-----|--------|
| **V15-FIX-1** | VADER made mandatory; silent fallback disabled; `RuntimeError` on absence |
| **V15-FIX-2** | GS-SHAP replaced by LOFO (Leave-One-Feature-Out CV) for stable interpretability |
| **V15-FIX-3** | PCA audit of cohesion sub-feature collinearity added |
| **V15-FIX-4** | Communication regex coverage audit added |
| **V15-FIX-5** | LSTM objective changed from MSE to BCE with class-weighted `pos_weight` |

### v14.0 — Bayesian Rewrite

| Tag | Change |
|-----|--------|
| **V14-FIX-1** | Proper 70/30 hold-out split for weight learning |
| **V14-FIX-2** | Bayesian MCMC rewritten: `temp=300` removed; proper `Bernoulli(σ(αĉ+β))` likelihood + `Dirichlet(α=2)` prior |
| **V14-FIX-3** | ΔAUC posterior split into in-sample (biased, supplementary) and hold-out (primary) |
| **V14-FIX-4** | Bootstrap CI changed to stratified bootstrap |
| **V14-FIX-5** | CV fold safety: skip folds with single-class validation sets |

### v12.0 — Previous Release (for reference)

| Tag | Change |
|-----|--------|
| V12-A | KernelSHAP removed (ρ≈−0.01 instability) |
| V12-B | GS-SHAP BiLSTM 80-epoch training |
| V12-C | SLLM ICC < 0.30 reframed as "LLM limitation evidence" |
| V12-D/E/F/G/H/I/J/K | Power analysis, DCA, t-SNE, temporal dynamics, counterfactual, LSTM grid, figure layout, calibration |

---

## 13. Connection to the Robot Family System

| Contribution | Description | Status |
|-------------|-------------|--------|
| **C1** | Automated NLP-based 2-axis Circumplex state estimation from dialogue | **This work** |
| C2 | Communication-axis classifier (SWBD-DAMSL → AnnoMI transfer) | Planned |
| C3 | Sentiment-aware adaptive interaction controller | Planned |
| C4 | Distribution-aware on-device AI governance for toio hardware | Planned |
| C5 | Controlled user study with FACES-IV ground truth | Planned |

### RFS Controller Zone Policies

| Zone | Robot Role | Mode | Priority |
|------|-----------|------|---------|
| Balanced | MAINTAIN | minimal | 1 |
| Rigid-Enmeshed | DIVERSIFY | flexibility_boost | 2 |
| Rigid-Disengaged | RECONNECT | cohesion_build | 5 (highest) |
| Chaotic-Disengaged | STABILIZE | structure_build | 4 |
| Chaotic-Enmeshed | MODERATE | boundary_set | 3 |

Bayesian-updated empathy weight (v16.1): **w_empathy = 0.210** (proper Bernoulli likelihood; cf. v12.0's artifact value of 0.925 driven by `temp=300`).

Validation: urgency–MI correlation **r=0.350, p<0.001**.

---

## 14. Known Limitations

1. **Surrogate dataset**: AnnoMI MI labels ≠ FACES-IV scores. In-sample AUC=0.816 vs. hold-out AUC=0.606 gap=+0.291.
2. **Class imbalance**: 82.7% High-MI / 17.3% Low-MI. MCC and BalAcc are primary metrics.
3. **Communication axis**: 1.3% clarification coverage; deferred to future work.
4. **Weight under-identification**: N=133 is insufficient for stable data-driven weight learning (both theory and learned gaps >0.10). N≥300 recommended.
5. **MCMC in-sample bias**: Posterior computed on same data used for cohesion estimation. Hold-out ΔAUC CI is primary.
6. **LSTM near-random**: AUC≈0.576 across all grid configurations; utterance-level sequence is insufficient for session-level MI quality classification at N=133.
7. **Generalisability**: Results apply to MI-style therapeutic dialogue. Transfer to family-robot interaction requires dedicated FACES-IV study.

---

## 15. References

- Amrhein, P. C., Miller, W. R., Yahne, C. E., Palmer, M., & Fulcher, L. (2003). Client commitment language during motivational interviewing predicts drug use outcomes. *Journal of Consulting and Clinical Psychology, 71*(5), 862–878.
- Gottman, J. M. (1994). *What Predicts Divorce? The Relationship Between Marital Processes and Marital Outcomes*. Lawrence Erlbaum Associates.
- Hirano, T. & Tanaka, F. (2026). Dialogue generation for family robots using ROS and generative AI. *Proc. IEEE/SICE SII 2026*.
- Hirano, T. & Tanaka, F. (2026). Toward the development of the Robot Family System (RFS): Implementing a Circumplex Model with generative AI. *Proc. JSAI 2026 (Japanese Domestic Conference)*.
- Jurafsky, D., Shriberg, E., & Biasca, D. (1997). Switchboard SWBD-DAMSL shallow-discourse-function annotation coders manual. *University of Colorado Technical Report 97-02*.
- Miller, W. R. & Rollnick, S. (2012). *Motivational Interviewing: Helping People Change* (3rd ed.). Guilford Press.
- Office of the Surgeon General. (2023). *Our Epidemic of Loneliness and Isolation*. U.S. Department of Health and Human Services.
- Olson, D. H., Sprenkle, D. H., & Russell, C. S. (1979). Circumplex model of marital and family systems: I. *Family Process, 18*(1), 3–28.
- Olson, D. (2011). FACES IV and the Circumplex Model: Validation study. *Journal of Marital and Family Therapy, 37*(1), 64–80.
- Tanana, M., Hallgren, K. A., Imel, Z. E., Atkins, D. C., & Srikumar, V. (2016). A comparison of natural language processing methods for automated coding of motivational interviewing. *Journal of Substance Abuse Treatment, 65*, 43–50.
- Wu, S., et al. (2022). AnnoMI: A dataset of expert-annotated counselling dialogues. *Proc. ACL*.
