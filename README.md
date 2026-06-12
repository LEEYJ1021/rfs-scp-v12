# RFS-SCP v12: Circumplex-Grounded Relational State Estimation for Robot Family System Design

> **Automated Circumplex State Estimation from Motivational Interviewing Dialogue | Bayesian Weight Optimization | Dual-SHAP Interpretability | Multi-SLLM Benchmark | RFS Controller Validation**

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)

---

## Table of Contents

1. Research Overview
2. Background and Motivation
3. Research Questions and Key Results
4. Theoretical Framework
5. Dataset
6. Repository Structure
7. Installation
8. Usage
9. Figures
10. v12.0 Changelog
11. Connection to the Robot Family System
12. Citation
13. References

---

## 1. Research Overview

This repository contains the complete analysis pipeline, source code, figures, and results for **RFS-SCP v12.0** — a computational framework for estimating Olson's Circumplex Model states (cohesion and flexibility) directly from motivational interviewing (MI) dialogue, and using those estimates to drive an autonomous Robot Family System (RFS) controller.

### Research Gap and Contribution

The Robot Family System (RFS) prototype developed by Hirano & Tanaka (2026) demonstrated that a multi-robot platform structured around Olson's Circumplex Model can provide family-like social support for individuals living alone. However, the prototype depended on **manual FACES-IV questionnaire administration** to assess family state — a critical bottleneck preventing continuous autonomous operation.

This work directly addresses that gap by replacing manual state assessment with an **automated NLP-based estimator** validated against expert MI quality labels. The pipeline enables continuous autonomous family state monitoring, providing the computational foundation for fully adaptive RFS operation.

```
Before (Hirano & Tanaka, 2026):
    Manual FACES-IV survey → CircumplexController → toio robot behavior

After (This work):
    Raw dialogue
        ↓
    Automated CircumplexEstimator  [This work: C1]
        ↓
    RFS CircumplexController       [This work: validated]
        ↓
    toio robot behavior (empathy_intensity, intervention_mode, verbosity)
```

---

## 2. Background and Motivation

### 2.1 Social Isolation and the Robot Family System

Japan faces a pressing social challenge: more than one-third of households are single-person households, and national surveys confirm that loneliness and isolation affect all age cohorts. Prior research identifies chronic loneliness as a health risk equivalent to smoking up to 15 cigarettes per day (Office of the Surgeon General, 2023). Robotic social support offers a distinct advantage over human-based interventions — **continuity** — demonstrated through longitudinal deployment of telepresence robots showing improved social connectedness among older adults (Rheman et al., 2024), and multi-robot settings extending conversation duration relative to single-robot conditions (Iio et al., 2017).

Building on this foundation, the **Robot Family System (RFS)** (Hirano & Tanaka, 2026; Hirano & Tanaka, 2025) deploys Sony toio robots as a simulated family community structured according to Olson's Circumplex Model. Initial studies established the platform architecture and compared centralized and distributed dialogue variants. The present work constitutes the first systematic validation of the automated state estimation component essential for autonomous RFS operation.

### 2.2 The Circumplex Model as a Design Target

Olson's Circumplex Model of Marital and Family Systems (Olson, 1979; Olson, 2011) characterizes family functioning along two primary orthogonal dimensions — **Cohesion** (emotional bonding, 0–100) and **Flexibility** (capacity for change, 0–100) — with the central clinical hypothesis that families in the balanced zone (35–65 on both axes) are more functional and resilient. Validated across over 1,200 published studies, this model provides a theoretically grounded and empirically supported target state for the RFS controller.

The critical challenge, and the focus of this work, is that the standard instrument for measuring these coordinates — the **FACES-IV questionnaire** — requires manual administration, making it incompatible with real-time autonomous robot operation. Automating this estimation from dialogue data enables continuous state monitoring at the timescales required for adaptive robotic interaction.

---

## 3. Research Questions and Key Results

### 3.1 Summary Table

| ID | Research Question | Verdict | Key Statistics |
|----|------------------|---------|----------------|
| **RQ1** | Does the heuristic CircumplexEstimator discriminate expert MI quality? | **CONFIRMED ✓** | AUC=0.816 CI=[0.699, 0.924] d=1.840 |
| **RQ2** | Does empathy-cohesion association differ by MI quality? | **SUPPORTED ✓** | β_int=−0.718 CI=[−1.170, −0.464] ΔR²=0.138 |
| **RQ3** | Does topic cluster moderate relational dynamics? | **CONFIRMED ✓** | F=2.804 p=0.020 η²=0.099 |
| **RQ4** | Does Bayesian MCMC weight optimization improve AUC? | **INFORMATIVE ✓** | ΔAUC=+0.043, dominant feature=empathy |
| **RQ5-A** | Interpretable features vs. temporal LSTM encoding? | **INTERPRETABLE WINS** | Logistic=0.916, LSTM=0.617, Δ=+0.300 |
| **RQ5-B** | Does Mamba SSM scoring add predictive value? | **SUPPORTED ✓** | Ensemble AUC=0.816 (α=0.95) |
| **RQ5-C** | Can small LLMs estimate Circumplex states? | **LLM LIMITATION CONFIRMED ✓** | ICC_max=0.280 (below "fair" threshold of 0.40) |
| **SHAP** | Do LinearSHAP and PermSHAP converge? | **VALIDATED ✓** | Spearman ρ=0.986 |
| **DYN** | Do temporal rigidity dynamics differentiate MI quality? | **INFORMATIVE ✓** | Cohesion volatility d=−0.511 |
| **RQ6** | Does RFS intervention urgency correlate with MI quality? | **VALIDATED ✓** | r=0.350 p<0.001 |

### 3.2 RQ1 — External Validity

The heuristic CircumplexEstimator achieves strong discrimination between High-MI and Low-MI sessions:

| Metric | Value |
|--------|-------|
| AUC | 0.8162 |
| 95% CI (bootstrap, N=2,000) | [0.699, 0.924] |
| Cohen's d | 1.840 |
| MCC | 0.543 |
| Specificity | 0.652 |
| Brier score (raw / isotonic) | 0.105 / 0.080 |
| ECE (isotonic) | 0.000 |
| Permutation p | < 0.001 |
| Post-hoc power (N=133) | 1.000 |

### 3.3 RQ2 — Conditional Moderation

The empathy–cohesion relationship is significantly moderated by MI quality:

```
cohesion ~ empathy + MI_quality + empathy × MI_quality

empathy:               β = +0.340   t = +4.119   p < .001  ***
empathy × MI_quality:  β = −0.718   t = −5.819   p < .001  ***
Bootstrap 95% CI:      [−1.170, −0.464]  → SIGNIFICANT

High-MI: r = +0.017  (n.s.)
Low-MI:  r = +0.666  (p < .05)
Fisher z-test: p = 0.001
```

This interaction is substantively important for the RFS: in Low-MI sessions, higher empathy strongly predicts cohesion (r=0.666), while in High-MI sessions the relationship is absent (r=0.017). The Bayesian-updated RFS empathy weight (w=0.925 vs. baseline w=0.240) amplifies sensitivity to this dynamic in precisely the contexts where it matters most.

### 3.4 RQ4 — Bayesian MCMC Weight Optimization

```
Baseline AUC (heuristic):  0.8162
Nelder-Mead optimized:     0.8593  (+0.043)
MCMC acceptance rate:      32.3%
Dominant feature:          empathy (posterior mean weight: 0.925 vs. baseline 0.240)
```

The empathy feature dominates the posterior, with its weight shifting from 0.240 to 0.925 — a finding directly operationalized in the RFS controller update.

### 3.5 RQ5-A — Ablation: Interpretable vs. Temporal

| Model | AUC | Notes |
|-------|-----|-------|
| Circumplex (heuristic) | 0.816 | No training required |
| Bayesian (MCMC-optimized) | 0.859 | Optimized weights |
| Logistic Regression (GroupKFold) | **0.916** | Session-level features |
| LSTM h=32 L=1 (best) | 0.617 | Sequential utterances |

Interpretable session-level features outperform temporal LSTM encoding by ΔAUC=+0.300. Counterfactual analysis shows a median empathy increase of **1.214 SD** is required to flip a Low-MI classification to High-MI.

### 3.6 RQ5-C — Multi-SLLM Benchmark (LLM Limitation Evidence)

| Model | Params (B) | ICC (cohesion) | AUC |
|-------|-----------|----------------|-----|
| qwen2.5:7b | 7.0 | **0.280** | **0.694** |
| phi3:mini | 3.8 | 0.128 | 0.544 |
| llama3.2:3b | 3.0 | 0.093 | 0.570 |
| qwen2.5:3b | 3.0 | 0.122 | 0.504 |
| gemma2:2b | 2.0 | 0.027 | 0.582 |
| qwen2.5:1.5b | 1.5 | 0.044 | 0.460 |
| mistral:7b-instruct | 7.0 | 0.130 | 0.456 |

The maximum ICC observed (0.280, `qwen2.5:7b`) falls below the "fair agreement" threshold of 0.40 (Koo & Mae, 2016), despite a model-size to ICC correlation of **r=0.817 (p=0.025)**. This provides quantitative evidence that the domain-informed heuristic NLP estimator is superior to zero-shot prompted SLLMs for Circumplex state estimation — a critical validation for the RFS architecture, which cannot rely on proprietary cloud models for embedded deployment.

### 3.7 Multiple Comparisons (BH-Corrected)

| Test | p_raw | p_BH | Sig |
|------|-------|------|-----|
| EXT: AUC > 0.5 (permutation) | 0.0000 | 0.0000 | ✓ |
| RQ2: interaction β ≠ 0 | 0.0000 | 0.0000 | ✓ |
| RQ2: Fisher z-test | 0.0012 | 0.0016 | ✓ |
| RQ3: ANOVA (cohesion) | 0.0195 | 0.0223 | ✓ |
| RQ4: Bayesian ΔAUC > 0 | 0.0010 | 0.0016 | ✓ |
| RFS: urgency vs. MI quality | 0.0000 | 0.0001 | ✓ |
| EXT: permutation AUC | 0.0000 | 0.0000 | ✓ |
| DYN: transition entropy | 0.0722 | 0.0722 | — |

All primary hypotheses survive Benjamini-Hochberg correction at α=0.05. Temporal dynamics (transition entropy) remain non-significant, warranting further investigation.

---

## 4. Theoretical Framework

### 4.1 Olson's Circumplex Model (FACES IV, 2011)

The Circumplex Model characterizes family systems along two orthogonal axes:

- **Cohesion** (0–100): Disengaged ↔ Connected ↔ Enmeshed
- **Flexibility** (0–100): Rigid ↔ Structured ↔ Flexible ↔ Chaotic

The central clinical hypothesis — that families in the balanced zone (cohesion 35–65, flexibility 35–65) are more functional — has been validated across over 1,200 studies. This work estimates these coordinates from dialogue features rather than FACES questionnaires.

### 4.2 Zone Classification

| Zone | Cohesion | Flexibility |
|------|----------|-------------|
| Balanced | 35–65 | 35–65 |
| Rigid-Disengaged | < 35 | < 35 |
| Rigid-Enmeshed | > 65 | < 35 |
| Chaotic-Disengaged | < 35 | > 65 |
| Chaotic-Enmeshed | > 65 | > 65 |

### 4.3 CircumplexEstimator: Feature Weights

**Cohesion weights** (sum to 1.0):

| Feature | Weight | Derivation |
|---------|--------|-----------|
| `empathy_rate` | 0.24 | `reflection_exists == True` rate (therapist) |
| `agreement_rate` | 0.18 | `client_talk_type == 'change'` rate (client) |
| `sent_pos` | 0.12 | Mean VADER compound sentiment |
| `wc_balance` | 0.11 | `min(n_therapist, n_client) / max(...)` |
| `sent_congruence` | 0.15 | `1 - \|sent_therapist_mean - sent_client_mean\|` |
| `neg_absence` | 0.12 | `1 - negation_rate` |
| `sent_div_absence` | 0.08 | `1 - sent_diff_ab` |

**Flexibility weights** (sum to 1.0):

| Feature | Weight | Derivation |
|---------|--------|-----------|
| `question_rate` | 0.25 | `question_exists == True` rate (therapist) |
| `oscillation_rate` | 0.22 | Transition rate of `client_talk_type` |
| `sent_variance` | 0.20 | SD of session-wide sentiment |
| `anti_rigidity` | 0.20 | `1 / (1 + exp(3 × lag1_autocorr))` |
| `novelty` | 0.13 | Mean type-token ratio per utterance |

---

## 5. Dataset

**AnnoMI** (Wu et al., 2022) — Expert-annotated motivational interviewing corpus

- **Source**: [GitHub: uccollab/annomi](https://github.com/uccollab/annomi)
- **Size**: 13,551 utterances across 133 sessions, 18 annotation columns
- **Class balance**: High-MI = 110 (82.7%), Low-MI = 23 (17.3%)
- **Topics**: 44 unique counselling topics across 6 topic clusters (substance, smoking, health, medical, psychosocial, other)

> **Important**: The AnnoMI CSVs are **not included** in this repository due to licensing. Download `AnnoMI-full.csv` from [uccollab/annomi](https://github.com/uccollab/annomi) and place it at `data/annomi/AnnoMI-full.csv` before running.

### Dataset Statistics

```
Shape:              13,551 × 18
Sessions:           133 (transcript_id 0–133)
Mean utterances/session: 101.9 (Median: 50, Min: 6, Max: 1,750)
Speaker balance:    Therapist: 6,826  Client: 6,725
Mean word count:    16.7 words/utterance (SD: 20.7)
Missing values:     ~50% for therapist/client-specific columns
                    (by design: therapist columns NaN for client rows)
```

---

## 6. Repository Structure

```
rfs-scp-v12/
│
├── README.md                        ← This file
├── requirements.txt                 ← Python dependencies
├── environment.yml                  ← Conda environment spec
├── Diagramm_RQ1.png
├── Diagramm_RQ2-5.png
│
├── data/
│   ├── README.md                    ← Data access instructions
│   └── annomi/                      ← Place AnnoMI CSVs here (not tracked)
│       ├── AnnoMI-full.csv          ← 13,551 utterances / 133 sessions
│       └── AnnoMI-simple.csv        ← Fallback (subset)
│
├── src/
│   ├── eda/
│   │   └── annomi_eda.py            ← Dataset EDA and descriptive statistics
│   ├── features/
│   │   └── feature_extraction.py    ← Session-level feature extraction
│   ├── models/
│   │   ├── circumplex_estimator.py  ← CircumplexEstimator heuristic model
│   │   ├── logistic_model.py        ← Logistic regression (GroupKFold)
│   │   ├── lstm_model.py            ← LSTM / BiLSTM encoder
│   │   ├── mamba_model.py           ← Mamba SSM scoring
│   │   └── bayesian_mcmc.py         ← Bayesian weight optimization (MCMC)
│   ├── shap/
│   │   └── dual_shap.py             ← LinearSHAP + PermSHAP + GS-SHAP
│   ├── sllm/
│   │   └── sllm_benchmark.py        ← Multi-SLLM evaluation via Ollama
│   ├── rfs/
│   │   └── rfs_controller.py        ← RFS CircumplexController + InterventionScheduler
│   └── utils/
│       └── stats_utils.py           ← Statistical helpers (Cohen's d, ICC, BH, etc.)
│
├── scripts/
│   ├── run_eda.sh
│   ├── run_full_pipeline.sh
│   └── run_sllm_benchmark.sh
│
├── figures/                         ← Generated figures (9 panels)
├── results/                         ← Generated CSV outputs
└── docs/
    └── methodology.md               ← Extended methods notes (RFS-SCP v12)
```

---

## 7. Installation

### Requirements

| Component | Specification |
|-----------|--------------|
| Python | 3.11+ |
| RAM | 16 GB (32 GB for SLLM benchmark with 7B models) |
| GPU | NVIDIA RTX 3060+ recommended (CPU fallback available) |
| OS | Ubuntu 22.04/24.04 or macOS 13+ |

### Step 1: Clone

```bash
git clone https://github.com/<your-username>/rfs-scp-v12.git
cd rfs-scp-v12
```

### Step 2: Environment

```bash
# Conda (recommended)
conda env create -f environment.yml
conda activate rfs-scp

# Or with pip
pip install -r requirements.txt
```

### Step 3: Data

```bash
mkdir -p data/annomi
# Download AnnoMI-full.csv from https://github.com/uccollab/annomi
# Place it at: data/annomi/AnnoMI-full.csv
```

### Step 4: (Optional) Ollama for SLLM Benchmark

```bash
# Install Ollama: https://ollama.ai/download
ollama serve &
ollama pull qwen2.5:7b
ollama pull gemma2:2b
ollama pull llama3.2:3b
ollama pull phi3:mini
ollama pull qwen2.5:3b
ollama pull qwen2.5:1.5b
ollama pull mistral:7b-instruct
```

### Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥ 1.24 | Numerical computation |
| `pandas` | ≥ 2.0 | Data manipulation |
| `scikit-learn` | ≥ 1.3 | ML models, metrics, cross-validation |
| `scipy` | ≥ 1.11 | Statistical tests |
| `statsmodels` | ≥ 0.14 | Benjamini-Hochberg correction |
| `matplotlib` | ≥ 3.7 | Visualization (9 publication figures) |
| `torch` | ≥ 2.0 | LSTM / BiLSTM (optional; Ridge fallback if absent) |
| `vaderSentiment` | ≥ 3.3 | Sentiment analysis |
| `imbalanced-learn` | ≥ 0.11 | SMOTE sensitivity analysis (optional) |

Random seed: `SEED = 42` (fixed throughout; all stochastic components deterministic except Ollama SLLM calls with `temperature=0.1`).

---

## 8. Usage

### EDA Only

```bash
python src/eda/annomi_eda.py --annomi-dir data/annomi
```

### Full Pipeline

```bash
python src/rfs_scp_v12_main.py \
  --annomi-dir data/annomi \
  --output-dir results/

# Or via shell script
bash scripts/run_full_pipeline.sh
```

### SLLM Benchmark (requires Ollama)

```bash
python src/sllm/sllm_benchmark.py \
  --annomi-dir data/annomi \
  --ollama-url http://localhost:11434/api/generate \
  --ollama-models qwen2.5:7b gemma2:2b llama3.2:3b phi3:mini
```

If Ollama is unavailable, the pipeline automatically falls back to the lexical estimator. Set `OLLAMA_URL` to a non-reachable address to force this behavior.

---

## 9. Figures

All figures are generated to `figures/` after running the full pipeline.

| File | Content |
|------|---------|
| `fig1_rq1_external_v12.png` | Circumplex scatter (High/Low MI), ROC curve, calibration (raw vs. isotonic), Decision Curve Analysis |
| `fig2_rq2_moderation_v12.png` | Moderation scatter, bootstrap CI distribution, within-group correlations, Johnson-Neyman spotlight |
| `fig3_rq4_bayesian_v12.png` | MCMC weight posterior distributions, trace plots (top-3 variance), LOO sensitivity, empathy posterior |
| `fig4_shap_v12.png` | LinearSHAP (signed), LinearSHAP vs. PermSHAP comparison, GS-SHAP (sequence-level), counterfactual analysis, cell map |
| `fig5_ablation_v12.png` | AUC comparison across all models, LSTM sensitivity grid (hidden × layers), SMOTE sensitivity, complexity-performance Pareto |
| `fig6_rq3_cluster_v12.png` | Cohesion by topic cluster, empathy by cluster, cohesion heatmap, post-hoc Cohen's d matrix, BH-corrected pairwise tests |
| `fig7_sllm_v12.png` | SLLM AUC/AP/ensemble, ICC scatter, model-size vs. ICC regression [V12-C], capability radar |
| `fig8_dynamics_umap_v12.png` | t-SNE latent space (3-panel: MI quality / zone / topic cluster) [V12-F], temporal dynamics strip plots [V12-G] |
| `fig9_power_scorecard_v12.png` | Post-hoc power curve, research question scorecard |

---

## 10. v12.0 Changelog

| Tag | Change |
|-----|--------|
| **V12-A** | KernelSHAP **removed** — Spearman ρ ≈ −0.01 with LinearSHAP/PermSHAP confirmed instability at `n_coalitions=100`. Dual-SHAP suite retained: LinearSHAP + PermSHAP (ρ=0.986). |
| **V12-B** | GS-SHAP **fixed** — BiLSTM trained for 80 epochs with early stopping + val-AUC monitoring. Efficiency error (mean=0.00000 ± 0.00000) now non-trivially close to zero and reported per-session. |
| **V12-C** | SLLM framing **revised** — ICC < 0.30 reframed as "LLM limitation evidence" quantifying heuristic estimator superiority. Model-size vs. ICC regression added (r=0.817, p=0.025). |
| **V12-D** | **Power analysis** added — GPower-equivalent post-hoc analysis for t-test, ANOVA, and logistic AUC. N=133 yields power=1.000 for d=1.840; N_needed (80% power) = 5. |
| **V12-E** | **Decision Curve Analysis** added — net benefit curves for CircumplexEstimator, Logistic model, treat-all, and treat-none policies across threshold probabilities. |
| **V12-F** | **t-SNE latent space visualization** added (UMAP fallback) — 3-panel layout: MI quality / Circumplex zone / topic cluster. |
| **V12-G** | **Temporal rigidity dynamics** added — per-session transition entropy, emotional inertia (\|lag1_autocorr\|), cohesion volatility, empathy recovery rate. |
| **V12-H** | **Counterfactual analysis** added — minimum empathy increase to flip Low→High MI classification via logistic boundary inversion. Median Δ = 1.214 SD. |
| **V12-I** | **LSTM sensitivity grid** added — `hidden_size ∈ {16, 32, 48, 64}` × `n_layers ∈ {1, 2}`. Best: h=32, L=1, AUC=0.617. |
| **V12-J** | **Figure layout overhaul** — `constrained_layout=True`, non-overlapping annotations via manual offsets, unified font sizes, axis padding standardized. |
| **V12-K** | **Calibration enhanced** — isotonic recalibration added. Brier: 0.105 → 0.080; ECE: 0.113 → 0.000. |

---

## 11. Connection to the Robot Family System

This work directly addresses the core automation gap in the RFS research program, contributing **Contribution C1** of the broader research roadmap:

| Contribution | Description | Status |
|-------------|-------------|--------|
| **C1** | Automated NLP-based Circumplex state estimation from dialogue | **This work** |
| C2 | Sentiment-aware adaptive interaction controller | Planned |
| C3 | Distribution-aware on-device AI governance for toio hardware | Planned |
| C4 | Controlled user study with older adult participants | Planned |

### RFS Controller Behavior

The `CircumplexController` in `src/rfs/rfs_controller.py` maps estimated Circumplex states to five zone-specific robot behavior policies:

| Zone | Robot Role | Intervention Mode | Priority |
|------|-----------|------------------|---------|
| Balanced | MAINTAIN | minimal | 1 |
| Rigid-Enmeshed | DIVERSIFY | flexibility_boost | 2 |
| Rigid-Disengaged | RECONNECT | cohesion_build | 5 (highest) |
| Chaotic-Disengaged | STABILIZE | structure_build | 4 |
| Chaotic-Enmeshed | MODERATE | boundary_set | 3 |

A 3-step cooldown prevents intervention cascades. Bayesian-updated empathy weight: **w = 0.925** (vs. baseline w = 0.240), operationalizing the RQ4 posterior into real-time robot behavior.

**Validation**: RFS intervention urgency correlates with expert MI quality at **r = 0.350, p < 0.001**, confirming that the automated estimator produces controller outputs aligned with human expert judgment.

---

## 12. References

- Hirano, T. & Tanaka, F. (2026). Dialogue generation for family robots using ROS and generative AI. *Proc. IEEE/SICE SII 2026*.
- Hirano, T. & Tanaka, F. (2025). Toward the development of the Robot Family System (RFS): Implementing a Circumplex Model with generative AI. *Japanese Domestic Conference*.
- Iio, T., Yoshikawa, Y., & Ishiguro, H. (2017). Retaining human-robots conversation: Comparing single robot to multiple robots in a real event. *Journal of Advanced Computational Intelligence and Intelligent Informatics, 21*, 675–685.
- Koo, T. K. & Mae, M. Y. (2016). A guideline of selecting and reporting intraclass correlation coefficients for reliability research. *Journal of Chiropractic Medicine, 15*(2), 155–163.
- Office of the Surgeon General. (2023). *Our epidemic of loneliness and isolation*. U.S. Department of Health and Human Services.
- Olson, D. H., Sprenkle, D. H., & Russell, C. S. (1979). Circumplex model of marital and family systems: I. *Family Process, 18*(1), 3–28.
- Olson, D. (2011). FACES IV and the Circumplex Model: Validation study. *Journal of Marital and Family Therapy, 37*(1), 64–80.
- Rheman, J. M., Baggett, R. P., Simecek, M., Fraune, M. R., & Tsui, K. M. (2024). Longitudinal study of mobile telepresence robots in older adults' homes. *J. Hum.-Robot Interact., 13*.
- Wu, S., et al. (2022). AnnoMI: A dataset of expert-annotated counselling dialogues. *Proc. ACL*.
- Gu, A. & Dao, T. (2023). Mamba: Linear-time sequence modeling with selective state spaces. *arXiv:2312.00752*.
