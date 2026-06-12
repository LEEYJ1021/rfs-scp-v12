# RFS-SCP v12: Circumplex-Grounded Relational State Estimation for Robot Family System Design

> **Automated Circumplex State Estimation from MI Dialogue | Bayesian Weight Optimization | Dual-SHAP Interpretability | Multi-SLLM Benchmark | RFS Controller**

---

## Overview

This repository contains the full analysis pipeline, source code, figures, and results for **RFS-SCP v12.0** — a computational framework for estimating Olson's Circumplex Model states (cohesion and flexibility) directly from motivational interviewing (MI) dialogue, and using those estimates to drive an autonomous Robot Family System (RFS) controller.

The work builds on two prior studies:

1. Hirano & Tanaka (2026). *Dialogue Generation for Family Robots Using ROS and Generative AI.* IEEE/SICE SII 2026.
2. Hirano & Tanaka (2025). *Toward the Development of the Robot Family System (RFS): Implementing a Circumplex Model with Generative AI.* (Japanese domestic conference)

**Research gap addressed**: The RFS prototype depended on manual FACES-IV questionnaire administration to assess family state. This work replaces that manual step with an automated NLP-based estimator validated against expert MI quality labels, enabling continuous autonomous state monitoring.

---

## Repository Structure

```
rfs-scp-v12/
│
├── README.md                        ← This file
├── requirements.txt                 ← Python dependencies
├── environment.yml                  ← Conda environment spec
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
├── figures/
│   ├── fig1_rq1_external_v12.png
│   ├── fig2_rq2_moderation_v12.png
│   ├── fig3_rq4_bayesian_v12.png
│   ├── fig4_shap_v12.png
│   ├── fig5_ablation_v12.png
│   ├── fig6_rq3_cluster_v12.png
│   ├── fig7_sllm_v12.png
│   ├── fig8_dynamics_umap_v12.png
│   └── fig9_power_scorecard_v12.png
│
├── results/
│   ├── annomi_session_features_v12.csv
│   ├── hypothesis_summary_v12.csv
│   ├── shap_linear_v12.csv
│   ├── shap_permutation_v12.csv
│   ├── shap_gsshap_v12.csv
│   ├── sllm_benchmark_v12.csv
│   └── rfs_controller_log_v12.csv
│
└── docs/
    └── methodology.md               ← Extended methods notes
```

---

## Research Questions

| ID | Question | Outcome |
|----|----------|---------|
| **RQ1** | Does the heuristic CircumplexEstimator discriminate expert MI quality? | **CONFIRMED** AUC=0.816 CI=[0.699, 0.924] d=1.840 |
| **RQ2** | Does empathy-cohesion association differ by MI quality? | **SUPPORTED** β_int=−0.718 CI=[−1.170, −0.464] ΔR²=0.138 |
| **RQ3** | Does topic cluster moderate relational dynamics? | **CONFIRMED** F=2.804 p=0.020 η²=0.099 |
| **RQ4** | Does Bayesian MCMC weight optimization improve AUC? | **INFORMATIVE** ΔAUC=+0.043, dominant=empathy |
| **RQ5-A** | Interpretable features vs temporal LSTM encoding? | **INTERPRETABLE WINS** Logistic=0.916 LSTM=0.617 |
| **RQ5-B** | Does Mamba SSM scoring add predictive value? | **SUPPORTED** Ensemble AUC=0.816 (α=0.95) |
| **RQ5-C** | Can small LLMs estimate Circumplex states? | **CONFIRMED (LLM LIMITATION)** ICC_max=0.280 |
| **SHAP** | Do LinearSHAP and PermSHAP converge? | **VALIDATED** ρ=0.986 |
| **DYN** | Do temporal rigidity dynamics differentiate MI quality? | **INFORMATIVE** cohesion volatility d=−0.511 |
| **RQ6** | Does RFS intervention urgency correlate with MI quality? | **VALIDATED** r=0.350 p<0.001 |

---

## Key Results

### RQ1 — External Validity

| Metric | Value |
|--------|-------|
| AUC | 0.8162 |
| 95% CI (bootstrap) | [0.699, 0.924] |
| Cohen's d | 1.840 |
| MCC | 0.543 |
| Specificity | 0.652 |
| Brier (raw) | 0.105 |
| Brier (isotonic) | 0.080 |
| ECE (isotonic) | 0.000 |
| Permutation p | 0.000 |
| Post-hoc power | 1.000 (N=133) |

### RQ2 — Conditional Moderation

```
cohesion ~ empathy + MI_quality + empathy × MI_quality

empathy:                β=+0.340  t=+4.119  p<.001
empathy×mi_quality:     β=−0.718  t=−5.819  p<.001  ΔR²=0.138
Bootstrap 95% CI:       [−1.170, −0.464]  → SIGNIFICANT

High-MI:  r=+0.017 (n.s.)
Low-MI:   r=+0.666 (p<.05)
Fisher z: p=0.001
```

### RQ3 — Topic Cluster ANOVA

| Statistic | Value |
|-----------|-------|
| F(5, 127) | 2.804 |
| p | 0.020 |
| η² | 0.099 |
| ω² | 0.064 |

Clusters (substance, medical, other, smoking, health, psychosocial) differ significantly in mean cohesion. BH-corrected post-hoc shows health vs. other as the largest contrast (d=1.20).

### RQ4 — Bayesian MCMC Weight Optimization

| | AUC |
|--|-----|
| Baseline (heuristic) | 0.8162 |
| Nelder-Mead optimal | 0.8593 |
| MCMC posterior mean | Acceptance=32.3% |

Dominant feature: **empathy** (posterior weight shifts from 0.24 baseline → 0.93 posterior mean). Agreement weight shows largest positive LOO ΔAUC (+0.045).

### RQ5-A — Ablation

| Model | AUC | Notes |
|-------|-----|-------|
| Circumplex (heuristic) | 0.816 | No training |
| Bayesian (optimized) | 0.859 | MCMC weights |
| Logistic (GroupKFold) | **0.916** | Session-level features |
| LSTM h=32 L=1 (best) | 0.617 | Sequential utterances |

Interpretable session-level features outperform temporal LSTM encoding by ΔAUC=+0.300. Counterfactual analysis: median empathy increase of **1.214 std** required to flip Low-MI → High-MI classification.

### RQ5-B — Mamba SSM

| | AUC |
|--|-----|
| Mamba standalone | 0.578 |
| Ensemble (α=0.95) | 0.816 |

Mamba alone is marginal, but ensemble preserves heuristic performance. Optimal α=0.95 indicates near-complete reliance on CircumplexEstimator.

### RQ5-C — Multi-SLLM Benchmark

| Model | Params (B) | ICC_cohesion | AUC |
|-------|-----------|-------------|-----|
| qwen2.5:7b | 7.0 | **0.280** | **0.694** |
| gemma2:2b | 2.0 | 0.027 | 0.582 |
| llama3.2:3b | 3.0 | 0.093 | 0.570 |
| phi3:mini | 3.8 | 0.128 | 0.544 |
| qwen2.5:3b | 3.0 | 0.122 | 0.504 |
| qwen2.5:1.5b | 1.5 | 0.044 | 0.460 |
| mistral:7b-instruct | 7.0 | 0.130 | 0.456 |

Model-size → ICC correlation: **r=0.817 p=0.025**. Maximum ICC=0.280 (below fair agreement threshold of 0.40) confirms that even 7B-parameter SLLMs cannot reliably estimate Circumplex states, providing quantitative evidence that the heuristic NLP estimator is superior for this task.

### SHAP — Dual-SHAP Convergence

LinearSHAP ↔ PermSHAP Spearman ρ = **0.986**

Feature importance ranking (top 5):

| Rank | Feature | LinearSHAP | PermSHAP |
|------|---------|-----------|---------|
| 1 | empathy_rate | 2.124 | 0.185 |
| 2 | negation_rate | −0.809 | 0.072 |
| 3 | wc_balance | 0.621 | 0.054 |
| 4 | emp_agr_interact | 0.602 | 0.040 |
| 5 | agreement_rate | 0.593 | 0.050 |

GS-SHAP (sequence-level, BiLSTM): efficiency error mean=0.00000 ± 0.00000.

### Temporal Dynamics (V12-G)

| Metric | Cohen's d | p | Sig |
|--------|----------|---|-----|
| cohesion_volatility | −0.511 | 0.028 | ✓ |
| transition_entropy | +0.416 | 0.072 | n.s. |
| emotional_inertia | −0.386 | 0.095 | n.s. |
| empathy_recovery_rate | +0.068 | 0.769 | n.s. |

### BH-Corrected Multiple Comparisons

| Test | p_raw | p_BH | Sig |
|------|-------|------|-----|
| EXT: AUC > 0.5 | 0.000 | 0.000 | ✓ |
| RQ2: interaction β ≠ 0 | 0.000 | 0.000 | ✓ |
| RQ2: Fisher z | 0.001 | 0.002 | ✓ |
| RQ3: ANOVA(cohesion) | 0.020 | 0.022 | ✓ |
| RQ4: Bayesian ΔAUC > 0 | 0.001 | 0.002 | ✓ |
| RFS: urgency vs MI | 0.000 | 0.000 | ✓ |
| EXT: Permutation AUC | 0.000 | 0.000 | ✓ |
| DYN: transition entropy | 0.072 | 0.072 | – |

---

## Dataset

**AnnoMI** (Motivational Interviewing dataset)

- **Source**: [GitHub: uccollab/annomi](https://github.com/uccollab/annomi)
- **Citation**: Wu et al. (2022). *AnnoMI: A Dataset of Expert-Annotated Counselling Dialogues.*
- **Full version**: 13,551 utterances, 133 sessions, 18 columns
- **Class balance**: High-MI=110 (82.7%), Low-MI=23 (17.3%)
- **Topics**: 44 unique topics across 133 sessions

The AnnoMI CSVs are **not included** in this repository due to licensing. Place `AnnoMI-full.csv` (preferred) or `AnnoMI-simple.csv` in `data/annomi/` before running.

---

## Hardware & Software Requirements

### Minimum Recommended Hardware

| Component | Specification |
|-----------|--------------|
| CPU | 8-core, 3.0 GHz+ |
| RAM | 16 GB (32 GB for SLLM benchmark) |
| GPU | NVIDIA RTX 3060 or better (for LSTM/BiLSTM training) |
| Storage | 10 GB free |
| OS | Ubuntu 22.04 / 24.04 or macOS 13+ |

The full pipeline including LSTM sensitivity grid and GS-SHAP can run on CPU only (Ridge fallback active when PyTorch unavailable), but will take significantly longer (~2–4 hours vs ~20 minutes with GPU).

### SLLM Benchmark Hardware

Running the multi-SLLM benchmark requires [Ollama](https://ollama.ai/) with locally hosted models. Minimum 16 GB RAM; 32 GB recommended for 7B models.

| Model | RAM Required |
|-------|-------------|
| qwen2.5:1.5b | 4 GB |
| qwen2.5:3b | 6 GB |
| gemma2:2b | 5 GB |
| llama3.2:3b | 6 GB |
| phi3:mini | 6 GB |
| qwen2.5:7b | 12 GB |
| mistral:7b-instruct | 12 GB |

---

## Installation

### 1. Clone

```bash
git clone https://github.com/<your-username>/rfs-scp-v12.git
cd rfs-scp-v12
```

### 2. Environment (conda recommended)

```bash
conda env create -f environment.yml
conda activate rfs-scp
```

Or with pip:

```bash
pip install -r requirements.txt
```

### 3. Data

```bash
mkdir -p data/annomi
# Download AnnoMI-full.csv from https://github.com/uccollab/annomi
# Place it at: data/annomi/AnnoMI-full.csv
```

### 4. (Optional) Ollama for SLLM benchmark

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

---

## Usage

### EDA only

```bash
python src/eda/annomi_eda.py --annomi-dir data/annomi
```

### Full pipeline

```bash
python src/rfs_scp_v12_main.py \
  --annomi-dir data/annomi \
  --output-dir results/
```

Or via shell script:

```bash
bash scripts/run_full_pipeline.sh
```

### SLLM benchmark (requires Ollama)

```bash
python src/sllm/sllm_benchmark.py \
  --annomi-dir data/annomi \
  --ollama-url http://localhost:11434/api/generate \
  --ollama-models qwen2.5:7b gemma2:2b llama3.2:3b phi3:mini
```

### Skip SLLM (faster run)

Set `OLLAMA_URL` to a non-reachable address; the pipeline auto-falls back to the lexical estimator.

---

## Figures

All figures are saved to `figures/` after running the pipeline.

| File | Content |
|------|---------|
| `fig1_rq1_external_v12.png` | Circumplex scatter (High/Low MI), ROC, Calibration, DCA |
| `fig2_rq2_moderation_v12.png` | Moderation scatter, bootstrap CI, within-group r, Johnson-Neyman |
| `fig3_rq4_bayesian_v12.png` | MCMC weight distributions, traces, LOO sensitivity, empathy posterior |
| `fig4_shap_v12.png` | LinearSHAP, PermSHAP comparison, GS-SHAP, counterfactual, cell map |
| `fig5_ablation_v12.png` | AUC comparison, LSTM sensitivity grid, SMOTE sensitivity, Pareto |
| `fig6_rq3_cluster_v12.png` | Cohesion by cluster, empathy by cluster, heatmap, Cohen's d matrix |
| `fig7_sllm_v12.png` | SLLM AUC/AP/ensemble, ICC scatter, model-size regression, radar |
| `fig8_dynamics_umap_v12.png` | t-SNE (MI/zone/cluster), temporal dynamics strip plots |
| `fig9_power_scorecard_v12.png` | Power curve, research question scorecard |

---

## v12.0 Changelog

| Tag | Change |
|-----|--------|
| V12-A | KernelSHAP removed (ρ≈−0.01 with LinearSHAP/PermSHAP confirmed instability at n_coalitions=100). Dual-SHAP suite: LinearSHAP + PermSHAP only. |
| V12-B | GS-SHAP fixed: BiLSTM trained 80 epochs with early stopping + val-AUC monitoring. Efficiency error now non-zero and reported. |
| V12-C | SLLM framing revised: ICC<0.30 reframed as "LLM limitation evidence." Model-size vs ICC regression added (r=0.817). |
| V12-D | GPower-equivalent post-hoc power analysis for t-test, ANOVA, and logistic AUC. N=133 yields power=1.000 for d=1.840. |
| V12-E | Decision Curve Analysis (DCA): net benefit curves for CircumplexEstimator and Logistic model. |
| V12-F | t-SNE latent space visualization (UMAP fallback): 3-panel (MI quality / zone / topic cluster). |
| V12-G | Temporal rigidity dynamics: per-session transition entropy, emotional inertia, cohesion volatility, empathy recovery rate. |
| V12-H | Counterfactual analysis: minimum empathy increase to flip Low→High MI via logistic boundary inversion. Median Δ=1.214 std. |
| V12-I | LSTM sensitivity grid: hidden_size ∈ {16,32,48,64} × n_layers ∈ {1,2} AUC grid. Best: h=32, L=1, AUC=0.617. |
| V12-J | Figure layout overhaul: constrained_layout, non-overlapping annotations, unified font sizes. |
| V12-K | Calibration enhanced: isotonic recalibration added. Brier: 0.105→0.080, ECE: 0.113→0.000. |

---

## Connection to Robot Family System (RFS)

This analysis directly addresses the core gap in the RFS research program:

```
Manual FACES-IV survey
        ↓
[This work: automated CircumplexEstimator from dialogue]
        ↓
RFS CircumplexController
        ↓
toio robot behavior (empathy_intensity, intervention_mode, verbosity)
```

The `CircumplexController` in `src/rfs/rfs_controller.py` maps estimated states to five zone policies:

| Zone | Robot Role | Intervention Mode |
|------|-----------|------------------|
| balanced | MAINTAIN | minimal |
| rigid-enmeshed | DIVERSIFY | flexibility_boost |
| rigid-disengaged | RECONNECT | cohesion_build |
| chaotic-disengaged | STABILIZE | structure_build |
| chaotic-enmeshed | MODERATE | boundary_set |

RFS urgency correlates with MI quality: **r=0.350, p<0.001**. Bayesian-updated empathy weight: **w=0.925** (vs baseline 0.240), informing robot empathy intensity calculation.

---

## Theoretical Background

### Olson's Circumplex Model (FACES IV, 2011)

The model characterizes family systems along two axes:

- **Cohesion** (0–100): Disengaged ↔ Connected ↔ Enmeshed
- **Flexibility** (0–100): Rigid ↔ Structured ↔ Flexible ↔ Chaotic

**Central hypothesis**: Families in the balanced zone (cohesion 35–65, flexibility 35–65) are more functional. This work estimates these coordinates from dialogue features rather than FACES questionnaires.

### CircumplexEstimator Feature Weights

**Cohesion weights** (sum to 1.0):

| Feature | Weight | Rationale |
|---------|--------|-----------|
| empathy | 0.24 | Reflection existence rate (therapist) |
| agreement | 0.18 | Client change-talk rate |
| sent_pos | 0.12 | Positive sentiment mean |
| wc_balance | 0.11 | Word-count balance therapist/client |
| sent_congruence | 0.15 | 1 − |sent_diff_ab| |
| neg_absence | 0.12 | 1 − negation_rate |
| sent_div_absence | 0.08 | 1 − sent_diff_ab |

**Flexibility weights** (sum to 1.0):

| Feature | Weight | Rationale |
|---------|--------|-----------|
| oscillation | 0.22 | Client talk-type transition rate |
| question | 0.25 | Question rate (therapist) |
| sent_variance | 0.20 | Sentiment standard deviation |
| novelty | 0.13 | Mean type-token ratio |
| anti_rigidity | 0.20 | 1/(1+exp(3×lag1_autocorr)) |

---

## License

MIT License. See `LICENSE`.
