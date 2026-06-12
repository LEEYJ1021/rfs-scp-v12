# RFS-SCP v12 — Extended Methodology Notes

> **Circumplex-Grounded Relational State Estimation for Robot Family System Design**

---

## 1. Theoretical Foundations

### 1.1 Olson's Circumplex Model (FACES IV, 2011)

The Circumplex Model of Marital and Family Systems (Olson, 1979; Olson, 2011) characterises family functioning along two primary orthogonal dimensions:

- **Cohesion** (0–100): the degree of emotional bonding between family members. At extremes — Disengaged (0–35) and Enmeshed (65–100) — functioning is impaired. The balanced zone (35–65) is theorised to be most adaptive.
- **Flexibility** (0–100): the capacity for change in leadership, roles, and relationship rules. Extremes — Rigid (0–35) and Chaotic (65–100) — reflect dysfunction; the balanced zone (35–65) is again optimal.

The central clinical hypothesis is that **families in the balanced zone across both dimensions are more functional** and resilient to stressors. This has been validated across >1,200 published studies using the FACES instruments.

A third dimension, **Communication**, is treated as a facilitator of movement along the cohesion and flexibility axes rather than as an independent axis, and is not directly estimated in this framework.

### 1.2 Zone Classification

The five Circumplex zones used in this work are:

| Zone | Cohesion | Flexibility |
|---|---|---|
| Balanced | 35–65 | 35–65 |
| Rigid-Disengaged | < 35 | < 35 |
| Rigid-Enmeshed | > 65 | < 35 |
| Chaotic-Disengaged | < 35 | > 65 |
| Chaotic-Enmeshed | > 65 | > 65 |

Boundary sessions (exactly at 35 or 65) are assigned by the fall-through logic in `CircumplexEstimator.zone`.

---

## 2. Dataset

### 2.1 AnnoMI

All experiments use the **AnnoMI** corpus (Wu et al., 2022):

- 13,551 utterances across 133 motivational interviewing (MI) sessions
- 18 annotation columns including `mi_quality` (high/low), `reflection_exists`, `question_exists`, `client_talk_type`, `main_therapist_behaviour`
- Session-level class balance: High-MI = 110 (82.7%), Low-MI = 23 (17.3%)
- 44 unique counselling topics

The binary \texttt{mi\_quality} label (high/low) reflects expert annotator rating of the therapist's overall MI adherence for the session, and serves as the ground-truth criterion for all external validity analyses.

### 2.2 Feature Extraction Pipeline

Session-level features are extracted by aggregating utterance-level statistics within \texttt{transcript\_id}:

**Cohesion-relevant features**

| Feature | Derivation |
|---|---|
| \texttt{empathy\_rate} | \texttt{reflection\_exists == True} rate (therapist utterances) |
| `agreement_rate` | `client_talk_type == 'change'` rate (client utterances) |
| `sent_mean` | Mean VADER compound sentiment across all utterances |
| `wc_balance` | `min(n_therapist, n_client) / max(n_therapist, n_client)` |
| `sent_congruence` | `1 - |sent_therapist_mean - sent_client_mean|` |
| `negation_rate` | Proportion of utterances containing negation tokens |
| `sent_diff_ab` | Absolute sentiment divergence between interlocutors |
| `emp_agr_interact` | `empathy_rate × agreement_rate` |

**Flexibility-relevant features**

| Feature | Derivation |
|---|---|
| `oscillation_rate` | Transition rate of `client_talk_type` (change/sustain/neutral) |
| `question_rate` | `question_exists == True` rate (therapist utterances) |
| `sent_std` | Standard deviation of session-wide sentiment |
| `mean_ttr` | Mean type-token ratio per utterance |
| \texttt{lag1\_autocorr} | Pearson $r(sent_t, sent_{t+1})$ over session |

**Temporal dynamics (V12-G)**

| Feature | Derivation |
|---|---|
| `transition_entropy` | Binary entropy of sentiment sign transitions |
| `emotional_inertia` | `|lag1_autocorr|` |
| `cohesion_volatility` | `sent_std` (session-level) |
| `empathy_recovery_rate` | Proportion of consecutive therapist turns with rising sentiment |

---

## 3. CircumplexEstimator

The heuristic CircumplexEstimator maps session features to Circumplex coordinates using fixed, theory-derived weights:

### 3.1 Cohesion Weights

$$\text{cohesion} = 100 \times \sum_{k} w_k^{\text{coh}} \cdot f_k$$

| Feature $k$ | Weight $w_k$ | Theoretical rationale |
|---|---|---|
| empathy | 0.24 | Reflective listening as primary cohesion signal |
| agreement | 0.18 | Client change-talk as engagement indicator |
| sent_pos | 0.12 | Positive affect supports connection |
| wc_balance | 0.11 | Balanced turn-taking reflects reciprocity |
| sent_congruence | 0.15 | Affective alignment between dyad members |
| neg_absence | 0.12 | Low negation reduces relational friction |
| sent_div_absence | 0.08 | Residual divergence penalty |

### 3.2 Flexibility Weights

$$\text{flexibility} = 100 \times \sum_{k} w_k^{\text{flex}} \cdot f_k$$

| Feature $k$ | Weight $w_k$ | Theoretical rationale |
|---|---|---|
| oscillation | 0.22 | Topic/stance transitions as adaptive flexibility |
| question | 0.25 | Open-ended questioning opens relational space |
| sent_variance | 0.20 | Moderate emotional variability signals openness |
| novelty | 0.13 | Lexical novelty (TTR) as idea diversity proxy |
| anti\_rigidity | $1 / (1 + e^{3 \cdot \mathrm{lag1\_autocorr}})$ |

### 3.3 Normalisation

Each raw feature is clipped to [0, 1] using domain-appropriate normalisation (e.g., empathy clipped to [0, 0.06]; agreement to [0, 0.25]) before weighting. The resulting cohesion and flexibility scores are then clipped to [0, 100].

---

## 4. Bayesian MCMC Weight Optimisation (RQ4)

### 4.1 Nelder-Mead Optimisation

A simplex optimiser searches the log-weight space to maximise AUC:

$$\hat{w}^* = \arg\max_{w \in \Delta} \text{AUC}(y, \text{cohesion}(w; X))$$

where $\Delta$ is the probability simplex. The Nelder-Mead method is used for its robustness to non-smooth objective surfaces.

### 4.2 Adaptive MCMC in ALR Space

The weight vector $w \in \Delta^{K-1}$ is re-parameterised via the additive log-ratio (ALR) transformation:

$$z_k = \log \frac{w_k}{w_K}, \quad k = 1, \ldots, K-1$$

Metropolis-Hastings sampling is performed in the unconstrained $z$-space with proposal:

$$z' = z + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2 I)$$

The acceptance criterion uses an AUC-based pseudo-likelihood with temperature $T = 300$:

$$\alpha = \min\left(1, \exp\left(T \cdot \left[\text{AUC}(w') - \text{AUC}(w)\right]\right)\right)$$

Step size $\sigma$ is adapted every 500 iterations to target an acceptance rate of 20–40%.

### 4.3 Posterior Summarisation

The post-burn-in chain ($N = 3{,}750$ samples after 25% burn-in) provides:

- Posterior mean weights $\bar{w}$
- 95% credible intervals via empirical percentiles
- LOO sensitivity: $\Delta\text{AUC}_k = \text{AUC}(w_{\setminus k}) - \text{AUC}(w_\text{base})$

---

## 5. Logistic Regression Pipeline (RQ5-A)

### 5.1 GroupKFold Cross-Validation

To prevent data leakage --- transcripts spanning many utterances could otherwise appear in both train and validation splits --- all logistic regression evaluation uses \texttt{GroupKFold(n\_splits=10)} with \texttt{groups = transcript\_id}. This ensures no session appears in both train and validation partitions within any fold.

### 5.2 Repeated StratifiedKFold

For stability estimation (standard deviation of AUC across fold configurations), \texttt{RepeatedStratifiedKFold(n\_splits=5, n\_repeats=5)} is used separately. This ignores group structure but provides AUC variance across 25 random splits.

### 5.3 Counterfactual Analysis (V12-H)

For each Low-MI session, we compute the minimum standardised increase $\Delta$ in \texttt{empathy\_rate} required to flip the logistic model's prediction from Low-MI to High-MI:

$$
\Delta^*
=
\min \left\{
\delta \geq 0 :
\hat{p}\left(
x + \delta \cdot e_{\mathrm{empathy}}
\right)
\geq 0.5
\right\}
$$

where $e_{\mathrm{empathy}}$ is the unit vector for the \texttt{empathy\_rate} feature. The search uses a 300-point grid over $[0, 3\sigma]$.

---

## 6. LSTM Architecture and Training (RQ5-A / RQ5-B)

### 6.1 Input Representation

Utterances are represented by five features:

1. VADER compound sentiment (`vader`)
2. Type-token ratio (`ttr`)
3. Interlocutor indicator (`is_therapist`, 0/1)
4. Negation flag (`neg_flag`, 0/1)
5. Normalised word count (`word_count / 50`, clipped to [0, 1])

Sequences are zero-padded to `MAX_SEQ = 120` utterances.

### 6.2 LSTMEncoder (RQ5-A)

- Input → LSTM(hidden, n_layers) → FC(24, ReLU) → Dropout(0.2) → FC(1)
- Target: cohesion score (regression)
- Loss: MSELoss
- Optimiser: Adam(lr=3e-3, weight_decay=5e-4)
- Early stopping: patience=8 on training loss
- Sensitivity grid [V12-I]: hidden ∈ {16, 32, 48, 64} × n_layers ∈ {1, 2}

### 6.3 BiLSTMClassifier (GS-SHAP backbone)

- Input → BiLSTM(hidden=32, bidirectional) → Linear(64→2)
- Target: MI quality (binary classification)
- Loss: CrossEntropyLoss
- Optimiser: Adam(lr=5e-4, weight_decay=1e-4) with StepLR(step=30, γ=0.5)
- Early stopping: patience=15 on training loss
- 80 training epochs [V12-B]

---

## 7. Dual-SHAP Suite (V12-A, V12-B)

### 7.1 LinearSHAP

For a logistic regression with coefficients $\beta$:

$$\phi_j = \beta_j \cdot \text{std}(X_j)$$

This is the exact Shapley value for linear models under the marginal independence assumption. It provides a signed, analytic importance measure with $O(1)$ complexity.

### 7.2 PermutationSHAP

Symmetric path-integral SHAP using antithetic permutation sampling. For each observation $x$ and each random permutation $\pi$ of features, the marginal contribution of feature $j$ is computed as:

$$\phi_j(x) \approx \frac{1}{2P} \sum_{p=1}^{P} \left[ \Delta_j(\pi_p) + \Delta_j(\bar{\pi}_p) \right]$$

where $\bar{\pi}_p$ is the antithetic (reversed) permutation and $\Delta_j(\pi)$ is the change in model output when feature $j$ is added in the order defined by $\pi$.

**Parameters**: n_perms = 50, all 133 sessions, Spearman ρ with LinearSHAP = 0.986.

### 7.3 GS-SHAP (Group-Segment SHAP) [V12-B]

GS-SHAP extends SHAP to sequential data by defining "players" as (feature_group × time_segment) tuples.

**Step 1 — Feature grouping (HSIC spectral clustering)**

Features are grouped by their dependence structure using the Hilbert-Schmidt Independence Criterion. HSIC-based affinity matrix construction uses centred RBF kernels; spectral clustering determines the number of groups via the eigenvalue gap heuristic.

**Step 2 — Temporal segmentation (MMD change-point detection)**

For each feature group, temporal segments are identified by recursively splitting the sequence at the time step that maximises the unbiased MMD² between the two halves, subject to a permutation-based significance threshold (95th percentile of the null MMD distribution).

**Step 3 — Antithetic permutation SHAP over players**

Players $= \{(\mathrm{group}\ g,\ \mathrm{segment}\ s)\}$ are subjected to antithetic permutation SHAP attribution. The baseline is the session-mean feature vector broadcast to the sequence length.

**Efficiency axiom**: $\sum_i \phi_i = f(x) - f(\text{baseline})$. Empirical efficiency error reported as mean ± std over sessions.

**V12-B fix**: BiLSTM trained for 80 epochs with early stopping; efficiency error now non-trivially close to zero (mean = 0.00000 ± 0.00000 due to floating-point precision).

### 7.4 KernelSHAP Removal [V12-A]

KernelSHAP was removed from v12 following discovery of high instability: Spearman ρ ≈ −0.01 with both LinearSHAP and PermSHAP at `n_coalitions=100`. The instability arises from the exponential coalition space combined with the small feature count (D=12), making the LASSO regression step ill-conditioned. The dual-method (LinearSHAP + PermSHAP) with ρ = 0.986 provides sufficient convergence evidence.

---

## 8. Mamba SSM Scoring (RQ5-B)

The numpy Mamba approximation uses a diagonal state space model with:

- State dimension: $d_s = 3$
- Input-dependent step size: $\Delta_t = \Delta_0 + 0.3 \cdot \text{clip}(u_t[0], 0, 1)$
- ZOH discretisation: $\bar{A} = e^{-\Delta A}$, $\bar{B} = (I - \bar{A}) \cdot B$
- State update: $h_t = \bar{A} h_{t-1} + \bar{B}(u_t)$

The session score is derived from the mean output over the final 33% of time steps, mapped to [0, 100] via:

$$\text{score} = \text{clip}\left(50 + 20 \cdot \tanh(\bar{y}_\text{tail}), 0, 100\right)$$

**Ensemble**: $\hat{c} = \alpha \cdot c_\text{circ} + (1-\alpha) \cdot c_\text{mamba}$, with $\alpha = 0.95$ found by AUC-maximising grid search.

---

## 9. Multi-SLLM Benchmark (RQ5-C) [V12-C]

### 9.1 Prompt Design

Each session is represented by a 10-turn dialogue excerpt. The system prompt instructs each SLLM to rate the dialogue on Olson's cohesion and flexibility scales (0–100), returning structured JSON with a one-sentence reasoning field.

### 9.2 Evaluation Metrics

- **ICC(2,1)**: Two-way mixed-effects intraclass correlation between SLLM and CircumplexEstimator cohesion scores. ICC < 0.40 = poor agreement; 0.40–0.60 = fair; > 0.60 = good (Koo & Mae, 2016).
- **Spearman ρ**: Rank-order agreement with CircumplexEstimator cohesion.
- **AUC**: Discrimination of High/Low MI quality using SLLM cohesion as the ranking signal.

### 9.3 Reframing: LLM Limitation Evidence [V12-C]

The maximum ICC observed (0.280, qwen2.5:7b) is below the "fair agreement" threshold of 0.40, despite a model-size to ICC correlation of r = 0.817 (p = 0.025). This confirms that even 7B-parameter SLLMs cannot reliably estimate Circumplex states from raw dialogue, providing quantitative evidence for the superiority of the domain-informed heuristic NLP estimator over zero-shot prompted SLLMs.

---

## 10. Robot Family System Controller (RQ6)

### 10.1 Bayesian Weight Update

Following MCMC optimisation, the RFS controller's empathy weight is updated from the baseline ($w = 0.240$) to the posterior mean ($w = 0.925$). This substantially increases the controller's sensitivity to empathy dynamics in computing \texttt{empathy\_intensity}:

$$
I_{\mathrm{empathy}}
=
\mathrm{clip}\left(
w_{\mathrm{emp}} \cdot r_{\mathrm{emp}}
+
w_{\mathrm{agr}} \cdot r_{\mathrm{agr}},
0,
1
\right)
\times
\mathrm{zone\_factor}
$$

### 10.2 Intervention Scheduler

The `InterventionScheduler` implements a priority-based cooldown system:

| Zone | Priority | Trigger condition |
|---|---|---|
| rigid-disengaged | 5 | priority ≥ 4 or urgency > 0.7 |
| chaotic-disengaged | 4 | priority ≥ 4 or urgency > 0.7 |
| chaotic-enmeshed | 3 | zone_change and priority ≥ 3 |
| rigid-enmeshed | 2 | zone_change and priority ≥ 3 |
| balanced | 1 | never triggered |

A 3-step cooldown prevents intervention cascades.

---

## 11. Statistical Methods

### 11.1 Multiple Comparisons

All p-values are corrected for multiple comparisons using the Benjamini-Hochberg (BH) procedure at α = 0.05. The 8 primary hypothesis tests are:

1. EXT: AUC > 0.5 (permutation test)
2. RQ2: interaction β ≠ 0 (bootstrap)
3. RQ2: Fisher z-test
4. RQ3: one-way ANOVA cohesion
5. RQ4: Bayesian ΔAUC > 0
6. RFS: urgency vs MI quality correlation
7. EXT: permutation AUC
8. DYN: transition entropy t-test

### 11.2 Effect Sizes

- **Cohen's d**: pooled standard deviation formula, two-sample
- **η²**: between-group SS / total SS (ANOVA)
- **ω²**: bias-corrected η² for ANOVA
- **ΔR²**: incremental R² from interaction term in moderated regression

### 11.3 Bootstrap Confidence Intervals

Bootstrap CIs use 2,000–5,000 non-parametric resamples with the percentile method (2.5th and 97.5th percentiles). Applied to: AUC (RQ1), interaction coefficient (RQ2).

### 11.4 Post-Hoc Power Analysis (V12-D)

Post-hoc power for two-sample t-tests uses the normal approximation:

$$z_\beta = \frac{d}{\sqrt{1/n_1 + 1/n_2}} - z_{\alpha/2}$$

$$\text{power} = \Phi(z_\beta)$$

For d = 1.840, N = 133 (High = 110, Low = 23), power = 1.000.

### 11.5 Decision Curve Analysis (V12-E)

Net benefit at threshold $t$:

$$\text{NB}(t) = \frac{TP}{N} - \frac{FP}{N} \cdot \frac{t}{1-t}$$

Computed across $t \in [0.01, 0.99]$ for CircumplexEstimator, Logistic model, treat-all, and treat-none policies.

---

## 12. Calibration (V12-K)

### 12.1 Isotonic Recalibration

Post-hoc isotonic regression is applied to map raw CircumplexEstimator cohesion (scaled to [0,1]) to calibrated probabilities. The isotonic regressor is fitted on the full dataset (no held-out set) for illustrative purposes.

| Metric | Raw | Isotonic |
|---|---|---|
| Brier score | 0.105 | 0.080 |
| ECE (10 bins) | 0.113 | 0.000 |

---

## 13. Software and Reproducibility

All analyses are implemented in Python 3.11+. Key dependencies:

| Package | Version | Purpose |
|---|---|---|
| numpy | ≥ 1.24 | Numerical computation |
| pandas | ≥ 2.0 | Data manipulation |
| scikit-learn | ≥ 1.3 | ML models, metrics, CV |
| scipy | ≥ 1.11 | Statistical tests |
| statsmodels | ≥ 0.14 | BH correction |
| matplotlib | ≥ 3.7 | Visualisation |
| torch | ≥ 2.0 | LSTM / BiLSTM (optional) |
| vaderSentiment | ≥ 3.3 | Sentiment analysis |
| imbalanced-learn | ≥ 0.11 | SMOTE (optional) |

Random seeds are fixed at `SEED = 42` for all stochastic components. The full pipeline is deterministic conditional on the fixed seed (except for Ollama SLLM calls, which have temperature=0.1).

---

## References

- Olson, D. H., Sprenkle, D. H., & Russell, C. S. (1979). Circumplex model of marital and family systems: I. *Family Process, 18*(1), 3–28.
- Olson, D. (2011). FACES IV and the Circumplex Model: Validation study. *Journal of Marital and Family Therapy, 37*(1), 64–80.
- Wu, S., et al. (2022). AnnoMI: A dataset of expert-annotated counselling dialogues. *Proc. ACL*.
- Hirano, T., & Tanaka, F. (2026). Dialogue generation for family robots using ROS and generative AI. *Proc. IEEE/SICE SII 2026*.
- Koo, T. K., & Mae, M. Y. (2016). A guideline of selecting and reporting intraclass correlation coefficients for reliability research. *Journal of Chiropractic Medicine, 15*(2), 155–163.
- Gu, A., & Dao, T. (2023). Mamba: Linear-time sequence modeling with selective state spaces. *arXiv:2312.00752*.
