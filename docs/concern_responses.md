# RFS-SCP v16.1 — Detailed Concern Responses

This document provides full responses to all reviewer and collaborator concerns raised against v12.0. Each concern is numbered to match the `§3. Major Concerns Addressed` section in the README.

---

## Concern 1 — Missing Main Script (`rfs_scp_v12_main.py`)

**Issue**: `run_full_pipeline.sh` referenced a main script that was not committed to the repository.

**Root cause**: The main script was developed and executed locally but not staged for commit. The shell script pointed to a path that existed on the development machine but not in the repository tree.

**Resolution**:
- The main script is now `src/rfs_scp_v16_main.py` and is committed.
- `scripts/run_full_pipeline.sh` updated to reference `src/rfs_scp_v16_main.py`.
- A reproducibility manifest is auto-generated at `results/reproducibility_manifest_v16.json` on every pipeline run. It records: Python version, library versions, random seed, VADER backend, AnnoMI MD5 hash, and AUC floor citation.
- `src/validate_concerns.py` check 6 independently verifies that both the main script and the shell script reference are present.

**Verification**:
```bash
bash scripts/run_validation.sh
# [PASS ✓]  src/rfs_scp_v16_main.py exists
# [PASS ✓]  run_full_pipeline.sh references rfs_scp_v16_main.py
```

---

## Concern 2 — Figure 7(A) Threshold Line (AUC = 0.55) Basis

**Issue**: The horizontal dashed line at AUC = 0.55 appeared arbitrary; red bars were coloured without explanation.

**Resolution**: The threshold is now formalised as:

```python
AUC_FLOOR = 0.55
AUC_FLOOR_CITATION = "Tanana et al. (2016) J. Subst. Abuse Treat."
```

**Citation**: Tanana, M., Hallgren, K. A., Imel, Z. E., Atkins, D. C., & Srikumar, V. (2016). A comparison of natural language processing methods for automated coding of motivational interviewing. *Journal of Substance Abuse Treatment, 65*, 43–50.

This paper reports baseline AUC ≈ 0.55 for automated MI coding, which represents the minimum validity bar above random chance (AUC = 0.50) for this task class. The threshold is embedded in the `SURROGATE_DISCLAIMER` banner, all relevant figure titles, and the Scorecard (Figure 8).

Red bars in Figure 4 (interpretability premium) now carry the explicit legend label: "Below floor (AUC < 0.55; Tanana et al. 2016)."

**Verification**:
```bash
bash scripts/run_validation.sh
# [PASS ✓]  AUC_FLOOR = 0.55 defined
# [PASS ✓]  Tanana et al. (2016) citation embedded
```

---

## Concern 3 — All RQs Confirmed: Over-Clean Results

**Issue**: Every research question was confirmed or supported, raising concerns about insufficient critical evaluation and possible p-hacking.

**Resolution**: Several results are downgraded or explicitly qualified:

| RQ | v12 Verdict | v16.1 Revised Verdict | Reason for change |
|----|------------|----------------------|------------------|
| RQ4 | INFORMATIVE ✓ | **SUPPORTED (hold-out)** / **in-sample = biased supplementary** | ΔAUC now split: hold-out CI [+0.015,+0.126] is primary; in-sample CI is labelled as biased |
| RQ5-B | SUPPORTED ✓ | **interpretability premium demonstrated via LSTM gap** | LSTM[BCE] AUC = 0.576 (near-random); gap vs. Circumplex = +0.24; this is evidence for, not against, the interpretability claim |
| §S (new row) | — | **LIMITATION ⚠** | Hold-out weight learning: theory gap +0.291, learned gap +0.155; both flagged as primary limitations |

The LOFO analysis identifies 6 of 12 features as noise or redundant (LOFO drop ≤ 0). These are reported and not suppressed.

The Scorecard (Figure 8) includes a `§S — Hold-out weight sensitivity` row explicitly marked `LIMITATION ⚠`, with both theory and learned weight train/test gaps.

---

## Concern 4 — Communication Axis: Why Excluded?

**Issue**: Olson's model includes Communication as its third dimension, which is arguably the most directly measurable from dialogue text. Exclusion requires explicit justification.

**Response**: The exclusion is a methodological limitation, not a theoretical preference.

The Communication proxy relies on five dialogue signals: question rate, turn balance, topic shift rate, clarification rate, and listener response rate. Of these, clarification rate is the most specific to the Communication dimension and is the primary distinguishing signal from the Flexibility axis.

**Coverage audit (AnnoMI-full, 13,551 utterances)**:

| Signal | Regex hit rate | Status |
|--------|---------------|--------|
| Clarification | 1.3% | ⚠ NEAR-ZERO |
| Listener response | 30.2% | ✓ Usable |
| Question rate | (already in Flexibility) | Collinear risk |
| Turn balance | (already in Cohesion) | Collinear risk |

With clarification coverage at 1.3%, the Communication score is almost entirely driven by question rate and listener responses. These features are already components of the Flexibility and Cohesion proxies respectively, creating multicollinearity that would inflate Communication ANOVA statistics without reflecting a genuinely distinct dimension.

**ANOVA results for transparency**: F = 2.050, p = 0.076, η² = 0.075. The BH-corrected p = 0.085 is not significant at α = 0.05, which is consistent with the low-quality operationalisation.

**Planned resolution (Contribution C2)**:
1. Train a clarification-act classifier using the SWBD-DAMSL dialogue act corpus (Jurafsky et al., 1997)
2. Apply to AnnoMI to re-estimate clarification coverage
3. Re-run 3-axis model if coverage exceeds 10%

The Communication axis is fully implemented in code (`CircumplexState.communication`, `W_COMM_THEORY`, `ESTIMATOR.estimate()`) and computed for all sessions. All outputs include communication scores. It is labelled **[FUTURE WORK]** throughout.

**Verification**:
```bash
bash scripts/run_validation.sh
# [PASS ✓]  Clarification coverage = 1.32% (< 10% → FUTURE WORK)
```

---

## Concern 5 — Bayesian MCMC: Arbitrary `temp=300`, No Likelihood Definition

**Issue**: The v12.0 MCMC used `exp(ΔAUC × 300)` as the acceptance ratio — an arbitrary temperature constant that is not a proper likelihood.

**Resolution**: The MCMC is completely rewritten with a proper generative model:

```
Likelihood:  y_i ~ Bernoulli(σ(α·ĉ_i(w) + β))
Prior:       w   ~ Dirichlet(α=2.0)
Posterior:   log p(w|data) = Σ [y_i log σ(ĉ_i) + (1−y_i) log(1−σ(ĉ_i))]
                            + (α−1) Σ log(w_k)
```

The acceptance ratio uses exact log-posterior differences: `log u < min(0, log_post(w') − log_post(w))`. There is no temperature constant.

**Consequences**: The dominant feature shifted from `empathy` (v12, driven by temp=300 artifact) to `wc_balance` (v16.1, proper likelihood). This is a substantive correction: turn balance appears as the most uncertain weight under the proper posterior, consistent with its PC3 loading in the PCA collinearity audit.

In-sample bias is explicitly disclosed: the posterior is computed on the same N=133 data used for cohesion estimation. The hold-out ΔAUC CI [+0.015, +0.126] is the primary claim.

**Verification**:
```bash
bash scripts/run_validation.sh
# [PASS ✓]  temp=300 absent from source
# [PASS ✓]  Bernoulli likelihood keyword present
# [PASS ✓]  Dirichlet prior keyword present
```

---

## Concern 6 — Feature Weight Derivation: Circular Reasoning Risk

**Issue**: Cohesion sub-feature weights may have been tuned on AnnoMI, making the model circular.

**Response**: The baseline weights (`BASE_W`) are theory-derived and set prior to any AnnoMI analysis. Their provenance is documented in `docs/weight_provenance.md`.

**Circularity risk acknowledged**: The feature-to-construct mapping was chosen because these features are known to correlate with MI quality. This is an inherent limitation of using a surrogate dataset with theoretical mappings. The hold-out weight analysis (§4c) quantifies this risk:

- Both theory and learned weights show train/test gaps > 0.10 at N=133
- This indicates weight-space under-identification, not necessarily circularity
- Theory weights are preferred for parsimony and transparency

**Hold-out sensitivity**: Saved in `results/holdout_weight_sensitivity_v16.csv`. The `both_gaps_flagged` column is `True`, ensuring the limitation is machine-readable.

---

## Concern 7 — AnnoMI Relevance: What Does MI Data Have to Do with FACES-IV?

**Issue**: AnnoMI contains therapist-client MI sessions; Olson's model targets family systems. The connection requires explicit justification.

**Response**: Two levels of justification:

**Level 1 — Construct mapping**: MI therapeutic alliance and family Cohesion share overlapping theoretical constructs. Reflective listening maps to emotional bonding; change talk elicitation maps to adaptability; question patterning maps to role flexibility. The full table appears in `README.md §4.3`.

**Level 2 — Surrogate methodology**: AnnoMI is used as a surrogate development corpus because no publicly available dataset combines family dialogue with FACES-IV scores. MI quality labels serve as a proxy for "relational quality," which is the target construct.

**Explicit limitation**: AnnoMI results do not constitute FACES-IV validation. The conservative generalisation estimate is AUC ≈ 0.606 (hold-out, N_test=40). Transfer to family-robot interaction requires a dedicated study (Contribution C5: controlled user study with FACES-IV ground truth).

The `SURROGATE_DISCLAIMER` banner is printed at every pipeline run and embedded in all figure titles and the reproducibility manifest.

---

## Summary of Changes by Concern

| Concern | Primary change | Validation |
|---------|---------------|-----------|
| 1 | `src/rfs_scp_v16_main.py` committed; `run_full_pipeline.sh` updated | `validate_concerns.py` check 6 |
| 2 | `AUC_FLOOR = 0.55` with Tanana et al. (2016) citation | `validate_concerns.py` check 2 |
| 3 | §S LIMITATION row in Scorecard; LOFO noise features reported | `results/holdout_weight_sensitivity_v16.csv` |
| 4 | Communication → [FUTURE WORK]; coverage audit saved | `validate_concerns.py` check 5 |
| 5 | MCMC rewritten: proper Bernoulli likelihood + Dirichlet prior | `validate_concerns.py` check 3 |
| 6 | Hold-out weight learning; both gaps reported in CSV | `validate_concerns.py` check 4 |
| 7 | `SURROGATE_DISCLAIMER` in code, all figure titles, manifest | `results/reproducibility_manifest_v16.json` |
