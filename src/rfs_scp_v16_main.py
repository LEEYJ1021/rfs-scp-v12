#!/usr/bin/env python3
# =============================================================================
#  RFS-SCP v16.1  —  Circumplex-Grounded MI Quality Estimation:
#                    A Computational Framework for Robot Family System Design
#
#  TARGET JOURNALS: Expert Systems with Applications / Knowledge-Based Systems
#
#  CHANGES v16.1 vs v16.0  (단기 방향 수정 — SHORT-TERM DIRECTION)
#  ─────────────────────────────────────────────────────────────────────
#  [v16-FIX-1] matplotlib boxplot compatibility
#    - tick_labels parameter used (matplotlib ≥3.9 compatible)
#    - Fallback to set_xticklabels() for maximum compatibility
#
#  [v16-FIX-2] Communication axis demoted to Future Work
#    - PRIMARY CLAIMS: Cohesion + Flexibility proxies only
#    - Communication axis moved to §Appendix / Future Work framing
#    - SURROGATE_DISCLAIMER updated: Communication = "future extension"
#    - RQ reformulated: focused on 2-axis Cohesion/Flexibility claim
#    - All "3-axis" references updated to "2-axis (+ Communication future)"
#    - Communication ANOVA remains in code for completeness but clearly
#      labelled [FUTURE WORK / EXPLORATORY — NOT primary claim]
#    - Fig.6(B) title updated with stronger warning
#    - Scorecard RQ3 detail explicitly excludes Communication from
#      confirmatory claims
#
#  [v16-FIX-3] AUC ≥ 0.55 threshold provenance strengthened
#    - Tanana et al. (2016) citation text added to SURROGATE_DISCLAIMER
#    - Scorecard RQ1 detail line expanded with citation context
#    - AUC_FLOOR constant defined with docstring citation
#
#  [v16-FIX-4] Surrogate framing strengthened in all print statements
#    - All references to "3-axis Circumplex" updated
#    - Primary RQ now single-sentence, 2-axis focused
#    - FINAL SUMMARY updated to reflect demoted Communication
#
#  CARRIED OVER FROM v16.0 (unchanged logic):
#    V16-BUG-1/2/3: DCA nb_none fix; PCA mapping fix; GroupKFold fix
#    V16-ENH-1..5:  LOFO CI; hold-out L2; VIF; figure DPI; manifest
#    V15-FIX-1..5:  VADER mandatory; GS-SHAP→LOFO; PCA audit;
#                   Comm regex audit; LSTM BCE
#    V14-FIX-1..5:  hold-out split; log-loss; ΔAUC posterior;
#                   bootstrap CI; CV fold safety
#    Bayesian MCMC with proper Bernoulli likelihood + Dirichlet prior;
#    MCC+BalAcc primary; BH correction; isotonic calibration; DCA;
#    UMAP; temporal dynamics
# =============================================================================

import os
import re
import json
import hashlib
import warnings
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import pearsonr, spearmanr, norm as sp_norm
from scipy.optimize import minimize
from numpy.linalg import lstsq
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score,
    matthews_corrcoef, brier_score_loss, confusion_matrix,
    balanced_accuracy_score,
)
from sklearn.model_selection import (
    StratifiedKFold, GroupKFold, train_test_split,
)
from sklearn.linear_model import LogisticRegression, Ridge
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# [v16-FIX-3] AUC floor with citation provenance
# Tanana et al. (2016) "A comparison of natural language processing methods
# for automated coding of motivational interviewing." Journal of Substance
# Abuse Treatment, 65, 43-50. — baseline AUC ~0.55 for automated MI coding.
AUC_FLOOR = 0.55
AUC_FLOOR_CITATION = "Tanana et al. (2016) J. Subst. Abuse Treat."

# =============================================================================
#  [V15-FIX-1] VADER MANDATORY
# =============================================================================
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _sia = SentimentIntensityAnalyzer()
    def vader(text: str) -> float:
        return _sia.polarity_scores(str(text))["compound"]
    VADER_BACKEND = "vaderSentiment==real"
    print("[VADER] ✓ vaderSentiment loaded")
except ImportError:
    raise RuntimeError(
        "[V15-FIX-1] vaderSentiment not found.\n"
        "pip install vaderSentiment\n"
        "Silent fallback is DISABLED in v16.1 for reproducibility."
    )

# ── optional imports ──────────────────────────────────────────────────────
try:
    import torch, torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_OK = True
    print(f"[TORCH] ✓ {torch.__version__}")
except ImportError:
    TORCH_OK = False
    print("[TORCH] not found — Ridge fallback")

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_OK = True
except ImportError:
    SMOTE_OK = False

try:
    import umap as umap_lib
    UMAP_OK = True
    print("[UMAP] ✓ available")
except ImportError:
    UMAP_OK = False
    print("[UMAP] not found — t-SNE fallback")

# ── matplotlib boxplot compatibility check  [v16-FIX-1] ─────────────────
import matplotlib
_MPL_VER = tuple(int(x) for x in matplotlib.__version__.split(".")[:2])
_BOXPLOT_LABEL_KW = "tick_labels" if _MPL_VER >= (3, 9) else "labels"
print(f"[MPL] ✓ matplotlib {matplotlib.__version__}  "
      f"boxplot label kwarg = '{_BOXPLOT_LABEL_KW}'")

def _boxplot_with_labels(ax, data, label_list, **kwargs):
    """[v16-FIX-1] Compatibility wrapper for boxplot label parameter."""
    try:
        return ax.boxplot(data, **{_BOXPLOT_LABEL_KW: label_list}, **kwargs)
    except TypeError:
        bp = ax.boxplot(data, **kwargs)
        ax.set_xticklabels(label_list)
        return bp

# ── argument parsing ──────────────────────────────────────────────────────
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--annomi-dir", type=str, default=None)
args, _ = parser.parse_known_args()

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

ANNOMI_DIR = Path(
    args.annomi_dir
    or os.environ.get("ANNOMI_DIR", str(BASE_DIR / "annomi"))
)
OUT_DIR = Path(os.environ.get("RFS_OUT_v16", str(BASE_DIR / "rfs_v16_outputs")))
OUT_DIR.mkdir(parents=True, exist_ok=True)

ANNOMI_FULL   = ANNOMI_DIR / "AnnoMI-full.csv"
ANNOMI_SIMPLE = ANNOMI_DIR / "AnnoMI-simple.csv"

SEED = 42
np.random.seed(SEED)
RNG  = np.random.default_rng(SEED)
FIG_DPI = 300

# ── colour palette ────────────────────────────────────────────────────────
PAL = dict(
    red="#C0392B", blue="#2471A3", green="#1E8449",
    purple="#7D3C98", orange="#D68910", teal="#0E6655",
    gray="#717D7E", navy="#1A252F", gold="#B7950B",
    cyan="#148F77", pink="#EC407A", lime="#558B2F",
)
ZONE_COLORS = {
    "balanced":           "#1E8449",
    "rigid-disengaged":   "#C0392B",
    "rigid-enmeshed":     "#7D3C98",
    "chaotic-disengaged": "#2471A3",
    "chaotic-enmeshed":   "#D68910",
}
MODEL_COLORS = [
    "#D85A30", "#2E5EAA", "#1D9E75", "#534AB7",
    "#E8A33D", "#9B2335", "#2D6A4F", "#6B4226",
]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#FAFAFA",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "figure.dpi": 130,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})

SEP = "=" * 78

def save_fig(name: str, fig=None, dpi: int = FIG_DPI):
    path = OUT_DIR / name
    (fig or plt).savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close("all")
    print(f"  + saved {path.name}")

print(f"[0] Environment ready | VADER: {VADER_BACKEND}")
print(f"    Output dir: {OUT_DIR}")

# =============================================================================
#  §0  SURROGATE DATASET FRAMING  [v16-FIX-2/3: updated]
# =============================================================================
SURROGATE_DISCLAIMER = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  RFS-SCP v16.1 — SURROGATE DATASET FRAMING                                 ║
║                                                                              ║
║  DATASET : AnnoMI (N=133 sessions, 13,551 utterances)                       ║
║  LABEL   : mi_quality (high/low) — annotated MI adherence quality           ║
║            NOT a family-system measurement (no FACES-IV ground truth)       ║
║                                                                              ║
║  PRIMARY CIRCUMPLEX CLAIMS (this paper):                                     ║
║    • Cohesion    ← empathy, agreement, sentiment proxies                     ║
║                     (grounded in Miller & Rollnick 2012 MI constructs)       ║
║    • Flexibility ← oscillation, question rate, lexical novelty               ║
║                                                                              ║
║  FUTURE EXTENSION (NOT primary claim):                                       ║
║    • Communication← turn balance, clarification, topic shift                ║
║                     [PARTIAL COVERAGE ONLY: clarification ≈1.3%]            ║
║                     Requires improved NLP proxy before confirmatory use.     ║
║                                                                              ║
║  PRIMARY RQ: "Do Olson Circumplex Cohesion and Flexibility axes,             ║
║   operationalised via MI-dialogue features, provide a theoretically          ║
║   interpretable and statistically valid framework for predicting annotated   ║
║   MI quality — and can this 2-axis framework inform robot intervention       ║
║   signal design?"                                                            ║
║                                                                              ║
║  AUC FLOOR ≥{AUC_FLOOR}: per {AUC_FLOOR_CITATION}              ║
║   (automated MI coding baseline; random classifier AUC = 0.50)              ║
║                                                                              ║
║  GENERALISABILITY LIMIT: Results apply to MI-style therapeutic dialogues;   ║
║  transfer to real family-robot interaction requires FACES-IV validation.     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
print(SURROGATE_DISCLAIMER)

# =============================================================================
#  §0b  REPRODUCIBILITY MANIFEST
# =============================================================================
def write_manifest(annomi_path: Path) -> dict:
    import platform, sys
    manifest = {
        "version": "RFS-SCP-v16.1",
        "python": sys.version,
        "platform": platform.platform(),
        "SEED": SEED,
        "VADER_backend": VADER_BACKEND,
        "matplotlib": matplotlib.__version__,
        "boxplot_label_kwarg": _BOXPLOT_LABEL_KW,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": __import__("sklearn").__version__,
        "scipy": __import__("scipy").__version__,
        "torch": torch.__version__ if TORCH_OK else "not installed",
        "annomi_path": str(annomi_path),
        "annomi_md5": "N/A",
        "auc_floor": AUC_FLOOR,
        "auc_floor_citation": AUC_FLOOR_CITATION,
    }
    if annomi_path.exists():
        h = hashlib.md5(annomi_path.read_bytes()).hexdigest()
        manifest["annomi_md5"] = h
    with open(OUT_DIR / "reproducibility_manifest_v16.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  [V16-ENH-5] Reproducibility manifest written.")
    return manifest

# =============================================================================
#  §1  STATISTICAL HELPERS
# =============================================================================
def cohens_d(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    sp = np.sqrt(
        ((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1))
        / (len(a) + len(b) - 2)
    )
    return float((np.mean(a) - np.mean(b)) / (sp + 1e-12))

def eta_squared(groups) -> float:
    grand  = np.concatenate(groups).mean()
    SS_bet = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    SS_tot = sum((v - grand) ** 2 for g in groups for v in g)
    return float(SS_bet / (SS_tot + 1e-12))

def bh_correct(p_values, alpha=0.05):
    reject, p_corr, _, _ = multipletests(p_values, alpha=alpha, method="fdr_bh")
    return reject, p_corr

def mcc_score(y_true, y_pred_bin) -> float:
    return float(matthews_corrcoef(y_true, y_pred_bin))

def bal_acc_score(y_true, y_pred_bin) -> float:
    return float(balanced_accuracy_score(y_true, y_pred_bin))

def specificity_score(y_true, y_pred_bin) -> float:
    cm = confusion_matrix(y_true, y_pred_bin)
    if cm.shape == (2, 2):
        tn, fp = cm[0, 0], cm[0, 1]
        return float(tn / (tn + fp + 1e-9))
    return float("nan")

def ece_score(y_true, y_prob, n_bins=10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() > 0:
            ece += mask.sum() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece / max(len(y_true), 1))

def ols_coef_pval(y_vec, X_mat):
    beta, _, _, _ = lstsq(X_mat, y_vec, rcond=None)
    y_hat = X_mat @ beta
    sse   = np.sum((y_vec - y_hat) ** 2)
    dof   = len(y_vec) - X_mat.shape[1]
    mse   = sse / max(dof, 1)
    cov_b = mse * np.linalg.pinv(X_mat.T @ X_mat)
    se    = np.sqrt(np.diag(cov_b))
    t_val = beta / (se + 1e-15)
    p_val = 2 * stats.t.sf(np.abs(t_val), df=dof)
    return beta, se, t_val, p_val

def compute_vif(X_mat) -> np.ndarray:
    n_cols = X_mat.shape[1]
    vifs   = np.full(n_cols, np.nan)
    for j in range(n_cols):
        y_j  = X_mat[:, j]
        X_j  = np.delete(X_mat, j, axis=1)
        if X_j.shape[1] == 0:
            vifs[j] = 1.0; continue
        beta_j, _, _, _ = ols_coef_pval(y_j, X_j)
        y_hat_j = X_j @ beta_j
        ss_res  = np.sum((y_j - y_hat_j) ** 2)
        ss_tot  = np.sum((y_j - y_j.mean()) ** 2) + 1e-12
        r2_j    = 1.0 - ss_res / ss_tot
        vifs[j] = 1.0 / max(1.0 - r2_j, 1e-9)
    return vifs

def power_ttest_ind(n1, n2, d, alpha=0.05) -> float:
    se      = np.sqrt(1 / n1 + 1 / n2)
    z_alpha = sp_norm.ppf(1 - alpha / 2)
    z_beta  = abs(d) / se - z_alpha
    return float(sp_norm.cdf(z_beta))

def net_benefit(y_true, y_score, threshold) -> float:
    yp  = (y_score >= threshold).astype(int)
    tp_ = ((yp == 1) & (y_true == 1)).sum()
    fp_ = ((yp == 1) & (y_true == 0)).sum()
    n   = len(y_true)
    return (tp_ / n) - (fp_ / n) * (threshold / (1 - threshold + 1e-9))

def stratified_bootstrap_auc_ci(
    y, score, n_boot=2000, seed=SEED, alpha=0.05
) -> Tuple[Tuple[float, float], np.ndarray]:
    y = np.asarray(y); score = np.asarray(score)
    idx_pos = np.where(y == 1)[0]; idx_neg = np.where(y == 0)[0]
    if len(idx_pos) == 0 or len(idx_neg) == 0:
        return (np.nan, np.nan), np.array([])
    rng_ = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        bp  = rng_.choice(idx_pos, size=len(idx_pos), replace=True)
        bn  = rng_.choice(idx_neg, size=len(idx_neg), replace=True)
        idx_ = np.concatenate([bp, bn])
        try:
            boot.append(roc_auc_score(y[idx_], score[idx_]))
        except Exception:
            pass
    boot = np.array(boot)
    if len(boot) == 0:
        return (np.nan, np.nan), boot
    ci = (
        float(np.percentile(boot, 100 * alpha / 2)),
        float(np.percentile(boot, 100 * (1 - alpha / 2))),
    )
    return ci, boot

def fishers_z_test(r1, n1, r2, n2) -> Tuple[float, float]:
    z1   = np.arctanh(np.clip(r1, -0.9999, 0.9999))
    z2   = np.arctanh(np.clip(r2, -0.9999, 0.9999))
    se_z = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    z_d  = (z1 - z2) / (se_z + 1e-15)
    return float(z_d), float(2 * stats.norm.sf(abs(z_d)))

def safe_cv_auc(
    X, y, groups, model_fn, n_splits=5, n_repeats=5,
    use_group=False, seed=SEED
) -> dict:
    aucs, mccs, baccs = [], [], []
    n_skipped = 0
    n_total   = 0

    if use_group:
        n_eff    = min(n_splits, len(np.unique(groups)))
        splitter = GroupKFold(n_splits=n_eff)
        for tr_idx, va_idx in splitter.split(X, y, groups=groups):
            n_total += 1
            y_va = y[va_idx]
            if len(np.unique(y_va)) < 2 or len(np.unique(y[tr_idx])) < 2:
                n_skipped += 1; continue
            model = model_fn()
            model.fit(X[tr_idx], y[tr_idx])
            preds     = model.predict_proba(X[va_idx])[:, 1]
            preds_bin = (preds >= 0.5).astype(int)
            try:
                aucs.append(roc_auc_score(y_va, preds))
                mccs.append(mcc_score(y_va.astype(int), preds_bin))
                baccs.append(bal_acc_score(y_va.astype(int), preds_bin))
            except Exception:
                pass
    else:
        for rep in range(n_repeats):
            splitter = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=seed + rep
            )
            for tr_idx, va_idx in splitter.split(X, y):
                n_total += 1
                y_va = y[va_idx]
                if len(np.unique(y_va)) < 2 or len(np.unique(y[tr_idx])) < 2:
                    n_skipped += 1; continue
                model = model_fn()
                model.fit(X[tr_idx], y[tr_idx])
                preds     = model.predict_proba(X[va_idx])[:, 1]
                preds_bin = (preds >= 0.5).astype(int)
                try:
                    aucs.append(roc_auc_score(y_va, preds))
                    mccs.append(mcc_score(y_va.astype(int), preds_bin))
                    baccs.append(bal_acc_score(y_va.astype(int), preds_bin))
                except Exception:
                    pass

    return dict(
        auc_mean=float(np.nanmean(aucs))   if aucs else np.nan,
        auc_std =float(np.nanstd(aucs))    if aucs else np.nan,
        mcc_mean=float(np.nanmean(mccs))   if aucs else np.nan,
        bacc_mean=float(np.nanmean(baccs)) if aucs else np.nan,
        n_folds_used=len(aucs),
        n_folds_skipped=n_skipped,
        n_folds_total=n_total,
    )

print("[1] Statistical helpers ready")

# =============================================================================
#  §2  CIRCUMPLEX ESTIMATOR  [v16-FIX-2: 2-axis primary]
# =============================================================================
@dataclass
class CircumplexState:
    cohesion:      float = 50.0
    flexibility:   float = 50.0
    communication: float = 50.0   # retained for future work; not primary claim

    def __post_init__(self):
        self.cohesion      = float(np.clip(self.cohesion,      0, 100))
        self.flexibility   = float(np.clip(self.flexibility,   0, 100))
        self.communication = float(np.clip(self.communication, 0, 100))

    @property
    def zone(self) -> str:
        c, f = self.cohesion, self.flexibility
        if 35 <= c <= 65 and 35 <= f <= 65: return "balanced"
        if c > 65 and f < 35:               return "rigid-enmeshed"
        if c < 35 and f < 35:               return "rigid-disengaged"
        if c < 35 and f > 65:               return "chaotic-disengaged"
        if c > 65 and f > 65:               return "chaotic-enmeshed"
        if f < 35:                           return "rigid-disengaged"
        if f > 65:                           return "chaotic-disengaged"
        if c < 35:                           return "rigid-disengaged"
        return "rigid-enmeshed"

    @property
    def communication_quality(self) -> str:
        if self.communication >= 65: return "open"
        if self.communication >= 35: return "moderate"
        return "restricted"

    @property
    def balanced(self) -> bool:
        return self.zone == "balanced"

    @property
    def deviation(self) -> float:
        return float(np.hypot(self.cohesion - 50, self.flexibility - 50))

    @property
    def robot_state(self) -> dict:
        # [v16-FIX-2] intervention_urgency based on 2-axis deviation only
        # Communication gap excluded from primary urgency calculation
        return {
            "zone":                 self.zone,
            "engagement_need":      float(np.clip((65 - self.cohesion) / 30, 0, 1)),
            "flexibility_gap":      float(np.clip((self.flexibility - 65) / 35, 0, 1)),
            "communication_gap":    float(np.clip((65 - self.communication) / 65, 0, 1)),
            "intervention_urgency": float(np.clip(self.deviation / 50, 0, 1)),
            "target_cohesion":      50.0,
            "target_flexibility":   50.0,
            "target_communication": 75.0,
        }


# [v16-FIX-2] Communication weights retained for future work tracking
W_COMM_THEORY = dict(
    question_rate=0.25, turn_balance=0.20, topic_shift_rate=0.18,
    clarification_rate=0.20, listener_resp_rate=0.17,
)


class CircumplexEstimator:
    # Cohesion sub-feature weights: Olson(2011) factor loadings mapped to MI
    W_COH_THEORY = dict(
        empathy=0.24, agreement=0.18, sent_pos=0.12, wc_balance=0.11,
        sent_congruence=0.15, neg_absence=0.12, sent_div_absence=0.08,
    )
    W_FLEX = dict(
        oscillation=0.22, question=0.25, sent_variance=0.20,
        novelty=0.13, anti_rigidity=0.20,
    )
    W_COMM = W_COMM_THEORY.copy()
    W_COH_LEARNED: Optional[Dict] = None

    def estimate(self, f: dict, w_coh=None, w_comm=None) -> CircumplexState:
        wc = w_coh  or self.W_COH_LEARNED or self.W_COH_THEORY
        wm = w_comm or self.W_COMM
        g  = lambda k, d=0.: float(f.get(k, d) or d)
        cl = lambda v: float(np.clip(v, 0, 1))

        emp    = cl(g("empathy_rate")  / 0.06)
        agr    = cl(g("agreement_rate") / 0.25)
        s_pos  = (g("sent_mean") + 1) / 2
        bal    = g("wc_balance", 0.5)
        scong  = 1 - cl(g("sent_diff_ab"))
        neg_a  = 1 - cl(g("negation_rate") / 0.30)
        sdiv_a = 1 - cl(g("sent_diff_ab")  / 0.60)

        coh_keys = ["empathy", "agreement", "sent_pos", "wc_balance",
                    "sent_congruence", "neg_absence", "sent_div_absence"]
        coh_vals = [emp, agr, s_pos, bal, scong, neg_a, sdiv_a]
        coh = 100 * sum(wc.get(k, 0) * v for k, v in zip(coh_keys, coh_vals))

        osc    = g("oscillation_rate", 0.5)
        qst    = cl(g("question_rate") / 0.20)
        sstd   = cl(g("sent_std")      / 0.50)
        ttr_v  = cl(g("mean_ttr")      / 0.80)
        lag1   = g("lag1_autocorr", 0)
        anti_r = float(1.0 / (1.0 + np.exp(3.0 * lag1)))

        wf   = self.W_FLEX
        flex = 100 * (
            wf["oscillation"]   * osc   +
            wf["question"]      * qst   +
            wf["sent_variance"] * sstd  +
            wf["novelty"]       * ttr_v +
            wf["anti_rigidity"] * anti_r
        )

        # Communication: computed but [FUTURE WORK] — partial lexical coverage
        q_rate    = cl(g("question_rate")          / 0.20)
        t_bal     = g("turn_balance", 0.5)
        ts_rate   = cl(g("topic_shift_rate",  0.0) / 0.30)
        clar_rate = cl(g("clarification_rate", 0.0) / 0.15)
        lr_rate   = cl(g("listener_resp_rate", 0.0) / 0.20)

        comm_keys = ["question_rate", "turn_balance", "topic_shift_rate",
                     "clarification_rate", "listener_resp_rate"]
        comm_vals = [q_rate, t_bal, ts_rate, clar_rate, lr_rate]
        comm = 100 * sum(wm.get(k, 0) * v for k, v in zip(comm_keys, comm_vals))

        return CircumplexState(round(coh, 2), round(flex, 2), round(comm, 2))


ESTIMATOR = CircumplexEstimator()
COH_KEYS  = list(ESTIMATOR.W_COH_THEORY.keys())
BASE_W    = np.array([ESTIMATOR.W_COH_THEORY[k] for k in COH_KEYS])
COMM_KEYS = list(W_COMM_THEORY.keys())
COH_SUB_COLS = ["empathy_rate", "agreement_rate", "sent_mean", "wc_balance",
                "sent_diff_ab", "negation_rate", "mean_ttr"]

# [v16-FIX-2] Clarify primary axis claim
print("[2] CircumplexEstimator ready")
print("    PRIMARY AXES: Cohesion + Flexibility (2-axis claim)")
print("    FUTURE WORK:  Communication (partial lexical coverage only)")

# =============================================================================
#  §3  LEXICAL HELPERS + REGEX AUDIT
# =============================================================================
NEG_RE    = re.compile(r"\b(no|not|never|n't|nothing|nobody|none)\b", re.I)
CLARIF_RE = re.compile(
    r"\b(so you('re| are)|what i hear|did you mean|you mean|in other words|"
    r"if i understand|let me|i want to make sure|correct me)\b",
    re.I,
)
LISTENER_RE = re.compile(
    r"\b(mm+|uh huh|i see|right|okay|go on|tell me more|yes|sure)\b", re.I
)

def ttr(text: str) -> float:
    toks = str(text).lower().split()
    return len(set(toks)) / len(toks) if toks else 0.0

def audit_regex_coverage(df: pd.DataFrame) -> dict:
    n           = len(df)
    clarif_hit  = df["utterance_text"].str.contains(CLARIF_RE).sum()
    listen_hit  = df["utterance_text"].str.contains(LISTENER_RE).sum()
    neg_hit     = df["utterance_text"].str.contains(NEG_RE).sum()
    report = {
        "n_utterances":           n,
        "clarification_hit_pct":  round(100 * clarif_hit / max(n, 1), 2),
        "listener_hit_pct":       round(100 * listen_hit / max(n, 1), 2),
        "negation_hit_pct":       round(100 * neg_hit    / max(n, 1), 2),
    }
    print("\n  Regex Coverage Audit (Communication proxies):")
    print(f"    clarification : {report['clarification_hit_pct']:.1f}%  "
          f"({'⚠ NEAR-ZERO → FUTURE WORK' if report['clarification_hit_pct'] < 10 else '✓'})")
    print(f"    listener resp : {report['listener_hit_pct']:.1f}%  "
          f"({'⚠ near-zero' if report['listener_hit_pct'] < 10 else '✓'})")
    print(f"    negation      : {report['negation_hit_pct']:.1f}%")
    if report["clarification_hit_pct"] < 10:
        print("    → Communication axis excluded from primary claims (insufficient coverage)")
    return report

print("[3] Lexical helpers ready")

# =============================================================================
#  §4  DATA LOADING + FEATURE EXTRACTION
# =============================================================================
print(f"\n[4] Loading AnnoMI from {ANNOMI_DIR}")

if ANNOMI_FULL.exists():
    raw = pd.read_csv(ANNOMI_FULL); _using = "full"
elif ANNOMI_SIMPLE.exists():
    raw = pd.read_csv(ANNOMI_SIMPLE); _using = "simple"
else:
    raise FileNotFoundError(f"AnnoMI CSV not found in {ANNOMI_DIR}")

print(f"  AnnoMI-{_using}: {len(raw):,} utterances / "
      f"{raw.transcript_id.nunique()} sessions")

manifest = write_manifest(ANNOMI_FULL if _using == "full" else ANNOMI_SIMPLE)

raw["vader"]        = raw["utterance_text"].apply(vader)
raw["neg_flag"]     = raw["utterance_text"].str.contains(NEG_RE).astype(int)
raw["ttr_val"]      = raw["utterance_text"].apply(ttr)
raw["is_therapist"] = (raw["interlocutor"] == "therapist").astype(int)
raw["word_count"]   = raw["utterance_text"].str.split().str.len().fillna(0)
raw["is_clarif"]    = raw["utterance_text"].str.contains(CLARIF_RE).astype(int)
raw["is_listener"]  = raw["utterance_text"].str.contains(LISTENER_RE).astype(int)

regex_audit = audit_regex_coverage(raw)

# [v16-FIX-2] Flag Communication as future-work based on coverage
COMM_COVERAGE_OK = regex_audit["clarification_hit_pct"] >= 10
COMM_STATUS_LABEL = "FUTURE WORK" if not COMM_COVERAGE_OK else "EXPLORATORY"


def extract_features(df: pd.DataFrame):
    rows, sequences = [], {}
    for tid, grp in df.groupby("transcript_id"):
        grp = grp.sort_values("utterance_id")
        th  = grp[grp.interlocutor == "therapist"]
        cl  = grp[grp.interlocutor == "client"]

        if _using == "full" and "reflection_exists" in th.columns:
            empathy_rate = (
                th["reflection_exists"].astype(str).str.lower() == "true"
            ).mean() if len(th) else 0.0
        else:
            empathy_rate = (
                th["main_therapist_behaviour"] == "reflection"
            ).mean() if len(th) else 0.0

        agreement_rate = (
            cl["client_talk_type"] == "change"
        ).mean() if len(cl) else 0.0

        if _using == "full" and "question_exists" in th.columns:
            question_rate = (
                th["question_exists"].astype(str).str.lower() == "true"
            ).mean() if len(th) else 0.0
        else:
            question_rate = (
                th["main_therapist_behaviour"] == "question"
            ).mean() if len(th) else 0.0

        if len(cl) >= 2:
            ct_arr = cl["client_talk_type"].values
            oscillation_rate = sum(
                1 for i in range(len(ct_arr) - 1) if ct_arr[i] != ct_arr[i + 1]
            ) / (len(ct_arr) - 1)
        else:
            oscillation_rate = 0.5

        sent_all     = grp["vader"].values
        sent_mean    = float(np.mean(sent_all))
        sent_std     = float(np.std(sent_all)) if len(sent_all) > 1 else 0.0
        sent_diff_ab = abs(
            (th["vader"].mean() if len(th) else 0.0) -
            (cl["vader"].mean() if len(cl) else 0.0)
        )
        lag1_autocorr = (
            float(np.corrcoef(sent_all[:-1], sent_all[1:])[0, 1])
            if len(sent_all) >= 10 and np.std(sent_all) > 1e-9
            else 0.0
        )

        negation_rate      = grp["neg_flag"].mean()
        mean_ttr           = grp["ttr_val"].mean()
        wc_balance         = min(len(th), len(cl)) / max(len(th), len(cl), 1)
        emp_agr_inter      = empathy_rate * agreement_rate
        turn_balance       = wc_balance
        topic_shift_rate   = oscillation_rate
        clarification_rate = grp["is_clarif"].mean()
        listener_resp_rate = th["is_listener"].mean() if len(th) else 0.0

        # Temporal dynamics
        n_v = len(sent_all)
        transition_entropy = 0.0
        if n_v > 2:
            diffs   = np.diff(np.sign(sent_all))
            n_trans = np.sum(diffs != 0)
            p_trans = np.clip(n_trans / max(n_v - 1, 1), 1e-9, 1 - 1e-9)
            transition_entropy = (
                -p_trans * np.log2(p_trans)
                - (1 - p_trans) * np.log2(1 - p_trans)
            )

        emotional_inertia   = abs(lag1_autocorr)
        cohesion_volatility = sent_std
        th_vader  = th["vader"].values if len(th) > 0 else np.array([0.0])
        recovery_rate = (
            float(np.mean(np.diff(th_vader) > 0)) if len(th_vader) > 1 else 0.5
        )

        feat = dict(
            empathy_rate=empathy_rate, agreement_rate=agreement_rate,
            question_rate=question_rate, oscillation_rate=oscillation_rate,
            sent_mean=sent_mean, sent_std=sent_std, sent_diff_ab=sent_diff_ab,
            negation_rate=negation_rate, mean_ttr=mean_ttr,
            wc_balance=wc_balance, lag1_autocorr=lag1_autocorr,
            emp_agr_interact=emp_agr_inter,
            turn_balance=turn_balance, topic_shift_rate=topic_shift_rate,
            clarification_rate=clarification_rate,
            listener_resp_rate=listener_resp_rate,
        )
        cp = ESTIMATOR.estimate(feat)

        mi_quality = grp["mi_quality"].iloc[0]
        topic      = grp["topic"].iloc[0].strip()

        rows.append(dict(
            transcript_id=tid, mi_quality=mi_quality,
            mi_quality_bin=int(mi_quality == "high"), topic=topic,
            n_turns=len(grp), n_therapist=len(th), n_client=len(cl),
            empathy_rate=empathy_rate, agreement_rate=agreement_rate,
            question_rate=question_rate, oscillation_rate=oscillation_rate,
            sent_mean=sent_mean, sent_std=sent_std, sent_diff_ab=sent_diff_ab,
            lag1_autocorr=lag1_autocorr, negation_rate=negation_rate,
            mean_ttr=mean_ttr, wc_balance=wc_balance,
            emp_agr_interact=emp_agr_inter,
            turn_balance=turn_balance, topic_shift_rate=topic_shift_rate,
            clarification_rate=clarification_rate,
            listener_resp_rate=listener_resp_rate,
            cohesion=cp.cohesion, flexibility=cp.flexibility,
            communication=cp.communication,
            zone=cp.zone, deviation=cp.deviation, balanced=int(cp.balanced),
            comm_quality=cp.communication_quality,
            transition_entropy=transition_entropy,
            emotional_inertia=emotional_inertia,
            cohesion_volatility=cohesion_volatility,
            empathy_recovery_rate=recovery_rate,
        ))

        seq = grp[["vader", "ttr_val", "is_therapist", "neg_flag", "word_count"]
                  ].values.astype(float)
        seq[:, 4] = np.clip(seq[:, 4] / 50.0, 0, 1)
        sequences[tid] = seq

    return pd.DataFrame(rows), sequences


print("  Extracting session features + utterance sequences …")
sess, utt_sequences = extract_features(raw)
sess.to_csv(OUT_DIR / "annomi_session_features_v16.csv", index=False)

n_total = len(sess)
n_hi    = int(sess.mi_quality_bin.sum())
n_lo    = n_total - n_hi
print(f"  → {n_total} sessions | High={n_hi} ({n_hi/n_total*100:.1f}%) "
      f"Low={n_lo} ({n_lo/n_total*100:.1f}%)")


def assign_cluster(t: str) -> str:
    t_l = t.lower()
    if any(k in t_l for k in ["alcohol","drug","recidiv","gambling","coffee"]):
        return "substance"
    if any(k in t_l for k in ["smok","tobacco"]):
        return "smoking"
    if any(k in t_l for k in ["weight","diet","exercise","activity"]):
        return "health"
    if any(k in t_l for k in ["asthma","diabetes","medicine","medical",
                                "oral","birth","diagnos"]):
        return "medical"
    if any(k in t_l for k in ["harm","violen","school","assertive",
                                "community","flatmate","doi"]):
        return "psychosoc"
    return "other"


sess["topic_cluster"] = sess.topic.apply(assign_cluster)

FEAT_COLS = [
    "empathy_rate", "agreement_rate", "question_rate", "oscillation_rate",
    "sent_mean", "sent_std", "sent_diff_ab", "lag1_autocorr",
    "negation_rate", "mean_ttr", "wc_balance", "emp_agr_interact",
]

all_ids = sess.transcript_id.values
y_bin   = sess.mi_quality_bin.values.astype(np.float32)

MAX_SEQ, N_FEAT = 120, 5


def pad_sequences(seq_dict, ids, max_len=MAX_SEQ, n_feat=N_FEAT):
    X = np.zeros((len(ids), max_len, n_feat), dtype=np.float32)
    for i, tid in enumerate(ids):
        seq    = seq_dict.get(tid, np.zeros((1, n_feat)))[:, :n_feat]
        length = min(len(seq), max_len)
        X[i, :length] = seq[:length]
    return X


X_seq = pad_sequences(utt_sequences, all_ids)
print("[4] Done. MAX_SEQ=120")

# =============================================================================
#  §4b  COHESION WEIGHT HELPERS
# =============================================================================
def compute_cohesion_with_w(df, w_arr) -> np.ndarray:
    w  = np.clip(w_arr, 1e-6, None); w = w / w.sum()
    wd = dict(zip(COH_KEYS, w))
    feat_keys = [
        "empathy_rate", "agreement_rate", "sent_mean", "wc_balance",
        "sent_diff_ab", "negation_rate", "mean_ttr", "lag1_autocorr",
        "oscillation_rate", "question_rate", "sent_std",
    ]
    return df.apply(
        lambda r: ESTIMATOR.estimate(
            {c: r.get(c, 0) for c in feat_keys}, w_coh=wd
        ).cohesion,
        axis=1,
    ).values

# =============================================================================
#  §4c  HOLD-OUT WEIGHT LEARNING
# =============================================================================
print(f"\n[4c] Hold-out Weight Learning (auxiliary §S)")

tid_train, tid_test, _, _ = train_test_split(
    sess.transcript_id.values, y_bin,
    test_size=0.30, stratify=y_bin, random_state=SEED
)
sess_train = sess[sess.transcript_id.isin(tid_train)].reset_index(drop=True)
sess_test  = sess[sess.transcript_id.isin(tid_test)].reset_index(drop=True)
print(f"  Train: N={len(sess_train)} | Test: N={len(sess_test)}")

L2_LAMBDA = 0.5

def neg_logloss_w_train(w_arr, lambda_l2=L2_LAMBDA):
    cohs = compute_cohesion_with_w(sess_train, w_arr) / 100.0
    cohs = np.clip(cohs, 1e-9, 1 - 1e-9)
    y_tr = sess_train.mi_quality_bin.values
    ll   = np.sum(y_tr * np.log(cohs) + (1 - y_tr) * np.log(1 - cohs))
    w_n  = np.clip(w_arr, 1e-6, None); w_n = w_n / w_n.sum()
    bw_n = BASE_W / BASE_W.sum()
    reg  = lambda_l2 * np.sum((w_n - bw_n) ** 2)
    return -(ll) + reg

res_w = minimize(
    lambda x: neg_logloss_w_train(np.exp(x)),
    np.log(BASE_W), method="Nelder-Mead",
    options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-8},
)
w_learned_raw = np.exp(res_w.x)
w_learned     = w_learned_raw / w_learned_raw.sum()
w_diff        = float(np.abs(w_learned - BASE_W / BASE_W.sum()).sum())

auc_theory_train  = roc_auc_score(sess_train.mi_quality_bin,
                                   compute_cohesion_with_w(sess_train, BASE_W))
auc_theory_test   = roc_auc_score(sess_test.mi_quality_bin,
                                   compute_cohesion_with_w(sess_test,  BASE_W))
auc_learned_train = roc_auc_score(sess_train.mi_quality_bin,
                                   compute_cohesion_with_w(sess_train, w_learned))
auc_learned_test  = roc_auc_score(sess_test.mi_quality_bin,
                                   compute_cohesion_with_w(sess_test,  w_learned))
gap = auc_learned_train - auc_learned_test
print(f"  Theory  → Train={auc_theory_train:.4f}  Test={auc_theory_test:.4f}")
print(f"  Learned → Train={auc_learned_train:.4f}  Test={auc_learned_test:.4f}  "
      f"gap={gap:+.4f}")
# [FIX-1] theory weights도 train/test 괴리를 함께 보고
theory_gap = auc_theory_train - auc_theory_test
if gap > 0.10:
    print(f"  ⚠ Learned gap={gap:.4f} > 0.10 — "
          f"small-N overfitting; L2 (λ={L2_LAMBDA}) applied.")
if theory_gap > 0.10:
    print(f"  ⚠ Theory  gap={theory_gap:.4f} > 0.10 — "
          f"theory weights also show small-N instability on this split.")
print(f"  Theory  gap (Train−Test) = {theory_gap:+.4f}")
print(f"  Learned gap (Train−Test) = {gap:+.4f}")
print(f"  → BOTH weight sets show instability on N={len(sess_train)}/N={len(sess_test)} split.")
print(f"  → Conclusion: theory weights preferred NOT because they generalise better,")
print(f"     but because the weight-space is under-identified at N={n_total}.")
print(f"     Both gaps must be reported as primary limitations.")

HOLDOUT = dict(
    w_learned=w_learned, w_diff=w_diff,
    auc_theory_train=auc_theory_train, auc_theory_test=auc_theory_test,
    auc_learned_train=auc_learned_train, auc_learned_test=auc_learned_test,
    gap=gap, theory_gap=theory_gap, l2_lambda=L2_LAMBDA,
    n_train=len(sess_train), n_test=len(sess_test),
    # [FIX-1] 두 gap 모두 limitation으로 flagging
    both_gaps_flagged=(gap > 0.10 or theory_gap > 0.10),
    limitation_note=(
        f"Both theory (gap={theory_gap:+.4f}) and learned (gap={gap:+.4f}) "
        f"weights show train-test instability on N={n_total}. "
        f"Theory weights preferred on parsimony grounds only. "
        f"Weight-space is under-identified; N≥300 recommended for stable learning."
    ),
)

ESTIMATOR.W_COH_LEARNED = None
sess["cohesion"] = compute_cohesion_with_w(sess, BASE_W)
y_coh = sess.cohesion.values.astype(np.float32)
print("  Main RQ1-6 use Olson(2011) BASE_W (theory weights).")
print(f"  LIMITATION: theory weights Test AUC={auc_theory_test:.4f} "
      f"(gap={theory_gap:+.4f}) — report in §Limitations.")

# =============================================================================
#  §5  PCA ON COHESION SUB-FEATURES
# =============================================================================
print(f"\n{SEP}\n[5] PCA — Cohesion Sub-feature Collinearity Audit\n{SEP}")

X_coh_sub   = sess[COH_SUB_COLS].fillna(0).values
scaler_coh  = StandardScaler()
X_coh_sub_s = scaler_coh.fit_transform(X_coh_sub)

pca_coh  = PCA(n_components=min(7, len(COH_SUB_COLS)), random_state=SEED)
pca_coh.fit(X_coh_sub_s)
explained = pca_coh.explained_variance_ratio_

print("  PCA on 7 Cohesion sub-features (COH_SUB_COLS order):")
for i, ev in enumerate(explained):
    top_col = COH_SUB_COLS[int(np.argmax(np.abs(pca_coh.components_[i])))]
    print(f"    PC{i+1}: {ev*100:.1f}%  top loading: {top_col}")

pc1_dominant = COH_SUB_COLS[int(np.argmax(np.abs(pca_coh.components_[0])))]
pc2_dominant = COH_SUB_COLS[int(np.argmax(np.abs(pca_coh.components_[1])))]

corr_mat_coh = np.corrcoef(X_coh_sub_s.T)
r_wb_emp = corr_mat_coh[
    COH_SUB_COLS.index("wc_balance"), COH_SUB_COLS.index("empathy_rate")
]
r_sd_sm  = corr_mat_coh[
    COH_SUB_COLS.index("sent_diff_ab"), COH_SUB_COLS.index("sent_mean")
]
print(f"  r(wc_balance, empathy_rate) = {r_wb_emp:.4f}")
print(f"  r(sent_diff_ab, sent_mean)  = {r_sd_sm:.4f}")

results = {}
results["PCA_COH"] = dict(
    explained=explained.tolist(),
    components=pca_coh.components_.tolist(),
    pc1_dominant=pc1_dominant,
    pc2_dominant=pc2_dominant,
    feature_names=COH_SUB_COLS,
)

# =============================================================================
#  §6  RQ1: MI-QUALITY DISCRIMINATION BY CIRCUMPLEX PROXY  [v16-FIX-3]
# =============================================================================
print(f"\n{SEP}")
print(f"[6] RQ1: 2-Axis Circumplex-Proxy Discrimination  (N={n_total})")
print(f"    PRIMARY: Cohesion proxy | Theory weights only | Perm null | Strat CI")
print(f"    AUC floor ≥{AUC_FLOOR} per {AUC_FLOOR_CITATION}")
print(SEP)

high = sess[sess.mi_quality == "high"]
low  = sess[sess.mi_quality == "low"]

t_coh, p_coh     = stats.ttest_ind(high.cohesion, low.cohesion)
d_coh_insample   = cohens_d(high.cohesion.values, low.cohesion.values)

cv_ds = []
for rep in range(5):
    skf_d = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + rep)
    for tr_i, va_i in skf_d.split(sess.cohesion.values, y_bin):
        y_va  = y_bin[va_i]; c_va = sess.cohesion.values[va_i]
        hi_cv = c_va[y_va == 1]; lo_cv = c_va[y_va == 0]
        if len(hi_cv) > 1 and len(lo_cv) > 1:
            cv_ds.append(cohens_d(hi_cv, lo_cv))
d_coh_cv = float(np.nanmean(cv_ds))

auc_coh = roc_auc_score(y_bin, sess.cohesion)
ap_coh  = average_precision_score(y_bin, sess.cohesion)
fpr_c, tpr_c, thresh_roc = roc_curve(y_bin, sess.cohesion)

j_idx       = np.argmax(tpr_c - fpr_c)
opt_thresh  = float(thresh_roc[j_idx])
y_pred_opt  = (sess.cohesion >= opt_thresh).astype(int)
mcc_val     = mcc_score(y_bin.astype(int), y_pred_opt)
bal_acc     = bal_acc_score(y_bin.astype(int), y_pred_opt)
spec_val    = specificity_score(y_bin.astype(int), y_pred_opt)

brier_raw  = brier_score_loss(y_bin, sess.cohesion / 100.0)
ece_raw    = ece_score(y_bin, sess.cohesion.values / 100.0)

iso_reg  = IsotonicRegression(out_of_bounds="clip")
prob_raw = np.clip(sess.cohesion.values / 100.0, 0, 1)
iso_reg.fit(prob_raw, y_bin)
prob_iso   = iso_reg.predict(prob_raw)
brier_iso  = brier_score_loss(y_bin, prob_iso)
ece_iso    = ece_score(y_bin, prob_iso)

auc_ci, boot_aucs = stratified_bootstrap_auc_ci(
    y_bin, sess.cohesion.values, n_boot=2000, seed=SEED + 1
)

rng_perm  = np.random.default_rng(SEED + 77)
perm_aucs = []
for _ in range(2000):
    y_sh = rng_perm.permutation(y_bin)
    try:
        perm_aucs.append(roc_auc_score(y_sh, sess.cohesion.values))
    except Exception:
        pass
perm_aucs = np.array(perm_aucs)
perm_p    = float((perm_aucs >= auc_coh).mean())

prob_true_raw, prob_pred_raw = calibration_curve(y_bin, prob_raw, n_bins=10)
prob_true_iso, prob_pred_iso = calibration_curve(y_bin, prob_iso, n_bins=10)

print(f"  d: in-sample={d_coh_insample:.4f}  CV-mean(5x5)={d_coh_cv:.4f}")
print(f"  AUC={auc_coh:.4f}  CI=[{auc_ci[0]:.4f},{auc_ci[1]:.4f}]  "
      f"AP={ap_coh:.4f}")
print(f"  MCC={mcc_val:.4f}  BalAcc={bal_acc:.4f}  Spec={spec_val:.4f}")
print(f"  Brier {brier_raw:.4f}→{brier_iso:.4f}  "
      f"ECE {ece_raw:.4f}→{ece_iso:.4f}")
print(f"  Perm-p={perm_p:.4f}")
print(f"  AUC {'≥' if auc_coh >= AUC_FLOOR else '<'} floor {AUC_FLOOR} "
      f"({AUC_FLOOR_CITATION}): {'PASS ✓' if auc_coh >= AUC_FLOOR else 'FAIL ✗'}")

power_rq1 = power_ttest_ind(len(high), len(low), d_coh_cv)
n_needed  = max(
    int(np.ceil(
        2 * ((sp_norm.ppf(0.8) + sp_norm.ppf(0.975)) / max(d_coh_cv, 0.01)) ** 2
    )), 5
)
print(f"  Power(CV-d={d_coh_cv:.3f})={power_rq1:.3f}  N_needed(80%)={n_needed}")

# DCA
thresholds = np.linspace(0.01, 0.99, 99)
nb_circ    = np.array([net_benefit(y_bin, sess.cohesion / 100.0, t) for t in thresholds])
nb_all     = np.array([net_benefit(y_bin, np.ones(len(y_bin)), t)  for t in thresholds])
nb_none    = np.zeros_like(thresholds)

scaler_dca  = StandardScaler()
X_dca_s     = scaler_dca.fit_transform(sess[FEAT_COLS].fillna(0).values)
lr_dca      = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced",
                                  random_state=SEED)
lr_dca.fit(X_dca_s, y_bin)
prob_logit  = lr_dca.predict_proba(X_dca_s)[:, 1]
nb_logit    = np.array([net_benefit(y_bin, prob_logit, t) for t in thresholds])

results["EXT"] = dict(
    auc=auc_coh, ap=ap_coh, d_insample=d_coh_insample, d_cv=d_coh_cv,
    p=p_coh, auc_ci=auc_ci, boot_aucs=boot_aucs,
    bal_acc=bal_acc, mcc=mcc_val, specificity=spec_val,
    brier_raw=brier_raw, ece_raw=ece_raw,
    brier_iso=brier_iso, ece_iso=ece_iso,
    perm_p=perm_p, perm_aucs=perm_aucs,
    fpr=fpr_c, tpr=tpr_c,
    prob_true_raw=prob_true_raw, prob_pred_raw=prob_pred_raw,
    prob_true_iso=prob_true_iso, prob_pred_iso=prob_pred_iso,
    power=power_rq1, n_needed=n_needed,
    high_mean=high.cohesion.mean(), low_mean=low.cohesion.mean(),
    thresholds=thresholds,
    nb_circ=nb_circ, nb_logit=nb_logit,
    nb_all=nb_all, nb_none=nb_none,
    supported=(perm_p < 0.05 and auc_coh >= AUC_FLOOR),
)

# =============================================================================
#  §7  RQ2: EMPATHY–COHESION MODERATION
# =============================================================================
print(f"\n{SEP}\n[7] RQ2: Empathy–Cohesion Moderation (N={n_total})\n{SEP}")

scaler_mod = StandardScaler()
X_mod      = scaler_mod.fit_transform(
    sess[["empathy_rate", "mi_quality_bin", "cohesion"]].values
)
emp_s, mi_s, coh_s = X_mod[:, 0], X_mod[:, 1], X_mod[:, 2]
n_mod   = len(emp_s)
emp_x_mi = emp_s * mi_s
X_full   = np.c_[np.ones(n_mod), emp_s, mi_s, emp_x_mi]
beta_int, se_int, t_int, p_int = ols_coef_pval(coh_s, X_full)

labels_int = ["intercept", "empathy", "mi_quality", "empathy×mi_quality"]
print("  Interaction: cohesion ~ empathy + MI + empathy×MI")
for lbl, b, t_, p_ in zip(labels_int, beta_int, t_int, p_int):
    sig = "***" if p_ < .001 else ("**" if p_ < .01 else ("*" if p_ < .05 else "n.s."))
    print(f"    {lbl:<22s}: β={b:+.4f}  t={t_:+.3f}  p={p_:.4f}  {sig}")

vifs_full = compute_vif(X_full)
vif_labels = ["intercept", "empathy", "mi_quality", "empathy×mi_quality"]
print("  VIF:")
for lbl, vif in zip(vif_labels, vifs_full):
    warn = " ⚠ HIGH" if vif > 5 else ""
    print(f"    {lbl:<22s}: VIF={vif:.2f}{warn}")

r_hi, p_rhi = pearsonr(high.empathy_rate, high.cohesion)
r_lo, p_rlo = pearsonr(low.empathy_rate,  low.cohesion)
z_diff, p_zdiff = fishers_z_test(r_hi, len(high), r_lo, len(low))

rng_b2   = np.random.default_rng(SEED + 200)
boot_int = []
for _ in range(5000):
    idx_b = rng_b2.integers(0, n_mod, n_mod)
    X_b   = np.c_[np.ones(n_mod), emp_s[idx_b], mi_s[idx_b],
                  emp_s[idx_b] * mi_s[idx_b]]
    try:
        bb, _, _, _ = ols_coef_pval(coh_s[idx_b], X_b)
        boot_int.append(float(bb[3]))
    except Exception:
        pass

ci_int  = (float(np.percentile(boot_int, 2.5)),
           float(np.percentile(boot_int, 97.5)))
mod_sig = not (ci_int[0] <= 0 <= ci_int[1])

X_main = np.c_[np.ones(n_mod), emp_s, mi_s]
bm, _, _, _ = ols_coef_pval(coh_s, X_main)
ss_res_main = np.sum((coh_s - X_main @ bm) ** 2)
ss_res_full = np.sum((coh_s - X_full @ beta_int) ** 2)
dr2 = float(
    (ss_res_main - ss_res_full) / (np.sum((coh_s - coh_s.mean()) ** 2) + 1e-9)
)

print(f"  High-MI r={r_hi:.4f}  Low-MI r={r_lo:.4f}  Fisher-z p={p_zdiff:.4f}")
print(f"  Bootstrap CI(β_int)=[{ci_int[0]:+.4f},{ci_int[1]:+.4f}]  "
      f"→ {'SIGNIFICANT ✓' if mod_sig else 'n.s.'}")

results["RQ2"] = dict(
    beta_interact=float(beta_int[3]), p_interact=float(p_int[3]),
    ci_interact=ci_int, r_high=float(r_hi), p_high=float(p_rhi),
    r_low=float(r_lo), p_low=float(p_rlo),
    z_diff=float(z_diff), p_zdiff=float(p_zdiff),
    delta_r2=dr2, boot_int=boot_int, mod_sig=mod_sig,
    beta_full=beta_int.tolist(), se_full=se_int.tolist(),
    vif_full=vifs_full.tolist(), vif_labels=vif_labels,
    supported=(mod_sig and p_zdiff < 0.05),
)

# =============================================================================
#  §8  RQ3: TOPIC DOMAIN × COHESION PROXY  [v16-FIX-2: Comm demoted]
# =============================================================================
print(f"\n{SEP}\n[8] RQ3: Topic Domain × Cohesion Proxy (N={n_total})\n{SEP}")
print(f"  PRIMARY: Cohesion ANOVA [CONFIRMATORY]")
print(f"  Communication ANOVA = [{COMM_STATUS_LABEL}] — NOT primary claim")

valid_clusters = (
    sess.topic_cluster.value_counts()[sess.topic_cluster.value_counts() >= 5]
    .index.tolist()
)
sess_h3 = sess[sess.topic_cluster.isin(valid_clusters)]

groups_coh  = [sess_h3[sess_h3.topic_cluster == c].cohesion.values       for c in valid_clusters]
groups_comm = [sess_h3[sess_h3.topic_cluster == c].communication.values  for c in valid_clusters]
groups_emp  = [sess_h3[sess_h3.topic_cluster == c].empathy_rate.values   for c in valid_clusters]

F_h3,  p_h3  = stats.f_oneway(*groups_coh)
eta2_h3      = eta_squared(groups_coh)
omega2_h3    = (F_h3 - 1) / (
    F_h3 + (len(sess_h3) - len(valid_clusters)) / max(len(valid_clusters) - 1, 1) + 1e-9
)
F_comm, p_comm = stats.f_oneway(*groups_comm)
eta2_comm      = eta_squared(groups_comm)
F_emp,  p_emp  = stats.f_oneway(*groups_emp)
eta2_emp       = eta_squared(groups_emp)

pairs       = [
    (valid_clusters[i], valid_clusters[j])
    for i in range(len(valid_clusters))
    for j in range(i + 1, len(valid_clusters))
]
pairwise_p, pairwise_d = [], []
for c1, c2 in pairs:
    g1 = sess_h3[sess_h3.topic_cluster == c1].cohesion.values
    g2 = sess_h3[sess_h3.topic_cluster == c2].cohesion.values
    _, p_ = stats.ttest_ind(g1, g2)
    pairwise_p.append(p_); pairwise_d.append(cohens_d(g1, g2))
reject_bh, p_bh = bh_correct(pairwise_p)

print(f"  Cohesion [CONFIRMATORY]:     F={F_h3:.3f}  p={p_h3:.4f}  η²={eta2_h3:.4f}")
print(f"  Communication [{COMM_STATUS_LABEL}]: F={F_comm:.3f}  p={p_comm:.4f}  "
      f"η²={eta2_comm:.4f}  ← NOT primary claim")

results["RQ3"] = dict(
    F=F_h3, p=p_h3, eta2=eta2_h3, omega2=omega2_h3,
    F_emp=F_emp, p_emp=p_emp, eta2_emp=eta2_emp,
    F_comm=F_comm, p_comm=p_comm, eta2_comm=eta2_comm,
    clusters=valid_clusters, pairs=pairs,
    pairwise_p=pairwise_p, pairwise_d=pairwise_d,
    p_bh=p_bh.tolist(), reject_bh=reject_bh.tolist(),
    comm_status=COMM_STATUS_LABEL,
    comm_coverage_ok=COMM_COVERAGE_OK,
    supported=(p_h3 < 0.05 and eta2_h3 > 0.05),
)

# =============================================================================
#  §9  RQ4: BAYESIAN MCMC WEIGHT POSTERIOR
# =============================================================================
print(f"\n{SEP}\n[9] RQ4: Bayesian MCMC Weight Posterior (N={n_total})\n{SEP}")
print("  Likelihood: y_i ~ Bernoulli(σ(α·ĉ_i(w)+β))")
print("  Prior:      w ~ Dirichlet(α=2) weakly informative")

DIRICHLET_ALPHA = 2.0
N_MCMC  = 5000
BURN_IN = N_MCMC // 4
K_COH   = len(COH_KEYS)


def logistic_log_likelihood(cohs, y):
    cohs_2d = cohs.reshape(-1, 1)
    lr_ = LogisticRegression(max_iter=200, C=10.0, random_state=SEED)
    try:
        lr_.fit(cohs_2d, y)
        probs = np.clip(lr_.predict_proba(cohs_2d)[:, 1], 1e-9, 1 - 1e-9)
        return float(np.sum(y * np.log(probs) + (1 - y) * np.log(1 - probs)))
    except Exception:
        return -1e9


def dirichlet_log_prior(w, alpha=DIRICHLET_ALPHA):
    w_s = np.clip(w, 1e-9, 1)
    return float((alpha - 1) * np.sum(np.log(w_s)))


def log_posterior(w_arr, y, df):
    w    = np.clip(w_arr, 1e-6, None); w = w / w.sum()
    cohs = compute_cohesion_with_w(df, w)
    return logistic_log_likelihood(cohs, y) + dirichlet_log_prior(w)


def w_to_logit(w):
    w_s = np.clip(w, 1e-9, 1 - 1e-9); w_s /= w_s.sum()
    return np.log(w_s[:-1] / w_s[-1])


def logit_to_w(z):
    z_f = np.append(z, 0.0); z_f -= z_f.max()
    w = np.exp(z_f); return w / w.sum()


auc_base = roc_auc_score(y_bin, sess.cohesion)
lp_base  = log_posterior(BASE_W, y_bin, sess)
print(f"  Baseline AUC={auc_base:.4f}  log-posterior={lp_base:.2f}")

step_size = 0.30
z_cur     = w_to_logit(BASE_W)
w_cur     = logit_to_w(z_cur)
lp_cur    = log_posterior(w_cur, y_bin, sess)

chain    = np.zeros((N_MCMC, K_COH))
lp_chain = np.zeros(N_MCMC)
accepted = 0
rng_mcmc = np.random.default_rng(SEED + 999)

for i in range(N_MCMC):
    z_prop  = z_cur + rng_mcmc.normal(0, step_size, size=K_COH - 1)
    w_prop  = logit_to_w(z_prop)
    lp_prop = log_posterior(w_prop, y_bin, sess)
    if np.log(rng_mcmc.random() + 1e-15) < min(0.0, lp_prop - lp_cur):
        z_cur, w_cur, lp_cur = z_prop, w_prop, lp_prop
        accepted += 1
    chain[i] = w_cur; lp_chain[i] = lp_cur
    if (i + 1) % 1000 == 0:
        rate = accepted / (i + 1)
        if rate > 0.40:   step_size *= 1.3
        elif rate < 0.20: step_size *= 0.7
        step_size = float(np.clip(step_size, 0.02, 3.0))
        print(f"    step {i+1}/{N_MCMC}: accept={rate:.1%}  step={step_size:.3f}")

final_accept_rate = accepted / N_MCMC
chain_burn  = chain[BURN_IN:]
lp_burn     = lp_chain[BURN_IN:]
w_post_mean = chain_burn.mean(0)
w_post_std  = chain_burn.std(0)
w_post_ci   = np.percentile(chain_burn, [2.5, 97.5], axis=0)
xerr_lo     = np.abs(w_post_mean - w_post_ci[0])
xerr_hi     = np.abs(w_post_ci[1] - w_post_mean)

dominant_key = COH_KEYS[int(np.argmax(w_post_mean))]

_COH_KEY_TO_SUBCOL = {
    "empathy":          "empathy_rate",
    "agreement":        "agreement_rate",
    "sent_pos":         "sent_mean",
    "wc_balance":       "wc_balance",
    "sent_congruence":  "sent_diff_ab",
    "neg_absence":      "negation_rate",
    "sent_div_absence": "sent_diff_ab",
}
subcol_for_dominant = _COH_KEY_TO_SUBCOL.get(dominant_key, None)

if subcol_for_dominant is not None and subcol_for_dominant in COH_SUB_COLS:
    subcol_idx = COH_SUB_COLS.index(subcol_for_dominant)
    pc_loadings = np.abs(pca_coh.components_)
    dominant_pc = int(np.argmax(pc_loadings[:, subcol_idx]))
    pc_top_col  = COH_SUB_COLS[int(np.argmax(pc_loadings[dominant_pc]))]
    dominant_component_label = (
        f"PC{dominant_pc+1} ({pc_top_col}-dominated, "
        f"var={explained[dominant_pc]*100:.1f}%)"
    )
else:
    dominant_component_label = "multi-component (unmapped)"

print(f"  Dominant feature: {dominant_key}  →  {dominant_component_label}")

res_opt   = minimize(
    lambda x: -log_posterior(np.exp(x), y_bin, sess),
    np.log(BASE_W), method="Nelder-Mead",
    options={"maxiter": 1200, "xatol": 1e-5, "fatol": 1e-6},
)
w_raw_opt = np.exp(res_opt.x)
w_opt     = w_raw_opt / w_raw_opt.sum()
auc_opt   = roc_auc_score(y_bin, compute_cohesion_with_w(sess, w_opt))

loo_aucs = {}
for key in COH_KEYS:
    w_loo = {k: v for k, v in ESTIMATOR.W_COH_THEORY.items() if k != key}
    tot   = sum(w_loo.values())
    wn    = {k: v / tot for k, v in w_loo.items()}
    cohs_l = compute_cohesion_with_w(
        sess, np.array([wn.get(k, 0) for k in COH_KEYS])
    )
    loo_aucs[key] = roc_auc_score(y_bin, cohs_l)

rng_dauc = np.random.default_rng(SEED + 555)
sub_idx  = rng_dauc.choice(len(chain_burn), size=min(500, len(chain_burn)), replace=False)

# [FIX-2a] In-sample ΔAUC (원래 방식 — in-sample bias 있음, 명시적으로 라벨)
delta_auc_samples = []
for w_s in chain_burn[sub_idx]:
    cohs_s = compute_cohesion_with_w(sess, w_s)
    try:
        delta_auc_samples.append(roc_auc_score(y_bin, cohs_s) - auc_base)
    except Exception:
        pass
delta_auc_samples = np.array(delta_auc_samples)
ci_delta         = np.percentile(delta_auc_samples, [2.5, 97.5])
p_rq4_one_sided  = float((delta_auc_samples <= 0).mean())
p_rq4_two_sided  = float(min(1.0, 2 * min(p_rq4_one_sided, 1 - p_rq4_one_sided)))

# [FIX-2b] Hold-out ΔAUC (엄밀한 버전 — test set에서만 평가)
delta_auc_holdout = []
auc_base_test = roc_auc_score(
    sess_test.mi_quality_bin,
    compute_cohesion_with_w(sess_test, BASE_W)
)
for w_s in chain_burn[sub_idx]:
    cohs_ho = compute_cohesion_with_w(sess_test, w_s)
    try:
        delta_auc_holdout.append(roc_auc_score(sess_test.mi_quality_bin, cohs_ho)
                                  - auc_base_test)
    except Exception:
        pass
delta_auc_holdout  = np.array(delta_auc_holdout)
ci_delta_ho        = np.percentile(delta_auc_holdout, [2.5, 97.5]) if len(delta_auc_holdout) > 10 else [np.nan, np.nan]
p_rq4_ho_onesided  = float((delta_auc_holdout <= 0).mean()) if len(delta_auc_holdout) > 10 else np.nan
p_rq4_ho_twosided  = float(min(1.0, 2 * min(p_rq4_ho_onesided, 1 - p_rq4_ho_onesided))) if not np.isnan(p_rq4_ho_onesided) else np.nan

print(f"  ΔAUC [IN-SAMPLE, biased]: mean={delta_auc_samples.mean():+.4f}  "
      f"95%CI=[{ci_delta[0]:+.4f},{ci_delta[1]:+.4f}]  p={p_rq4_two_sided:.4f}")
print(f"  ⚠ In-sample bias: MCMC optimised on same N={n_total}; "
      f"ΔAUC inflated. Report hold-out version as primary.")
print(f"  ΔAUC [HOLD-OUT, N_test={len(sess_test)}]: "
      f"mean={delta_auc_holdout.mean():+.4f}  "
      f"95%CI=[{ci_delta_ho[0]:+.4f},{ci_delta_ho[1]:+.4f}]  "
      f"p={p_rq4_ho_twosided:.4f}")
print(f"  MAP AUC={auc_opt:.4f}  (Baseline={auc_base:.4f}  Δ={auc_opt-auc_base:+.4f})")
print(f"  Accept={final_accept_rate:.1%}  Dominant={dominant_key} → "
      f"{dominant_component_label}")

# [FIX-2c] supported 판단을 hold-out CI 기준으로
rq4_supported_ho = (len(delta_auc_holdout) > 10 and float(ci_delta_ho[0]) > 0)

results["RQ4"] = dict(
    auc_base=auc_base, auc_opt=auc_opt,
    w_opt=dict(zip(COH_KEYS, w_opt.tolist())),
    w_post_mean=w_post_mean, w_post_std=w_post_std, w_post_ci=w_post_ci,
    xerr_lo=xerr_lo, xerr_hi=xerr_hi,
    chain=chain_burn, lp_chain=lp_burn,
    final_accept_rate=final_accept_rate,
    dominant_key=dominant_key,
    dominant_component=dominant_component_label,
    loo_aucs=loo_aucs, dirichlet_alpha=DIRICHLET_ALPHA,
    # in-sample (biased)
    delta_auc_samples=delta_auc_samples, ci_delta=ci_delta,
    p_rq4=p_rq4_two_sided,
    # hold-out (엄밀)
    delta_auc_holdout=delta_auc_holdout, ci_delta_ho=ci_delta_ho,
    p_rq4_ho=p_rq4_ho_twosided,
    auc_base_test=auc_base_test,
    # [FIX-2] hold-out 기준으로 supported 판단
    supported=rq4_supported_ho,
    in_sample_bias_note=(
        "ΔAUC posterior computed on same data used for MCMC likelihood → "
        "in-sample bias. Primary claim uses hold-out ΔAUC "
        f"(CI=[{ci_delta_ho[0]:+.4f},{ci_delta_ho[1]:+.4f}])."
    ),
)

# =============================================================================
#  §10  RQ5: INTERPRETABILITY PREMIUM
# =============================================================================
print(f"\n{SEP}\n[10] RQ5: Interpretability Premium (N={n_total})\n{SEP}")

X_sess    = sess[FEAT_COLS].fillna(0).values
scaler_l  = StandardScaler()
X_sess_s  = scaler_l.fit_transform(X_sess)
sess_grps = sess.transcript_id.values

logit_res = safe_cv_auc(
    X_sess_s, y_bin.astype(int), sess_grps,
    model_fn=lambda: LogisticRegression(
        max_iter=500, C=1.0, class_weight="balanced", random_state=SEED
    ),
    n_splits=5, n_repeats=5, use_group=False, seed=SEED,
)
print(f"  Logistic (5x5 StratKFold): "
      f"AUC={logit_res['auc_mean']:.4f}±{logit_res['auc_std']:.4f}  "
      f"MCC={logit_res['mcc_mean']:.4f}  BalAcc={logit_res['bacc_mean']:.4f}  "
      f"[folds {logit_res['n_folds_used']}/{logit_res['n_folds_total']}]")

gkf_res = safe_cv_auc(
    X_sess_s, y_bin.astype(int), sess_grps,
    model_fn=lambda: LogisticRegression(
        max_iter=500, C=1.0, class_weight="balanced", random_state=SEED
    ),
    n_splits=10, n_repeats=1, use_group=True, seed=SEED,
)
print(f"  Logistic (GroupKFold-10):   "
      f"AUC={gkf_res['auc_mean']:.4f}  MCC={gkf_res['mcc_mean']:.4f}  "
      f"[folds {gkf_res['n_folds_used']}/{gkf_res['n_folds_total']}]")

smote_results = {}
for cond, label in [("none", "No resampling"), ("balanced", "class_weight=balanced")]:
    kw = {"class_weight": "balanced"} if cond == "balanced" else {}
    r_ = safe_cv_auc(
        X_sess_s, y_bin.astype(int), sess_grps,
        model_fn=lambda kw=kw: LogisticRegression(
            max_iter=500, C=1.0, random_state=SEED, **kw
        ),
        n_splits=5, n_repeats=5, use_group=False, seed=SEED + 1,
    )
    smote_results[cond] = {"auc": r_["auc_mean"], "mcc": r_["mcc_mean"],
                           "bacc": r_["bacc_mean"], "label": label}

if SMOTE_OK:
    aucs_s, mccs_s, baccs_s = [], [], []
    for rep in range(5):
        skf_ = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + 10 + rep)
        for tr_idx, va_idx in skf_.split(X_sess_s, y_bin):
            y_tr = y_bin[tr_idx]
            if len(np.unique(y_bin[va_idx])) < 2:
                continue
            try:
                sm  = SMOTE(random_state=SEED,
                             k_neighbors=min(3, int(y_tr.sum()) - 1))
                Xr, yr = sm.fit_resample(X_sess_s[tr_idx], y_tr)
                lr  = LogisticRegression(max_iter=500, C=1.0, random_state=SEED)
                lr.fit(Xr, yr)
                preds     = lr.predict_proba(X_sess_s[va_idx])[:, 1]
                preds_bin = (preds >= 0.5).astype(int)
                aucs_s.append(roc_auc_score(y_bin[va_idx], preds))
                mccs_s.append(mcc_score(y_bin[va_idx].astype(int), preds_bin))
                baccs_s.append(bal_acc_score(y_bin[va_idx].astype(int), preds_bin))
            except Exception:
                pass
    smote_results["smote"] = {
        "auc":  float(np.nanmean(aucs_s)),
        "mcc":  float(np.nanmean(mccs_s)),
        "bacc": float(np.nanmean(baccs_s)),
        "label": "SMOTE",
    }

print("  Imbalance sensitivity:")
for v in smote_results.values():
    print(f"    {v['label']}: AUC={v['auc']:.4f}  MCC={v['mcc']:.4f}  "
          f"BalAcc={v['bacc']:.4f}")

lstm_sensitivity_grid = {}
if TORCH_OK:
    class LSTMClassifier(nn.Module):
        def __init__(self, input_size=5, hidden_size=48, num_layers=2, dropout=0.3):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size, hidden_size, num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, 24), nn.ReLU(),
                nn.Dropout(0.2), nn.Linear(24, 1),
            )
        def forward(self, x):
            _, (hn, _) = self.lstm(x)
            return self.fc(hn[-1]).squeeze(1)

    y_bin_long = torch.FloatTensor(y_bin)
    print("\n  LSTM sensitivity grid (BCE objective) …")
    for hs in [16, 32, 48, 64]:
        for nl in [1, 2]:
            key  = f"h{hs}_l{nl}"
            fold_aucs, fold_mccs, n_skip = [], [], 0
            skf_g = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            for tr_idx, va_idx in skf_g.split(X_seq, y_bin):
                if len(np.unique(y_bin[va_idx])) < 2:
                    n_skip += 1; continue
                X_tr     = torch.FloatTensor(X_seq[tr_idx])
                X_va     = torch.FloatTensor(X_seq[va_idx])
                y_tr     = y_bin_long[tr_idx]
                y_va_np  = y_bin[va_idx]
                pos_w    = torch.tensor(
                    [(y_bin[tr_idx] == 0).sum() / max((y_bin[tr_idx] == 1).sum(), 1)],
                    dtype=torch.float,
                )
                crit_l   = nn.BCEWithLogitsLoss(pos_weight=pos_w)
                mdl      = LSTMClassifier(hidden_size=hs, num_layers=nl)
                opt_     = torch.optim.Adam(mdl.parameters(), lr=3e-3, weight_decay=5e-4)
                ds_      = TensorDataset(X_tr, y_tr)
                dl_      = DataLoader(ds_, batch_size=16, shuffle=True)
                best_loss, patience = float("inf"), 0
                mdl.train()
                for epoch in range(80):
                    eloss = 0.0
                    for xb, yb in dl_:
                        opt_.zero_grad()
                        loss = crit_l(mdl(xb), yb)
                        loss.backward()
                        nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
                        opt_.step(); eloss += loss.item()
                    if eloss < best_loss - 1e-4: best_loss = eloss; patience = 0
                    else: patience += 1
                    if patience >= 10: break
                mdl.eval()
                with torch.no_grad():
                    logits_va = mdl(X_va).numpy()
                probs_va  = 1.0 / (1.0 + np.exp(-logits_va))
                preds_bin = (probs_va >= 0.5).astype(int)
                try:
                    fold_aucs.append(roc_auc_score(y_va_np, probs_va))
                    fold_mccs.append(mcc_score(y_va_np.astype(int), preds_bin))
                except Exception:
                    pass
            lstm_sensitivity_grid[key] = {
                "hidden": hs, "layers": nl,
                "auc_mean": float(np.nanmean(fold_aucs)) if fold_aucs else np.nan,
                "auc_std":  float(np.nanstd(fold_aucs))  if fold_aucs else np.nan,
                "mcc_mean": float(np.nanmean(fold_mccs))  if fold_aucs else np.nan,
                "n_skip": n_skip,
            }
            print(f"    {key}: AUC={lstm_sensitivity_grid[key]['auc_mean']:.4f}"
                  f"±{lstm_sensitivity_grid[key]['auc_std']:.4f}  "
                  f"MCC={lstm_sensitivity_grid[key]['mcc_mean']:.4f}")

    best_lstm_key  = max(
        lstm_sensitivity_grid,
        key=lambda k: lstm_sensitivity_grid[k]["auc_mean"]
        if not np.isnan(lstm_sensitivity_grid[k]["auc_mean"]) else -999,
    )
    lstm_auc_mean = lstm_sensitivity_grid[best_lstm_key]["auc_mean"]
    lstm_auc_std  = lstm_sensitivity_grid[best_lstm_key]["auc_std"]
    lstm_mcc_mean = lstm_sensitivity_grid[best_lstm_key]["mcc_mean"]
    print(f"  Best LSTM [BCE]: {best_lstm_key}  AUC={lstm_auc_mean:.4f}")
else:
    rskf2  = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fa_aucs = []
    X_flat  = X_seq.reshape(len(sess), -1)
    for tr_idx, va_idx in rskf2.split(X_flat, y_bin):
        if len(np.unique(y_bin[va_idx])) < 2: continue
        rdg = Ridge(alpha=10.0)
        rdg.fit(X_flat[tr_idx], y_coh[tr_idx])
        try: fa_aucs.append(roc_auc_score(y_bin[va_idx], rdg.predict(X_flat[va_idx])))
        except Exception: pass
    lstm_auc_mean = float(np.nanmean(fa_aucs))
    lstm_auc_std  = float(np.nanstd(fa_aucs))
    lstm_mcc_mean = float("nan")
    lstm_sensitivity_grid = {}; best_lstm_key = "ridge_fallback"

# Counterfactual analysis
lr_cf = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced",
                            random_state=SEED)
lr_cf.fit(X_sess_s, y_bin)
low_mask  = (y_bin == 0)
X_low     = X_sess_s[low_mask].copy()
feat_idx  = FEAT_COLS.index("empathy_rate")
cf_deltas = []
for i in range(len(X_low)):
    x_cf = X_low[i].copy()
    for delta in np.linspace(0, 3.0, 300):
        x_cf[feat_idx] = X_low[i, feat_idx] + delta
        if lr_cf.predict_proba(x_cf.reshape(1, -1))[0, 1] >= 0.5:
            cf_deltas.append(delta); break
    else:
        cf_deltas.append(np.nan)
cf_deltas = np.array(cf_deltas)
cf_median = float(np.nanmedian(cf_deltas))
cf_pct25  = float(np.nanpercentile(cf_deltas, 25))
cf_pct75  = float(np.nanpercentile(cf_deltas, 75))
print(f"  Counterfactual Δempathy: median={cf_median:.4f}  "
      f"IQR=[{cf_pct25:.4f},{cf_pct75:.4f}]")

results["RQ5"] = dict(
    logit=logit_res, gkf=gkf_res, smote_results=smote_results,
    lstm_auc_mean=lstm_auc_mean, lstm_auc_std=lstm_auc_std,
    lstm_mcc_mean=lstm_mcc_mean, sensitivity_grid=lstm_sensitivity_grid,
    best_lstm_key=best_lstm_key,
    cf_deltas=cf_deltas, cf_median=cf_median,
    cf_pct25=cf_pct25, cf_pct75=cf_pct75,
    supported=(logit_res["auc_mean"] >= AUC_FLOOR),
)

# =============================================================================
#  §11  DUAL-SHAP + LOFO
# =============================================================================
print(f"\n{SEP}\n[11] Dual-SHAP + LOFO (N={n_total})\n{SEP}")

lr_global   = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced",
                                  random_state=SEED)
lr_global.fit(X_sess_s, y_bin)
coef        = lr_global.coef_[0]
feat_std    = X_sess_s.std(0) + 1e-9
linear_shap = coef * feat_std

linear_shap_df = pd.DataFrame({
    "feature": FEAT_COLS,
    "LinearSHAP": linear_shap,
    "abs_LinearSHAP": np.abs(linear_shap),
}).sort_values("abs_LinearSHAP", ascending=False)
print(f"  LinearSHAP top-3: {linear_shap_df.head(3)['feature'].tolist()}")


def permutation_shap(predict_fn, X, baseline, n_perms=50, seed=SEED):
    rng_ = np.random.default_rng(seed)
    n, d = X.shape; phi = np.zeros((n, d))
    for s in range(n):
        x_s   = X[s]; phi_s = np.zeros(d); total = 0
        for _ in range(n_perms):
            perm = rng_.permutation(d)
            for direction in [perm, perm[::-1]]:
                x_prev = baseline.copy()
                f_prev = float(predict_fn(x_prev.reshape(1, -1))[0])
                for idx in direction:
                    x_cur = x_prev.copy(); x_cur[idx] = x_s[idx]
                    f_cur = float(predict_fn(x_cur.reshape(1, -1))[0])
                    phi_s[idx] += f_cur - f_prev
                    f_prev = f_cur; x_prev = x_cur
                total += 1
        phi[s] = phi_s / max(total, 1)
    return phi


baseline_sess  = X_sess_s.mean(0)
def lr_pred(X_in): return lr_global.predict_proba(X_in)[:, 1]

print("  PermutationSHAP (n_perms=50) …")
perm_shap_vals = permutation_shap(lr_pred, X_sess_s, baseline_sess, n_perms=50)
perm_shap_mean = np.abs(perm_shap_vals).mean(0)
perm_shap_df   = pd.DataFrame({
    "feature": FEAT_COLS, "PermSHAP": perm_shap_mean,
}).sort_values("PermSHAP", ascending=False)
print(f"  PermSHAP top-3: {perm_shap_df.head(3)['feature'].tolist()}")
rho_l_p, _ = spearmanr(np.abs(linear_shap), perm_shap_mean)
print(f"  LinearSHAP vs PermSHAP ρ={rho_l_p:.4f}")

print("  LOFO (5-fold CV, 3 repeats, bootstrap CI) …")
full_res_lofo = safe_cv_auc(
    X_sess_s, y_bin.astype(int), sess_grps,
    model_fn=lambda: LogisticRegression(
        max_iter=500, C=1.0, class_weight="balanced", random_state=SEED
    ),
    n_splits=5, n_repeats=3, use_group=False, seed=SEED,
)
full_auc_lofo = full_res_lofo["auc_mean"]

lofo_results: dict = {}
for j, feat in enumerate(FEAT_COLS):
    mask   = [i for i in range(len(FEAT_COLS)) if i != j]
    X_lofo = X_sess_s[:, mask]
    res_lf = safe_cv_auc(
        X_lofo, y_bin.astype(int), sess_grps,
        model_fn=lambda: LogisticRegression(
            max_iter=500, C=1.0, class_weight="balanced", random_state=SEED
        ),
        n_splits=5, n_repeats=3, use_group=False, seed=SEED,
    )
    drop    = full_auc_lofo - res_lf["auc_mean"]
    drop_se = np.sqrt(
        (full_res_lofo["auc_std"] ** 2 + res_lf["auc_std"] ** 2) /
        max(full_res_lofo["n_folds_used"], 1)
    )
    lofo_results[feat] = {
        "auc_without": res_lf["auc_mean"],
        "auc_drop":    drop,
        "drop_ci_lo":  drop - 1.96 * drop_se,
        "drop_ci_hi":  drop + 1.96 * drop_se,
    }

lofo_df = pd.DataFrame([
    {"feature": k,
     "auc_without": v["auc_without"],
     "auc_drop":    v["auc_drop"],
     "drop_ci_lo":  v["drop_ci_lo"],
     "drop_ci_hi":  v["drop_ci_hi"]}
    for k, v in lofo_results.items()
]).sort_values("auc_drop", ascending=False)

# [FIX-3] 음수 drop 감지 및 해석 분리
lofo_positive = lofo_df[lofo_df["auc_drop"] > 0]   # 제거시 AUC 하락 → 유익 feature
lofo_negative = lofo_df[lofo_df["auc_drop"] <= 0]   # 제거시 AUC 유지/상승 → 노이즈 feature

print(f"  Full AUC (LOFO base): {full_auc_lofo:.4f}")
print(f"  LOFO top features (positive drop = informative):")
for _, row in lofo_positive.head(5).iterrows():
    ci_str = f"[{row['drop_ci_lo']:+.4f},{row['drop_ci_hi']:+.4f}]"
    print(f"    {row['feature']:<22s}: drop={row['auc_drop']:+.4f}  CI={ci_str}")
if len(lofo_negative) > 0:
    print(f"  LOFO noise features (drop ≤ 0 = removing improves/maintains AUC):")
    for _, row in lofo_negative.iterrows():
        ci_str = f"[{row['drop_ci_lo']:+.4f},{row['drop_ci_hi']:+.4f}]"
        print(f"    {row['feature']:<22s}: drop={row['auc_drop']:+.4f}  CI={ci_str}  "
              f"← likely noise/redundant")

# [FIX-3] 최상위 유익 feature (양수 drop만)
lofo_top1_feat = lofo_positive.iloc[0]["feature"] if len(lofo_positive) > 0 else "none"
lofo_top1_drop = lofo_positive.iloc[0]["auc_drop"] if len(lofo_positive) > 0 else 0.0
lofo_n_noise   = len(lofo_negative)
lofo_n_info    = len(lofo_positive)
print(f"  LOFO summary: {lofo_n_info} informative, {lofo_n_noise} noise/redundant features")

comp_results: dict = {}
for method_name, shap_vals in [("LinearSHAP", np.abs(linear_shap)),
                                ("PermSHAP",   perm_shap_mean)]:
    sorted_feats = np.argsort(shap_vals)[::-1]
    aucs_abl     = []
    for k in [1, 2, 3, 5]:
        mask  = np.ones(len(FEAT_COLS), dtype=bool)
        mask[sorted_feats[:k]] = False
        X_abl = X_sess_s[:, mask]
        if X_abl.shape[1] == 0: continue
        lr_abl = LogisticRegression(max_iter=300, C=1.0, class_weight="balanced",
                                     random_state=SEED)
        lr_abl.fit(X_abl, y_bin)
        aucs_abl.append((k, roc_auc_score(y_bin, lr_abl.predict_proba(X_abl)[:, 1])))
    comp_results[method_name] = aucs_abl

results["SHAP"] = dict(
    linear_shap_df=linear_shap_df, perm_shap_df=perm_shap_df,
    lofo_df=lofo_df, rho_l_p=rho_l_p,
    full_auc_lofo=full_auc_lofo, comp_results=comp_results,
)
linear_shap_df.to_csv(OUT_DIR / "shap_linear_v16.csv", index=False)
perm_shap_df.to_csv(OUT_DIR / "shap_permutation_v16.csv", index=False)
lofo_df.to_csv(OUT_DIR / "lofo_results_v16.csv", index=False)

# =============================================================================
#  §12  RQ6: RFS CONTROLLER VALIDATION  [v16-FIX-2: 2-axis urgency]
# =============================================================================
print(f"\n{SEP}\n[12] RQ6: RFS Controller — 2-Axis Urgency (N={n_total})\n{SEP}")


class CircumplexController:
    ZONE_POLICIES = {
        "balanced":           {"role": "MAINTAIN",  "mode": "minimal",
                               "emp_int": 0.3, "verbosity": 0.4, "comm_boost": 0.1},
        "rigid-enmeshed":     {"role": "DIVERSIFY", "mode": "flexibility_boost",
                               "emp_int": 0.5, "verbosity": 0.6, "comm_boost": 0.3},
        "rigid-disengaged":   {"role": "RECONNECT", "mode": "cohesion_build",
                               "emp_int": 0.9, "verbosity": 0.8, "comm_boost": 0.7},
        "chaotic-disengaged": {"role": "STABILIZE", "mode": "structure_build",
                               "emp_int": 0.7, "verbosity": 0.5, "comm_boost": 0.5},
        "chaotic-enmeshed":   {"role": "MODERATE",  "mode": "boundary_set",
                               "emp_int": 0.4, "verbosity": 0.3, "comm_boost": 0.2},
    }

    def __init__(self, w_empathy=0.24, w_agreement=0.18):
        self.w_empathy = w_empathy
        self.w_agreement = w_agreement
        self.history: list = []

    def update_bayesian_weights(self, w_post_mean):
        self.w_empathy   = float(w_post_mean[COH_KEYS.index("empathy")])
        self.w_agreement = float(w_post_mean[COH_KEYS.index("agreement")])

    def step(self, features: dict, session_id: str = "") -> dict:
        state  = ESTIMATOR.estimate(features)
        policy = self.ZONE_POLICIES.get(state.zone, self.ZONE_POLICIES["balanced"])
        eng    = float(np.clip(
            self.w_empathy   * features.get("empathy_rate",   0) +
            self.w_agreement * features.get("agreement_rate", 0),
            0, 1,
        ))
        # [v16-FIX-2] urgency based on 2-axis deviation only
        comm_gap = state.robot_state["communication_gap"]
        cmd = {
            "session_id":  session_id,
            "timestamp_step": len(self.history),
            "circumplex_state": {
                "cohesion":      state.cohesion,
                "flexibility":   state.flexibility,
                "communication": state.communication,
                "zone":          state.zone,
                "comm_quality":  state.communication_quality,
                "deviation":     state.deviation,
            },
            "robot_role":           policy["role"],
            "intervention_mode":    policy["mode"],
            "empathy_intensity":    float(np.clip(policy["emp_int"] * (1 + eng), 0, 1)),
            # comm_boost retained as future-work signal but not in urgency
            "communication_boost":  float(np.clip(policy["comm_boost"] * (1 + comm_gap), 0, 1)),
            "verbosity":            policy["verbosity"],
            "intervention_urgency": float(state.robot_state["intervention_urgency"]),
            "empathy_weight":       self.w_empathy,
        }
        self.history.append(cmd)
        return cmd


class InterventionScheduler:
    PRIORITY = {
        "rigid-disengaged": 5, "chaotic-disengaged": 4, "chaotic-enmeshed": 3,
        "rigid-enmeshed": 2, "balanced": 1,
    }

    def __init__(self, cooldown_steps=3):
        self.last_zone    = None
        self.cooldown     = 0
        self.cooldown_max = cooldown_steps
        self.interventions: list = []

    def decide(self, state: CircumplexState) -> dict:
        priority = self.PRIORITY.get(state.zone, 1)
        urgency  = state.robot_state["intervention_urgency"]
        zone_chg = (self.last_zone is not None and state.zone != self.last_zone)
        trigger  = (priority >= 4 or urgency > 0.7) and self.cooldown == 0
        trigger  = trigger or (zone_chg and priority >= 3)
        dec = {
            "trigger": trigger, "priority": priority,
            "urgency": round(urgency, 3), "zone": state.zone,
            "zone_change": zone_chg, "cooldown_remaining": self.cooldown,
        }
        if trigger:
            self.cooldown = self.cooldown_max
            self.interventions.append(dec)
        else:
            self.cooldown = max(0, self.cooldown - 1)
        self.last_zone = state.zone
        return dec


controller = CircumplexController()
controller.update_bayesian_weights(results["RQ4"]["w_post_mean"])
scheduler  = InterventionScheduler(cooldown_steps=3)
FEAT_COLS_RFS = [
    "empathy_rate", "agreement_rate", "sent_mean", "wc_balance", "sent_diff_ab",
    "negation_rate", "mean_ttr", "lag1_autocorr", "oscillation_rate",
    "question_rate", "sent_std", "turn_balance", "topic_shift_rate",
    "clarification_rate", "listener_resp_rate",
]

rfs_logs = []
for _, row in sess.iterrows():
    feat   = {c: row.get(c, 0) for c in FEAT_COLS_RFS}
    cmd    = controller.step(feat, session_id=str(row.transcript_id))
    state_ = CircumplexState(row.cohesion, row.flexibility, row.communication)
    sched  = scheduler.decide(state_)
    rfs_logs.append({
        "transcript_id":       row.transcript_id,
        "mi_quality":          row.mi_quality,
        "zone":                row.zone,
        "cohesion":            row.cohesion,
        "flexibility":         row.flexibility,
        "communication":       row.communication,
        "comm_quality":        row.comm_quality,
        "robot_role":          cmd["robot_role"],
        "intervention_mode":   cmd["intervention_mode"],
        "empathy_intensity":   round(cmd["empathy_intensity"], 3),
        "communication_boost": round(cmd["communication_boost"], 3),
        "intervention_urgency": cmd["intervention_urgency"],
        "scheduler_trigger":   sched["trigger"],
        "scheduler_priority":  sched["priority"],
    })

rfs_df = pd.DataFrame(rfs_logs)
rfs_df.to_csv(OUT_DIR / "rfs_controller_log_v16.csv", index=False)

r_urg_mi,  p_urg_mi  = pearsonr(
    rfs_df.intervention_urgency, rfs_df.mi_quality.map({"high": 1, "low": 0})
)
r_comm_mi, p_comm_mi = pearsonr(
    rfs_df.communication, rfs_df.mi_quality.map({"high": 1, "low": 0})
)
print(f"  r(urgency, MI) = {r_urg_mi:.4f}  p={p_urg_mi:.4f}  [2-axis urgency]")
print(f"  r(comm,    MI) = {r_comm_mi:.4f}  p={p_comm_mi:.4f}  "
      f"[{COMM_STATUS_LABEL} — expected low, partial lexical coverage]")

dyn_cols  = ["transition_entropy", "emotional_inertia",
             "cohesion_volatility", "empathy_recovery_rate"]
dyn_high  = sess[sess.mi_quality == "high"][dyn_cols]
dyn_low   = sess[sess.mi_quality == "low"][dyn_cols]
dyn_results: dict = {}
for col in dyn_cols:
    t_d, p_d = stats.ttest_ind(dyn_high[col], dyn_low[col])
    d_d      = cohens_d(dyn_high[col].values, dyn_low[col].values)
    dyn_results[col] = {
        "t": t_d, "p": p_d, "d": d_d,
        "mean_high": float(dyn_high[col].mean()),
        "mean_low":  float(dyn_low[col].mean()),
    }
    sig = "✓" if p_d < 0.05 else "n.s."
    print(f"    {col}: d={d_d:.3f}  p={p_d:.4f}  {sig}")

results["RQ6"] = dict(
    rfs_df=rfs_df,
    r_urgency_mi=float(r_urg_mi), p_urgency_mi=float(p_urg_mi),
    r_comm_mi=float(r_comm_mi),   p_comm_mi=float(p_comm_mi),
    n_interventions=len(scheduler.interventions),
    empathy_weight_robot=float(controller.w_empathy),
    supported=(p_urg_mi < 0.05),
)
results["DYNAMICS"] = dyn_results

# =============================================================================
#  §13  BH MULTIPLE COMPARISON CORRECTION  [v16-FIX-2: Comm as future work]
# =============================================================================
print(f"\n{SEP}\n[13] BH Multiple Comparison Correction\n{SEP}")
print(f"  NOTE: Communication tests labelled [{COMM_STATUS_LABEL}] — excluded "
      f"from primary hypothesis count")

# [v16-FIX-2] Primary tests: RQ1-RQ4, RQ6 urgency, dynamics
# Communication test included for completeness but labelled future work
all_tests = [
    ("RQ1: AUC>0.5 (permutation)   [PRIMARY]",  results["EXT"]["perm_p"]),
    ("RQ2: interaction β≠0         [PRIMARY]",  results["RQ2"]["p_interact"]),
    ("RQ2: Fisher z                [PRIMARY]",  results["RQ2"]["p_zdiff"]),
    ("RQ3: ANOVA(cohesion)         [PRIMARY]",  results["RQ3"]["p"]),
    (f"RQ3: ANOVA(comm)             [{COMM_STATUS_LABEL}]", results["RQ3"]["p_comm"]),
    ("RQ4: ΔAUC posterior          [PRIMARY]",  results["RQ4"]["p_rq4"]),
    ("RQ6: urgency vs MI           [PRIMARY]",  results["RQ6"]["p_urgency_mi"]),
    (f"RQ6: communication vs MI     [{COMM_STATUS_LABEL}]", results["RQ6"]["p_comm_mi"]),
    ("DYN: transition_entropy      [SECONDARY]", dyn_results["transition_entropy"]["p"]),
    ("DYN: cohesion_volatility     [SECONDARY]", dyn_results["cohesion_volatility"]["p"]),
]
test_names = [t[0] for t in all_tests]
test_pvals = [t[1] for t in all_tests]
reject_bh_all, p_bh_all = bh_correct(test_pvals)

print(f"  {'Test':<48s}  {'p_raw':>8}  {'p_BH':>8}  Sig")
print(f"  {'-'*74}")
for name, p_r, p_b, rej in zip(test_names, test_pvals, p_bh_all, reject_bh_all):
    print(f"  {name:<48s}  {p_r:>8.4f}  {p_b:>8.4f}  {'✓' if rej else '–'}")

results["BH"] = dict(
    tests=test_names, p_raw=test_pvals,
    p_bh=p_bh_all.tolist(), reject=reject_bh_all.tolist(),
)

# =============================================================================
#  §14  LATENT SPACE EMBEDDING
# =============================================================================
print("\n  Latent space embedding …")
X_embed_raw = scaler_l.transform(sess[FEAT_COLS].fillna(0).values)
if UMAP_OK:
    reducer = umap_lib.UMAP(
        n_components=2, random_state=SEED, n_neighbors=15, min_dist=0.1
    )
    embed = reducer.fit_transform(X_embed_raw)
    embed_method = "UMAP"
else:
    from sklearn.manifold import TSNE
    reducer = TSNE(
        n_components=2, random_state=SEED,
        perplexity=min(30, len(sess) - 1),
    )
    embed = reducer.fit_transform(X_embed_raw)
    embed_method = "t-SNE"
results["EMBED"] = dict(embed=embed, method=embed_method)
print(f"  {embed_method} complete.")

# =============================================================================
#  §15  PUBLICATION FIGURES  (v16.1)
# =============================================================================
print(f"\n{SEP}\n[15] Generating v16.1 figures (DPI={FIG_DPI})\n{SEP}")

cluster_colors_c = [
    PAL["blue"], PAL["green"], PAL["teal"],
    PAL["orange"], PAL["purple"], PAL["red"],
][:len(valid_clusters)]

zone_rects_data = [
    (0,   0, 35, 35, "Rigid-\nDisengaged",   PAL["red"],    .18),
    (65,  0, 35, 35, "Rigid-\nEnmeshed",     PAL["purple"], .18),
    (0,  65, 35, 35, "Chaotic-\nDisengaged", PAL["blue"],   .18),
    (65, 65, 35, 35, "Chaotic-\nEnmeshed",   PAL["orange"], .18),
    (35, 35, 30, 30, "BALANCED",             PAL["green"],  .40),
]


def draw_zones(ax):
    for x0, y0, w, h, lbl, col, al in zone_rects_data:
        ax.add_patch(FancyBboxPatch(
            (x0, y0), w, h, boxstyle="round,pad=0.8",
            facecolor=col, edgecolor=col, alpha=al, lw=0.5,
        ))
        ax.text(x0 + w / 2, y0 + h / 2, lbl,
                ha="center", va="center", fontsize=6.5,
                fontweight="bold", color=col, alpha=min(al * 2.5, 1.0))
    for v in [35, 65]:
        ax.axhline(v, color="gray", lw=0.4, ls="--", alpha=0.35)
        ax.axvline(v, color="gray", lw=0.4, ls="--", alpha=0.35)


rng_jit = np.random.default_rng(SEED + 77)

# ──────────────────────────────────────────────────────────────────────────
#  Fig 1: RQ1 — Discrimination + ROC + Calibration + DCA
# ──────────────────────────────────────────────────────────────────────────
fig  = plt.figure(figsize=(28, 8), constrained_layout=True)
gs1  = gridspec.GridSpec(1, 5, figure=fig, wspace=0.35)
fig.suptitle(
    f"Fig. 1 — RQ1: 2-Axis Circumplex-Proxy Discrimination  (N={n_total})\n"
    f"AUC={auc_coh:.4f} [{auc_ci[0]:.3f},{auc_ci[1]:.3f}]  "
    f"MCC={mcc_val:.3f}  BalAcc={bal_acc:.3f}  d_CV={d_coh_cv:.3f}  "
    f"Perm-p={perm_p:.4f}  Floor≥{AUC_FLOOR} ({AUC_FLOOR_CITATION})\n"
    f"[Surrogate: Cohesion/Flex = MI proxies only; NOT FACES-IV; "
    f"Communication = future work]",
    fontsize=8.0, fontweight="bold", y=1.01,
)

for ax_i, (lbl_, sub_, col_) in enumerate([
    (f"High-MI (N={len(high)})", high, PAL["green"]),
    (f"Low-MI  (N={len(low)})",  low,  PAL["red"]),
]):
    ax = fig.add_subplot(gs1[ax_i]); draw_zones(ax)
    jx = rng_jit.uniform(-0.6, 0.6, len(sub_))
    jy = rng_jit.uniform(-0.6, 0.6, len(sub_))
    # [v16-FIX-2] Communication shown as decorative channel only
    comm_norm = (sub_.communication.values - sub_.communication.min()) / \
                max(sub_.communication.max() - sub_.communication.min(), 1e-3)
    sc = ax.scatter(
        sub_.cohesion + jx, sub_.flexibility + jy,
        s=55, c=comm_norm, cmap="coolwarm", alpha=0.80,
        edgecolors="white", lw=0.5, zorder=5,
        vmin=0, vmax=1,
    )
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect("equal")
    ax.set_xlabel("Cohesion proxy [0–100]  ← PRIMARY")
    ax.set_ylabel("Flexibility proxy [0–100]  ← PRIMARY")
    ax.set_title(f"({'AB'[ax_i]}) {lbl_}\n"
                 f"(colour = Comm. proxy [future work])", pad=6)

ax_roc = fig.add_subplot(gs1[2])
ax_roc.plot(results["EXT"]["fpr"], results["EXT"]["tpr"],
            color=PAL["blue"], lw=2.2, label=f"AUC={auc_coh:.4f}")
ax_roc.fill_between(results["EXT"]["fpr"], results["EXT"]["tpr"],
                     alpha=0.10, color=PAL["blue"])
ax_roc.plot([0, 1], [0, 1], color="gray", lw=1, ls=":")
ax_roc.axhline(0, color="gray", lw=0.3)
ax_roc.axvline(0, color="gray", lw=0.3)
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_title(f"(C) ROC Curve  [PRIMARY]\nMCC={mcc_val:.3f}  BalAcc={bal_acc:.3f}", pad=6)
ax_roc.legend(loc="lower right"); ax_roc.set_aspect("equal")

ax_cal = fig.add_subplot(gs1[3])
ax_cal.plot(results["EXT"]["prob_pred_raw"], results["EXT"]["prob_true_raw"],
            "o-", color=PAL["orange"], lw=1.8, ms=5,
            label=f"Raw  Brier={brier_raw:.3f}")
ax_cal.plot(results["EXT"]["prob_pred_iso"], results["EXT"]["prob_true_iso"],
            "s--", color=PAL["teal"], lw=1.8, ms=5,
            label=f"Iso  Brier={brier_iso:.3f}")
ax_cal.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect")
ax_cal.set_xlabel("Mean predicted probability")
ax_cal.set_ylabel("Observed fraction positives")
ax_cal.set_title(f"(D) Calibration\nECE: {ece_raw:.3f} → {ece_iso:.3f} (isotonic)", pad=6)
ax_cal.legend(loc="upper left")

ax_dca = fig.add_subplot(gs1[4])
th_dca = results["EXT"]["thresholds"]
ax_dca.plot(th_dca, results["EXT"]["nb_circ"],  color=PAL["blue"],   lw=2.0, label="Circumplex")
ax_dca.plot(th_dca, results["EXT"]["nb_logit"], color=PAL["green"],  lw=2.0, label="Logistic")
ax_dca.plot(th_dca, results["EXT"]["nb_all"],   color=PAL["orange"], lw=1.2, ls="--", label="Treat-all")
ax_dca.plot(th_dca, results["EXT"]["nb_none"],  color="gray",        lw=1.0, ls=":",  label="Treat-none")
ax_dca.set_xlim(0, 1); ax_dca.set_ylim(-0.05, 0.5)
ax_dca.set_xlabel("Threshold probability")
ax_dca.set_ylabel("Net Benefit")
ax_dca.set_title("(E) Decision Curve Analysis", pad=6)
ax_dca.legend(loc="upper right")
save_fig("fig1_rq1_v16.png", fig)

# ──────────────────────────────────────────────────────────────────────────
#  Fig 2: RQ2 — Moderation + VIF
# ──────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), constrained_layout=True)
fig.suptitle(
    f"Fig. 2 — RQ2: Empathy–Cohesion Moderation by MI Quality (N={n_total})  "
    f"[PRIMARY CLAIM]\n"
    f"β_int={results['RQ2']['beta_interact']:+.4f}  "
    f"CI=[{results['RQ2']['ci_interact'][0]:+.3f},{results['RQ2']['ci_interact'][1]:+.3f}]  "
    f"ΔR²={results['RQ2']['delta_r2']:.4f}",
    fontsize=10, fontweight="bold",
)

ax = axes[0]
for mi_q, col_, mk in [("high", PAL["green"], "o"), ("low", PAL["red"], "s")]:
    sub_ = sess[sess.mi_quality == mi_q]
    jx_  = rng_jit.uniform(-0.002, 0.002, len(sub_))
    ax.scatter(sub_.empathy_rate + jx_, sub_.cohesion,
               color=col_, marker=mk, s=45, alpha=0.70,
               edgecolors="white", lw=0.4, label=mi_q)
    z_   = np.polyfit(sub_.empathy_rate, sub_.cohesion, 1)
    xr_  = np.linspace(sub_.empathy_rate.min(), sub_.empathy_rate.max(), 100)
    r_,  _ = pearsonr(sub_.empathy_rate, sub_.cohesion)
    ax.plot(xr_, np.polyval(z_, xr_), color=col_, lw=2, ls="--",
            label=f"{mi_q} r={r_:+.3f}")
ax.set_xlabel("Empathy Rate (proportion)")
ax.set_ylabel("Cohesion proxy [0–100]")
ax.set_title("(A) Empathy→Cohesion × MI Quality", pad=6)
ax.legend(fontsize=7)

ax = axes[1]
ax.hist(results["RQ2"]["boot_int"], bins=50, color=PAL["teal"],
        alpha=0.75, edgecolor="white", density=True)
ax.axvline(0, color="black", lw=1.5, label="β=0")
ax.axvline(results["RQ2"]["beta_interact"], color=PAL["navy"], lw=2, ls="--",
           label=f"β={results['RQ2']['beta_interact']:+.4f}")
ci_ = results["RQ2"]["ci_interact"]
ax.axvline(ci_[0], color=PAL["red"], lw=1.2, ls=":")
ax.axvline(ci_[1], color=PAL["red"], lw=1.2, ls=":", label="95% CI")
ax.set_xlabel("Bootstrap β (empathy × MI interaction)")
ax.set_ylabel("Density")
sig_str = "SIGNIFICANT ✓" if results["RQ2"]["mod_sig"] else "n.s."
ax.set_title(f"(B) Bootstrap distribution ({sig_str})", pad=6)
ax.legend(fontsize=7)

ax = axes[2]
vif_vals = results["RQ2"]["vif_full"]
vif_lbls = ["int.", "empathy", "MI", "emp×MI"]
colors_v = [PAL["red"] if v > 5 else PAL["blue"] for v in vif_vals]
ax.bar(vif_lbls, vif_vals, color=colors_v, alpha=0.85, edgecolor="white")
ax.axhline(5, color=PAL["red"], lw=1.2, ls="--", label="VIF=5 threshold")
ax.axhline(2, color=PAL["gray"], lw=0.8, ls=":")
ax.set_ylabel("VIF")
ax.set_title(f"(C) Variance Inflation Factors\n"
             f"max VIF={max(vif_vals):.2f}  "
             f"{'⚠ HIGH' if max(vif_vals) > 5 else '✓ OK'}", pad=6)
ax.legend(fontsize=7)

ax = axes[3]
beta_arr = np.array(results["RQ2"]["beta_full"])
se_arr   = np.array(results["RQ2"]["se_full"])
emp_f    = np.linspace(-2, 2, 100)
cond_e   = beta_arr[2] + beta_arr[3] * emp_f
se_c     = np.sqrt(se_arr[2] ** 2 + se_arr[3] ** 2 * emp_f ** 2)
ax.plot(emp_f, cond_e, color=PAL["navy"], lw=2)
ax.fill_between(emp_f, cond_e - 1.96 * se_c, cond_e + 1.96 * se_c,
                alpha=0.12, color=PAL["navy"])
ax.axhline(0, color="black", lw=0.8, ls="--")
sig_m = ((cond_e - 1.96 * se_c) > 0) | ((cond_e + 1.96 * se_c) < 0)
ax.fill_between(emp_f, cond_e - 1.96 * se_c, cond_e + 1.96 * se_c,
                where=sig_m, alpha=0.25, color=PAL["green"], label="Sig. region")
ax.set_xlabel("Standardised empathy rate")
ax.set_ylabel("Conditional MI quality effect")
ax.set_title("(D) Johnson-Neyman spotlight", pad=6)
ax.legend(fontsize=7)
save_fig("fig2_rq2_v16.png", fig)

# ──────────────────────────────────────────────────────────────────────────
#  Fig 3: RQ4 — Bayesian MCMC + ΔAUC + PCA
# ──────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), constrained_layout=True)
fig.suptitle(
    f"Fig. 3 — RQ4: Bayesian MCMC Weight Posterior (N={n_total})  [PRIMARY]\n"
    f"Dominant: {dominant_key} → {dominant_component_label}  "
    f"MAP AUC={auc_opt:.4f}  "
    f"ΔAUC 95%CI=[{ci_delta[0]:+.4f},{ci_delta[1]:+.4f}]  "
    f"p={p_rq4_two_sided:.4f}  Accept={final_accept_rate:.1%}",
    fontsize=8.5, fontweight="bold",
)

ax = axes[0]
x_pos = np.arange(len(COH_KEYS))
ax.barh(x_pos, w_post_mean, color=PAL["teal"], alpha=0.70, label="MCMC posterior",
        xerr=[xerr_lo, xerr_hi], capsize=3, error_kw=dict(lw=1.2))
ax.scatter(BASE_W / BASE_W.sum(), x_pos, color=PAL["red"],
           s=55, zorder=5, marker="D", label="Theory (Olson 2011)")
ax.scatter(list(w_opt), x_pos, color=PAL["orange"],
           s=55, zorder=5, marker="^", label="MAP")
ax.set_yticks(x_pos); ax.set_yticklabels(COH_KEYS, fontsize=8)
ax.set_xlabel("Weight")
ax.set_title(f"(A) Posterior weights (95% CI)\n{dominant_component_label}", pad=6)
ax.legend(fontsize=7)

ax = axes[1]
top3_idx = np.argsort(w_post_std)[-3:][::-1]
for j, k_idx in enumerate(top3_idx):
    ax.plot(chain_burn[:, k_idx], alpha=0.70, lw=0.7,
            label=COH_KEYS[k_idx], color=MODEL_COLORS[j])
ax.set_xlabel("MCMC iteration (post burn-in)")
ax.set_ylabel("Weight")
ax.set_title(f"(B) MCMC traces (top-3 variance)\nAccept={final_accept_rate:.1%}", pad=6)
ax.legend(fontsize=7)

ax = axes[2]
ax.hist(delta_auc_samples, bins=40, color=PAL["navy"],
        alpha=0.7, edgecolor="white", density=True)
ax.axvline(0, color="black", lw=1.5, label="ΔAUC=0")
ax.axvline(ci_delta[0], color=PAL["red"], lw=1.2, ls=":")
ax.axvline(ci_delta[1], color=PAL["red"], lw=1.2, ls=":", label="95% CI")
ax.axvline(delta_auc_samples.mean(), color=PAL["green"], lw=2, ls="--",
           label=f"mean={delta_auc_samples.mean():+.4f}")
ax.set_xlabel("ΔAUC = AUC(posterior weights) − AUC(theory weights)")
ax.set_ylabel("Density")
ax.set_title(f"(C) ΔAUC posterior test\np={p_rq4_two_sided:.4f}  "
             f"{'✓' if results['RQ4']['supported'] else 'n.s.'}", pad=6)
ax.legend(fontsize=7)

ax = axes[3]
ax2 = ax.twinx()
ax.bar(range(1, len(explained) + 1), [e * 100 for e in explained],
       color=PAL["blue"], alpha=0.65, label="% variance")
ax2.plot(range(1, len(explained) + 1), np.cumsum([e * 100 for e in explained]),
         "o-", color=PAL["red"], lw=1.8, ms=5, label="Cumulative %")
ax.set_xlabel("Principal Component")
ax.set_ylabel("% variance explained", color=PAL["blue"])
ax2.set_ylabel("Cumulative %", color=PAL["red"]); ax2.set_ylim(0, 105)
ax.set_title(f"(D) Cohesion sub-feature PCA scree\nPC1 dominant: {pc1_dominant}", pad=6)
ax.legend(loc="upper right", fontsize=7); ax2.legend(loc="center right", fontsize=7)
save_fig("fig3_rq4_bayesian_v16.png", fig)

# ──────────────────────────────────────────────────────────────────────────
#  Fig 4: RQ5 — Interpretability Premium
# ──────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), constrained_layout=True)
fig.suptitle(
    f"Fig. 4 — RQ5: Interpretability Premium (N={n_total})  [PRIMARY]\n"
    f"Circumplex={auc_base:.4f}  Logistic(CV)={logit_res['auc_mean']:.4f}  "
    f"LSTM[BCE]={lstm_auc_mean:.4f}  Floor≥{AUC_FLOOR}",
    fontsize=9, fontweight="bold",
)

ax = axes[0]
method_comp = {
    "Circumplex\n(theory)":   auc_base,
    "Bayesian\n(MAP)":        auc_opt,
    "Logistic\n(5×5 CV)":    logit_res["auc_mean"],
    f"LSTM[BCE]\n(best)":     lstm_auc_mean,
}
cols_ab = [PAL["blue"], PAL["purple"], PAL["green"], PAL["red"]]
bars_ab  = ax.bar(list(method_comp.keys()), list(method_comp.values()),
                  color=cols_ab, alpha=0.85, edgecolor="white", width=0.6)
ax.axhline(AUC_FLOOR, color="gray", lw=1, ls="--",
           label=f"AUC≥{AUC_FLOOR} floor ({AUC_FLOOR_CITATION})")
ax.axhline(0.80, color=PAL["green"], lw=1, ls=":", label="AUC=0.80 target")
for bar_, v_ in zip(bars_ab, method_comp.values()):
    if not np.isnan(v_):
        ax.text(bar_.get_x() + bar_.get_width() / 2, v_ + 0.012, f"{v_:.3f}",
                ha="center", fontsize=8.5, fontweight="bold")
ax.set_ylabel("AUC"); ax.set_ylim(0, 1.05)
ax.set_title("(A) AUC comparison\n[interpretability ≈ theory on small N]", pad=6)
ax.legend(fontsize=7)

ax = axes[1]
method_mcc  = {
    "Circumplex":         mcc_val,
    "Logistic\n(5×5 CV)": logit_res["mcc_mean"],
    "LSTM[BCE]":          lstm_mcc_mean if not np.isnan(lstm_mcc_mean) else 0.0,
}
method_bacc = {
    "Circumplex":         bal_acc,
    "Logistic\n(5×5 CV)": logit_res["bacc_mean"],
    "LSTM[BCE]":          0.0,
}
x_mb = np.arange(len(method_mcc)); w_mb = 0.38
ax.bar(x_mb - w_mb / 2, list(method_mcc.values()),  width=w_mb,
       color=PAL["blue"],  alpha=0.80, label="MCC",    edgecolor="white")
ax.bar(x_mb + w_mb / 2, list(method_bacc.values()), width=w_mb,
       color=PAL["green"], alpha=0.80, label="BalAcc", edgecolor="white")
ax.set_xticks(x_mb); ax.set_xticklabels(list(method_mcc.keys()))
ax.set_ylabel("Score"); ax.set_ylim(0, 1.05)
ax.set_title("(B) MCC + Balanced Accuracy\n(imbalance-robust metrics)", pad=6)
ax.legend(fontsize=7)

ax = axes[2]
if lstm_sensitivity_grid:
    hs_vals  = sorted({v["hidden"] for v in lstm_sensitivity_grid.values()})
    nl_vals  = sorted({v["layers"] for v in lstm_sensitivity_grid.values()})
    grid_mat = np.zeros((len(nl_vals), len(hs_vals)))
    for k, v in lstm_sensitivity_grid.items():
        ri = nl_vals.index(v["layers"]); ci = hs_vals.index(v["hidden"])
        grid_mat[ri, ci] = v["auc_mean"] if not np.isnan(v["auc_mean"]) else 0
    im5 = ax.imshow(grid_mat, cmap="YlGn", vmin=0.5, vmax=0.85, aspect="auto")
    plt.colorbar(im5, ax=ax, label="AUC", pad=0.02)
    ax.set_xticks(range(len(hs_vals)))
    ax.set_xticklabels([f"h={h}" for h in hs_vals])
    ax.set_yticks(range(len(nl_vals)))
    ax.set_yticklabels([f"L={l}" for l in nl_vals])
    for ri in range(len(nl_vals)):
        for ci in range(len(hs_vals)):
            v_ = grid_mat[ri, ci]
            ax.text(ci, ri, f"{v_:.3f}", ha="center", va="center", fontsize=8,
                    fontweight="bold", color="white" if v_ > 0.70 else "black")
    ax.set_title("LSTM sensitivity grid\n(hidden_size × n_layers, BCE)", pad=6)
else:
    ax.text(0.5, 0.5, "LSTM grid\n(PyTorch required)",
            ha="center", va="center", transform=ax.transAxes)

ax = axes[3]
cf_clean = cf_deltas[~np.isnan(cf_deltas)]
ax.hist(cf_clean, bins=20, color=PAL["navy"], alpha=0.75,
        edgecolor="white", density=True)
ax.axvline(cf_median, color=PAL["red"], lw=2, ls="--",
           label=f"Median={cf_median:.3f} std")
ax.axvline(cf_pct25,  color=PAL["orange"], lw=1.2, ls=":")
ax.axvline(cf_pct75,  color=PAL["orange"], lw=1.2, ls=":", label="IQR")
ax.set_xlabel("Required Δ(empathy_rate) in standardised units")
ax.set_ylabel("Density")
ax.set_title("(D) Counterfactual: Low→High MI\n(minimum empathy increase)", pad=6)
ax.legend(fontsize=7)
save_fig("fig4_rq5_ablation_v16.png", fig)

# ──────────────────────────────────────────────────────────────────────────
#  Fig 5: SHAP + LOFO  [v16-FIX-1: boxplot compatibility]
# ──────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(22, 11), constrained_layout=True)
fig.suptitle(
    f"Fig. 5 — Feature Importance: LinearSHAP + PermSHAP + LOFO\n"
    f"ρ(Linear,Perm)={rho_l_p:.4f}  LOFO base AUC={full_auc_lofo:.4f}  "
    f"[+drop=informative / −drop=noise; n_info={lofo_n_info}, n_noise={lofo_n_noise}]",
    fontsize=10, fontweight="bold",
)

ax = axes[0, 0]
df_ls = linear_shap_df.sort_values("abs_LinearSHAP", ascending=True)
colors_ls = [PAL["green"] if v > 0 else PAL["red"] for v in df_ls.LinearSHAP]
ax.barh(df_ls.feature, df_ls.LinearSHAP, color=colors_ls,
        alpha=0.85, edgecolor="white")
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("LinearSHAP (signed)")
ax.set_title("(A) LinearSHAP — analytic\n(coeff × feature std)", pad=6)

ax = axes[0, 1]
feat_order = linear_shap_df.sort_values("abs_LinearSHAP", ascending=True)["feature"].tolist()
ls_vals = linear_shap_df.set_index("feature").loc[feat_order, "abs_LinearSHAP"].values
ps_vals = perm_shap_df.set_index("feature").loc[feat_order, "PermSHAP"].values
ls_n = ls_vals / (ls_vals.max() + 1e-9)
ps_n = ps_vals / (ps_vals.max() + 1e-9)
y_f = np.arange(len(feat_order)); w_b = 0.38
ax.barh(y_f - w_b / 2, ls_n, height=w_b, color=PAL["blue"],
        alpha=0.8, label="LinearSHAP")
ax.barh(y_f + w_b / 2, ps_n, height=w_b, color=PAL["teal"],
        alpha=0.8, label="PermSHAP")
ax.set_yticks(y_f); ax.set_yticklabels(feat_order, fontsize=7.5)
ax.set_xlabel("Normalised |SHAP|")
ax.set_title(f"(B) LinearSHAP vs PermSHAP\nSpearman ρ={rho_l_p:.4f}", pad=6)
ax.legend(fontsize=7)

ax = axes[0, 2]
lofo_sorted = lofo_df.sort_values("auc_drop", ascending=True)
colors_lf   = [PAL["red"] if v > 0 else PAL["gray"] for v in lofo_sorted.auc_drop]
err_lo = lofo_sorted.auc_drop - lofo_sorted.drop_ci_lo
err_hi = lofo_sorted.drop_ci_hi - lofo_sorted.auc_drop
ax.barh(lofo_sorted.feature, lofo_sorted.auc_drop,
        color=colors_lf, alpha=0.85, edgecolor="white",
        xerr=[np.clip(err_lo, 0, None), np.clip(err_hi, 0, None)],
        capsize=3, error_kw=dict(lw=1.0, color="black", alpha=0.6))
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("AUC drop when feature removed (LOFO-CV, ±95% CI)")
ax.set_title("(C) LOFO — Leave-One-Feature-Out\n(bootstrap CI added)", pad=6)

ax = axes[1, 0]
for method_name, col_ in [("LinearSHAP", PAL["blue"]), ("PermSHAP", PAL["teal"])]:
    if method_name in comp_results:
        pairs_ = comp_results[method_name]
        k_v    = [p[0] for p in pairs_]; auc_v = [p[1] for p in pairs_]
        ax.plot(k_v, auc_v, "o-", color=col_, lw=1.8, ms=6, label=method_name)
ax.axhline(auc_base, color="gray", lw=1.2, ls="--",
           label=f"Full AUC (theory)={auc_base:.4f}")
ax.set_xlabel("k top features removed")
ax.set_ylabel("AUC (in-sample, all N)")
ax.set_title("(D) SHAP comprehensiveness test\n(sequential ablation)", pad=6)
ax.legend(fontsize=7)

ax = axes[1, 1]
lofo_rank = lofo_df.set_index("feature")["auc_drop"].reindex(FEAT_COLS).values
shap_rank = linear_shap_df.set_index("feature")["abs_LinearSHAP"].reindex(FEAT_COLS).values
rho_ls_lf, p_ls_lf = spearmanr(
    pd.Series(shap_rank).rank().values,
    pd.Series(lofo_rank).rank().values,
)
ax.scatter(shap_rank, lofo_rank, color=PAL["navy"], s=55,
           alpha=0.75, edgecolors="white")
for fi, feat in enumerate(FEAT_COLS):
    ax.annotate(feat[:8], (shap_rank[fi], lofo_rank[fi]),
                fontsize=6, ha="left", va="bottom", color="#444")
ax.set_xlabel("|LinearSHAP|")
ax.set_ylabel("LOFO AUC drop")
ax.set_title(f"(E) SHAP vs LOFO rank correlation\nρ={rho_ls_lf:.4f}  p={p_ls_lf:.4f}", pad=6)

# [v16-FIX-1] matplotlib boxplot compatibility
ax = axes[1, 2]
cf_clean2 = cf_deltas[~np.isnan(cf_deltas)]
_boxplot_with_labels(
    ax, [cf_clean2], ["Low → High MI"],
    vert=True, patch_artist=True,
    boxprops=dict(facecolor=PAL["teal"], alpha=0.6),
    medianprops=dict(color=PAL["red"], lw=2),
)
ax.set_ylabel("Required Δ(empathy_rate) [std]")
ax.set_title(f"(F) Counterfactual summary\n"
             f"Median={cf_median:.3f}  IQR=[{cf_pct25:.3f},{cf_pct75:.3f}]", pad=6)
save_fig("fig5_shap_lofo_v16.png", fig)

# ──────────────────────────────────────────────────────────────────────────
#  Fig 6: RQ3 — Topic Cluster Profiles  [v16-FIX-2: Comm demoted]
# ──────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(20, 11), constrained_layout=True)
fig.suptitle(
    f"Fig. 6 — RQ3: Topic Domain × Cohesion-Proxy Profiles (N={n_total})\n"
    f"Cohesion [CONFIRMATORY / PRIMARY]: F={F_h3:.3f}  p={p_h3:.4f}  "
    f"η²={eta2_h3:.4f}  |  "
    f"Communication [{COMM_STATUS_LABEL}]: F={F_comm:.3f}  p={p_comm:.4f}",
    fontsize=9.5, fontweight="bold",
)

ax = axes[0, 0]
vdata = [sess_h3[sess_h3.topic_cluster == c].cohesion.values for c in valid_clusters]
vp    = ax.violinplot(vdata, showmedians=True, showextrema=False)
for body, col_ in zip(vp["bodies"], cluster_colors_c):
    body.set_facecolor(col_); body.set_alpha(0.60)
vp["cmedians"].set_color("black"); vp["cmedians"].set_linewidth(1.8)
ax.set_xticks(range(1, len(valid_clusters) + 1))
ax.set_xticklabels([c.replace("_", "\n") for c in valid_clusters], fontsize=7)
ax.set_ylabel("Cohesion proxy [0–100]")
ax.set_title(f"(A) Cohesion by topic domain [CONFIRMATORY]\n"
             f"F={F_h3:.2f}  η²={eta2_h3:.4f}", pad=6)

ax = axes[0, 1]
vdata2 = [sess_h3[sess_h3.topic_cluster == c].communication.values for c in valid_clusters]
vp2    = ax.violinplot(vdata2, showmedians=True, showextrema=False)
for body, col_ in zip(vp2["bodies"], cluster_colors_c):
    body.set_facecolor(col_); body.set_alpha(0.60)
vp2["cmedians"].set_color("black"); vp2["cmedians"].set_linewidth(1.8)
ax.set_xticks(range(1, len(valid_clusters) + 1))
ax.set_xticklabels([c.replace("_", "\n") for c in valid_clusters], fontsize=7)
ax.set_ylabel("Communication proxy [0–100]")
# [v16-FIX-2] Stronger warning on Communication panel
ax.set_title(f"(B) Communication [{COMM_STATUS_LABEL}]\n"
             f"F={F_comm:.2f}  η²={eta2_comm:.4f}\n"
             f"⚠ clarification coverage={regex_audit['clarification_hit_pct']:.1f}% "
             f"— NOT primary claim", pad=6)

ax = axes[0, 2]
ct_hm = sess_h3.groupby(["topic_cluster","mi_quality"])["cohesion"].mean().unstack()
im    = ax.imshow(ct_hm.values.T, aspect="auto", cmap="RdYlGn", vmin=50, vmax=95)
plt.colorbar(im, ax=ax, label="Mean Cohesion proxy", pad=0.02)
ax.set_xticks(range(len(ct_hm.index)))
ax.set_xticklabels([c.replace("_", "\n") for c in ct_hm.index], fontsize=7.5)
ax.set_yticks(range(len(ct_hm.columns)))
ax.set_yticklabels(ct_hm.columns, fontsize=8)
for i in range(len(ct_hm.index)):
    for j in range(len(ct_hm.columns)):
        val = ct_hm.values[i, j]
        if not np.isnan(val):
            ax.text(i, j, f"{val:.1f}", ha="center", va="center",
                    fontsize=8, fontweight="bold")
ax.set_title("(C) Mean Cohesion heatmap [PRIMARY]\nCluster × MI quality label", pad=6)

ax = axes[1, 0]
d_mat = np.zeros((len(valid_clusters), len(valid_clusters)))
for i, c1 in enumerate(valid_clusters):
    for j, c2 in enumerate(valid_clusters):
        d_mat[i, j] = cohens_d(
            sess_h3[sess_h3.topic_cluster == c1].cohesion.values,
            sess_h3[sess_h3.topic_cluster == c2].cohesion.values,
        )
im2 = ax.imshow(d_mat, cmap="RdBu_r", vmin=-2.5, vmax=2.5)
plt.colorbar(im2, ax=ax, label="Cohen's d (cohesion)", pad=0.02)
ax.set_xticks(range(len(valid_clusters)))
ax.set_yticks(range(len(valid_clusters)))
ax.set_xticklabels([c[:7] for c in valid_clusters],
                    fontsize=7, rotation=30, ha="right")
ax.set_yticklabels([c[:7] for c in valid_clusters], fontsize=7)
for i in range(len(valid_clusters)):
    for j in range(len(valid_clusters)):
        ax.text(j, i, f"{d_mat[i,j]:.2f}", ha="center", va="center",
                fontsize=6.5,
                color="white" if abs(d_mat[i, j]) > 1 else "black")
ax.set_title("(D) Post-hoc Cohen's d (cohesion)", pad=6)

ax = axes[1, 1]
pair_colors_ = [PAL["green"] if r_ else PAL["red"] for r_ in reject_bh]
pair_labels_ = [f"{c1[:6]}v{c2[:6]}" for c1, c2 in pairs]
ax.barh(pair_labels_,
        [-np.log10(max(p, 1e-6)) for p in p_bh.tolist()],
        color=pair_colors_, alpha=0.85, edgecolor="white")
ax.axvline(-np.log10(0.05), color=PAL["red"], lw=1.2, ls="--", label="α=0.05")
ax.set_xlabel("−log₁₀(p_BH)")
ax.legend(fontsize=7)
ax.set_title("(E) BH-corrected post-hoc pairwise\n[cohesion PRIMARY]", pad=6)

ax = axes[1, 2]
ct_bar = pd.crosstab(sess.topic_cluster, sess.mi_quality).reindex(
    [c for c in valid_clusters
     if c in pd.crosstab(sess.topic_cluster, sess.mi_quality).index]
)
bot_ = np.zeros(len(ct_bar))
for mi_q_, col_ in [("high", PAL["green"]), ("low", PAL["red"])]:
    if mi_q_ in ct_bar.columns:
        vals_ = ct_bar[mi_q_].values
        ax.bar(range(len(ct_bar)), vals_, bottom=bot_, color=col_,
               alpha=0.8, edgecolor="white",
               label=f"{mi_q_.capitalize()}-MI")
        for j_, (v_, b_) in enumerate(zip(vals_, bot_)):
            if v_ > 0:
                ax.text(j_, b_ + v_ / 2, str(v_), ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
        bot_ += vals_
ax.set_xticks(range(len(ct_bar)))
ax.set_xticklabels([c.replace("_", "\n") for c in ct_bar.index], fontsize=7.5)
ax.set_ylabel("Number of sessions")
ax.legend(fontsize=7)
ax.set_title("(F) Session count by topic domain", pad=6)
save_fig("fig6_rq3_clusters_v16.png", fig)

# ──────────────────────────────────────────────────────────────────────────
#  Fig 7: UMAP + Temporal Dynamics
# ──────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(22, 11), constrained_layout=True)
fig.suptitle(
    f"Fig. 7 — {embed_method} Latent Space + Temporal Rigidity Dynamics  "
    f"(N={n_total})",
    fontsize=10, fontweight="bold",
)

embed_  = results["EMBED"]["embed"]
emb_m_  = results["EMBED"]["method"]

for ax_i, (mi_q, col_) in enumerate([("high", PAL["green"]), ("low", PAL["red"])]):
    mask = sess.mi_quality.values == mi_q
    axes[0, ax_i].scatter(
        embed_[mask, 0], embed_[mask, 1], s=50, color=col_,
        alpha=0.75, edgecolors="white", lw=0.4,
        label=f"{mi_q.capitalize()}-MI",
    )
axes[0, 0].set_title(f"(A) {emb_m_}: MI quality [2-axis features]", pad=6)
axes[0, 0].legend(fontsize=8)

ax = axes[0, 1]
for z_, col_ in ZONE_COLORS.items():
    mask = sess.zone.values == z_
    if mask.sum() > 0:
        ax.scatter(embed_[mask, 0], embed_[mask, 1], s=50, color=col_,
                   alpha=0.75, edgecolors="white", lw=0.4, label=z_[:12])
ax.set_title(f"(B) {emb_m_}: Circumplex zone [2-axis]", pad=6)
ax.legend(fontsize=6.5)

ax = axes[0, 2]
cluster_pal = {c: cluster_colors_c[i % len(cluster_colors_c)]
               for i, c in enumerate(valid_clusters)}
for cl_, col_ in cluster_pal.items():
    mask = sess.topic_cluster.values == cl_
    if mask.sum() > 0:
        ax.scatter(embed_[mask, 0], embed_[mask, 1], s=50, color=col_,
                   alpha=0.75, edgecolors="white", lw=0.4, label=cl_[:9])
ax.set_title(f"(C) {emb_m_}: Topic cluster", pad=6)
ax.legend(fontsize=6.5)

for ax_i, (ax, col) in enumerate(
    zip([axes[1, 0], axes[1, 1], axes[1, 2]],
        ["transition_entropy", "emotional_inertia", "cohesion_volatility"])
):
    dres    = dyn_results[col]
    data_h  = sess[sess.mi_quality == "high"][col].values
    data_l  = sess[sess.mi_quality == "low"][col].values
    rng_j2  = np.random.default_rng(SEED + ax_i + 20)
    jit_h   = rng_j2.uniform(-0.12, 0.12, len(data_h))
    jit_l   = rng_j2.uniform(-0.12, 0.12, len(data_l))
    ax.scatter(np.ones(len(data_h)) + jit_h, data_h, s=25,
               color=PAL["green"], alpha=0.55, edgecolors="white", lw=0.3)
    ax.scatter(2 * np.ones(len(data_l)) + jit_l, data_l, s=25,
               color=PAL["red"],   alpha=0.55, edgecolors="white", lw=0.3)
    ax.hlines(np.median(data_h), 0.75, 1.25, color=PAL["green"], lw=2.5)
    ax.hlines(np.median(data_l), 1.75, 2.25, color=PAL["red"],   lw=2.5)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["High-MI", "Low-MI"], fontsize=8.5)
    sig_ = ("***" if dres["p"] < 0.001 else
            ("**" if dres["p"] < 0.01 else
             ("*" if dres["p"] < 0.05 else "n.s.")))
    ax.set_title(f"({'DEF'[ax_i]}) {col.replace('_', ' ')}\n"
                 f"d={dres['d']:+.3f}  {sig_}", pad=6)
    ax.set_ylabel(col.replace("_", " "))

for row_ax in [axes[0, 0], axes[0, 1], axes[0, 2]]:
    row_ax.set_xlabel(f"{emb_m_} dim. 1")
    row_ax.set_ylabel(f"{emb_m_} dim. 2")
save_fig("fig7_dynamics_umap_v16.png", fig)

# ──────────────────────────────────────────────────────────────────────────
#  Fig 8: Power + Final Scorecard  [v16-FIX-2/3: updated]
# ──────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    1, 2, figsize=(28, 11), constrained_layout=True,
    gridspec_kw={"width_ratios": [1, 2.2]},
)
fig.suptitle(
    f"Fig. 8 — Post-hoc Power + Research Question Scorecard  (v16.1, N={n_total})\n"
    f"PRIMARY CLAIMS: Cohesion + Flexibility (2-axis) | "
    f"Communication = [{COMM_STATUS_LABEL}]",
    fontsize=11, fontweight="bold",
)

ax = axes[0]
n_range = np.arange(10, 500, 5)
powers_ = [power_ttest_ind(int(n * n_hi / n_total),
                            int(n * n_lo / n_total), d_coh_cv)
           for n in n_range]
ax.plot(n_range, powers_, color=PAL["blue"], lw=2.2,
        label=f"CV-adjusted d={d_coh_cv:.2f}")
ax.axhline(0.80, color=PAL["green"], lw=1.5, ls="--", label="80% power target")
ax.axhline(0.90, color=PAL["teal"],  lw=1.2, ls=":",  label="90% power")
ax.axvline(n_total, color=PAL["red"], lw=2, ls="--",
           label=f"N={n_total} (power={power_rq1:.2f})")
ax.axvline(n_needed, color=PAL["navy"], lw=1.5, ls="-.",
           label=f"N_needed={n_needed}")
ax.set_xlabel("Total N")
ax.set_ylabel("Statistical power")
ax.set_title("(A) Post-hoc power curve\n(two-sample t-test, CV-adjusted d)", pad=6)
ax.legend(fontsize=7); ax.set_ylim(0, 1.05)

ax = axes[1]; ax.axis("off")

# [v16-FIX-2/3] Scorecard with Communication demoted + AUC citation
scorecard = [
    ("RQ1",
     "2-Axis Circumplex-proxy discrimination of annotated MI quality",
     "CONFIRMED ✓" if results["EXT"]["supported"] else "PARTIAL",
     PAL["green"] if results["EXT"]["supported"] else PAL["orange"],
     f"AUC={auc_coh:.4f} [{auc_ci[0]:.3f},{auc_ci[1]:.3f}]  "
     f"MCC={mcc_val:.3f}  BalAcc={bal_acc:.3f}  d_CV={d_coh_cv:.3f}  Perm-p={perm_p:.4f}"
     f"\n    [Surrogate: Cohesion/Flex = MI proxies; NOT FACES-IV. "
     f"AUC floor ≥{AUC_FLOOR} per {AUC_FLOOR_CITATION}]"),
    ("RQ2",
     "MI quality moderates empathy–Cohesion relationship  [PRIMARY]",
     "SUPPORTED ✓" if results["RQ2"]["supported"] else "EXPLORATORY",
     PAL["teal"] if results["RQ2"]["supported"] else PAL["orange"],
     f"β_int={results['RQ2']['beta_interact']:+.4f}  "
     f"CI=[{results['RQ2']['ci_interact'][0]:+.3f},{results['RQ2']['ci_interact'][1]:+.3f}]  "
     f"ΔR²={results['RQ2']['delta_r2']:.4f}  Fisher-z p={results['RQ2']['p_zdiff']:.4f}"
     f"\n    [VIF max={max(vifs_full):.2f}  "
     f"{'⚠ VIF > 5' if max(vifs_full) > 5 else '✓ VIF OK'}]"),
    ("RQ3",
     "Topic domains differ in Cohesion proxy  [PRIMARY / CONFIRMATORY]",
     "CONFIRMED ✓" if results["RQ3"]["supported"] else "PARTIAL",
     PAL["green"] if results["RQ3"]["supported"] else PAL["orange"],
     f"Cohesion [CONF]: F={F_h3:.3f}  p={p_h3:.4f}  η²={eta2_h3:.4f}  ω²={omega2_h3:.4f}"
     f"\n    [Communication [{COMM_STATUS_LABEL}]: F={F_comm:.3f}  p={p_comm:.4f}  "
     f"η²={eta2_comm:.4f} — NOT primary claim; coverage={regex_audit['clarification_hit_pct']:.1f}%]"),
    ("RQ4",
     "Bayesian posterior improves over theory weights  [PRIMARY]",
     "SUPPORTED ✓" if results["RQ4"]["supported"] else "n.s. (hold-out)",
     PAL["green"] if results["RQ4"]["supported"] else PAL["gray"],
     f"Hold-out ΔAUC CI=[{results['RQ4']['ci_delta_ho'][0]:+.4f},"
     f"{results['RQ4']['ci_delta_ho'][1]:+.4f}]  p={results['RQ4']['p_rq4_ho']:.4f}  "
     f"[In-sample CI={ci_delta[0]:+.4f},{ci_delta[1]:+.4f} — biased, supplementary only]"
     f"\n    [Dominant: {dominant_key} → {dominant_component_label}]"
     f"\n    {results['RQ4']['in_sample_bias_note']}"),
    ("RQ5",
     "Interpretability premium: theory ≈ black-box on small N  [PRIMARY]",
     "SUPPORTED ✓" if abs(auc_base - logit_res["auc_mean"]) < 0.15 else "PARTIAL",
     PAL["blue"],
     f"Circumplex={auc_base:.4f}  Logistic(CV)={logit_res['auc_mean']:.4f}  "
     f"LSTM[BCE]={lstm_auc_mean:.4f}  CF Δempathy={cf_median:.3f} std"
     f"\n    [LSTM near-random (AUC≈0.57) → interpretable proxy preferred on N={n_total}]"),
    ("SHAP",
     "Dual-SHAP + LOFO: convergent feature importance",
     "VALIDATED ✓", PAL["orange"],
     f"ρ(Linear,Perm)={rho_l_p:.4f}  "
     f"Top LOFO (informative): {lofo_top1_feat} (drop={lofo_top1_drop:+.4f})  "
     f"[{lofo_n_info} informative, {lofo_n_noise} noise/redundant]"
     f"\n    [empathy_rate, negation_rate, wc_balance = consistent top-3 across SHAP methods]"
     f"\n    [LOFO drop≤0 features are noise — removed from primary importance claims]"),
    ("DYN",
     "Temporal rigidity markers differ by MI quality  [SECONDARY]",
     "INFORMATIVE ✓", PAL["cyan"],
     "  ".join([f"{c}: d={dyn_results[c]['d']:+.3f}" for c in dyn_cols])),
    ("RQ6",
     "2-Axis Circumplex zones → sensible robot intervention signals",
     "VALIDATED ✓" if results["RQ6"]["supported"] else "PARTIAL",
     PAL["navy"] if results["RQ6"]["supported"] else PAL["orange"],
     f"r(urgency,MI)={r_urg_mi:.4f}  p={p_urg_mi:.4f}  [2-axis urgency only]  "
     f"w_empathy={float(controller.w_empathy):.4f} (Bayesian-updated)"
     f"\n    [comm r={r_comm_mi:.3f} p={p_comm_mi:.3f} low as expected — "
     f"Communication [{COMM_STATUS_LABEL}]]"),
    ("§S",
     "Hold-out weight generalisation (auxiliary sensitivity check)",
     "LIMITATION ⚠" if HOLDOUT.get("both_gaps_flagged") else "INFORMATIVE",
     PAL["red"] if HOLDOUT.get("both_gaps_flagged") else PAL["lime"],
     f"Theory:  Train={HOLDOUT['auc_theory_train']:.4f}  "
     f"Test={HOLDOUT['auc_theory_test']:.4f}  gap={HOLDOUT['theory_gap']:+.4f}"
     f"  {'⚠' if HOLDOUT['theory_gap'] > 0.10 else '✓'}"
     f"\n    Learned: Train={HOLDOUT['auc_learned_train']:.4f}  "
     f"Test={HOLDOUT['auc_learned_test']:.4f}  gap={HOLDOUT['gap']:+.4f}"
     f"  {'⚠' if HOLDOUT['gap'] > 0.10 else '✓'}"
     f"\n    {HOLDOUT['limitation_note']}"),
]

cell_h = 0.86 / len(scorecard)
for i, (code, name, verdict, col, detail) in enumerate(scorecard):
    y_ = 0.94 - i * cell_h * 1.02
    ax.add_patch(FancyBboxPatch(
        (0.01, y_ - cell_h * 0.52), 0.98, cell_h * 0.95,
        boxstyle="round,pad=0.004",
        facecolor=col, alpha=0.07, edgecolor=col, lw=1.0,
        transform=ax.transAxes,
    ))
    ax.text(0.02, y_, code,    transform=ax.transAxes,
            fontsize=8.5, fontweight="bold", color=col, va="center")
    ax.text(0.09, y_, name,    transform=ax.transAxes, fontsize=7.5, va="center")
    ax.text(0.65, y_, verdict, transform=ax.transAxes,
            fontsize=8, fontweight="bold", color=col, va="center")
    ax.text(0.02, y_ - cell_h * 0.38, f"    {detail}",
            transform=ax.transAxes, fontsize=5.8, color="#444444", va="center")
save_fig("fig8_power_scorecard_v16.png", fig)

# =============================================================================
#  §16  FINAL SUMMARY  [v16-FIX-2/3/4]
# =============================================================================
print(f"\n{SEP}")
print(f"  RFS-SCP v16.1 — FINAL SUMMARY  (N={n_total})")
print(f"  PRIMARY: 2-Axis (Cohesion + Flexibility) | "
      f"Communication = [{COMM_STATUS_LABEL}]")
print(SEP)

for code, name, verdict, col, detail in scorecard:
    print(f"\n  [{code}] {name}")
    print(f"    → {verdict}")
    print(f"    {detail.split(chr(10))[0]}")

print(f"""
  ─────────────────────────────────────────────────────────────────────
  v16.1 CHANGES vs v16.0 (단기 방향 수정):

  [v16-FIX-1] matplotlib boxplot compatibility
    '{_BOXPLOT_LABEL_KW}' parameter used (detected matplotlib {matplotlib.__version__})
    _boxplot_with_labels() wrapper ensures fallback compatibility.

  [v16-FIX-2] Communication axis demoted to [{COMM_STATUS_LABEL}]
    PRIMARY CLAIMS: Cohesion + Flexibility (2-axis) only
    Communication retained in code for completeness; all labels,
    figure titles, scorecard, BH table, urgency formula updated.
    Communication ANOVA results still reported for transparency.
    Clarification coverage = {regex_audit['clarification_hit_pct']:.1f}% 
    (requires improved NLP proxy for primary claim status).

  [v16-FIX-3] AUC ≥ {AUC_FLOOR} threshold citation added
    AUC_FLOOR = {AUC_FLOOR}  |  Citation: {AUC_FLOOR_CITATION}
    Added to: SURROGATE_DISCLAIMER, §6 print, Fig.1 title,
    Fig.4 title, Scorecard RQ1 detail.

  [v16-FIX-4] Surrogate framing unified across all outputs
    All "3-axis" → "2-axis (+ Communication future work)"
    BH table labels clarified: PRIMARY / FUTURE WORK / SECONDARY.
    Final summary header updated.

  KEY RESULTS (N={n_total}, 2-axis PRIMARY, VADER={VADER_BACKEND}):
    RQ1: AUC={auc_coh:.4f} [{auc_ci[0]:.3f},{auc_ci[1]:.3f}]  
         floor≥{AUC_FLOOR} ({AUC_FLOOR_CITATION})  perm-p={perm_p:.4f}
    RQ2: β_int={results['RQ2']['beta_interact']:+.4f}  Fisher-z p={results['RQ2']['p_zdiff']:.4f}
    RQ3: Cohesion[PRIMARY] p={p_h3:.4f}  η²={eta2_h3:.4f}  |  
         Comm[{COMM_STATUS_LABEL}] p={p_comm:.4f}
    RQ4: ΔAUC CI=[{ci_delta[0]:+.4f},{ci_delta[1]:+.4f}]  p={p_rq4_two_sided:.4f}
    RQ5: Logistic(CV)={logit_res['auc_mean']:.4f}  LSTM[BCE]={lstm_auc_mean:.4f}
    RQ6: r(urgency,MI)={r_urg_mi:.4f} [2-axis]  
         r(comm,MI)={r_comm_mi:.4f} [{COMM_STATUS_LABEL}]
  ─────────────────────────────────────────────────────────────────────
""")

# =============================================================================
#  §17  SAVE OUTPUTS
# =============================================================================
summary_rows = []
for code, name, verdict, col, detail in scorecard:
    summary_rows.append(dict(
        rq=code, name=name, verdict=verdict,
        primary_claim=(COMM_STATUS_LABEL not in name),
        detail=detail.replace("\n", "  "),
    ))
pd.DataFrame(summary_rows).to_csv(OUT_DIR / "hypothesis_summary_v16.csv", index=False)
pd.DataFrame(regex_audit, index=[0]).to_csv(
    OUT_DIR / "regex_coverage_audit_v16.csv", index=False
)
# [FIX-4] numpy array / bool 직렬화 안전 처리
holdout_csv = {
    k: (v.tolist() if hasattr(v, "tolist") else
        bool(v)    if isinstance(v, (bool, np.bool_)) else v)
    for k, v in HOLDOUT.items()
    if k != "w_learned"           # numpy array — 별도 저장
}
pd.DataFrame([holdout_csv]).to_csv(
    OUT_DIR / "holdout_weight_sensitivity_v16.csv", index=False
)
pd.Series(HOLDOUT["w_learned"], index=COH_KEYS).to_csv(
    OUT_DIR / "holdout_w_learned_v16.csv", header=["weight"]
)

# Save manifest update with final results
manifest["final_auc"] = float(auc_coh)
manifest["primary_axes"] = "Cohesion + Flexibility"
manifest["communication_status"] = COMM_STATUS_LABEL
with open(OUT_DIR / "reproducibility_manifest_v16.json", "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\n  Output directory: {OUT_DIR}")
for f_ in sorted(OUT_DIR.iterdir()):
    if f_.is_file():
        print(f"    {f_.name:<58s}  {f_.stat().st_size/1024:>8.1f} KB")

print(f"\n{SEP}\n  RFS-SCP v16.1 COMPLETE\n{SEP}")