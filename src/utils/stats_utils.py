"""
stats_utils.py
==============
Shared statistical helpers for the RFS-SCP pipeline.

Functions
---------
cohens_d                      — pooled standard deviation effect size
eta_squared                   — one-way ANOVA effect size
bh_correct                    — Benjamini-Hochberg FDR correction
mcc_score / bal_acc_score     — imbalance-robust classification metrics
specificity_score             — true negative rate
ece_score                     — expected calibration error
ols_coef_pval                 — OLS regression with t-statistics
compute_vif                   — variance inflation factors
power_ttest_ind               — post-hoc power for two-sample t-test
net_benefit                   — decision curve analysis net benefit
stratified_bootstrap_auc_ci   — stratified bootstrap AUC CI
fishers_z_test                — Fisher's z test for two correlations
safe_cv_auc                   — cross-validated AUC with fold-skip logic
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.linalg import lstsq
from scipy import stats
from scipy.stats import norm as sp_norm
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold
from statsmodels.stats.multitest import multipletests


# ── Effect sizes ──────────────────────────────────────────────────────────────

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled Cohen's d."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    sp = np.sqrt(
        ((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1))
        / (len(a) + len(b) - 2)
    )
    return float((np.mean(a) - np.mean(b)) / (sp + 1e-12))


def eta_squared(groups: list[np.ndarray]) -> float:
    """One-way ANOVA η²."""
    grand = np.concatenate(groups).mean()
    SS_bet = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    SS_tot = sum((v - grand) ** 2 for g in groups for v in g)
    return float(SS_bet / (SS_tot + 1e-12))


# ── Multiple comparisons ─────────────────────────────────────────────────────

def bh_correct(p_values: list[float],
               alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR correction."""
    reject, p_corr, _, _ = multipletests(p_values, alpha=alpha, method="fdr_bh")
    return reject, p_corr


# ── Classification metrics ───────────────────────────────────────────────────

def mcc_score(y_true: np.ndarray, y_pred_bin: np.ndarray) -> float:
    return float(matthews_corrcoef(y_true, y_pred_bin))


def bal_acc_score(y_true: np.ndarray, y_pred_bin: np.ndarray) -> float:
    return float(balanced_accuracy_score(y_true, y_pred_bin))


def specificity_score(y_true: np.ndarray, y_pred_bin: np.ndarray) -> float:
    cm = confusion_matrix(y_true, y_pred_bin)
    if cm.shape == (2, 2):
        tn, fp = cm[0, 0], cm[0, 1]
        return float(tn / (tn + fp + 1e-9))
    return float("nan")


def ece_score(y_true: np.ndarray, y_prob: np.ndarray,
              n_bins: int = 10) -> float:
    """Expected calibration error."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() > 0:
            ece += mask.sum() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece / max(len(y_true), 1))


# ── Regression helpers ───────────────────────────────────────────────────────

def ols_coef_pval(y_vec: np.ndarray,
                  X_mat: np.ndarray
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    OLS coefficients with t-statistics and p-values.

    Returns
    -------
    beta, se, t_val, p_val  (all shape (p,))
    """
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


def compute_vif(X_mat: np.ndarray) -> np.ndarray:
    """Variance inflation factors for each column of X_mat."""
    n_cols = X_mat.shape[1]
    vifs   = np.full(n_cols, np.nan)
    for j in range(n_cols):
        y_j  = X_mat[:, j]
        X_j  = np.delete(X_mat, j, axis=1)
        if X_j.shape[1] == 0:
            vifs[j] = 1.0
            continue
        beta_j, _, _, _ = ols_coef_pval(y_j, X_j)
        y_hat_j = X_j @ beta_j
        ss_res  = np.sum((y_j - y_hat_j) ** 2)
        ss_tot  = np.sum((y_j - y_j.mean()) ** 2) + 1e-12
        r2_j    = 1.0 - ss_res / ss_tot
        vifs[j] = 1.0 / max(1.0 - r2_j, 1e-9)
    return vifs


# ── Power analysis ───────────────────────────────────────────────────────────

def power_ttest_ind(n1: int, n2: int, d: float,
                    alpha: float = 0.05) -> float:
    """Post-hoc power for two-sample independent t-test."""
    se      = np.sqrt(1 / max(n1, 1) + 1 / max(n2, 1))
    z_alpha = sp_norm.ppf(1 - alpha / 2)
    z_beta  = abs(d) / (se + 1e-12) - z_alpha
    return float(sp_norm.cdf(z_beta))


# ── Decision Curve Analysis ──────────────────────────────────────────────────

def net_benefit(y_true: np.ndarray,
                y_score: np.ndarray,
                threshold: float) -> float:
    """Net benefit at a given probability threshold."""
    yp  = (y_score >= threshold).astype(int)
    tp_ = ((yp == 1) & (y_true == 1)).sum()
    fp_ = ((yp == 1) & (y_true == 0)).sum()
    n   = len(y_true)
    return (tp_ / n) - (fp_ / n) * (threshold / (1 - threshold + 1e-9))


# ── Bootstrap AUC CI ─────────────────────────────────────────────────────────

def stratified_bootstrap_auc_ci(
    y: np.ndarray,
    score: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Tuple[Tuple[float, float], np.ndarray]:
    """
    Stratified bootstrap 95% CI for AUC.

    Stratified = resample positive and negative classes separately to
    maintain prevalence in each bootstrap sample.
    """
    y = np.asarray(y)
    score = np.asarray(score)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    if len(idx_pos) == 0 or len(idx_neg) == 0:
        return (float("nan"), float("nan")), np.array([])
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
        return (float("nan"), float("nan")), boot
    ci = (
        float(np.percentile(boot, 100 * alpha / 2)),
        float(np.percentile(boot, 100 * (1 - alpha / 2))),
    )
    return ci, boot


# ── Fisher's z test ──────────────────────────────────────────────────────────

def fishers_z_test(r1: float, n1: int,
                   r2: float, n2: int) -> Tuple[float, float]:
    """Test equality of two Pearson correlations using Fisher's z transform."""
    z1   = np.arctanh(np.clip(r1, -0.9999, 0.9999))
    z2   = np.arctanh(np.clip(r2, -0.9999, 0.9999))
    se_z = np.sqrt(1 / max(n1 - 3, 1) + 1 / max(n2 - 3, 1))
    z_d  = (z1 - z2) / (se_z + 1e-15)
    return float(z_d), float(2 * stats.norm.sf(abs(z_d)))


# ── Cross-validated AUC with fold-skip ───────────────────────────────────────

def safe_cv_auc(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_fn,
    n_splits: int = 5,
    n_repeats: int = 5,
    use_group: bool = False,
    seed: int = 42,
) -> dict:
    """
    Cross-validated AUC + MCC + BalAcc with single-class fold safety.

    Single-class validation folds are skipped and counted in n_folds_skipped.
    This prevents ValueError on roc_auc_score for highly imbalanced datasets.
    """
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
                n_skipped += 1
                continue
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
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                       random_state=seed + rep)
            for tr_idx, va_idx in splitter.split(X, y):
                n_total += 1
                y_va = y[va_idx]
                if len(np.unique(y_va)) < 2 or len(np.unique(y[tr_idx])) < 2:
                    n_skipped += 1
                    continue
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
        auc_mean=float(np.nanmean(aucs))    if aucs else float("nan"),
        auc_std =float(np.nanstd(aucs))     if aucs else float("nan"),
        mcc_mean=float(np.nanmean(mccs))    if aucs else float("nan"),
        bacc_mean=float(np.nanmean(baccs))  if aucs else float("nan"),
        n_folds_used=len(aucs),
        n_folds_skipped=n_skipped,
        n_folds_total=n_total,
    )
