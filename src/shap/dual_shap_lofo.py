"""
dual_shap_lofo.py
=================
Dual-SHAP + Leave-One-Feature-Out (LOFO) interpretability module.

Provides three complementary feature importance methods:
  1. LinearSHAP  — analytic: coeff × feature_std
  2. PermSHAP    — permutation-based approximation (n_perms configurable)
  3. LOFO        — cross-validated AUC drop when each feature is removed

All three are combined in the main pipeline (§11 of rfs_scp_v16_main.py)
with Spearman rank correlation to verify convergent validity.

Usage (standalone):
    from src.shap.dual_shap_lofo import linear_shap, permutation_shap, lofo_importance
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


# ── 1. LinearSHAP ─────────────────────────────────────────────────────────────

def linear_shap(model: LogisticRegression,
                X: np.ndarray,
                feature_names: list[str]) -> pd.DataFrame:
    """
    Analytic LinearSHAP: signed importance = coeff_j × std(X_j).

    This is exact for linear models and avoids the Monte Carlo variance
    of kernel/permutation methods.

    Parameters
    ----------
    model : fitted LogisticRegression
    X     : (n_samples, n_features) standardised feature matrix
    feature_names : list of length n_features

    Returns
    -------
    DataFrame with columns [feature, LinearSHAP, abs_LinearSHAP],
    sorted descending by |LinearSHAP|.
    """
    coef = model.coef_[0]
    feat_std = X.std(axis=0) + 1e-9
    shap_vals = coef * feat_std
    df = pd.DataFrame({
        "feature": feature_names,
        "LinearSHAP": shap_vals,
        "abs_LinearSHAP": np.abs(shap_vals),
    }).sort_values("abs_LinearSHAP", ascending=False).reset_index(drop=True)
    return df


# ── 2. PermutationSHAP ───────────────────────────────────────────────────────

def permutation_shap(predict_fn,
                     X: np.ndarray,
                     baseline: np.ndarray,
                     n_perms: int = 50,
                     seed: int = 42) -> np.ndarray:
    """
    Permutation-based SHAP approximation.

    Estimates φ_j(i) for each sample i and feature j by averaging the
    marginal contribution of feature j across random feature orderings
    (both forward and reverse sweeps for variance reduction).

    Parameters
    ----------
    predict_fn : callable, X → 1-D probability array
    X          : (n_samples, n_features)
    baseline   : (n_features,) reference point (typically X.mean(0))
    n_perms    : number of random orderings per sample
    seed       : random seed

    Returns
    -------
    phi : (n_samples, n_features)  — mean |phi| aggregated per feature
          by the caller.
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape
    phi = np.zeros((n, d))

    for s in range(n):
        x_s = X[s]
        phi_s = np.zeros(d)
        total = 0
        for _ in range(n_perms):
            perm = rng.permutation(d)
            for direction in [perm, perm[::-1]]:
                x_prev = baseline.copy()
                f_prev = float(predict_fn(x_prev.reshape(1, -1))[0])
                for idx in direction:
                    x_cur = x_prev.copy()
                    x_cur[idx] = x_s[idx]
                    f_cur = float(predict_fn(x_cur.reshape(1, -1))[0])
                    phi_s[idx] += f_cur - f_prev
                    f_prev = f_cur
                    x_prev = x_cur
                total += 1
        phi[s] = phi_s / max(total, 1)

    return phi


# ── 3. LOFO ──────────────────────────────────────────────────────────────────

def lofo_importance(X: np.ndarray,
                    y: np.ndarray,
                    feature_names: list[str],
                    n_splits: int = 5,
                    n_repeats: int = 3,
                    seed: int = 42,
                    C: float = 1.0) -> pd.DataFrame:
    """
    Leave-One-Feature-Out cross-validated AUC importance.

    For each feature j, computes:
        auc_drop_j = auc_full − auc_{-j}

    Positive drop → feature is informative.
    Zero or negative drop → feature is noise/redundant.

    Bootstrap CI is approximated as ±1.96·SE where SE combines the
    standard deviation of the full and leave-one-out AUC estimates.

    Parameters
    ----------
    X            : (n_samples, n_features) — already standardised
    y            : (n_samples,) binary labels
    feature_names: list of feature names
    n_splits     : cross-validation folds per repeat
    n_repeats    : number of CV repeats
    seed         : random seed
    C            : logistic regression regularisation strength

    Returns
    -------
    DataFrame sorted by auc_drop descending, with columns:
    [feature, auc_without, auc_drop, drop_ci_lo, drop_ci_hi]
    """
    def _cv_auc(X_sub):
        aucs = []
        for rep in range(n_repeats):
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                  random_state=seed + rep)
            for tr, va in skf.split(X_sub, y):
                if len(np.unique(y[va])) < 2:
                    continue
                lr = LogisticRegression(C=C, max_iter=500,
                                        class_weight="balanced",
                                        random_state=seed)
                lr.fit(X_sub[tr], y[tr])
                try:
                    aucs.append(roc_auc_score(y[va],
                                              lr.predict_proba(X_sub[va])[:, 1]))
                except Exception:
                    pass
        return (float(np.nanmean(aucs)) if aucs else float("nan"),
                float(np.nanstd(aucs))  if aucs else float("nan"),
                len(aucs))

    full_auc, full_std, full_n = _cv_auc(X)
    rows = []
    for j, feat in enumerate(feature_names):
        mask = [i for i in range(len(feature_names)) if i != j]
        lo_auc, lo_std, lo_n = _cv_auc(X[:, mask])
        drop = full_auc - lo_auc
        drop_se = np.sqrt((full_std ** 2 + lo_std ** 2) / max(full_n, 1))
        rows.append({
            "feature":    feat,
            "auc_without": lo_auc,
            "auc_drop":    drop,
            "drop_ci_lo":  drop - 1.96 * drop_se,
            "drop_ci_hi":  drop + 1.96 * drop_se,
        })

    df = pd.DataFrame(rows).sort_values("auc_drop", ascending=False).reset_index(drop=True)
    return df


# ── 4. Convergent validity check ─────────────────────────────────────────────

def shap_lofo_rank_correlation(linear_shap_df: pd.DataFrame,
                                lofo_df: pd.DataFrame,
                                feature_names: list[str]) -> dict:
    """
    Compute Spearman rank correlation between |LinearSHAP| and LOFO AUC drop.

    Returns
    -------
    dict with keys: rho, p_value
    """
    ls_rank = linear_shap_df.set_index("feature")["abs_LinearSHAP"].reindex(feature_names).values
    lf_rank = lofo_df.set_index("feature")["auc_drop"].reindex(feature_names).values
    rho, pval = spearmanr(
        pd.Series(ls_rank).rank().values,
        pd.Series(lf_rank).rank().values,
    )
    return dict(rho=float(rho), p_value=float(pval))
