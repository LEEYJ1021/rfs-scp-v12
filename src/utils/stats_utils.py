"""
stats_utils.py
Statistical helper functions for RFS-SCP v12.

Includes: Cohen's d, eta_squared, BH correction, ICC(2,1),
          MCC, specificity, ECE, OLS coef/pval, power analysis.
"""

import numpy as np
from scipy import stats
from scipy.stats import norm as sp_norm
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from numpy.linalg import lstsq
from statsmodels.stats.multitest import multipletests


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled Cohen's d for two independent groups."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    sp = np.sqrt(((len(a) - 1) * np.var(a, ddof=1) +
                  (len(b) - 1) * np.var(b, ddof=1)) / (len(a) + len(b) - 2))
    return float((np.mean(a) - np.mean(b)) / (sp + 1e-12))


def eta_squared(groups: list) -> float:
    """One-way eta-squared effect size."""
    grand = np.concatenate(groups).mean()
    ss_bet = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_tot = sum((v - grand) ** 2 for g in groups for v in g)
    return float(ss_bet / (ss_tot + 1e-12))


def bh_correct(p_values: list, alpha: float = 0.05):
    """Benjamini-Hochberg FDR correction.

    Returns
    -------
    reject : ndarray of bool
    p_corr : ndarray of float
    """
    reject, p_corr, _, _ = multipletests(p_values, alpha=alpha, method="fdr_bh")
    return reject, p_corr


def icc_2_1(a: np.ndarray, b: np.ndarray) -> float:
    """ICC(2,1) — two-way random effects, single measures, absolute agreement."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) < 3:
        return np.nan
    data = np.column_stack([a, b])
    n, k = data.shape
    grand = data.mean()
    ss_r = k * np.sum((data.mean(1) - grand) ** 2)
    ss_c = n * np.sum((data.mean(0) - grand) ** 2)
    ss_t = np.sum((data - grand) ** 2)
    ss_e = ss_t - ss_r - ss_c
    ms_r = ss_r / (n - 1)
    ms_e = ss_e / max((n - 1) * (k - 1), 1)
    return float(np.clip((ms_r - ms_e) / (ms_r + (k - 1) * ms_e + 1e-12), -1, 1))


def mcc_score(y_true: np.ndarray, y_pred_bin: np.ndarray) -> float:
    """Matthews Correlation Coefficient."""
    return float(matthews_corrcoef(y_true, y_pred_bin))


def specificity_score(y_true: np.ndarray, y_pred_bin: np.ndarray) -> float:
    """Specificity (true negative rate)."""
    cm = confusion_matrix(y_true, y_pred_bin)
    if cm.shape == (2, 2):
        tn, fp = cm[0, 0], cm[0, 1]
        return float(tn / (tn + fp + 1e-9))
    return float("nan")


def ece_score(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() > 0:
            ece += mask.sum() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece / max(len(y_true), 1))


def ols_coef_pval(y_vec: np.ndarray, X_mat: np.ndarray):
    """OLS coefficients, SE, t-values, and p-values.

    Returns
    -------
    beta, se, t_val, p_val : ndarray
    """
    beta, _, _, _ = lstsq(X_mat, y_vec, rcond=None)
    y_hat = X_mat @ beta
    sse = np.sum((y_vec - y_hat) ** 2)
    dof = len(y_vec) - X_mat.shape[1]
    mse = sse / max(dof, 1)
    cov_b = mse * np.linalg.pinv(X_mat.T @ X_mat)
    se = np.sqrt(np.diag(cov_b))
    t_val = beta / (se + 1e-15)
    p_val = 2 * stats.t.sf(np.abs(t_val), df=dof)
    return beta, se, t_val, p_val


def fishers_z_test(r1: float, n1: int, r2: float, n2: int):
    """Fisher's z-test for difference between two Pearson correlations.

    Returns
    -------
    z_diff, p_value : float
    """
    z1 = np.arctanh(np.clip(r1, -0.9999, 0.9999))
    z2 = np.arctanh(np.clip(r2, -0.9999, 0.9999))
    se_z = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    z_d = (z1 - z2) / (se_z + 1e-15)
    return float(z_d), float(2 * stats.norm.sf(abs(z_d)))


def power_ttest_ind(n1: int, n2: int, d: float, alpha: float = 0.05) -> float:
    """Approximate power for two-sample independent t-test."""
    se = np.sqrt(1 / n1 + 1 / n2)
    z_alpha = sp_norm.ppf(1 - alpha / 2)
    z_beta = abs(d) / se - z_alpha
    return float(sp_norm.cdf(z_beta))


def n_needed_ttest(d: float, power: float = 0.80, alpha: float = 0.05) -> int:
    """Minimum N per group for two-sample t-test at given power."""
    return int(np.ceil(2 * ((sp_norm.ppf(power) + sp_norm.ppf(1 - alpha / 2)) / d) ** 2))
