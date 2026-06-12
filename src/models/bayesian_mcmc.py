"""
bayesian_mcmc.py
Bayesian weight optimisation for CircumplexEstimator cohesion weights.

Two methods:
1. Nelder-Mead (scipy.optimize.minimize) in log-space
2. Adaptive MCMC in additive log-ratio (ALR) space

Both maximise ROC-AUC of cohesion against session-level MI quality labels.
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from typing import List, Tuple

from src.models.circumplex_estimator import CircumplexEstimator

ESTIMATOR = CircumplexEstimator()
COH_KEYS = list(ESTIMATOR.W_COH.keys())
BASE_W = np.array([ESTIMATOR.W_COH[k] for k in COH_KEYS])


def compute_cohesion_with_w(sess_df, w_arr: np.ndarray) -> np.ndarray:
    """Recompute session cohesion scores given a weight vector."""
    w = np.clip(w_arr, 1e-6, None)
    w = w / w.sum()
    w_dict = dict(zip(COH_KEYS, w))

    feat_keys = ["empathy_rate", "agreement_rate", "sent_mean", "wc_balance",
                 "sent_diff_ab", "negation_rate", "mean_ttr", "lag1_autocorr",
                 "oscillation_rate", "question_rate", "sent_std"]

    return sess_df.apply(
        lambda r: ESTIMATOR.estimate(
            {c: r.get(c, 0) for c in feat_keys}, w_coh=w_dict
        ).cohesion,
        axis=1,
    ).values


def neg_auc_w(w_arr: np.ndarray, sess_df, y_bin: np.ndarray) -> float:
    """Negative AUC objective (for minimization)."""
    cohs = compute_cohesion_with_w(sess_df, w_arr)
    try:
        return -roc_auc_score(y_bin, cohs)
    except Exception:
        return 0.0


def nelder_mead_optimize(sess_df, y_bin: np.ndarray) -> Tuple[np.ndarray, float]:
    """Nelder-Mead optimisation in log-space.

    Returns
    -------
    w_opt : ndarray  (normalised weights)
    auc_opt : float
    """
    res = minimize(
        lambda x: neg_auc_w(np.exp(x), sess_df, y_bin),
        np.log(BASE_W),
        method="Nelder-Mead",
        options={"maxiter": 1200, "xatol": 1e-5, "fatol": 1e-6},
    )
    w_raw = np.exp(res.x)
    w_opt = w_raw / w_raw.sum()
    return w_opt, -res.fun


def _w_to_logit(w: np.ndarray) -> np.ndarray:
    w_s = np.clip(w, 1e-9, 1 - 1e-9)
    w_s /= w_s.sum()
    return np.log(w_s[:-1] / w_s[-1])


def _logit_to_w(z: np.ndarray) -> np.ndarray:
    z_f = np.append(z, 0.0)
    z_f -= z_f.max()
    w = np.exp(z_f)
    return w / w.sum()


def adaptive_mcmc(
    sess_df,
    y_bin: np.ndarray,
    n_mcmc: int = 5000,
    burn_in_frac: float = 0.25,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Adaptive MCMC in additive log-ratio (ALR) space.

    Uses a Metropolis-Hastings accept/reject step with likelihood
    proportional to exp(AUC × 300).

    Parameters
    ----------
    sess_df : DataFrame
    y_bin : ndarray of int
    n_mcmc : int
    burn_in_frac : float
    seed : int
    verbose : bool

    Returns
    -------
    dict with keys:
      chain, w_post_mean, w_post_std, w_post_ci, final_accept_rate
    """
    burn_in = int(n_mcmc * burn_in_frac)
    K = len(COH_KEYS)

    step_size = 0.30
    z_cur = _w_to_logit(BASE_W)
    w_cur = _logit_to_w(z_cur)
    auc_cur = -neg_auc_w(w_cur, sess_df, y_bin)

    chain = np.zeros((n_mcmc, K))
    accepted = 0
    rng = np.random.default_rng(seed)

    for i in range(n_mcmc):
        z_prop = z_cur + rng.normal(0, step_size, size=K - 1)
        w_prop = _logit_to_w(z_prop)
        auc_prop = -neg_auc_w(w_prop, sess_df, y_bin)

        if np.log(rng.random() + 1e-15) < min(0.0, (auc_prop - auc_cur) * 300.0):
            z_cur, w_cur, auc_cur = z_prop, w_prop, auc_prop
            accepted += 1

        chain[i] = w_cur

        if verbose and (i + 1) % 500 == 0:
            rate = accepted / (i + 1)
            if rate > 0.40:
                step_size *= 1.3
            elif rate < 0.20:
                step_size *= 0.7
            step_size = float(np.clip(step_size, 0.02, 3.0))
            print(f"  MCMC step {i+1}/{n_mcmc}: accept={rate:.1%}  step={step_size:.3f}")

    final_accept_rate = accepted / n_mcmc
    chain_burn = chain[burn_in:]
    w_post_mean = chain_burn.mean(0)
    w_post_std = chain_burn.std(0)
    w_post_ci = np.percentile(chain_burn, [2.5, 97.5], axis=0)

    return {
        "chain": chain_burn,
        "w_post_mean": w_post_mean,
        "w_post_std": w_post_std,
        "w_post_ci": w_post_ci,
        "final_accept_rate": final_accept_rate,
        "dominant_key": COH_KEYS[int(np.argmax(w_post_mean))],
    }


def loo_sensitivity(sess_df, y_bin: np.ndarray) -> dict:
    """Leave-one-weight-out AUC sensitivity analysis."""
    base_auc = roc_auc_score(y_bin, compute_cohesion_with_w(sess_df, BASE_W))
    loo_aucs = {}
    for key in COH_KEYS:
        w_loo = {k: v for k, v in ESTIMATOR.W_COH.items() if k != key}
        tot = sum(w_loo.values())
        w_n = {k: v / tot for k, v in w_loo.items()}
        cohs = compute_cohesion_with_w(sess_df, np.array([w_n.get(k, 0) for k in COH_KEYS]))
        loo_aucs[key] = roc_auc_score(y_bin, cohs)
    return {k: v - base_auc for k, v in loo_aucs.items()}
