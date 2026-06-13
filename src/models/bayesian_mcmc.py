#!/usr/bin/env python3
"""
bayesian_mcmc.py — Bayesian MCMC weight optimisation for Circumplex Estimator.

Likelihood : y_i ~ Bernoulli(σ(α·ĉ_i(w) + β))
Prior      : w ~ Dirichlet(α=2.0)  [weakly informative]
MCMC       : Metropolis-Hastings in additive log-ratio (ALR) transform space

NOTE: No temperature constant.  The acceptance ratio uses the exact
log-posterior difference, making posterior widths genuinely data-driven.

This is a refactor of the v16.1 inline MCMC (§9 in rfs_scp_v16_main.py)
into a standalone importable module.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression


# ── ALR / inverse-ALR transforms ─────────────────────────────────────────

def w_to_logit(w: np.ndarray) -> np.ndarray:
    """Additive log-ratio transform: simplex → R^{K-1}."""
    w_s = np.clip(w, 1e-9, 1 - 1e-9)
    w_s /= w_s.sum()
    return np.log(w_s[:-1] / w_s[-1])


def logit_to_w(z: np.ndarray) -> np.ndarray:
    """Inverse ALR: R^{K-1} → simplex."""
    z_f = np.append(z, 0.0)
    z_f -= z_f.max()
    w = np.exp(z_f)
    return w / w.sum()


# ── Log-likelihood and log-prior ─────────────────────────────────────────

def bernoulli_log_likelihood(cohs: np.ndarray, y: np.ndarray) -> float:
    """
    Fit logistic regression to (cohesion scores, labels) and return
    the Bernoulli log-likelihood under the fitted model.
    """
    cohs_2d = cohs.reshape(-1, 1)
    lr = LogisticRegression(max_iter=200, C=10.0, random_state=42)
    try:
        lr.fit(cohs_2d, y)
        probs = np.clip(lr.predict_proba(cohs_2d)[:, 1], 1e-9, 1 - 1e-9)
        return float(np.sum(y * np.log(probs) + (1 - y) * np.log(1 - probs)))
    except Exception:
        return -1e9


def dirichlet_log_prior(w: np.ndarray, alpha: float = 2.0) -> float:
    """Dirichlet(α=2) log-prior: weakly informative."""
    w_s = np.clip(w, 1e-9, 1)
    return float((alpha - 1) * np.sum(np.log(w_s)))


# ── MCMC runner ───────────────────────────────────────────────────────────

def run_mcmc(
    coh_fn: Callable[[np.ndarray], np.ndarray],
    y: np.ndarray,
    init_w: np.ndarray,
    n_steps: int = 5000,
    burn_in: int = 1250,
    dirichlet_alpha: float = 2.0,
    step_size_init: float = 0.30,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, float, Dict]:
    """
    Run Metropolis-Hastings MCMC over cohesion sub-feature weight space.

    Parameters
    ----------
    coh_fn         : callable w_array → cohesion_scores (length N)
    y              : binary labels (0/1), length N
    init_w         : initial weight vector (length K, sums to 1)
    n_steps        : total MCMC iterations
    burn_in        : burn-in iterations (discarded)
    dirichlet_alpha: Dirichlet prior concentration
    step_size_init : initial proposal step size (adaptive)
    seed           : random seed

    Returns
    -------
    chain      : (n_steps - burn_in, K) post-burn-in weight samples
    lp_chain   : log-posterior values for each post-burn-in sample
    accept_rate: overall acceptance rate
    diagnostics: dict with step_size_final, n_accepted, n_total
    """
    K        = len(init_w)
    rng      = np.random.default_rng(seed)
    step_sz  = step_size_init

    def log_posterior(w: np.ndarray) -> float:
        w = np.clip(w, 1e-6, None); w = w / w.sum()
        return (bernoulli_log_likelihood(coh_fn(w), y)
                + dirichlet_log_prior(w, dirichlet_alpha))

    z_cur  = w_to_logit(init_w)
    w_cur  = logit_to_w(z_cur)
    lp_cur = log_posterior(w_cur)

    chain_raw = np.zeros((n_steps, K))
    lp_raw    = np.zeros(n_steps)
    accepted  = 0

    for i in range(n_steps):
        z_prop  = z_cur + rng.normal(0, step_sz, size=K - 1)
        w_prop  = logit_to_w(z_prop)
        lp_prop = log_posterior(w_prop)
        if np.log(rng.random() + 1e-15) < min(0.0, lp_prop - lp_cur):
            z_cur, w_cur, lp_cur = z_prop, w_prop, lp_prop
            accepted += 1
        chain_raw[i] = w_cur
        lp_raw[i]    = lp_cur

        if (i + 1) % 1000 == 0:
            rate = accepted / (i + 1)
            if rate > 0.40:   step_sz *= 1.3
            elif rate < 0.20: step_sz *= 0.7
            step_sz = float(np.clip(step_sz, 0.02, 3.0))

    chain    = chain_raw[burn_in:]
    lp_chain = lp_raw[burn_in:]

    return chain, lp_chain, accepted / n_steps, {
        "step_size_final": step_sz,
        "n_accepted": accepted,
        "n_total": n_steps,
    }
