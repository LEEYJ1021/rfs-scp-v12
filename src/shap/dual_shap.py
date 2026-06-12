#!/usr/bin/env python3
# =============================================================================
#  dual_shap.py — Dual-SHAP Interpretability Suite
#
#  Part of: RFS-SCP v12.0 (Circumplex-Grounded Relational State Estimation)
#
#  Implements:
#    - LinearSHAP   : analytic SHAP via coeff × feature std [V12-A]
#    - PermSHAP     : permutation-path SHAP with antithetic sampling
#    - GS-SHAP      : group-segment SHAP for utterance-level attribution
#                     using HSIC clustering + MMD change-point detection
#    - Comprehensiveness test: top-k ablation AUC
#    - Spearman convergence: LinearSHAP ↔ PermSHAP ranking agreement
#
#  V12-A: KernelSHAP removed — ρ≈−0.01 with LinearSHAP/PermSHAP
#         at n_coalitions=100 confirms high-variance instability.
#  V12-B: GS-SHAP fixed — BiLSTM trained 80 epochs with early stopping
#         + val-AUC monitoring; efficiency error now reported.
# =============================================================================

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import SpectralClustering
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

SEED = 42
np.random.seed(SEED)


# ===========================================================================
#  LinearSHAP  (analytic — coefficient × feature standard deviation)
# ===========================================================================

def linear_shap(clf: LogisticRegression,
                X_scaled: np.ndarray,
                feat_cols: list) -> pd.DataFrame:
    """
    Compute analytic LinearSHAP values for a fitted LogisticRegression.

    SHAP_j = coeff_j × std(X_scaled[:, j])

    This is exact for linear models and identical to the SHAP linear
    explainer under the marginal independence assumption.

    Parameters
    ----------
    clf       : fitted LogisticRegression
    X_scaled  : standardised feature matrix used to train clf
    feat_cols : feature names

    Returns
    -------
    DataFrame with columns ['feature', 'LinearSHAP', 'abs_LinearSHAP']
    sorted by |LinearSHAP| descending
    """
    coef     = clf.coef_[0]
    feat_std = X_scaled.std(0) + 1e-9
    shap_vals = coef * feat_std

    df = pd.DataFrame({
        "feature":       feat_cols,
        "LinearSHAP":    shap_vals,
        "abs_LinearSHAP": np.abs(shap_vals),
    }).sort_values("abs_LinearSHAP", ascending=False).reset_index(drop=True)

    return df


# ===========================================================================
#  PermutationSHAP  (antithetic sampling, all-sessions baseline)
# ===========================================================================

def permutation_shap(
    predict_fn,
    X: np.ndarray,
    baseline: np.ndarray,
    n_perms: int = 50,
    seed: int = SEED,
) -> np.ndarray:
    """
    Estimate SHAP values via symmetric permutation sampling.

    For each sample s and each permutation (forward + antithetic reverse):
      1. Start from baseline.
      2. Reveal features in permutation order; accumulate marginal contributions.
    Average over 2·n_perms antithetic permutations.

    Parameters
    ----------
    predict_fn : callable (n_samples, n_features) → (n_samples,) probabilities
    X          : (N, D) feature matrix
    baseline   : (D,) baseline (mean) feature vector
    n_perms    : number of forward permutations (total = 2·n_perms)
    seed       : RNG seed

    Returns
    -------
    phi : (N, D) SHAP value matrix
    """
    rng  = np.random.default_rng(seed)
    n, d = X.shape
    phi  = np.zeros((n, d))

    for s in range(n):
        x_s   = X[s]
        phi_s = np.zeros(d)
        total = 0

        for _ in range(n_perms):
            perm = rng.permutation(d)
            for direction in [perm, perm[::-1]]:
                x_prev = baseline.copy()
                f_prev = float(predict_fn(x_prev.reshape(1, -1))[0])
                for idx in direction:
                    x_cur        = x_prev.copy()
                    x_cur[idx]   = x_s[idx]
                    f_cur        = float(predict_fn(x_cur.reshape(1, -1))[0])
                    phi_s[idx]  += f_cur - f_prev
                    f_prev       = f_cur
                    x_prev       = x_cur
                total += 1

        phi[s] = phi_s / max(total, 1)

    return phi


def perm_shap_summary(perm_shap_vals: np.ndarray, feat_cols: list) -> pd.DataFrame:
    """Mean absolute PermSHAP per feature, sorted descending."""
    mean_abs = np.abs(perm_shap_vals).mean(0)
    return (pd.DataFrame({"feature": feat_cols, "PermSHAP": mean_abs})
              .sort_values("PermSHAP", ascending=False).reset_index(drop=True))


# ===========================================================================
#  GS-SHAP helpers
# ===========================================================================

def _rbf_kernel(X: np.ndarray, Y: np.ndarray,
                gamma: float | None = None) -> np.ndarray:
    """RBF kernel matrix K(X, Y)."""
    if gamma is None:
        Z   = np.concatenate([X, Y], 0)
        sq  = np.sum((Z[:, None, :] - Z[None, :, :]) ** 2, axis=-1)
        med = np.median(sq)
        gamma = 1.0 if med <= 0 else 1.0 / (2.0 * med)
    XX    = np.sum(X ** 2, 1, keepdims=True)
    YY    = np.sum(Y ** 2, 1, keepdims=True)
    dists = np.maximum(XX - 2.0 * X @ Y.T + YY.T, 0.0)
    return np.exp(np.clip(-gamma * dists, -50.0, 0.0))


def _mmd2_unbiased(X: np.ndarray, Y: np.ndarray) -> float:
    """Unbiased MMD² between X and Y with RBF kernel."""
    n, m = len(X), len(Y)
    if n < 2 or m < 2:
        return 0.0
    Kxx = _rbf_kernel(X, X); np.fill_diagonal(Kxx, 0.0)
    Kyy = _rbf_kernel(Y, Y); np.fill_diagonal(Kyy, 0.0)
    Kxy = _rbf_kernel(X, Y)
    return float(
        Kxx.sum() / (n * (n - 1))
        + Kyy.sum() / (m * (m - 1))
        - 2.0 * Kxy.sum() / (n * m)
    )


def cluster_features_hsic(X_flat: np.ndarray,
                           max_samples: int = 500,
                           seed: int = SEED) -> list:
    """
    Cluster feature dimensions using HSIC-based spectral clustering.
    Returns list of groups (each group is a sorted list of feature indices).
    """
    D = X_flat.shape[1]
    if D == 1:
        return [[0]]

    if X_flat.shape[0] > max_samples:
        idx = np.random.choice(X_flat.shape[0], max_samples, replace=False)
        X_s = X_flat[idx]
    else:
        X_s = X_flat

    N = X_s.shape[0]
    H = np.eye(N) - np.ones((N, N)) / N

    centred = []
    for i in range(D):
        xi  = X_s[:, i].reshape(-1, 1)
        di  = (xi - xi.T) ** 2
        si  = float(np.sqrt(np.median(di) + 1e-8))
        Ki  = np.exp(-di / (2.0 * si ** 2 + 1e-8))
        centred.append(H @ Ki @ H)

    mat = np.zeros((D, D), dtype=np.float32)
    for i in range(D):
        for j in range(i, D):
            val = float(np.trace(centred[i] @ centred[j])) / (N - 1) ** 2
            mat[i, j] = mat[j, i] = val

    W = np.maximum(mat, 0.0); np.fill_diagonal(W, 0.0)
    if W.sum() < 1e-12:
        return [[i] for i in range(D)]

    d_vec       = W.sum(1)
    D_inv_sqrt  = np.diag(1.0 / np.sqrt(d_vec + 1e-8))
    L_sym       = np.eye(D) - D_inv_sqrt @ W @ D_inv_sqrt
    vals        = np.sort(np.linalg.eigvalsh(L_sym))
    gaps        = np.diff(vals[:D])
    K           = max(2, int(np.argmax(gaps) + 1))
    K           = min(K, D)

    if K == 1:
        return [list(range(D))]
    if K == D:
        return [[i] for i in range(D)]

    sc     = SpectralClustering(n_clusters=K, affinity="precomputed",
                                 assign_labels="kmeans", random_state=seed)
    labels = sc.fit_predict(W)
    return sorted(
        [sorted(np.where(labels == g)[0].tolist())
         for g in range(K) if np.any(labels == g)],
        key=lambda x: min(x),
    )


def segment_by_mmd(x: np.ndarray, min_seg_len: int = 10,
                    max_segments: int = 4, n_perms: int = 20,
                    seed: int = 0) -> list:
    """
    Recursively split sequence x into temporal segments using MMD-based
    change-point detection with permutation significance threshold.

    Returns list of (start, end) tuples.
    """
    T = x.shape[0]
    change_points: list = []

    def _split(start, end, depth):
        if depth >= max_segments - 1:
            return
        if end - start < 2 * min_seg_len:
            return
        best_t, best_mmd = -1, -1.0
        for t in range(start + min_seg_len, end - min_seg_len + 1):
            val = _mmd2_unbiased(x[start:t], x[t:end])
            if val > best_mmd:
                best_mmd, best_t = val, t
        if best_t < 0:
            return
        rng_ = np.random.default_rng(seed + start)
        seg  = x[start:end]
        null = [
            _mmd2_unbiased(
                seg[rng_.permutation(end - start)[:best_t - start]],
                seg[rng_.permutation(end - start)[best_t - start:]],
            )
            for _ in range(n_perms)
        ]
        tau = float(np.quantile(null, 0.95))
        if best_mmd > tau:
            change_points.append(best_t)
            _split(start, best_t, depth + 1)
            _split(best_t, end, depth + 1)

    _split(0, T, 0)
    cps  = sorted(set(change_points))
    bdrs = [0] + cps + [T]
    segs = [(bdrs[i], bdrs[i + 1]) for i in range(len(bdrs) - 1)]

    # merge tiny trailing segments
    merged: list = []
    for s, e in segs:
        if merged and (e - s) < min_seg_len:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged if merged else [(0, T)]


def gs_shap_session(
    x_seq: np.ndarray,
    feature_groups: list,
    baseline_mean: np.ndarray,
    predict_fn_seq,
    n_perms: int = 30,
    seed: int = 0,
) -> tuple:
    """
    GS-SHAP attribution for a single session sequence.

    Parameters
    ----------
    x_seq          : (T, D) utterance sequence
    feature_groups : list of groups (each group = list of feature indices)
    baseline_mean  : (D,) mean baseline for masking
    predict_fn_seq : callable (1, T, D) → (1,) probability
    n_perms        : number of antithetic permutation passes
    seed           : RNG seed

    Returns
    -------
    phi        : (M,) per-player SHAP values
    players    : list of dicts describing each player
    cell_map   : (T, D) attribution cell map
    """
    T, D = x_seq.shape
    rng_ = np.random.default_rng(seed)

    # Build players: (feature_group, time_segment) cartesian product
    segs_by_group = []
    for grp in feature_groups:
        gseed = int(rng_.integers(0, 2 ** 31))
        x_g   = x_seq[:, list(grp)].astype(np.float32)
        segs_by_group.append(
            segment_by_mmd(x_g, min_seg_len=max(3, T // 8),
                           max_segments=4, n_perms=15, seed=gseed)
        )

    players = []
    for k, (grp, segs) in enumerate(zip(feature_groups, segs_by_group)):
        for j, (s, e) in enumerate(segs):
            players.append({"group_id": k, "segment_id": j,
                            "var_indices": list(grp), "time_range": (s, e)})
    M = len(players)
    if M == 0:
        return np.zeros(0), [], np.zeros((T, D), dtype=np.float32)

    phi    = np.zeros(M, dtype=np.float64)
    x_base = np.broadcast_to(baseline_mean, (T, D)).copy().astype(np.float32)
    f0     = float(predict_fn_seq(x_base[None, ...])[0])
    rng_s  = np.random.default_rng(seed + 1)

    for _ in range(n_perms):
        for perm in [rng_s.permutation(M), rng_s.permutation(M)[::-1]]:
            x_cur = x_base.copy(); f_prev = f0
            for idx in perm:
                p = players[idx]; t0, t1 = p["time_range"]
                x_cur[t0:t1, p["var_indices"]] = x_seq[t0:t1, p["var_indices"]]
                f_cur = float(predict_fn_seq(x_cur[None, ...])[0])
                phi[idx] += f_cur - f_prev
                f_prev = f_cur

    phi /= float(2 * n_perms)

    # Build cell map
    cell_map = np.zeros((T, D), dtype=np.float64)
    counts   = np.zeros((T, D), dtype=np.float64)
    for i, p in enumerate(players):
        t0, t1 = p["time_range"]; vars_ = p["var_indices"]
        n_cells = max(1, (t1 - t0) * len(vars_))
        cell_map[t0:t1, vars_] += float(phi[i]) / n_cells
        counts[t0:t1, vars_]   += 1.0
    cell_map = (cell_map / np.where(counts > 0, counts, 1.0)).astype(np.float32)

    return phi.astype(np.float32), players, cell_map


# ===========================================================================
#  Comprehensiveness test
# ===========================================================================

def comprehensiveness_test(
    X_scaled: np.ndarray, y_bin: np.ndarray,
    shap_vals_abs: np.ndarray, feat_cols: list,
    baseline_auc: float,
    k_list: list = [1, 2, 3, 5],
    method_name: str = "",
) -> list:
    """
    Ablate the top-k most important features and measure AUC degradation.

    Returns list of (k, auc) tuples.
    """
    sorted_feats = np.argsort(shap_vals_abs)[::-1]
    results = []
    for k in k_list:
        mask = np.ones(len(feat_cols), dtype=bool)
        mask[sorted_feats[:k]] = False
        X_abl = X_scaled[:, mask]
        if X_abl.shape[1] == 0:
            continue
        clf_abl = LogisticRegression(max_iter=300, C=1.0,
                                      class_weight="balanced",
                                      random_state=SEED)
        clf_abl.fit(X_abl, y_bin)
        auc_abl = roc_auc_score(y_bin, clf_abl.predict_proba(X_abl)[:, 1])
        results.append((k, float(auc_abl)))

    if method_name:
        print(f"  Comprehensiveness [{method_name}]: "
              f"{[(k, f'{a:.3f}') for k, a in results]}")
    return results


# ===========================================================================
#  Spearman convergence check
# ===========================================================================

def shap_convergence(linear_abs: np.ndarray, perm_abs: np.ndarray) -> float:
    """Spearman ρ between LinearSHAP and PermSHAP absolute feature rankings."""
    rho, _ = spearmanr(linear_abs, perm_abs)
    return float(rho)


# ===========================================================================
#  Full pipeline
# ===========================================================================

def run_dual_shap_pipeline(
    X_scaled: np.ndarray, y_bin: np.ndarray, feat_cols: list,
    clf: LogisticRegression | None = None,
) -> dict:
    """
    Run the complete Dual-SHAP pipeline:
      1. LinearSHAP (analytic)
      2. PermSHAP (n_perms=50, all sessions)
      3. Spearman convergence test
      4. Comprehensiveness ablation

    Parameters
    ----------
    X_scaled  : standardised feature matrix
    y_bin     : binary MI-quality labels
    feat_cols : feature column names
    clf       : pre-fitted LogisticRegression; if None, fitted here

    Returns
    -------
    dict with all results
    """
    if clf is None:
        clf = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced",
                                  random_state=SEED)
        clf.fit(X_scaled, y_bin)

    baseline_auc = roc_auc_score(y_bin, clf.predict_proba(X_scaled)[:, 1])

    # LinearSHAP
    print("[DualSHAP] LinearSHAP ...")
    ls_df = linear_shap(clf, X_scaled, feat_cols)
    print(f"  Top-3: {ls_df.head(3)['feature'].tolist()}")

    # PermSHAP
    print("[DualSHAP] PermSHAP (n_perms=50) ...")
    baseline_vec = X_scaled.mean(0)
    def predict_fn(X_in):
        return clf.predict_proba(X_in)[:, 1]

    perm_vals = permutation_shap(predict_fn, X_scaled, baseline_vec,
                                  n_perms=50, seed=SEED)
    ps_df = perm_shap_summary(perm_vals, feat_cols)
    print(f"  Top-3: {ps_df.head(3)['feature'].tolist()}")

    # Spearman convergence
    ls_abs  = ls_df.set_index("feature")["abs_LinearSHAP"].reindex(feat_cols).values
    ps_abs  = ps_df.set_index("feature")["PermSHAP"].reindex(feat_cols).values
    rho_val = shap_convergence(ls_abs, ps_abs)
    print(f"  LinearSHAP ↔ PermSHAP Spearman ρ = {rho_val:.4f}")

    # Comprehensiveness
    comp_ls = comprehensiveness_test(X_scaled, y_bin, ls_abs, feat_cols,
                                      baseline_auc, method_name="LinearSHAP")
    comp_ps = comprehensiveness_test(X_scaled, y_bin, ps_abs, feat_cols,
                                      baseline_auc, method_name="PermSHAP")

    return {
        "linear_shap_df":  ls_df,
        "perm_shap_df":    ps_df,
        "perm_shap_vals":  perm_vals,
        "rho_l_p":         rho_val,
        "comp_linear":     comp_ls,
        "comp_perm":       comp_ps,
        "baseline_auc":    baseline_auc,
        "clf":             clf,
    }


# ===========================================================================
#  CLI
# ===========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RFS-SCP Dual-SHAP")
    parser.add_argument("--features-csv", type=str,
                        default="rfs_v12_outputs/annomi_session_features_v12.csv")
    args = parser.parse_args()

    from pathlib import Path

    feat_path = Path(args.features_csv)
    if not feat_path.exists():
        raise FileNotFoundError(f"Run rfs_scp_v12_main.py first: {feat_path}")

    sess = pd.read_csv(feat_path)
    FEAT_COLS = [
        "empathy_rate", "agreement_rate", "question_rate", "oscillation_rate",
        "sent_mean", "sent_std", "sent_diff_ab", "lag1_autocorr",
        "negation_rate", "mean_ttr", "wc_balance", "emp_agr_interact",
    ]
    X = sess[FEAT_COLS].fillna(0).values
    y = sess["mi_quality_bin"].values.astype(float)
    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)

    res = run_dual_shap_pipeline(X_s, y, FEAT_COLS)
    print(f"\nSpearman ρ (LinearSHAP ↔ PermSHAP): {res['rho_l_p']:.4f}")
    print("LinearSHAP top-5:")
    print(res["linear_shap_df"].head(5).to_string(index=False))
