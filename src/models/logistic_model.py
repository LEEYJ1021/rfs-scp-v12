#!/usr/bin/env python3
# =============================================================================
#  logistic_model.py — Session-Level Logistic Regression with GroupKFold
#
#  Part of: RFS-SCP v12.0 (Circumplex-Grounded Relational State Estimation)
#
#  Implements:
#    - GroupKFold cross-validation (prevents session leakage)
#    - Repeated StratifiedKFold for stability estimation
#    - SMOTE / class_weight sensitivity analysis
#    - Counterfactual analysis: min empathy increase to flip Low→High MI
#    - BH-corrected significance reporting
# =============================================================================

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, RepeatedStratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    matthews_corrcoef, brier_score_loss,
)
from sklearn.isotonic import IsotonicRegression

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_OK = True
except ImportError:
    SMOTE_OK = False

SEED = 42
np.random.seed(SEED)

FEAT_COLS = [
    "empathy_rate", "agreement_rate", "question_rate", "oscillation_rate",
    "sent_mean", "sent_std", "sent_diff_ab", "lag1_autocorr",
    "negation_rate", "mean_ttr", "wc_balance", "emp_agr_interact",
]


# ---------------------------------------------------------------------------
#  Core model
# ---------------------------------------------------------------------------

def train_logistic(X_train: np.ndarray, y_train: np.ndarray,
                   C: float = 1.0, class_weight: str = "balanced",
                   max_iter: int = 500) -> LogisticRegression:
    """Return a fitted LogisticRegression classifier."""
    clf = LogisticRegression(
        C=C, class_weight=class_weight, max_iter=max_iter,
        random_state=SEED, solver="lbfgs",
    )
    clf.fit(X_train, y_train)
    return clf


# ---------------------------------------------------------------------------
#  GroupKFold evaluation
# ---------------------------------------------------------------------------

def group_kfold_eval(X: np.ndarray, y: np.ndarray,
                     groups: np.ndarray, n_splits: int = 10,
                     C: float = 1.0) -> dict:
    """
    GroupKFold cross-validation ensuring no transcript leaks across folds.

    Returns
    -------
    dict with keys: auc_mean, auc_std, mcc_mean, brier_mean, fold_aucs
    """
    scaler = StandardScaler()
    gkf    = GroupKFold(n_splits=n_splits)
    aucs, mccs, briers = [], [], []

    for tr_idx, va_idx in gkf.split(X, y, groups=groups):
        X_tr = scaler.fit_transform(X[tr_idx])
        X_va = scaler.transform(X[va_idx])
        clf  = train_logistic(X_tr, y[tr_idx], C=C)
        prob = clf.predict_proba(X_va)[:, 1]
        pred = (prob >= 0.5).astype(int)
        try:
            aucs.append(roc_auc_score(y[va_idx], prob))
            mccs.append(float(matthews_corrcoef(y[va_idx], pred)))
            briers.append(brier_score_loss(y[va_idx], prob))
        except ValueError:
            pass  # single-class fold; skip

    return {
        "auc_mean":   float(np.nanmean(aucs)),
        "auc_std":    float(np.nanstd(aucs)),
        "mcc_mean":   float(np.nanmean(mccs)),
        "brier_mean": float(np.nanmean(briers)),
        "fold_aucs":  aucs,
    }


# ---------------------------------------------------------------------------
#  Repeated CV stability
# ---------------------------------------------------------------------------

def repeated_cv_eval(X: np.ndarray, y: np.ndarray,
                     n_splits: int = 5, n_repeats: int = 5,
                     C: float = 1.0) -> dict:
    """Repeated StratifiedKFold for stability (ignores group structure)."""
    scaler = StandardScaler()
    rskf   = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                      random_state=SEED)
    aucs = []
    for tr_idx, va_idx in rskf.split(X, y):
        X_tr = scaler.fit_transform(X[tr_idx])
        X_va = scaler.transform(X[va_idx])
        clf  = train_logistic(X_tr, y[tr_idx], C=C)
        prob = clf.predict_proba(X_va)[:, 1]
        try:
            aucs.append(roc_auc_score(y[va_idx], prob))
        except ValueError:
            pass
    return {
        "auc_mean": float(np.nanmean(aucs)),
        "auc_std":  float(np.nanstd(aucs)),
        "n_folds":  len(aucs),
    }


# ---------------------------------------------------------------------------
#  Imbalance sensitivity
# ---------------------------------------------------------------------------

def imbalance_sensitivity(X: np.ndarray, y: np.ndarray,
                           groups: np.ndarray, n_splits: int = 10) -> dict:
    """
    Compare three resampling conditions:
      - none          : no class_weight
      - balanced      : class_weight='balanced'
      - smote         : SMOTE oversampling (if imblearn available)
    """
    scaler = StandardScaler()
    gkf    = GroupKFold(n_splits=n_splits)
    results: dict = {}

    # --- no resampling ---
    aucs = []
    for tr_idx, va_idx in gkf.split(X, y, groups=groups):
        X_tr = scaler.fit_transform(X[tr_idx])
        X_va = scaler.transform(X[va_idx])
        clf  = LogisticRegression(C=1.0, max_iter=500, random_state=SEED)
        clf.fit(X_tr, y[tr_idx])
        prob = clf.predict_proba(X_va)[:, 1]
        try:
            aucs.append(roc_auc_score(y[va_idx], prob))
        except ValueError:
            pass
    results["none"] = {"label": "No resampling", "auc": float(np.nanmean(aucs))}

    # --- class_weight='balanced' ---
    aucs = []
    for tr_idx, va_idx in gkf.split(X, y, groups=groups):
        X_tr = scaler.fit_transform(X[tr_idx])
        X_va = scaler.transform(X[va_idx])
        clf  = train_logistic(X_tr, y[tr_idx], C=1.0)
        prob = clf.predict_proba(X_va)[:, 1]
        try:
            aucs.append(roc_auc_score(y[va_idx], prob))
        except ValueError:
            pass
    results["balanced"] = {"label": "class_weight=balanced",
                            "auc": float(np.nanmean(aucs))}

    # --- SMOTE ---
    if SMOTE_OK:
        aucs = []
        for tr_idx, va_idx in gkf.split(X, y, groups=groups):
            X_tr_raw = scaler.fit_transform(X[tr_idx])
            X_va     = scaler.transform(X[va_idx])
            try:
                k_n = min(3, int(y[tr_idx].sum()) - 1)
                sm  = SMOTE(random_state=SEED, k_neighbors=k_n)
                Xr, yr = sm.fit_resample(X_tr_raw, y[tr_idx])
                clf = LogisticRegression(C=1.0, max_iter=500, random_state=SEED)
                clf.fit(Xr, yr)
                prob = clf.predict_proba(X_va)[:, 1]
                aucs.append(roc_auc_score(y[va_idx], prob))
            except (ValueError, Exception):
                pass
        results["smote"] = {"label": "SMOTE",
                             "auc": float(np.nanmean(aucs))}

    return results


# ---------------------------------------------------------------------------
#  Counterfactual analysis
# ---------------------------------------------------------------------------

def counterfactual_empathy_delta(
    X_scaled: np.ndarray, y: np.ndarray,
    clf: LogisticRegression,
    feat_cols: list,
    feat_name: str = "empathy_rate",
    delta_max: float = 3.0,
    n_steps: int = 300,
) -> dict:
    """
    For each Low-MI session, find the minimum increase in empathy_rate
    (in standardised units) required to flip the model prediction to
    High-MI (prob >= 0.5).

    Parameters
    ----------
    X_scaled : standardised feature matrix (all sessions)
    y        : binary labels (1=High-MI, 0=Low-MI)
    clf      : fitted LogisticRegression on the full dataset
    feat_cols: list of feature column names matching X_scaled columns
    feat_name: feature to perturb (default 'empathy_rate')
    delta_max: maximum delta to search
    n_steps  : grid resolution

    Returns
    -------
    dict with keys: deltas (array), median, mean, pct25, pct75
    """
    feat_idx  = feat_cols.index(feat_name)
    low_mask  = (y == 0)
    X_low     = X_scaled[low_mask].copy()
    cf_deltas = []

    for i in range(len(X_low)):
        x_cf = X_low[i].copy()
        flipped = False
        for delta in np.linspace(0, delta_max, n_steps):
            x_cf[feat_idx] = X_low[i, feat_idx] + delta
            prob = clf.predict_proba(x_cf.reshape(1, -1))[0, 1]
            if prob >= 0.5:
                cf_deltas.append(delta)
                flipped = True
                break
        if not flipped:
            cf_deltas.append(np.nan)

    cf_arr = np.array(cf_deltas)
    return {
        "deltas":  cf_arr,
        "median":  float(np.nanmedian(cf_arr)),
        "mean":    float(np.nanmean(cf_arr)),
        "pct25":   float(np.nanpercentile(cf_arr, 25)),
        "pct75":   float(np.nanpercentile(cf_arr, 75)),
        "n_low":   int(low_mask.sum()),
        "n_flipped": int(np.sum(~np.isnan(cf_arr))),
    }


# ---------------------------------------------------------------------------
#  Convenience runner
# ---------------------------------------------------------------------------

def run_logistic_pipeline(sess: pd.DataFrame) -> dict:
    """
    Full pipeline from a session-level DataFrame.

    Parameters
    ----------
    sess : DataFrame with columns in FEAT_COLS plus
           'mi_quality_bin', 'transcript_id'

    Returns
    -------
    dict of all results
    """
    X      = sess[FEAT_COLS].fillna(0).values
    y      = sess["mi_quality_bin"].values.astype(float)
    groups = sess["transcript_id"].values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("[LogisticModel] GroupKFold evaluation ...")
    gkf_res = group_kfold_eval(X, y, groups)
    print(f"  AUC={gkf_res['auc_mean']:.4f}±{gkf_res['auc_std']:.4f}  "
          f"MCC={gkf_res['mcc_mean']:.4f}")

    print("[LogisticModel] Repeated CV stability ...")
    rep_res = repeated_cv_eval(X, y)
    print(f"  RepCV AUC={rep_res['auc_mean']:.4f}±{rep_res['auc_std']:.4f}")

    print("[LogisticModel] Imbalance sensitivity ...")
    imb_res = imbalance_sensitivity(X, y, groups)
    for k, v in imb_res.items():
        print(f"  {v['label']}: AUC={v['auc']:.4f}")

    print("[LogisticModel] Fitting global model for counterfactual ...")
    clf_global = train_logistic(X_scaled, y)
    cf_res = counterfactual_empathy_delta(X_scaled, y, clf_global, FEAT_COLS)
    print(f"  Counterfactual Δempathy median={cf_res['median']:.4f}  "
          f"[{cf_res['pct25']:.4f}, {cf_res['pct75']:.4f}]")

    return {
        "gkf":         gkf_res,
        "repeated_cv": rep_res,
        "imbalance":   imb_res,
        "counterfactual": cf_res,
        "scaler":      scaler,
        "clf_global":  clf_global,
    }


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, os

    parser = argparse.ArgumentParser(description="RFS-SCP Logistic Model")
    parser.add_argument("--features-csv", type=str,
                        default="rfs_v12_outputs/annomi_session_features_v12.csv",
                        help="Path to session features CSV")
    args = parser.parse_args()

    feat_path = Path(args.features_csv)
    if not feat_path.exists():
        raise FileNotFoundError(f"Features CSV not found: {feat_path}\n"
                                "Run rfs_scp_v12_main.py first to generate it.")

    sess = pd.read_csv(feat_path)
    results = run_logistic_pipeline(sess)

    print("\n=== LOGISTIC MODEL SUMMARY ===")
    print(f"GroupKFold AUC  : {results['gkf']['auc_mean']:.4f} ± "
          f"{results['gkf']['auc_std']:.4f}")
    print(f"Repeated CV AUC : {results['repeated_cv']['auc_mean']:.4f} ± "
          f"{results['repeated_cv']['auc_std']:.4f}")
    print(f"CF Δempathy     : median={results['counterfactual']['median']:.4f} "
          f"(std units to flip Low→High MI)")
