"""
logistic_model.py
=================
Logistic regression wrapper used in the RFS-SCP pipeline.

Provides:
  - build_logistic()       : returns a configured LogisticRegression instance
  - evaluate_logistic()    : fits and returns a dict of evaluation metrics
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    balanced_accuracy_score,
)


def build_logistic(C: float = 1.0, max_iter: int = 500,
                   class_weight: str = "balanced",
                   random_state: int = 42) -> LogisticRegression:
    """Return a configured LogisticRegression estimator."""
    return LogisticRegression(
        C=C,
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=random_state,
        solver="lbfgs",
    )


def evaluate_logistic(model: LogisticRegression,
                      X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Fit model on train set, evaluate on test set.

    Returns
    -------
    dict with keys: auc, ap, mcc, bal_acc, threshold_opt
    """
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    auc = float(roc_auc_score(y_test, proba)) if len(np.unique(y_test)) > 1 else float("nan")
    ap  = float(average_precision_score(y_test, proba)) if len(np.unique(y_test)) > 1 else float("nan")
    mcc = float(matthews_corrcoef(y_test, pred))
    bal = float(balanced_accuracy_score(y_test, pred))

    return dict(auc=auc, ap=ap, mcc=mcc, bal_acc=bal)
