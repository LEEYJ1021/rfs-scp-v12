#!/usr/bin/env python3
"""
src/models/lstm_model.py

RFS-SCP v16.1 — LSTM Sensitivity Module
=========================================
Extracted LSTM classifier + sensitivity grid logic from the v16.1
main pipeline (§10, RQ5). Provides a BCE-objective LSTM classifier
for session-level MI-quality prediction from utterance sequences,
plus a Ridge fallback when PyTorch is unavailable.

V15-FIX-5: LSTM objective changed from MSE to BCE with class-weighted
           pos_weight (handles 82.7% / 17.3% class imbalance).
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import Ridge

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_OK = True
except ImportError:
    TORCH_OK = False


# =============================================================================
#  LSTM Classifier
# =============================================================================
if TORCH_OK:
    class LSTMClassifier(nn.Module):
        """
        Session-level binary classifier over padded utterance sequences.

        Input shape: (batch, seq_len, input_size)
        Output: logits of shape (batch,) — apply sigmoid for probabilities.
        """
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


# =============================================================================
#  Sensitivity grid (hidden_size × num_layers), BCE objective
# =============================================================================
def run_lstm_sensitivity_grid(
    X_seq: np.ndarray,
    y_bin: np.ndarray,
    seed: int = 42,
    hidden_sizes=(16, 32, 48, 64),
    num_layers_list=(1, 2),
    n_splits=5,
    max_epochs=80,
    patience_limit=10,
    batch_size=16,
    lr=3e-3,
    weight_decay=5e-4,
    verbose=True,
):
    """
    Run a grid search over (hidden_size, num_layers) using StratifiedKFold
    CV with a BCEWithLogitsLoss objective and class-imbalance pos_weight.

    Returns
    -------
    sensitivity_grid : dict
        Keyed by "h{hidden}_l{layers}" -> dict with auc_mean, auc_std,
        mcc_mean, n_skip.
    best_key : str
        Key of the best-performing configuration by auc_mean.
    """
    if not TORCH_OK:
        raise RuntimeError(
            "[LSTM] PyTorch not found — use run_ridge_fallback() instead."
        )

    from sklearn.metrics import matthews_corrcoef

    def mcc_score(y_true, y_pred_bin):
        return float(matthews_corrcoef(y_true, y_pred_bin))

    y_bin_long = torch.FloatTensor(y_bin)
    sensitivity_grid = {}

    if verbose:
        print("\n  LSTM sensitivity grid (BCE objective) …")

    for hs in hidden_sizes:
        for nl in num_layers_list:
            key = f"h{hs}_l{nl}"
            fold_aucs, fold_mccs, n_skip = [], [], 0
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

            for tr_idx, va_idx in skf.split(X_seq, y_bin):
                if len(np.unique(y_bin[va_idx])) < 2:
                    n_skip += 1
                    continue

                X_tr = torch.FloatTensor(X_seq[tr_idx])
                X_va = torch.FloatTensor(X_seq[va_idx])
                y_tr = y_bin_long[tr_idx]
                y_va_np = y_bin[va_idx]

                # Class-imbalance weighting [V15-FIX-5]
                pos_w = torch.tensor(
                    [(y_bin[tr_idx] == 0).sum() / max((y_bin[tr_idx] == 1).sum(), 1)],
                    dtype=torch.float,
                )
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

                model = LSTMClassifier(hidden_size=hs, num_layers=nl)
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=lr, weight_decay=weight_decay
                )
                dataset = TensorDataset(X_tr, y_tr)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

                best_loss, patience = float("inf"), 0
                model.train()
                for epoch in range(max_epochs):
                    epoch_loss = 0.0
                    for xb, yb in dataloader:
                        optimizer.zero_grad()
                        loss = criterion(model(xb), yb)
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        epoch_loss += loss.item()
                    if epoch_loss < best_loss - 1e-4:
                        best_loss = epoch_loss
                        patience = 0
                    else:
                        patience += 1
                    if patience >= patience_limit:
                        break

                model.eval()
                with torch.no_grad():
                    logits_va = model(X_va).numpy()
                probs_va = 1.0 / (1.0 + np.exp(-logits_va))
                preds_bin = (probs_va >= 0.5).astype(int)

                try:
                    fold_aucs.append(roc_auc_score(y_va_np, probs_va))
                    fold_mccs.append(mcc_score(y_va_np.astype(int), preds_bin))
                except Exception:
                    pass

            sensitivity_grid[key] = {
                "hidden": hs,
                "layers": nl,
                "auc_mean": float(np.nanmean(fold_aucs)) if fold_aucs else np.nan,
                "auc_std": float(np.nanstd(fold_aucs)) if fold_aucs else np.nan,
                "mcc_mean": float(np.nanmean(fold_mccs)) if fold_aucs else np.nan,
                "n_skip": n_skip,
            }

            if verbose:
                g = sensitivity_grid[key]
                print(
                    f"    {key}: AUC={g['auc_mean']:.4f}±{g['auc_std']:.4f}  "
                    f"MCC={g['mcc_mean']:.4f}"
                )

    best_key = max(
        sensitivity_grid,
        key=lambda k: sensitivity_grid[k]["auc_mean"]
        if not np.isnan(sensitivity_grid[k]["auc_mean"]) else -999,
    )

    if verbose:
        best = sensitivity_grid[best_key]
        print(f"  Best LSTM [BCE]: {best_key}  AUC={best['auc_mean']:.4f}")

    return sensitivity_grid, best_key


# =============================================================================
#  Ridge fallback (no PyTorch)
# =============================================================================
def run_ridge_fallback(X_seq, y_bin, y_coh, seed=42, n_splits=5, alpha=10.0):
    """
    Fallback path when PyTorch is unavailable: flatten sequences and fit
    a Ridge regressor against the cohesion score, evaluated via AUC on
    the binary MI-quality label.

    Returns
    -------
    auc_mean, auc_std : float
    sensitivity_grid : dict (empty)
    best_key : str ("ridge_fallback")
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X_flat = X_seq.reshape(len(X_seq), -1)
    fold_aucs = []

    for tr_idx, va_idx in skf.split(X_flat, y_bin):
        if len(np.unique(y_bin[va_idx])) < 2:
            continue
        model = Ridge(alpha=alpha)
        model.fit(X_flat[tr_idx], y_coh[tr_idx])
        try:
            fold_aucs.append(roc_auc_score(y_bin[va_idx], model.predict(X_flat[va_idx])))
        except Exception:
            pass

    auc_mean = float(np.nanmean(fold_aucs)) if fold_aucs else np.nan
    auc_std = float(np.nanstd(fold_aucs)) if fold_aucs else np.nan

    return auc_mean, auc_std, {}, "ridge_fallback"


# =============================================================================
#  Unified entry point
# =============================================================================
def run_lstm_or_fallback(X_seq, y_bin, y_coh=None, seed=42, **kwargs):
    """
    Convenience wrapper used by the main pipeline (§10, RQ5).

    If PyTorch is available, runs the full LSTM sensitivity grid.
    Otherwise, falls back to Ridge regression on flattened sequences.

    Returns
    -------
    lstm_auc_mean, lstm_auc_std, lstm_mcc_mean, sensitivity_grid, best_key
    """
    if TORCH_OK:
        grid, best_key = run_lstm_sensitivity_grid(X_seq, y_bin, seed=seed, **kwargs)
        best = grid[best_key]
        return (
            best["auc_mean"],
            best["auc_std"],
            best["mcc_mean"],
            grid,
            best_key,
        )
    else:
        if y_coh is None:
            raise ValueError("y_coh is required for Ridge fallback.")
        auc_mean, auc_std, grid, best_key = run_ridge_fallback(
            X_seq, y_bin, y_coh, seed=seed
        )
        return auc_mean, auc_std, float("nan"), grid, best_key


if __name__ == "__main__":
    print(f"[LSTM] TORCH_OK = {TORCH_OK}")
    if TORCH_OK:
        print(f"[LSTM] torch version: {torch.__version__}")
    else:
        print("[LSTM] PyTorch not found — Ridge fallback available.")