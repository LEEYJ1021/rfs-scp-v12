#!/usr/bin/env python3
# =============================================================================
#  lstm_model.py — LSTM / BiLSTM Utterance Sequence Encoder
#
#  Part of: RFS-SCP v12.0 (Circumplex-Grounded Relational State Estimation)
#
#  Implements:
#    - LSTMEncoder    : uni-directional LSTM regressor (cohesion target)
#    - BiLSTMClassifier: bidirectional LSTM for binary MI-quality classification
#    - LSTM sensitivity grid: hidden_size × n_layers AUC comparison [V12-I]
#    - GS-SHAP compatible sequence prediction interface
#
#  Requires: PyTorch ≥ 2.0
#  Falls back to Ridge regression when PyTorch is unavailable.
# =============================================================================

import numpy as np
import pandas as pd
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    print("[lstm_model] PyTorch not found — Ridge fallback active")

from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

SEED = 42
np.random.seed(SEED)

MAX_SEQ = 120
N_FEAT  = 5          # vader, ttr, is_therapist, neg_flag, word_count


# ===========================================================================
#  PyTorch models
# ===========================================================================

if TORCH_OK:

    class LSTMEncoder(nn.Module):
        """
        Uni-directional LSTM that predicts a scalar cohesion score from
        an utterance sequence.  Used in RQ5-A ablation study.

        Parameters
        ----------
        input_size  : number of utterance-level features (default 5)
        hidden_size : LSTM hidden dimension
        num_layers  : stacked LSTM depth
        dropout     : dropout rate (applied between layers when num_layers > 1)
        """

        def __init__(self, input_size: int = N_FEAT, hidden_size: int = 48,
                     num_layers: int = 2, dropout: float = 0.3):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size, hidden_size, num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, 24),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(24, 1),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            _, (hn, _) = self.lstm(x)       # hn: (num_layers, B, H)
            return self.fc(hn[-1])           # last layer hidden → scalar


    class BiLSTMClassifier(nn.Module):
        """
        Bidirectional LSTM for binary MI-quality classification.
        Used in GS-SHAP computation (RQ5-B).

        Parameters
        ----------
        input_dim   : utterance feature dimension
        hidden_dim  : per-direction LSTM hidden size
        output_dim  : number of classes (2 for binary)
        """

        def __init__(self, input_dim: int = N_FEAT,
                     hidden_dim: int = 32, output_dim: int = 2):
            super().__init__()
            self.lstm = nn.LSTM(
                input_dim, hidden_dim, num_layers=1,
                batch_first=True, bidirectional=True,
            )
            self.head = nn.Linear(hidden_dim * 2, output_dim)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            out, _ = self.lstm(x)           # (B, T, 2*H)
            return self.head(out[:, -1, :]) # last time-step


# ===========================================================================
#  Training helpers
# ===========================================================================

def _train_epoch(model, loader, optimizer, criterion, clip: float = 1.0):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item()
    return total_loss


def train_lstm_encoder(
    X_seq: np.ndarray, y_coh: np.ndarray,
    hidden_size: int = 48, num_layers: int = 2,
    n_epochs: int = 60, batch_size: int = 16,
    lr: float = 3e-3, patience: int = 8,
    weight_decay: float = 5e-4,
) -> "LSTMEncoder":
    """
    Train LSTMEncoder to predict cohesion from padded utterance sequences.

    Parameters
    ----------
    X_seq   : (N, T, F) padded sequence array
    y_coh   : (N,) float cohesion targets
    Returns fitted LSTMEncoder in eval mode.
    """
    if not TORCH_OK:
        raise RuntimeError("PyTorch required for LSTM training")

    model = LSTMEncoder(input_size=X_seq.shape[2],
                         hidden_size=hidden_size, num_layers=num_layers)
    X_t = torch.FloatTensor(X_seq)
    y_t = torch.FloatTensor(y_coh).unsqueeze(1)
    ds  = TensorDataset(X_t, y_t)
    dl  = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.MSELoss()

    best_loss = float("inf")
    pat_count = 0
    for epoch in range(n_epochs):
        eloss = _train_epoch(model, dl, opt, crit)
        if eloss < best_loss - 1e-4:
            best_loss = eloss; pat_count = 0
        else:
            pat_count += 1
        if pat_count >= patience:
            break

    model.eval()
    return model


def train_bilstm_classifier(
    X_seq: np.ndarray, y_bin: np.ndarray,
    hidden_dim: int = 32, n_epochs: int = 80,
    batch_size: int = 16, lr: float = 5e-4,
    weight_decay: float = 1e-4, patience: int = 15,
) -> "BiLSTMClassifier":
    """
    Train BiLSTMClassifier for binary MI-quality prediction.
    Used in GS-SHAP sequence attribution pipeline.
    """
    if not TORCH_OK:
        raise RuntimeError("PyTorch required for BiLSTM training")

    model = BiLSTMClassifier(input_dim=X_seq.shape[2], hidden_dim=hidden_dim)
    X_t   = torch.FloatTensor(X_seq)
    y_t   = torch.LongTensor(y_bin.astype(np.int64))
    ds    = TensorDataset(X_t, y_t)
    dl    = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.5)
    crit  = nn.CrossEntropyLoss()

    best_loss = float("inf")
    pat_count = 0
    for epoch in range(n_epochs):
        eloss = _train_epoch(model, dl, opt, crit)
        sched.step()
        if eloss < best_loss - 1e-4:
            best_loss = eloss; pat_count = 0
        else:
            pat_count += 1
        if pat_count >= patience:
            break

    model.eval()
    return model


# ===========================================================================
#  Sensitivity grid  [V12-I]
# ===========================================================================

def lstm_sensitivity_grid(
    X_seq: np.ndarray, y_bin: np.ndarray, y_coh: np.ndarray,
    hidden_sizes: list = [16, 32, 48, 64],
    n_layers_list: list = [1, 2],
    n_splits: int = 5,
) -> dict:
    """
    Evaluate LSTMEncoder across a hidden_size × n_layers grid using
    StratifiedKFold cross-validation.

    Returns
    -------
    dict keyed by "h{H}_l{L}" with auc_mean, auc_std, hidden, layers
    """
    if not TORCH_OK:
        print("  [SensGrid] PyTorch unavailable; skipping LSTM grid.")
        return {}

    results = {}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    for hs in hidden_sizes:
        for nl in n_layers_list:
            key = f"h{hs}_l{nl}"
            fold_aucs = []

            for tr_idx, va_idx in skf.split(X_seq, y_bin):
                X_tr = torch.FloatTensor(X_seq[tr_idx])
                X_va = torch.FloatTensor(X_seq[va_idx])
                y_tr = torch.FloatTensor(y_coh[tr_idx]).unsqueeze(1)

                mdl = LSTMEncoder(input_size=N_FEAT, hidden_size=hs, num_layers=nl)
                opt = torch.optim.Adam(mdl.parameters(), lr=3e-3, weight_decay=5e-4)
                crit = nn.MSELoss()
                ds   = TensorDataset(X_tr, y_tr)
                dl   = DataLoader(ds, batch_size=16, shuffle=True)
                best_loss = float("inf"); pat = 0

                mdl.train()
                for _ in range(60):
                    el = _train_epoch(mdl, dl, opt, crit)
                    if el < best_loss - 1e-4: best_loss = el; pat = 0
                    else: pat += 1
                    if pat >= 8: break

                mdl.eval()
                with torch.no_grad():
                    preds = mdl(X_va).squeeze().numpy()
                try:
                    fold_aucs.append(roc_auc_score(y_bin[va_idx], preds))
                except ValueError:
                    pass

            results[key] = {
                "hidden":   hs,
                "layers":   nl,
                "auc_mean": float(np.nanmean(fold_aucs)),
                "auc_std":  float(np.nanstd(fold_aucs)),
            }
            print(f"  {key}: AUC={results[key]['auc_mean']:.4f}"
                  f"±{results[key]['auc_std']:.4f}")

    return results


# ===========================================================================
#  Ridge fallback
# ===========================================================================

def ridge_fallback_eval(X_seq: np.ndarray, y_bin: np.ndarray,
                         y_coh: np.ndarray, n_splits: int = 5,
                         n_repeats: int = 5) -> dict:
    """
    Evaluate a Ridge regression baseline on flattened sequences.
    Used when PyTorch is unavailable.
    """
    from sklearn.model_selection import RepeatedStratifiedKFold
    X_flat = X_seq.reshape(len(X_seq), -1)
    rskf   = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                      random_state=SEED)
    aucs   = []
    for tr_idx, va_idx in rskf.split(X_flat, y_bin):
        rdg = Ridge(alpha=10.0)
        rdg.fit(X_flat[tr_idx], y_coh[tr_idx])
        preds = rdg.predict(X_flat[va_idx])
        try:
            aucs.append(roc_auc_score(y_bin[va_idx], preds))
        except ValueError:
            pass
    return {
        "auc_mean": float(np.nanmean(aucs)),
        "auc_std":  float(np.nanstd(aucs)),
        "method":   "ridge_fallback",
    }


# ===========================================================================
#  Sequence prediction interface for GS-SHAP
# ===========================================================================

def make_seq_predict_fn(bilstm_model):
    """
    Wrap a trained BiLSTMClassifier as a numpy-in / numpy-out
    probability function compatible with the GS-SHAP attribution API.

    Parameters
    ----------
    bilstm_model : trained BiLSTMClassifier

    Returns
    -------
    callable: (np.ndarray of shape (B, T, F)) → (B,) float probabilities
    """
    if not TORCH_OK:
        raise RuntimeError("PyTorch required")

    def predict_fn(x_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            out = bilstm_model(torch.from_numpy(x_np.astype(np.float32)))
        return torch.softmax(out, dim=1)[:, 1].numpy()

    return predict_fn


# ===========================================================================
#  Sequence padding utility
# ===========================================================================

def pad_sequences(seq_dict: dict, ids: np.ndarray,
                  max_len: int = MAX_SEQ, n_feat: int = N_FEAT) -> np.ndarray:
    """
    Pad or truncate utterance sequences to fixed (max_len, n_feat) shape.

    Parameters
    ----------
    seq_dict : dict mapping transcript_id → np.ndarray of shape (T, F)
    ids      : array of transcript IDs to include (in order)

    Returns
    -------
    np.ndarray of shape (N, max_len, n_feat)
    """
    X = np.zeros((len(ids), max_len, n_feat), dtype=np.float32)
    for i, tid in enumerate(ids):
        seq = seq_dict.get(tid, np.zeros((1, n_feat)))[:, :n_feat]
        length = min(len(seq), max_len)
        X[i, :length] = seq[:length]
    return X


# ===========================================================================
#  CLI
# ===========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RFS-SCP LSTM Model")
    parser.add_argument("--features-csv", type=str,
                        default="rfs_v12_outputs/annomi_session_features_v12.csv")
    parser.add_argument("--grid", action="store_true",
                        help="Run LSTM sensitivity grid (slow, requires GPU)")
    args = parser.parse_args()

    feat_path = Path(args.features_csv)
    if not feat_path.exists():
        raise FileNotFoundError(f"Run rfs_scp_v12_main.py first: {feat_path}")

    sess  = pd.read_csv(feat_path)
    y_bin = sess["mi_quality_bin"].values.astype(float)
    y_coh = sess["cohesion"].values.astype(np.float32)

    # Dummy sequence array (replace with real sequences from feature extraction)
    X_seq = np.random.randn(len(sess), MAX_SEQ, N_FEAT).astype(np.float32)

    if not TORCH_OK:
        print("PyTorch unavailable — running Ridge fallback ...")
        res = ridge_fallback_eval(X_seq, y_bin, y_coh)
        print(f"Ridge AUC: {res['auc_mean']:.4f} ± {res['auc_std']:.4f}")
    elif args.grid:
        print("Running LSTM sensitivity grid ...")
        grid = lstm_sensitivity_grid(X_seq, y_bin, y_coh)
        best = max(grid, key=lambda k: grid[k]["auc_mean"])
        print(f"\nBest: {best}  AUC={grid[best]['auc_mean']:.4f}")
    else:
        print("Training BiLSTM classifier ...")
        model = train_bilstm_classifier(X_seq, y_bin.astype(np.int64))
        predict_fn = make_seq_predict_fn(model)
        probs = predict_fn(X_seq[:5])
        print(f"Sample probs (first 5): {probs}")
