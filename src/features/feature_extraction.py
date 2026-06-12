"""
feature_extraction.py
Session-level feature extraction from AnnoMI dialogue data.

Extracts 12 core features + 4 temporal rigidity dynamics per session.
Also builds utterance-level padded sequences for LSTM/BiLSTM training.
"""

from __future__ import annotations
import re
import numpy as np
import pandas as pd
from typing import Dict, Tuple

from src.models.circumplex_estimator import CircumplexEstimator

ESTIMATOR = CircumplexEstimator()

# ── Sentiment (VADER preferred, regex fallback) ─────────────────────────────
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _sia = SentimentIntensityAnalyzer()

    def vader(text: str) -> float:
        return _sia.polarity_scores(str(text))["compound"]

except ImportError:
    _POS = re.compile(r'\b(good|great|love|happy|thanks|okay|yes|sure|right)\b', re.I)
    _NEG = re.compile(r'\b(bad|no|not|never|hate|wrong|stop|leave)\b', re.I)

    def vader(text: str) -> float:  # noqa: F811
        pos = len(_POS.findall(str(text)))
        neg = len(_NEG.findall(str(text)))
        return (pos - neg) / max(pos + neg + 1, 1)


NEG_RE = re.compile(r"\b(no|not|never|n't|nothing|nobody|none)\b", re.I)


def ttr(text: str) -> float:
    """Type-token ratio."""
    toks = str(text).lower().split()
    return len(set(toks)) / len(toks) if toks else 0.0


def assign_cluster(topic: str) -> str:
    """Map raw topic string to one of 6 broad clusters."""
    t = topic.lower()
    if any(k in t for k in ["alcohol", "drug", "recidiv", "gambling", "coffee"]):
        return "substance"
    if any(k in t for k in ["smok", "tobacco"]):
        return "smoking"
    if any(k in t for k in ["weight", "diet", "exercise", "activity"]):
        return "health"
    if any(k in t for k in ["asthma", "diabetes", "medicine", "medical",
                             "oral", "birth", "diagnos"]):
        return "medical"
    if any(k in t for k in ["harm", "violen", "school", "assertive",
                             "community", "flatmate", "doi"]):
        return "psychosoc"
    return "other"


def extract_features_with_sequences(
    df: pd.DataFrame,
    dataset_type: str = "full",
    max_seq: int = 120,
    n_feat: int = 5,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[int, np.ndarray]]:
    """Extract session-level features and utterance sequences.

    Parameters
    ----------
    df : pd.DataFrame
        Raw AnnoMI DataFrame (full or simple variant).
    dataset_type : str
        "full" or "simple". Controls which columns are used for
        empathy_rate and question_rate.
    max_seq : int
        Maximum sequence length for padding.
    n_feat : int
        Number of utterance-level features (vader, ttr, is_therapist,
        neg_flag, word_count_norm).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    sess_df : pd.DataFrame
        Session-level features (one row per transcript_id).
    utt_sequences : dict
        Mapping transcript_id → ndarray of shape (T, n_feat).
    """
    np.random.seed(seed)

    # Utterance-level derived columns
    df = df.copy()
    df["vader"] = df["utterance_text"].apply(vader)
    df["neg_flag"] = df["utterance_text"].str.contains(NEG_RE).astype(int)
    df["ttr_val"] = df["utterance_text"].apply(ttr)
    df["is_therapist"] = (df["interlocutor"] == "therapist").astype(int)
    df["word_count"] = df["utterance_text"].str.split().str.len().fillna(0)

    rows, sequences = [], {}

    for tid, grp in df.groupby("transcript_id"):
        grp = grp.sort_values("utterance_id")
        th = grp[grp.interlocutor == "therapist"]
        cl = grp[grp.interlocutor == "client"]

        # empathy_rate
        if dataset_type == "full" and "reflection_exists" in th.columns:
            empathy_rate = (
                (th["reflection_exists"].astype(str).str.lower() == "true").mean()
                if len(th) else 0.0
            )
        else:
            empathy_rate = (
                (th["main_therapist_behaviour"] == "reflection").mean()
                if len(th) else 0.0
            )

        # agreement_rate
        agreement_rate = (
            (cl["client_talk_type"] == "change").mean() if len(cl) else 0.0
        )

        # question_rate
        if dataset_type == "full" and "question_exists" in th.columns:
            question_rate = (
                (th["question_exists"].astype(str).str.lower() == "true").mean()
                if len(th) else 0.0
            )
        else:
            question_rate = (
                (th["main_therapist_behaviour"] == "question").mean()
                if len(th) else 0.0
            )

        # oscillation_rate
        if len(cl) >= 2:
            ct_arr = cl["client_talk_type"].values
            oscillation_rate = sum(
                1 for i in range(len(ct_arr) - 1) if ct_arr[i] != ct_arr[i + 1]
            ) / (len(ct_arr) - 1)
        else:
            oscillation_rate = 0.5

        sent_all = grp["vader"].values
        sent_mean = float(np.mean(sent_all))
        sent_std = float(np.std(sent_all)) if len(sent_all) > 1 else 0.0
        sent_diff_ab = abs(
            (th["vader"].mean() if len(th) else 0.0)
            - (cl["vader"].mean() if len(cl) else 0.0)
        )

        lag1_autocorr = (
            float(np.corrcoef(sent_all[:-1], sent_all[1:])[0, 1])
            if len(sent_all) >= 10 and np.std(sent_all) > 1e-9
            else 0.0
        )

        negation_rate = grp["neg_flag"].mean()
        mean_ttr = grp["ttr_val"].mean()
        wc_balance = min(len(th), len(cl)) / max(len(th), len(cl), 1)
        emp_agr_interact = empathy_rate * agreement_rate

        # Temporal rigidity dynamics (V12-G)
        n_v = len(sent_all)
        transition_entropy = 0.0
        if n_v > 2:
            diffs = np.diff(np.sign(sent_all))
            n_trans = np.sum(diffs != 0)
            p_trans = np.clip(n_trans / max(n_v - 1, 1), 1e-9, 1 - 1e-9)
            transition_entropy = (
                -p_trans * np.log2(p_trans) - (1 - p_trans) * np.log2(1 - p_trans)
            )

        emotional_inertia = abs(lag1_autocorr)
        cohesion_volatility = sent_std
        th_vader = th["vader"].values if len(th) > 0 else np.array([0.0])
        recovery_rate = float(np.mean(np.diff(th_vader) > 0)) if len(th_vader) > 1 else 0.5

        feat = dict(
            empathy_rate=empathy_rate,
            agreement_rate=agreement_rate,
            question_rate=question_rate,
            oscillation_rate=oscillation_rate,
            sent_mean=sent_mean,
            sent_std=sent_std,
            sent_diff_ab=sent_diff_ab,
            negation_rate=negation_rate,
            mean_ttr=mean_ttr,
            wc_balance=wc_balance,
            lag1_autocorr=lag1_autocorr,
            emp_agr_interact=emp_agr_interact,
        )

        cp = ESTIMATOR.estimate(feat)
        mi_quality = grp["mi_quality"].iloc[0]
        topic = grp["topic"].iloc[0].strip()

        rows.append(dict(
            transcript_id=tid,
            mi_quality=mi_quality,
            mi_quality_bin=int(mi_quality == "high"),
            topic=topic,
            topic_cluster=assign_cluster(topic),
            n_turns=len(grp),
            n_therapist=len(th),
            n_client=len(cl),
            **feat,
            cohesion=cp.cohesion,
            flexibility=cp.flexibility,
            zone=cp.zone,
            deviation=cp.deviation,
            balanced=int(cp.balanced),
            transition_entropy=transition_entropy,
            emotional_inertia=emotional_inertia,
            cohesion_volatility=cohesion_volatility,
            empathy_recovery_rate=recovery_rate,
        ))

        seq = grp[["vader", "ttr_val", "is_therapist", "neg_flag", "word_count"]
                  ].values.astype(float)
        seq[:, 4] = np.clip(seq[:, 4] / 50.0, 0, 1)
        sequences[tid] = seq

    sess_df = pd.DataFrame(rows)
    return sess_df, sequences


def pad_sequences(
    seq_dict: dict,
    ids: np.ndarray,
    max_len: int = 120,
    n_feat: int = 5,
) -> np.ndarray:
    """Pad utterance sequences to fixed length.

    Returns
    -------
    ndarray of shape (N, max_len, n_feat)
    """
    X = np.zeros((len(ids), max_len, n_feat), dtype=np.float32)
    for i, tid in enumerate(ids):
        seq = seq_dict.get(tid, np.zeros((1, n_feat)))[:, :n_feat]
        length = min(len(seq), max_len)
        X[i, :length] = seq[:length]
    return X
