#!/usr/bin/env python3
"""
feature_extraction.py — Session-level feature extraction from AnnoMI utterances.

Called by rfs_scp_v16_main.py (§4).  Also importable as a standalone module.

Key outputs per session
-----------------------
Cohesion proxies     : empathy_rate, agreement_rate, sent_mean, wc_balance,
                       sent_diff_ab, negation_rate, mean_ttr, emp_agr_interact
Flexibility proxies  : oscillation_rate, question_rate, sent_std, lag1_autocorr
Communication proxies: turn_balance, topic_shift_rate, clarification_rate,
                       listener_resp_rate  [FUTURE WORK — 1.3% lexical coverage]
Temporal dynamics    : transition_entropy, emotional_inertia,
                       cohesion_volatility, empathy_recovery_rate
Circumplex estimates : cohesion, flexibility, communication, zone, deviation
"""

from __future__ import annotations

import re
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# ── Lexical regex patterns ────────────────────────────────────────────────
NEG_RE    = re.compile(r"\b(no|not|never|n't|nothing|nobody|none)\b", re.I)
CLARIF_RE = re.compile(
    r"\b(so you('re| are)|what i hear|did you mean|you mean|in other words|"
    r"if i understand|let me|i want to make sure|correct me)\b",
    re.I,
)
LISTENER_RE = re.compile(
    r"\b(mm+|uh huh|i see|right|okay|go on|tell me more|yes|sure)\b", re.I
)


def type_token_ratio(text: str) -> float:
    """Lexical diversity — type/token ratio."""
    tokens = str(text).lower().split()
    return len(set(tokens)) / len(tokens) if tokens else 0.0


def extract_session_features(
    grp: pd.DataFrame,
    vader_fn,
    using: str = "full",
) -> Tuple[Dict, np.ndarray]:
    """
    Extract all features for one transcript session.

    Parameters
    ----------
    grp      : DataFrame for a single transcript_id (already VADER-scored)
    vader_fn : callable str → float
    using    : "full" or "simple" (affects column availability)

    Returns
    -------
    feat_dict : flat dict of feature values
    seq       : (T × 5) numpy array — utterance-level sequence for LSTM
                columns: [vader, ttr_val, is_therapist, neg_flag, word_count_norm]
    """
    grp = grp.sort_values("utterance_id")
    th  = grp[grp.interlocutor == "therapist"]
    cl  = grp[grp.interlocutor == "client"]

    # ── Cohesion proxies ─────────────────────────────────────────────────
    if using == "full" and "reflection_exists" in th.columns:
        empathy_rate = (
            th["reflection_exists"].astype(str).str.lower() == "true"
        ).mean() if len(th) else 0.0
    else:
        empathy_rate = (
            th["main_therapist_behaviour"] == "reflection"
        ).mean() if len(th) else 0.0

    agreement_rate = (
        cl["client_talk_type"] == "change"
    ).mean() if len(cl) else 0.0

    if using == "full" and "question_exists" in th.columns:
        question_rate = (
            th["question_exists"].astype(str).str.lower() == "true"
        ).mean() if len(th) else 0.0
    else:
        question_rate = (
            th["main_therapist_behaviour"] == "question"
        ).mean() if len(th) else 0.0

    # ── Flexibility proxies ───────────────────────────────────────────────
    if len(cl) >= 2:
        ct_arr = cl["client_talk_type"].values
        oscillation_rate = sum(
            1 for i in range(len(ct_arr) - 1) if ct_arr[i] != ct_arr[i + 1]
        ) / (len(ct_arr) - 1)
    else:
        oscillation_rate = 0.5

    sent_all     = grp["vader"].values
    sent_mean    = float(np.mean(sent_all))
    sent_std     = float(np.std(sent_all)) if len(sent_all) > 1 else 0.0
    sent_diff_ab = abs(
        (th["vader"].mean() if len(th) else 0.0) -
        (cl["vader"].mean() if len(cl) else 0.0)
    )
    lag1_autocorr = (
        float(np.corrcoef(sent_all[:-1], sent_all[1:])[0, 1])
        if len(sent_all) >= 10 and np.std(sent_all) > 1e-9
        else 0.0
    )

    negation_rate = grp["neg_flag"].mean()
    mean_ttr      = grp["ttr_val"].mean()
    wc_balance    = min(len(th), len(cl)) / max(len(th), len(cl), 1)
    emp_agr_inter = empathy_rate * agreement_rate

    # ── Communication proxies [FUTURE WORK] ──────────────────────────────
    turn_balance       = wc_balance
    topic_shift_rate   = oscillation_rate
    clarification_rate = grp["is_clarif"].mean()
    listener_resp_rate = th["is_listener"].mean() if len(th) else 0.0

    # ── Temporal dynamics ─────────────────────────────────────────────────
    n_v = len(sent_all)
    transition_entropy = 0.0
    if n_v > 2:
        diffs   = np.diff(np.sign(sent_all))
        n_trans = np.sum(diffs != 0)
        p_trans = np.clip(n_trans / max(n_v - 1, 1), 1e-9, 1 - 1e-9)
        transition_entropy = (
            -p_trans * np.log2(p_trans)
            - (1 - p_trans) * np.log2(1 - p_trans)
        )

    emotional_inertia   = abs(lag1_autocorr)
    cohesion_volatility = sent_std
    th_vader = th["vader"].values if len(th) > 0 else np.array([0.0])
    recovery_rate = (
        float(np.mean(np.diff(th_vader) > 0)) if len(th_vader) > 1 else 0.5
    )

    feat_dict = dict(
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
        emp_agr_interact=emp_agr_inter,
        turn_balance=turn_balance,
        topic_shift_rate=topic_shift_rate,
        clarification_rate=clarification_rate,
        listener_resp_rate=listener_resp_rate,
        transition_entropy=transition_entropy,
        emotional_inertia=emotional_inertia,
        cohesion_volatility=cohesion_volatility,
        empathy_recovery_rate=recovery_rate,
    )

    # ── Utterance sequence for LSTM ───────────────────────────────────────
    wc_norm = np.clip(grp["word_count"].values / 50.0, 0, 1)
    seq = np.stack([
        grp["vader"].values,
        grp["ttr_val"].values,
        grp["is_therapist"].values.astype(float),
        grp["neg_flag"].values.astype(float),
        wc_norm,
    ], axis=1).astype(np.float32)

    return feat_dict, seq


def audit_regex_coverage(df: pd.DataFrame) -> dict:
    """Return lexical regex hit rates for Communication proxy audit."""
    n          = len(df)
    clarif_hit = df["utterance_text"].str.contains(CLARIF_RE).sum()
    listen_hit = df["utterance_text"].str.contains(LISTENER_RE).sum()
    neg_hit    = df["utterance_text"].str.contains(NEG_RE).sum()
    return {
        "n_utterances":          n,
        "clarification_hit_pct": round(100 * clarif_hit / max(n, 1), 2),
        "listener_hit_pct":      round(100 * listen_hit / max(n, 1), 2),
        "negation_hit_pct":      round(100 * neg_hit    / max(n, 1), 2),
    }
