#!/usr/bin/env python3
"""
circumplex_estimator.py — Olson Circumplex state estimation from MI features.

Theory-derived feature weights are documented in docs/weight_provenance.md.
Communication axis is retained but labelled [FUTURE WORK] throughout.

Primary axes : Cohesion + Flexibility
Future work  : Communication (lexical coverage ~1.3% in AnnoMI)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class CircumplexState:
    """
    Olson Circumplex state for a single session.

    cohesion      : 0–100  [PRIMARY CLAIM]
    flexibility   : 0–100  [PRIMARY CLAIM]
    communication : 0–100  [FUTURE WORK — partial lexical coverage only]
    """
    cohesion:      float = 50.0
    flexibility:   float = 50.0
    communication: float = 50.0

    def __post_init__(self):
        self.cohesion      = float(np.clip(self.cohesion,      0, 100))
        self.flexibility   = float(np.clip(self.flexibility,   0, 100))
        self.communication = float(np.clip(self.communication, 0, 100))

    @property
    def zone(self) -> str:
        c, f = self.cohesion, self.flexibility
        if 35 <= c <= 65 and 35 <= f <= 65: return "balanced"
        if c > 65 and f < 35:               return "rigid-enmeshed"
        if c < 35 and f < 35:               return "rigid-disengaged"
        if c < 35 and f > 65:               return "chaotic-disengaged"
        if c > 65 and f > 65:               return "chaotic-enmeshed"
        if f < 35:                           return "rigid-disengaged"
        if f > 65:                           return "chaotic-disengaged"
        if c < 35:                           return "rigid-disengaged"
        return "rigid-enmeshed"

    @property
    def communication_quality(self) -> str:
        if self.communication >= 65: return "open"
        if self.communication >= 35: return "moderate"
        return "restricted"

    @property
    def balanced(self) -> bool:
        return self.zone == "balanced"

    @property
    def deviation(self) -> float:
        """Euclidean distance from the balanced centre (50, 50)."""
        return float(np.hypot(self.cohesion - 50, self.flexibility - 50))

    @property
    def robot_state(self) -> dict:
        """
        Robot intervention signals derived from the 2-axis state.
        intervention_urgency is based on 2-axis deviation only.
        Communication gap retained as a future-work auxiliary signal.
        """
        return {
            "zone":                 self.zone,
            "engagement_need":      float(np.clip((65 - self.cohesion) / 30, 0, 1)),
            "flexibility_gap":      float(np.clip((self.flexibility - 65) / 35, 0, 1)),
            "communication_gap":    float(np.clip((65 - self.communication) / 65, 0, 1)),
            "intervention_urgency": float(np.clip(self.deviation / 50, 0, 1)),
            "target_cohesion":      50.0,
            "target_flexibility":   50.0,
            "target_communication": 75.0,
        }


# ── Theory-derived feature weights (Olson 2011; Miller & Rollnick 2012) ──
# See docs/weight_provenance.md for full literature mapping.
W_COH_THEORY: Dict[str, float] = {
    "empathy":          0.24,  # Olson(2011) Table 3; MI reflection behaviour
    "agreement":        0.18,  # Olson(2011) togetherness; MI change talk
    "sent_pos":         0.12,  # Gottman(1994) positivity ratio
    "wc_balance":       0.11,  # Olson(2011) turn balance
    "sent_congruence":  0.15,  # Olson(2011) emotional congruence
    "neg_absence":      0.12,  # Gottman(1994) negative affect reduction
    "sent_div_absence": 0.08,  # Olson(2011) shared reality
}

W_FLEX_THEORY: Dict[str, float] = {
    "oscillation":    0.22,   # Adaptability; client talk-type transitions
    "question":       0.25,   # Role flexibility via questioning
    "sent_variance":  0.20,   # Affective range
    "novelty":        0.13,   # Lexical diversity as cognitive flexibility
    "anti_rigidity":  0.20,   # Inverse of emotional inertia
}

W_COMM_THEORY: Dict[str, float] = {
    "question_rate":       0.25,  # FUTURE WORK
    "turn_balance":        0.20,
    "topic_shift_rate":    0.18,
    "clarification_rate":  0.20,
    "listener_resp_rate":  0.17,
}


class CircumplexEstimator:
    """
    Estimate Olson Circumplex (Cohesion, Flexibility, Communication) from
    a feature dictionary produced by feature_extraction.py.

    Weights default to theory-derived values (W_COH_THEORY, W_FLEX_THEORY).
    W_COH_LEARNED can be set externally after Bayesian MCMC optimisation.
    """

    W_COH_THEORY   = W_COH_THEORY
    W_FLEX         = W_FLEX_THEORY
    W_COMM         = W_COMM_THEORY
    W_COH_LEARNED: Optional[Dict[str, float]] = None

    def estimate(
        self,
        f: dict,
        w_coh: Optional[Dict[str, float]] = None,
        w_comm: Optional[Dict[str, float]] = None,
    ) -> CircumplexState:
        """
        Compute CircumplexState from a feature dict.

        Parameters
        ----------
        f      : feature dict from feature_extraction.extract_session_features()
        w_coh  : override cohesion weights (default: W_COH_LEARNED or W_COH_THEORY)
        w_comm : override communication weights (default: W_COMM)
        """
        wc = w_coh  or self.W_COH_LEARNED or self.W_COH_THEORY
        wm = w_comm or self.W_COMM

        g  = lambda k, d=0.: float(f.get(k, d) or d)
        cl = lambda v: float(np.clip(v, 0, 1))

        # ── Cohesion sub-components ───────────────────────────────────────
        emp    = cl(g("empathy_rate")  / 0.06)
        agr    = cl(g("agreement_rate") / 0.25)
        s_pos  = (g("sent_mean") + 1) / 2
        bal    = g("wc_balance", 0.5)
        scong  = 1 - cl(g("sent_diff_ab"))
        neg_a  = 1 - cl(g("negation_rate") / 0.30)
        sdiv_a = 1 - cl(g("sent_diff_ab") / 0.60)

        coh_keys = ["empathy", "agreement", "sent_pos", "wc_balance",
                    "sent_congruence", "neg_absence", "sent_div_absence"]
        coh_vals = [emp, agr, s_pos, bal, scong, neg_a, sdiv_a]
        coh = 100 * sum(wc.get(k, 0) * v for k, v in zip(coh_keys, coh_vals))

        # ── Flexibility sub-components ────────────────────────────────────
        osc    = g("oscillation_rate", 0.5)
        qst    = cl(g("question_rate") / 0.20)
        sstd   = cl(g("sent_std")      / 0.50)
        ttr_v  = cl(g("mean_ttr")      / 0.80)
        lag1   = g("lag1_autocorr", 0)
        anti_r = float(1.0 / (1.0 + np.exp(3.0 * lag1)))

        wf   = self.W_FLEX
        flex = 100 * (
            wf["oscillation"]   * osc   +
            wf["question"]      * qst   +
            wf["sent_variance"] * sstd  +
            wf["novelty"]       * ttr_v +
            wf["anti_rigidity"] * anti_r
        )

        # ── Communication sub-components [FUTURE WORK] ───────────────────
        q_rate    = cl(g("question_rate")          / 0.20)
        t_bal     = g("turn_balance", 0.5)
        ts_rate   = cl(g("topic_shift_rate",  0.0) / 0.30)
        clar_rate = cl(g("clarification_rate", 0.0) / 0.15)
        lr_rate   = cl(g("listener_resp_rate", 0.0) / 0.20)

        comm_keys = ["question_rate", "turn_balance", "topic_shift_rate",
                     "clarification_rate", "listener_resp_rate"]
        comm_vals = [q_rate, t_bal, ts_rate, clar_rate, lr_rate]
        comm = 100 * sum(wm.get(k, 0) * v for k, v in zip(comm_keys, comm_vals))

        return CircumplexState(
            round(coh,  2),
            round(flex, 2),
            round(comm, 2),
        )
