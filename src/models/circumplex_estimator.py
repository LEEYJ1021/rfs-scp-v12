"""
circumplex_estimator.py
Heuristic CircumplexEstimator: maps session-level dialogue features to
Olson Circumplex cohesion and flexibility coordinates (0–100 scale).

Based on: Olson (2011). FACES IV and the Circumplex Model: Validation Study.
          Journal of Marital and Family Therapy, 37(1), 64–80.
"""

from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np


@dataclass
class CircumplexState:
    """Represents a point on the Olson Circumplex plane.

    Parameters
    ----------
    cohesion : float
        0 = Disengaged, 100 = Enmeshed. Balanced zone: 35–65.
    flexibility : float
        0 = Rigid, 100 = Chaotic. Balanced zone: 35–65.
    """
    cohesion: float = 50.0
    flexibility: float = 50.0

    def __post_init__(self):
        self.cohesion = float(np.clip(self.cohesion, 0, 100))
        self.flexibility = float(np.clip(self.flexibility, 0, 100))

    @property
    def zone(self) -> str:
        """Five-zone classification."""
        c, f = self.cohesion, self.flexibility
        if 35 <= c <= 65 and 35 <= f <= 65:
            return "balanced"
        if c > 65 and f < 35:
            return "rigid-enmeshed"
        if c < 35 and f < 35:
            return "rigid-disengaged"
        if c < 35 and f > 65:
            return "chaotic-disengaged"
        if c > 65 and f > 65:
            return "chaotic-enmeshed"
        # boundary cases
        if f < 35:
            return "rigid-disengaged"
        if f > 65:
            return "chaotic-disengaged"
        if c < 35:
            return "rigid-disengaged"
        return "rigid-enmeshed"

    @property
    def balanced(self) -> bool:
        return self.zone == "balanced"

    @property
    def deviation(self) -> float:
        """Euclidean distance from the centre (50, 50)."""
        return float(np.hypot(self.cohesion - 50, self.flexibility - 50))

    @property
    def robot_state(self) -> dict:
        """Derived signals for RFS controller."""
        return {
            "zone": self.zone,
            "engagement_need": float(np.clip((65 - self.cohesion) / 30, 0, 1)),
            "flexibility_gap": float(np.clip((self.flexibility - 65) / 35, 0, 1)),
            "intervention_urgency": float(np.clip(self.deviation / 50, 0, 1)),
            "target_cohesion": 50.0,
            "target_flexibility": 50.0,
        }


class CircumplexEstimator:
    """Estimates Circumplex cohesion and flexibility from dialogue features.

    Cohesion weights (W_COH) reflect relational bonding signals.
    Flexibility weights (W_FLEX) reflect adaptability signals.

    Both weight sets are normalised to sum to 1.0 at estimation time.
    Weights can be overridden (e.g., by Bayesian MCMC optimisation).
    """

    W_COH: Dict[str, float] = dict(
        empathy=0.24,
        agreement=0.18,
        sent_pos=0.12,
        wc_balance=0.11,
        sent_congruence=0.15,
        neg_absence=0.12,
        sent_div_absence=0.08,
    )

    W_FLEX: Dict[str, float] = dict(
        oscillation=0.22,
        question=0.25,
        sent_variance=0.20,
        novelty=0.13,
        anti_rigidity=0.20,
    )

    def estimate(
        self,
        f: dict,
        w_coh: Optional[Dict[str, float]] = None,
    ) -> CircumplexState:
        """Compute CircumplexState from a feature dictionary.

        Parameters
        ----------
        f : dict
            Session-level feature dictionary. Expected keys:
            empathy_rate, agreement_rate, question_rate,
            oscillation_rate, sent_mean, sent_std, sent_diff_ab,
            negation_rate, mean_ttr, wc_balance, lag1_autocorr,
            emp_agr_interact.
        w_coh : dict, optional
            Override cohesion weight dictionary.

        Returns
        -------
        CircumplexState
        """
        w = w_coh or self.W_COH

        def g(key, default=0.0):
            return float(f.get(key, default) or default)

        def cl(v):
            return float(np.clip(v, 0, 1))

        # --- Cohesion components ---
        emp = cl(g("empathy_rate") / 0.06)
        agr = cl(g("agreement_rate") / 0.25)
        s_pos = (g("sent_mean") + 1) / 2
        bal = g("wc_balance", 0.5)
        scong = 1 - cl(g("sent_diff_ab"))
        neg_a = 1 - cl(g("negation_rate") / 0.30)
        sdiv_a = 1 - cl(g("sent_diff_ab") / 0.60)

        keys = ["empathy", "agreement", "sent_pos", "wc_balance",
                "sent_congruence", "neg_absence", "sent_div_absence"]
        vals = [emp, agr, s_pos, bal, scong, neg_a, sdiv_a]
        coh = 100 * sum(w.get(k, 0) * v for k, v in zip(keys, vals))

        # --- Flexibility components ---
        osc = g("oscillation_rate", 0.5)
        qst = cl(g("question_rate") / 0.20)
        sstd = cl(g("sent_std") / 0.50)
        ttr_v = cl(g("mean_ttr") / 0.80)
        lag1 = g("lag1_autocorr", 0)
        anti_r = float(1.0 / (1.0 + np.exp(3.0 * lag1)))

        wf = self.W_FLEX
        flex = 100 * (
            wf["oscillation"] * osc
            + wf["question"] * qst
            + wf["sent_variance"] * sstd
            + wf["novelty"] * ttr_v
            + wf["anti_rigidity"] * anti_r
        )

        return CircumplexState(round(coh, 2), round(flex, 2))
