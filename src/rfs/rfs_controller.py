"""
rfs_controller.py
=================
RFS CircumplexController and InterventionScheduler.

These classes drive robot behaviour based on the estimated Circumplex zone.
The urgency signal is derived from 2-axis (Cohesion + Flexibility) deviation
only; the Communication axis is retained as a future-work signal but is
explicitly excluded from the primary urgency calculation (v16-FIX-2).

Design notes
------------
- Zone policies map each of the five Circumplex zones to a robot role and
  intervention mode.  Priority order: Rigid-Disengaged (5) > Chaotic-
  Disengaged (4) > Chaotic-Enmeshed (3) > Rigid-Enmeshed (2) > Balanced (1).
- The Bayesian MCMC posterior (RQ4) updates the empathy weight used in the
  engagement intensity calculation.
- InterventionScheduler implements a cooldown to prevent over-triggering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


# ── CircumplexState ───────────────────────────────────────────────────────────

@dataclass
class CircumplexState:
    """Scalar Circumplex coordinates for a single session."""
    cohesion:      float = 50.0
    flexibility:   float = 50.0
    communication: float = 50.0   # future work — not primary claim

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
        """2-axis Euclidean deviation from the balanced centre (50, 50)."""
        return float(np.hypot(self.cohesion - 50, self.flexibility - 50))

    @property
    def robot_state(self) -> dict:
        """
        Signals passed to the robot controller.
        intervention_urgency is based on 2-axis deviation only
        (Communication gap excluded from primary urgency; v16-FIX-2).
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


# ── CircumplexController ─────────────────────────────────────────────────────

class CircumplexController:
    """
    Map Circumplex zones to robot behaviour commands.

    Zone policies define role, intervention mode, and signal intensities.
    Empathy weight is initially set from Olson (2011) theory; it is updated
    by calling update_bayesian_weights() with MCMC posterior mean weights.
    """

    ZONE_POLICIES: Dict[str, dict] = {
        "balanced": {
            "role": "MAINTAIN",  "mode": "minimal",
            "emp_int": 0.3, "verbosity": 0.4, "comm_boost": 0.1,
        },
        "rigid-enmeshed": {
            "role": "DIVERSIFY", "mode": "flexibility_boost",
            "emp_int": 0.5, "verbosity": 0.6, "comm_boost": 0.3,
        },
        "rigid-disengaged": {
            "role": "RECONNECT", "mode": "cohesion_build",
            "emp_int": 0.9, "verbosity": 0.8, "comm_boost": 0.7,
        },
        "chaotic-disengaged": {
            "role": "STABILIZE", "mode": "structure_build",
            "emp_int": 0.7, "verbosity": 0.5, "comm_boost": 0.5,
        },
        "chaotic-enmeshed": {
            "role": "MODERATE",  "mode": "boundary_set",
            "emp_int": 0.4, "verbosity": 0.3, "comm_boost": 0.2,
        },
    }

    # Theory weights from Olson (2011); overwritten by MCMC posterior
    COH_KEYS = ["empathy", "agreement", "sent_pos", "wc_balance",
                "sent_congruence", "neg_absence", "sent_div_absence"]

    def __init__(self, w_empathy: float = 0.24, w_agreement: float = 0.18):
        self.w_empathy   = w_empathy
        self.w_agreement = w_agreement
        self.history: List[dict] = []

    def update_bayesian_weights(self, w_post_mean: np.ndarray) -> None:
        """
        Update empathy and agreement weights from MCMC posterior mean.

        Parameters
        ----------
        w_post_mean : array of shape (len(COH_KEYS),) in COH_KEYS order.
        """
        idx_emp = self.COH_KEYS.index("empathy")
        idx_agr = self.COH_KEYS.index("agreement")
        self.w_empathy   = float(w_post_mean[idx_emp])
        self.w_agreement = float(w_post_mean[idx_agr])

    def step(self, state: CircumplexState,
             features: Optional[dict] = None,
             session_id: str = "") -> dict:
        """
        Produce a robot command for the given CircumplexState.

        Parameters
        ----------
        state      : CircumplexState instance
        features   : raw feature dict (used to compute engagement)
        session_id : identifier for logging

        Returns
        -------
        Command dict with keys: zone, robot_role, empathy_intensity,
        communication_boost, verbosity, intervention_urgency.
        """
        features = features or {}
        policy   = self.ZONE_POLICIES.get(state.zone,
                                          self.ZONE_POLICIES["balanced"])
        eng = float(np.clip(
            self.w_empathy   * features.get("empathy_rate",   0) +
            self.w_agreement * features.get("agreement_rate", 0),
            0, 1,
        ))
        # [v16-FIX-2] urgency based on 2-axis deviation only
        comm_gap = state.robot_state["communication_gap"]

        cmd = {
            "session_id":           session_id,
            "timestamp_step":       len(self.history),
            "circumplex_state": {
                "cohesion":       state.cohesion,
                "flexibility":    state.flexibility,
                "communication":  state.communication,
                "zone":           state.zone,
                "comm_quality":   state.communication_quality,
                "deviation":      state.deviation,
            },
            "robot_role":           policy["role"],
            "intervention_mode":    policy["mode"],
            "empathy_intensity":    float(np.clip(policy["emp_int"] * (1 + eng), 0, 1)),
            # comm_boost retained as future-work signal; not in urgency
            "communication_boost":  float(np.clip(policy["comm_boost"] * (1 + comm_gap), 0, 1)),
            "verbosity":            policy["verbosity"],
            "intervention_urgency": float(state.robot_state["intervention_urgency"]),
            "empathy_weight":       self.w_empathy,
        }
        self.history.append(cmd)
        return cmd


# ── InterventionScheduler ─────────────────────────────────────────────────────

class InterventionScheduler:
    """
    Cooldown-based scheduler that decides whether to trigger an intervention.

    Prevents over-triggering by enforcing a minimum gap of `cooldown_steps`
    between successive interventions.  High-urgency zones can override the
    cooldown when a zone change is detected.
    """

    PRIORITY: Dict[str, int] = {
        "rigid-disengaged":   5,
        "chaotic-disengaged": 4,
        "chaotic-enmeshed":   3,
        "rigid-enmeshed":     2,
        "balanced":           1,
    }

    def __init__(self, cooldown_steps: int = 3):
        self.last_zone:    Optional[str] = None
        self.cooldown:     int = 0
        self.cooldown_max: int = cooldown_steps
        self.interventions: List[dict] = []

    def decide(self, state: CircumplexState) -> dict:
        priority  = self.PRIORITY.get(state.zone, 1)
        urgency   = state.robot_state["intervention_urgency"]
        zone_chg  = (self.last_zone is not None
                     and state.zone != self.last_zone)
        trigger = ((priority >= 4 or urgency > 0.7) and self.cooldown == 0)
        trigger = trigger or (zone_chg and priority >= 3)

        decision = {
            "trigger":            trigger,
            "priority":           priority,
            "urgency":            round(urgency, 3),
            "zone":               state.zone,
            "zone_change":        zone_chg,
            "cooldown_remaining": self.cooldown,
        }

        if trigger:
            self.cooldown = self.cooldown_max
            self.interventions.append(decision)
        else:
            self.cooldown = max(0, self.cooldown - 1)

        self.last_zone = state.zone
        return decision
