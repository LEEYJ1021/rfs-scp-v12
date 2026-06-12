"""
rfs_controller.py
Robot Family System (RFS) CircumplexController and InterventionScheduler.

Maps Circumplex zone estimates to robot behaviour policies and determines
when intervention triggers should fire.

Policy table derived from:
  Olson (2011). FACES IV and the Circumplex Model.
  Hirano & Tanaka (2026). Dialogue Generation for Family Robots.
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional

from src.models.circumplex_estimator import CircumplexEstimator, CircumplexState

ESTIMATOR = CircumplexEstimator()
COH_KEYS = list(ESTIMATOR.W_COH.keys())


class CircumplexController:
    """Maps Circumplex state → robot intervention policy.

    Zone policies define how the robot ensemble should behave to
    steer the family state toward the balanced zone.

    Parameters
    ----------
    w_empathy : float
        Initial empathy feature weight (overrideable by Bayesian update).
    w_agreement : float
        Initial agreement feature weight.
    """

    ZONE_POLICIES = {
        "balanced": {
            "role": "MAINTAIN",
            "intervention_mode": "minimal",
            "empathy_intensity": 0.3,
            "verbosity": 0.4,
        },
        "rigid-enmeshed": {
            "role": "DIVERSIFY",
            "intervention_mode": "flexibility_boost",
            "empathy_intensity": 0.5,
            "verbosity": 0.6,
        },
        "rigid-disengaged": {
            "role": "RECONNECT",
            "intervention_mode": "cohesion_build",
            "empathy_intensity": 0.9,
            "verbosity": 0.8,
        },
        "chaotic-disengaged": {
            "role": "STABILIZE",
            "intervention_mode": "structure_build",
            "empathy_intensity": 0.7,
            "verbosity": 0.5,
        },
        "chaotic-enmeshed": {
            "role": "MODERATE",
            "intervention_mode": "boundary_set",
            "empathy_intensity": 0.4,
            "verbosity": 0.3,
        },
    }

    def __init__(self, w_empathy: float = 0.24, w_agreement: float = 0.18):
        self.w_empathy = w_empathy
        self.w_agreement = w_agreement
        self.history: List[dict] = []

    def update_bayesian_weights(self, w_post_mean: np.ndarray) -> None:
        """Update controller weights from Bayesian MCMC posterior means.

        Parameters
        ----------
        w_post_mean : np.ndarray
            Posterior mean weights aligned to COH_KEYS order.
        """
        self.w_empathy = float(w_post_mean[COH_KEYS.index("empathy")])
        self.w_agreement = float(w_post_mean[COH_KEYS.index("agreement")])

    def step(self, features: dict, session_id: str = "") -> dict:
        """Compute one control step.

        Parameters
        ----------
        features : dict
            Session-level dialogue feature dictionary.
        session_id : str
            Identifier for logging.

        Returns
        -------
        dict
            Robot command including zone, role, empathy_intensity, urgency.
        """
        state = ESTIMATOR.estimate(features)
        policy = self.ZONE_POLICIES.get(state.zone, self.ZONE_POLICIES["balanced"])

        eng = float(
            np.clip(
                self.w_empathy * features.get("empathy_rate", 0)
                + self.w_agreement * features.get("agreement_rate", 0),
                0, 1,
            )
        )

        cmd = {
            "session_id": session_id,
            "timestamp_step": len(self.history),
            "circumplex_state": {
                "cohesion": state.cohesion,
                "flexibility": state.flexibility,
                "zone": state.zone,
                "deviation": state.deviation,
            },
            "robot_role": policy["role"],
            "intervention_mode": policy["intervention_mode"],
            "empathy_intensity": float(
                np.clip(policy["empathy_intensity"] * (1 + eng), 0, 1)
            ),
            "verbosity": policy["verbosity"],
            "intervention_urgency": float(state.robot_state["intervention_urgency"]),
            "empathy_weight": self.w_empathy,
        }
        self.history.append(cmd)
        return cmd


class InterventionScheduler:
    """Determines when robot intervention should fire.

    Priority table (higher = more urgent):
        rigid-disengaged    → 5
        chaotic-disengaged  → 4
        chaotic-enmeshed    → 3
        rigid-enmeshed      → 2
        balanced            → 1

    A trigger fires when:
      - priority >= 4 OR urgency > 0.7 AND cooldown == 0
      - OR zone changed AND priority >= 3

    Parameters
    ----------
    cooldown_steps : int
        Minimum steps between consecutive interventions.
    """

    PRIORITY = {
        "rigid-disengaged": 5,
        "chaotic-disengaged": 4,
        "chaotic-enmeshed": 3,
        "rigid-enmeshed": 2,
        "balanced": 1,
    }

    def __init__(self, cooldown_steps: int = 3):
        self.last_zone: Optional[str] = None
        self.cooldown: int = 0
        self.cooldown_max: int = cooldown_steps
        self.interventions: List[dict] = []

    def decide(self, state: CircumplexState) -> dict:
        """Decide whether to trigger intervention.

        Parameters
        ----------
        state : CircumplexState

        Returns
        -------
        dict with keys: trigger, priority, urgency, zone, zone_change, cooldown_remaining
        """
        priority = self.PRIORITY.get(state.zone, 1)
        urgency = state.robot_state["intervention_urgency"]
        zone_chg = self.last_zone is not None and state.zone != self.last_zone

        trigger = (priority >= 4 or urgency > 0.7) and self.cooldown == 0
        trigger = trigger or (zone_chg and priority >= 3)

        dec = {
            "trigger": trigger,
            "priority": priority,
            "urgency": round(urgency, 3),
            "zone": state.zone,
            "zone_change": zone_chg,
            "cooldown_remaining": self.cooldown,
        }

        if trigger:
            self.cooldown = self.cooldown_max
            self.interventions.append(dec)
        else:
            self.cooldown = max(0, self.cooldown - 1)

        self.last_zone = state.zone
        return dec
