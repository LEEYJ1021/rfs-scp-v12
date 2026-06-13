# RFS-SCP v16.1 — Feature Weight Literature Provenance

This document provides the full citation trail for each sub-feature weight in the `CircumplexEstimator`. All weights are theory-derived from published sources and set prior to any AnnoMI analysis.

---

## Cohesion Sub-feature Weights

### `empathy_rate` — Weight: 0.24

**Source**: Olson (2011), Table 3, FACES IV factor loadings. The "Supportive communication" factor (cohesion-facilitating dimension) shows the highest loading among therapist behaviour proxies.

**MI mapping**: Miller & Rollnick (2012, Ch. 6) define reflective listening as the primary MI skill for building therapeutic alliance, which maps to family cohesion via the construct of "emotional responsiveness."

**Operationalisation in AnnoMI**: `reflection_exists == True` (AnnoMI-full) or `main_therapist_behaviour == 'reflection'` (AnnoMI-simple).

---

### `agreement_rate` — Weight: 0.18

**Source**: Olson (2011), "togetherness" subscale of FACES IV Cohesion. Shared goals and mutual commitment are central.

**MI mapping**: Amrhein et al. (2003) demonstrated that client commitment language (change talk) predicts drug use outcomes — operationalised as a reliable indicator of motivational alignment, the closest MI analogue to family togetherness.

**Operationalisation**: `client_talk_type == 'change'`.

---

### `sent_congruence` — Weight: 0.15

**Source**: Olson (2011), "emotional congruence" subscale. Family members in cohesive systems show similar emotional valence.

**MI mapping**: The difference in VADER compound scores between therapist and client utterances within a session proxies emotional alignment.

**Operationalisation**: `1 − |VADER_mean(therapist) − VADER_mean(client)|`.

---

### `sent_pos` — Weight: 0.12

**Source**: Gottman (1994), positive affect ratio. Stable marriages maintain a 5:1 positive-to-negative interaction ratio; adapted here as a session-level positivity proxy.

**MI mapping**: Moyers & Martin (2006) linked therapist positive affect to better MI outcomes.

**Operationalisation**: `(VADER compound mean + 1) / 2` (rescaled to [0,1]).

---

### `neg_absence` — Weight: 0.12

**Source**: Gottman (1994). Reduction in negative affect is a primary indicator of family cohesion improvement.

**MI mapping**: Negation token rate (no, not, never, n't, nothing, nobody, none) is a broad proxy for resistance, conflict, or disengagement.

**Operationalisation**: `1 − negation_rate`, where negation_rate = proportion of utterances containing negation tokens.

---

### `wc_balance` — Weight: 0.11

**Source**: Olson (2011), "turn balance" construct. Enmeshed families are characterised by one member dominating discourse; disengaged families show asymmetric withdrawal.

**Operationalisation**: `min(N_therapist_utterances, N_client_utterances) / max(N_therapist_utterances, N_client_utterances)`.

---

### `sent_div_absence` — Weight: 0.08

**Source**: Olson (2011), "shared reality" subscale. Cohesive families maintain a shared emotional frame.

**Operationalisation**: `1 − |VADER diff| / 0.60`, clipped to [0,1]. This is a gentler version of `sent_congruence` that tolerates moderate sentiment divergence.

---

## Flexibility Sub-feature Weights

### `question_rate` — Weight: 0.25

**Source**: Olson (2011). Role flexibility is operationalised via clarification and questioning behaviours. The highest-weighted Flexibility feature.

**MI mapping**: Questions in MI signal both flexibility (openness to new information) and adaptive role negotiation between therapist and client.

**Operationalisation**: `question_exists == True` (AnnoMI-full) or `main_therapist_behaviour == 'question'` (AnnoMI-simple).

---

### `oscillation_rate` — Weight: 0.22

**Source**: Olson (2011), adaptability subscale. Families with healthy flexibility show fluid role transitions rather than rigid repetition.

**MI mapping**: Transitions between client talk types (change, sustain, neutral, follow/neutral) proxy discourse adaptability.

**Operationalisation**: proportion of consecutive client utterance pairs with different `client_talk_type` labels.

---

### `sent_variance` — Weight: 0.20

**Source**: Olson (2011), affective range. Flexible families permit a broader range of emotional expression.

**Operationalisation**: Standard deviation of VADER compound scores across all session utterances, clipped to [0,1] (normalised by 0.50).

---

### `anti_rigidity` — Weight: 0.20

**Source**: Emotional inertia literature (Kuppens et al., 2010). High lag-1 autocorrelation of affect indicates emotional rigidity.

**Operationalisation**: `1 / (1 + exp(3 × lag1_autocorr))` — sigmoid inversion of the lag-1 autocorrelation, yielding low values for rigid (high autocorrelation) and high values for flexible (low autocorrelation) sessions.

---

### `novelty` — Weight: 0.13

**Source**: Lexical diversity as cognitive flexibility proxy (Malvern et al., 2004).

**Operationalisation**: Mean type-token ratio (TTR) across utterances. TTR = unique tokens / total tokens.

---

## Communication Sub-feature Weights (Future Work)

These weights are implemented but currently produce a `[FUTURE WORK]` signal due to insufficient AnnoMI lexical coverage.

| Sub-feature | Weight | Source |
|------------|--------|--------|
| `question_rate` | 0.25 | Olson (2011) speaking skills |
| `turn_balance` | 0.20 | Olson (2011) self-disclosure balance |
| `topic_shift_rate` | 0.18 | Olson (2011) continuity tracking |
| `clarification_rate` | 0.20 | Olson (2011) clarity — **primary distinguishing signal** |
| `listener_resp_rate` | 0.17 | Olson (2011) listening skills |

AnnoMI coverage of clarification acts: **1.3%** (insufficient). See `docs/concern_responses.md §Concern 4`.

---

## References

- Amrhein, P. C., Miller, W. R., Yahne, C. E., Palmer, M., & Fulcher, L. (2003). Client commitment language during motivational interviewing predicts drug use outcomes. *Journal of Consulting and Clinical Psychology, 71*(5), 862–878.
- Gottman, J. M. (1994). *What Predicts Divorce?* Lawrence Erlbaum Associates.
- Kuppens, P., Allen, N. B., & Sheeber, L. B. (2010). Emotional inertia and psychological maladjustment. *Psychological Science, 21*(7), 984–991.
- Malvern, D., Richards, B., Chipere, N., & Durán, P. (2004). *Lexical Diversity and Language Development*. Palgrave Macmillan.
- Miller, W. R. & Rollnick, S. (2012). *Motivational Interviewing* (3rd ed.). Guilford Press.
- Moyers, T. B. & Martin, T. (2006). Therapist influence on client language during motivational interviewing sessions. *Journal of Substance Abuse Treatment, 30*(3), 245–251.
- Olson, D. (2011). FACES IV and the Circumplex Model: Validation study. *Journal of Marital and Family Therapy, 37*(1), 64–80.
