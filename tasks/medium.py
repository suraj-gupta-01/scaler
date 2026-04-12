"""
Task 2: Resource-Constrained Triage  (Medium)
=============================================

Objective
---------
Maximise incident resolution quality under a hard per-step investigation
budget (K = 3 INVESTIGATE actions per step).  The agent must learn to
prioritise which alerts deserve expensive investigations versus which can be
cheaply ignored or escalated.

Difficulty contract
-------------------
  - max_investigations = 3 per step (enforced by env.py)
  - Alerts are still mostly independent (20 % chance of correlated chain)
  - False-positive rate elevated vs easy task
  - Agent must balance breadth (handle many alerts) vs depth (investigate few)

Grading formula  (all inputs are deterministic given the same seed)
-------------------------------------------------------------------
    resolved_score  = Σ  severity_weight(alert) · resolution_quality(action)
                       for all handled alerts

    fp_penalty      = 0.30 · (unnecessary_investigations / total_investigations)
                      — fraction of budget spent on false positives

    critical_miss_penalty = 0.20 · min(critical_missed / max(critical_total, 1), 1.0)

    raw = resolved_score / max_possible_score              ∈ [0, 1]
    score = max(0.0, raw − fp_penalty − critical_miss_penalty)  ∈ [0.0, 1.0]

Resolution quality per action
------------------------------
    INVESTIGATE on critical/high (≥ 0.75)     → 1.00 × severity
    INVESTIGATE on medium       (0.50–0.74)   → 0.80 × severity
    INVESTIGATE on false-pos    (< 0.30)      → 0.00  + marks unnecessary
    ESCALATE    on critical/high(≥ 0.70)      → 0.85 × severity
    ESCALATE    on medium                     → 0.60 × severity
    ESCALATE    on false-pos                  → 0.00  + marks unnecessary
    IGNORE      on false-pos    (< 0.30)      → 0.15 × (1 − severity)  (filtering bonus)
    IGNORE      on medium                     → 0.00
    IGNORE      on critical                   → 0.00 + critical_missed++
    DELAY       on any                        → 0.00 (deferral, no score now)

Success threshold: ≥ 0.55

Integration with env.py
-----------------------
Consume info["processed_alerts"] inside the step loop:

    obs, reward, done, info = env.step(action)
    for alert_data in info.get("processed_alerts", []):
        grader.process_step(alert_data, info)
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from adaptive_alert_triage.models import Action, Alert, Reward

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_CRITICAL_THRESHOLD: float       = 0.75
_HIGH_THRESHOLD: float           = 0.70
_MEDIUM_LOWER: float             = 0.50
_FALSE_POSITIVE_THRESHOLD: float = 0.30

# Grading weights
_FP_PENALTY_WEIGHT: float            = 0.30
_CRITICAL_MISS_PENALTY_WEIGHT: float = 0.20

# Filtering-bonus cap so ignoring FPs never inflates score above 1.0
_FP_BONUS_CAP_PER_ALERT: float = 0.15

SUCCESS_THRESHOLD: float = 0.549


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------

class MediumTaskGrader:
    """
    Grader for Task 2: Resource-Constrained Triage.

    Lifecycle (one episode)
    -----------------------
    1. Instantiate once per episode.
    2. After every env.step(action), iterate info["processed_alerts"] and
       call process_step(alert_data, info) for each entry.
    3. At episode end call get_episode_score() → float in [0.0, 1.0].
    4. Optionally call get_metrics() for a full breakdown.
    5. Call reset() to reuse for a new episode.

    The score is deterministic: same seed + same policy → same score.
    """

    def __init__(self, max_investigations_per_step: int = 3) -> None:
        self._K = max_investigations_per_step

        # Accumulators
        self._resolved_score: float      = 0.0   # weighted resolution quality
        self._max_possible_score: float  = 0.0   # theoretical max if all handled optimally
        self._total_investigations: int  = 0
        self._unnecessary_invest: int    = 0      # INVESTIGATE on FP or low severity
        self._critical_total: int        = 0
        self._critical_missed: int       = 0
        self._total_actions: int         = 0

        self._action_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    def process_step(
        self,
        alert_data: Dict[str, Any],
        info: Dict[str, Any],  # noqa: ARG002
    ) -> float:
        """
        Evaluate one action using ground-truth data from env.step().

        Args:
            alert_data: One entry from info["processed_alerts"].
            info:       Full info dict (unused here, kept for API symmetry).

        Returns:
            Raw score contribution for this action (not normalised).
        """
        self._total_actions += 1

        true_sev:    float = float(alert_data.get("true_severity", 0.0))
        action_type: str   = str(alert_data.get("action_taken", ""))
        is_fp:       bool  = bool(alert_data.get("is_false_positive",
                                  true_sev < _FALSE_POSITIVE_THRESHOLD))

        # The theoretical max contribution for this alert (investigating optimally)
        optimal = self._optimal_contribution(true_sev)
        self._max_possible_score += optimal

        if true_sev >= _CRITICAL_THRESHOLD:
            self._critical_total += 1

        contribution = self._contribution(action_type, true_sev, is_fp)
        self._resolved_score += contribution

        if action_type == "INVESTIGATE":
            self._total_investigations += 1
            if is_fp or true_sev < _FALSE_POSITIVE_THRESHOLD:
                self._unnecessary_invest += 1

        if action_type == "IGNORE" and true_sev >= _CRITICAL_THRESHOLD:
            self._critical_missed += 1

        self._action_history.append({
            "alert_id":      alert_data.get("alert_id", ""),
            "action":        action_type,
            "true_severity": true_sev,
            "is_fp":         is_fp,
            "contribution":  contribution,
            "optimal":       optimal,
        })

        return contribution

    # ------------------------------------------------------------------
    # Legacy API
    # ------------------------------------------------------------------

    def grade_action(self, action: Action, alert: Alert, reward: Reward) -> float:
        """Grade a single action-alert pair (legacy / unit-test API)."""
        alert_data = {
            "alert_id":         alert.id,
            "true_severity":    alert.true_severity,
            "visible_severity": alert.visible_severity,
            "confidence":       alert.confidence,
            "alert_type":       alert.alert_type,
            "age":              alert.age,
            "action_taken":     action.action_type,
            "is_false_positive": alert.true_severity < _FALSE_POSITIVE_THRESHOLD,
        }
        return self.process_step(alert_data, {})

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def get_episode_score(self) -> float:
        """
        Return final normalised score strictly in (0, 1) — never 0.0 or 1.0.
        """
        if self._max_possible_score <= 0.0:
            return 0.5

        raw = min(self._resolved_score / self._max_possible_score, 1.0)

        if self._total_investigations > 0:
            fp_rate = self._unnecessary_invest / self._total_investigations
        else:
            fp_rate = 0.0
        fp_penalty = _FP_PENALTY_WEIGHT * fp_rate

        if self._critical_total > 0:
            miss_rate = min(self._critical_missed / self._critical_total, 1.0)
        else:
            miss_rate = 0.0
        miss_penalty = _CRITICAL_MISS_PENALTY_WEIGHT * miss_rate

        base_score = max(0.0, raw - fp_penalty - miss_penalty)
        rbase=round(base_score,2)
        if rbase == 0.00:
            return 0.01
        if rbase == 1.00:
            return 0.99
        return float(rbase)


    def passed(self) -> bool:
        """Return True if the agent meets the medium-task success threshold."""
        return self.get_episode_score() >= SUCCESS_THRESHOLD

    def calculate_resource_efficiency(self) -> float:
        """
        Fraction of INVESTIGATE + ESCALATE actions that were productive.

        Productive = action on an alert with true_severity ≥ 0.50.
        Returns 1.0 when no costly actions were taken (or 1.0 for perfect efficiency).
        """
        costly = [h for h in self._action_history
                  if h["action"] in ("INVESTIGATE", "ESCALATE")]
        if not costly:
            return 1.0
        productive = sum(1 for h in costly if h["true_severity"] >= _MEDIUM_LOWER)
        raw = productive / len(costly)
        return raw

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """Return a full breakdown of episode performance."""
        score = self.get_episode_score()
        eff   = self.calculate_resource_efficiency()

        breakdown: Dict[str, int] = {
            "INVESTIGATE": 0, "IGNORE": 0, "ESCALATE": 0, "DELAY": 0,
        }
        for h in self._action_history:
            breakdown[h["action"]] = breakdown.get(h["action"], 0) + 1

        return {
            "overall_score":          score,
            "passed":                 self.passed(),
            "success_threshold":      SUCCESS_THRESHOLD,
            "resolved_score":         self._resolved_score,
            "max_possible_score":     self._max_possible_score,
            "normalised_resolved":    (self._resolved_score / self._max_possible_score
                                       if self._max_possible_score > 0 else 0.0),
            "resource_efficiency":    eff,
            "total_investigations":   self._total_investigations,
            "unnecessary_invest":     self._unnecessary_invest,
            "critical_total":         self._critical_total,
            "critical_missed":        self._critical_missed,
            "total_actions":          self._total_actions,
            "action_breakdown":       breakdown,
        }

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all state for a new episode."""
        self._resolved_score       = 0.0
        self._max_possible_score   = 0.0
        self._total_investigations = 0
        self._unnecessary_invest   = 0
        self._critical_total       = 0
        self._critical_missed      = 0
        self._total_actions        = 0
        self._action_history       = []

    def __repr__(self) -> str:
        score = self.get_episode_score()
        eff   = self.calculate_resource_efficiency()
        return (
            f"MediumTaskGrader(score={score:.3f}, "
            f"efficiency={eff:.3f}, "
            f"investigations={self._total_investigations}, "
            f"passed={self.passed()})"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _optimal_contribution(true_sev: float) -> float:
        """What's the best possible contribution for this alert?"""
        if true_sev >= _CRITICAL_THRESHOLD:
            return 1.00 * true_sev
        if true_sev >= _MEDIUM_LOWER:
            return 0.80 * true_sev
        if true_sev < _FALSE_POSITIVE_THRESHOLD:
            return _FP_BONUS_CAP_PER_ALERT * (1.0 - true_sev)
        # Low-medium: best action is still INVESTIGATE
        return 0.80 * true_sev

    @staticmethod
    def _contribution(action_type: str, true_sev: float, is_fp: bool) -> float:
        """
        Deterministic contribution for one action.

        Returns a non-normalised float; caller accumulates into
        _resolved_score and later normalises by _max_possible_score.
        """
        if action_type == "INVESTIGATE":
            if is_fp or true_sev < _FALSE_POSITIVE_THRESHOLD:
                return 0.0          # budget wasted; penalty applied separately
            if true_sev >= _CRITICAL_THRESHOLD:
                return 1.00 * true_sev
            if true_sev >= _MEDIUM_LOWER:
                return 0.80 * true_sev
            # Low-medium investigation is barely useful
            return 0.40 * true_sev

        if action_type == "ESCALATE":
            if is_fp or true_sev < _FALSE_POSITIVE_THRESHOLD:
                return 0.0
            if true_sev >= _HIGH_THRESHOLD:
                return 0.85 * true_sev
            if true_sev >= _MEDIUM_LOWER:
                return 0.60 * true_sev
            return 0.30 * true_sev

        if action_type == "IGNORE":
            if is_fp or true_sev < _FALSE_POSITIVE_THRESHOLD:
                # Efficient noise filtering — small bonus
                return _FP_BONUS_CAP_PER_ALERT * (1.0 - true_sev)
            # Ignoring a non-FP alert gives zero (or negative for criticals,
            # tracked separately via critical_missed)
            return 0.0

        # DELAY — deferred, no score contribution this step
        return 0.0


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def run_episode_evaluation(
    agent,
    env,
    num_episodes: int = 10,
    seed_offset: int = 0,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run multiple episodes and return aggregated grading results.

    Args:
        agent:        Agent with .act(observation) -> Action method.
        env:          AdaptiveAlertTriageEnv(task_id="medium") instance.
        num_episodes: Number of episodes to run.
        seed_offset:  Added to episode index for the reset seed.
        verbose:      Print per-episode summary when True.

    Returns:
        Dict with keys: mean_score, std_score, min_score, max_score,
        success_rate, mean_efficiency, episode_scores, episode_metrics.
    """
    episode_scores:  List[float]          = []
    episode_metrics: List[Dict[str, Any]] = []

    for ep in range(num_episodes):
        grader = MediumTaskGrader(max_investigations_per_step=3)
        obs    = env.reset(seed=seed_offset + ep)
        done   = False

        while not done:
            if not obs.alerts:
                break

            action = agent.act(obs)
            obs, _reward, done, info = env.step(action)

            for alert_data in info.get("processed_alerts", []):
                grader.process_step(alert_data, info)

        score   = grader.get_episode_score()
        metrics = grader.get_metrics()
        episode_scores.append(score)
        episode_metrics.append(metrics)

        if verbose:
            print(
                f"  ep {ep + 1:02d}  score={score:.3f}  "
                f"eff={metrics['resource_efficiency']:.3f}  "
                f"invest={metrics['total_investigations']}  "
                f"passed={metrics['passed']}"
            )

    scores_arr = np.array(episode_scores)
    eff_arr    = np.array([m["resource_efficiency"] for m in episode_metrics])
    return {
        "mean_score":       float(scores_arr.mean()),
        "std_score":        float(scores_arr.std()),
        "min_score":        float(scores_arr.min()),
        "max_score":        float(scores_arr.max()),
        "success_rate":     float((scores_arr >= SUCCESS_THRESHOLD).mean()),
        "mean_efficiency":  float(eff_arr.mean()),
        "episode_scores":   episode_scores,
        "episode_metrics":  episode_metrics,
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("MediumTaskGrader — self-test\n" + "=" * 55)

    from adaptive_alert_triage.models import Alert, Action, Reward

    def _alert(aid: str, true_sev: float, is_fp: bool = False) -> Alert:
        return Alert(
            id=aid, visible_severity=0.6, confidence=0.85,
            alert_type="CPU", age=1, true_severity=true_sev,
            metadata={"false_positive": is_fp},
        )

    grader = MediumTaskGrader()
    cases = [
        ("Critical + INVESTIGATE (best)",        "INVESTIGATE", 0.90, False),
        ("High     + ESCALATE",                  "ESCALATE",    0.80, False),
        ("Medium   + INVESTIGATE",               "INVESTIGATE", 0.60, False),
        ("FP       + IGNORE (efficient)",        "IGNORE",      0.15, True),
        ("FP       + INVESTIGATE (wasteful)",    "INVESTIGATE", 0.15, True),
        ("Critical + IGNORE (miss)",             "IGNORE",      0.90, False),
        ("Medium   + DELAY",                     "DELAY",       0.60, False),
    ]

    all_pass = True
    for desc, act, sev, is_fp in cases:
        alert  = _alert("ax", sev, is_fp)
        action = Action(alert_id="ax", action_type=act)
        contrib = grader.grade_action(action, alert, Reward(value=0.0))
        print(f"  {desc:45s}  contrib={contrib:+.4f}")

    score = grader.get_episode_score()
    m     = grader.get_metrics()
    print(f"\nEpisode score      : {score:.4f}")
    print(f"Passed             : {m['passed']}")
    print(f"Resource efficiency: {m['resource_efficiency']:.4f}")
    print(f"Critical missed    : {m['critical_missed']}/{m['critical_total']}")
    print(f"Unnecessary invest : {m['unnecessary_invest']}/{m['total_investigations']}")
    print(f"Action breakdown   : {m['action_breakdown']}")