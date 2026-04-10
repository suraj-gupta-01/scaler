"""
Task 1: Basic Alert Prioritisation  (Easy)
==========================================

Objective
---------
Learn to correctly classify and respond to individual, independent alerts.
There are no resource constraints and no alert correlations — the agent simply
needs to read each alert's visible attributes and choose the right action.

Difficulty contract
-------------------
  - No resource budget (unlimited INVESTIGATE per step)
  - Alerts are independent (no correlated chains)
  - Correct policy is purely threshold-based on visible severity

Grading formula
---------------
    score = correct_actions / total_actions          ∈ [0.0, 1.0]

Correctness rules (deterministic)
----------------------------------
    true_severity ≥ 0.75  → INVESTIGATE or ESCALATE
    true_severity < 0.30  → IGNORE
    0.30 ≤ severity < 0.75 → INVESTIGATE (always OK)
                             ESCALATE    (OK if severity ≥ 0.60)
                             IGNORE      (OK if severity < 0.50)
                             DELAY       → always wrong in the easy task

Success threshold: ≥ 0.70 (70 % correct action rate)

Integration with env.py
-----------------------
Every call to env.step(action) returns an ``info`` dict containing:

    info["processed_alerts"]  — list of dicts, one per action this step
        Keys: alert_id, true_severity, visible_severity, confidence,
              alert_type, age, is_correlated, is_false_positive,
              action_taken, correlation_group_index

The grader consumes those dicts via process_step(); this guarantees that
ground-truth fields are used even after the alert has been removed from the
environment queue.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from adaptive_alert_triage.models import Action, Alert, Reward

# ---------------------------------------------------------------------------
# Severity band boundaries  (kept in sync with utils.py constants)
# ---------------------------------------------------------------------------
_CRITICAL_THRESHOLD: float       = 0.75
_FALSE_POSITIVE_THRESHOLD: float = 0.30
_MEDIUM_ESCALATE_MIN: float      = 0.60   # ESCALATE acceptable above this
_MEDIUM_IGNORE_MAX: float        = 0.50   # IGNORE acceptable below this

# Pass threshold
SUCCESS_THRESHOLD: float = 0.696


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------

class EasyTaskGrader:
    """
    Grader for Task 1: Basic Alert Prioritisation.

    Lifecycle (one episode)
    -----------------------
    1. Instantiate once per episode.
    2. After every env.step(action), iterate info["processed_alerts"] and
       call process_step(alert_data, info) for each entry.
    3. At episode end call get_episode_score() → float in [0.0, 1.0].
    4. Optionally call get_metrics() for a full breakdown.
    5. Call reset() to reuse the grader for a new episode.

    Scoring is fully deterministic: same alert + same action → same score.
    """

    def __init__(self) -> None:
        self.correct_actions: int = 0
        self.total_actions: int = 0
        self.action_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Primary interface  (production)
    # ------------------------------------------------------------------

    def process_step(
        self,
        alert_data: Dict[str, Any],
        info: Dict[str, Any],  # noqa: ARG002  (kept for API symmetry)
    ) -> float:
        """
        Evaluate one action using ground-truth data from env.step().

        Args:
            alert_data: One entry from info["processed_alerts"].
                        Must contain: true_severity, action_taken.
            info:       Full info dict from env.step() (unused here but
                        kept for consistent API across all three graders).

        Returns:
            1.0 if the action was correct, 0.0 otherwise.
        """
        self.total_actions += 1

        true_severity: float = float(alert_data.get("true_severity", 0.0))
        action_type: str     = str(alert_data.get("action_taken", ""))
        is_correct: bool     = self._is_action_correct(action_type, true_severity)

        if is_correct:
            self.correct_actions += 1

        self.action_history.append({
            "alert_id":        alert_data.get("alert_id", ""),
            "action":          action_type,
            "true_severity":   true_severity,
            "visible_severity":alert_data.get("visible_severity", 0.0),
            "confidence":      alert_data.get("confidence", 0.0),
            "alert_type":      alert_data.get("alert_type", ""),
            "is_false_positive":alert_data.get("is_false_positive", False),
            "correct":         is_correct,
            "score":           1.0 if is_correct else 0.0,
        })

        return 1.0 if is_correct else 0.0

    # ------------------------------------------------------------------
    # Legacy API  (unit tests / backward compat)
    # ------------------------------------------------------------------

    def grade_action(self, action: Action, alert: Alert, reward: Reward) -> float:
        """
        Grade a single action-alert pair (legacy / unit-test API).

        Prefer process_step() in production — this wrapper exists only for
        backward compatibility with existing unit tests.
        """
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
        if self.total_actions == 0:
            return 0.5

        raw = self.correct_actions / self.total_actions
        # Map [0,1] -> (0,1) with a small epsilon margin, no rounding
        score = 0.001 + 0.998 * float(raw)
        return max(0.001, min(0.999, score))


    def passed(self) -> bool:
        """Return True if the agent meets the easy-task success threshold."""
        return self.get_episode_score() >= SUCCESS_THRESHOLD

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """
        Return a detailed per-band accuracy breakdown.

        Severity bands:
            critical       true_severity ≥ 0.75
            medium         0.30 ≤ true_severity < 0.75
            false_positive true_severity < 0.30
        """
        score = self.get_episode_score()

        critical_h = [h for h in self.action_history if h["true_severity"] >= _CRITICAL_THRESHOLD]
        medium_h   = [h for h in self.action_history
                      if _FALSE_POSITIVE_THRESHOLD <= h["true_severity"] < _CRITICAL_THRESHOLD]
        fp_h       = [h for h in self.action_history if h["true_severity"] < _FALSE_POSITIVE_THRESHOLD]

        def _acc(subset: list) -> float:
            if not subset:
                return 0.0
            return sum(h["correct"] for h in subset) / len(subset)

        breakdown: Dict[str, int] = {
            "INVESTIGATE": 0, "IGNORE": 0, "ESCALATE": 0, "DELAY": 0,
        }
        for h in self.action_history:
            breakdown[h["action"]] = breakdown.get(h["action"], 0) + 1

        return {
            "overall_score":           score,
            "passed":                  self.passed(),
            "success_threshold":       SUCCESS_THRESHOLD,
            "correct_actions":         self.correct_actions,
            "total_actions":           self.total_actions,
            "critical_accuracy":       _acc(critical_h),
            "medium_accuracy":         _acc(medium_h),
            "false_positive_accuracy": _acc(fp_h),
            "critical_count":          len(critical_h),
            "medium_count":            len(medium_h),
            "false_positive_count":    len(fp_h),
            "action_breakdown":        breakdown,
        }

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all state for a new episode."""
        self.correct_actions = 0
        self.total_actions   = 0
        self.action_history  = []

    def __repr__(self) -> str:
        score = self.get_episode_score()
        return (
            f"EasyTaskGrader(score={score:.3f}, "
            f"correct={self.correct_actions}/{self.total_actions}, "
            f"passed={self.passed()})"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_action_correct(action_type: str, true_severity: float) -> bool:
        """
        Deterministic correctness rule for Task 1.

        Critical (≥ 0.75)   → INVESTIGATE or ESCALATE
        False positive (< 0.30) → IGNORE only
        Medium (0.30–0.74)  → INVESTIGATE always OK
                              ESCALATE OK if severity ≥ 0.60
                              IGNORE OK if severity < 0.50
                              DELAY never OK (creates unnecessary backlog)
        """
        if true_severity >= _CRITICAL_THRESHOLD:
            return action_type in ("INVESTIGATE", "ESCALATE")

        if true_severity < _FALSE_POSITIVE_THRESHOLD:
            return action_type == "IGNORE"

        # Medium band
        if action_type == "INVESTIGATE":
            return True
        if action_type == "ESCALATE":
            return true_severity >= _MEDIUM_ESCALATE_MIN
        if action_type == "IGNORE":
            return true_severity < _MEDIUM_IGNORE_MAX
        # DELAY
        return False


# ---------------------------------------------------------------------------
# Evaluation helper (used by evaluation/evaluate.py)
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
        env:          AdaptiveAlertTriageEnv(task_id="easy") instance.
        num_episodes: Number of episodes to run.
        seed_offset:  Added to episode index to produce the reset seed.
        verbose:      Print per-episode summary when True.

    Returns:
        Dict with keys: mean_score, std_score, min_score, max_score,
        success_rate, episode_scores, episode_metrics.
    """
    episode_scores:  List[float]         = []
    episode_metrics: List[Dict[str, Any]] = []

    for ep in range(num_episodes):
        grader = EasyTaskGrader()
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
                f"correct={metrics['correct_actions']}/{metrics['total_actions']}  "
                f"passed={metrics['passed']}"
            )

    scores_arr = np.array(episode_scores)
    return {
        "mean_score":    float(scores_arr.mean()),
        "std_score":     float(scores_arr.std()),
        "min_score":     float(scores_arr.min()),
        "max_score":     float(scores_arr.max()),
        "success_rate":  float((scores_arr >= SUCCESS_THRESHOLD).mean()),
        "episode_scores":  episode_scores,
        "episode_metrics": episode_metrics,
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("EasyTaskGrader — self-test\n" + "=" * 50)

    from adaptive_alert_triage.models import Alert, Action, Reward

    def _alert(aid: str, true_sev: float, vis_sev: float = 0.5) -> Alert:
        return Alert(
            id=aid, visible_severity=vis_sev, confidence=0.9,
            alert_type="CPU", age=1, true_severity=true_sev,
        )

    cases = [
        # (desc, action, true_sev, expected_score)
        ("Critical + INVESTIGATE",        "INVESTIGATE", 0.90, 1.0),
        ("Critical + ESCALATE",           "ESCALATE",    0.90, 1.0),
        ("Critical + IGNORE  (wrong)",    "IGNORE",      0.90, 0.0),
        ("Critical + DELAY   (wrong)",    "DELAY",       0.90, 0.0),
        ("FP      + IGNORE",              "IGNORE",      0.10, 1.0),
        ("FP      + INVESTIGATE (wrong)", "INVESTIGATE", 0.10, 0.0),
        ("Medium  + INVESTIGATE",         "INVESTIGATE", 0.55, 1.0),
        ("Medium  + ESCALATE hi (ok)",    "ESCALATE",    0.65, 1.0),
        ("Medium  + ESCALATE lo (wrong)", "ESCALATE",    0.45, 0.0),
        ("Medium  + IGNORE lo (ok)",      "IGNORE",      0.40, 1.0),
        ("Medium  + IGNORE hi (wrong)",   "IGNORE",      0.60, 0.0),
        ("Medium  + DELAY  (wrong)",      "DELAY",       0.55, 0.0),
    ]

    grader = EasyTaskGrader()
    all_pass = True
    for desc, act, sev, expected in cases:
        alert = _alert("a1", sev)
        action = Action(alert_id="a1", action_type=act)
        result = grader.grade_action(action, alert, Reward(value=0.0))
        ok = result == expected
        if not ok:
            all_pass = False
        print(f"  [{'PASS' if ok else 'FAIL'}]  {desc}")
        if not ok:
            print(f"         got {result}, expected {expected}")

    final = grader.get_episode_score()
    print(f"\nEpisode score : {final:.3f}")
    print(f"Passed        : {grader.passed()}")
    m = grader.get_metrics()
    print(f"Critical acc  : {m['critical_accuracy']:.3f}")
    print(f"Medium acc    : {m['medium_accuracy']:.3f}")
    print(f"FP acc        : {m['false_positive_accuracy']:.3f}")
    print("\nAll tests passed!" if all_pass else "\nSome FAILED — check above.")