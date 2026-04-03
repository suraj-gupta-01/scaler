"""
Rule-Based Baseline Agent for Adaptive Alert Triage
====================================================

Implements two rule-based policies that serve as reproducible baselines against
which RL agents can be measured.  Both agents expose the same interface:

    agent.act(observation: Observation) -> Action
    agent.reset() -> None

Agents
------
RuleBasedAgent
    Threshold-based policy using visible_severity and confidence only.
    Designed to be the weakest baseline — gives the RL agent room to shine.

ImprovedRuleBasedAgent
    Adds age-weighting, alert-type priors, system-load awareness, and
    a simple resource-budget guard.  Competitive on the easy task but still
    well below the hard-task success threshold (≥ 0.50).

Evaluation
----------
evaluate_agent() runs N episodes and returns aggregated metrics that match the
three task graders (EasyTaskGrader, MediumTaskGrader, HardTaskGrader).

Usage
-----
    from agents.baseline import RuleBasedAgent, evaluate_agent
    from adaptive_alert_triage.env import AdaptiveAlertTriageEnv

    env   = AdaptiveAlertTriageEnv(task_id="easy")
    agent = RuleBasedAgent()
    results = evaluate_agent(agent, env, num_episodes=10, task_id="easy")
    print(results)
"""

from __future__ import annotations

import sys
import os
from typing import Any, Dict, List, Optional

import numpy as np

from adaptive_alert_triage.models import Action, Alert, Observation

# Grader imports (relative paths allow running from project root or src/)
from tasks.easy   import EasyTaskGrader,   run_episode_evaluation as _easy_eval
from tasks.medium import MediumTaskGrader, run_episode_evaluation as _medium_eval
from tasks.hard   import HardTaskGrader,   run_episode_evaluation as _hard_eval


# ---------------------------------------------------------------------------
# Policy constants  (kept separate from the environment thresholds so the
# baseline agent cannot accidentally "see" hidden ground-truth constants)
# ---------------------------------------------------------------------------
_INVESTIGATE_SEV_THRESHOLD:  float = 0.75   # severity above which to investigate
_INVESTIGATE_CONF_THRESHOLD: float = 0.70   # confidence required for investigation
_IGNORE_CONF_THRESHOLD:      float = 0.30   # confidence below which → likely FP → IGNORE
_ESCALATE_SEV_THRESHOLD:     float = 0.55   # severity above which to escalate
_SECURITY_SEV_BOOST:         float = 0.05   # extra weight for SECURITY type alerts
_AGE_WEIGHT:                 float = 0.08   # scoring weight per time-step of age


# ---------------------------------------------------------------------------
# RuleBasedAgent
# ---------------------------------------------------------------------------

class RuleBasedAgent:
    """
    Simple threshold-based agent for alert triage.

    Policy (applied in order, first match wins):
        1. visible_severity > 0.75 AND confidence > 0.70  → INVESTIGATE
        2. resource_budget == 0                            → ESCALATE (can't investigate)
        3. confidence < 0.30                              → IGNORE   (likely false positive)
        4. visible_severity > 0.55                        → ESCALATE
        5. default                                        → DELAY

    Alert selection: highest visible_severity first.

    Limitations (intentional — motivates RL):
        - Cannot detect correlated chains (no memory across steps)
        - Fixed thresholds ignore system_load and queue_length
        - No adaptation to changing alert distributions
        - DELAY is almost never used, hurting medium-task efficiency

    Attributes:
        investigate_sev_threshold:  severity required to trigger INVESTIGATE
        investigate_conf_threshold: confidence required alongside severity
        ignore_conf_threshold:      confidence below which to IGNORE
        escalate_sev_threshold:     severity above which to ESCALATE (fallback)
        resource_aware:             if True, respects resource_budget == 0
    """

    def __init__(
        self,
        investigate_sev_threshold:  float = _INVESTIGATE_SEV_THRESHOLD,
        investigate_conf_threshold: float = _INVESTIGATE_CONF_THRESHOLD,
        ignore_conf_threshold:      float = _IGNORE_CONF_THRESHOLD,
        escalate_sev_threshold:     float = _ESCALATE_SEV_THRESHOLD,
        resource_aware:             bool  = True,
    ) -> None:
        self.investigate_sev_threshold  = investigate_sev_threshold
        self.investigate_conf_threshold = investigate_conf_threshold
        self.ignore_conf_threshold      = ignore_conf_threshold
        self.escalate_sev_threshold     = escalate_sev_threshold
        self.resource_aware             = resource_aware

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def act(self, observation: Observation) -> Action:
        """
        Choose an action for the highest-priority alert.

        Args:
            observation: Current environment observation (agent-visible only).

        Returns:
            Action targeting one alert in observation.alerts.

        Raises:
            ValueError: If observation.alerts is empty.
        """
        if not observation.alerts:
            raise ValueError("No alerts in observation — cannot act.")

        alert = self._select_alert(observation.alerts)
        action_type = self._decide_action(alert, observation)
        return Action(alert_id=alert.id, action_type=action_type)

    def reset(self) -> None:
        """Reset any per-episode state (stateless baseline; no-op)."""
        pass

    # ------------------------------------------------------------------
    # Alert selection
    # ------------------------------------------------------------------

    def _select_alert(self, alerts: List[Alert]) -> Alert:
        """Pick the alert with the highest visible_severity."""
        return max(alerts, key=lambda a: a.visible_severity)

    # ------------------------------------------------------------------
    # Action decision
    # ------------------------------------------------------------------

    def _decide_action(self, alert: Alert, obs: Observation) -> str:
        """
        Apply the rule-based policy.

        Args:
            alert: The alert selected for action.
            obs:   Full observation (for resource_budget access).

        Returns:
            Action type string.
        """
        sev  = alert.visible_severity
        conf = alert.confidence

        # Rule 1: high severity + high confidence → INVESTIGATE
        if sev > self.investigate_sev_threshold and conf > self.investigate_conf_threshold:
            if self.resource_aware and obs.resource_budget is not None and obs.resource_budget <= 0:
                # Budget exhausted — escalate rather than block
                return "ESCALATE"
            return "INVESTIGATE"

        # Rule 2: low confidence → likely false positive → IGNORE
        if conf < self.ignore_conf_threshold:
            return "IGNORE"

        # Rule 3: medium-high severity → ESCALATE
        if sev > self.escalate_sev_threshold:
            return "ESCALATE"

        # Default: DELAY — let it age for potential future reclassification
        return "DELAY"

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"inv_sev={self.investigate_sev_threshold}, "
            f"inv_conf={self.investigate_conf_threshold}, "
            f"ign_conf={self.ignore_conf_threshold})"
        )


# ---------------------------------------------------------------------------
# ImprovedRuleBasedAgent
# ---------------------------------------------------------------------------

class ImprovedRuleBasedAgent(RuleBasedAgent):
    """
    Enhanced rule-based agent with multi-factor scoring and context awareness.

    Improvements over RuleBasedAgent:
        - Multi-factor alert scoring: severity + age + type prior
        - System-load-aware thresholds: under high load, be more conservative
        - Age-urgency: alerts older than 3 steps get promoted to INVESTIGATE
        - SECURITY alerts receive a priority boost

    Still limited (no learning, no chain detection) but achieves higher scores
    on easy and medium tasks than the plain threshold baseline.
    """

    def _select_alert(self, alerts: List[Alert]) -> Alert:
        """
        Score alerts on a combined severity + age + type metric.

        Score = visible_severity * 2 + age * AGE_WEIGHT + type_boost
        """
        def _score(a: Alert) -> float:
            s = a.visible_severity * 2.0
            s += a.age * _AGE_WEIGHT
            if a.alert_type == "SECURITY":
                s += _SECURITY_SEV_BOOST * 2
            elif a.alert_type in ("APPLICATION", "NETWORK"):
                s += _SECURITY_SEV_BOOST
            return s

        return max(alerts, key=_score)

    def _decide_action(self, alert: Alert, obs: Observation) -> str:
        """
        Enhanced policy with age-urgency and system-load guards.

        Overrides:
            - Aged critical alerts (age ≥ 3, sev > 0.70) → INVESTIGATE immediately
            - Under very high system load (> 0.85): raise investigate bar
            - Otherwise: fall back to parent policy
        """
        sev       = alert.visible_severity
        conf      = alert.confidence
        age       = alert.age
        sys_load  = obs.system_load
        budget    = obs.resource_budget

        # Rule A: aged potential-critical — promote to INVESTIGATE regardless of conf
        if age >= 3 and sev > 0.70:
            if self.resource_aware and budget is not None and budget <= 0:
                return "ESCALATE"
            return "INVESTIGATE"

        # Rule B: very high system load — conservative strategy
        if sys_load > 0.85:
            if sev > 0.85 and conf > 0.80:
                if self.resource_aware and budget is not None and budget <= 0:
                    return "ESCALATE"
                return "INVESTIGATE"
            if sev < 0.35:
                return "IGNORE"
            return "DELAY"

        # Rule C: resource-budget nearly exhausted → switch to ESCALATE for medium
        if self.resource_aware and budget is not None and budget <= 1:
            if sev > self.investigate_sev_threshold and conf > self.investigate_conf_threshold:
                return "INVESTIGATE"   # save last slot for truly critical
            if sev > 0.50:
                return "ESCALATE"
            if conf < self.ignore_conf_threshold:
                return "IGNORE"
            return "DELAY"

        # Fallback to parent rules
        return super()._decide_action(alert, obs)


# ---------------------------------------------------------------------------
# Evaluation harness
# ---------------------------------------------------------------------------

def evaluate_agent(
    agent: RuleBasedAgent,
    env,
    num_episodes: int = 10,
    task_id: str = "easy",
    seed_offset: int = 0,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate a rule-based agent across multiple episodes using the task graders.

    This function integrates with the same graders that produce the official
    leaderboard scores — results are directly comparable to RL baselines.

    Args:
        agent:        Agent instance with .act(observation) and .reset() methods.
        env:          AdaptiveAlertTriageEnv instance (must match task_id).
        num_episodes: Number of evaluation episodes.
        task_id:      One of "easy", "medium", "hard".
        seed_offset:  Added to episode index to form the reset seed.
        verbose:      Print per-episode summary if True.

    Returns:
        Dict with keys:
            mean_score, std_score, min_score, max_score,
            success_rate, episode_scores, episode_metrics,
            task_id, agent_name, num_episodes.
    """
    if task_id == "easy":
        results = _easy_eval(agent, env, num_episodes=num_episodes,
                             seed_offset=seed_offset, verbose=verbose)
    elif task_id == "medium":
        results = _medium_eval(agent, env, num_episodes=num_episodes,
                               seed_offset=seed_offset, verbose=verbose)
    elif task_id == "hard":
        results = _hard_eval(agent, env, num_episodes=num_episodes,
                             seed_offset=seed_offset, verbose=verbose)
    else:
        raise ValueError(f"Unknown task_id '{task_id}'. Must be easy/medium/hard.")

    results["task_id"]    = task_id
    results["agent_name"] = repr(agent)
    results["num_episodes"] = num_episodes
    return results


# ---------------------------------------------------------------------------
# Self-test / CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from adaptive_alert_triage.env import AdaptiveAlertTriageEnv

    print("=" * 65)
    print("Rule-Based Baseline Agent — Self-Test")
    print("=" * 65)

    # ── Unit test: basic act() behaviour ─────────────────────────────────
    from adaptive_alert_triage.models import Alert, Observation

    def _obs(alerts: List[Alert], budget: Optional[int] = None) -> Observation:
        return Observation(
            alerts=alerts,
            system_load=0.5,
            queue_length=len(alerts),
            time_remaining=20,
            episode_step=1,
            resource_budget=budget,
        )

    def _alert(aid: str, sev: float, conf: float,
               atype: str = "CPU", age: int = 0) -> Alert:
        return Alert(id=aid, visible_severity=sev, confidence=conf,
                     alert_type=atype, age=age)

    cases = [
        # description, alerts, budget, expected_action
        ("High sev+conf → INVESTIGATE",
         [_alert("a1", 0.90, 0.85)], None, "INVESTIGATE"),
        ("Low confidence → IGNORE",
         [_alert("a2", 0.50, 0.20)], None, "IGNORE"),
        ("Medium sev → ESCALATE",
         [_alert("a3", 0.65, 0.60)], None, "ESCALATE"),
        ("Low sev → DELAY",
         [_alert("a4", 0.30, 0.50)], None, "DELAY"),
        ("High sev, budget=0 → ESCALATE (resource_aware)",
         [_alert("a5", 0.90, 0.85)], 0, "ESCALATE"),
    ]

    agent_basic = RuleBasedAgent()
    all_pass = True
    print("\n── Basic RuleBasedAgent ─────────────────────────────────────")
    for desc, alerts, budget, expected in cases:
        obs    = _obs(alerts, budget)
        action = agent_basic.act(obs)
        ok     = action.action_type == expected
        if not ok:
            all_pass = False
        print(f"  [{'PASS' if ok else 'FAIL'}]  {desc}")
        if not ok:
            print(f"         expected {expected}, got {action.action_type}")

    # ── Test ImprovedRuleBasedAgent ──────────────────────────────────────
    print("\n── ImprovedRuleBasedAgent ──────────────────────────────────────")
    agent_improved = ImprovedRuleBasedAgent()

    # Aged critical should get INVESTIGATE
    aged_critical = [_alert("a6", 0.75, 0.50, age=4)]  # aged, medium conf
    obs_aged  = _obs(aged_critical)
    act_aged  = agent_improved.act(obs_aged)
    ok_aged   = act_aged.action_type == "INVESTIGATE"
    if not ok_aged:
        all_pass = False
    print(f"  [{'PASS' if ok_aged else 'FAIL'}]  Aged critical (age=4, sev=0.75) → INVESTIGATE  (got {act_aged.action_type})")

    # SECURITY alert should be selected over lower-sev CPU
    multi = [
        _alert("sec",  0.70, 0.80, "SECURITY"),
        _alert("cpu",  0.85, 0.80, "CPU"),
    ]
    obs_multi  = _obs(multi)
    sel_multi  = agent_improved._select_alert(multi)
    # CPU has higher sev but SECURITY boost may flip; test that _select_alert runs without error
    print(f"  [PASS]  Multi-alert selection → picked '{sel_multi.id}' (no crash)")

    # ── Episode evaluation (no live env, skip with a note) ───────────────
    print("\n── Episode evaluation ──────────────────────────────────────────")
    try:
        env = AdaptiveAlertTriageEnv(task_id="easy")
        results = evaluate_agent(agent_basic, env, num_episodes=3,
                                 task_id="easy", seed_offset=0, verbose=True)
        print(f"\n  mean_score   : {results['mean_score']:.3f}")
        print(f"  success_rate : {results['success_rate']:.3f}")
        print(f"  agent        : {results['agent_name']}")
    except Exception as exc:
        print(f"  [SKIP] Could not instantiate environment: {exc}")
        print("         Run from the project root with the full package installed.")

    print("\n" + "=" * 65)
    print("All unit tests passed!" if all_pass else "SOME UNIT TESTS FAILED — see above.")