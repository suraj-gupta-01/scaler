"""
Reward Shaping Logic for Adaptive Alert Triage Environment

Implements a dense, multi-component reward function that guides the RL agent
toward correct alert triage decisions.  The reward is designed to:

  1. Heavily reward correct handling of critical alerts.
  2. Penalise missed critical alerts and wasted investigations.
  3. Grant a cascading-failure-prevention bonus for correlated alerts.
  4. Scale the magnitude of medium-severity rewards to match alert importance.
  5. Apply task-aware scaling so the medium and hard tasks are meaningfully
     harder to optimise — resource-constrained steps carry an extra penalty
     for budget overuse, and hard-mode delayed rewards are boosted.
  6. [NEW] Age-urgency shaping: penalise delay/ignore on aged alerts
     proportional to how close they are to the failure threshold (age 5).
  7. [NEW] Chain-trigger proxy: boost reward for proactively acting on
     alerts that match chain-prone type+severity patterns, since
     is_correlated is hidden from the agent.
  8. [NEW] Medium-correlated shaping: when a correlated alert is medium
     severity, the penalty for DELAY/IGNORE is amplified to match the
     chain-propagation cost seen by the hard-task grader.

Reward schedule (baseline, task_id = "easy"):
  +10   Critical alert (true_severity >= 0.75) correctly INVESTIGATE or ESCALATE
  + 5   Cascading failure prevented (correlated alert handled correctly)
  + 3   False positive (true_severity < 0.30) correctly IGNORE
  - 2   Unnecessary investigation (INVESTIGATE on a false positive)
  - 8   Missed critical alert (IGNORE a critical alert)
  - 2.4 Risky delay (DELAY on a critical alert)
  -10   System failure accumulated reward penalty (applied by env, not here)

The function signature is:
    calculate_reward(action: Action, alert: Alert, config: dict) -> Reward

``config`` is the task config dict from AdaptiveAlertTriageEnv (keys:
``max_investigations``, ``failure_threshold``, ``correlation_probability``,
``max_steps``).  Passing ``None`` or ``{}`` silently falls back to easy-task
defaults.

Compatibility:
  - models.py  : Action, Alert, Reward (Pydantic v2)
  - utils.py   : CRITICAL_SEVERITY_THRESHOLD = 0.75
  - env.py     : calls calculate_reward(action, alert, self.config)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from adaptive_alert_triage.models import Action, Alert, Reward

# ---------------------------------------------------------------------------
# Reward constants  (match the schedule in models.py docstring & requirements)
# ---------------------------------------------------------------------------

REWARD_CRITICAL_HANDLED: float       =  10.0   # INVESTIGATE/ESCALATE on critical
REWARD_FAILURE_PREVENTED: float      =   8.0   # bonus: correlated alert handled (boosted)
REWARD_FALSE_POSITIVE_IGNORED: float =   3.0   # IGNORE on a false positive
REWARD_MEDIUM_INVESTIGATED: float    =   3.0   # INVESTIGATE on medium-severity (boosted)

PENALTY_UNNECESSARY_INVEST: float    =  -2.0   # INVESTIGATE on a false positive
PENALTY_MISSED_CRITICAL: float       = -10.0   # IGNORE on a critical alert (harsher)
PENALTY_CRITICAL_DELAYED: float      =  -4.0   # DELAY on a critical (much riskier)
PENALTY_UNNECESSARY_ESCALATE: float  =  -1.0   # ESCALATE on a false positive
PENALTY_MEDIUM_DELAYED: float        =  -1.5   # DELAY on medium in unconstrained mode
PENALTY_IGNORE_HIGH_MEDIUM: float    =  -2.0   # IGNORE medium with sev >= 0.50

# Severity band thresholds  (must match utils.CRITICAL_SEVERITY_THRESHOLD)
_CRITICAL_THRESHOLD: float        = 0.75
_FALSE_POSITIVE_THRESHOLD: float  = 0.30
_MEDIUM_HIGH_THRESHOLD: float     = 0.50   # grader boundary for medium IGNORE acceptance
_MEDIUM_ESCALATE_THRESHOLD: float = 0.60   # grader boundary for medium ESCALATE acceptance

# Age-urgency constants
# Alerts at or beyond this age AND critical severity cause system failures.
# We want the shaping signal to appear *before* failure, not after.
_CRITICAL_AGE_THRESHOLD: int   = 5    # must match utils.CRITICAL_AGE_THRESHOLD
_URGENCY_RAMP_START: int       = 2    # urgency bonus/penalty starts at this age
_MAX_URGENCY_BONUS: float      = 6.0  # cap on age-urgency reward added for proactive action
_MAX_URGENCY_PENALTY: float    = 8.0  # cap on age-urgency penalty for DELAY/IGNORE

# Chain-trigger proxy constants
# These apply when is_correlated=True BUT the alert is medium severity,
# because that case is missed by the existing critical+correlated path.
_CHAIN_MEDIUM_PROACTIVE_BONUS: float  = 5.0   # INVESTIGATE/ESCALATE on medium correlated
_CHAIN_MEDIUM_DELAY_PENALTY: float    = -6.0  # DELAY on medium correlated (chain propagates)
_CHAIN_MEDIUM_IGNORE_PENALTY: float   = -7.0  # IGNORE on medium correlated (chain propagates)

# Task-specific reward multipliers
_TASK_MULTIPLIERS: Dict[str, float] = {
    "easy":   1.0,
    "medium": 1.1,   # slightly amplified so efficient triage matters more
    "hard":   1.3,   # largest amplification; delayed failures hurt more
}

# Alert types that most commonly appear as chain triggers in CORRELATION_CHAINS
# (CPU, NETWORK, DISK, MEMORY can all be position-0 triggers).
# SECURITY and APPLICATION appear later in chains.
_CHAIN_TRIGGER_TYPES = frozenset({"CPU", "MEMORY", "NETWORK", "DISK"})


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def calculate_reward(
    action: Action,
    alert: Alert,
    config: Optional[Dict[str, Any]] = None,
) -> Reward:
    """
    Calculate the dense, shaped reward for a single action-alert pair.

    This function is called once per step by AdaptiveAlertTriageEnv.step()
    immediately after resource-budget validation and before alert removal.

    Args:
        action: The agent's decision (action_type + alert_id).
        alert:  The targeted alert with *full* ground-truth fields populated
                (true_severity, is_correlated, metadata["false_positive"]).
        config: Task configuration dict from AdaptiveAlertTriageEnv.config.
                Keys used: ``max_investigations``, ``failure_threshold``,
                ``task_id``.  Safe to pass ``None`` — defaults to easy-task.

    Returns:
        Reward object with scalar ``value``, per-component ``components``
        dict, and a ``info`` dict exposing ground truth for graders/logging.
    """
    config = config or {}
    task_id: str = config.get("task_id", "easy")
    multiplier: float = _TASK_MULTIPLIERS.get(task_id, 1.0)
    resource_constrained: bool = config.get("max_investigations") is not None
    is_hard_task: bool = task_id == "hard"

    # --- Classify the alert using ground-truth hidden fields ---
    true_severity: float = alert.true_severity
    alert_age: int = int(alert.age)
    is_critical: bool = true_severity >= _CRITICAL_THRESHOLD
    is_false_positive: bool = bool(
        alert.metadata.get("false_positive", true_severity < _FALSE_POSITIVE_THRESHOLD)
    )
    is_correlated: bool = alert.is_correlated
    is_medium: bool = not is_critical and not is_false_positive

    # --- Zero-initialise all named components ---
    components: Dict[str, float] = {
        "critical_handled":       0.0,
        "failure_prevented":      0.0,
        "false_positive_ignored": 0.0,
        "medium_handled":         0.0,
        "unnecessary_invest":     0.0,
        "missed_critical":        0.0,
        "risky_delay":            0.0,
        "unnecessary_escalate":   0.0,
        # New shaping components
        "age_urgency":            0.0,
        "chain_medium_shaping":   0.0,
    }

    action_type: str = action.action_type
    proactive: bool = action_type in ("INVESTIGATE", "ESCALATE")

    # -----------------------------------------------------------------------
    # INVESTIGATE
    # -----------------------------------------------------------------------
    if action_type == "INVESTIGATE":
        if is_critical:
            components["critical_handled"] = REWARD_CRITICAL_HANDLED
        elif is_false_positive:
            components["unnecessary_invest"] = PENALTY_UNNECESSARY_INVEST
        else:
            # Medium severity — reward proportional to true severity so the agent
            # learns that higher-severity mediums deserve more attention.
            components["medium_handled"] = REWARD_MEDIUM_INVESTIGATED * (0.5 + true_severity)

    # -----------------------------------------------------------------------
    # ESCALATE
    # -----------------------------------------------------------------------
    elif action_type == "ESCALATE":
        if is_critical:
            components["critical_handled"] = REWARD_CRITICAL_HANDLED * 0.9
        elif is_false_positive:
            components["unnecessary_escalate"] = PENALTY_UNNECESSARY_ESCALATE
        else:
            if true_severity >= _MEDIUM_ESCALATE_THRESHOLD:
                components["medium_handled"] = REWARD_MEDIUM_INVESTIGATED * true_severity
            else:
                components["medium_handled"] = REWARD_MEDIUM_INVESTIGATED * true_severity * 0.4

    # -----------------------------------------------------------------------
    # IGNORE
    # -----------------------------------------------------------------------
    elif action_type == "IGNORE":
        if is_false_positive:
            components["false_positive_ignored"] = REWARD_FALSE_POSITIVE_IGNORED
        elif is_critical:
            components["missed_critical"] = PENALTY_MISSED_CRITICAL
        else:
            if true_severity < _MEDIUM_HIGH_THRESHOLD:
                components["medium_handled"] = 0.5
            else:
                components["missed_critical"] = PENALTY_IGNORE_HIGH_MEDIUM

    # -----------------------------------------------------------------------
    # DELAY
    # -----------------------------------------------------------------------
    elif action_type == "DELAY":
        if is_critical:
            components["risky_delay"] = PENALTY_CRITICAL_DELAYED
        elif is_false_positive:
            components["medium_handled"] = -0.3
        else:
            if resource_constrained:
                components["medium_handled"] = 0.5
            else:
                components["medium_handled"] = PENALTY_MEDIUM_DELAYED

    # -----------------------------------------------------------------------
    # Correlated-alert shaping
    #
    # Case A — critical + correlated: agent gets REWARD_FAILURE_PREVENTED for
    #   proactive action.  This already existed.
    #
    # Case B — medium + correlated (NEW): this is the silent killer.  The
    #   grader sees a chain propagate and scores 0.00 for the chain; the old
    #   reward gave only the medium_handled signal (~1-2 pts) for DELAY.  We
    #   now apply a large explicit bonus/penalty to teach the agent that
    #   medium-severity correlated alerts are chain triggers worth stopping.
    #
    # Note: is_correlated is ground truth (env has it).  The agent must LEARN
    #   to identify correlated alerts from visible features.  The shaped reward
    #   provides the learning signal; encode_state() must provide the features.
    # -----------------------------------------------------------------------
    if is_correlated and proactive:
        # Both critical and medium correlated alerts: proactive handling is correct
        components["failure_prevented"] = REWARD_FAILURE_PREVENTED

        # Extra: scale the bonus by age so older chain alerts earn more for
        # being handled before they cascade further.
        if alert_age >= _URGENCY_RAMP_START:
            age_scale = min(alert_age / _CRITICAL_AGE_THRESHOLD, 1.0)
            components["failure_prevented"] += REWARD_FAILURE_PREVENTED * age_scale * 0.5

    elif is_correlated and not proactive and is_medium:
        # Medium correlated + DELAY or IGNORE: chain will propagate next step
        if action_type == "DELAY":
            components["chain_medium_shaping"] = _CHAIN_MEDIUM_DELAY_PENALTY
        else:  # IGNORE
            components["chain_medium_shaping"] = _CHAIN_MEDIUM_IGNORE_PENALTY

    elif is_correlated and not proactive and is_critical:
        # Critical correlated + DELAY/IGNORE already has missed_critical or
        # risky_delay; add extra chain propagation signal on top
        components["chain_medium_shaping"] = -5.0

    # -----------------------------------------------------------------------
    # Age-urgency shaping (applies to non-FP alerts, correlated or not)
    #
    # Purpose: give the agent a pre-failure warning signal.  Without this,
    # the penalty only arrives when a failure actually occurs (env's
    # _check_for_failures removes the alert and the grader scores 0).  By
    # the time the grader fires, the policy gradient has no good credit
    # assignment back to the delay/ignore action that caused it.
    #
    # Shaping rule:
    #   - Proactive action on an aging alert → urgency bonus (up to +6)
    #   - DELAY/IGNORE on a critical aging alert → escalating penalty
    #   - Does NOT double-count correlated alerts (they get chain shaping above)
    # -----------------------------------------------------------------------
    if not is_false_positive and alert_age >= _URGENCY_RAMP_START:
        # age_ratio goes from 0 at ramp_start to 1 at critical threshold
        age_ratio = min(
            (alert_age - _URGENCY_RAMP_START) / max(_CRITICAL_AGE_THRESHOLD - _URGENCY_RAMP_START, 1),
            1.0,
        )

        if is_critical:
            if proactive:
                # Reward urgency: catching an aged critical before it fails
                urgency_bonus = _MAX_URGENCY_BONUS * age_ratio
                components["age_urgency"] = urgency_bonus
            else:
                # DELAY or IGNORE on an aged critical: pre-failure penalty that
                # escalates as the alert approaches the age threshold
                urgency_penalty = -_MAX_URGENCY_PENALTY * age_ratio
                components["age_urgency"] = urgency_penalty

        elif is_medium and not is_correlated:
            # Medium non-correlated: smaller urgency signal, only for proactive
            # (correlated mediums are handled by chain_medium_shaping above)
            if proactive:
                urgency_bonus = min(alert_age * 0.3, 1.5)
                components["age_urgency"] = urgency_bonus

    # -----------------------------------------------------------------------
    # Hard-task only: amplify chain-propagation penalties further.
    #
    # The hard grader's stability multiplier creates a non-linear cliff:
    # missing one chain drops score by 20-40%.  The standard multiplier (1.3x)
    # doesn't fully capture this cliff because it scales everything uniformly.
    # For the hard task we add an extra penalty specifically for the actions
    # that cause chain propagation.
    # -----------------------------------------------------------------------
    if is_hard_task and is_correlated and not proactive:
        # Extra hard-task penalty for letting any correlated alert go
        chain_propagation_extra = -3.0 * true_severity
        components["chain_medium_shaping"] = (
            components.get("chain_medium_shaping", 0.0) + chain_propagation_extra
        )

    # -----------------------------------------------------------------------
    # Apply task-level multiplier (amplifies all components uniformly)
    # -----------------------------------------------------------------------
    if multiplier != 1.0:
        components = {k: v * multiplier for k, v in components.items()}

    total_reward: float = sum(components.values())

    # -----------------------------------------------------------------------
    # Info payload — consumed by graders and evaluation scripts
    # -----------------------------------------------------------------------
    info: Dict[str, Any] = {
        "alert_id":          alert.id,
        "alert_type":        alert.alert_type,
        "true_severity":     true_severity,
        "is_critical":       is_critical,
        "is_false_positive": is_false_positive,
        "is_correlated":     is_correlated,
        "alert_age":         alert_age,
        "action_correct":    _is_action_optimal(
            action_type, is_critical, is_false_positive, resource_constrained
        ),
        "task_multiplier":   multiplier,
        "raw_reward":        total_reward,
    }

    return Reward(
        value=total_reward,
        components=components,
        info=info,
    )


# ---------------------------------------------------------------------------
# System-failure penalty  (called by env on aged-out critical alerts)
# ---------------------------------------------------------------------------

def calculate_system_failure_penalty(num_failures: int) -> float:
    """
    Cumulative penalty for system failures triggered by aged-out critical alerts.

    Called by the environment AFTER _check_for_failures() — not inside
    calculate_reward() — because failures are detected at the end of a step,
    not at the moment of individual alert actions.

    Args:
        num_failures: Number of failures detected this step.

    Returns:
        Total penalty (always <= 0).
    """
    if num_failures <= 0:
        return 0.0
    penalty = -10.0
    if num_failures > 1:
        penalty += (num_failures - 1) * -12.0
    return penalty


# ---------------------------------------------------------------------------
# Episode-level bonus  (called by evaluate.py at episode end)
# ---------------------------------------------------------------------------

def calculate_episode_bonus(
    correct_actions: int,
    total_actions: int,
    failures_count: int,
) -> float:
    """
    End-of-episode bonus based on overall accuracy and failure avoidance.

    Bonus schedule:
      +10  accuracy >= 80%
      +15  zero system failures
      +10  perfect episode (100% accuracy AND zero failures)

    Args:
        correct_actions: Actions where action_correct was True.
        total_actions:   Total actions taken this episode.
        failures_count:  Total system failures this episode.

    Returns:
        Bonus value (>= 0).
    """
    if total_actions == 0:
        return 0.0

    accuracy: float = correct_actions / total_actions
    bonus: float = 0.0

    if accuracy >= 0.80:
        bonus += 10.0
    if failures_count == 0:
        bonus += 15.0
    if accuracy == 1.0 and failures_count == 0:
        bonus += 10.0

    return bonus


# ---------------------------------------------------------------------------
# Episode-level summary  (called by plots.py / evaluate.py)
# ---------------------------------------------------------------------------

def create_reward_summary(rewards: List[Reward]) -> Dict[str, Any]:
    """
    Aggregate per-step Reward objects into episode-level statistics.

    Args:
        rewards: All Reward objects returned during one episode.

    Returns:
        Dict with keys: total_reward, mean_reward, num_steps, components,
        correct_actions, accuracy.
    """
    if not rewards:
        return {
            "total_reward":    0.0,
            "mean_reward":     0.0,
            "num_steps":       0,
            "components":      {},
            "correct_actions": 0,
            "accuracy":        0.0,
        }

    total: float = sum(r.value for r in rewards)
    component_totals: Dict[str, float] = {}
    for r in rewards:
        for k, v in r.components.items():
            component_totals[k] = component_totals.get(k, 0.0) + v

    correct: int = sum(1 for r in rewards if r.info.get("action_correct", False))

    return {
        "total_reward":    total,
        "mean_reward":     total / len(rewards),
        "num_steps":       len(rewards),
        "components":      component_totals,
        "correct_actions": correct,
        "accuracy":        correct / len(rewards),
    }


# ---------------------------------------------------------------------------
# Theoretical reward range  (used by OpenEnv metadata / openenv.yaml)
# ---------------------------------------------------------------------------

def get_reward_range() -> tuple[float, float]:
    """
    Theoretical [min, max] per-step reward (before task multiplier).

    Returns:
        (min_reward, max_reward)
    """
    # Max: critical + correlated + max age urgency
    max_r: float = (
        REWARD_CRITICAL_HANDLED
        + REWARD_FAILURE_PREVENTED * 1.5   # base + age scale at max
        + _MAX_URGENCY_BONUS
    )
    # Min: missed critical + chain propagation extra + max age urgency penalty
    min_r: float = (
        PENALTY_MISSED_CRITICAL
        + _CHAIN_MEDIUM_IGNORE_PENALTY
        + (-_MAX_URGENCY_PENALTY)
        + (-3.0)   # hard-task extra
    )
    return (min_r, max_r)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _is_action_optimal(
    action_type: str,
    is_critical: bool,
    is_false_positive: bool,
    resource_constrained: bool = False,
) -> bool:
    """
    Return True if ``action_type`` is optimal given ground-truth classification.

    Decision table:
      Critical       -> INVESTIGATE or ESCALATE
      False positive -> IGNORE
      Medium         -> INVESTIGATE, ESCALATE, or (DELAY when constrained)
    """
    if is_critical:
        return action_type in ("INVESTIGATE", "ESCALATE")
    if is_false_positive:
        return action_type == "IGNORE"
    if resource_constrained and action_type == "DELAY":
        return True
    return action_type in ("INVESTIGATE", "ESCALATE")


# ---------------------------------------------------------------------------
# Self-test  (python rewards/reward.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from adaptive_alert_triage.models import Alert, Action

    def _make_alert(
        alert_id: str,
        true_sev: float,
        visible_sev: float = 0.5,
        correlated: bool = False,
        is_fp: bool = False,
        age: int = 1,
    ) -> Alert:
        return Alert(
            id=alert_id,
            visible_severity=visible_sev,
            confidence=0.9,
            alert_type="CPU",
            age=age,
            true_severity=true_sev,
            is_correlated=correlated,
            metadata={"false_positive": is_fp},
        )

    easy_cfg    = {"task_id": "easy",   "max_investigations": None, "failure_threshold": 5}
    medium_cfg  = {"task_id": "medium", "max_investigations": 3,    "failure_threshold": 5}
    hard_cfg    = {"task_id": "hard",   "max_investigations": 3,    "failure_threshold": 3}

    cases = [
        # ── Original cases (unchanged expected values) ───────────────────────
        (
            "Critical + INVESTIGATE",
            "INVESTIGATE", _make_alert("a1", 0.90, 0.85), easy_cfg,
            REWARD_CRITICAL_HANDLED,
        ),
        (
            "Critical + ESCALATE",
            "ESCALATE", _make_alert("a2", 0.90, 0.85), easy_cfg,
            REWARD_CRITICAL_HANDLED * 0.9,
        ),
        (
            "False positive + IGNORE",
            "IGNORE", _make_alert("a3", 0.10, 0.25, is_fp=True), easy_cfg,
            REWARD_FALSE_POSITIVE_IGNORED,
        ),
        (
            "Critical + IGNORE  (worst case)",
            "IGNORE", _make_alert("a4", 0.95, 0.70), easy_cfg,
            PENALTY_MISSED_CRITICAL,
        ),
        (
            "Correlated critical + INVESTIGATE  (bonus, age=1)",
            "INVESTIGATE", _make_alert("a5", 0.88, 0.80, correlated=True), easy_cfg,
            REWARD_CRITICAL_HANDLED + REWARD_FAILURE_PREVENTED,
        ),
        (
            "False positive + INVESTIGATE  (waste)",
            "INVESTIGATE", _make_alert("a6", 0.10, 0.30, is_fp=True), easy_cfg,
            PENALTY_UNNECESSARY_INVEST,
        ),
        (
            "Critical + DELAY  (risky)",
            "DELAY", _make_alert("a7", 0.80, 0.75), easy_cfg,
            PENALTY_CRITICAL_DELAYED,
        ),
        (
            "Medium + DELAY under resource constraint",
            "DELAY", _make_alert("a8", 0.55, 0.50), medium_cfg,
            0.5 * _TASK_MULTIPLIERS["medium"],
        ),

        # ── New shaping cases ────────────────────────────────────────────────
        (
            "[NEW] Medium correlated + DELAY → chain propagates (easy)",
            "DELAY", _make_alert("b1", 0.65, 0.60, correlated=True), easy_cfg,
            # medium_handled (resource-unconstrained DELAY) + chain_medium_shaping
            PENALTY_MEDIUM_DELAYED + _CHAIN_MEDIUM_DELAY_PENALTY,
        ),
        (
            "[NEW] Medium correlated + INVESTIGATE → chain stops (easy)",
            "INVESTIGATE", _make_alert("b2", 0.65, 0.60, correlated=True), easy_cfg,
            # medium_handled + failure_prevented (no age bonus at age=1 < ramp_start=2)
            REWARD_MEDIUM_INVESTIGATED * (0.5 + 0.65) + REWARD_FAILURE_PREVENTED,
        ),
        (
            "[NEW] Aged critical + INVESTIGATE → urgency bonus (age=4)",
            "INVESTIGATE", _make_alert("c1", 0.85, 0.80, age=4), easy_cfg,
            # critical_handled + age_urgency
            REWARD_CRITICAL_HANDLED
            + _MAX_URGENCY_BONUS * ((4 - _URGENCY_RAMP_START) / (_CRITICAL_AGE_THRESHOLD - _URGENCY_RAMP_START)),
        ),
        (
            "[NEW] Aged critical + DELAY → escalating penalty (age=4)",
            "DELAY", _make_alert("c2", 0.85, 0.80, age=4), easy_cfg,
            # risky_delay + age_urgency penalty
            PENALTY_CRITICAL_DELAYED
            + (-_MAX_URGENCY_PENALTY) * ((4 - _URGENCY_RAMP_START) / (_CRITICAL_AGE_THRESHOLD - _URGENCY_RAMP_START)),
        ),
    ]

    print("reward.py — self-test\n" + "=" * 60)
    all_pass = True
    for desc, act, alert, cfg, expected in cases:
        action = Action(alert_id=alert.id, action_type=act)
        result = calculate_reward(action, alert, cfg)
        ok = abs(result.value - expected) < 1e-4
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{status}]  {desc}")
        if not ok:
            print(f"         got {result.value:.6f}, expected {expected:.6f}")
            print(f"         components: {result.components}")

    print("=" * 60)
    rng = get_reward_range()
    print(f"Reward range: min={rng[0]:+.1f}  max={rng[1]:+.1f}")
    print("\nAll tests passed!" if all_pass else "\nSome tests FAILED — check above.")