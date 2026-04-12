"""
Task 3: Cascading Failure Prevention  (Hard)
============================================

Objective
---------
Detect correlated alert chains and stop them *before* they cascade into
system failures.  This is the defining challenge of the hard task: each
chain's trigger alert arrives first; if the agent fails to handle it
correctly, the *next* alert in the chain is spawned by the environment in
a future step — and so on until the chain terminates in a system failure.

How cascading chains work in this environment
---------------------------------------------
Unlike the easy and medium tasks where alerts are mostly independent, the
hard task environment (correlation_probability = 0.40) frequently spawns
*correlated chains*.  The full chain is NOT delivered all at once.  Instead:

  Step N   → trigger alert arrives  (chain position 0)
  Step N+k → if trigger was IGNORED/DELAYED, child alert arrives (position 1)
  Step N+m → if child was missed, grandchild arrives (position 2)  …etc.

This means the agent must:
  1. Recognise a trigger alert before seeing any siblings (the siblings
     haven't spawned yet).
  2. INVESTIGATE or ESCALATE the trigger, which *stops* the chain from
     propagating further.
  3. If the trigger is missed, handle each subsequent child aggressively.

The grader tracks chain-level outcomes (was the chain stopped at the
trigger? at the first child? did it run all the way to a system failure?)
and awards bonus/penalty accordingly.

Grading formula  (all inputs are deterministic given the same seed)
-------------------------------------------------------------------
    component_score = chain_outcome_score + isolation_bonus
                    + stability_score − timing_penalty

    where:

    chain_outcome_score:
        Each chain contributes up to 1.0 depending on where it was stopped:
            stopped at trigger  (position 0)  → 1.00
            stopped at child    (position 1)  → 0.70
            stopped at position 2             → 0.40
            ran to system failure             → 0.00
        Weighted by the chain's max true_severity so dangerous chains
        matter more.

    isolation_bonus:
        +0.10 per independent (non-correlated) alert correctly handled.
        Cap: 0.20 total (so non-correlated work can contribute at most 20 %)

    stability_score:
        Based on total system failures:
            0 failures  → 1.00
            1           → 0.80
            2           → 0.60
            3           → 0.30
            4+          → 0.00

    timing_penalty:
        For each chain: 0.05 × (stop_position - 1).clamp(0, 3)
        Penalises the agent for letting the chain run multiple steps.

    Final normalisation:
        raw = component_score / max_possible_score   ∈ [0, 1]
        final = max(0.0, min(raw * stability_score, 1.0))

Success threshold: ≥ 0.50  (intentionally hard to challenge frontier models)

Why this is genuinely hard
--------------------------
  - The agent sees only one alert at a time.  It cannot know whether an
    alert is a chain trigger without correlating its type, timing, and
    visible_severity pattern.
  - The chain siblings are hidden — they haven't spawned yet when the
    trigger arrives, so `is_correlated` can only be known from ground truth.
  - Delayed action is penalised: DELAYing a trigger causes a child to spawn.
  - The stability multiplier creates a non-linear cliff: even a single
    missed chain that reaches failure drops the score significantly.

Integration with env.py
-----------------------
    obs, reward, done, info = env.step(action)

    # 1. Update correlation state (called EVERY step — chains grow dynamically)
    grader.update_correlation_state(info.get("correlation_groups", []))

    # 2. Process all actions this step
    for alert_data in info.get("processed_alerts", []):
        grader.process_step(alert_data, info)

    # 3. Record any system failures
    grader.record_failures(info.get("failures_this_step", 0))
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from adaptive_alert_triage.models import Action, Alert, Reward

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_CRITICAL_THRESHOLD: float       = 0.75
_MEDIUM_THRESHOLD: float         = 0.50
_FALSE_POSITIVE_THRESHOLD: float = 0.30

# Chain-stop rewards by position index (0 = trigger, 1 = first child, …)
_CHAIN_STOP_REWARDS: Dict[int, float] = {
    0: 1.00,   # stopped at trigger  — best outcome
    1: 0.70,   # stopped at first child
    2: 0.40,   # stopped two steps in
    3: 0.15,   # barely caught it
}
_CHAIN_FAILURE_REWARD: float   = 0.00   # ran all the way to failure
_TIMING_PENALTY_PER_STEP: float = 0.05  # extra penalty per position beyond 0
_MAX_TIMING_PENALTY_STEPS: int  = 3

# Isolation bonus for correctly handling non-correlated alerts
_ISOLATION_BONUS_PER_ALERT: float = 0.10
_ISOLATION_BONUS_CAP: float       = 0.20

# Stability score by failure count (step-function approximation)
_STABILITY_BY_FAILURES: List[Tuple[int, float]] = [
    (0, 1.00),
    (1, 0.80),
    (2, 0.60),
    (3, 0.30),
]
_STABILITY_FLOOR: float = 0.00

SUCCESS_THRESHOLD: float = 0.50


# ---------------------------------------------------------------------------
# Internal data class for one chain
# ---------------------------------------------------------------------------

class _ChainRecord:
    """Bookkeeping for a single correlated alert chain."""

    __slots__ = (
        "chain_id", "alert_ids", "max_severity",
        "stop_position", "completed", "hit_failure",
    )

    def __init__(self, chain_id: int, alert_ids: List[str]) -> None:
        self.chain_id:     int            = chain_id
        self.alert_ids:    List[str]      = list(alert_ids)
        self.max_severity: float          = 0.0   # updated as alerts arrive
        self.stop_position: Optional[int] = None   # index where chain was halted
        self.completed:    bool           = False
        self.hit_failure:  bool           = False

    def position_of(self, alert_id: str) -> Optional[int]:
        try:
            return self.alert_ids.index(alert_id)
        except ValueError:
            return None

    def mark_stopped(self, position: int, severity: float) -> None:
        if not self.completed:
            self.stop_position = position
            self.completed     = True
            self.max_severity  = max(self.max_severity, severity)

    def mark_failure(self) -> None:
        self.hit_failure = True
        self.completed   = True

    def outcome_score(self) -> float:
        """Score for this chain's outcome weighted by severity."""
        if self.hit_failure:
            return _CHAIN_FAILURE_REWARD
        if self.stop_position is None:
            return _CHAIN_FAILURE_REWARD   # chain never handled
        base   = _CHAIN_STOP_REWARDS.get(self.stop_position, _CHAIN_FAILURE_REWARD)
        timing = _TIMING_PENALTY_PER_STEP * min(self.stop_position,
                                                _MAX_TIMING_PENALTY_STEPS)
        return max(0.0, base - timing) * max(self.max_severity, 0.5)

    def max_possible(self) -> float:
        """Maximum possible score for this chain."""
        return _CHAIN_STOP_REWARDS[0] * max(self.max_severity, 0.5)


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------

class HardTaskGrader:
    """
    Grader for Task 3: Cascading Failure Prevention.

    Lifecycle (one episode)
    -----------------------
    1. Instantiate once per episode.
    2. After every env.step(action):
         a. grader.update_correlation_state(info["correlation_groups"])
         b. for alert_data in info["processed_alerts"]:
                grader.process_step(alert_data, info)
         c. grader.record_failures(info["failures_this_step"])
    3. At episode end: grader.get_episode_score() → float ∈ [0.0, 1.0].
    4. grader.get_metrics() for a full breakdown.
    5. grader.reset() to reuse for a new episode.
    """

    def __init__(
        self,
        correlation_chains: Optional[List[List[str]]] = None,
    ) -> None:
        # Map from chain_id → _ChainRecord (built up over the episode)
        self._chains: Dict[int, _ChainRecord] = {}

        # Map from alert_id → chain_id for O(1) lookup
        self._alert_to_chain: Dict[str, int] = {}

        # Independent-alert counters
        self._isolation_correct: int = 0
        self._isolation_total: int   = 0

        # System failures
        self._system_failures: int = 0

        # Overall accumulator (non-correlated alerts only)
        self._action_history: List[Dict[str, Any]] = []
        self._total_actions: int = 0

        # Seed with any chains already known at episode start
        if correlation_chains:
            self.update_correlation_state(correlation_chains)

    # ------------------------------------------------------------------
    # State update  (called every step)
    # ------------------------------------------------------------------

    def update_correlation_state(self, chains: List[List[str]]) -> None:
        """
        Sync the grader's chain knowledge with the environment's live state.

        MUST be called EVERY step with info["correlation_groups"] because
        new chains spawn mid-episode as the agent misses trigger alerts.

        Args:
            chains: Current list-of-lists of correlated alert IDs from
                    info["correlation_groups"].
        """
        for chain_id, alert_ids in enumerate(chains):
            if chain_id not in self._chains:
                # New chain discovered this step
                record = _ChainRecord(chain_id, alert_ids)
                self._chains[chain_id] = record
                for aid in alert_ids:
                    self._alert_to_chain[aid] = chain_id
            else:
                # Existing chain may have grown (new child spawned)
                existing = self._chains[chain_id]
                for aid in alert_ids:
                    if aid not in existing.alert_ids:
                        existing.alert_ids.append(aid)
                    self._alert_to_chain[aid] = chain_id

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

        For correlated alerts: updates the chain record (stopped / missed).
        For independent alerts: accumulates the isolation bonus.

        Args:
            alert_data: One entry from info["processed_alerts"].
            info:       Full info dict (unused here, kept for API symmetry).

        Returns:
            Immediate contribution to the score (for logging — the true
            episode score is only finalised at get_episode_score()).
        """
        self._total_actions += 1

        alert_id:    str   = str(alert_data.get("alert_id", ""))
        true_sev:    float = float(alert_data.get("true_severity", 0.0))
        action_type: str   = str(alert_data.get("action_taken", ""))
        is_corr:     bool  = bool(alert_data.get("is_correlated", False))
        chain_idx:   Optional[int] = alert_data.get("correlation_group_index")

        # Normalise: use our own chain map if env didn't set the index
        if chain_idx is None:
            chain_idx = self._alert_to_chain.get(alert_id)

        contribution = 0.0

        if is_corr and chain_idx is not None and chain_idx in self._chains:
            contribution = self._handle_correlated(
                alert_id, true_sev, action_type, chain_idx
            )
        else:
            # Independent alert
            contribution = self._handle_independent(true_sev, action_type)

        self._action_history.append({
            "alert_id":    alert_id,
            "action":      action_type,
            "true_severity": true_sev,
            "is_correlated": is_corr,
            "chain_idx":   chain_idx,
            "contribution": contribution,
        })

        return contribution

    def record_failures(self, count: int) -> None:
        """
        Record system failures detected this step.

        Call with info["failures_this_step"] after every env.step().

        Args:
            count: Number of failures this step (0 is fine).
        """
        self._system_failures += max(0, int(count))
        # Mark all incomplete chains as hit_failure (conservative)
        for record in self._chains.values():
            if not record.completed and count > 0:
                record.mark_failure()

    # ------------------------------------------------------------------
    # Legacy API
    # ------------------------------------------------------------------

    def grade_action(
        self,
        action: Action,
        alert: Alert,
        reward: Reward,
        current_alerts: Optional[List[Alert]] = None,
    ) -> float:
        """Grade a single action-alert pair (legacy / unit-test API)."""
        alert_data = {
            "alert_id":              alert.id,
            "true_severity":         alert.true_severity,
            "visible_severity":      alert.visible_severity,
            "confidence":            alert.confidence,
            "alert_type":            alert.alert_type,
            "age":                   alert.age,
            "action_taken":          action.action_type,
            "is_correlated":         alert.is_correlated,
            "is_false_positive":     alert.true_severity < _FALSE_POSITIVE_THRESHOLD,
            "correlation_group_index": self._alert_to_chain.get(alert.id),
        }
        return self.process_step(alert_data, {})

    def record_system_failure(self, alert_id: Optional[str] = None) -> None:
        """Legacy single-failure recorder (kept for backward compat)."""
        self.record_failures(1)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def get_episode_score(self) -> float:
        """
        Return final normalised score strictly in (0, 1) — never 0.0 or 1.0.
        """
        chain_score = sum(c.outcome_score() for c in self._chains.values())
        max_chain   = sum(c.max_possible()  for c in self._chains.values())

        isolation = min(
            self._isolation_correct * _ISOLATION_BONUS_PER_ALERT,
            _ISOLATION_BONUS_CAP,
        )

        denominator = max(max_chain, 1.0)
        raw = min((chain_score + isolation) / denominator, 1.0)

        stability = self._stability_score(self._system_failures)
        final_base = max(0.0, min(raw * stability, 1.0))
        rfinal=round(final_base,2)
        if rfinal == 0.0:
            return 0.01
        if rfinal == 1.00:
            return 0.99
        return float(rfinal)


    def passed(self) -> bool:
        """Return True if the agent meets the hard-task success threshold."""
        return self.get_episode_score() >= SUCCESS_THRESHOLD

    def calculate_correlation_detection_rate(self) -> float:
        """
        Fraction of chains that were successfully stopped (any position).

        Returns 1.0 when no chains exist (nothing to detect).
        """
        if not self._chains:
            return 1.0
        stopped = sum(
            1 for c in self._chains.values()
            if c.completed and not c.hit_failure
        )
        raw = stopped / len(self._chains)
        return raw

    def calculate_stability_score(self) -> float:
        """Return the stability multiplier for the current failure count."""
        return self._stability_score(self._system_failures)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """Return a full breakdown of episode performance."""
        score          = self.get_episode_score()
        corr_rate      = self.calculate_correlation_detection_rate()
        stability      = self.calculate_stability_score()

        chain_details = []
        for cid, rec in sorted(self._chains.items()):
            chain_details.append({
                "chain_id":      cid,
                "length":        len(rec.alert_ids),
                "max_severity":  rec.max_severity,
                "stop_position": rec.stop_position,
                "hit_failure":   rec.hit_failure,
                "outcome_score": rec.outcome_score(),
            })

        breakdown: Dict[str, int] = {
            "INVESTIGATE": 0, "IGNORE": 0, "ESCALATE": 0, "DELAY": 0,
        }
        for h in self._action_history:
            breakdown[h["action"]] = breakdown.get(h["action"], 0) + 1

        return {
            "overall_score":              score,
            "passed":                     self.passed(),
            "success_threshold":          SUCCESS_THRESHOLD,
            "chain_score":                sum(c.outcome_score() for c in self._chains.values()),
            "max_chain_score":            sum(c.max_possible()  for c in self._chains.values()),
            "correlation_detection_rate": corr_rate,
            "total_chains":               len(self._chains),
            "chains_stopped":             sum(1 for c in self._chains.values()
                                              if c.completed and not c.hit_failure),
            "chains_at_trigger":          sum(1 for c in self._chains.values()
                                              if c.stop_position == 0),
            "chains_hit_failure":         sum(1 for c in self._chains.values()
                                              if c.hit_failure),
            "chain_details":              chain_details,
            "isolation_correct":          self._isolation_correct,
            "isolation_total":            self._isolation_total,
            "system_failures":            self._system_failures,
            "stability_score":            stability,
            "total_actions":              self._total_actions,
            "action_breakdown":           breakdown,
        }

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all state for a new episode."""
        self._chains          = {}
        self._alert_to_chain  = {}
        self._isolation_correct = 0
        self._isolation_total   = 0
        self._system_failures   = 0
        self._action_history    = []
        self._total_actions     = 0

    def __repr__(self) -> str:
        score    = self.get_episode_score()
        corr_r   = self.calculate_correlation_detection_rate()
        return (
            f"HardTaskGrader(score={score:.3f}, "
            f"failures={self._system_failures}, "
            f"chains={len(self._chains)}, "
            f"detection_rate={corr_r:.3f}, "
            f"passed={self.passed()})"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _handle_correlated(
        self,
        alert_id: str,
        true_sev: float,
        action_type: str,
        chain_idx: int,
    ) -> float:
        """
        Update chain record based on action and return immediate contribution.

        If the action is INVESTIGATE or ESCALATE → chain is stopped here.
        If IGNORE or DELAY → chain continues (child will spawn next step).
        """
        record   = self._chains[chain_idx]
        position = record.position_of(alert_id)
        if position is None:
            position = len(record.alert_ids)  # fallback: treat as end of chain

        record.max_severity = max(record.max_severity, true_sev)

        proactive = action_type in ("INVESTIGATE", "ESCALATE")

        if proactive:
            # Agent stopped the chain at this position
            record.mark_stopped(position, true_sev)
            base    = _CHAIN_STOP_REWARDS.get(position, _CHAIN_FAILURE_REWARD)
            timing  = _TIMING_PENALTY_PER_STEP * min(position, _MAX_TIMING_PENALTY_STEPS)
            contrib = max(0.0, base - timing) * max(true_sev, 0.5)
        else:
            # Agent missed this alert — chain propagates
            # Give a small negative signal so the agent learns
            if true_sev >= _CRITICAL_THRESHOLD:
                contrib = -0.30 * true_sev
            else:
                contrib = -0.10

        return contrib

    def _handle_independent(self, true_sev: float, action_type: str) -> float:
        """
        Score a non-correlated alert action and accumulate isolation bonus.

        Returns a small contribution that feeds into the isolation bonus pool.
        """
        self._isolation_total += 1

        is_correct = self._independent_correct(action_type, true_sev)
        if is_correct:
            self._isolation_correct += 1
            return _ISOLATION_BONUS_PER_ALERT
        return 0.0

    @staticmethod
    def _independent_correct(action_type: str, true_sev: float) -> bool:
        """Correctness rule for non-correlated alerts (same as easy task)."""
        if true_sev >= _CRITICAL_THRESHOLD:
            return action_type in ("INVESTIGATE", "ESCALATE")
        if true_sev < _FALSE_POSITIVE_THRESHOLD:
            return action_type == "IGNORE"
        # Medium
        return action_type in ("INVESTIGATE", "ESCALATE")

    @staticmethod
    def _stability_score(failures: int) -> float:
        """Step-function stability multiplier."""
        for threshold, score in _STABILITY_BY_FAILURES:
            if failures <= threshold:
                return score
        return _STABILITY_FLOOR


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
        env:          AdaptiveAlertTriageEnv(task_id="hard") instance.
        num_episodes: Number of episodes to run.
        seed_offset:  Added to episode index for the reset seed.
        verbose:      Print per-episode summary when True.

    Returns:
        Dict with keys: mean_score, std_score, min_score, max_score,
        success_rate, mean_failures, mean_detection_rate,
        episode_scores, episode_metrics.
    """
    episode_scores:  List[float]          = []
    episode_metrics: List[Dict[str, Any]] = []

    for ep in range(num_episodes):
        grader = HardTaskGrader()
        obs    = env.reset(seed=seed_offset + ep)
        done   = False

        while not done:
            if not obs.alerts:
                break

            action = agent.act(obs)
            obs, _reward, done, info = env.step(action)

            # 1. Update chain knowledge (MUST be before process_step)
            grader.update_correlation_state(info.get("correlation_groups", []))

            # 2. Grade actions
            for alert_data in info.get("processed_alerts", []):
                grader.process_step(alert_data, info)

            # 3. Record failures
            grader.record_failures(info.get("failures_this_step", 0))

        score   = grader.get_episode_score()
        metrics = grader.get_metrics()
        episode_scores.append(score)
        episode_metrics.append(metrics)

        if verbose:
            print(
                f"  ep {ep + 1:02d}  score={score:.3f}  "
                f"failures={metrics['system_failures']}  "
                f"chains={metrics['chains_stopped']}/{metrics['total_chains']}  "
                f"passed={metrics['passed']}"
            )

    scores_arr = np.array(episode_scores)
    fail_arr   = np.array([m["system_failures"]          for m in episode_metrics])
    det_arr    = np.array([m["correlation_detection_rate"] for m in episode_metrics])

    return {
        "mean_score":          float(scores_arr.mean()),
        "std_score":           float(scores_arr.std()),
        "min_score":           float(scores_arr.min()),
        "max_score":           float(scores_arr.max()),
        "success_rate":        float((scores_arr >= SUCCESS_THRESHOLD).mean()),
        "mean_failures":       float(fail_arr.mean()),
        "mean_detection_rate": float(det_arr.mean()),
        "episode_scores":      episode_scores,
        "episode_metrics":     episode_metrics,
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("HardTaskGrader — self-test\n" + "=" * 60)

    from adaptive_alert_triage.models import Alert, Action, Reward

    def _alert(aid: str, true_sev: float, correlated: bool = False) -> Alert:
        return Alert(
            id=aid, visible_severity=true_sev * 0.95, confidence=0.85,
            alert_type="CPU", age=1, true_severity=true_sev,
            is_correlated=correlated,
        )

    # ── Scenario 1: Agent stops chain at trigger ──────────────────────
    print("\n[Scenario 1] Agent catches trigger alert — chain stops immediately")
    grader = HardTaskGrader()
    grader.update_correlation_state([["cpu_01", "mem_01", "app_01"]])

    a = _alert("cpu_01", 0.80, correlated=True)
    contrib = grader.grade_action(Action(alert_id="cpu_01", action_type="INVESTIGATE"), a, Reward(value=0))
    print(f"  Trigger INVESTIGATE  contrib={contrib:+.4f}")

    grader.record_failures(0)
    score = grader.get_episode_score()
    m     = grader.get_metrics()
    print(f"  Episode score : {score:.4f}  (expected ≥ 0.5)")
    print(f"  Chains stopped at trigger: {m['chains_at_trigger']}")
    assert score >= 0.50, f"Expected ≥0.50, got {score}"
    assert m["chains_at_trigger"] == 1

    # ── Scenario 2: Agent misses trigger → child spawns → agent catches child ──
    print("\n[Scenario 2] Agent misses trigger, catches first child")
    grader2 = HardTaskGrader()
    grader2.update_correlation_state([["cpu_02", "mem_02", "app_02"]])

    # Miss trigger
    a1 = _alert("cpu_02", 0.78, correlated=True)
    c1 = grader2.grade_action(Action(alert_id="cpu_02", action_type="IGNORE"), a1, Reward(value=0))
    print(f"  Trigger IGNORE   contrib={c1:+.4f}  (penalty expected)")

    # Child alert spawned, agent catches it
    a2 = _alert("mem_02", 0.85, correlated=True)
    c2 = grader2.grade_action(Action(alert_id="mem_02", action_type="INVESTIGATE"), a2, Reward(value=0))
    print(f"  Child INVESTIGATE contrib={c2:+.4f}")

    grader2.record_failures(0)
    score2 = grader2.get_episode_score()
    print(f"  Episode score : {score2:.4f}  (expected < scenario 1 score)")
    assert score2 < score, "Catching child should score less than catching trigger"

    # ── Scenario 3: Full failure — agent misses entire chain ───────────
    print("\n[Scenario 3] Agent misses all chain alerts → system failure")
    grader3 = HardTaskGrader()
    grader3.update_correlation_state([["cpu_03", "mem_03", "app_03"]])

    for aid, sev in [("cpu_03", 0.80), ("mem_03", 0.88), ("app_03", 0.92)]:
        a = _alert(aid, sev, correlated=True)
        grader3.grade_action(Action(alert_id=aid, action_type="IGNORE"), a, Reward(value=0))

    grader3.record_failures(1)   # system failure registered
    score3 = grader3.get_episode_score()
    print(f"  Episode score : {score3:.4f}  (expected ≈ 0)")
    assert score3 < 0.3, f"Missed entire chain + failure should score < 0.3, got {score3}"

    # ── Determinism check ─────────────────────────────────────────────
    print("\n[Determinism] Same inputs → same score")
    def _run_fixed() -> float:
        g = HardTaskGrader()
        g.update_correlation_state([["x1", "x2"]])
        a = _alert("x1", 0.82, correlated=True)
        g.grade_action(Action(alert_id="x1", action_type="INVESTIGATE"), a, Reward(value=0))
        g.record_failures(0)
        return g.get_episode_score()

    s_a, s_b = _run_fixed(), _run_fixed()
    assert s_a == s_b, f"Non-deterministic: {s_a} != {s_b}"
    print(f"  Score both runs: {s_a:.6f}  ✓")

    print("\n" + "=" * 60)
    print("All self-tests passed!")