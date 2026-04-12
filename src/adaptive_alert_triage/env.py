"""
Adaptive Alert Triage Environment — OpenEnv-compliant OpenEnv Environment

Implements a partially observable RL environment that simulates a real-world
DevOps / SOC alert-triage workflow.  An agent must process a continuous stream
of system alerts under time and resource constraints, deciding for each alert:

    INVESTIGATE  — allocate resources to diagnose (costly)
    IGNORE       — dismiss as noise (efficient for false positives)
    ESCALATE     — route to specialist team
    DELAY        — defer to the next time-step

The environment supports three difficulty tasks:

    easy   (30 steps, no resource constraint, 10 % correlation probability)
    medium (40 steps, K=3 investigations/step,  20 % correlation probability)
    hard   (50 steps, K=3 investigations/step,  40 % correlation probability,
            stricter failure threshold)

OpenEnv interface
-----------------
    reset(seed?, options?) -> Observation
    step(action)           -> (Observation, Reward, done, info)
    state()                -> EpisodeState

Info dict keys (required by graders)
-------------------------------------
    processed_alerts  : list[dict]  — ground-truth data for every action taken
                                      this step (alert_id, true_severity,
                                      is_false_positive, action_taken, etc.)
    correlation_groups: list[list]  — current correlated-chain groups (alert IDs)
    failures_this_step: int         — failures triggered this step
    system_failure    : bool        — True if the episode is in a failure state
    step              : int         — current step index
    cumulative_reward : float       — total reward so far
    failures_count    : int         — total failures so far
    action_correct    : bool        — whether the most recent action was optimal
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import openenv_shim as gym
from openenv_shim import spaces

from adaptive_alert_triage.models import (
    Action,
    Alert,
    EpisodeState,
    Observation,
    Reward,
)
from adaptive_alert_triage import utils

# Import reward calculation with graceful fallback for development mode
import os as _os
import sys as _sys

try:
    from rewards.reward import calculate_reward
except ImportError:
    _project_root = _os.path.dirname(
        _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    )
    if _project_root not in _sys.path:
        _sys.path.insert(0, _project_root)
    from rewards.reward import calculate_reward  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Task configurations
# ---------------------------------------------------------------------------

_TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "easy": {
        "max_steps": 10,
        "failure_threshold": 2,
        "max_investigations": None,   # unconstrained
        "correlation_probability": 0.10,
        "description": "Basic alert prioritisation — no resource constraint.",
    },
    "medium": {
        "max_steps": 15,
        "failure_threshold": 3,
        "max_investigations": 3,      # K = 3 per step
        "correlation_probability": 0.20,
        "description": "Resource-constrained triage — K=3 investigations/step.",
    },
    "hard": {
        "max_steps": 20,
        "failure_threshold": 2,       # stricter
        "max_investigations": 3,
        "correlation_probability": 0.40,
        "description": (
            "Cascading-failure prevention — correlated alerts, delayed failures, "
            "hidden severity, strict failure threshold."
        ),
    },
}


# ---------------------------------------------------------------------------
# Main environment class
# ---------------------------------------------------------------------------

class AdaptiveAlertTriageEnv(gym.Env):
    """
    OpenEnv environment for adaptive alert triage.

    Parameters
    ----------
    task_id : str
        Difficulty level: ``"easy"``, ``"medium"``, or ``"hard"``.
    max_steps : int, optional
        Override the task-default episode length.
    seed : int, optional
        Fixed random seed for full reproducibility.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        task_id: str = "easy",
        max_steps: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        if task_id not in _TASK_CONFIGS:
            raise ValueError(
                f"Unknown task_id '{task_id}'. "
                f"Valid options: {sorted(_TASK_CONFIGS.keys())}"
            )

        self.task_id: str = task_id
        self.config: Dict[str, Any] = dict(_TASK_CONFIGS[task_id])
        self.max_steps: int = max_steps or self.config["max_steps"]
        self.failure_threshold: int = self.config["failure_threshold"]
        self.max_investigations_per_step: Optional[int] = self.config["max_investigations"]

        # Episode state — initialised properly in reset()
        self.current_step: int = 0
        self.alerts: List[Alert] = []
        self.failures_count: int = 0
        self.cumulative_reward: float = 0.0
        self.investigations_used: int = 0

        # Hidden state
        self.correlation_groups: List[List[str]] = []

        # Action history (for state() and checkpointing)
        self._action_history: List[Action] = []

        # Real-alert ingestion queue (Datadog / Kafka webhook mode)
        self.real_alerts_queue: deque = deque(maxlen=50)

        # Per-step grading data — populated in step(), consumed by graders
        self._processed_alerts_this_step: List[Dict[str, Any]] = []
        self._failures_this_step: int = 0

        # Seed
        self._seed: Optional[int] = seed
        if seed is not None:
            utils.set_seed(seed)

        # OpenEnv spaces (abstract; real actions are Action Pydantic objects)
        self.action_space = spaces.Discrete(4)   # 4 ActionType values
        self.observation_space = spaces.Dict(
            {
                "system_load": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                "queue_length": spaces.Box(0, 100, shape=(1,), dtype=np.int32),
                "time_remaining": spaces.Box(
                    0, self.max_steps, shape=(1,), dtype=np.int32
                ),
            }
        )

    # ------------------------------------------------------------------
    # OpenEnv interface — reset
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Observation:
        """
        Reset the environment to a clean initial state.

        Args:
            seed:    Override seed for this episode.
            options: Reserved for future use (ignored).

        Returns:
            Initial Observation with no agent-visible hidden fields.
        """
        if seed is not None:
            self._seed = seed
        if self._seed is not None:
            utils.set_seed(self._seed)

        # Reset all episode counters
        self.current_step = 0
        self.failures_count = 0
        self.cumulative_reward = 0.0
        self.investigations_used = 0
        self.correlation_groups = []
        self._action_history = []
        self._processed_alerts_this_step = []
        self._failures_this_step = 0

        # Generate the initial alert batch
        self.alerts = self._generate_initial_alerts()

        return self._create_observation()

    # ------------------------------------------------------------------
    # OpenEnv interface — step
    # ------------------------------------------------------------------

    def step(
        self, action: Action
    ) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Execute one environment step.

        The agent submits one Action per call; the environment:
          1. Validates the alert ID and resource budget.
          2. Records ground-truth data for the graders.
          3. Calculates the dense reward.
          4. Applies the action (removes / keeps alert).
          5. Ages remaining alerts.
          6. Checks for delayed failures.
          7. Generates new alerts (Poisson arrivals + possible correlation chain).
          8. Increments step counter and resets per-step budget.
          9. Returns (Observation, Reward, done, info).

        The ``info`` dict always contains:
            processed_alerts  — list of ground-truth dicts, one per action
            correlation_groups — current correlation chains
            failures_this_step — failures triggered this step
            system_failure    — whether the system is in a failure state
            step              — current step index
            cumulative_reward — total reward this episode
            failures_count    — total failures this episode
            action_correct    — whether the action matched the optimal policy

        Args:
            action: Agent's Action targeting one alert by ID.

        Returns:
            (next_observation, reward, done, info)
        """
        # --- Reset per-step tracking ---
        self._processed_alerts_this_step = []
        self._failures_this_step = 0

        # --- Validate alert ID ---
        alert = self._get_alert_by_id(action.alert_id)
        if alert is None:
            reward = Reward(
                value=0.01,
                components={"invalid_action": -5.0},
                info={"error": f"Alert ID '{action.alert_id}' not found in queue"},
            )
            obs = self._create_observation()
            return obs, reward, True, self._build_info(
                action_correct=False,
                extra={"error": "Invalid alert ID"},
            )

        # --- Resource-budget enforcement ---
        if (
            self.max_investigations_per_step is not None
            and action.action_type == "INVESTIGATE"
        ):
            if self.investigations_used >= self.max_investigations_per_step:
                reward = Reward(
                    value=0.01,
                    components={"resource_budget_exceeded": -3.0},
                    info={
                        "error": "Investigation budget exhausted for this step",
                        "budget": self.max_investigations_per_step,
                        "used": self.investigations_used,
                    },
                )
                obs = self._create_observation()
                return obs, reward, False, self._build_info(
                    action_correct=False,
                    extra={"resource_constraint_violated": True},
                )
            self.investigations_used += 1

        # --- Record ground-truth for graders BEFORE removing the alert ---
        processed: Dict[str, Any] = {
            "alert_id": alert.id,
            "true_severity": alert.true_severity,
            "visible_severity": alert.visible_severity,
            "confidence": alert.confidence,
            "alert_type": alert.alert_type,
            "age": alert.age,
            "is_correlated": alert.is_correlated,
            "is_false_positive": bool(alert.metadata.get("false_positive", False)),
            "action_taken": action.action_type,
            "correlation_group_index": self._find_correlation_group(alert.id),
        }
        self._processed_alerts_this_step.append(processed)

        # --- Track action history ---
        self._action_history.append(action)

        # --- Calculate dense reward ---
        reward = calculate_reward(action, alert, self.config)
        self.cumulative_reward += reward.value

        # --- Apply action to alert queue ---
        self._process_action(action, alert)

        # --- Age all remaining unresolved alerts ---
        self._age_alerts()

        # --- Check for failures triggered by aged critical alerts ---
        self._failures_this_step = self._check_for_failures()
        self.failures_count += self._failures_this_step

        # --- Generate new alerts (Poisson arrivals + possible chain) ---
        if utils.should_generate_new_alerts(self.current_step, len(self.alerts)):
            new_alerts = self._generate_new_alerts()
            self.alerts.extend(new_alerts)

        # --- Advance step and reset per-step investigation budget ---
        self.current_step += 1
        self.investigations_used = 0

        # --- Termination check ---
        done: bool = self._is_terminal()

        # --- Build next observation (hidden fields masked) ---
        obs = self._create_observation()

        # --- Determine overall failure state ---
        system_in_failure: bool = (
            self.failures_count >= self.failure_threshold
            or self._failures_this_step > 0
        )

        info = self._build_info(
            action_correct=bool(reward.info.get("action_correct", False)),
            extra={
                "system_failure": system_in_failure,
                "alert_handled": alert.id,
            },
        )

        return obs, reward, done, info

    # ------------------------------------------------------------------
    # OpenEnv interface — state
    # ------------------------------------------------------------------

    def state(self) -> EpisodeState:
        """
        Return the complete internal episode state (visible + hidden).

        Used by evaluation scripts, replay tools, and the hard-task grader
        for root-cause analysis.  NOT intended to be passed to the agent.

        Returns:
            EpisodeState with full ground-truth information.
        """
        hidden: Dict[str, Any] = {
            "true_severities": {a.id: a.true_severity for a in self.alerts},
            "correlation_groups": [list(g) for g in self.correlation_groups],
            "false_positives": [
                a.id for a in self.alerts
                if a.metadata.get("false_positive", False)
            ],
            # Pending failures: alerts that are critical AND close to the age threshold
            "pending_failures": {
                a.id: utils.CRITICAL_AGE_THRESHOLD - a.age
                for a in self.alerts
                if utils.is_critical_alert(a)
                and a.age < utils.CRITICAL_AGE_THRESHOLD
            },
        }

        return EpisodeState(
            observation=self._create_observation(),
            hidden_state=hidden,
            cumulative_reward=self.cumulative_reward,
            failures_count=self.failures_count,
            actions_taken=[a.model_dump() for a in self._action_history],
            seed=self._seed,
        )

    # ------------------------------------------------------------------
    # Internal helpers — observation construction
    # ------------------------------------------------------------------

    def _create_observation(self) -> Observation:
        """
        Build the agent-facing Observation by masking all hidden fields.

        true_severity and is_correlated are zeroed-out; metadata is stripped.
        The agent must infer hidden information from visible_severity,
        confidence, alert_type, and age alone.
        """
        system_load: float = utils.calculate_system_load(len(self.alerts))

        visible_alerts: List[Alert] = []
        for a in self.alerts:
            visible_alerts.append(
                Alert(
                    id=a.id,
                    visible_severity=a.visible_severity,
                    confidence=a.confidence,
                    alert_type=a.alert_type,
                    age=a.age,
                    # Hidden fields zeroed out
                    true_severity=0.0,
                    is_correlated=False,
                    metadata={},
                )
            )

        resource_budget: Optional[int] = None
        if self.max_investigations_per_step is not None:
            resource_budget = self.max_investigations_per_step - self.investigations_used

        return Observation(
            alerts=visible_alerts,
            system_load=system_load,
            queue_length=len(self.alerts),
            time_remaining=max(0, self.max_steps - self.current_step),
            episode_step=self.current_step,
            resource_budget=resource_budget,
        )

    # ------------------------------------------------------------------
    # Internal helpers — alert generation
    # ------------------------------------------------------------------

    def _generate_initial_alerts(self) -> List[Alert]:
        """
        Generate the starting alert batch for a fresh episode.

        Real alerts from the ingestion queue are prioritised; any remaining
        slots are filled with synthetic alerts.
        """
        num_initial: int = int(np.random.randint(3, 7))
        alerts: List[Alert] = []

        # Drain real alerts first
        while self.real_alerts_queue and len(alerts) < num_initial:
            raw = self.real_alerts_queue.popleft()
            alerts.append(self._ingest_real_alert(raw))

        # Fill with synthetic
        for i in range(len(alerts), num_initial):
            alerts.append(
                utils.generate_alert(step=0, alert_index=i)
            )

        return alerts

    def _generate_new_alerts(self) -> List[Alert]:
        """
        Generate alerts to append to the queue this step.

        If real alerts are queued they are processed first (no synthetic
        alerts generated that step).  Otherwise, a Poisson-sampled batch of
        independent alerts is generated, with a task-configured probability
        that a correlated chain replaces the batch entirely.
        """
        # Priority: real ingest queue
        if self.real_alerts_queue:
            raw = self.real_alerts_queue.popleft()
            return [self._ingest_real_alert(raw)]

        # Correlated chain vs independent batch
        if np.random.random() < self.config["correlation_probability"]:
            chain_alerts = utils.generate_correlated_alerts(
                self.current_step, num_alerts=3
            )
            self.correlation_groups.append([a.id for a in chain_alerts])
            return chain_alerts

        num_new: int = utils.sample_num_new_alerts()
        return [
            utils.generate_alert(
                step=self.current_step,
                alert_index=i,
            )
            for i in range(num_new)
        ]

    @staticmethod
    def _ingest_real_alert(raw: Dict[str, Any]) -> Alert:
        """
        Convert a raw real-alert dict into an Alert with synthesised ground truth.

        Ground truth is estimated by adding Gaussian noise to visible_severity,
        reflecting that real monitoring tools provide imperfect severity scores.
        """
        true_severity: float = float(
            np.clip(
                float(raw["visible_severity"]) + np.random.normal(0.0, 0.10),
                0.0,
                1.0,
            )
        )
        return Alert(
            id=raw["id"],
            visible_severity=float(raw["visible_severity"]),
            confidence=float(raw["confidence"]),
            alert_type=raw["type"],
            age=0,
            true_severity=true_severity,
            is_correlated=False,
            metadata={"source": "real_ingest", "raw": raw},
        )

    # ------------------------------------------------------------------
    # Internal helpers — action processing
    # ------------------------------------------------------------------

    def _process_action(self, action: Action, alert: Alert) -> None:
        """
        Apply the agent's action to the alert queue.

        INVESTIGATE, ESCALATE, and IGNORE all resolve the alert (remove it
        from the queue).  DELAY keeps the alert in the queue; its age will
        be incremented by _age_alerts().
        """
        if action.action_type in ("INVESTIGATE", "ESCALATE", "IGNORE"):
            self.alerts = [a for a in self.alerts if a.id != alert.id]
        # DELAY: no-op — alert remains; age increment handled in _age_alerts()

    def _age_alerts(self) -> None:
        """Increment the age of every unresolved alert by one step."""
        for alert in self.alerts:
            alert.age += 1

    def _check_for_failures(self) -> int:
        """
        Detect and remove alerts that have caused system failures.

        A failure occurs when a critical alert (true_severity ≥ 0.75) has
        been in the queue for CRITICAL_AGE_THRESHOLD or more steps without
        being resolved.  Each such alert contributes one failure event.

        Returns:
            Number of new failure events detected this step.
        """
        failures: int = 0
        failed_ids: List[str] = []

        for alert in self.alerts:
            if (
                utils.is_critical_alert(alert)
                and alert.age >= utils.CRITICAL_AGE_THRESHOLD
            ):
                failures += 1
                failed_ids.append(alert.id)

        # Remove failed alerts (they've escalated out of the triage queue)
        if failed_ids:
            self.alerts = [a for a in self.alerts if a.id not in failed_ids]

        return failures

    # ------------------------------------------------------------------
    # Internal helpers — utilities
    # ------------------------------------------------------------------

    def _get_alert_by_id(self, alert_id: str) -> Optional[Alert]:
        """Return the Alert with the given ID, or None if not found."""
        for alert in self.alerts:
            if alert.id == alert_id:
                return alert
        return None

    def _find_correlation_group(self, alert_id: str) -> Optional[int]:
        """
        Return the index of the correlation group that contains alert_id, or None.

        Used to populate the ``correlation_group_index`` field in processed_alerts
        so the hard-task grader can score root-cause identification.
        """
        for idx, group in enumerate(self.correlation_groups):
            if alert_id in group:
                return idx
        return None

    def _is_terminal(self) -> bool:
        """Return True if the episode should end."""
        return (
            self.current_step >= self.max_steps
            or self.failures_count >= self.failure_threshold
        )

    def _build_info(
        self,
        action_correct: bool,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Assemble the standard info dict returned from step().

        Always includes the keys required by all three task graders.
        Additional keys can be merged in via ``extra``.
        """
        info: Dict[str, Any] = {
            # Core grading keys (required)
            "processed_alerts": list(self._processed_alerts_this_step),
            "correlation_groups": [list(g) for g in self.correlation_groups],
            "failures_this_step": self._failures_this_step,
            "system_failure": self.failures_count >= self.failure_threshold
            or self._failures_this_step > 0,
            # Convenience telemetry
            "step": self.current_step,
            "cumulative_reward": self.cumulative_reward,
            "failures_count": self.failures_count,
            "action_correct": action_correct,
        }
        if extra:
            info.update(extra)
        return info

    # ------------------------------------------------------------------
    # OpenEnv render
    # ------------------------------------------------------------------

    def render(self, mode: str = "human") -> Optional[str]:
        """
        Render a text summary of the current environment state.

        Args:
            mode: ``"human"`` (prints to stdout) or ``"ansi"`` (returns string).

        Returns:
            String if mode is ``"ansi"``, otherwise None.
        """
        lines = [
            f"\n=== Step {self.current_step}/{self.max_steps}"
            f"  [{self.task_id}] ===",
            f"  Failures     : {self.failures_count}/{self.failure_threshold}",
            f"  Cum. reward  : {self.cumulative_reward:+.1f}",
            f"  Active alerts: {len(self.alerts)}",
        ]

        if self.max_investigations_per_step is not None:
            lines.append(
                f"  Inv. budget  : "
                f"{self.max_investigations_per_step - self.investigations_used}"
                f"/{self.max_investigations_per_step} remaining"
            )

        if self.alerts:
            lines.append("\n  Alerts (first 5):")
            for a in self.alerts[:5]:
                lines.append(
                    f"    {a.id}  sev={a.visible_severity:.2f}"
                    f"  conf={a.confidence:.2f}"
                    f"  type={a.alert_type:<12}"
                    f"  age={a.age}"
                )
            if len(self.alerts) > 5:
                lines.append(f"    … and {len(self.alerts) - 5} more")

        output = "\n".join(lines) + "\n"

        if mode == "human":
            print(output)
            return None
        return output


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

def main() -> None:
    """Run a short demo episode with a simple heuristic policy."""
    print("Adaptive Alert Triage Environment — Demo\n")

    env = AdaptiveAlertTriageEnv(task_id="easy", seed=42)
    obs: Observation = env.reset()
    print(f"Initial observation: {len(obs.alerts)} alerts  "
          f"(system_load={obs.system_load:.2f})\n")

    done = False
    step_count = 0

    while not done and step_count < 5:
        env.render()

        if not obs.alerts:
            print("No alerts in queue — nothing to handle.")
            break

        # Heuristic: pick the alert with the highest visible_severity
        best_alert = max(obs.alerts, key=lambda a: a.visible_severity)
        action = Action(
            alert_id=best_alert.id,
            action_type=(
                "INVESTIGATE" if best_alert.visible_severity >= 0.7 else "IGNORE"
            ),
        )

        obs, reward, done, info = env.step(action)
        print(
            f"  Action: {action.action_type} → {best_alert.id}"
            f"  Reward: {reward.value:+.1f}"
            f"  Correct: {info.get('action_correct', '?')}"
        )
        step_count += 1

    print(f"\nDemo finished after {step_count} steps.")
    print(f"Final cumulative reward : {env.cumulative_reward:+.1f}")
    print(f"Total system failures   : {env.failures_count}")


if __name__ == "__main__":
    main()