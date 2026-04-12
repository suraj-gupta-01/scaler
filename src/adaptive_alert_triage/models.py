"""
Pydantic Models for OpenEnv Compliance

Defines the Observation, Action, and Reward models using Pydantic for
validation and serialization. These models form the core API contract
for the Adaptive Alert Triage environment.

OpenEnv compliance:
  - Observation, Action, Reward are all Pydantic BaseModel subclasses
  - step(action) -> (Observation, Reward, bool, dict)
  - reset() -> Observation
  - state() -> EpisodeState
"""

from typing import List, Literal, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

AlertType = Literal["CPU", "MEMORY", "DISK", "NETWORK", "APPLICATION", "SECURITY"]
ActionType = Literal["INVESTIGATE", "IGNORE", "ESCALATE", "DELAY"]


# ---------------------------------------------------------------------------
# Alert
# ---------------------------------------------------------------------------

class Alert(BaseModel):
    """
    Represents a single alert in the system.

    Visible fields are exposed to the agent via Observation.
    Hidden fields (true_severity, is_correlated) are only used internally
    by the environment for reward calculation and failure checking — they
    are zeroed-out / masked before being returned to the agent.

    Attributes:
        id:               Unique alert identifier.
        visible_severity: Noisy, observable severity score in [0.0, 1.0].
        confidence:       Detection confidence level in [0.0, 1.0].
        alert_type:       Category of the alert (CPU, MEMORY, etc.).
        age:              Number of time-steps since the alert was generated.
        true_severity:    Ground-truth severity (hidden from agent).
        is_correlated:    Whether the alert is part of a correlated failure
                          chain (hidden from agent).
        metadata:         Optional key/value context bag.
    """

    id: str = Field(..., description="Unique alert identifier")
    visible_severity: float = Field(
        ..., ge=0.0, le=1.0, description="Observable severity score (noisy)"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Detection confidence level"
    )
    alert_type: AlertType = Field(..., description="Alert category")
    age: int = Field(..., ge=0, description="Time-steps since generation")

    # Hidden attributes — never returned to the agent in Observation
    true_severity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Ground-truth severity (hidden from agent)",
    )
    is_correlated: bool = Field(
        default=False,
        description="Part of a correlated failure chain (hidden from agent)",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Optional context bag"
    )

    @field_validator("visible_severity", "confidence", "true_severity", mode="before")
    @classmethod
    def clamp_to_unit_interval(cls, v: float) -> float:
        """Silently clamp float fields to [0.0, 1.0] to absorb small fp drift."""
        return float(max(0.0, min(1.0, float(v))))

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "alert_0001_00",
                "visible_severity": 0.82,
                "confidence": 0.91,
                "alert_type": "CPU",
                "age": 1,
            }
        }
    }


# ---------------------------------------------------------------------------
# Observation  (what the agent sees each step)
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """
    Environment observation returned by reset() and step().

    Contains only the information that is visible to the agent.
    Hidden fields (true_severity, is_correlated) are stripped before
    this object is constructed — see AdaptiveAlertTriageEnv._create_observation().

    Attributes:
        alerts:          Active alerts requiring triage decisions.
        system_load:     Current infrastructure utilisation in [0.0, 1.0].
        queue_length:    Total alerts currently in the processing queue.
        time_remaining:  Steps left before episode ends.
        episode_step:    Current step index (0-based).
        resource_budget: Remaining investigation actions this step (None if
                         task has no resource constraint).
    """

    alerts: List[Alert] = Field(default_factory=list, description="Active alerts")
    system_load: float = Field(
        ..., ge=0.0, le=1.0, description="System resource utilisation"
    )
    queue_length: int = Field(..., ge=0, description="Alerts currently in queue")
    time_remaining: int = Field(..., ge=0, description="Steps left in episode")
    episode_step: int = Field(..., ge=0, description="Current step (0-based)")
    resource_budget: Optional[int] = Field(
        None,
        description="Remaining INVESTIGATE actions this step (None = unconstrained)",
    )

    @model_validator(mode="after")
    def queue_length_matches_alerts(self) -> "Observation":
        """queue_length must equal len(alerts) for consistency."""
        if self.queue_length != len(self.alerts):
            # Auto-correct rather than raise — keeps the server robust
            object.__setattr__(self, "queue_length", len(self.alerts))
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "alerts": [
                    {
                        "id": "alert_0003_01",
                        "visible_severity": 0.85,
                        "confidence": 0.92,
                        "alert_type": "CPU",
                        "age": 3,
                    }
                ],
                "system_load": 0.72,
                "queue_length": 1,
                "time_remaining": 47,
                "episode_step": 3,
                "resource_budget": 2,
            }
        }
    }


# ---------------------------------------------------------------------------
# Action  (what the agent sends each step)
# ---------------------------------------------------------------------------

class Action(BaseModel):
    """
    Agent action targeting a single alert.

    The environment processes one Action per call to step().  Resource-
    constrained tasks (medium / hard) count INVESTIGATE actions against a
    per-step budget tracked in Observation.resource_budget.

    Attributes:
        alert_id:    ID of the alert to act upon.  Must match an id in the
                     current Observation.alerts list.
        action_type: Decision to apply:
                       INVESTIGATE — deep-dive, costs resource budget.
                       IGNORE      — dismiss as noise (zero cost).
                       ESCALATE    — route to specialist (medium cost).
                       DELAY       — keep in queue, re-evaluate next step.
        metadata:    Optional free-form context (e.g. escalation reason).
    """

    alert_id: str = Field(..., description="Target alert identifier")
    action_type: ActionType = Field(..., description="Action to perform")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Optional action context"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "alert_id": "alert_0003_01",
                "action_type": "INVESTIGATE",
                "metadata": {"reason": "high visible severity"},
            }
        }
    }


# ---------------------------------------------------------------------------
# Reward  (what step() returns alongside the next Observation)
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    """
    Scalar reward signal returned after each action.

    The dense reward is decomposed into named components so that graders,
    evaluation scripts, and debugging tools can inspect each contribution.

    Reward schedule:
        +10 : Critical alert correctly handled (INVESTIGATE or ESCALATE).
        +5  : Action prevented a future cascading failure.
        +3  : False positive correctly ignored.
        -2  : Unnecessary investigation (benign alert investigated).
        -8  : Critical alert missed (IGNORE or excessive DELAY).
        -10 : System failure triggered by accumulated unhandled alerts.

    Attributes:
        value:      Total scalar reward for this step.
        components: Breakdown mapping component name -> float contribution.
        info:       Debugging / logging extras (ground-truth reveal, etc.).
    """

    value: float = Field(..., description="Total scalar reward")

    components: Dict[str, float] = Field(
        default_factory=dict, description="Per-component reward breakdown"
    )
    info: Dict[str, Any] = Field(
        default_factory=dict, description="Debugging / ground-truth context"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "value": 10.0,
                "components": {
                    "critical_handled": 10.0,
                    "false_positive_penalty": 0.0,
                    "resource_penalty": 0.0,
                    "failure_penalty": 0.0,
                },
                "info": {
                    "alert_id": "alert_0003_01",
                    "true_severity": 0.90,
                    "action_correct": True,
                    "was_false_positive": False,
                    "was_critical": True,
                },
            }
        }
    }


# ---------------------------------------------------------------------------
# EpisodeState  (full internal state — used by state() and checkpointing)
# ---------------------------------------------------------------------------

class EpisodeState(BaseModel):
    """
    Complete internal snapshot of a running episode.

    Returned by AdaptiveAlertTriageEnv.state().  Contains both the visible
    observation AND the hidden ground-truth information — intended for
    evaluation scripts, replay, and debugging only (never exposed to the
    agent during training).

    Attributes:
        observation:       Current agent-visible observation.
        hidden_state:      Ground-truth data not exposed to the agent:
                             true_severities  — {alert_id: float}
                             correlation_groups — [[alert_id, ...], ...]
                             false_positives  — [alert_id, ...]
                             pending_failures — {alert_id: steps_until_failure}
        cumulative_reward: Total reward accumulated so far this episode.
        failures_count:    Number of system failures that have occurred.
        actions_taken:     Ordered history of every Action in this episode.
        seed:              Random seed used to initialise this episode.
    """

    observation: Observation = Field(..., description="Current agent-visible state")
    hidden_state: Dict[str, Any] = Field(
        default_factory=dict,
        description="Ground-truth data (not exposed to agent)",
    )
    cumulative_reward: float = Field(
        default=0.0, description="Total reward accumulated this episode"
    )
    failures_count: int = Field(
        default=0, ge=0, description="System failures so far"
    )
    actions_taken: List[Dict[str, Any]] = Field(
        default_factory=list, description="Full action history for this episode"
    )
    seed: Optional[int] = Field(
        None, description="Random seed for episode reproducibility"
    )

    model_config = {"arbitrary_types_allowed": True}