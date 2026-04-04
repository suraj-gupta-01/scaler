"""
Utility Functions for the Adaptive Alert Triage Environment

Provides deterministic, seed-controlled helpers for:
  - Alert generation (individual and correlated chains)
  - Severity / noise / false-positive logic
  - System-load calculation
  - Alert-queue arrival modelling
  - Action-correctness evaluation (used by graders)

All randomness flows through numpy so that a single set_seed() call at
episode start guarantees full reproducibility.
"""

import random
from typing import List, Dict, Tuple, Optional

import numpy as np

from adaptive_alert_triage.models import Alert, AlertType


# ---------------------------------------------------------------------------
# Alert-type configuration
# ---------------------------------------------------------------------------

# Each entry defines the baseline true-severity and the false-positive rate
# for that alert class.  These values were chosen to reflect realistic SOC
# distributions (SECURITY is rare but almost never a false positive; APPLICATION
# is the noisiest signal).
ALERT_TYPE_CONFIG: Dict[str, Dict[str, float]] = {
    "CPU":         {"base_severity": 0.60, "false_positive_rate": 0.15},
    "MEMORY":      {"base_severity": 0.70, "false_positive_rate": 0.20},
    "DISK":        {"base_severity": 0.50, "false_positive_rate": 0.25},
    "NETWORK":     {"base_severity": 0.65, "false_positive_rate": 0.10},
    "APPLICATION": {"base_severity": 0.75, "false_positive_rate": 0.30},
    "SECURITY":    {"base_severity": 0.90, "false_positive_rate": 0.05},
}

# Cascade chains: each sub-list is a typical multi-alert failure sequence.
# The environment uses these when generating correlated alert groups.
CORRELATION_CHAINS: List[List[str]] = [
    ["CPU",      "MEMORY",  "APPLICATION"],
    ["NETWORK",  "APPLICATION", "APPLICATION"],
    ["DISK",     "MEMORY",  "APPLICATION"],
    ["SECURITY", "NETWORK", "APPLICATION"],
    ["MEMORY",   "CPU",     "APPLICATION"],
]

# Thresholds used across the environment and graders
CRITICAL_SEVERITY_THRESHOLD: float = 0.75   # true_severity >= this → critical
CRITICAL_AGE_THRESHOLD: int = 5             # age >= this AND critical → failure


# ---------------------------------------------------------------------------
# Seed management
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """
    Set random seeds for numpy and the stdlib random module.

    Must be called before any alert-generation functions to guarantee
    reproducible episodes.

    Args:
        seed: Non-negative integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# ID helpers
# ---------------------------------------------------------------------------

def generate_alert_id(step: int, alert_index: int) -> str:
    """
    Build a deterministic, human-readable alert identifier.

    Format: ``alert_<step:04d>_<index:02d>``

    Args:
        step:        Episode step at which the alert was generated.
        alert_index: Position of this alert within the batch generated
                     at that step.

    Returns:
        Unique alert ID string, e.g. ``"alert_0007_02"``.
    """
    return f"alert_{step:04d}_{alert_index:02d}"


# ---------------------------------------------------------------------------
# Alert-type sampling
# ---------------------------------------------------------------------------

def sample_alert_type() -> AlertType:
    """
    Sample a random alert type using empirically motivated class weights.

    APPLICATION alerts are most common (25 %); SECURITY alerts are rarest
    (5 %) but carry the highest baseline severity.

    Returns:
        One of the six AlertType literals.
    """
    alert_types: List[str] = [
        "CPU", "MEMORY", "DISK", "NETWORK", "APPLICATION", "SECURITY",
    ]
    weights: List[float] = [0.20, 0.20, 0.15, 0.15, 0.25, 0.05]
    idx: int = int(np.random.choice(len(alert_types), p=weights))
    return alert_types[idx]  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Severity helpers
# ---------------------------------------------------------------------------

def calculate_true_severity(
    alert_type: AlertType,
    is_correlated: bool = False,
) -> float:
    """
    Sample ground-truth severity for a *non*-false-positive alert.

    Adds Gaussian noise (σ=0.10) around the type's baseline severity.
    Correlated alerts receive a 1.3× boost (capped at 1.0) to model the
    increased risk of cascading failures.

    Args:
        alert_type:    Category of the alert.
        is_correlated: Whether the alert belongs to a correlated chain.

    Returns:
        True severity in [0.0, 1.0].
    """
    base: float = ALERT_TYPE_CONFIG[alert_type]["base_severity"]
    noise: float = float(np.random.normal(0.0, 0.10))
    severity: float = float(np.clip(base + noise, 0.0, 1.0))
    if is_correlated:
        severity = float(min(severity * 1.3, 1.0))
    return severity


def add_observation_noise(true_severity: float, confidence: float) -> float:
    """
    Simulate partial-observability by adding confidence-weighted noise.

    Lower confidence → higher observation noise, making it harder for the
    agent to distinguish true positives from false alarms.

    Args:
        true_severity: Ground-truth severity value.
        confidence:    Sensor/detector confidence level.

    Returns:
        Noisy visible severity in [0.0, 1.0].
    """
    noise_std: float = 0.15 * (1.0 - confidence)
    noise: float = float(np.random.normal(0.0, noise_std))
    return float(np.clip(true_severity + noise, 0.0, 1.0))


# ---------------------------------------------------------------------------
# False-positive determination
# ---------------------------------------------------------------------------

def is_false_positive(alert_type: AlertType) -> bool:
    """
    Stochastically decide whether an alert is a false positive.

    Uses the per-type false-positive rate from ALERT_TYPE_CONFIG.

    Args:
        alert_type: Category of the alert.

    Returns:
        True if the alert should be treated as a false positive.
    """
    fp_rate: float = ALERT_TYPE_CONFIG[alert_type]["false_positive_rate"]
    return bool(np.random.random() < fp_rate)


# ---------------------------------------------------------------------------
# Single-alert generation
# ---------------------------------------------------------------------------

def generate_alert(
    step: int,
    alert_index: int,
    is_correlated: bool = False,
    force_critical: bool = False,
) -> Alert:
    """
    Generate a single synthetic alert with both visible and hidden attributes.

    Workflow:
      1. Sample alert type.
      2. Determine if false positive (unless force_critical=True).
      3. Set true_severity: low for FPs, high for forced-critical, otherwise
         sampled via calculate_true_severity().
      4. Sample confidence (type-dependent baseline + noise).
      5. Generate noisy visible_severity via add_observation_noise().

    Args:
        step:          Current episode step (used for ID generation).
        alert_index:   Index within this step's batch.
        is_correlated: Mark the alert as part of a correlated failure chain.
        force_critical: Override FP logic and set severity in [0.8, 1.0].

    Returns:
        Fully populated Alert object.
    """
    alert_id: str = generate_alert_id(step, alert_index)
    alert_type: AlertType = sample_alert_type()

    # False-positive logic
    is_fp: bool = is_false_positive(alert_type) and not force_critical

    # True severity
    if is_fp:
        true_severity = float(np.random.uniform(0.0, 0.30))
    elif force_critical:
        true_severity = float(np.random.uniform(0.80, 1.0))
    else:
        true_severity = calculate_true_severity(alert_type, is_correlated)

    # Confidence — inversely related to FP rate, with Gaussian jitter
    base_confidence: float = 1.0 - ALERT_TYPE_CONFIG[alert_type]["false_positive_rate"]
    confidence: float = float(
        np.clip(base_confidence + np.random.normal(0.0, 0.10), 0.0, 1.0)
    )

    # Observable severity (noisy)
    visible_severity: float = add_observation_noise(true_severity, confidence)

    return Alert(
        id=alert_id,
        visible_severity=visible_severity,
        confidence=confidence,
        alert_type=alert_type,
        age=0,
        true_severity=true_severity,
        is_correlated=is_correlated,
        metadata={
            "false_positive": is_fp,
            "generated_at_step": step,
        },
    )


# ---------------------------------------------------------------------------
# Correlated-alert chain generation
# ---------------------------------------------------------------------------

def generate_correlated_alerts(step: int, num_alerts: int = 3) -> List[Alert]:
    """
    Generate a sequence of alerts that share a hidden root cause.

    Simulates cascading failures (e.g. high CPU → memory pressure →
    application crash).  Severity escalates along the chain so that later
    members are more dangerous than the trigger.

    The IDs of all alerts in the chain should be tracked in
    ``AdaptiveAlertTriageEnv.correlation_groups`` so the hard-task grader
    can reward root-cause identification.

    Args:
        step:       Current episode step (used for ID generation).
        num_alerts: Number of alerts to produce (1 – len(chain), capped
                    at 3 by default to match a typical failure chain).

    Returns:
        List of correlated Alert objects in causal order.
    """
    chain: List[str] = random.choice(CORRELATION_CHAINS)[:num_alerts]
    alerts: List[Alert] = []

    for i, alert_type in enumerate(chain):
        alert_id = generate_alert_id(step, i)

        # Severity increases along the chain
        base_sev: float = 0.60 + i * 0.15
        true_severity: float = float(
            np.clip(base_sev + np.random.normal(0.0, 0.05), 0.0, 1.0)
        )
        confidence: float = float(
            np.clip(0.80 + np.random.normal(0.0, 0.10), 0.0, 1.0)
        )
        visible_severity: float = add_observation_noise(true_severity, confidence)

        alert = Alert(
            id=alert_id,
            visible_severity=visible_severity,
            confidence=confidence,
            alert_type=alert_type,  # type: ignore[arg-type]
            age=0,
            true_severity=true_severity,
            is_correlated=True,
            metadata={
                "false_positive": False,
                "correlation_chain": chain,
                "chain_position": i,
                "generated_at_step": step,
            },
        )
        alerts.append(alert)

    return alerts


# ---------------------------------------------------------------------------
# System-load calculation
# ---------------------------------------------------------------------------

def calculate_system_load(num_active_alerts: int, base_load: float = 0.30) -> float:
    """
    Estimate current system resource utilisation.

    Each unresolved alert contributes 0.05 to load, plus a small Gaussian
    jitter to model background variability.

    Args:
        num_active_alerts: Number of alerts currently in the queue.
        base_load:         Steady-state load with no active alerts.

    Returns:
        System load in [0.0, 1.0].
    """
    alert_contribution: float = num_active_alerts * 0.05
    jitter: float = float(np.random.normal(0.0, 0.02))
    return float(np.clip(base_load + alert_contribution + jitter, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Alert-arrival modelling
# ---------------------------------------------------------------------------

def should_generate_new_alerts(step: int, current_queue: int) -> bool:
    """
    Decide whether the environment should produce new alerts this step.

    Uses a Poisson-inspired arrival model with back-pressure: a growing queue
    reduces arrival probability, preventing runaway queue growth and forcing
    the agent to drain alerts before new ones overwhelm the system.

    Args:
        step:          Current episode step (unused but available for
                       future step-dependent patterns).
        current_queue: Number of alerts already in the queue.

    Returns:
        True if new alerts should be generated.
    """
    base_prob: float = 0.70
    # Back-pressure: every queued alert reduces arrival probability by 0.05,
    # capped at a maximum reduction of 0.40.
    queue_penalty: float = min(current_queue * 0.05, 0.40)
    arrival_prob: float = base_prob - queue_penalty
    return bool(np.random.random() < arrival_prob)


def sample_num_new_alerts() -> int:
    """
    Sample the number of alerts to generate this step (Poisson, λ=2).

    Capped at 5 to prevent single-step queue explosions.

    Returns:
        Integer in [0, 5].
    """
    return int(min(int(np.random.poisson(2)), 5))


# ---------------------------------------------------------------------------
# Alert criticality
# ---------------------------------------------------------------------------

def is_critical_alert(alert: Alert, threshold: float = CRITICAL_SEVERITY_THRESHOLD) -> bool:
    """
    Determine whether an alert is critical based on its *true* severity.

    Note: the agent cannot observe true_severity directly; this function is
    used internally by the reward calculator and failure checker.

    Args:
        alert:     The alert to evaluate.
        threshold: Minimum true_severity for criticality (default 0.75).

    Returns:
        True if the alert's true severity meets or exceeds the threshold.
    """
    return alert.true_severity >= threshold


# ---------------------------------------------------------------------------
# Action-correctness evaluation  (used by task graders)
# ---------------------------------------------------------------------------

def calculate_action_correctness(
    action_type: str,
    alert: Alert,
    resource_constrained: bool = False,
) -> Tuple[bool, str]:
    """
    Evaluate whether an action matches the ground-truth optimal policy.

    Decision logic:
      - Critical alert  → INVESTIGATE or ESCALATE is correct.
      - False positive  → IGNORE is correct; anything else wastes resources.
      - Medium severity → INVESTIGATE is correct; DELAY is acceptable when
                          resource-constrained.

    This is intentionally strict for critical alerts (the agent should never
    ignore or indefinitely delay them) and lenient for medium-severity alerts
    (a delayed medium alert is acceptable if the budget is exhausted).

    Args:
        action_type:          The action taken ("INVESTIGATE", "IGNORE", etc.).
        alert:                Alert being evaluated (with true hidden fields).
        resource_constrained: Whether the task enforces a per-step action budget.

    Returns:
        Tuple of (is_correct: bool, reason: str).
    """
    is_critical: bool = is_critical_alert(alert)
    is_fp: bool = bool(alert.metadata.get("false_positive", False))

    if is_critical:
        if action_type in ("INVESTIGATE", "ESCALATE"):
            return True, "Correctly handled critical alert"
        return False, "Missed critical alert — should INVESTIGATE or ESCALATE"

    if is_fp:
        if action_type == "IGNORE":
            return True, "Correctly ignored false positive"
        return False, "Wasted resources on false positive"

    # Medium-severity alert
    if action_type == "INVESTIGATE":
        return True, "Investigated medium-severity alert"
    if action_type == "DELAY" and resource_constrained:
        return True, "Delayed medium alert under resource constraints (acceptable)"
    if action_type == "ESCALATE":
        return True, "Escalated medium alert (acceptable)"
    return True, "Acceptable action for medium-severity alert"