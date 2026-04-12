#!/usr/bin/env python
"""
OpenEnv Validation CLI Tool

Usage:
    openenv validate                          # via registered entry point (pyproject.toml)
    python -m adaptive_alert_triage.validate  # direct module invocation
    python validate.py                        # from repo root

Validates that the Adaptive Alert Triage environment meets the full OpenEnv
interface specification:
  1. Typed Observation, Action, and Reward Pydantic models
  2. step(action) → returns (observation, reward, done, info)
  3. reset() → returns initial observation
  4. state() → returns current EpisodeState
  5. openenv.yaml with required metadata

Exit codes:
    0 — all checks passed
    1 — one or more checks failed
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

# ---------------------------------------------------------------------------
# Make sure the package is importable regardless of CWD.
# The entry-point may be called from any directory (e.g. the repo root),
# so we add both the src/ directory and the repo root to sys.path.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()

# src/ directory (where the package lives)
_SRC = _HERE.parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# repo root (where openenv.yaml lives)
_REPO_ROOT = _SRC.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from adaptive_alert_triage.env import AdaptiveAlertTriageEnv
from adaptive_alert_triage.models import (
    Action,
    Observation,
    Reward,
    Alert,
    EpisodeState,
)


class OpenEnvValidator:
    """Validates OpenEnv compliance of the environment."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.checks_passed: List[str] = []
        self.checks_failed: List[Tuple[str, str]] = []

    def log(self, message: str, level: str = "INFO"):
        if self.verbose:
            print(f"[{level}] {message}")

    def check(self, name: str, condition: bool, details: str = "") -> bool:
        if condition:
            self.checks_passed.append(name)
            self.log(f"[PASS] {name}", "PASS")
            if details:
                self.log(f"  {details}", "INFO")
            return True
        else:
            self.checks_failed.append((name, details))
            self.log(f"[FAIL] {name}", "FAIL")
            if details:
                self.log(f"  {details}", "ERROR")
            return False

    def validate_pydantic_models(self) -> bool:
        self.log("\n=== Validating Pydantic Models ===", "INFO")
        from pydantic import BaseModel
        checks = [
            ("Observation is Pydantic BaseModel", issubclass(Observation, BaseModel)),
            ("Action is Pydantic BaseModel",      issubclass(Action, BaseModel)),
            ("Reward is Pydantic BaseModel",       issubclass(Reward, BaseModel)),
            ("EpisodeState is Pydantic BaseModel", issubclass(EpisodeState, BaseModel)),
            ("Alert is Pydantic BaseModel",        issubclass(Alert, BaseModel)),
        ]
        return all(self.check(name, cond) for name, cond in checks)

    def validate_required_fields(self) -> bool:
        self.log("\n=== Validating Model Fields ===", "INFO")
        checks = [
            (
                "Observation has required fields",
                {"alerts", "system_load", "queue_length", "time_remaining", "episode_step"}.issubset(
                    set(Observation.model_fields.keys())
                ),
                f"Fields: {', '.join(sorted(Observation.model_fields.keys()))}",
            ),
            (
                "Action has required fields",
                {"alert_id", "action_type"}.issubset(set(Action.model_fields.keys())),
                f"Fields: {', '.join(sorted(Action.model_fields.keys()))}",
            ),
            (
                "Reward has required fields",
                {"value", "components"}.issubset(set(Reward.model_fields.keys())),
                f"Fields: {', '.join(sorted(Reward.model_fields.keys()))}",
            ),
        ]
        return all(self.check(name, cond, details) for name, cond, details in checks)

    def validate_serialization(self) -> bool:
        self.log("\n=== Validating Serialization ===", "INFO")
        try:
            action   = Action(alert_id="test", action_type="INVESTIGATE")
            restored = Action.model_validate_json(action.model_dump_json())
            action_ok = restored.alert_id == action.alert_id
            self.check("Action serialization round-trip", action_ok)

            reward   = Reward(value=0.5, components={"test": 0.5})
            restored = Reward.model_validate_json(reward.model_dump_json())
            reward_ok = restored.value == reward.value
            self.check("Reward serialization round-trip", reward_ok)

            return action_ok and reward_ok
        except Exception as e:
            self.check("Serialization", False, str(e))
            return False

    def validate_reset_method(self) -> bool:
        self.log("\n=== Validating reset() Method ===", "INFO")
        try:
            env = AdaptiveAlertTriageEnv(task_id="easy", seed=42)

            has_method = hasattr(env, "reset")
            self.check("reset() method exists", has_method)
            if not has_method:
                return False

            obs = env.reset()
            returns_obs = isinstance(obs, Observation)
            self.check("reset() returns Observation", returns_obs)

            env2 = AdaptiveAlertTriageEnv(task_id="easy")
            obs2 = env2.reset(seed=42)
            reproducible = len(env.alerts) == len(env2.alerts)
            self.check("reset() is reproducible with seed", reproducible)

            return has_method and returns_obs and reproducible
        except Exception as e:
            self.check("reset() validation", False, str(e))
            return False

    def validate_step_method(self) -> bool:
        self.log("\n=== Validating step() Method ===", "INFO")
        try:
            env = AdaptiveAlertTriageEnv(task_id="easy", seed=42)
            obs = env.reset()

            has_method = hasattr(env, "step")
            self.check("step() method exists", has_method)
            if not has_method or not obs.alerts:
                return False

            action = Action(alert_id=obs.alerts[0].id, action_type="INVESTIGATE")
            result = env.step(action)

            is_tuple = isinstance(result, tuple)
            self.check("step() returns tuple", is_tuple)
            if not is_tuple:
                return False

            correct_len = len(result) == 4
            self.check("step() returns 4-tuple", correct_len, f"Got {len(result)} elements")
            if not correct_len:
                return False

            next_obs, reward, done, info = result

            obs_ok    = isinstance(next_obs, Observation)
            reward_ok = isinstance(reward, Reward)
            done_ok   = isinstance(done, bool)
            info_ok   = isinstance(info, dict)

            self.check("step() returns Observation",  obs_ok)
            self.check("step() returns Reward",        reward_ok)
            self.check("step() returns bool (done)",   done_ok)
            self.check("step() returns dict (info)",   info_ok)

            if info_ok:
                self.check(
                    "info contains 'processed_alerts'",
                    "processed_alerts" in info,
                    f"Keys: {', '.join(sorted(info.keys()))}",
                )
                self.check("info contains 'correlation_groups'", "correlation_groups" in info)

            return obs_ok and reward_ok and done_ok and info_ok
        except Exception as e:
            self.check("step() validation", False, str(e))
            return False

    def validate_state_method(self) -> bool:
        self.log("\n=== Validating state() Method ===", "INFO")
        try:
            env = AdaptiveAlertTriageEnv(task_id="easy", seed=42)
            env.reset()

            has_method = hasattr(env, "state")
            self.check("state() method exists", has_method)
            if not has_method:
                return False

            state = env.state()
            is_episode_state = isinstance(state, EpisodeState)
            self.check("state() returns EpisodeState", is_episode_state)
            if not is_episode_state:
                return False

            has_obs = hasattr(state, "observation") and isinstance(state.observation, Observation)
            self.check("EpisodeState has observation (Observation)", has_obs)

            has_hidden = hasattr(state, "hidden_state") and isinstance(state.hidden_state, dict)
            self.check("EpisodeState has hidden_state (dict)", has_hidden)

            if has_hidden:
                self.check("hidden_state contains true_severities",   "true_severities"   in state.hidden_state)
                self.check("hidden_state contains correlation_groups", "correlation_groups" in state.hidden_state)

            self.check("EpisodeState has cumulative_reward", hasattr(state, "cumulative_reward"))

            return is_episode_state and has_obs and has_hidden
        except Exception as e:
            self.check("state() validation", False, str(e))
            return False

    def validate_openenv_yaml(self) -> bool:
        self.log("\n=== Validating openenv.yaml ===", "INFO")
        try:
            # Search for openenv.yaml relative to the repo root (not CWD)
            candidates = [
                Path("openenv.yaml"),          # CWD (most common)
                _REPO_ROOT / "openenv.yaml",   # repo root
                Path(__file__).parent / "openenv.yaml",  # package dir
            ]
            yaml_path = next((p for p in candidates if p.exists()), None)

            exists = yaml_path is not None
            self.check("openenv.yaml exists", exists, str(yaml_path or candidates[0].absolute()))
            if not exists:
                return False

            with open(yaml_path) as f:
                data = yaml.safe_load(f)

            is_dict = isinstance(data, dict)
            self.check("openenv.yaml is valid YAML dict", is_dict)
            if not is_dict:
                return False

            required_fields = {
                ("name",        "Environment name"),
                ("version",     "Version string"),
                ("description", "Description"),
                ("tasks",       "Task definitions"),
            }
            all_present = True
            for field, description in required_fields:
                present = field in data
                self.check(f"'{field}' present ({description})", present)
                all_present = all_present and present

            if "tasks" in data:
                tasks   = data["tasks"]
                is_list = isinstance(tasks, list)
                self.check("tasks is a list", is_list, f"Got {type(tasks)}")
                if is_list:
                    self.check("tasks list is not empty", len(tasks) > 0, f"{len(tasks)} tasks defined")
                    all_have_ids = all("id" in task for task in tasks)
                    task_ids = [task.get("id", "?") for task in tasks]
                    self.check("all tasks have 'id'", all_have_ids, f"IDs: {', '.join(task_ids)}")

            has_config = "config" in data
            self.check("'config' section present", has_config)

            if has_config and "actions" in data["config"]:
                expected = {"INVESTIGATE", "IGNORE", "ESCALATE", "DELAY"}
                found    = set(data["config"]["actions"])
                self.check(
                    "config.actions includes all required actions",
                    expected.issubset(found),
                    f"Found: {', '.join(sorted(found))}",
                )

            return all_present
        except Exception as e:
            self.check("openenv.yaml validation", False, str(e))
            return False

    def validate_all_tasks(self) -> bool:
        self.log("\n=== Validating All Tasks ===", "INFO")
        try:
            all_ok = True
            for task_id in ["easy", "medium", "hard"]:
                try:
                    env = AdaptiveAlertTriageEnv(task_id=task_id, seed=42)
                    obs = env.reset()
                    obs_ok = isinstance(obs, Observation)

                    if obs.alerts:
                        action = Action(alert_id=obs.alerts[0].id, action_type="INVESTIGATE")
                        next_obs, reward, done, info = env.step(action)
                        step_ok = (
                            isinstance(next_obs, Observation)
                            and isinstance(reward, Reward)
                            and isinstance(done, bool)
                            and isinstance(info, dict)
                        )
                    else:
                        step_ok = True

                    state_ok = isinstance(env.state(), EpisodeState)
                    task_ok  = obs_ok and step_ok and state_ok
                    self.check(f"Task '{task_id}' is OpenEnv compliant", task_ok)
                    all_ok = all_ok and task_ok
                except Exception as e:
                    self.check(f"Task '{task_id}' is OpenEnv compliant", False, str(e))
                    all_ok = False
            return all_ok
        except Exception as e:
            self.check("Task validation", False, str(e))
            return False

    def run_all_checks(self) -> bool:
        self.log("=" * 60)
        self.log("OpenEnv Compliance Validator", "INFO")
        self.log("=" * 60)

        results = [
            self.validate_pydantic_models(),
            self.validate_required_fields(),
            self.validate_serialization(),
            self.validate_reset_method(),
            self.validate_step_method(),
            self.validate_state_method(),
            self.validate_openenv_yaml(),
            self.validate_all_tasks(),
        ]

        self.log("\n" + "=" * 60, "INFO")
        self.log("VALIDATION SUMMARY", "INFO")
        self.log("=" * 60, "INFO")

        total_passed = len(self.checks_passed)
        total_failed = len(self.checks_failed)
        total_checks = total_passed + total_failed

        self.log(f"Passed: {total_passed}/{total_checks}", "INFO")

        if self.checks_failed:
            self.log(f"Failed: {total_failed}/{total_checks}", "ERROR")
            for name, details in self.checks_failed:
                self.log(f"  - {name}", "ERROR")
                if details:
                    self.log(f"    {details}", "ERROR")
        else:
            self.log("All checks passed!", "PASS")

        self.log("=" * 60 + "\n", "INFO")
        return len(self.checks_failed) == 0


def main():
    """
    Entry point for the `openenv validate` CLI command.

    Registered in pyproject.toml as:
        openenv = "adaptive_alert_triage.validate:main"

    This means `pip install -e .` makes `openenv validate` available system-wide
    (the `validate` sub-argument is ignored by argparse; the script always
    runs the full compliance suite).
    """
    # Accept (and ignore) an optional positional argument so that
    # `openenv validate` doesn't fail with "unrecognised argument: validate".
    import argparse
    parser = argparse.ArgumentParser(
        prog="openenv",
        description="OpenEnv compliance validator for Adaptive Alert Triage",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="validate",
        choices=["validate"],
        help="Sub-command (only 'validate' is supported)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress per-check output; only print the final summary",
    )
    args = parser.parse_args()

    validator = OpenEnvValidator(verbose=not args.quiet)
    success   = validator.run_all_checks()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()