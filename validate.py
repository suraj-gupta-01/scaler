#!/usr/bin/env python
"""
OpenEnv Validation CLI Tool

Usage:
    python -m src.adaptive_alert_triage.validate
    openenv validate  (if registered as entry point)

Validates that the Adaptive Alert Triage environment meets the full OpenEnv
interface specification:
  1. Typed Observation, Action, and Reward Pydantic models
  2. step(action) → returns (observation, reward, done, info)
  3. reset() → returns initial observation
  4. state() → returns current state
  5. openenv.yaml with metadata
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
import yaml

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
        self.checks_passed = []
        self.checks_failed = []

    def log(self, message: str, level: str = "INFO"):
        """Log a message with level."""
        if self.verbose:
            print(f"[{level}] {message}")

    def check(self, name: str, condition: bool, details: str = "") -> bool:
        """Record a check result."""
        if condition:
            self.checks_passed.append(name)
            self.log(f"✓ {name}", "PASS")
            if details:
                self.log(f"  {details}", "INFO")
            return True
        else:
            self.checks_failed.append((name, details))
            self.log(f"✗ {name}", "FAIL")
            if details:
                self.log(f"  {details}", "ERROR")
            return False

    def validate_pydantic_models(self) -> bool:
        """1. Check that models are Pydantic BaseModels."""
        self.log("\n=== Validating Pydantic Models ===", "INFO")
        
        from pydantic import BaseModel
        
        checks = [
            ("Observation is Pydantic BaseModel", issubclass(Observation, BaseModel)),
            ("Action is Pydantic BaseModel", issubclass(Action, BaseModel)),
            ("Reward is Pydantic BaseModel", issubclass(Reward, BaseModel)),
            ("EpisodeState is Pydantic BaseModel", issubclass(EpisodeState, BaseModel)),
            ("Alert is Pydantic BaseModel", issubclass(Alert, BaseModel)),
        ]
        
        return all(self.check(name, cond) for name, cond in checks)

    def validate_required_fields(self) -> bool:
        """Check that models have required fields."""
        self.log("\n=== Validating Model Fields ===", "INFO")
        
        checks = [
            (
                "Observation has required fields",
                {"alerts", "system_load", "queue_length", "time_remaining", "episode_step"}.issubset(
                    set(Observation.model_fields.keys())
                ),
                f"Fields: {', '.join(sorted(Observation.model_fields.keys()))}"
            ),
            (
                "Action has required fields",
                {"alert_id", "action_type"}.issubset(set(Action.model_fields.keys())),
                f"Fields: {', '.join(sorted(Action.model_fields.keys()))}"
            ),
            (
                "Reward has required fields",
                {"value", "components"}.issubset(set(Reward.model_fields.keys())),
                f"Fields: {', '.join(sorted(Reward.model_fields.keys()))}"
            ),
        ]
        
        return all(self.check(name, cond, details) for name, cond, details in checks)

    def validate_serialization(self) -> bool:
        """Check that models can be serialized/deserialized."""
        self.log("\n=== Validating Serialization ===", "INFO")
        
        try:
            # Test Action serialization
            action = Action(alert_id="test", action_type="INVESTIGATE")
            json_str = action.model_dump_json()
            restored = Action.model_validate_json(json_str)
            action_ok = restored.alert_id == action.alert_id
            self.check("Action serialization round-trip", action_ok)
            
            # Test Reward serialization
            reward = Reward(value=10.0, components={"test": 10.0})
            json_str = reward.model_dump_json()
            restored = Reward.model_validate_json(json_str)
            reward_ok = restored.value == reward.value
            self.check("Reward serialization round-trip", reward_ok)
            
            return action_ok and reward_ok
        except Exception as e:
            self.check("Serialization", False, str(e))
            return False

    def validate_reset_method(self) -> bool:
        """2. Check reset() method."""
        self.log("\n=== Validating reset() Method ===", "INFO")
        
        try:
            env = AdaptiveAlertTriageEnv(task_id="easy", seed=42)
            
            # Check method exists
            has_method = hasattr(env, "reset")
            self.check("reset() method exists", has_method)
            if not has_method:
                return False
            
            # Check return type
            obs = env.reset()
            returns_observation = isinstance(obs, Observation)
            self.check("reset() returns Observation", returns_observation)
            
            # Check reproducibility
            env2 = AdaptiveAlertTriageEnv(task_id="easy")
            obs2 = env2.reset(seed=42)
            is_reproducible = len(env.alerts) == len(env2.alerts)
            self.check("reset() is reproducible with seed", is_reproducible)
            
            return has_method and returns_observation and is_reproducible
        except Exception as e:
            self.check("reset() validation", False, str(e))
            return False

    def validate_step_method(self) -> bool:
        """3. Check step(action) method."""
        self.log("\n=== Validating step() Method ===", "INFO")
        
        try:
            env = AdaptiveAlertTriageEnv(task_id="easy", seed=42)
            obs = env.reset()
            
            # Check method exists
            has_method = hasattr(env, "step")
            self.check("step() method exists", has_method)
            if not has_method or not obs.alerts:
                return False
            
            # Take a step
            action = Action(alert_id=obs.alerts[0].id, action_type="INVESTIGATE")
            result = env.step(action)
            
            # Check return type is tuple
            is_tuple = isinstance(result, tuple)
            self.check("step() returns tuple", is_tuple)
            
            if not is_tuple:
                return False
            
            # Check tuple length
            correct_length = len(result) == 4
            self.check("step() returns 4-tuple", correct_length, f"Got {len(result)} elements")
            
            if not correct_length:
                return False
            
            next_obs, reward, done, info = result
            
            # Check return types
            obs_ok = isinstance(next_obs, Observation)
            self.check("step() returns Observation", obs_ok)
            
            reward_ok = isinstance(reward, Reward)
            self.check("step() returns Reward", reward_ok)
            
            done_ok = isinstance(done, bool)
            self.check("step() returns bool (done)", done_ok)
            
            info_ok = isinstance(info, dict)
            self.check("step() returns dict (info)", info_ok)
            
            # Check info contents
            if info_ok:
                has_processed_alerts = "processed_alerts" in info
                self.check(
                    "info contains 'processed_alerts'",
                    has_processed_alerts,
                    f"Keys: {', '.join(sorted(info.keys()))}"
                )
                
                has_correlation_groups = "correlation_groups" in info
                self.check("info contains 'correlation_groups'", has_correlation_groups)
            
            return obs_ok and reward_ok and done_ok and info_ok
        except Exception as e:
            self.check("step() validation", False, str(e))
            return False

    def validate_state_method(self) -> bool:
        """4. Check state() method."""
        self.log("\n=== Validating state() Method ===", "INFO")
        
        try:
            env = AdaptiveAlertTriageEnv(task_id="easy", seed=42)
            env.reset()
            
            # Check method exists
            has_method = hasattr(env, "state")
            self.check("state() method exists", has_method)
            if not has_method:
                return False
            
            # Get state
            state = env.state()
            
            # Check return type
            is_episode_state = isinstance(state, EpisodeState)
            self.check("state() returns EpisodeState", is_episode_state)
            
            if not is_episode_state:
                return False
            
            # Check required attributes
            has_observation = hasattr(state, "observation") and isinstance(state.observation, Observation)
            self.check("EpisodeState has observation (Observation)", has_observation)
            
            has_hidden_state = hasattr(state, "hidden_state") and isinstance(state.hidden_state, dict)
            self.check("EpisodeState has hidden_state (dict)", has_hidden_state)
            
            if has_hidden_state:
                has_true_severities = "true_severities" in state.hidden_state
                self.check("hidden_state contains true_severities", has_true_severities)
                
                has_correlation_groups = "correlation_groups" in state.hidden_state
                self.check("hidden_state contains correlation_groups", has_correlation_groups)
            
            has_cumulative_reward = hasattr(state, "cumulative_reward")
            self.check("EpisodeState has cumulative_reward", has_cumulative_reward)
            
            return is_episode_state and has_observation and has_hidden_state
        except Exception as e:
            self.check("state() validation", False, str(e))
            return False

    def validate_openenv_yaml(self) -> bool:
        """5. Check openenv.yaml metadata."""
        self.log("\n=== Validating openenv.yaml ===", "INFO")
        
        try:
            yaml_path = Path("openenv.yaml")
            
            # Check file exists
            exists = yaml_path.exists()
            self.check("openenv.yaml exists", exists, str(yaml_path.absolute()))
            
            if not exists:
                return False
            
            # Check valid YAML
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
            
            is_dict = isinstance(data, dict)
            self.check("openenv.yaml is valid YAML dict", is_dict)
            
            if not is_dict:
                return False
            
            # Check required fields
            required_fields = {
                ("name", "Environment name"),
                ("version", "Version string"),
                ("description", "Description"),
                ("tasks", "Task definitions"),
            }
            
            all_present = True
            for field, description in required_fields:
                present = field in data
                self.check(f"'{field}' present ({description})", present)
                all_present = all_present and present
            
            # Check tasks structure
            if "tasks" in data:
                tasks = data["tasks"]
                is_list = isinstance(tasks, list)
                self.check("tasks is a list", is_list, f"Got {type(tasks)}")
                
                if is_list:
                    has_tasks = len(tasks) > 0
                    self.check("tasks list is not empty", has_tasks, f"{len(tasks)} tasks defined")
                    
                    # Check each task has ID
                    all_have_ids = all("id" in task for task in tasks)
                    task_ids = [task.get("id", "?") for task in tasks]
                    self.check("all tasks have 'id'", all_have_ids, f"IDs: {', '.join(task_ids)}")
            
            # Check config section
            has_config = "config" in data
            self.check("'config' section present", has_config)
            
            if has_config and "actions" in data["config"]:
                expected_actions = {"INVESTIGATE", "IGNORE", "ESCALATE", "DELAY"}
                yaml_actions = set(data["config"]["actions"])
                has_all_actions = expected_actions.issubset(yaml_actions)
                self.check("config.actions includes all required actions", has_all_actions,
                          f"Found: {', '.join(sorted(yaml_actions))}")
            
            return all_present
        except Exception as e:
            self.check("openenv.yaml validation", False, str(e))
            return False

    def validate_all_tasks(self) -> bool:
        """Verify all tasks work correctly."""
        self.log("\n=== Validating All Tasks ===", "INFO")
        
        try:
            all_ok = True
            for task_id in ["easy", "medium", "hard"]:
                try:
                    env = AdaptiveAlertTriageEnv(task_id=task_id, seed=42)
                    obs = env.reset()
                    
                    # Verify structure
                    obs_ok = isinstance(obs, Observation)
                    
                    # Take one step
                    if obs.alerts:
                        action = Action(alert_id=obs.alerts[0].id, action_type="INVESTIGATE")
                        next_obs, reward, done, info = env.step(action)
                        
                        step_ok = (
                            isinstance(next_obs, Observation) and
                            isinstance(reward, Reward) and
                            isinstance(done, bool) and
                            isinstance(info, dict)
                        )
                    else:
                        step_ok = True
                    
                    # Get state
                    state_ok = isinstance(env.state(), EpisodeState)
                    
                    task_ok = obs_ok and step_ok and state_ok
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
        """Run all validation checks."""
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
        
        # Print summary
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
            self.log("All checks passed! ✓", "PASS")
        
        self.log("=" * 60 + "\n", "INFO")
        
        return len(self.checks_failed) == 0


def main():
    """Entry point for CLI."""
    validator = OpenEnvValidator(verbose=True)
    success = validator.run_all_checks()
    
    # Return appropriate exit code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()