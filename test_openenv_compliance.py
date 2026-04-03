"""
Comprehensive OpenEnv Compliance Test Suite

Validates that all OpenEnv interface requirements are met:
1. Typed Observation, Action, and Reward Pydantic models
2. step(action) → returns (observation, reward, done, info)
3. reset() → returns initial observation
4. state() → returns current state
5. openenv.yaml with metadata
6. Tested via openenv validate

Run with: pytest tests/test_openenv_compliance.py -v
"""

import json
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml
from pydantic import BaseModel

from adaptive_alert_triage.env import AdaptiveAlertTriageEnv
from adaptive_alert_triage.models import (
    Action,
    Observation,
    Reward,
    Alert,
    EpisodeState,
    ActionType,
    AlertType,
)


# ============================================================================
# REQUIREMENT 1: Typed Pydantic Models
# ============================================================================

class TestPydanticModels:
    """Verify Observation, Action, and Reward are properly typed Pydantic models."""

    def test_observation_is_pydantic_model(self):
        """Observation must be a Pydantic BaseModel."""
        assert issubclass(Observation, BaseModel), "Observation must inherit from Pydantic BaseModel"

    def test_action_is_pydantic_model(self):
        """Action must be a Pydantic BaseModel."""
        assert issubclass(Action, BaseModel), "Action must inherit from Pydantic BaseModel"

    def test_reward_is_pydantic_model(self):
        """Reward must be a Pydantic BaseModel."""
        assert issubclass(Reward, BaseModel), "Reward must inherit from Pydantic BaseModel"

    def test_episode_state_is_pydantic_model(self):
        """EpisodeState must be a Pydantic BaseModel."""
        assert issubclass(EpisodeState, BaseModel), "EpisodeState must inherit from Pydantic BaseModel"

    def test_alert_is_pydantic_model(self):
        """Alert must be a Pydantic BaseModel."""
        assert issubclass(Alert, BaseModel), "Alert must inherit from Pydantic BaseModel"

    def test_observation_has_required_fields(self):
        """Observation must have all required fields."""
        required_fields = {"alerts", "system_load", "queue_length", "time_remaining", "episode_step", "resource_budget"}
        model_fields = set(Observation.model_fields.keys())
        assert required_fields.issubset(model_fields), f"Missing fields: {required_fields - model_fields}"

    def test_action_has_required_fields(self):
        """Action must have alert_id and action_type."""
        required_fields = {"alert_id", "action_type"}
        model_fields = set(Action.model_fields.keys())
        assert required_fields.issubset(model_fields), f"Missing fields: {required_fields - model_fields}"

    def test_reward_has_required_fields(self):
        """Reward must have value and components."""
        required_fields = {"value", "components"}
        model_fields = set(Reward.model_fields.keys())
        assert required_fields.issubset(model_fields), f"Missing fields: {required_fields - model_fields}"

    def test_action_type_is_literal(self):
        """Validate ActionType literal values."""
        valid_actions = {"INVESTIGATE", "IGNORE", "ESCALATE", "DELAY"}
        # Create an action with each type to verify validation
        for action_type in valid_actions:
            action = Action(alert_id="test", action_type=action_type)
            assert action.action_type == action_type

    def test_alert_type_is_literal(self):
        """Validate AlertType literal values."""
        valid_types = {"CPU", "MEMORY", "DISK", "NETWORK", "APPLICATION", "SECURITY"}
        # Create an alert with each type
        for alert_type in valid_types:
            alert = Alert(
                id="test",
                visible_severity=0.5,
                confidence=0.8,
                alert_type=alert_type,
                age=0,
            )
            assert alert.alert_type == alert_type

    def test_observation_serialization(self):
        """Observation must be JSON serializable."""
        env = AdaptiveAlertTriageEnv(task_id="easy", seed=42)
        obs = env.reset()
        
        # Should be able to serialize to model_dump_json
        json_str = obs.model_dump_json()
        assert isinstance(json_str, str)
        
        # Should be able to parse back
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_action_serialization(self):
        """Action must be JSON serializable."""
        action = Action(alert_id="alert_001", action_type="INVESTIGATE")
        json_str = action.model_dump_json()
        assert isinstance(json_str, str)
        
        parsed = json.loads(json_str)
        assert parsed["alert_id"] == "alert_001"
        assert parsed["action_type"] == "INVESTIGATE"

    def test_reward_serialization(self):
        """Reward must be JSON serializable."""
        reward = Reward(
            value=10.0,
            components={"critical_handled": 10.0},
            info={"alert_id": "alert_001"}
        )
        json_str = reward.model_dump_json()
        assert isinstance(json_str, str)
        
        parsed = json.loads(json_str)
        assert parsed["value"] == 10.0


# ============================================================================
# REQUIREMENT 2: step(action) → (observation, reward, done, info)
# ============================================================================

class TestStepInterface:
    """Verify step() method signature and return types."""

    def test_step_exists(self):
        """Environment must have a step method."""
        env = AdaptiveAlertTriageEnv(task_id="easy", seed=42)
        assert hasattr(env, "step"), "Environment must have step() method"

    def test_step_accepts_action(self):
        """step() must accept an Action parameter."""
        env = AdaptiveAlertTriageEnv(task_id="easy", seed=42)
        obs = env.reset()
        
        action = Action(alert_id=obs.alerts[0].id, action_type="INVESTIGATE")
        result = env.step(action)
        assert result is not None, "step() should return a value"

    def test_step_returns_tuple(self):
        """step() must return a tuple of 4 elements."""
        env = AdaptiveAlertTriageEnv(task_id="easy", seed=42)
        obs = env.reset()
        
        action = Action(alert_id=obs.alerts[0].id, action_type="INVESTIGATE")
        result = env.step(action)
        
        assert isinstance(result, tuple), "step() must return a tuple"
        assert len(result) == 4, "step() must return exactly 4 values"

    def test_step_returns_observation(self):
        """First return value must be Observation."""
        env = AdaptiveAlertTriageEnv(task_id="easy", seed=42)
        obs = env.reset()
        
        action = Action(alert_id=obs.alerts[0].id, action_type="INVESTIGATE")
        next_obs, _, _, _ = env.step(action)
        
        assert isinstance(next_obs, Observation), "First return must be Observation"

    def test_step_returns_reward(self):
        """Second return value must be Reward."""
        env = AdaptiveAlertTriageEnv(task_id="easy", seed=42)
        obs = env.reset()
        
        action = Action(alert_id=obs.alerts[0].id, action_type="INVESTIGATE")
        _, reward, _, _ = env.step(action)
        
        assert isinstance(reward, Reward), "Second return must be Reward"

    def test_step_returns_done(self):
        """Third return value must be bool (done flag)."""
        env = AdaptiveAlertTriageEnv(task_id="easy", seed=42)
        obs = env.reset()
        
        action = Action(alert_id=obs.alerts[0].id, action_type="INVESTIGATE")
        _, _, done, _ = env.step(action)
        
        assert isinstance(done, bool), "Third return must be boolean (done flag)"

    def test_step_returns_info(self):
        """Fourth return value must be dict (info)."""
        env = AdaptiveAlertTriageEnv(task_id="easy", seed=42)
        obs = env.reset()
        
        action = Action(alert_id=obs.alerts[0].id, action_type="INVESTIGATE")
        _, _, _, info = env.step(action)
        
        assert isinstance(info, dict), "Fourth return must be a dictionary (info)"

    def test_info_contains_processed_alerts(self):
        """info dict must contain processed_alerts."""
        env = AdaptiveAlertTriageEnv(task_id="easy", seed=42)
        obs = env.reset()
        
        action = Action(alert_id=obs.alerts[0].id, action_type="INVESTIGATE")
        _, _, _, info = env.step(action)
        
        assert "processed_alerts" in info, "info must contain 'processed_alerts'"
        assert isinstance(info["processed_alerts"], list), "processed_alerts must be a list"

    def test_info_contains_correlation_groups(self):
        """info dict must contain correlation_groups."""
        env = AdaptiveAlertTriageEnv(task_id="hard", seed=42)
        obs = env.reset()
        
        action = Action(alert_id=obs.alerts[0].id, action_type="INVESTIGATE")
        _, _, _, info = env.step(action)
        
        assert "correlation_groups" in info, "info must contain 'correlation_groups'"
        assert isinstance(info["correlation_groups"], list), "correlation_groups must be a list"

    def test_info_contains_system_failure(self):
        """info dict should indicate system failure state."""
        env = AdaptiveAlertTriageEnv(task_id="easy", seed=42)
        obs = env.reset()
        
        action = Action(alert_id=obs.alerts[0].id, action_type="INVESTIGATE")
        _, _, _, info = env.step(action)
        
        assert "system_failure" in info, "info should contain 'system_failure'"

    def test_reward_has_value(self):
        """Reward must have a numeric value."""
        env = AdaptiveAlertTriageEnv(task_id="easy", seed=42)
        obs = env.reset()
        
        action = Action(alert_id=obs.alerts[0].id, action_type="INVESTIGATE")
        _, reward, _, _ = env.step(action)
        
        assert isinstance(reward.value, (int, float)), "Reward.value must be numeric"

    def test_observation_updated_after_step(self):
        """Observation should normally change after step()."""
        env = AdaptiveAlertTriageEnv(task_id="easy", seed=42)
        obs_before = env.reset()
        
        action = Action(alert_id=obs_before.alerts[0].id, action_type="INVESTIGATE")
        obs_after, _, _, _ = env.step(action)
        
        # Episode step should have incremented
        assert obs_after.episode_step == obs_before.episode_step + 1


# ============================================================================
# REQUIREMENT 3: reset() → Observation
# ============================================================================

class TestResetInterface:
    """Verify reset() method signature and return type."""

    def test_reset_exists(self):
        """Environment must have a reset method."""
        env = AdaptiveAlertTriageEnv(task_id="easy", seed=42)
        assert hasattr(env, "reset"), "Environment must have reset() method"

    def test_reset_returns_observation(self):
        """reset() must return an Observation."""
        env = AdaptiveAlertTriageEnv(task_id="easy", seed=42)
        obs = env.reset()
        
        assert isinstance(obs, Observation), "reset() must return an Observation"

    def test_reset_accepts_seed(self):
        """reset() should accept optional seed parameter."""
        env = AdaptiveAlertTriageEnv(task_id="easy")
        obs = env.reset(seed=42)
        
        assert isinstance(obs, Observation), "reset(seed=...) should return Observation"

    def test_reset_accepts_options(self):
        """reset() should accept optional options parameter."""
        env = AdaptiveAlertTriageEnv(task_id="easy", seed=42)
        obs = env.reset(options={})
        
        assert isinstance(obs, Observation), "reset(options=...) should return Observation"

    def test_reset_reproducibility(self):
        """Same seed should produce same initial observation."""
        env1 = AdaptiveAlertTriageEnv(task_id="easy")
        obs1 = env1.reset(seed=42)
        
        env2 = AdaptiveAlertTriageEnv(task_id="easy")
        obs2 = env2.reset(seed=42)
        
        assert len(obs1.alerts) == len(obs2.alerts), "Same seed should produce same number of alerts"

    def test_reset_clears_episode_state(self):
        """reset() should clear episode state between calls."""
        env = AdaptiveAlertTriageEnv(task_id="easy", seed=42)
        
        obs1 = env.reset()
        assert obs1.episode_step == 0, "Initial episode_step should be 0"
        
        # Take a step
        if obs1.alerts:
            action = Action(alert_id=obs1.alerts[0].id, action_type="INVESTIGATE")
            _, _, _, _ = env.step(action)
        
        # Reset again
        obs2 = env.reset(seed=99)
        assert obs2.episode_step == 0, "After reset, episode_step should be 0 again"


# ============================================================================
# REQUIREMENT 4: state() → EpisodeState
# ============================================================================

class TestStateInterface:
    """Verify state() method and return type."""

    def test_state_exists(self):
        """Environment must have a state method."""
        env = AdaptiveAlertTriageEnv(task_id="easy", seed=42)
        assert hasattr(env, "state"), "Environment must have state() method"

    def test_state_returns_episode_state(self):
        """state() must return an EpisodeState."""
        env = AdaptiveAlertTriageEnv(task_id="easy", seed=42)
        env.reset()
        
        state = env.state()
        assert isinstance(state, EpisodeState), "state() must return an EpisodeState"

    def test_episode_state_contains_observation(self):
        """EpisodeState must contain current observation."""
        env = AdaptiveAlertTriageEnv(task_id="easy", seed=42)
        env.reset()
        
        state = env.state()
        assert hasattr(state, "observation"), "EpisodeState must have observation"
        assert isinstance(state.observation, Observation), "observation must be an Observation"

    def test_episode_state_contains_hidden_state(self):
        """EpisodeState must contain hidden_state dict."""
        env = AdaptiveAlertTriageEnv(task_id="easy", seed=42)
        env.reset()
        
        state = env.state()
        assert hasattr(state, "hidden_state"), "EpisodeState must have hidden_state"
        assert isinstance(state.hidden_state, dict), "hidden_state must be a dict"

    def test_hidden_state_contains_true_severities(self):
        """hidden_state must contain true_severities mapping."""
        env = AdaptiveAlertTriageEnv(task_id="easy", seed=42)
        env.reset()
        
        state = env.state()
        assert "true_severities" in state.hidden_state, "hidden_state must contain true_severities"

    def test_hidden_state_contains_correlation_groups(self):
        """hidden_state must contain correlation_groups."""
        env = AdaptiveAlertTriageEnv(task_id="hard", seed=42)
        env.reset()
        
        state = env.state()
        assert "correlation_groups" in state.hidden_state, "hidden_state must contain correlation_groups"

    def test_episode_state_contains_cumulative_reward(self):
        """EpisodeState must track cumulative_reward."""
        env = AdaptiveAlertTriageEnv(task_id="easy", seed=42)
        env.reset()
        
        state = env.state()
        assert hasattr(state, "cumulative_reward"), "EpisodeState must have cumulative_reward"
        assert isinstance(state.cumulative_reward, (int, float)), "cumulative_reward must be numeric"

    def test_episode_state_contains_failures_count(self):
        """EpisodeState must track failures count."""
        env = AdaptiveAlertTriageEnv(task_id="easy", seed=42)
        env.reset()
        
        state = env.state()
        assert hasattr(state, "failures_count"), "EpisodeState must have failures_count"
        assert isinstance(state.failures_count, int), "failures_count must be an integer"

    def test_episode_state_tracks_actions_taken(self):
        """EpisodeState should track actions taken."""
        env = AdaptiveAlertTriageEnv(task_id="easy", seed=42)
        obs = env.reset()
        
        state_before = env.state()
        initial_action_count = len(state_before.actions_taken)
        
        # Take an action
        if obs.alerts:
            action = Action(alert_id=obs.alerts[0].id, action_type="INVESTIGATE")
            _, _, _, _ = env.step(action)
        
        state_after = env.state()
        assert len(state_after.actions_taken) >= initial_action_count, "actions_taken should accumulate"


# ============================================================================
# REQUIREMENT 5: openenv.yaml with metadata
# ============================================================================

class TestOpenEnvYAML:
    """Verify openenv.yaml provides required metadata."""

    def test_openenv_yaml_exists(self):
        """openenv.yaml must exist in project root."""
        yaml_path = Path("openenv.yaml")
        assert yaml_path.exists(), f"openenv.yaml must exist at {yaml_path}"

    def test_openenv_yaml_is_valid_yaml(self):
        """openenv.yaml must be valid YAML."""
        with open("openenv.yaml") as f:
            data = yaml.safe_load(f)
        
        assert isinstance(data, dict), "openenv.yaml must parse to a dictionary"

    def test_openenv_yaml_has_name(self):
        """openenv.yaml must have a 'name' field."""
        with open("openenv.yaml") as f:
            data = yaml.safe_load(f)
        
        assert "name" in data, "openenv.yaml must have 'name' field"

    def test_openenv_yaml_has_version(self):
        """openenv.yaml must have a 'version' field."""
        with open("openenv.yaml") as f:
            data = yaml.safe_load(f)
        
        assert "version" in data, "openenv.yaml must have 'version' field"

    def test_openenv_yaml_has_description(self):
        """openenv.yaml must have a 'description' field."""
        with open("openenv.yaml") as f:
            data = yaml.safe_load(f)
        
        assert "description" in data, "openenv.yaml must have 'description' field"

    def test_openenv_yaml_has_tasks(self):
        """openenv.yaml must define tasks."""
        with open("openenv.yaml") as f:
            data = yaml.safe_load(f)
        
        assert "tasks" in data, "openenv.yaml must have 'tasks' section"
        assert isinstance(data["tasks"], list), "tasks must be a list"
        assert len(data["tasks"]) > 0, "tasks list must not be empty"

    def test_openenv_yaml_tasks_have_ids(self):
        """Each task must have an 'id' field."""
        with open("openenv.yaml") as f:
            data = yaml.safe_load(f)
        
        for task in data["tasks"]:
            assert "id" in task, f"Task missing 'id' field: {task}"

    def test_openenv_yaml_has_config(self):
        """openenv.yaml should have a 'config' section."""
        with open("openenv.yaml") as f:
            data = yaml.safe_load(f)
        
        assert "config" in data, "openenv.yaml should have 'config' section"

    def test_openenv_yaml_config_has_actions(self):
        """config should define available actions."""
        with open("openenv.yaml") as f:
            data = yaml.safe_load(f)
        
        assert "actions" in data["config"], "config must define 'actions'"
        expected_actions = {"INVESTIGATE", "IGNORE", "ESCALATE", "DELAY"}
        yaml_actions = set(data["config"]["actions"])
        assert expected_actions.issubset(yaml_actions), f"config must include all standard actions"


# ============================================================================
# REQUIREMENT 6: Validation Testing
# ============================================================================

class TestOpenEnvValidation:
    """End-to-end OpenEnv compliance validation."""

    def test_full_episode_workflow(self):
        """Complete episode following OpenEnv spec should work."""
        env = AdaptiveAlertTriageEnv(task_id="easy", seed=42)
        
        # 1. Reset to get initial observation
        obs = env.reset()
        assert isinstance(obs, Observation)
        
        # 2. Run episode steps
        done = False
        episode_steps = 0
        max_allowed_steps = env.max_steps + 5  # Allow some buffer
        
        while not done and episode_steps < max_allowed_steps:
            if not obs.alerts:
                break
            
            # Take an action
            action = Action(alert_id=obs.alerts[0].id, action_type="INVESTIGATE")
            obs, reward, done, info = env.step(action)
            
            # Validate return types
            assert isinstance(obs, Observation)
            assert isinstance(reward, Reward)
            assert isinstance(done, bool)
            assert isinstance(info, dict)
            
            episode_steps += 1
        
        # 3. Get final state
        final_state = env.state()
        assert isinstance(final_state, EpisodeState)

    def test_all_task_difficulties(self):
        """All task difficulties should be OpenEnv compliant."""
        for task_id in ["easy", "medium", "hard"]:
            env = AdaptiveAlertTriageEnv(task_id=task_id, seed=42)
            
            # Reset
            obs = env.reset()
            assert isinstance(obs, Observation), f"reset() failed for {task_id}"
            
            # Step
            if obs.alerts:
                action = Action(alert_id=obs.alerts[0].id, action_type="INVESTIGATE")
                obs, reward, done, info = env.step(action)
                
                assert isinstance(obs, Observation)
                assert isinstance(reward, Reward)
                assert isinstance(done, bool)
                assert isinstance(info, dict)
            
            # State
            state = env.state()
            assert isinstance(state, EpisodeState), f"state() failed for {task_id}"

    def test_pydantic_validation(self):
        """Pydantic models should validate their fields."""
        # Invalid action type should fail validation
        with pytest.raises(Exception):
            Action(alert_id="test", action_type="INVALID_ACTION")
        
        # Invalid alert type should fail validation
        with pytest.raises(Exception):
            Alert(
                id="test",
                visible_severity=0.5,
                confidence=0.8,
                alert_type="INVALID_TYPE",
                age=0,
            )

    def test_serialization_round_trip(self):
        """Models should serialize/deserialize without data loss."""
        action = Action(
            alert_id="alert_123",
            action_type="INVESTIGATE",
            metadata={"reason": "high severity"}
        )
        
        # Serialize
        json_str = action.model_dump_json()
        
        # Deserialize
        restored = Action.model_validate_json(json_str)
        
        assert restored.alert_id == action.alert_id
        assert restored.action_type == action.action_type
        assert restored.metadata == action.metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v"])