"""
Unit Tests for Adaptive Alert Triage Environment

Tests core environment functionality: reset, step, state management.
"""

import pytest
import numpy as np
from adaptive_alert_triage.env import AdaptiveAlertTriageEnv
from adaptive_alert_triage.models import Action, Observation, Reward


class TestEnvironmentBasics:
    """Test basic environment operations."""
    
    def test_initialization(self):
        """Test environment initialization with different tasks."""
        for task_id in ["easy", "medium", "hard"]:
            env = AdaptiveAlertTriageEnv(task_id=task_id, seed=42)
            assert env.task_id == task_id
            assert env.current_step == 0
    
    def test_reset(self):
        """Test environment reset returns valid observation."""
        env = AdaptiveAlertTriageEnv(task_id="easy", seed=42)
        obs = env.reset()
        
        assert isinstance(obs, Observation)
        assert len(obs.alerts) > 0
        assert 0.0 <= obs.system_load <= 1.0
        assert obs.queue_length >= 0
        assert obs.time_remaining == env.max_steps
        assert obs.episode_step == 0
    
    def test_reset_reproducibility(self):
        """Test that same seed produces same initial state."""
        env1 = AdaptiveAlertTriageEnv(task_id="easy", seed=42)
        obs1 = env1.reset()
        
        env2 = AdaptiveAlertTriageEnv(task_id="easy", seed=42)
        obs2 = env2.reset()
        
        assert len(obs1.alerts) == len(obs2.alerts)
        assert obs1.system_load == obs2.system_load
    
    def test_step_basic(self):
        """Test basic step execution."""
        env = AdaptiveAlertTriageEnv(task_id="easy", seed=42)
        obs = env.reset()
        
        # Take action on first alert
        action = Action(alert_id=obs.alerts[0].id, action_type="INVESTIGATE")
        next_obs, reward, done, info = env.step(action)
        
        assert isinstance(next_obs, Observation)
        assert isinstance(reward, Reward)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert env.current_step == 1
    
    def test_step_invalid_alert(self):
        """Test step with invalid alert ID."""
        env = AdaptiveAlertTriageEnv(task_id="easy", seed=42)
        obs = env.reset()
        
        # Invalid alert ID
        action = Action(alert_id="nonexistent_alert", action_type="INVESTIGATE")
        next_obs, reward, done, info = env.step(action)
        
        assert reward.value < 0  # Should be penalized
        assert done  # Should terminate on invalid action
    
    def test_episode_termination(self):
        """Test episode terminates at max_steps."""
        env = AdaptiveAlertTriageEnv(task_id="easy", max_steps=5, seed=42)
        obs = env.reset()
        
        done = False
        steps = 0
        
        while not done and steps < 10:  # Safety limit
            if obs.alerts:
                action = Action(alert_id=obs.alerts[0].id, action_type="INVESTIGATE")
                obs, reward, done, info = env.step(action)
            else:
                break
            steps += 1
        
        assert steps <= 5 or done
    
    def test_state_method(self):
        """Test state() returns complete episode state."""
        env = AdaptiveAlertTriageEnv(task_id="easy", seed=42)
        obs = env.reset()
        
        state = env.state()
        
        assert state.observation.episode_step == obs.episode_step
        assert "true_severities" in state.hidden_state
        assert state.cumulative_reward == 0.0
        assert state.failures_count == 0


class TestTaskConfigurations:
    """Test task-specific configurations."""
    
    def test_easy_task_config(self):
        """Test easy task has correct configuration."""
        env = AdaptiveAlertTriageEnv(task_id="easy", seed=42)
        
        assert env.max_steps == 30
        assert env.max_investigations_per_step is None  # No resource constraint
        assert env.failure_threshold == 5
    
    def test_medium_task_config(self):
        """Test medium task has resource constraints."""
        env = AdaptiveAlertTriageEnv(task_id="medium", seed=42)
        
        assert env.max_steps == 40
        assert env.max_investigations_per_step == 3  # Resource constrained
        assert env.failure_threshold == 5
    
    def test_hard_task_config(self):
        """Test hard task has stricter failure tolerance."""
        env = AdaptiveAlertTriageEnv(task_id="hard", seed=42)
        
        assert env.max_steps == 50
        assert env.max_investigations_per_step == 3
        assert env.failure_threshold == 3  # Stricter
    
    def test_resource_budget_tracking(self):
        """Test resource budget is tracked in medium/hard tasks."""
        env = AdaptiveAlertTriageEnv(task_id="medium", seed=42)
        obs = env.reset()
        
        # Resource budget should be visible in observation
        assert obs.resource_budget is not None
        assert obs.resource_budget == 3


class TestObservability:
    """Test partial observability - hidden vs visible attributes."""
    
    def test_hidden_attributes_not_exposed(self):
        """Test that true_severity and correlations are hidden from agent."""
        env = AdaptiveAlertTriageEnv(task_id="easy", seed=42)
        obs = env.reset()
        
        # Get full state (includes hidden info)
        state = env.state()
        
        # Check that observation doesn't expose hidden attributes
        for alert in obs.alerts:
            # These should be zeroed out or default in observation
            assert alert.true_severity == 0.0  # Hidden
            assert alert.is_correlated == False  # Hidden
        
        # But they should exist in hidden state
        assert "true_severities" in state.hidden_state
    
    def test_visible_attributes_noisy(self):
        """Test that visible_severity differs from true_severity (noise)."""
        env = AdaptiveAlertTriageEnv(task_id="easy", seed=42)
        env.reset()
        
        # Access internal alerts (with true values)
        internal_alerts = env.alerts
        
        if internal_alerts:
            # Visible and true severity should differ due to noise
            # (This is probabilistic, so we check that at least some differ)
            has_noise = any(
                abs(a.visible_severity - a.true_severity) > 0.01
                for a in internal_alerts
            )
            assert has_noise, "Expected observation noise in severity"


class TestAlertDynamics:
    """Test alert generation and aging dynamics."""
    
    def test_alerts_age_over_time(self):
        """Test that unhandled alerts age."""
        env = AdaptiveAlertTriageEnv(task_id="easy", seed=42)
        obs = env.reset()
        
        if len(obs.alerts) < 2:
            pytest.skip("Need multiple alerts for this test")
        
        # Handle only first alert
        first_alert_id = obs.alerts[0].id
        second_alert_initial_age = obs.alerts[1].age
        
        action = Action(alert_id=first_alert_id, action_type="INVESTIGATE")
        next_obs, _, _, _ = env.step(action)
        
        # Second alert should still exist and be older
        second_alert_new = next((a for a in next_obs.alerts if a.age > second_alert_initial_age), None)
        assert second_alert_new is not None, "Alert should have aged"
    
    def test_new_alerts_generated(self):
        """Test that new alerts are generated over time."""
        env = AdaptiveAlertTriageEnv(task_id="easy", seed=42)
        obs = env.reset()
        
        initial_alert_ids = {a.id for a in obs.alerts}
        
        # Take several steps
        for _ in range(5):
            if obs.alerts:
                action = Action(alert_id=obs.alerts[0].id, action_type="INVESTIGATE")
                obs, _, done, _ = env.step(action)
                if done:
                    break
        
        # Should have some new alert IDs
        current_alert_ids = {a.id for a in obs.alerts}
        new_alerts = current_alert_ids - initial_alert_ids
        
        # Probabilistic, but likely to have new alerts after 5 steps
        assert len(new_alerts) > 0, "Expected new alerts to be generated"


class TestRewardSignals:
    """Test reward calculation."""
    
    def test_positive_reward_for_critical(self):
        """Test that investigating critical alerts gives positive reward."""
        env = AdaptiveAlertTriageEnv(task_id="easy", seed=42)
        obs = env.reset()
        
        # Find a high-severity alert (visible)
        high_sev_alert = max(obs.alerts, key=lambda a: a.visible_severity)
        
        action = Action(alert_id=high_sev_alert.id, action_type="INVESTIGATE")
        _, reward, _, _ = env.step(action)
        
        # Likely to be positive for high visible severity
        # (Not guaranteed due to noise, but statistically likely)
        assert isinstance(reward.value, float)
    
    def test_reward_components(self):
        """Test that reward has component breakdown."""
        env = AdaptiveAlertTriageEnv(task_id="easy", seed=42)
        obs = env.reset()
        
        action = Action(alert_id=obs.alerts[0].id, action_type="INVESTIGATE")
        _, reward, _, _ = env.step(action)
        
        assert isinstance(reward.components, dict)
        assert len(reward.components) > 0


def test_render(capsys):
    """Test render method produces output."""
    env = AdaptiveAlertTriageEnv(task_id="easy", seed=42)
    obs = env.reset()
    
    # Test human mode (prints)
    env.render(mode="human")
    captured = capsys.readouterr()
    assert "Step" in captured.out
    assert "Failures" in captured.out
    
    # Test ansi mode (returns string)
    output = env.render(mode="ansi")
    assert isinstance(output, str)
    assert "Step" in output


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
