"""
Unit Tests for Reward Calculation

Tests reward shaping logic and component breakdown.
"""

import pytest
from adaptive_alert_triage.models import Action, Alert, Reward
from rewards.reward import (
    calculate_reward,
    calculate_system_failure_penalty,
    calculate_episode_bonus,
    get_reward_range,
    create_reward_summary,
    REWARD_CRITICAL_HANDLED,
    REWARD_FAILURE_PREVENTED,
    REWARD_FALSE_POSITIVE_IGNORED,
    PENALTY_MISSED_CRITICAL,
)


class TestRewardCalculation:
    """Test core reward calculation logic."""
    
    def test_critical_investigated_reward(self):
        """Test reward for correctly investigating critical alert."""
        alert = Alert(
            id="alert_001",
            visible_severity=0.85,
            confidence=0.9,
            alert_type="CPU",
            age=1,
            true_severity=0.90,  # Critical
            is_correlated=False,
        )
        action = Action(alert_id="alert_001", action_type="INVESTIGATE")
        
        reward = calculate_reward(action, alert)
        
        assert reward.value == REWARD_CRITICAL_HANDLED, \
            f"Expected {REWARD_CRITICAL_HANDLED}, got {reward.value}"
        assert reward.components["critical_handled"] == REWARD_CRITICAL_HANDLED
        assert reward.info["action_correct"] is True
    
    def test_critical_escalated_reward(self):
        """Test reward for escalating critical alert."""
        alert = Alert(
            id="alert_002",
            visible_severity=0.9,
            confidence=0.95,
            alert_type="SECURITY",
            age=2,
            true_severity=0.95,
            is_correlated=False,
        )
        action = Action(alert_id="alert_002", action_type="ESCALATE")
        
        reward = calculate_reward(action, alert)
        
        # Escalate gets 90% of investigate reward
        expected = REWARD_CRITICAL_HANDLED * 0.9
        assert abs(reward.value - expected) < 0.01, \
            f"Expected ~{expected}, got {reward.value}"
        assert reward.info["action_correct"] is True
    
    def test_false_positive_ignored_reward(self):
        """Test reward for correctly ignoring false positive."""
        alert = Alert(
            id="alert_003",
            visible_severity=0.3,
            confidence=0.4,
            alert_type="DISK",
            age=0,
            true_severity=0.15,  # False positive
            is_correlated=False,
        )
        action = Action(alert_id="alert_003", action_type="IGNORE")
        
        reward = calculate_reward(action, alert)
        
        assert reward.value == REWARD_FALSE_POSITIVE_IGNORED, \
            f"Expected {REWARD_FALSE_POSITIVE_IGNORED}, got {reward.value}"
        assert reward.components["false_positive_ignored"] == REWARD_FALSE_POSITIVE_IGNORED
        assert reward.info["action_correct"] is True
    
    def test_critical_ignored_penalty(self):
        """Test penalty for ignoring critical alert."""
        alert = Alert(
            id="alert_004",
            visible_severity=0.7,
            confidence=0.8,
            alert_type="SECURITY",
            age=2,
            true_severity=0.95,  # Critical
            is_correlated=False,
        )
        action = Action(alert_id="alert_004", action_type="IGNORE")
        
        reward = calculate_reward(action, alert)
        
        assert reward.value == PENALTY_MISSED_CRITICAL, \
            f"Expected {PENALTY_MISSED_CRITICAL}, got {reward.value}"
        assert reward.components["missed_critical"] == PENALTY_MISSED_CRITICAL
        assert reward.info["action_correct"] is False
    
    def test_unnecessary_investigation_penalty(self):
        """Test penalty for investigating false positive."""
        alert = Alert(
            id="alert_005",
            visible_severity=0.35,
            confidence=0.45,
            alert_type="NETWORK",
            age=0,
            true_severity=0.20,  # False positive
            is_correlated=False,
        )
        action = Action(alert_id="alert_005", action_type="INVESTIGATE")
        
        reward = calculate_reward(action, alert)
        
        assert reward.value < 0.0, "Should be negative for wasted resources"
        assert reward.components["unnecessary_investigation"] < 0.0
    
    def test_correlated_alert_bonus(self):
        """Test bonus for handling correlated alerts."""
        alert = Alert(
            id="alert_006",
            visible_severity=0.8,
            confidence=0.85,
            alert_type="CPU",
            age=1,
            true_severity=0.85,
            is_correlated=True,  # Correlated
        )
        action = Action(alert_id="alert_006", action_type="INVESTIGATE")
        
        reward = calculate_reward(action, alert)
        
        # Should get critical + failure prevention bonus
        expected = REWARD_CRITICAL_HANDLED + REWARD_FAILURE_PREVENTED
        assert reward.value == expected, \
            f"Expected {expected}, got {reward.value}"
        assert reward.components["failure_prevented"] == REWARD_FAILURE_PREVENTED
    
    def test_medium_alert_handling(self):
        """Test reward for medium severity alerts."""
        alert = Alert(
            id="alert_007",
            visible_severity=0.6,
            confidence=0.7,
            alert_type="MEMORY",
            age=1,
            true_severity=0.55,  # Medium
            is_correlated=False,
        )
        action = Action(alert_id="alert_007", action_type="INVESTIGATE")
        
        reward = calculate_reward(action, alert)
        
        # Medium alerts get scaled reward based on severity
        assert 0.0 < reward.value < REWARD_CRITICAL_HANDLED, \
            "Medium alert should get moderate positive reward"
        assert "medium_handled" in reward.components
    
    def test_delay_action_rewards(self):
        """Test rewards for DELAY action."""
        # Delaying medium alert (acceptable)
        alert_medium = Alert(
            id="alert_008",
            visible_severity=0.5,
            confidence=0.6,
            alert_type="DISK",
            age=0,
            true_severity=0.50,
            is_correlated=False,
        )
        action_delay = Action(alert_id="alert_008", action_type="DELAY")
        
        reward_medium = calculate_reward(action_delay, alert_medium)
        assert reward_medium.value >= 0.0, "Delaying medium alert should be acceptable"
        
        # Delaying critical alert (risky)
        alert_critical = Alert(
            id="alert_009",
            visible_severity=0.85,
            confidence=0.9,
            alert_type="CPU",
            age=2,
            true_severity=0.90,
            is_correlated=False,
        )
        action_delay_crit = Action(alert_id="alert_009", action_type="DELAY")
        
        reward_critical = calculate_reward(action_delay_crit, alert_critical)
        assert reward_critical.value < 0.0, "Delaying critical alert should be penalized"


class TestRewardComponents:
    """Test reward component breakdown."""
    
    def test_reward_has_components(self):
        """Test that all rewards include component breakdown."""
        alert = Alert(
            id="a1", visible_severity=0.9, confidence=0.9, alert_type="CPU",
            age=1, true_severity=0.9
        )
        action = Action(alert_id="a1", action_type="INVESTIGATE")
        
        reward = calculate_reward(action, alert)
        
        assert isinstance(reward.components, dict)
        assert len(reward.components) > 0
        assert sum(reward.components.values()) == reward.value
    
    def test_reward_info_fields(self):
        """Test that reward info contains useful debugging information."""
        alert = Alert(
            id="a1", visible_severity=0.9, confidence=0.9, alert_type="CPU",
            age=1, true_severity=0.9
        )
        action = Action(alert_id="a1", action_type="INVESTIGATE")
        
        reward = calculate_reward(action, alert)
        
        assert "alert_id" in reward.info
        assert "true_severity" in reward.info
        assert "is_critical" in reward.info
        assert "is_false_positive" in reward.info
        assert "action_correct" in reward.info


class TestAuxiliaryFunctions:
    """Test auxiliary reward functions."""
    
    def test_system_failure_penalty(self):
        """Test system failure penalty calculation."""
        penalty_1 = calculate_system_failure_penalty(1)
        penalty_3 = calculate_system_failure_penalty(3)
        
        assert penalty_1 < 0.0
        assert penalty_3 == penalty_1 * 3
    
    def test_episode_bonus_high_accuracy(self):
        """Test episode bonus for high accuracy."""
        bonus = calculate_episode_bonus(correct_actions=85, total_actions=100, failures_count=0)
        
        assert bonus > 0.0, "High accuracy should give bonus"
    
    def test_episode_bonus_perfect(self):
        """Test episode bonus for perfect performance."""
        bonus_perfect = calculate_episode_bonus(
            correct_actions=100, total_actions=100, failures_count=0
        )
        
        bonus_high = calculate_episode_bonus(
            correct_actions=85, total_actions=100, failures_count=0
        )
        
        assert bonus_perfect > bonus_high, "Perfect should get higher bonus"
    
    def test_episode_bonus_with_failures(self):
        """Test that failures reduce episode bonus."""
        bonus_no_fail = calculate_episode_bonus(
            correct_actions=80, total_actions=100, failures_count=0
        )
        bonus_with_fail = calculate_episode_bonus(
            correct_actions=80, total_actions=100, failures_count=2
        )
        
        assert bonus_no_fail > bonus_with_fail, "Failures should reduce bonus"
    
    def test_reward_range(self):
        """Test reward range calculation."""
        min_r, max_r = get_reward_range()
        
        assert min_r < 0.0, "Min reward should be negative (penalty)"
        assert max_r > 0.0, "Max reward should be positive"
        assert max_r > abs(min_r), "Max reward magnitude should exceed penalty"
    
    def test_reward_summary_empty(self):
        """Test reward summary with empty list."""
        summary = create_reward_summary([])
        
        assert summary["total_reward"] == 0.0
        assert summary["num_rewards"] == 0
    
    def test_reward_summary_aggregation(self):
        """Test reward summary aggregates correctly."""
        rewards = [
            Reward(value=10.0, components={"critical_handled": 10.0}, 
                   info={"action_correct": True}),
            Reward(value=3.0, components={"false_positive_ignored": 3.0}, 
                   info={"action_correct": True}),
            Reward(value=-2.0, components={"unnecessary_investigation": -2.0}, 
                   info={"action_correct": False}),
        ]
        
        summary = create_reward_summary(rewards)
        
        assert summary["total_reward"] == 11.0
        assert summary["mean_reward"] == 11.0 / 3
        assert summary["num_rewards"] == 3
        assert summary["correct_actions"] == 2
        assert summary["accuracy"] == 2/3
        assert "critical_handled" in summary["components"]


class TestRewardConsistency:
    """Test consistency and edge cases."""
    
    def test_same_input_same_reward(self):
        """Test deterministic reward calculation."""
        alert = Alert(
            id="a1", visible_severity=0.9, confidence=0.9, alert_type="CPU",
            age=1, true_severity=0.9
        )
        action = Action(alert_id="a1", action_type="INVESTIGATE")
        
        reward1 = calculate_reward(action, alert)
        reward2 = calculate_reward(action, alert)
        
        assert reward1.value == reward2.value
    
    def test_all_action_types_covered(self):
        """Test that all action types produce rewards."""
        alert = Alert(
            id="a1", visible_severity=0.6, confidence=0.7, alert_type="CPU",
            age=1, true_severity=0.6
        )
        
        action_types = ["INVESTIGATE", "IGNORE", "ESCALATE", "DELAY"]
        
        for action_type in action_types:
            action = Action(alert_id="a1", action_type=action_type)
            reward = calculate_reward(action, alert)
            
            assert isinstance(reward.value, float), \
                f"Action {action_type} should return numeric reward"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
