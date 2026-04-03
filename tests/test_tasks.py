"""
Unit Tests for Task Graders

Tests grading logic for easy, medium, and hard tasks.
"""

import pytest
from adaptive_alert_triage.models import Action, Alert, Reward
from tasks.easy import EasyTaskGrader
from tasks.medium import MediumTaskGrader
from tasks.hard import HardTaskGrader


class TestEasyTaskGrader:
    """Test easy task grading logic."""
    
    def test_critical_alert_correct(self):
        """Test correct handling of critical alert."""
        grader = EasyTaskGrader()
        
        alert = Alert(
            id="alert_001",
            visible_severity=0.85,
            confidence=0.9,
            alert_type="CPU",
            age=1,
            true_severity=0.90,  # Critical
        )
        action = Action(alert_id="alert_001", action_type="INVESTIGATE")
        reward = Reward(value=10.0)
        
        score = grader.grade_action(action, alert, reward)
        
        assert score == 1.0, "Should get full score for correct action"
        assert grader.correct_actions == 1
        assert grader.total_actions == 1
    
    def test_critical_alert_incorrect(self):
        """Test incorrect handling of critical alert (ignored)."""
        grader = EasyTaskGrader()
        
        alert = Alert(
            id="alert_002",
            visible_severity=0.7,
            confidence=0.8,
            alert_type="SECURITY",
            age=2,
            true_severity=0.95,  # Critical
        )
        action = Action(alert_id="alert_002", action_type="IGNORE")
        reward = Reward(value=-8.0)
        
        score = grader.grade_action(action, alert, reward)
        
        assert score == 0.0, "Should get zero score for missed critical"
        assert grader.correct_actions == 0
        assert grader.total_actions == 1
    
    def test_false_positive_correct(self):
        """Test correct handling of false positive (ignored)."""
        grader = EasyTaskGrader()
        
        alert = Alert(
            id="alert_003",
            visible_severity=0.3,
            confidence=0.4,
            alert_type="DISK",
            age=0,
            true_severity=0.15,  # False positive
        )
        action = Action(alert_id="alert_003", action_type="IGNORE")
        reward = Reward(value=3.0)
        
        score = grader.grade_action(action, alert, reward)
        
        assert score == 1.0, "Should get full score for ignoring FP"
        assert grader.correct_actions == 1
    
    def test_episode_score_calculation(self):
        """Test episode score aggregation."""
        grader = EasyTaskGrader()
        
        # 3 actions: 2 correct, 1 incorrect
        alerts_actions = [
            (Alert(id="a1", visible_severity=0.9, confidence=0.9, alert_type="CPU", 
                   age=1, true_severity=0.9), "INVESTIGATE", True),
            (Alert(id="a2", visible_severity=0.3, confidence=0.4, alert_type="DISK", 
                   age=0, true_severity=0.2), "IGNORE", True),
            (Alert(id="a3", visible_severity=0.8, confidence=0.8, alert_type="SECURITY", 
                   age=1, true_severity=0.95), "IGNORE", False),
        ]
        
        for alert, action_type, _ in alerts_actions:
            action = Action(alert_id=alert.id, action_type=action_type)
            reward = Reward(value=0.0)
            grader.grade_action(action, alert, reward)
        
        score = grader.get_episode_score()
        assert abs(score - 2/3) < 0.01, f"Expected 0.667, got {score}"
    
    def test_metrics_breakdown(self):
        """Test detailed metrics generation."""
        grader = EasyTaskGrader()
        
        alert = Alert(
            id="a1", visible_severity=0.9, confidence=0.9, alert_type="CPU",
            age=1, true_severity=0.9
        )
        action = Action(alert_id="a1", action_type="INVESTIGATE")
        reward = Reward(value=10.0)
        
        grader.grade_action(action, alert, reward)
        metrics = grader.get_metrics()
        
        assert "overall_score" in metrics
        assert "correct_actions" in metrics
        assert "critical_accuracy" in metrics
        assert "action_breakdown" in metrics


class TestMediumTaskGrader:
    """Test medium task grading logic with resource constraints."""
    
    def test_productive_investigation(self):
        """Test high-value investigation scores well."""
        grader = MediumTaskGrader(max_investigations_per_step=3)
        
        alert = Alert(
            id="alert_001",
            visible_severity=0.85,
            confidence=0.9,
            alert_type="CPU",
            age=1,
            true_severity=0.90,
        )
        action = Action(alert_id="alert_001", action_type="INVESTIGATE")
        reward = Reward(value=10.0)
        
        contribution = grader.grade_action(action, alert, reward)
        
        assert contribution > 0.0, "High-value investigation should contribute positively"
        assert grader.investigations_used == 1
    
    def test_wasteful_investigation(self):
        """Test investigation on false positive is penalized."""
        grader = MediumTaskGrader(max_investigations_per_step=3)
        
        alert = Alert(
            id="alert_002",
            visible_severity=0.3,
            confidence=0.4,
            alert_type="DISK",
            age=0,
            true_severity=0.15,  # False positive
        )
        action = Action(alert_id="alert_002", action_type="INVESTIGATE")
        reward = Reward(value=-2.0)
        
        contribution = grader.grade_action(action, alert, reward)
        
        assert contribution < 0.0, "Wasteful investigation should be penalized"
        assert grader.unnecessary_investigations == 1
    
    def test_resource_efficiency_calculation(self):
        """Test resource efficiency metric."""
        grader = MediumTaskGrader(max_investigations_per_step=3)
        
        # 2 productive investigations, 1 wasteful
        alerts_actions = [
            (0.9, "INVESTIGATE", True),   # Productive
            (0.8, "INVESTIGATE", True),   # Productive
            (0.15, "INVESTIGATE", False), # Wasteful
        ]
        
        for true_sev, action_type, _ in alerts_actions:
            alert = Alert(
                id=f"a_{true_sev}", visible_severity=true_sev, confidence=0.8,
                alert_type="CPU", age=1, true_severity=true_sev
            )
            action = Action(alert_id=alert.id, action_type=action_type)
            reward = Reward(value=0.0)
            grader.grade_action(action, alert, reward)
        
        efficiency = grader.calculate_resource_efficiency()
        assert abs(efficiency - 2/3) < 0.01, f"Expected 0.667, got {efficiency}"
    
    def test_episode_score_with_efficiency(self):
        """Test that episode score considers efficiency factor."""
        grader = MediumTaskGrader(max_investigations_per_step=3)
        
        # Add some actions
        alert = Alert(
            id="a1", visible_severity=0.9, confidence=0.9, alert_type="CPU",
            age=1, true_severity=0.9
        )
        action = Action(alert_id="a1", action_type="INVESTIGATE")
        reward = Reward(value=10.0)
        
        grader.grade_action(action, alert, reward)
        score = grader.get_episode_score()
        
        assert 0.0 <= score <= 1.0, "Score should be normalized"
    
    def test_critical_missed_penalty(self):
        """Test missing critical alerts incurs penalty."""
        grader = MediumTaskGrader(max_investigations_per_step=3)
        
        alert = Alert(
            id="a1", visible_severity=0.8, confidence=0.8, alert_type="SECURITY",
            age=1, true_severity=0.95  # Critical
        )
        action = Action(alert_id="a1", action_type="IGNORE")
        reward = Reward(value=-8.0)
        
        grader.grade_action(action, alert, reward)
        
        assert grader.critical_missed == 1
        # Score should be penalized
        score = grader.get_episode_score()
        assert score < 0.5, "Missing critical should heavily impact score"


class TestHardTaskGrader:
    """Test hard task grading with correlation detection."""
    
    def test_correlation_detection(self):
        """Test bonus for handling correlated alerts."""
        correlation_chains = [["alert_001", "alert_002", "alert_003"]]
        grader = HardTaskGrader(correlation_chains=correlation_chains)
        
        alert = Alert(
            id="alert_001",
            visible_severity=0.8,
            confidence=0.85,
            alert_type="CPU",
            age=1,
            true_severity=0.85,
            is_correlated=True,
        )
        action = Action(alert_id="alert_001", action_type="INVESTIGATE")
        reward = Reward(value=10.0)
        
        contribution = grader.grade_action(action, alert, reward)
        
        # Should get base score + correlation bonus
        assert contribution > alert.true_severity, "Should get correlation bonus"
        assert grader.correlation_bonus > 0.0
    
    def test_failure_prevention_bonus(self):
        """Test bonus for preventing cascading failures."""
        correlation_chains = [["alert_001", "alert_002", "alert_003"]]
        grader = HardTaskGrader(correlation_chains=correlation_chains)
        
        # Handle first alert in chain (early detection)
        alert = Alert(
            id="alert_001",
            visible_severity=0.75,
            confidence=0.85,
            alert_type="CPU",
            age=1,
            true_severity=0.80,
            is_correlated=True,
        )
        action = Action(alert_id="alert_001", action_type="INVESTIGATE")
        reward = Reward(value=10.0)
        
        grader.grade_action(action, alert, reward)
        
        assert grader.failures_prevented >= 1, "Should register failure prevention"
    
    def test_system_failure_penalty(self):
        """Test heavy penalty for system failures."""
        grader = HardTaskGrader()
        
        # Record a failure
        grader.record_system_failure("alert_001")
        
        assert grader.system_failures == 1
        assert grader.stability_penalty > 0.0
        
        # Stability score should be reduced
        stability = grader.calculate_stability_score()
        assert stability < 1.0
    
    def test_missed_correlated_alert_penalty(self):
        """Test extra penalty for missing correlated alerts."""
        correlation_chains = [["alert_001", "alert_002"]]
        grader = HardTaskGrader(correlation_chains=correlation_chains)
        
        alert = Alert(
            id="alert_001",
            visible_severity=0.7,
            confidence=0.8,
            alert_type="CPU",
            age=1,
            true_severity=0.85,
            is_correlated=True,
        )
        action = Action(alert_id="alert_001", action_type="IGNORE")
        reward = Reward(value=-8.0)
        
        contribution = grader.grade_action(action, alert, reward)
        
        # Should have heavy penalty for missing correlated critical
        assert contribution < -2.0, "Should have extra penalty for correlated miss"
    
    def test_correlation_detection_rate(self):
        """Test calculation of correlation detection rate."""
        correlation_chains = [
            ["alert_001", "alert_002"],
            ["alert_003", "alert_004"],
        ]
        grader = HardTaskGrader(correlation_chains=correlation_chains)
        
        # Handle one chain
        grader.chains_handled.add(0)
        
        rate = grader.calculate_correlation_detection_rate()
        assert abs(rate - 0.5) < 0.01, "Should detect 50% of chains"
    
    def test_stability_score_perfect(self):
        """Test perfect stability (zero failures)."""
        grader = HardTaskGrader()
        
        stability = grader.calculate_stability_score()
        assert stability == 1.0, "Zero failures should give perfect stability"
    
    def test_stability_score_degraded(self):
        """Test degraded stability with failures."""
        grader = HardTaskGrader()
        
        # Multiple failures
        for _ in range(3):
            grader.record_system_failure()
        
        stability = grader.calculate_stability_score()
        assert stability < 1.0, "Failures should reduce stability"


def test_grader_reset():
    """Test that graders can be reset between episodes."""
    grader = EasyTaskGrader()
    
    # Do some actions
    alert = Alert(
        id="a1", visible_severity=0.9, confidence=0.9, alert_type="CPU",
        age=1, true_severity=0.9
    )
    action = Action(alert_id="a1", action_type="INVESTIGATE")
    reward = Reward(value=10.0)
    
    grader.grade_action(action, alert, reward)
    assert grader.total_actions == 1
    
    # Reset
    grader.reset()
    assert grader.total_actions == 0
    assert grader.correct_actions == 0
    assert len(grader.action_history) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
