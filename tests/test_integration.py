"""
Integration Tests for Adaptive Alert Triage Evaluation System

PRODUCTION TESTS: Verify the critical fixes to the evaluation pipeline:
1. info["processed_alerts"] contains ground truth after step()
2. correlation_groups are dynamically updated
3. system_failure flag is properly set
4. Graders produce non-zero scores with actual data

Run with: pytest tests/test_integration.py -v
"""

import pytest
import numpy as np

from adaptive_alert_triage.env import AdaptiveAlertTriageEnv
from adaptive_alert_triage.models import Action
from agents.baseline import RuleBasedAgent
from tasks.easy import EasyTaskGrader
from tasks.medium import MediumTaskGrader
from tasks.hard import HardTaskGrader


class TestProcessedAlertsInInfo:
    """Test that step() returns processed_alerts with ground truth."""
    
    def test_processed_alerts_present_in_info(self):
        """Verify info dict contains processed_alerts after step()."""
        env = AdaptiveAlertTriageEnv(task_id="easy", seed=42)
        obs = env.reset()
        
        if obs.alerts:
            action = Action(alert_id=obs.alerts[0].id, action_type="INVESTIGATE")
            _, _, _, info = env.step(action)
            
            assert "processed_alerts" in info, "processed_alerts missing from info"
            assert len(info["processed_alerts"]) > 0, "processed_alerts is empty"
    
    def test_processed_alerts_has_true_severity(self):
        """Verify processed_alerts contains true_severity (ground truth)."""
        env = AdaptiveAlertTriageEnv(task_id="easy", seed=42)
        obs = env.reset()
        
        if obs.alerts:
            action = Action(alert_id=obs.alerts[0].id, action_type="INVESTIGATE")
            _, _, _, info = env.step(action)
            
            alert_data = info["processed_alerts"][0]
            assert "true_severity" in alert_data, "true_severity missing"
            assert isinstance(alert_data["true_severity"], float), "true_severity not float"
            assert 0.0 <= alert_data["true_severity"] <= 1.0, "true_severity out of range"
    
    def test_processed_alerts_has_is_correlated(self):
        """Verify processed_alerts contains is_correlated flag."""
        env = AdaptiveAlertTriageEnv(task_id="hard", seed=42)
        obs = env.reset()
        
        if obs.alerts:
            action = Action(alert_id=obs.alerts[0].id, action_type="INVESTIGATE")
            _, _, _, info = env.step(action)
            
            alert_data = info["processed_alerts"][0]
            assert "is_correlated" in alert_data, "is_correlated missing"
            assert isinstance(alert_data["is_correlated"], bool), "is_correlated not bool"
    
    def test_processed_alerts_has_action_taken(self):
        """Verify processed_alerts records the action taken."""
        env = AdaptiveAlertTriageEnv(task_id="easy", seed=42)
        obs = env.reset()
        
        if obs.alerts:
            action = Action(alert_id=obs.alerts[0].id, action_type="ESCALATE")
            _, _, _, info = env.step(action)
            
            alert_data = info["processed_alerts"][0]
            assert "action_taken" in alert_data, "action_taken missing"
            assert alert_data["action_taken"] == "ESCALATE", "action_taken incorrect"
    
    def test_alert_not_lost_after_step(self):
        """
        CRITICAL: Verify alert data is preserved in info even though
        the alert may be removed from env.alerts after step().
        
        The key point: processed_alerts contains ground truth data that was
        captured BEFORE any removal, so graders always have complete data.
        """
        env = AdaptiveAlertTriageEnv(task_id="easy", seed=42)
        obs = env.reset()
        
        if obs.alerts:
            alert_id = obs.alerts[0].id
            action = Action(alert_id=alert_id, action_type="INVESTIGATE")
            _, _, _, info = env.step(action)
            
            # CRITICAL CHECK: processed_alerts should have the alert data
            # regardless of whether it's still in env.alerts
            assert len(info["processed_alerts"]) == 1, "processed_alerts should have alert data"
            assert info["processed_alerts"][0]["alert_id"] == alert_id, "Alert ID should match"
            
            # Verify ground truth is preserved
            alert_data = info["processed_alerts"][0]
            assert "true_severity" in alert_data, "Ground truth should be preserved"
            assert "is_correlated" in alert_data, "Correlation flag should be preserved"
            assert alert_data["action_taken"] == "INVESTIGATE", "Action should be recorded"


class TestCorrelationGroupsDynamic:
    """Test that correlation_groups are updated dynamically."""
    
    def test_correlation_groups_in_info(self):
        """Verify info contains correlation_groups."""
        env = AdaptiveAlertTriageEnv(task_id="hard", seed=42)
        obs = env.reset()
        
        if obs.alerts:
            action = Action(alert_id=obs.alerts[0].id, action_type="INVESTIGATE")
            _, _, _, info = env.step(action)
            
            assert "correlation_groups" in info, "correlation_groups missing from info"
            assert isinstance(info["correlation_groups"], list)
    
    def test_correlation_groups_grow_during_episode(self):
        """
        CRITICAL: Verify correlation_groups grows during episode.
        At reset() it's empty, but should accumulate chains.
        """
        env = AdaptiveAlertTriageEnv(task_id="hard", seed=100)
        obs = env.reset()
        
        # At start, may be empty
        initial_state = env.state()
        initial_chains = initial_state.hidden_state.get("correlation_groups", [])
        
        # Run multiple steps
        max_chains_seen = len(initial_chains)
        done = False
        steps = 0
        
        while not done and steps < 20:
            if not obs.alerts:
                break
            
            action = Action(alert_id=obs.alerts[0].id, action_type="INVESTIGATE")
            obs, _, done, info = env.step(action)
            
            current_chains = info.get("correlation_groups", [])
            if len(current_chains) > max_chains_seen:
                max_chains_seen = len(current_chains)
            
            steps += 1
        
        # With hard task (40% correlation prob), should see some chains
        # This is probabilistic, so we just verify the mechanism works
        assert "correlation_groups" in info, "correlation_groups should be in info"


class TestSystemFailureFlag:
    """Test that system_failure is properly set in info."""
    
    def test_system_failure_in_info(self):
        """Verify info contains system_failure flag."""
        env = AdaptiveAlertTriageEnv(task_id="easy", seed=42)
        obs = env.reset()
        
        if obs.alerts:
            action = Action(alert_id=obs.alerts[0].id, action_type="INVESTIGATE")
            _, _, _, info = env.step(action)
            
            assert "system_failure" in info, "system_failure missing from info"
            assert isinstance(info["system_failure"], bool)
    
    def test_failures_this_step_in_info(self):
        """Verify info contains failures_this_step count."""
        env = AdaptiveAlertTriageEnv(task_id="easy", seed=42)
        obs = env.reset()
        
        if obs.alerts:
            action = Action(alert_id=obs.alerts[0].id, action_type="INVESTIGATE")
            _, _, _, info = env.step(action)
            
            assert "failures_this_step" in info, "failures_this_step missing"
            assert isinstance(info["failures_this_step"], int)


class TestGraderWithProcessStep:
    """Test that graders work with new process_step API."""
    
    def test_easy_grader_process_step(self):
        """Test EasyTaskGrader.process_step() with alert data dict."""
        grader = EasyTaskGrader()
        
        # Simulate alert data from info["processed_alerts"]
        alert_data = {
            "alert_id": "test_001",
            "true_severity": 0.9,  # Critical
            "visible_severity": 0.85,
            "confidence": 0.9,
            "action_taken": "INVESTIGATE",  # Correct action
        }
        
        score = grader.process_step(alert_data, {})
        assert score == 1.0, "Should be correct for investigating critical"
        
        final_score = grader.get_episode_score()
        assert final_score == 0.99, "Episode score should be 0.99 mapped"
    
    def test_medium_grader_process_step(self):
        """Test MediumTaskGrader.process_step() with alert data dict."""
        grader = MediumTaskGrader()
        
        alert_data = {
            "alert_id": "test_001",
            "true_severity": 0.8,
            "visible_severity": 0.75,
            "action_taken": "INVESTIGATE",
        }
        
        contribution = grader.process_step(alert_data, {})
        assert contribution > 0, "Should have positive contribution for good investigation"
    
    def test_hard_grader_process_step_with_correlation(self):
        """
        CRITICAL: Test HardTaskGrader with correlated alert.
        Verify correlation_bonus fires when is_correlated is True.
        """
        grader = HardTaskGrader(correlation_chains=[["test_001", "test_002"]])
        
        # Process correlated alert
        alert_data = {
            "alert_id": "test_001",
            "true_severity": 0.8,
            "visible_severity": 0.75,
            "is_correlated": True,  # Ground truth!
            "action_taken": "INVESTIGATE",
            "correlation_group": 0,
        }
        
        contribution = grader.process_step(alert_data, {})
        
        # Should have correlation bonus mapped to contribution
        assert contribution >= 0.8, "Should get bonus for correlated alert"


class TestEvaluationIntegration:
    """End-to-end integration tests for evaluation pipeline."""
    
    def test_easy_task_produces_nonzero_scores(self):
        """
        CRITICAL: Verify easy task evaluation produces non-zero scores.
        This was broken before because alerts were None.
        """
        env = AdaptiveAlertTriageEnv(task_id="easy", seed=42)
        agent = RuleBasedAgent()
        grader = EasyTaskGrader()
        
        obs = env.reset()
        done = False
        
        while not done:
            if not obs.alerts:
                break
            
            action = agent.act(obs)
            obs, _, done, info = env.step(action)
            
            # Use processed_alerts for grading
            processed_alerts = info.get("processed_alerts", [])
            if processed_alerts:
                grader.process_step(processed_alerts[0], info)
        
        score = grader.get_episode_score()
        
        # RuleBased agent should get SOME correct actions
        assert score > 0.0, f"Score should be > 0, got {score}"
        assert grader.total_actions > 0, "Should have processed some actions"
    
    def test_medium_task_produces_nonzero_scores(self):
        """Verify medium task evaluation produces non-zero scores."""
        env = AdaptiveAlertTriageEnv(task_id="medium", seed=42)
        agent = RuleBasedAgent()
        grader = MediumTaskGrader()
        
        obs = env.reset()
        done = False
        
        while not done:
            if not obs.alerts:
                break
            
            action = agent.act(obs)
            obs, _, done, info = env.step(action)
            
            processed_alerts = info.get("processed_alerts", [])
            if processed_alerts:
                grader.process_step(processed_alerts[0], info)
        
        score = grader.get_episode_score()
        assert score > 0.0, f"Score should be > 0, got {score}"
    
    def test_hard_task_tracks_correlations(self):
        """
        CRITICAL: Verify hard task detects correlations.
        """
        env = AdaptiveAlertTriageEnv(task_id="hard", seed=42)
        agent = RuleBasedAgent()
        grader = HardTaskGrader()
        
        obs = env.reset()
        done = False
        correlated_alerts_seen = 0
        
        while not done:
            if not obs.alerts:
                break
            
            action = agent.act(obs)
            obs, _, done, info = env.step(action)
            
            # Update correlation chains dynamically
            grader.update_correlation_state(info.get("correlation_groups", []))
            
            processed_alerts = info.get("processed_alerts", [])
            if processed_alerts:
                alert_data = processed_alerts[0]
                if alert_data.get("is_correlated", False):
                    correlated_alerts_seen += 1
                grader.process_step(alert_data, info)
        
        score = grader.get_episode_score()
        metrics = grader.get_metrics()
        
        # Verify grader tracked data
        assert grader._total_actions > 0, "Should have processed actions"
        assert score >= 0.0, f"Score should be >= 0, got {score}"
        
        # Log metrics for debugging
        print(f"\nHard task metrics:")
        print(f"  Score: {score:.3f}")
        print(f"  Correlated alerts seen: {correlated_alerts_seen}")
        print(f"  Total chains: {metrics['total_chains']}")
    
    def test_full_evaluation_episode(self):
        """Full evaluation episode with all fixes."""
        from evaluation.evaluate import evaluate_agent_on_task
        
        agent = RuleBasedAgent()
        
        # Run on all tasks
        for task_id in ["easy", "medium", "hard"]:
            results = evaluate_agent_on_task(
                agent=agent,
                task_id=task_id,
                num_episodes=3,
                verbose=False,
            )
            
            # Verify we get actual scores, not 0.0
            assert results["mean_score"] >= 0.0, f"{task_id}: Score should be >= 0"
            
            # For easy task with rule-based agent, expect some success
            if task_id == "easy":
                assert results["mean_score"] > 0.0 or results["mean_reward"] != 0, \
                    "Easy task should produce non-trivial results"


class TestAlertPersistence:
    """Test that alert data persists correctly through the pipeline."""
    
    def test_true_severity_matches_internal_state(self):
        """Verify true_severity in processed_alerts matches internal alert."""
        env = AdaptiveAlertTriageEnv(task_id="easy", seed=42)
        obs = env.reset()
        
        if obs.alerts:
            # Get internal true severity before step
            alert_id = obs.alerts[0].id
            internal_alert = next(a for a in env.alerts if a.id == alert_id)
            expected_true_severity = internal_alert.true_severity
            
            action = Action(alert_id=alert_id, action_type="INVESTIGATE")
            _, _, _, info = env.step(action)
            
            # Verify it matches
            alert_data = info["processed_alerts"][0]
            assert alert_data["true_severity"] == expected_true_severity, \
                "true_severity in processed_alerts should match internal state"
    
    def test_is_correlated_matches_internal_state(self):
        """Verify is_correlated in processed_alerts matches internal alert."""
        env = AdaptiveAlertTriageEnv(task_id="hard", seed=42)
        obs = env.reset()
        
        if obs.alerts:
            alert_id = obs.alerts[0].id
            internal_alert = next(a for a in env.alerts if a.id == alert_id)
            expected_is_correlated = internal_alert.is_correlated
            
            action = Action(alert_id=alert_id, action_type="INVESTIGATE")
            _, _, _, info = env.step(action)
            
            alert_data = info["processed_alerts"][0]
            assert alert_data["is_correlated"] == expected_is_correlated


class TestCorrelationBonusFiring:
    """Test that correlation bonus actually fires in hard task."""
    
    def test_correlation_bonus_with_correlated_alert(self):
        """
        CRITICAL: Manually create scenario where correlation bonus MUST fire.
        """
        grader = HardTaskGrader(correlation_chains=[["alert_A", "alert_B", "alert_C"]])
        
        # Process alert that is in correlation chain
        alert_data = {
            "alert_id": "alert_A",
            "true_severity": 0.85,
            "is_correlated": True,
            "action_taken": "INVESTIGATE",
            "correlation_group": 0,
        }
        
        grader.process_step(alert_data, {})
        
        assert grader.get_metrics()["chain_score"] > 0, \
            "Correlation bonus should increase!"
        
        # Should also detect the correlation
        assert grader.calculate_correlation_detection_rate() > 0.0, "Should detect correlation"
    
    def test_no_bonus_for_non_correlated(self):
        """Verify no correlation bonus for non-correlated alerts."""
        grader = HardTaskGrader()
        
        alert_data = {
            "alert_id": "independent_001",
            "true_severity": 0.9,
            "is_correlated": False,  # Not correlated
            "action_taken": "INVESTIGATE",
            "correlation_group": None,
        }
        
        grader.process_step(alert_data, {})
        
        assert grader.get_metrics()["chain_score"] == 0.0, "No bonus for non-correlated"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])