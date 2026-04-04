"""
train_rl.py — Training + Evaluation Script for Adaptive Alert Triage
=====================================================================

Runs:
1. Rule-based baselines (RuleBasedAgent, ImprovedRuleBasedAgent)
2. PPO RL agent training across all 3 tasks
3. Saves all results to results.json for the comparison plot

Changes vs previous version:
    - Per-task episode budgets: hard gets 3× more episodes than easy
      because it has 40% chain probability and needs far more samples
      to observe enough chain outcomes for the policy to converge.
    - --episodes arg now sets the EASY budget; medium and hard scale up.
    - Grader is now passed into trainer.train() so terminal scores are
      injected into trajectories (was only logged after, not used).

Usage:
    python train_rl.py [--episodes 300] [--eval-episodes 20] [--seed 42]
"""

from __future__ import annotations

import json
import sys
import os
import argparse
import time
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from adaptive_alert_triage.env    import AdaptiveAlertTriageEnv
from adaptive_alert_triage.models import Action

from tasks.easy   import EasyTaskGrader,   SUCCESS_THRESHOLD as EASY_THRESH
from tasks.medium import MediumTaskGrader, SUCCESS_THRESHOLD as MED_THRESH
from tasks.hard   import HardTaskGrader,   SUCCESS_THRESHOLD as HARD_THRESH

from rl_agent import PPOTrainer, encode_state, _ACTION_NAMES

_GRADER_REGISTRY = {
    "easy":   (EasyTaskGrader,   {},              EASY_THRESH),
    "medium": (MediumTaskGrader, {"max_investigations_per_step": 3}, MED_THRESH),
    "hard":   (HardTaskGrader,   {},              HARD_THRESH),
}

# Per-task episode budgets.
# Hard needs ~3× easy: 40% chain probability over 50 steps means the agent
# sees ~20 chain alerts per episode but needs hundreds of episodes to learn
# which observable features (age, type, severity pattern) correlate with
# chain membership. Medium gets 1.5× for the resource-constraint curriculum.
_EPISODE_SCALE = {
    "easy":   1.0,
    "medium": 1.5,
    "hard":   3.0,
}


# ── Rule-based agents ─────────────────────────────────────────────────────

class SimpleRuleAgent:
    name = "RuleBased"

    def act(self, obs):
        if not obs.alerts:
            raise ValueError("No alerts")
        alert = max(obs.alerts, key=lambda a: a.visible_severity)
        sev, conf = alert.visible_severity, alert.confidence
        budget = obs.resource_budget

        if sev > 0.75 and conf > 0.70:
            if budget is not None and budget <= 0:
                return Action(alert_id=alert.id, action_type="ESCALATE")
            return Action(alert_id=alert.id, action_type="INVESTIGATE")
        if conf < 0.30:
            return Action(alert_id=alert.id, action_type="IGNORE")
        if sev > 0.55:
            return Action(alert_id=alert.id, action_type="ESCALATE")
        return Action(alert_id=alert.id, action_type="DELAY")

    def reset(self):
        pass


class ImprovedRuleAgent:
    name = "ImprovedRule"

    def act(self, obs):
        if not obs.alerts:
            raise ValueError("No alerts")

        def score(a):
            s = a.visible_severity * 2.0 + a.age * 0.08
            if a.alert_type == "SECURITY":
                s += 0.1
            return s

        alert = max(obs.alerts, key=score)
        sev, conf, age = alert.visible_severity, alert.confidence, alert.age
        budget, sys_load = obs.resource_budget, obs.system_load

        if age >= 3 and sev > 0.70:
            if budget is not None and budget <= 0:
                return Action(alert_id=alert.id, action_type="ESCALATE")
            return Action(alert_id=alert.id, action_type="INVESTIGATE")
        if sys_load > 0.85:
            if sev > 0.85 and conf > 0.80:
                return Action(alert_id=alert.id, action_type="INVESTIGATE")
            return Action(alert_id=alert.id, action_type="DELAY")
        if sev > 0.75 and conf > 0.70:
            if budget is not None and budget <= 0:
                return Action(alert_id=alert.id, action_type="ESCALATE")
            return Action(alert_id=alert.id, action_type="INVESTIGATE")
        if conf < 0.30:
            return Action(alert_id=alert.id, action_type="IGNORE")
        if sev > 0.55:
            return Action(alert_id=alert.id, action_type="ESCALATE")
        return Action(alert_id=alert.id, action_type="DELAY")

    def reset(self):
        pass


# ── Evaluation ────────────────────────────────────────────────────────────

def evaluate_agent(agent, task_id: str, n_episodes: int, seed_offset: int = 0) -> Dict:
    grader_cls, grader_kwargs, threshold = _GRADER_REGISTRY[task_id]
    env    = AdaptiveAlertTriageEnv(task_id=task_id)
    scores = []
    is_hard = (task_id == "hard")

    for ep in range(n_episodes):
        grader = grader_cls(**grader_kwargs)
        if hasattr(agent, 'reset'):
            agent.reset()
        obs  = env.reset(seed=seed_offset + ep)
        done = False

        while not done:
            if not obs.alerts:
                break
            action = agent.act(obs)
            obs, _r, done, info = env.step(action)

            if is_hard:
                grader.update_correlation_state(info.get("correlation_groups", []))
            for ad in info.get("processed_alerts", []):
                grader.process_step(ad, info)
            if is_hard:
                grader.record_failures(info.get("failures_this_step", 0))

        scores.append(grader.get_episode_score())

    arr = np.array(scores)
    return {
        "mean": float(arr.mean()),
        "std":  float(arr.std()),
        "min":  float(arr.min()),
        "max":  float(arr.max()),
        "success_rate": float((arr >= threshold).mean()),
        "scores": scores,
    }


# ── PPO wrapper ───────────────────────────────────────────────────────────

class PPOAgentWrapper:
    def __init__(self, trainer: PPOTrainer):
        self._trainer = trainer
        self.name = "PPO_LSTM"

    def act(self, obs):
        return self._trainer.act(obs)

    def reset(self):
        self._trainer.reset()


# ── Main ──────────────────────────────────────────────────────────────────

def run(args):
    results = {}

    for task_id in ["easy", "medium", "hard"]:
        grader_cls, grader_kwargs, threshold = _GRADER_REGISTRY[task_id]

        # Per-task episode budget
        n_episodes = int(args.episodes * _EPISODE_SCALE[task_id])

        print(f"\n{'='*60}")
        print(f"TASK: {task_id.upper()}  "
              f"(threshold ≥ {threshold}, episodes = {n_episodes})")
        print(f"{'='*60}")

        # 1. Rule-based baselines
        print(f"\n[1/3] Evaluating rule-based baselines…")
        rb_basic_res    = evaluate_agent(SimpleRuleAgent(),    task_id, args.eval_episodes, seed_offset=100)
        rb_improved_res = evaluate_agent(ImprovedRuleAgent(),  task_id, args.eval_episodes, seed_offset=100)

        print(f"  RuleBased    : mean={rb_basic_res['mean']:.3f}  "
              f"success={rb_basic_res['success_rate']:.0%}")
        print(f"  ImprovedRule : mean={rb_improved_res['mean']:.3f}  "
              f"success={rb_improved_res['success_rate']:.0%}")

        # 2. PPO training (grader passed in so terminal scores hit trajectories)
        print(f"\n[2/3] Training PPO agent ({n_episodes} episodes)…")
        env     = AdaptiveAlertTriageEnv(task_id=task_id)
        trainer = PPOTrainer(task_id=task_id, seed=args.seed, lr=3e-4)
        
        weight_path = f"weights/ppo_{task_id}.json"
        if os.path.exists(weight_path):
            try:
                trainer.load(weight_path)
                print(f"  Resuming continuous learning! Loaded existing weights from {weight_path}.")
            except Exception as e:
                print(f"  Could not load existing weights (starting fresh): {e}")

        t0 = time.time()
        history = trainer.train(
            env,
            n_episodes    = n_episodes,
            grader_cls    = grader_cls,
            grader_kwargs = grader_kwargs,
            log_interval  = max(1, n_episodes // 10),
            verbose       = True,
        )
        elapsed = time.time() - t0
        print(f"  Training done in {elapsed:.1f}s")

        os.makedirs("weights", exist_ok=True)
        trainer.save(f"weights/ppo_{task_id}.json")

        # 3. PPO evaluation
        print(f"\n[3/3] Evaluating PPO agent ({args.eval_episodes} episodes)…")
        ppo_agent = PPOAgentWrapper(trainer)
        ppo_res   = evaluate_agent(ppo_agent, task_id, args.eval_episodes, seed_offset=200)

        print(f"  PPO          : mean={ppo_res['mean']:.3f}  "
              f"success={ppo_res['success_rate']:.0%}")

        results[task_id] = {
            "threshold":     threshold,
            "n_episodes":    n_episodes,
            "rule_basic":    rb_basic_res,
            "rule_improved": rb_improved_res,
            "ppo":           ppo_res,
            "training": {
                "episode_rewards": history["episode_rewards"],
                "episode_scores":  history["episode_scores"],
                "policy_losses":   history["policy_losses"],
                "entropies":       history["entropies"],
            },
        }

    # Save results
    class _NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.floating,)):  return float(obj)
            if isinstance(obj, (np.integer,)):   return int(obj)
            if isinstance(obj, np.ndarray):      return obj.tolist()
            return super().default(obj)

    os.makedirs("results", exist_ok=True)
    out_path = "results/comparison_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, cls=_NumpyEncoder)
    print(f"\n✓ Results saved to {out_path}")

    # Summary table
    print(f"\n{'='*60}")
    print("FINAL COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Task':<10} {'Agent':<16} {'Mean':>8} {'Std':>7} {'Pass%':>8}")
    print("─" * 55)
    for task_id, res in results.items():
        for name, key in [("RuleBased", "rule_basic"),
                          ("ImprovedRule", "rule_improved"),
                          ("PPO+LSTM", "ppo")]:
            r = res[key]
            print(f"{task_id:<10} {name:<16} "
                  f"{r['mean']:>8.3f} "
                  f"{r['std']:>7.3f} "
                  f"{r['success_rate']*100:>7.1f}%")
        print("─" * 55)

    return results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes",      type=int, default=300,
                   help="Episode budget for easy task; medium=1.5×, hard=3×")
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--seed",          type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)