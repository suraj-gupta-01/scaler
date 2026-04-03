"""
Evaluation Script for Adaptive Alert Triage Environment
========================================================

Runs baseline agents on all tasks and computes performance metrics.

BUGS FIXED vs original:
  1. HardTaskGrader(correlation_chains=[]) — HardTaskGrader takes NO __init__
     args (chains come dynamically via update_correlation_state).
     Fixed: grader = HardTaskGrader()
  2. grader.record_system_failure() doesn't exist — the method is
     record_failures(n: int). Fixed to use the correct API.
  3. Success thresholds were wrong (medium=0.65, hard=0.60) — the actual
     grader constants are medium=0.55, hard=0.50.  Fixed to import from
     the grader modules.
  4. evaluate_agent_on_task consumed only processed_alerts[0] per step —
     env.step() may produce multiple processed alerts when there are batch
     actions. Fixed to iterate the full list.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

import numpy as np

from adaptive_alert_triage.env   import AdaptiveAlertTriageEnv
from agents.baseline import RuleBasedAgent, ImprovedRuleBasedAgent

from tasks.easy   import EasyTaskGrader,   SUCCESS_THRESHOLD as EASY_THRESH
from tasks.medium import MediumTaskGrader, SUCCESS_THRESHOLD as MED_THRESH
from tasks.hard   import HardTaskGrader,   SUCCESS_THRESHOLD as HARD_THRESH

_THRESHOLDS = {"easy": EASY_THRESH, "medium": MED_THRESH, "hard": HARD_THRESH}


# ──────────────────────────────────────────────────────────────────────────────
# Core evaluation function
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_agent_on_task(
    agent,
    task_id: str,
    num_episodes: int = 10,
    seed_start:   int = 0,
    verbose:      bool = False,
) -> Dict[str, Any]:
    """
    Evaluate any agent on a specific task using the official task graders.

    Args:
        agent:        Agent with .act(observation) → Action method.
        task_id:      "easy", "medium", or "hard".
        num_episodes: Episodes to run.
        seed_start:   Starting random seed.
        verbose:      Print per-episode stats.

    Returns:
        Dict with mean_score, std_score, success_rate, episode_scores, …
    """
    env          = AdaptiveAlertTriageEnv(task_id=task_id)
    is_hard      = task_id == "hard"
    threshold    = _THRESHOLDS[task_id]

    episode_scores   = []
    episode_rewards  = []
    episode_lengths  = []
    episode_failures = []

    for ep in range(num_episodes):
        obs = env.reset(seed=seed_start + ep)

        # ── Grader init (BUG FIX: HardTaskGrader takes NO args) ──────
        if task_id == "medium":
            grader = MediumTaskGrader(max_investigations_per_step=3)
        elif task_id == "hard":
            grader = HardTaskGrader()          # was wrongly HardTaskGrader(correlation_chains=[])
        else:
            grader = EasyTaskGrader()

        if hasattr(agent, "reset"):
            agent.reset()

        done         = False
        total_reward = 0.0
        steps        = 0

        while not done:
            if not obs.alerts:
                break

            try:
                action = agent.act(obs)
            except Exception as exc:
                if verbose:
                    print(f"  Agent error at step {steps}: {exc}")
                break

            next_obs, reward, done, info = env.step(action)

            # ── Hard task: update correlation state FIRST ─────────────
            if is_hard:
                grader.update_correlation_state(info.get("correlation_groups", []))

            # ── Grade every processed alert (BUG FIX: iterate all) ───
            for alert_data in info.get("processed_alerts", []):
                grader.process_step(alert_data, info)

            # ── Record failures (BUG FIX: correct method name + sig) ─
            if is_hard:
                grader.record_failures(info.get("failures_this_step", 0))   # was record_system_failure()

            total_reward += reward.value
            steps        += 1
            obs           = next_obs

        score = grader.get_episode_score()
        episode_scores.append(score)
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        episode_failures.append(env.failures_count)

        if verbose:
            print(
                f"  ep {ep+1:3d}  score={score:.3f}  "
                f"reward={total_reward:+7.1f}  "
                f"steps={steps}  failures={env.failures_count}"
            )

    arr = np.array(episode_scores)
    return {
        "task_id":        task_id,
        "num_episodes":   num_episodes,
        "mean_score":     float(arr.mean()),
        "std_score":      float(arr.std()),
        "min_score":      float(arr.min()),
        "max_score":      float(arr.max()),
        "success_rate":   float((arr >= threshold).mean()),
        "mean_reward":    float(np.mean(episode_rewards)),
        "std_reward":     float(np.std(episode_rewards)),
        "mean_length":    float(np.mean(episode_lengths)),
        "mean_failures":  float(np.mean(episode_failures)),
        "episode_scores":   episode_scores,
        "episode_rewards":  episode_rewards,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Full multi-agent evaluation
# ──────────────────────────────────────────────────────────────────────────────

def run_full_evaluation(
    num_episodes: int = 10,
    verbose:      bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Run all baseline agents on all tasks."""
    agents = {
        "RuleBased":              RuleBasedAgent(),
        "ImprovedRuleBased":      ImprovedRuleBasedAgent(),
        "RuleBased_ResourceAware": RuleBasedAgent(resource_aware=True),
    }
    all_results: Dict[str, Dict[str, Any]] = {}

    for agent_name, agent in agents.items():
        if verbose:
            print(f"\n{'='*60}\nEvaluating: {agent_name}\n{'='*60}")
        agent_results: Dict[str, Any] = {}

        for task_id in ("easy", "medium", "hard"):
            if verbose:
                print(f"\n--- Task: {task_id} ---")
            res = evaluate_agent_on_task(
                agent=agent,
                task_id=task_id,
                num_episodes=num_episodes,
                verbose=verbose,
            )
            agent_results[task_id] = res
            if verbose:
                print(f"  mean={res['mean_score']:.3f}  "
                      f"success={res['success_rate']:.0%}  "
                      f"reward={res['mean_reward']:.1f}")

        all_results[agent_name] = agent_results

    return all_results


# ──────────────────────────────────────────────────────────────────────────────
# Display helpers
# ──────────────────────────────────────────────────────────────────────────────

def print_summary_table(all_results: Dict[str, Dict[str, Any]]) -> None:
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    header = f"{'Agent':<28} {'Task':<10} {'Mean±Std':>14}  {'Pass%':>8}  {'Failures':>9}"
    print(header)
    print("-" * 80)
    for agent_name, agent_results in all_results.items():
        for task_id, res in agent_results.items():
            print(
                f"{(agent_name if task_id == 'easy' else ''):<28} "
                f"{task_id:<10} "
                f"{res['mean_score']:.3f}±{res['std_score']:.3f}  "
                f"{res['success_rate']:>8.0%}  "
                f"{res['mean_failures']:>9.2f}"
            )
        print()


def save_results(
    all_results: Dict[str, Dict[str, Any]],
    filename: str = "evaluation_results.json",
) -> None:
    def _cvt(obj):
        if isinstance(obj, np.ndarray):     return obj.tolist()
        if isinstance(obj, (np.int64, np.int32)):   return int(obj)
        if isinstance(obj, (np.float64, np.float32)): return float(obj)
        if isinstance(obj, dict):  return {k: _cvt(v) for k, v in obj.items()}
        if isinstance(obj, list):  return [_cvt(v) for v in obj]
        return obj

    with open(filename, "w") as f:
        json.dump(_cvt(all_results), f, indent=2)
    print(f"\nResults saved → {filename}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate baseline agents on Adaptive Alert Triage"
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--task",     choices=["easy", "medium", "hard", "all"], default="all")
    parser.add_argument("--agent",    choices=["rule", "improved", "resource", "all"], default="all")
    parser.add_argument("--verbose",  action="store_true")
    parser.add_argument("--output",   default="evaluation_results.json")
    args = parser.parse_args()

    print(f"Adaptive Alert Triage — Baseline Evaluation")
    print(f"Episodes/task: {args.episodes}  Task: {args.task}  Agent: {args.agent}\n")

    if args.task == "all" and args.agent == "all":
        all_results = run_full_evaluation(num_episodes=args.episodes, verbose=args.verbose)
    else:
        agents_map = {
            "rule":     ("RuleBased",               RuleBasedAgent()),
            "improved": ("ImprovedRuleBased",        ImprovedRuleBasedAgent()),
            "resource": ("RuleBased_ResourceAware",  RuleBasedAgent(resource_aware=True)),
        }
        agents = (
            {n: a for _, (n, a) in agents_map.items()}
            if args.agent == "all"
            else {agents_map[args.agent][0]: agents_map[args.agent][1]}
        )
        tasks  = ("easy", "medium", "hard") if args.task == "all" else (args.task,)
        all_results = {}
        for agent_name, agent in agents.items():
            agent_results = {}
            for task_id in tasks:
                agent_results[task_id] = evaluate_agent_on_task(
                    agent=agent, task_id=task_id,
                    num_episodes=args.episodes, verbose=args.verbose,
                )
            all_results[agent_name] = agent_results

    print_summary_table(all_results)
    if args.output:
        save_results(all_results, args.output)
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()