#!/usr/bin/env python3
"""
Adaptive Alert Triage — RL Trainer (FULLY FIXED)

Root-cause fixes:
  1. task_score: server never returns it — we compute it ourselves from
     action_correct ratio tracked per episode.
  2. SSL/ngrok drops: robust_request() retries with exponential back-off
     and re-creates the session on SSL/EOF errors.
  3. alert_type field: server returns "alert_type" in obs — handled.
  4. Removed stray self.ep reference that caused AttributeError.

Usage:
    python alert.py --count 500 --burst
    python train_external.py --task hard --timesteps 50000 --eval-eps 20
"""

import time
import requests
import numpy as np
import openenv as gym
from typing import Any, Dict, List, Tuple

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:
    print("ERROR: pip install stable-baselines3")
    exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

ACTION_TYPES     = ["INVESTIGATE", "IGNORE", "ESCALATE", "DELAY"]
ALERT_TYPES      = ["CPU", "MEMORY", "DISK", "NETWORK", "APPLICATION", "SECURITY"]
OBS_DIM          = 26
TASK_THRESHOLDS  = {"easy": 0.70, "medium": 0.65, "hard": 0.60}


# ─────────────────────────────────────────────────────────────────────────────
# Robust HTTP helper  (fixes SSL / ngrok drops)
# ─────────────────────────────────────────────────────────────────────────────

def robust_request(method: str, url: str, max_retries: int = 6, **kwargs) -> requests.Response:
    """
    Retry with exponential back-off.
    Re-creates Session on SSL/EOF errors (ngrok tunnel reset).
    """
    kwargs.setdefault("timeout", 20)
    session   = requests.Session()
    last_exc  = None

    for attempt in range(max_retries):
        try:
            return session.request(method, url, **kwargs)
        except Exception as exc:
            last_exc = exc
            wait     = min(2 ** attempt, 30)
            err      = str(exc)
            if attempt < max_retries - 1:
                print(f"  ⚠  [{url.split('/')[-1]}] attempt {attempt+1}: "
                      f"{err[:80]} — retry in {wait}s")
                time.sleep(wait)
                if any(k in err for k in ("SSL", "EOF", "RemoteDisconnected", "Connection")):
                    session = requests.Session()

    raise last_exc


# ─────────────────────────────────────────────────────────────────────────────
# Score computation  (fixes missing task_score key)
# ─────────────────────────────────────────────────────────────────────────────

def compute_score(info: Dict, correct: int, steps: int) -> float:
    """
    Server never puts task_score in info.
    We derive it as correct_actions / total_steps.
    """
    if steps > 0:
        return round(correct / steps, 4)
    cum = float(info.get("cumulative_reward", 0.0))
    return min(cum / 500.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Remote Gymnasium Environment
# ─────────────────────────────────────────────────────────────────────────────

class RemoteEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, server_url: str, task_id: str = "hard"):
        super().__init__()
        self.server  = server_url.rstrip("/")
        self.task_id = task_id

        self.action_space      = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )

        self.current_alerts:   List[Dict] = []
        self.current_alert_id: str = ""
        self.episode_scores:   List[float] = []
        self._correct  = 0
        self._steps    = 0

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self._correct = 0
        self._steps   = 0
        try:
            resp = robust_request("POST", f"{self.server}/env/reset/{self.task_id}")
            resp.raise_for_status()
            obs_dict = resp.json().get("obs", {})
            self.current_alerts = obs_dict.get("alerts", [])
            self._pick_alert()
            return self._flatten_obs(obs_dict), {}
        except Exception as e:
            print(f"[RemoteEnv.reset] {e}")
            return np.zeros(OBS_DIM, dtype=np.float32), {}

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action_type = ACTION_TYPES[int(action_idx)]
        self._steps += 1
        try:
            resp = robust_request(
                "POST", f"{self.server}/env/step",
                json={"alert_id": self.current_alert_id, "action_type": action_type},
            )
            resp.raise_for_status()
            data     = resp.json()
            obs_dict = data.get("obs", {})
            reward   = float(data.get("reward", 0.0))
            done     = bool(data.get("done", False))
            info     = data.get("info", {})

            if info.get("action_correct", False):
                self._correct += 1

            self.current_alerts = obs_dict.get("alerts", [])
            self._pick_alert()

            if done:
                score = compute_score(info, self._correct, self._steps)
                self.episode_scores.append(score)
                info["task_score"] = score   # inject so SB3 Monitor/callback sees it

            return self._flatten_obs(obs_dict), reward, done, False, info

        except Exception as e:
            print(f"[RemoteEnv.step] {e}")
            return np.zeros(OBS_DIM, dtype=np.float32), -1.0, True, False, {}

    def render(self): pass
    def close(self):  pass

    def _pick_alert(self):
        if not self.current_alerts:
            return
        best = max(self.current_alerts, key=lambda a: float(a.get("visible_severity", 0)))
        self.current_alert_id = best.get("id", "")

    def _flatten_obs(self, obs_dict: Dict) -> np.ndarray:
        values: List[float] = []
        alerts = obs_dict.get("alerts", [])
        for alert in (alerts[:5] + [{}] * 5)[:5]:
            values.append(float(alert.get("visible_severity", 0.0)))
            values.append(float(alert.get("confidence", 0.0)))
            values.append(min(float(alert.get("age", 0)) / 50.0, 1.0))
            atype = alert.get("alert_type", alert.get("type", "CPU"))
            values.append(
                ALERT_TYPES.index(atype) / len(ALERT_TYPES)
                if atype in ALERT_TYPES else 0.0
            )
        values.append(float(obs_dict.get("system_load", 0.0)))
        values.append(min(float(obs_dict.get("queue_length",   0))  / 50.0, 1.0))
        values.append(min(float(obs_dict.get("time_remaining", 0))  / 50.0, 1.0))
        values.append(min(float(obs_dict.get("resource_budget") or 0) / 5.0, 1.0))
        values.append(min(float(obs_dict.get("failures_count", 0))  / 5.0,  1.0))
        values.append(min(float(obs_dict.get("episode_step",   0))  / 50.0, 1.0))
        return np.clip(np.array(values[:OBS_DIM], dtype=np.float32), 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Rule-Based Agent
# ─────────────────────────────────────────────────────────────────────────────

class RuleBasedAgent:
    def act(self, obs_dict: Dict) -> Tuple[str, str]:
        alerts = obs_dict.get("alerts", [])
        if not alerts:
            return "", "DELAY"
        alert      = max(alerts, key=lambda a: float(a.get("visible_severity", 0)))
        alert_id   = alert.get("id", "")
        severity   = float(alert.get("visible_severity", 0.5))
        confidence = float(alert.get("confidence", 0.5))
        budget     = obs_dict.get("resource_budget")

        if severity > 0.8 and confidence > 0.7:
            if budget is not None and float(budget) <= 0:
                return alert_id, "ESCALATE"
            return alert_id, "INVESTIGATE"
        elif confidence < 0.3:
            return alert_id, "IGNORE"
        elif severity > 0.6:
            return alert_id, "ESCALATE"
        else:
            return alert_id, "DELAY"


def run_rule_based_episodes(server: str, task_id: str, n_episodes: int) -> Dict[str, Any]:
    agent           = RuleBasedAgent()
    episode_scores: List[float] = []
    episode_rewards:List[float] = []

    print(f"\n[RuleBased] Running {n_episodes} episodes on '{task_id}' task...")

    for ep in range(n_episodes):
        try:
            resp = robust_request("POST", f"{server}/env/reset/{task_id}")
            resp.raise_for_status()
            obs_dict = resp.json().get("obs", {})
        except Exception as e:
            print(f"  [ep {ep}] reset error: {e}")
            continue

        total_reward = 0.0
        correct      = 0
        steps        = 0
        done         = False

        while not done:
            alert_id, action_type = agent.act(obs_dict)
            if not alert_id:
                break
            try:
                resp = robust_request(
                    "POST", f"{server}/env/step",
                    json={"alert_id": alert_id, "action_type": action_type},
                )
                resp.raise_for_status()
                data          = resp.json()
                obs_dict      = data.get("obs", {})
                reward        = float(data.get("reward", 0.0))
                done          = bool(data.get("done", False))
                info          = data.get("info", {})
                total_reward += reward
                steps        += 1
                if info.get("action_correct", False):
                    correct += 1
                if done:
                    episode_scores.append(compute_score(info, correct, steps))
            except Exception as e:
                print(f"  [ep {ep}] step error: {e}")
                break

        episode_rewards.append(total_reward)
        if (ep + 1) % 5 == 0:
            ms = np.mean(episode_scores) if episode_scores else 0.0
            print(f"  ep {ep+1:3d}/{n_episodes}  mean_score={ms:.3f}")

    return {
        "episode_scores":  episode_scores,
        "episode_rewards": episode_rewards,
        "mean_score":  float(np.mean(episode_scores))  if episode_scores  else 0.0,
        "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "std_score":   float(np.std(episode_scores))   if episode_scores  else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SB3 Callback
# ─────────────────────────────────────────────────────────────────────────────

class ScoreCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.training_scores: List[float] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "task_score" in info:
                self.training_scores.append(float(info["task_score"]))
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(rl_train_scores, rl_eval_scores, rule_scores,
                    rl_eval_rewards, rule_rewards, task_id, output_path):
    threshold = TASK_THRESHOLDS.get(task_id, 0.60)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"RL (PPO) vs Rule-Based — Task: {task_id.upper()}",
                 fontsize=15, fontweight="bold")

    ax = axes[0, 0]
    if rl_train_scores:
        w = max(1, len(rl_train_scores) // 20)
        sm = np.convolve(rl_train_scores, np.ones(w)/w, mode="valid")
        ax.plot(rl_train_scores, alpha=0.25, color="steelblue", label="Raw")
        ax.plot(range(w-1, len(rl_train_scores)), sm, color="steelblue",
                lw=2, label=f"Smoothed (w={w})")
    ax.axhline(threshold, color="red", ls="--", lw=1.2, label=f"Threshold ({threshold})")
    ax.set_title("PPO training — task score"); ax.set_xlabel("Episode")
    ax.set_ylabel("Task score"); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    if rl_eval_scores:
        ax.plot(rl_eval_scores, color="steelblue", marker="o", markersize=4,
                label=f"PPO (mean={np.mean(rl_eval_scores):.3f})")
    if rule_scores:
        ax.plot(rule_scores, color="tomato", marker="s", markersize=4,
                label=f"Rule-Based (mean={np.mean(rule_scores):.3f})")
    ax.axhline(threshold, color="black", ls="--", lw=1)
    ax.set_title("Eval: episodic score (head-to-head)"); ax.set_xlabel("Episode")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    means = [np.mean(rl_eval_rewards) if rl_eval_rewards else 0,
             np.mean(rule_rewards)    if rule_rewards     else 0]
    stds  = [np.std(rl_eval_rewards)  if rl_eval_rewards else 0,
             np.std(rule_rewards)     if rule_rewards     else 0]
    bars = ax.bar(["PPO","Rule-Based"], means, yerr=stds, capsize=8,
                  color=["steelblue","tomato"], alpha=0.8, edgecolor="black")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f"{m:.1f}", ha="center", va="bottom", fontsize=10)
    ax.set_title("Mean episode reward ± std"); ax.set_ylabel("Total reward")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 1]
    if rl_eval_scores:
        ax.hist(rl_eval_scores, bins=12, alpha=0.6, color="steelblue",
                label="PPO", edgecolor="white")
    if rule_scores:
        ax.hist(rule_scores, bins=12, alpha=0.6, color="tomato",
                label="Rule-Based", edgecolor="white")
    ax.axvline(threshold, color="black", ls="--", lw=1.2)
    ax.set_title("Score distribution"); ax.set_xlabel("Task score")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n✅  Plot saved → {output_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def wait_for_server(url: str, attempts: int = 30) -> bool:
    print(f"Waiting for server at {url} ...")
    for i in range(attempts):
        try:
            r = robust_request("GET", f"{url}/health", max_retries=2)
            if r.status_code == 200 and r.json().get("status") == "ok":
                print(f"✅  Server ready (attempt {i+1})")
                return True
        except Exception:
            pass
        print(f"  attempt {i+1}/{attempts}...")
        time.sleep(3)
    print("❌  Server not ready"); return False


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--server",    default="http://localhost:8000")  # remote: https://scalar-hackathon.onrender.com
    p.add_argument("--task",      default="hard", choices=["easy","medium","hard"])
    p.add_argument("--timesteps", type=int, default=50_000)
    p.add_argument("--eval-eps",  type=int, default=20)
    p.add_argument("--output",    default="rl_vs_baseline.png")
    args = p.parse_args()

    print("=" * 65)
    print("  Adaptive Alert Triage — RL Trainer + Baseline Comparison")
    print("=" * 65)
    print(f"  Server:     {args.server}")
    print(f"  Task:       {args.task}")
    print(f"  Timesteps:  {args.timesteps:,}")
    print(f"  Eval eps:   {args.eval_eps}")

    if not wait_for_server(args.server):
        return

    # Step 1 — Rule-based baseline
    print("\n" + "─"*65)
    print("STEP 1 / 3 — Rule-based baseline")
    print("─"*65)
    rb = run_rule_based_episodes(args.server, args.task, args.eval_eps)
    print(f"\n  Rule-Based  mean_score={rb['mean_score']:.4f}  "
          f"±{rb['std_score']:.4f}  mean_reward={rb['mean_reward']:.1f}")

    # Step 2 — PPO training
    print("\n" + "─"*65)
    print("STEP 2 / 3 — PPO training")
    print("─"*65)
    train_env = Monitor(RemoteEnv(server_url=args.server, task_id=args.task))
    score_cb  = ScoreCallback()
    model = PPO(
        "MlpPolicy", train_env, verbose=1,
        n_steps=512, batch_size=64, n_epochs=10,
        learning_rate=3e-4, gamma=0.99, ent_coef=0.01,
    )
    model.learn(total_timesteps=args.timesteps, callback=score_cb)
    model.save(f"ppo_{args.task}_triage")
    print(f"\n✅  Model saved → ppo_{args.task}_triage.zip")

    # Step 3 — PPO eval
    print("\n" + "─"*65)
    print("STEP 3 / 3 — PPO evaluation")
    print("─"*65)
    eval_env        = RemoteEnv(server_url=args.server, task_id=args.task)
    rl_eval_scores: List[float] = []
    rl_eval_rewards:List[float] = []
    obs, _    = eval_env.reset()
    ep_reward = 0.0
    ep_count  = 0

    while ep_count < args.eval_eps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = eval_env.step(action)
        ep_reward += reward
        if done:
            ep_count += 1
            rl_eval_rewards.append(ep_reward)
            if "task_score" in info:
                rl_eval_scores.append(float(info["task_score"]))
            ep_reward = 0.0
            obs, _ = eval_env.reset()
            if ep_count % 5 == 0:
                ms = np.mean(rl_eval_scores) if rl_eval_scores else 0.0
                print(f"  eval ep {ep_count:3d}/{args.eval_eps}  mean_score={ms:.3f}")

    # Results
    rl_mean   = np.mean(rl_eval_scores)  if rl_eval_scores  else 0.0
    rl_std    = np.std(rl_eval_scores)   if rl_eval_scores  else 0.0
    threshold = TASK_THRESHOLDS.get(args.task, 0.60)
    rl_ok = sum(1 for s in rl_eval_scores       if s >= threshold)
    rb_ok = sum(1 for s in rb["episode_scores"] if s >= threshold)
    n     = args.eval_eps

    print("\n" + "="*65)
    print("  RESULTS")
    print("="*65)
    print(f"  {'Metric':<28} {'PPO':>12}   {'Rule-Based':>12}")
    print(f"  {'-'*52}")
    print(f"  {'Mean task score':<28} {rl_mean:>12.4f}   {rb['mean_score']:>12.4f}")
    print(f"  {'Std task score':<28} {rl_std:>12.4f}   {rb['std_score']:>12.4f}")
    print(f"  {'Mean reward':<28} "
          f"{np.mean(rl_eval_rewards) if rl_eval_rewards else 0:>12.1f}"
          f"   {rb['mean_reward']:>12.1f}")
    print(f"  {'Success rate (>={threshold})':<28} {rl_ok/n:>11.1%}   {rb_ok/n:>11.1%}")
    delta = rl_mean - rb["mean_score"]
    print(f"  {'Improvement':<28} {'+' if delta>=0 else ''}{delta*100:>10.2f}%")
    print("="*65)

    if HAS_MPL:
        plot_comparison(
            score_cb.training_scores, rl_eval_scores,
            rb["episode_scores"], rl_eval_rewards,
            rb["episode_rewards"], args.task, args.output,
        )
    print("\n✅  Done!")


if __name__ == "__main__":
    main()