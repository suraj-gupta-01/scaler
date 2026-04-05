"""
PPO + LSTM Reinforcement Learning Agent for Adaptive Alert Triage
=================================================================

Architecture (per RL_AGENT_METHODOLOGY.txt):
    Input → MLP Feature Encoder → LSTM → Policy Head + Value Head

Training:
    - PPO with clipped objective
    - GAE (Generalized Advantage Estimation)
    - Adam optimizer
    - Entropy regularization for exploration

State vector (20 features total):
    Per primary alert (highest visible_severity):
        [visible_severity, confidence, alert_type_one_hot(6),
         age_ratio, sev_x_conf, is_chain_type, budget_pressure]
        = 12 features

    Queue-level context:
        [system_load, queue_norm, time_ratio,
         max_age_ratio, mean_sev, n_chain_type_norm, budget_norm]
        = 7 features

    Budget flag:
        [has_budget]
        = 1 feature

    Total = 20

Fixes vs previous version:
    - state_dim corrected to 20 (was 12, encode_state returned 16 → crash)
    - Alert selection decoupled from action: agent picks BOTH alert AND action
      via a joint (alert_idx, action) softmax over top-K alerts
    - Removed duplicate age feature (was encoded at /10 AND /5 simultaneously)
    - Terminal grader score injected into final trajectory reward before GAE
    - Queue-context features added so agent sees full alert landscape per step
"""

from __future__ import annotations

import numpy as np
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# ── Minimal pure-numpy neural net ─────────────────────────────────────────

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


# ── LSTM cell ─────────────────────────────────────────────────────────────

class LSTMCell:
    """Single LSTM cell with Xavier-initialised weights."""

    def __init__(self, input_dim: int, hidden_dim: int, rng: np.random.Generator) -> None:
        self.hidden_dim = hidden_dim
        scale = np.sqrt(2.0 / (input_dim + hidden_dim))
        self.W = rng.normal(0, scale, (4 * hidden_dim, input_dim + hidden_dim))
        self.b = np.zeros(4 * hidden_dim)
        self.b[hidden_dim:2*hidden_dim] = 1.0  # forget gate bias = 1

    def forward(self, x: np.ndarray, h: np.ndarray, c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        combined = np.concatenate([x, h])
        gates = self.W @ combined + self.b
        hd = self.hidden_dim
        f = _sigmoid(gates[0*hd:1*hd])
        i = _sigmoid(gates[1*hd:2*hd])
        g = _tanh(   gates[2*hd:3*hd])
        o = _sigmoid(gates[3*hd:4*hd])
        c_new = f * c + i * g
        h_new = o * _tanh(c_new)
        return h_new, c_new


# ── Linear layer ──────────────────────────────────────────────────────────

class Linear:
    def __init__(self, in_dim: int, out_dim: int, rng: np.random.Generator) -> None:
        scale = np.sqrt(2.0 / in_dim)
        self.W = rng.normal(0, scale, (out_dim, in_dim))
        self.b = np.zeros(out_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.W @ x + self.b


# ── Policy + Value network ────────────────────────────────────────────────

class PPONetwork:
    """
    Actor-Critic: encoder → LSTM → policy_head (4 logits) + value_head (scalar).

    state_dim MUST match the output length of encode_state() exactly.
    Current value: 20.
    """

    ACTION_DIM = 4   # INVESTIGATE, IGNORE, ESCALATE, DELAY

    def __init__(
        self,
        state_dim:   int = 20,   # must match encode_state() output length
        encoder_dim: int = 64,
        lstm_dim:    int = 64,
        seed:        int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.enc1 = Linear(state_dim,   encoder_dim, rng)
        self.enc2 = Linear(encoder_dim, encoder_dim, rng)
        self.lstm = LSTMCell(encoder_dim, lstm_dim, rng)
        self.policy_head = Linear(lstm_dim, self.ACTION_DIM, rng)
        self.value_head  = Linear(lstm_dim, 1, rng)
        self.h = np.zeros(lstm_dim)
        self.c = np.zeros(lstm_dim)

    def reset_hidden(self) -> None:
        self.h = np.zeros_like(self.h)
        self.c = np.zeros_like(self.c)

    def forward(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        x = _relu(self.enc1.forward(state))
        x = _relu(self.enc2.forward(x))
        self.h, self.c = self.lstm.forward(x, self.h, self.c)
        logits = self.policy_head.forward(self.h)
        value  = float(self.value_head.forward(self.h)[0])
        return _softmax(logits), value

    def get_params(self) -> List[np.ndarray]:
        return [
            self.enc1.W, self.enc1.b,
            self.enc2.W, self.enc2.b,
            self.lstm.W, self.lstm.b,
            self.policy_head.W, self.policy_head.b,
            self.value_head.W,  self.value_head.b,
        ]

    def set_params(self, params: List[np.ndarray]) -> None:
        (self.enc1.W, self.enc1.b,
         self.enc2.W, self.enc2.b,
         self.lstm.W, self.lstm.b,
         self.policy_head.W, self.policy_head.b,
         self.value_head.W,  self.value_head.b) = params

    def copy_params(self) -> List[np.ndarray]:
        return [p.copy() for p in self.get_params()]


# ── Constants ─────────────────────────────────────────────────────────────

_ALERT_TYPE_MAP = {
    "CPU": 0, "MEMORY": 1, "DISK": 2,
    "NETWORK": 3, "APPLICATION": 4, "SECURITY": 5,
}
_ACTION_NAMES = ["INVESTIGATE", "IGNORE", "ESCALATE", "DELAY"]

# Alert types that commonly appear as chain triggers in CORRELATION_CHAINS
_CHAIN_TRIGGER_TYPES = frozenset({"CPU", "MEMORY", "NETWORK", "DISK"})

# Must match utils.CRITICAL_AGE_THRESHOLD
_CRITICAL_AGE_THRESHOLD = 5

# Hard task success threshold (must match hard.py SUCCESS_THRESHOLD)
_HARD_SUCCESS_THRESHOLD = 0.50
_EASY_SUCCESS_THRESHOLD = 0.70
_MEDIUM_SUCCESS_THRESHOLD = 0.65

_TASK_THRESHOLDS = {
    "easy":   _EASY_SUCCESS_THRESHOLD,
    "medium": _MEDIUM_SUCCESS_THRESHOLD,
    "hard":   _HARD_SUCCESS_THRESHOLD,
}


# ── State encoder ─────────────────────────────────────────────────────────

def encode_state(obs) -> np.ndarray:
    """
    Convert an Observation into a flat 20-element numpy feature vector.

    Layout:
        [0]     primary.visible_severity
        [1]     primary.confidence
        [2-7]   primary.alert_type one-hot (6 classes)
        [8]     age_ratio  = min(age / CRITICAL_AGE_THRESHOLD, 1.0)
                             (single age feature, normalised to failure threshold)
        [9]     sev_x_conf = visible_severity * confidence
        [10]    is_chain_type (1 if CPU/MEMORY/NETWORK/DISK else 0)
        [11]    budget_pressure = 1 - resource_budget/3  (0 if unconstrained)
        [12]    system_load
        [13]    queue_norm = min(queue_length / 10, 1.0)
        [14]    time_ratio = time_remaining / max_steps  (approx via /50)
        [15]    max_age_ratio across all alerts
        [16]    mean_visible_severity across all alerts
        [17]    n_chain_type_norm = fraction of alerts that are chain-trigger types
        [18]    budget_norm = resource_budget / 3  (0 if unconstrained)
        [19]    has_budget flag (1 if resource-constrained task, else 0)

    Total: 20 features. Must stay in sync with PPONetwork(state_dim=20).
    """
    if not obs.alerts:
        return np.zeros(20, dtype=np.float32)

    # Primary alert: highest visible severity (the one the agent will act on)
    primary = max(obs.alerts, key=lambda a: a.visible_severity)

    # --- Per-primary features ---
    type_oh = np.zeros(6, dtype=np.float32)
    type_oh[_ALERT_TYPE_MAP.get(primary.alert_type, 4)] = 1.0

    # Single age feature, normalised to the failure threshold (not /10)
    # This directly encodes "fraction of time until this alert causes a failure"
    age_ratio = min(primary.age / _CRITICAL_AGE_THRESHOLD, 1.0)

    sev_x_conf = primary.visible_severity * primary.confidence

    is_chain_type = 1.0 if primary.alert_type in _CHAIN_TRIGGER_TYPES else 0.0

    if obs.resource_budget is not None:
        budget_pressure = 1.0 - obs.resource_budget / 3.0
        budget_norm = obs.resource_budget / 3.0
        has_budget = 1.0
    else:
        budget_pressure = 0.0
        budget_norm = 1.0   # unconstrained = full budget
        has_budget = 0.0

    # --- Queue-level context features ---
    all_ages = [a.age for a in obs.alerts]
    all_sevs = [a.visible_severity for a in obs.alerts]
    n_chain  = sum(1 for a in obs.alerts if a.alert_type in _CHAIN_TRIGGER_TYPES)

    max_age_ratio  = min(max(all_ages) / _CRITICAL_AGE_THRESHOLD, 1.0)
    mean_sev       = float(np.mean(all_sevs))
    n_chain_norm   = n_chain / max(len(obs.alerts), 1)
    queue_norm     = min(obs.queue_length / 10.0, 1.0)
    # time_ratio: approximate max_steps as 50 (hard); exact value not exposed in obs
    time_ratio     = min(obs.time_remaining / 50.0, 1.0)

    feat = np.array([
        # Primary alert (12 features)
        primary.visible_severity,   # 0
        primary.confidence,         # 1
        *type_oh,                   # 2-7
        age_ratio,                  # 8  (single, normalised to failure threshold)
        sev_x_conf,                 # 9
        is_chain_type,              # 10
        budget_pressure,            # 11
        # Queue context (7 features)
        obs.system_load,            # 12
        queue_norm,                 # 13
        time_ratio,                 # 14
        max_age_ratio,              # 15  max age across ALL alerts in queue
        mean_sev,                   # 16  mean severity across queue
        n_chain_norm,               # 17  fraction of chain-type alerts
        budget_norm,                # 18
        # Budget flag (1 feature)
        has_budget,                 # 19
    ], dtype=np.float32)

    assert len(feat) == 20, f"encode_state returned {len(feat)} features, expected 20"
    return feat


def _select_alert(obs, action_idx: int):
    """
    Choose which alert to act on given the chosen action type.

    Strategy (decoupled from the policy's action choice):
      - INVESTIGATE / ESCALATE: pick the alert with highest urgency score
        (severity * confidence, boosted by age proximity to failure threshold)
      - IGNORE: pick the alert most likely to be a false positive
        (lowest visible_severity * confidence)
      - DELAY: pick the alert with lowest current urgency (safest to defer)

    This is a fixed heuristic for alert selection. The policy learns WHAT
    to do; this function implements WHERE to apply it.  Separating them keeps
    the action space at 4 (not 4 × N_alerts) while still allowing meaningful
    alert targeting.
    """
    action = _ACTION_NAMES[action_idx]

    def urgency(a):
        age_factor = min(a.age / _CRITICAL_AGE_THRESHOLD, 1.0)
        return a.visible_severity * a.confidence * (1.0 + age_factor)

    if action in ("INVESTIGATE", "ESCALATE"):
        return max(obs.alerts, key=urgency)
    elif action == "IGNORE":
        # Prefer low-confidence, low-severity alerts (likely false positives)
        return min(obs.alerts, key=lambda a: a.visible_severity * a.confidence)
    else:  # DELAY
        # Prefer the least urgent alert — safest to defer
        return min(obs.alerts, key=urgency)


# ── PPO Trainer ───────────────────────────────────────────────────────────

class PPOTrainer:
    """
    PPO with GAE using pure numpy.

    Key parameters:
        gamma    = 0.99   discount factor
        lam      = 0.95   GAE lambda
        clip_eps = 0.20   PPO clip range
        ent_coef = 0.01   entropy coefficient (increased for hard task)
        lr       = 3e-4   Adam learning rate
        epochs   = 4      update epochs per rollout
    """

    def __init__(
        self,
        task_id:    str   = "easy",
        seed:       int   = 0,
        lr:         float = 3e-4,
        gamma:      float = 0.99,
        lam:        float = 0.95,
        clip_eps:   float = 0.20,
        ent_coef:   float = 0.01,
        vf_coef:    float = 0.50,
        epochs:     int   = 4,
        batch_size: int   = 32,
    ) -> None:
        self.task_id    = task_id
        self.gamma      = gamma
        self.lam        = lam
        self.clip_eps   = clip_eps
        self.vf_coef    = vf_coef
        self.epochs     = epochs
        self.batch_size = batch_size
        self.threshold  = _TASK_THRESHOLDS.get(task_id, 0.65)

        # Higher entropy for hard task: the policy must not collapse to
        # "always INVESTIGATE" before it has learned chain patterns
        if task_id == "hard":
            self.ent_coef = max(ent_coef, 0.15)  # Bumped to 0.15 to break 'investigate' habit
        elif task_id == "easy":
            self.ent_coef = max(ent_coef, 0.03)
        else:
            self.ent_coef = ent_coef

        # Network: state_dim=20 must match encode_state() output
        self.net = PPONetwork(state_dim=20, seed=seed)

        # Adam optimiser state
        self._m = [np.zeros_like(p) for p in self.net.get_params()]
        self._v = [np.zeros_like(p) for p in self.net.get_params()]
        self._t = 0
        self.lr = lr

        # Training history
        self.episode_rewards: List[float] = []
        self.episode_scores:  List[float] = []
        self.policy_losses:   List[float] = []
        self.value_losses:    List[float] = []
        self.entropies:       List[float] = []

    # ------------------------------------------------------------------
    # Episode rollout
    # ------------------------------------------------------------------

    def collect_episode(
        self,
        env,
        grader_cls=None,
        grader_kwargs: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Run one episode, collecting (s, a, r, v, logp) tuples.

        If grader_cls is provided, the grader score is computed at episode
        end and injected into the final transition reward before returning
        the trajectory. This closes the gap between dense per-step rewards
        and the sparse episode-level grader score.
        """
        from adaptive_alert_triage.models import Action

        self.net.reset_hidden()
        obs  = env.reset(seed=int(np.random.randint(0, 10000)))
        done = False

        is_hard  = self.task_id == "hard"
        grader   = None
        if grader_cls is not None:
            grader = grader_cls(**(grader_kwargs or {}))

        states, actions, rewards, values, log_probs = [], [], [], [], []
        total_reward = 0.0
        steps = 0

        while not done:
            if not obs.alerts:
                break

            s = encode_state(obs)
            probs, v = self.net.forward(s)

            # Sample action index (policy chooses WHAT to do)
            a = int(np.random.choice(4, p=probs))
            log_p = float(np.log(probs[a] + 1e-8))

            # Select WHICH alert to act on (heuristic, decoupled from policy)
            alert = _select_alert(obs, a)
            action_obj = Action(alert_id=alert.id, action_type=_ACTION_NAMES[a])

            obs, reward, done, info = env.step(action_obj)
            r = float(reward.value)

            # Update grader if available (needed for terminal injection below)
            if grader is not None:
                if is_hard:
                    grader.update_correlation_state(
                        info.get("correlation_groups", []))
                for ad in info.get("processed_alerts", []):
                    grader.process_step(ad, info)
                if is_hard:
                    grader.record_failures(info.get("failures_this_step", 0))

            states.append(s)
            actions.append(a)
            rewards.append(r)
            values.append(v)
            log_probs.append(log_p)

            total_reward += r
            steps += 1

        # --- Terminal grader-score injection ---
        # The grader computes a single score at episode end that directly
        # determines whether the agent "passed". We inject this as an extra
        # reward on the final transition so GAE backpropagates the signal
        # through the entire episode.
        if grader is not None and len(rewards) > 0:
            grader_score  = grader.get_episode_score()
            # Scale: (score - threshold) * 30 so passing gives +9 to +15,
            # failing gives -15 to -9. Large enough to dominate dense noise.
            terminal_bonus = (grader_score - self.threshold) * 30.0
            rewards[-1] += terminal_bonus
            total_reward += terminal_bonus

        # Bootstrap value for GAE
        if not done and obs.alerts:
            s_last = encode_state(obs)
            _, v_last = self.net.forward(s_last)
        else:
            v_last = 0.0

        return {
            "states":       np.array(states,    dtype=np.float32),
            "actions":      np.array(actions,   dtype=np.int32),
            "rewards":      np.array(rewards,   dtype=np.float32),
            "values":       np.array(values,    dtype=np.float32),
            "log_probs":    np.array(log_probs, dtype=np.float32),
            "v_last":       v_last,
            "total_reward": total_reward,
            "steps":        steps,
            "grader_score": grader.get_episode_score() if grader else 0.0,
        }

    # ------------------------------------------------------------------
    # GAE
    # ------------------------------------------------------------------

    def compute_gae(
        self,
        rewards: np.ndarray,
        values:  np.ndarray,
        v_last:  float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae    = 0.0
        next_v = v_last

        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * next_v - values[t]
            gae   = delta + self.gamma * self.lam * gae
            advantages[t] = gae
            next_v = values[t]

        returns = advantages + values
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    # ------------------------------------------------------------------
    # PPO loss + finite-difference gradient
    # ------------------------------------------------------------------

    def _compute_loss(
        self,
        states:     np.ndarray,
        actions:    np.ndarray,
        old_lp:     np.ndarray,
        advantages: np.ndarray,
        returns:    np.ndarray,
    ) -> Tuple[float, float, float]:
        total_pl = total_vl = total_en = 0.0
        self.net.reset_hidden()

        for s, a, olp, adv, ret in zip(states, actions, old_lp, advantages, returns):
            probs, v = self.net.forward(s)
            log_p = float(np.log(probs[a] + 1e-8))
            ratio = np.exp(log_p - olp)

            pl = -min(ratio * adv,
                      np.clip(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv)
            vl = (v - ret) ** 2
            en = -float(np.sum(probs * np.log(probs + 1e-8)))

            total_pl += pl
            total_vl += vl
            total_en += en

        n = max(len(states), 1)
        return total_pl / n, total_vl / n, total_en / n

    def _finite_diff_gradient(
        self,
        states: np.ndarray, actions: np.ndarray, old_lp: np.ndarray,
        advantages: np.ndarray, returns: np.ndarray,
        eps: float = 1e-3,
    ) -> List[np.ndarray]:
        params = self.net.get_params()
        grads  = []

        base_pl, base_vl, base_en = self._compute_loss(
            states, actions, old_lp, advantages, returns)
        base_loss = base_pl + self.vf_coef * base_vl - self.ent_coef * base_en

        for i, p in enumerate(params):
            flat      = p.flatten()
            grad_flat = np.zeros_like(flat)
            n_sample  = min(len(flat), 20)
            indices   = np.random.choice(len(flat), n_sample, replace=False)

            for idx in indices:
                flat[idx] += eps
                p[:] = flat.reshape(p.shape)
                self.net.set_params(params)

                pl, vl, en = self._compute_loss(
                    states, actions, old_lp, advantages, returns)
                loss_p = pl + self.vf_coef * vl - self.ent_coef * en

                grad_flat[idx] = (loss_p - base_loss) / eps
                flat[idx] -= eps
                p[:] = flat.reshape(p.shape)

            grads.append(grad_flat.reshape(p.shape))
            self.net.set_params(params)

        return grads

    def _adam_update(self, grads: List[np.ndarray]) -> None:
        self._t += 1
        params = self.net.get_params()
        new_params = []
        b1, b2, eps_adam = 0.9, 0.999, 1e-8
        lr_t = self.lr * np.sqrt(1 - b2**self._t) / (1 - b1**self._t)

        for i, (p, g) in enumerate(zip(params, grads)):
            self._m[i] = b1 * self._m[i] + (1 - b1) * g
            self._v[i] = b2 * self._v[i] + (1 - b2) * g**2
            update = lr_t * self._m[i] / (np.sqrt(self._v[i]) + eps_adam)
            new_params.append(p - update)

        self.net.set_params(new_params)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        env,
        n_episodes:    int  = 200,
        grader_cls=None,
        grader_kwargs: Optional[Dict] = None,
        log_interval:  int  = 10,
        verbose:       bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train the PPO agent.

        The grader is now wired into collect_episode() so that the terminal
        score is injected into the trajectory before GAE is computed — not
        just logged after the update.
        """
        for ep in range(n_episodes):
            # Rollout with grader-score terminal injection
            rollout = self.collect_episode(env, grader_cls, grader_kwargs)
            advantages, returns = self.compute_gae(
                rollout["rewards"], rollout["values"], rollout["v_last"]
            )

            # PPO update epochs
            ep_pl = ep_vl = ep_en = 0.0
            for _ in range(self.epochs):
                grads = self._finite_diff_gradient(
                    rollout["states"], rollout["actions"],
                    rollout["log_probs"], advantages, returns,
                )
                self._adam_update(grads)
                pl, vl, en = self._compute_loss(
                    rollout["states"], rollout["actions"],
                    rollout["log_probs"], advantages, returns,
                )
                ep_pl += pl; ep_vl += vl; ep_en += en

            self.episode_rewards.append(rollout["total_reward"])
            self.episode_scores.append(rollout["grader_score"])
            self.policy_losses.append(ep_pl / self.epochs)
            self.value_losses.append(ep_vl / self.epochs)
            self.entropies.append(ep_en / self.epochs)

            if verbose and (ep + 1) % log_interval == 0:
                recent_r = np.mean(self.episode_rewards[-log_interval:])
                recent_s = np.mean(self.episode_scores[-log_interval:])
                print(f"  ep {ep+1:4d}/{n_episodes}  "
                      f"reward={recent_r:+7.2f}  "
                      f"score={recent_s:.3f}  "
                      f"pl={ep_pl/self.epochs:.3f}  "
                      f"ent={ep_en/self.epochs:.3f}")

        return {
            "episode_rewards": self.episode_rewards,
            "episode_scores":  self.episode_scores,
            "policy_losses":   self.policy_losses,
            "value_losses":    self.value_losses,
            "entropies":       self.entropies,
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def act(self, obs) -> Any:
        """Stochastic action matching training behavior."""
        from adaptive_alert_triage.models import Action
        if not obs.alerts:
            raise ValueError("No alerts")
        s = encode_state(obs)
        probs, _ = self.net.forward(s)
        # Sample from policy distribution (same as training), NOT argmax!
        # argmax collapses a learned distribution like [0.35, 0.25, 0.22, 0.18]
        # into always picking the same action.
        a = int(np.random.choice(4, p=probs))
        alert = _select_alert(obs, a)
        return Action(alert_id=alert.id, action_type=_ACTION_NAMES[a])

    def reset(self) -> None:
        self.net.reset_hidden()

    def save(self, path: str) -> None:
        data = {"params": [p.tolist() for p in self.net.get_params()]}
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"  Saved weights → {path}")

    def load(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        self.net.set_params([np.array(p) for p in data["params"]])