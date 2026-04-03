"""
inference.py — Baseline Inference Script (ROOT LEVEL)
======================================================

Pre-submission checklist requirements:
  ✅  Uses OpenAI Client for all LLM calls (NOT Gemini)
  ✅  Reads API_BASE_URL, MODEL_NAME, HF_TOKEN from environment variables
  ✅  File is named inference.py and lives in the ROOT of the project
  ✅  Emits strict [START], [STEP], [END] stdout log format
  ✅  Produces reproducible baseline scores on all 3 tasks (easy/medium/hard)
  ✅  Runtime < 20 min on 2 vCPU / 8 GB RAM (3 tasks × 3 eps ≈ 2–4 min)

Required environment variables:
    API_BASE_URL   — LLM endpoint, e.g. https://api.openai.com/v1
    MODEL_NAME     — Model identifier, e.g. gpt-4o-mini
    HF_TOKEN       — Hugging Face / API key used as the OpenAI api_key

Optional:
    OPENAI_API_KEY — fallback if HF_TOKEN not set

Usage:
    export API_BASE_URL="https://api.openai.com/v1"
    export MODEL_NAME="gpt-4o-mini"
    export HF_TOKEN="hf_..."
    python inference.py               # all 3 tasks, 3 episodes each
    python inference.py --task easy   # single task
    python inference.py --n 5         # 5 episodes per task

Stdout log format (one JSON line per event — DO NOT CHANGE field names):
    [START]  {"task":"easy","episode":1,"seed":42}
    [STEP]   {"step":1,"alert_id":"alert_0001_00","action":"INVESTIGATE","score":0.0,"reward":10.0,"done":false}
    [END]    {"task":"easy","episode":1,"score":0.823,"passed":true}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

# ── Path bootstrap ────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
for _p in (_ROOT, os.path.join(_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from adaptive_alert_triage.env    import AdaptiveAlertTriageEnv
from adaptive_alert_triage.models import Action, Observation

from tasks.easy   import EasyTaskGrader,   SUCCESS_THRESHOLD as EASY_THRESH
from tasks.medium import MediumTaskGrader, SUCCESS_THRESHOLD as MED_THRESH
from tasks.hard   import HardTaskGrader,   SUCCESS_THRESHOLD as HARD_THRESH

# ── OpenAI client ─────────────────────────────────────────────────────────────
try:
    from openai import OpenAI
    _OPENAI_OK = True
except ImportError:
    _OPENAI_OK = False

# ── Env-var config (checklist-specified names) ────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.x.ai/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "grok-4-1-fast-reasoning")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")
_API_KEY     = HF_TOKEN or os.environ.get("GROK_API_KEY", "no-key-set")

# ── Task registry ─────────────────────────────────────────────────────────────
_TASKS: Dict[str, Dict[str, Any]] = {
    "easy":   {"cls": EasyTaskGrader,   "kwargs": {},                               "thresh": EASY_THRESH},
    "medium": {"cls": MediumTaskGrader, "kwargs": {"max_investigations_per_step": 3}, "thresh": MED_THRESH},
    "hard":   {"cls": HardTaskGrader,   "kwargs": {},                               "thresh": HARD_THRESH},
}

# ── Structured log helpers — field names are fixed by the evaluator ───────────

def _emit(tag: str, payload: Dict[str, Any]) -> None:
    """Write one log line: '<TAG>  <json>' — no trailing whitespace."""
    print(f"{tag}  {json.dumps(payload, separators=(',', ':'))}", flush=True)


def log_start(task: str, episode: int, seed: int) -> None:
    _emit("[START]", {"task": task, "episode": episode, "seed": seed})


def log_step(step: int, alert_id: str, action: str,
             score: float, reward: float, done: bool) -> None:
    _emit("[STEP]", {
        "step":     step,
        "alert_id": alert_id,
        "action":   action,
        "score":    round(score, 4),
        "reward":   round(reward, 4),
        "done":     done,
    })


def log_end(task: str, episode: int, score: float, passed: bool) -> None:
    _emit("[END]", {"task": task, "episode": episode,
                    "score": round(score, 4), "passed": passed})


# ── LLM system prompt ─────────────────────────────────────────────────────────

_SYSTEM = (
    "You are an expert IT alert triage engineer. "
    "Given active alerts and system context, choose the BEST action for the "
    "highest-priority alert.\n\n"
    "Actions:\n"
    "  INVESTIGATE — deep diagnosis; costs investigation budget. "
    "Use for high-severity (>0.75), high-confidence (>0.60) alerts.\n"
    "  IGNORE — dismiss as noise. Use when confidence < 0.30 or severity < 0.30.\n"
    "  ESCALATE — route to specialist. Use when serious but budget exhausted, "
    "or confidence too low to investigate confidently.\n"
    "  DELAY — defer to next step. Only for medium alerts when budget is 0.\n\n"
    "Return ONLY valid JSON — no markdown, no explanation:\n"
    '{"alert_id":"<exact id>","action":"INVESTIGATE|IGNORE|ESCALATE|DELAY",'
    '"reasoning":"<one sentence>"}'
)


def _build_user_message(obs: Observation) -> str:
    parts = ["Active alerts:"]
    for a in obs.alerts:
        parts.append(
            f"  {a.id}: sev={a.visible_severity:.2f} conf={a.confidence:.2f} "
            f"type={a.alert_type} age={a.age}"
        )
    bud = str(obs.resource_budget) if obs.resource_budget is not None else "unlimited"
    parts.append(
        f"\nContext: system_load={obs.system_load:.2f} "
        f"queue={obs.queue_length} time_left={obs.time_remaining} "
        f"budget={bud}"
    )
    parts.append("\nReturn JSON only.")
    return "\n".join(parts)


# ── LLM agent ─────────────────────────────────────────────────────────────────

class LLMTriageAgent:
    """
    Alert triage agent that calls an OpenAI-compatible LLM endpoint.

    Uses API_BASE_URL + MODEL_NAME + HF_TOKEN as required by the checklist.
    Falls back to rule-based logic on API errors or JSON parse failures so
    episodes always complete (fallbacks are counted and reported at the end).
    """

    _VALID_ACTIONS = frozenset({"INVESTIGATE", "IGNORE", "ESCALATE", "DELAY"})

    def __init__(self) -> None:
        if not _OPENAI_OK:
            raise ImportError("openai package required.  Run: pip install openai")
        self._client   = OpenAI(api_key=_API_KEY, base_url=API_BASE_URL)
        self.model     = MODEL_NAME
        self.api_calls = 0
        self.fallbacks = 0

    # ── Public interface ──────────────────────────────────────────────────────

    def act(self, obs: Observation) -> Action:
        if not obs.alerts:
            raise ValueError("act() called with empty alerts")
        text = self._call_api(_build_user_message(obs))
        if text is None:
            self.fallbacks += 1
            return self._rule_fallback(obs)
        return self._parse(text, obs)

    def reset(self) -> None:
        pass   # stateless between episodes

    # ── API call ──────────────────────────────────────────────────────────────

    def _call_api(self, user_msg: str, retries: int = 2) -> Optional[str]:
        for attempt in range(retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model       = self.model,
                    messages    = [
                        {"role": "system", "content": _SYSTEM},
                        {"role": "user",   "content": user_msg},
                    ],
                    temperature     = 0.0,
                    max_tokens      = 150,
                    response_format = {"type": "json_object"},
                )
                self.api_calls += 1
                return (resp.choices[0].message.content or "").strip()
            except Exception as exc:
                wait = 2 ** attempt
                if attempt < retries:
                    print(f"  [LLM] attempt {attempt+1} failed: {exc}. "
                          f"Retrying in {wait}s", file=sys.stderr)
                    time.sleep(wait)
                else:
                    print(f"  [LLM] all retries exhausted: {exc}", file=sys.stderr)
                    return None

    # ── JSON parsing ──────────────────────────────────────────────────────────

    def _parse(self, raw: str, obs: Observation) -> Action:
        # Strip accidental markdown fences
        text = raw.strip()
        if text.startswith("```"):
            text = text.lstrip("`json").lstrip("`").rstrip("`").strip()
        try:
            data    = json.loads(text)
            aid     = str(data.get("alert_id", ""))
            action  = str(data.get("action",   "")).upper()
            valid   = {a.id for a in obs.alerts}
            if aid not in valid:
                # Case-insensitive fuzzy match
                low  = {i.lower(): i for i in valid}
                aid  = low.get(aid.lower(), obs.alerts[0].id)
            if action not in self._VALID_ACTIONS:
                action = self._rule_fallback(obs).action_type
            return Action(
                alert_id    = aid,
                action_type = action,
                metadata    = {"reasoning": data.get("reasoning", ""), "source": "llm"},
            )
        except Exception as exc:
            print(f"  [LLM] parse error: {exc} | raw: {raw[:80]}", file=sys.stderr)
            self.fallbacks += 1
            return self._rule_fallback(obs)

    # ── Rule-based fallback ───────────────────────────────────────────────────

    def _rule_fallback(self, obs: Observation) -> Action:
        """Simple threshold policy used when the LLM fails."""
        alert = max(obs.alerts, key=lambda a: a.visible_severity)
        sev, conf = alert.visible_severity, alert.confidence
        bud = obs.resource_budget
        no_budget = bud is not None and bud <= 0
        if sev >= 0.75 and conf >= 0.60:
            atype = "ESCALATE" if no_budget else "INVESTIGATE"
        elif conf < 0.30 or sev < 0.30:
            atype = "IGNORE"
        elif sev >= 0.55:
            atype = "ESCALATE"
        else:
            atype = "DELAY"
        return Action(alert_id=alert.id, action_type=atype)


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(agent: LLMTriageAgent, task_id: str, episode: int, seed: int) -> float:
    """
    Run one full episode, writing [START] / [STEP] / [END] to stdout.
    Returns the final grader score in [0.0, 1.0].
    """
    cfg     = _TASKS[task_id]
    env     = AdaptiveAlertTriageEnv(task_id=task_id)
    grader  = cfg["cls"](**cfg["kwargs"])
    is_hard = task_id == "hard"

    obs    = env.reset(seed=seed)
    done   = False
    step_n = 0

    log_start(task_id, episode, seed)

    while not done:
        if not obs.alerts:
            break

        action              = agent.act(obs)
        obs, reward, done, info = env.step(action)
        step_n             += 1

        # Feed grader
        if is_hard:
            grader.update_correlation_state(info.get("correlation_groups", []))
        for ad in info.get("processed_alerts", []):
            grader.process_step(ad, info)
        if is_hard:
            grader.record_failures(info.get("failures_this_step", 0))

        log_step(
            step     = step_n,
            alert_id = action.alert_id,
            action   = action.action_type,
            score    = grader.get_episode_score(),
            reward   = reward.value,
            done     = done,
        )

    final_score = grader.get_episode_score()
    log_end(task_id, episode, final_score, final_score >= cfg["thresh"])
    return final_score


# ── Main evaluation ───────────────────────────────────────────────────────────

def run_baseline(
    tasks:        List[str],
    num_episodes: int = 3,
    seed_offset:  int = 42,
) -> Dict[str, Any]:
    """
    Run LLM agent on all specified tasks, emit structured logs, return results.
    """
    if not _OPENAI_OK:
        print("[ERROR] openai package not installed. pip install openai",
              file=sys.stderr)
        sys.exit(1)

    # Validate required env vars
    missing = [v for v in ("API_BASE_URL", "MODEL_NAME", "HF_TOKEN")
               if not os.environ.get(v)]
    if missing:
        print(f"[WARN] Missing env vars: {missing}. "
              "Using defaults / OPENAI_API_KEY fallback.", file=sys.stderr)

    print("=" * 65, flush=True)
    print("Adaptive Alert Triage — LLM Baseline Inference", flush=True)
    print(f"API_BASE_URL : {API_BASE_URL}", flush=True)
    print(f"MODEL_NAME   : {MODEL_NAME}", flush=True)
    print(f"Tasks        : {tasks}", flush=True)
    print(f"Episodes/task: {num_episodes}", flush=True)
    print("=" * 65, flush=True)

    agent      = LLMTriageAgent()
    results: Dict[str, Any] = {}

    for task_id in tasks:
        thresh = _TASKS[task_id]["thresh"]
        print(f"\n{'─'*65}", flush=True)
        print(f"Task: {task_id.upper()}  (pass threshold >= {thresh})", flush=True)
        print(f"{'─'*65}", flush=True)

        scores = []
        for ep in range(1, num_episodes + 1):
            agent.reset()
            score = run_episode(agent, task_id, ep, seed_offset + ep - 1)
            scores.append(score)

        arr = np.array(scores)
        results[task_id] = {
            "mean_score":     float(arr.mean()),
            "std_score":      float(arr.std()),
            "min_score":      float(arr.min()),
            "max_score":      float(arr.max()),
            "success_rate":   float((arr >= thresh).mean()),
            "episode_scores": scores,
            "threshold":      thresh,
        }

    # Summary table
    print("\n" + "=" * 65, flush=True)
    print("BASELINE SCORE SUMMARY", flush=True)
    print(f"{'Task':<10} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Pass%':>8}",
          flush=True)
    print("─" * 52, flush=True)
    for t, r in results.items():
        print(f"{t:<10} {r['mean_score']:>8.3f} {r['std_score']:>8.3f} "
              f"{r['min_score']:>8.3f} {r['max_score']:>8.3f} "
              f"{r['success_rate']*100:>7.1f}%", flush=True)
    print("=" * 65, flush=True)
    print(f"LLM API calls : {agent.api_calls}", flush=True)
    print(f"Fallbacks     : {agent.fallbacks}", flush=True)

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="LLM baseline inference — Adaptive Alert Triage (OpenEnv)"
    )
    p.add_argument("--task",  choices=["easy", "medium", "hard"],
                   default=None, help="Single task (default: all three)")
    p.add_argument("--n",     type=int, default=3,
                   metavar="N",
                   help="Episodes per task (default: 3 — fits in 20 min budget)")
    p.add_argument("--seed",  type=int, default=42,
                   help="Base random seed (default: 42)")
    args = p.parse_args()

    task_list = [args.task] if args.task else ["easy", "medium", "hard"]
    run_baseline(tasks=task_list, num_episodes=args.n, seed_offset=args.seed)