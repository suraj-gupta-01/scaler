"""
FastAPI OpenEnv Server for Adaptive Alert Triage Environment — v0.3.1

Root-cause fixes from v0.3.0:
  FIX 1 — "No active episode" on /agent/recommend
  FIX 2 — Queued alerts (real_alerts_queue) never appeared in env.alerts
  FIX 3 — alert.dict() / obs.dict() removed in Pydantic v2
  FIX 4 — task_score missing from info dict
  FIX 5 — real_alerts_queue dropped on /env/reset
  FIX 6 — state.system_load AttributeError

New in v0.3.1 (pre-submission compliance):
  FIX 7 — Added POST /reset  (OpenEnv spec requires top-level /reset endpoint)
  FIX 8 — Added POST /env/reset  (alias without task_id, defaults to "hard")
  FIX 9 — Registered `openenv validate` CLI entry-point via pyproject.toml
           (see companion pyproject.toml fix)
"""

from __future__ import annotations

import asyncio
import os
import sys
import traceback
from collections import deque
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .env    import AdaptiveAlertTriageEnv
from .models import Action, Observation, Reward


# ── Try to load trained PPO agent (lazy import, server starts without it) ─────
_PPO_AVAILABLE = False
try:
    _project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    from rl_agent import PPOTrainer, encode_state, _ACTION_NAMES  # type: ignore
    _PPO_AVAILABLE = True
except ImportError:
    _project_root = ""


# ── Request / response models ─────────────────────────────────────────────────

class IngestAlert(BaseModel):
    id: str
    visible_severity: float
    confidence: float
    type: str


class StepRequest(BaseModel):
    alert_id: str
    action_type: str


class ResetRequest(BaseModel):
    """Optional body for POST /reset — task_id defaults to 'hard'."""
    task_id: Optional[str] = "hard"
    seed: Optional[int] = None


class HealthResponse(BaseModel):
    status: str
    env_ready: bool
    queue_size: int


# ── Alert-type normaliser ─────────────────────────────────────────────────────

_TYPE_REMAP: Dict[str, str] = {
    "cpu": "CPU", "cpu_spike": "CPU",
    "memory": "MEMORY", "memory_leak": "MEMORY",
    "disk": "DISK", "disk_full": "DISK",
    "network": "NETWORK", "net": "NETWORK", "network_latency": "NETWORK",
    "application": "APPLICATION", "app": "APPLICATION",
    "security": "SECURITY", "sec": "SECURITY",
}
_VALID = {"CPU", "MEMORY", "DISK", "NETWORK", "APPLICATION", "SECURITY"}


def _norm(raw: str) -> str:
    return _TYPE_REMAP.get(raw.lower(), raw.upper()) if raw else "APPLICATION"


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Adaptive Alert Triage RL Server", version="0.3.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

@app.middleware("http")
async def log_requests(request, call_next):
    print(f"REQUEST: {request.method} {request.url}")
    return await call_next(request)

# ── Global state ──────────────────────────────────────────────────────────────

env:            Optional[AdaptiveAlertTriageEnv] = None
episode_scores: List[float]                      = []
_ppo_agents:    Dict[str, Any]                   = {}   # task_id → PPOTrainer
_loop_task:     Optional[asyncio.Task]           = None
_last_action:   Optional[str]                    = None
_step_correct:  int = 0
_step_total:    int = 0

STEP_INTERVAL = 1.0   # seconds between autonomous episode-loop steps


# ── Score helpers ─────────────────────────────────────────────────────────────

def _reset_score() -> None:
    global _step_correct, _step_total
    _step_correct = _step_total = 0


def _tick(info: Dict) -> None:
    global _step_correct, _step_total
    _step_total += 1
    if info.get("action_correct", False):
        _step_correct += 1


def _score() -> float:
    return _step_correct / _step_total if _step_total else 0.0


# ── PPO helpers ───────────────────────────────────────────────────────────────

def _load_ppo(task_id: str) -> Optional[Any]:
    if not _PPO_AVAILABLE:
        return None
    path = os.path.join(_project_root, "weights", f"ppo_{task_id}.json")
    if not os.path.exists(path):
        print(f"   [PPO] weights not found: {path}")
        return None
    try:
        agent = PPOTrainer(task_id=task_id)
        agent.load(path)
        print(f"   [PPO] loaded {path}")
        return agent
    except Exception as e:
        print(f"   [PPO] load error: {e}")
        return None


def _ppo_act() -> Optional[Action]:
    if not env or not env.alerts:
        return None
    agent = _ppo_agents.get(env.task_id)
    if agent is None:
        return None
    try:
        obs = Observation(
            alerts         = list(env.alerts),
            system_load    = getattr(env, "_last_system_load", 0.5),
            queue_length   = len(env.alerts),
            time_remaining = env.max_steps - env.current_step,
            resource_budget=(
                env.max_investigations_per_step - env.investigations_used
                if env.max_investigations_per_step is not None else None
            ),
            episode_step   = env.current_step,
        )
        return agent.act(obs)
    except Exception:
        return None


def _rule_act() -> Optional[Action]:
    if not env or not env.alerts:
        return None
    top  = max(env.alerts, key=lambda a: a.visible_severity)
    sev  = top.visible_severity
    conf = top.confidence
    rem  = (env.max_investigations_per_step - env.investigations_used
            if env.max_investigations_per_step is not None else None)
    if sev >= 0.75 and conf >= 0.60:
        atype = "ESCALATE" if (rem is not None and rem <= 0) else "INVESTIGATE"
    elif conf < 0.30 or sev < 0.30:
        atype = "IGNORE"
    elif sev >= 0.55:
        atype = "ESCALATE"
    else:
        atype = "DELAY"
    return Action(alert_id=top.id, action_type=atype)


# ── Always-live episode loop ──────────────────────────────────────────────────

async def _episode_loop() -> None:
    global env, _last_action

    while True:
        try:
            if env is None:
                await asyncio.sleep(STEP_INTERVAL)
                continue

            if not env.alerts or env._is_terminal():
                if _step_total > 0:
                    episode_scores.append(_score())
                _reset_score()
                env.reset()

            if not env.alerts:
                await asyncio.sleep(STEP_INTERVAL)
                continue

            import time
            if time.time() - globals().get("_last_manual_step_time", 0.0) < 5.0:
                await asyncio.sleep(STEP_INTERVAL)
                continue

            action = _ppo_act() or _rule_act()
            if action is None:
                await asyncio.sleep(STEP_INTERVAL)
                continue

            _last_action = action.action_type
            _, reward, done, info = env.step(action)
            _tick(info)

            if done:
                episode_scores.append(_score())
                if len(episode_scores) > 1000:
                    episode_scores[:] = episode_scores[-1000:]
                _reset_score()
                env.reset()

        except Exception as exc:
            print(f"[episode_loop] {exc}")

        await asyncio.sleep(STEP_INTERVAL)


# ── Startup / shutdown ────────────────────────────────────────────────────────

def _restore_pristine_weights():
    import shutil
    pristine_dir = os.path.join(_project_root if _project_root else os.getcwd(), "weights_pristine")
    weights_dir  = os.path.join(_project_root if _project_root else os.getcwd(), "weights")

    if not os.path.exists(pristine_dir):
        print("   [STARTUP] No pristine weights found, skipping restore.")
        return

    os.makedirs(weights_dir, exist_ok=True)
    for f in os.listdir(pristine_dir):
        if f.startswith("ppo_") and f.endswith(".json"):
            src = os.path.join(pristine_dir, f)
            dst = os.path.join(weights_dir, f)
            shutil.copy2(src, dst)
            print(f"   [STARTUP] Restored pristine weights: {f}")


@app.on_event("startup")
async def startup():
    global env, _loop_task

    _restore_pristine_weights()

    env = AdaptiveAlertTriageEnv(task_id="hard")
    env.real_alerts_queue = deque(maxlen=50)
    env.reset()

    for tid in ("easy", "medium", "hard"):
        agent = _load_ppo(tid)
        if agent:
            _ppo_agents[tid] = agent

    _loop_task = asyncio.create_task(_episode_loop())

    print("✅ Alert Triage RL Server v0.3.1")
    print(f"   Active alerts : {len(env.alerts)}")
    print(f"   PPO loaded    : {list(_ppo_agents.keys()) or 'none (run train_rl.py first)'}")
    print(f"   Episode loop  : every {STEP_INTERVAL}s")


@app.on_event("shutdown")
async def shutdown():
    if _loop_task:
        _loop_task.cancel()


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status    = "ok",
        env_ready = env is not None and bool(env.alerts),
        queue_size= len(env.real_alerts_queue) if env and hasattr(env, "real_alerts_queue") else 0,
    )


@app.get("/metrics")
async def metrics():
    if not env:
        return {"error": "not initialized"}
    mean  = sum(episode_scores[-100:]) / len(episode_scores[-100:]) if episode_scores else 0.0
    delta = (mean - 0.61) * 100
    return {
        "mean_score":         round(mean, 3),
        "vs_baseline":        f"+{delta:.0f}%" if delta >= 0 else f"{delta:.0f}%",
        "active_alerts":      len(env.alerts),
        "episodes_completed": len(episode_scores),
        "current_step_score": round(_score(), 3),
        "current_step":       env.current_step,
        "last_action":        _last_action,
        "queue_size":         len(env.real_alerts_queue) if hasattr(env, "real_alerts_queue") else 0,
        "ppo_loaded":         list(_ppo_agents.keys()),
    }


# ── Alert ingestion ───────────────────────────────────────────────────────────

@app.post("/ingest/alerts")
async def ingest_one(alert: IngestAlert):
    if not env:
        return {"error": "not initialized"}
    if not hasattr(env, "real_alerts_queue"):
        env.real_alerts_queue = deque(maxlen=50)
    raw = alert.model_dump()
    raw["type"] = _norm(raw.get("type", "APPLICATION"))
    env.real_alerts_queue.appendleft(raw)
    return {
        "status": "queued", "queued": len(env.real_alerts_queue),
        "alert_id": alert.id, "resolved_type": raw["type"],
        "note": "Episode loop will process this within ~1s",
    }


@app.post("/ingest/alert-batch")
async def ingest_batch(alerts: List[IngestAlert]):
    if not env:
        return {"error": "not initialized"}
    if not hasattr(env, "real_alerts_queue"):
        env.real_alerts_queue = deque(maxlen=50)
    ingested = []
    for alert in alerts:
        raw = alert.model_dump()
        raw["type"] = _norm(raw.get("type", "APPLICATION"))
        env.real_alerts_queue.appendleft(raw)
        ingested.append({"alert_id": alert.id, "resolved_type": raw["type"]})
    return {"status": "queued", "queued": len(env.real_alerts_queue), "ingested": ingested}


# ── Environment control ───────────────────────────────────────────────────────

async def _do_reset(task_id: str = "hard", seed: Optional[int] = None) -> dict:
    """
    Shared reset logic used by all reset endpoints.
    Returns a dict suitable for JSON response.
    """
    global env
    if task_id not in ("easy", "medium", "hard"):
        return {"error": f"Invalid task_id '{task_id}'. Must be one of: easy, medium, hard"}
    try:
        saved = env.real_alerts_queue if (env and hasattr(env, "real_alerts_queue")) else None
        env = AdaptiveAlertTriageEnv(task_id=task_id)
        env.real_alerts_queue = saved if saved is not None else deque(maxlen=50)
        agent = _load_ppo(task_id)
        if agent:
            _ppo_agents[task_id] = agent
        obs = env.reset(seed=seed)
        _reset_score()
        return {"status": "reset", "task_id": task_id, "obs": obs.model_dump()}
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


# FIX 7 — Top-level /reset endpoint required by OpenEnv validator ping
# The pre-submission checker does: POST $PING_URL/reset
# This must return 200 and a valid Observation.
@app.post("/reset")
async def reset_top_level(request: Optional[ResetRequest] = None):
    """
    OpenEnv-required top-level reset endpoint.

    POST /reset
    Body (optional JSON): {"task_id": "easy"|"medium"|"hard", "seed": int}

    Returns the initial Observation for the new episode.
    This is the endpoint pinged by the pre-submission checker.
    """
    task_id = "hard"
    seed    = None
    if request is not None:
        task_id = request.task_id or "hard"
        seed    = request.seed
    return await _do_reset(task_id=task_id, seed=seed)


# FIX 8 — /env/reset without a path parameter (alias, defaults to "hard")
@app.post("/env/reset")
async def reset_env_default(request: Optional[ResetRequest] = None):
    """
    Alias for /env/reset/{task_id} without requiring a path parameter.
    Accepts the same optional JSON body as /reset.
    """
    task_id = "hard"
    seed    = None
    if request is not None:
        task_id = request.task_id or "hard"
        seed    = request.seed
    return await _do_reset(task_id=task_id, seed=seed)


@app.post("/env/reset/{task_id}")
async def reset_env(task_id: str = "hard"):
    """Reset with explicit task_id in path (original endpoint, kept for compatibility)."""
    return await _do_reset(task_id=task_id)


import time
_last_manual_step_time = 0.0

@app.post("/env/step")
async def step_env(request: StepRequest):
    global episode_scores, _last_manual_step_time
    _last_manual_step_time = time.time()

    if not env:
        return {"error": "not initialized"}
    if request.action_type not in {"INVESTIGATE", "IGNORE", "ESCALATE", "DELAY"}:
        return {"error": f"Invalid action '{request.action_type}'"}
    try:
        from rl_agent import encode_state  # type: ignore
        old_obs = Observation(
            alerts         = list(env.alerts),
            system_load    = getattr(env, "_last_system_load", 0.5),
            queue_length   = len(env.alerts),
            time_remaining = env.max_steps - env.current_step,
            resource_budget=(
                env.max_investigations_per_step - env.investigations_used
                if env.max_investigations_per_step is not None else None
            ),
            episode_step   = env.current_step,
        )

        action = Action(alert_id=request.alert_id, action_type=request.action_type)
        obs, reward, done, info = env.step(action)

        agent = _ppo_agents.get(env.task_id)
        if agent is not None:
            agent.net.forward(encode_state(old_obs))

        _tick(info)
        s = _score()
        info["task_score"] = s
        if done:
            episode_scores.append(s)
            _reset_score()
        return {"obs": obs.model_dump(), "reward": reward.value,
                "done": done, "info": info, "score": s}
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


@app.get("/env/state")
async def get_state():
    if not env:
        return {"error": "not initialized"}
    try:
        state = env.state()
        return {
            "visible_state": {
                "alerts":         [a.model_dump() for a in env.alerts],
                "current_step":   env.current_step,
                "max_steps":      env.max_steps,
                "failures_count": env.failures_count,
                "system_load":    state.observation.system_load,
                "queue_length":   len(env.alerts),
                "task_id":        env.task_id,
                "real_queue_size": len(env.real_alerts_queue) if hasattr(env, "real_alerts_queue") else 0,
            },
            "hidden_state":      state.hidden_state,
            "cumulative_reward": state.cumulative_reward,
        }
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


# ── Agent recommendation ──────────────────────────────────────────────────────

@app.get("/agent/recommend")
async def recommend():
    if not env or not env.alerts:
        return {
            "error": "No alerts yet — episode loop is starting, retry in 2s",
            "active_alerts": len(env.alerts) if env else 0,
        }

    task_id = env.task_id
    top     = max(env.alerts, key=lambda a: a.visible_severity)

    ppo = _ppo_agents.get(task_id)
    if ppo is not None:
        try:
            import numpy as np
            obs = Observation(
                alerts         = list(env.alerts),
                system_load    = getattr(env, "_last_system_load", 0.5),
                queue_length   = len(env.alerts),
                time_remaining = env.max_steps - env.current_step,
                resource_budget=(
                    env.max_investigations_per_step - env.investigations_used
                    if env.max_investigations_per_step is not None else None
                ),
                episode_step   = env.current_step,
            )
            s     = encode_state(obs)
            old_h, old_c = ppo.net.h.copy(), ppo.net.c.copy()
            probs, val = ppo.net.forward(s)
            ppo.net.h, ppo.net.c = old_h, old_c
            idx   = int(np.random.choice(4, p=probs))
            act   = _ACTION_NAMES[idx]
            conf  = round(float(probs[idx]) * 100, 1)
            return {
                "alert_id":         top.id,
                "action_type":      act,
                "reasoning":        f"PPO ({conf:.1f}% confidence)",
                "source":           "trained_ppo",
                "model_confidence": conf,
                "probabilities":    {_ACTION_NAMES[i]: round(float(probs[i]), 4) for i in range(4)},
                "value_estimate":   round(float(val), 3),
                "alert_severity":   top.visible_severity,
                "alert_confidence": top.confidence,
                "alert_age":        top.age,
                "alert_type":       top.alert_type,
                "active_alerts":    len(env.alerts),
                "episode_step":     env.current_step,
                "task_id":          task_id,
            }
        except Exception as exc:
            print(f"PPO recommend error: {exc}")

    # Rule-based fallback
    sev, conf = top.visible_severity, top.confidence
    rem = (env.max_investigations_per_step - env.investigations_used
           if env.max_investigations_per_step is not None else None)
    if sev >= 0.75 and conf >= 0.60:
        act = "ESCALATE" if (rem is not None and rem <= 0) else "INVESTIGATE"
    elif conf < 0.30 or sev < 0.30:
        act = "IGNORE"
    elif sev >= 0.55:
        act = "ESCALATE"
    else:
        act = "DELAY"

    return {
        "alert_id": top.id, "action_type": act,
        "source": "rule_based",
        "alert_severity": sev, "alert_confidence": conf,
        "alert_type": top.alert_type, "active_alerts": len(env.alerts),
        "task_id": task_id,
        "hint": "Run `python train_rl.py --episodes 300` to load PPO weights",
    }


@app.get("/agent/weights/{task_id}")
async def download_weights(task_id: str):
    from fastapi import HTTPException
    path = os.path.join(_project_root if _project_root else os.getcwd(), "weights", f"ppo_{task_id}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"No trained weights found for {task_id}")
    return FileResponse(path, media_type='application/json', filename=f"ppo_{task_id}.json")


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws/train")
async def ws_train(websocket: WebSocket):
    global env, episode_scores
    await websocket.accept()
    lc = lt = 0
    try:
        while True:
            data = await websocket.receive_json()
            if data.get("type") == "reset":
                tid  = data.get("task_id", "hard")
                saved = env.real_alerts_queue if (env and hasattr(env, "real_alerts_queue")) else None
                env  = AdaptiveAlertTriageEnv(task_id=tid)
                env.real_alerts_queue = saved or deque(maxlen=50)
                obs  = env.reset()
                lc = lt = 0
                await websocket.send_json({"obs": obs.model_dump(), "task_id": tid})
            elif data.get("type") == "step":
                if not env:
                    await websocket.send_json({"error": "Reset first"}); continue
                ad  = data.get("action", {})
                act = Action(alert_id=ad.get("alert_id",""), action_type=ad.get("action_type","IGNORE"))
                obs, reward, done, info = env.step(act)
                lt += 1
                if info.get("action_correct", False): lc += 1
                s = lc / lt if lt else 0.0
                if done: episode_scores.append(s)
                info["task_score"] = s
                await websocket.send_json({
                    "obs": obs.model_dump(), "reward": reward.value,
                    "done": done, "info": info, "task_score": s,
                    "action_correct": info.get("action_correct", False),
                    "failures_this_step": info.get("failures_this_step", 0),
                })
            elif data.get("type") == "close":
                break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try: await websocket.send_json({"error": str(e)})
        except Exception: pass


# ── Utility ───────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "name": "Adaptive Alert Triage RL Server", "version": "0.3.1",
        "openenv_endpoints": {
            "reset":  "POST /reset",
            "step":   "POST /env/step",
            "state":  "GET  /env/state",
            "health": "GET  /health",
        },
        "quick_start": [
            "1. python train_rl.py --episodes 300",
            "2. uvicorn src.adaptive_alert_triage.server:app --port 7860",
            "3. curl -X POST localhost:7860/reset",
            "4. curl localhost:7860/agent/recommend",
        ],
    }


import threading
import subprocess

_training_proc = None
_training_logs = []

def _run_training(episodes: int):
    global _training_proc, _training_logs, _ppo_agents
    _training_logs = [f"Starting training with --episodes {episodes}..."]
    try:
        _training_proc = subprocess.Popen(
            [sys.executable, "train_rl.py", "--episodes", str(episodes)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=_project_root if _project_root else os.getcwd()
        )
        for line in iter(_training_proc.stdout.readline, ''):
            if line:
                _training_logs.append(line.rstrip('\n'))
                if len(_training_logs) > 1000:
                    _training_logs.pop(0)
        _training_proc.wait()
        _training_logs.append(f"Training finished with exit code {_training_proc.returncode}")

        if _training_proc.returncode == 0:
            for tid in ("easy", "medium", "hard"):
                agent = _load_ppo(tid)
                if agent:
                    _ppo_agents[tid] = agent
            _training_logs.append("Successfully reloaded PPO weights for all tasks.")
    except Exception as e:
        _training_logs.append(f"Error starting training: {e}")

@app.post("/train")
async def start_training(episodes: int = 300):
    global _training_proc
    if _training_proc is not None and _training_proc.poll() is None:
        return {"status": "already running"}
    threading.Thread(target=_run_training, args=(episodes,), daemon=True).start()
    return {"status": "started"}

@app.get("/train/status")
async def get_training_status():
    global _training_proc, _training_logs
    is_running = _training_proc is not None and _training_proc.poll() is None
    return {"is_running": is_running, "logs": _training_logs}

@app.get("/web")
async def web_ui():
    import os
    dashboard_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "dashboard.html"
    )
    return FileResponse(dashboard_path, media_type="text/html")


@app.get("/tasks")
async def list_tasks():
    return {"tasks": [
        {"id": "easy",   "success_threshold": 0.70, "max_steps": 30},
        {"id": "medium", "success_threshold": 0.55, "max_steps": 40},
        {"id": "hard",   "success_threshold": 0.50, "max_steps": 50},
    ]}