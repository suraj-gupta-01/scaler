---
title: Adaptive Alert Triage & Incident Response
emoji: 🚨
colorFrom: red
colorTo: yellow
sdk: docker
sdk_version: "latest"
python_version: "3.11"
pinned: false
app_port: 7860
---

# Adaptive Alert Triage & Incident Response Environment (OpenEnv)

**Version**: 0.1.0  
**Framework**: OpenEnv  
**Status**: Alpha

## Overview

An OpenEnv-compliant reinforcement learning environment that simulates real-time IT alert triage and incident response. Agents must intelligently prioritize alerts under resource constraints while preventing cascading system failures in a partially observable, dynamic environment.

### Why RL Over Rule-Based Systems?

| **Challenge**               | **Rule-Based Limitation**                                  | **RL Advantage**                                       |
| --------------------------- | ---------------------------------------------------------- | ------------------------------------------------------ |
| **Dynamic Patterns**        | Static thresholds fail as alert patterns evolve            | Learns from feedback, adapts to changing distributions |
| **Context Awareness**       | Cannot capture alert correlations or temporal dependencies | Discovers hidden relationships through experience      |
| **Resource Optimization**   | Fixed allocation ignores varying system states             | Optimizes action selection under real-time constraints |
| **False Positive Handling** | Uniform treatment leads to alert fatigue                   | Learns nuanced confidence signals and noise patterns   |
| **Cascading Failures**      | Reactive approach misses early warning signs               | Proactive detection through predictive state modeling  |

## Environment Specification

### State Space (Partial Observability)

**Visible Features:**

- `alerts`: List of active alerts with:
  - `id`: Unique alert identifier
  - `visible_severity`: Noisy severity score (0.0-1.0)
  - `confidence`: Detection confidence (0.0-1.0)
  - `alert_type`: Category (CPU, MEMORY, DISK, NETWORK, APPLICATION, SECURITY)
  - `age`: Time steps since alert generation
- `system_load`: Current system resource utilization (0.0-1.0)
- `queue_length`: Number of unprocessed alerts
- `time_remaining`: Steps left in episode

**Hidden Features** (ground truth for reward computation):

- `true_severity`: Actual criticality of each alert
- `correlations`: Alert dependency graph
- `future_failures`: Predicted cascading failure probabilities

### Action Space

Per alert, the agent can execute:

- **INVESTIGATE**: Allocate resources to diagnose (costly but resolves critical issues)
- **IGNORE**: Mark as noise (efficient for false positives)
- **ESCALATE**: Route to specialist team (high-confidence critical alerts)
- **DELAY**: Defer to next time step (queue management)

**Resource Constraints**: Maximum K investigations per time step (task-dependent).

### Reward Structure

```python
+10  # Critical alert correctly investigated
+5   # Cascading failure prevented through correlation detection
+3   # False positive correctly ignored
-2   # Unnecessary investigation (resource waste)
-8   # Missed critical alert
-10  # System failure due to ignored critical issue
```

### Episode Dynamics

- **Length**: 20-50 time steps (task-dependent)
- **Termination**: Max steps reached OR failure threshold exceeded
- **Alert Generation**: Continuous stochastic process with temporal correlation
- **Failure Mechanics**: Ignored critical alerts accumulate damage, triggering cascading failures

## Tasks

### 1. Easy: Basic Alert Prioritization

**Objective**: Correctly classify and handle alerts based on visible signals.  
**Success Criteria**: ≥70% correct action rate  
**Key Challenge**: Distinguish genuine critical alerts from noise  
**Grading**: `correct_actions / total_actions`

### 2. Medium: Resource-Constrained Triage

**Objective**: Optimize triage under strict investigation limits.  
**Success Criteria**: ≥65% weighted efficiency score  
**Key Challenge**: Maximize critical alert resolution with limited resources  
**Grading**: `(weighted_resolved_alerts * resource_efficiency)`

### 3. Hard: Cascading Failures Prevention

**Objective**: Detect correlated alerts and prevent future failures.  
**Success Criteria**: ≥60% score with stability requirements  
**Key Challenge**: Infer hidden correlations and predict failure chains  
**Grading**: `(prevented_failures - system_instability_penalty) / max_possible`

## Installation

### Local Setup

```bash
# Clone repository
git clone https://github.com/scalar/adaptive-alert-triage.git
cd adaptive-alert-triage

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

### Docker Setup

```bash
# Build Docker image
docker build -t adaptive-alert-triage:latest .

# Run validation
docker run --rm adaptive-alert-triage:latest

# Run evaluation with OpenAI API key
docker run --rm -e OPENAI_API_KEY=your_key adaptive-alert-triage:latest python evaluation/evaluate.py
```

## Usage

### Quick Start

```python
from adaptive_alert_triage.env import AdaptiveAlertTriageEnv
from adaptive_alert_triage.models import Action

# Initialize environment with easy task
env = AdaptiveAlertTriageEnv(task_id="easy")

# Reset environment
observation = env.reset()

# Run episode
done = False
total_reward = 0

while not done:
    # Example: investigate first alert
    action = Action(
        alert_id=observation.alerts[0].id,
        action_type="INVESTIGATE"
    )

    observation, reward, done, info = env.step(action)
    total_reward += reward.value

print(f"Episode reward: {total_reward}")
print(f"Task score: {info['task_score']}")
```

### Running Baseline Agents

```bash
# Rule-based baseline
python agents/baseline.py --task easy

# OpenAI inference baseline (requires OPENAI_API_KEY)
export OPENAI_API_KEY=your_key_here
python agents/inference.py --task medium
```

### Evaluation

```bash
# Run all baselines on all tasks
python evaluation/evaluate.py

# Generate comparison plots
python evaluation/plots.py
```

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src/adaptive_alert_triage tests/

# Run specific test file
pytest tests/test_env.py -v
```

## Docker + RL Server

The environment includes a production-ready FastAPI server for remote RL training.

### Architecture

```
External World (Datadog/Kafka) ──POST /ingest/alerts──> Docker (FastAPI Server)
                                                        │
                                                        │ Internal: AdaptiveAlertTriageEnv
                                                        │ (real + synthetic alerts)
                                                        ↓
External RL Trainer (SB3)      ──/env/reset───────────> │ <──/env/step(action)── Obs/Reward/Done
                                                        │
                                                        ↓
                                                  RL beats baselines! (0.61 → 0.82+)
```

### Quick Start

```bash
# 1. Build and run the persistent RL server
docker compose up --build -d

# 2. Verify server health
curl http://localhost:8000/health

# 3. Send real alerts (simulate Datadog webhook)
bash scripts/demo_webhook.sh

# 4. Train external RL agent
pip install stable-baselines3
python train_external.py

# 5. View metrics
curl http://localhost:8000/metrics
```

### API Endpoints

| Endpoint               | Method | Description                             |
| ---------------------- | ------ | --------------------------------------- |
| `/health`              | GET    | Health check (env_ready, queue_size)    |
| `/metrics`             | GET    | RL score vs baseline comparison         |
| `/ingest/alerts`       | POST   | Webhook receiver for Datadog/Kafka      |
| `/env/reset/{task_id}` | POST   | Initialize episode (easy/medium/hard)   |
| `/env/step`            | POST   | Take RL action, receive obs/reward/done |
| `/env/state`           | GET    | Debug: current episode state            |
| `/tasks`               | GET    | List available tasks                    |
| `/ws/train`            | WS     | Real-time streaming RL loop             |

### WebSocket Training

```python
import websockets
import json

async with websockets.connect("ws://localhost:8000/ws/train") as ws:
    # Reset
    await ws.send(json.dumps({"type": "reset", "task_id": "hard"}))
    obs = await ws.recv()

    # Step loop
    while True:
        await ws.send(json.dumps({
            "type": "step",
            "action": {"alert_id": "A1", "action_type": "INVESTIGATE"}
        }))
        result = await ws.recv()
        if json.loads(result)["done"]:
            break
```

---

## Project Structure

```
adaptive_alert_triage_openenv/
├── README.md                   # This file
├── pyproject.toml              # Project metadata and dependencies
├── openenv.yaml                # OpenEnv specification
├── Dockerfile                  # Container build instructions
├── requirements.txt            # Python dependencies
│
├── src/adaptive_alert_triage/  # Core environment implementation
│   ├── __init__.py
│   ├── env.py                  # Main Gym environment
│   ├── models.py               # Pydantic Observation/Action/Reward models
│   └── utils.py                # Helper functions
│
├── tasks/                      # Task definitions and graders
│   ├── easy.py                 # Basic prioritization
│   ├── medium.py               # Resource-constrained triage
│   └── hard.py                 # Cascading failure prevention
│
├── rewards/                    # Reward shaping logic
│   └── reward.py
│
├── agents/                     # Baseline and example agents
│   ├── baseline.py             # Rule-based threshold agent
│   └── inference.py            # OpenAI API baseline
│
├── tests/                      # Unit and integration tests
│   ├── test_env.py
│   ├── test_tasks.py
│   └── test_rewards.py
│
├── evaluation/                 # Performance analysis
│   ├── evaluate.py             # Run benchmarks
│   └── plots.py                # Generate comparison charts
│
└── docker/                     # Docker utilities
    └── entrypoint.sh           # Container startup script
```

## OpenEnv Compliance

This environment adheres to the OpenEnv specification:

- ✅ Pydantic models for Observation, Action, and Reward
- ✅ OpenEnv-compatible API (`reset()`, `step()`, `state()`)
- ✅ Task-based evaluation with graders
- ✅ Reproducible seeding
- ✅ Docker containerization
- ✅ `openenv.yaml` metadata

## Contributing

Contributions are welcome! Please follow:

1. Black code formatting (`black .`)
2. Type hints for all functions
3. Docstrings in Google style
4. Unit tests for new features

## License

MIT License - see LICENSE file for details.
