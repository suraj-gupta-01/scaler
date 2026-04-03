#!/bin/bash
# Docker Entrypoint Script for Adaptive Alert Triage Environment

set -e

echo "================================"
echo "Adaptive Alert Triage - OpenEnv"
echo "================================"
echo ""

# Function to run validation
validate_env() {
    echo "Running OpenEnv validation..."
    echo ""
    
    # Check Python environment
    python --version
    pip list | grep -E "(pydantic|openenv|numpy)"
    
    echo ""
    echo "Checking package installation..."
    python -c "import adaptive_alert_triage; print(f'✓ Package version: {adaptive_alert_triage.__version__}')"
    
    echo ""
    echo "Validating environment structure..."
    python -c "
from adaptive_alert_triage.env import AdaptiveAlertTriageEnv
from adaptive_alert_triage.models import Action, Observation, Reward

# Test easy task
print('Testing easy task...')
env = AdaptiveAlertTriageEnv(task_id='easy', seed=42)
obs = env.reset()
assert isinstance(obs, Observation)
print(f'  ✓ Reset successful: {len(obs.alerts)} alerts')

action = Action(alert_id=obs.alerts[0].id, action_type='INVESTIGATE')
obs, reward, done, info = env.step(action)
assert isinstance(reward, Reward)
print(f'  ✓ Step successful: reward={reward.value}')

# Test medium task
print('Testing medium task...')
env = AdaptiveAlertTriageEnv(task_id='medium', seed=42)
obs = env.reset()
print(f'  ✓ Resource budget: {obs.resource_budget}')

# Test hard task
print('Testing hard task...')
env = AdaptiveAlertTriageEnv(task_id='hard', seed=42)
obs = env.reset()
state = env.state()
print(f'  ✓ Hidden state keys: {list(state.hidden_state.keys())}')

print('')
print('✅ All validation checks passed!')
"
    
    echo ""
    echo "Environment validated successfully!"
}

# Function to run tests
run_tests() {
    echo "Running test suite..."
    echo ""
    pytest tests/ -v --tb=short
    echo ""
    echo "Tests completed!"
}

# Function to run evaluation
run_evaluation() {
    echo "Running baseline evaluation..."
    echo ""
    python evaluation/evaluate.py --episodes 5 --verbose
    echo ""
    echo "Evaluation completed!"
}

# Function to start demo
run_demo() {
    echo "Running environment demo..."
    echo ""
    python src/adaptive_alert_triage/env.py
    echo ""
    echo "Demo completed!"
}

# Main command routing
case "$1" in
    validate|openenv)
        validate_env
        ;;
    test)
        run_tests
        ;;
    evaluate)
        run_evaluation
        ;;
    demo)
        run_demo
        ;;
    bash|sh|shell)
        exec /bin/bash
        ;;
    *)
        # Default: run validation
        if [ $# -eq 0 ]; then
            validate_env
        else
            # Pass through to command
            exec "$@"
        fi
        ;;
esac