"""
Verification Script - Check Project Structure and Integrity

Run this script to verify all files are present and properly configured.
"""

import os
import sys
from pathlib import Path


def check_file_exists(filepath: str, required: bool = True) -> bool:
    """Check if a file exists."""
    exists = os.path.exists(filepath)
    status = "✓" if exists else ("✗ MISSING" if required else "○ Optional")
    print(f"  {status} {filepath}")
    return exists


def main():
    """Run verification checks."""
    print("=" * 60)
    print("Adaptive Alert Triage - Project Verification")
    print("=" * 60)
    print()
    
    base_dir = Path(__file__).parent
    all_good = True
    
    # Check configuration files
    print("Configuration Files:")
    config_files = [
        "README.md",
        "SETUP.md",
        "pyproject.toml",
        "openenv.yaml",
        "requirements.txt",
        "Dockerfile",
    ]
    for f in config_files:
        if not check_file_exists(base_dir / f):
            all_good = False
    print()
    
    # Check source files
    print("Source Files:")
    src_files = [
        "src/adaptive_alert_triage/__init__.py",
        "src/adaptive_alert_triage/env.py",
        "src/adaptive_alert_triage/models.py",
        "src/adaptive_alert_triage/utils.py",
    ]
    for f in src_files:
        if not check_file_exists(base_dir / f):
            all_good = False
    print()
    
    # Check task files
    print("Task Files:")
    task_files = [
        "tasks/__init__.py",
        "tasks/easy.py",
        "tasks/medium.py",
        "tasks/hard.py",
    ]
    for f in task_files:
        if not check_file_exists(base_dir / f):
            all_good = False
    print()
    
    # Check reward files
    print("Reward Files:")
    if not check_file_exists(base_dir / "rewards/reward.py"):
        all_good = False
    print()
    
    # Check agent files
    print("Agent Files:")
    agent_files = [
        "agents/__init__.py",
        "agents/baseline.py",
        "agents/inference.py",
    ]
    for f in agent_files:
        if not check_file_exists(base_dir / f):
            all_good = False
    print()
    
    # Check test files
    print("Test Files:")
    test_files = [
        "tests/test_env.py",
        "tests/test_tasks.py",
        "tests/test_rewards.py",
    ]
    for f in test_files:
        if not check_file_exists(base_dir / f):
            all_good = False
    print()
    
    # Check evaluation files
    print("Evaluation Files:")
    eval_files = [
        "evaluation/evaluate.py",
        "evaluation/plots.py",
    ]
    for f in eval_files:
        if not check_file_exists(base_dir / f):
            all_good = False
    print()
    
    # Check docker files
    print("Docker Files:")
    if not check_file_exists(base_dir / "docker/entrypoint.sh"):
        all_good = False
    print()
    
    # File count summary
    print("=" * 60)
    if all_good:
        print("✅ All required files are present!")
        print()
        print("Next Steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Install package: pip install -e .")
        print("  3. Run tests: pytest tests/")
        print("  4. Try demo: python src/adaptive_alert_triage/env.py")
        print("  5. Run evaluation: python evaluation/evaluate.py")
        print()
        return 0
    else:
        print("❌ Some required files are missing!")
        print("Please ensure all files are created correctly.")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
