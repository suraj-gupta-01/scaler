"""
Plotting and Visualization for Evaluation Results

Generates comparison charts for RL vs baseline agents.
"""

import json
import argparse
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def load_results(filename: str = "evaluation_results.json") -> Dict[str, Any]:
    """
    Load evaluation results from JSON file.
    
    Args:
        filename: Path to results JSON file
        
    Returns:
        Results dictionary
    """
    with open(filename, "r") as f:
        return json.load(f)


def plot_score_comparison(results: Dict[str, Any], output_file: str = "score_comparison.png") -> None:
    """
    Create bar plot comparing mean scores across agents and tasks.
    
    Args:
        results: Evaluation results dictionary
        output_file: Output filename for plot
    """
    # Prepare data
    agents = []
    tasks = []
    scores = []
    errors = []
    
    for agent_name, agent_results in results.items():
        for task_id, task_results in agent_results.items():
            agents.append(agent_name)
            tasks.append(task_id.capitalize())
            scores.append(task_results["mean_score"])
            errors.append(task_results["std_score"])
    
    df = pd.DataFrame({
        "Agent": agents,
        "Task": tasks,
        "Score": scores,
        "Error": errors,
    })
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Group by task
    tasks_unique = df["Task"].unique()
    x = np.arange(len(tasks_unique))
    width = 0.25
    
    agents_unique = df["Agent"].unique()
    
    for i, agent in enumerate(agents_unique):
        agent_data = df[df["Agent"] == agent]
        scores_by_task = [
            agent_data[agent_data["Task"] == task]["Score"].values[0]
            if len(agent_data[agent_data["Task"] == task]) > 0 else 0
            for task in tasks_unique
        ]
        errors_by_task = [
            agent_data[agent_data["Task"] == task]["Error"].values[0]
            if len(agent_data[agent_data["Task"] == task]) > 0 else 0
            for task in tasks_unique
        ]
        
        plt.bar(
            x + i * width,
            scores_by_task,
            width,
            label=agent,
            yerr=errors_by_task,
            capsize=5,
            alpha=0.8,
        )
    
    # Add success thresholds
    thresholds = {"Easy": 0.7, "Medium": 0.65, "Hard": 0.6}
    for i, task in enumerate(tasks_unique):
        plt.axhline(
            y=thresholds.get(task, 0.6),
            xmin=(i - 0.4) / len(tasks_unique),
            xmax=(i + 0.4) / len(tasks_unique),
            color="red",
            linestyle="--",
            linewidth=1,
            alpha=0.5,
        )
    
    plt.xlabel("Task", fontsize=12, fontweight="bold")
    plt.ylabel("Score", fontsize=12, fontweight="bold")
    plt.title("Agent Performance Comparison Across Tasks", fontsize=14, fontweight="bold")
    plt.xticks(x + width, tasks_unique)
    plt.legend(loc="upper right", fontsize=10)
    plt.ylim(0, 1.0)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Score comparison plot saved to: {output_file}")
    plt.close()


def plot_success_rates(results: Dict[str, Any], output_file: str = "success_rates.png") -> None:
    """
    Create plot showing success rates (% episodes above threshold).
    
    Args:
        results: Evaluation results dictionary
        output_file: Output filename for plot
    """
    # Prepare data
    agents = []
    tasks = []
    success_rates = []
    
    for agent_name, agent_results in results.items():
        for task_id, task_results in agent_results.items():
            agents.append(agent_name)
            tasks.append(task_id.capitalize())
            success_rates.append(task_results["success_rate"] * 100)
    
    df = pd.DataFrame({
        "Agent": agents,
        "Task": tasks,
        "Success Rate": success_rates,
    })
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Grouped bar chart
    pivot_df = df.pivot(index="Task", columns="Agent", values="Success Rate")
    pivot_df.plot(kind="bar", ax=plt.gca(), width=0.8, alpha=0.8)
    
    plt.xlabel("Task", fontsize=12, fontweight="bold")
    plt.ylabel("Success Rate (%)", fontsize=12, fontweight="bold")
    plt.title("Success Rate: % Episodes Above Threshold", fontsize=14, fontweight="bold")
    plt.xticks(rotation=0)
    plt.legend(title="Agent", fontsize=10)
    plt.ylim(0, 100)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Success rates plot saved to: {output_file}")
    plt.close()


def plot_failure_analysis(results: Dict[str, Any], output_file: str = "failure_analysis.png") -> None:
    """
    Create plot showing mean system failures per episode.
    
    Args:
        results: Evaluation results dictionary
        output_file: Output filename for plot
    """
    # Prepare data
    agents = []
    tasks = []
    failures = []
    
    for agent_name, agent_results in results.items():
        for task_id, task_results in agent_results.items():
            agents.append(agent_name)
            tasks.append(task_id.capitalize())
            failures.append(task_results["mean_failures"])
    
    df = pd.DataFrame({
        "Agent": agents,
        "Task": tasks,
        "Mean Failures": failures,
    })
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Grouped bar chart
    pivot_df = df.pivot(index="Task", columns="Agent", values="Mean Failures")
    pivot_df.plot(kind="bar", ax=plt.gca(), width=0.8, alpha=0.8, color=["#d32f2f", "#f57c00", "#fbc02d"])
    
    plt.xlabel("Task", fontsize=12, fontweight="bold")
    plt.ylabel("Mean System Failures", fontsize=12, fontweight="bold")
    plt.title("System Failure Analysis (Lower is Better)", fontsize=14, fontweight="bold")
    plt.xticks(rotation=0)
    plt.legend(title="Agent", fontsize=10)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Failure analysis plot saved to: {output_file}")
    plt.close()


def plot_episode_curves(results: Dict[str, Any], task_id: str = "medium", 
                        output_file: str = "episode_curves.png") -> None:
    """
    Create line plots showing score progression across episodes.
    
    Args:
        results: Evaluation results dictionary
        task_id: Task to plot
        output_file: Output filename for plot
    """
    plt.figure(figsize=(12, 5))
    
    # Plot scores across episodes
    plt.subplot(1, 2, 1)
    for agent_name, agent_results in results.items():
        if task_id in agent_results:
            scores = agent_results[task_id]["episode_scores"]
            plt.plot(range(1, len(scores) + 1), scores, marker="o", label=agent_name, alpha=0.7)
    
    # Add threshold line
    thresholds = {"easy": 0.7, "medium": 0.65, "hard": 0.6}
    plt.axhline(y=thresholds[task_id], color="red", linestyle="--", label="Success Threshold", alpha=0.5)
    
    plt.xlabel("Episode", fontsize=11, fontweight="bold")
    plt.ylabel("Task Score", fontsize=11, fontweight="bold")
    plt.title(f"{task_id.capitalize()} Task: Score per Episode", fontsize=12, fontweight="bold")
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    
    # Plot rewards across episodes
    plt.subplot(1, 2, 2)
    for agent_name, agent_results in results.items():
        if task_id in agent_results:
            rewards = agent_results[task_id]["episode_rewards"]
            plt.plot(range(1, len(rewards) + 1), rewards, marker="s", label=agent_name, alpha=0.7)
    
    plt.xlabel("Episode", fontsize=11, fontweight="bold")
    plt.ylabel("Total Reward", fontsize=11, fontweight="bold")
    plt.title(f"{task_id.capitalize()} Task: Reward per Episode", fontsize=12, fontweight="bold")
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Episode curves plot saved to: {output_file}")
    plt.close()


def generate_all_plots(results_file: str = "evaluation_results.json", output_dir: str = ".") -> None:
    """
    Generate all visualization plots.
    
    Args:
        results_file: Path to evaluation results JSON
        output_dir: Directory to save plots
    """
    print(f"\nGenerating plots from: {results_file}\n")
    
    # Load results
    results = load_results(results_file)
    
    # Set style
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    # Generate plots
    plot_score_comparison(results, f"{output_dir}/score_comparison.png")
    plot_success_rates(results, f"{output_dir}/success_rates.png")
    plot_failure_analysis(results, f"{output_dir}/failure_analysis.png")
    
    # Generate episode curves for each task
    for task_id in ["easy", "medium", "hard"]:
        plot_episode_curves(results, task_id, f"{output_dir}/episode_curves_{task_id}.png")
    
    print("\n✅ All plots generated successfully!")


def main():
    """Main plotting entry point."""
    parser = argparse.ArgumentParser(
        description="Generate plots from evaluation results"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="evaluation_results.json",
        help="Input JSON file with evaluation results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--plot-type",
        type=str,
        choices=["all", "scores", "success", "failures", "episodes"],
        default="all",
        help="Type of plot to generate",
    )
    
    args = parser.parse_args()
    
    print("Adaptive Alert Triage - Result Visualization")
    
    # Load results
    try:
        results = load_results(args.input)
    except FileNotFoundError:
        print(f"Error: Results file not found: {args.input}")
        print("Run evaluate.py first to generate results.")
        return
    
    # Generate plots based on type
    if args.plot_type == "all":
        generate_all_plots(args.input, args.output_dir)
    elif args.plot_type == "scores":
        plot_score_comparison(results, f"{args.output_dir}/score_comparison.png")
    elif args.plot_type == "success":
        plot_success_rates(results, f"{args.output_dir}/success_rates.png")
    elif args.plot_type == "failures":
        plot_failure_analysis(results, f"{args.output_dir}/failure_analysis.png")
    elif args.plot_type == "episodes":
        for task_id in ["easy", "medium", "hard"]:
            plot_episode_curves(results, task_id, f"{args.output_dir}/episode_curves_{task_id}.png")


if __name__ == "__main__":
    main()
