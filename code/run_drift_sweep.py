"""
DRIFTBENCH Drift Dose Sweep Experiment

Creates controlled drift by mixing v1 and v2 corpus documents at different ratios.
Generates decay curves and computes reliability half-life.
"""

import os
import json
import sys
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from driftbench_core import (
    DriftBenchMetrics,
    EvaluationResult,
    DriftBenchReport,
    compute_reliability_half_life
)
from rag_baselines import (
    VanillaRAG,
    OracleDoc,
    IterativeRAG,
    SimpleRetriever,
    MockGenerator,
    load_corpus
)

# Try matplotlib
try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Plots will be saved as JSON.")


@dataclass
class DriftSweepConfig:
    """Configuration for drift dose sweep."""
    name: str = "DRIFTBENCH Drift Sweep"
    drift_doses: Tuple[float, ...] = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0)
    top_k: int = 3
    seed: int = 42


def create_drifted_corpus(
    corpus_v1: Dict[str, str],
    corpus_v2: Dict[str, str],
    drift_dose: float,
    seed: int = 42
) -> Dict[str, str]:
    """
    Create a corpus with controlled drift.

    drift_dose = 0.0: All v1 docs (baseline, no drift)
    drift_dose = 1.0: All v2 docs (full drift)
    drift_dose = 0.5: 50% v1, 50% v2

    For each task document, randomly decide if it should be v1 or v2.
    Shared docs always use v2 (background knowledge).
    """
    random.seed(seed)
    drifted_corpus = {}

    # Get task-specific doc ids (not shared_*)
    task_doc_ids = [k for k in corpus_v1.keys() if not k.startswith("shared_")]
    shared_doc_ids = [k for k in corpus_v1.keys() if k.startswith("shared_")]

    # For each task doc, decide based on drift_dose
    for doc_id in task_doc_ids:
        if random.random() < drift_dose:
            # Use v2 (drifted)
            drifted_corpus[doc_id] = corpus_v2.get(doc_id, corpus_v1[doc_id])
        else:
            # Use v1 (baseline)
            drifted_corpus[doc_id] = corpus_v1[doc_id]

    # Shared docs always from v2 (or v1 if identical)
    for doc_id in shared_doc_ids:
        drifted_corpus[doc_id] = corpus_v2.get(doc_id, corpus_v1[doc_id])

    return drifted_corpus


def evaluate_at_drift_dose(
    system,
    tasks: List[Dict],
    corpus_v1: Dict[str, str],
    corpus_v2: Dict[str, str],
    drift_dose: float,
    seed: int = 42
) -> Tuple[List[EvaluationResult], Dict[str, str]]:
    """Evaluate a system at a specific drift dose."""

    # Create drifted corpus
    drifted_corpus = create_drifted_corpus(corpus_v1, corpus_v2, drift_dose, seed)

    # Index corpus
    if hasattr(system, 'index'):
        system.index(drifted_corpus)

    results = []

    for task in tasks:
        task_id = task["task_id"]
        question = task["question"]

        # Determine expected answer based on which version the doc is
        doc_content = drifted_corpus.get(task_id, "")

        # Check if this doc is v1 or v2 content
        v1_content = corpus_v1.get(task_id, "")
        v2_content = corpus_v2.get(task_id, "")

        if doc_content == v2_content:
            expected = task["answer_v2"]
            version = "v2"
        else:
            expected = task["answer_v1"]
            version = "v1"

        # Get response
        response = system.answer(question)

        # Check correctness
        predicted = response.answer.lower()
        expected_lower = expected.lower()

        correct = (
            expected_lower in predicted or
            predicted in expected_lower or
            (expected_lower == "true" and "true" in predicted) or
            (expected_lower == "false" and "false" in predicted)
        )

        # Check retrieval success
        retrieval_success = False
        for doc in response.retrieved_docs:
            if expected_lower in doc.content.lower():
                retrieval_success = True
                break

        # Determine silent failure
        silent_failure = (
            not correct and
            response.confidence >= 0.8 and
            not response.expressed_uncertainty
        )

        results.append(EvaluationResult(
            task_id=task_id,
            correct=correct,
            confidence=response.confidence,
            silent_failure=silent_failure,
            retrieval_success=retrieval_success,
            version_tested=version
        ))

    return results, drifted_corpus


def run_drift_sweep(config: DriftSweepConfig = None) -> Dict:
    """Run the full drift dose sweep experiment."""
    if config is None:
        config = DriftSweepConfig()

    print("=" * 70)
    print("  DRIFTBENCH DRIFT DOSE SWEEP")
    print("=" * 70)
    print(f"  Config: {config.name}")
    print(f"  Drift doses: {config.drift_doses}")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    # Load data
    data_dir = Path(__file__).parent.parent / "data"
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    corpus_v1 = load_corpus(data_dir / "corpus_v1.json")
    corpus_v2 = load_corpus(data_dir / "corpus_v2.json")

    with open(data_dir / "fastapi_drift_tasks.json", 'r') as f:
        dataset = json.load(f)

    tasks = dataset["tasks"]
    print(f"\n  Loaded {len(tasks)} tasks")
    print(f"  Corpus v1: {len(corpus_v1)} docs")
    print(f"  Corpus v2: {len(corpus_v2)} docs")

    # Initialize systems
    vanilla_rag = VanillaRAG(
        retriever=SimpleRetriever(),
        generator=MockGenerator(),
        top_k=config.top_k
    )

    iterative_rag = IterativeRAG(
        retriever=SimpleRetriever(),
        generator=MockGenerator(),
        top_k=config.top_k,
        max_rounds=2
    )

    # Results storage
    sweep_results = {
        "vanilla_rag": {"doses": [], "success_rates": [], "sfr": [], "cer": [], "ece": []},
        "iterative_rag": {"doses": [], "success_rates": [], "sfr": [], "cer": [], "ece": []},
    }

    # Run sweep for each system
    for system_name, system in [("vanilla_rag", vanilla_rag), ("iterative_rag", iterative_rag)]:
        print(f"\n  Evaluating {system_name}...")

        for drift_dose in config.drift_doses:
            print(f"    Drift dose {drift_dose*100:.0f}%...", end=" ")

            results, _ = evaluate_at_drift_dose(
                system, tasks, corpus_v1, corpus_v2, drift_dose, config.seed
            )

            metrics = DriftBenchMetrics(results)

            sweep_results[system_name]["doses"].append(drift_dose)
            sweep_results[system_name]["success_rates"].append(metrics.success_rate())
            sweep_results[system_name]["sfr"].append(metrics.silent_failure_rate())
            sweep_results[system_name]["cer"].append(metrics.confident_error_rate())
            sweep_results[system_name]["ece"].append(metrics.expected_calibration_error())

            print(f"Success: {metrics.success_rate()*100:.1f}%")

    # Compute half-lives
    print("\n" + "=" * 70)
    print("  RELIABILITY HALF-LIFE")
    print("=" * 70)

    for system_name in ["vanilla_rag", "iterative_rag"]:
        doses = sweep_results[system_name]["doses"]
        rates = sweep_results[system_name]["success_rates"]

        half_life = compute_reliability_half_life(doses, rates)
        sweep_results[system_name]["half_life"] = half_life

        if half_life is not None:
            print(f"  {system_name}: d_1/2 = {half_life*100:.1f}% drift")
        else:
            print(f"  {system_name}: d_1/2 = N/A (never drops to 50%)")

    # Save results
    experiment_results = {
        "config": asdict(config),
        "timestamp": datetime.now().isoformat(),
        "n_tasks": len(tasks),
        "sweep_results": sweep_results,
    }

    results_path = results_dir / "drift_sweep_results.json"
    with open(results_path, 'w') as f:
        json.dump(experiment_results, f, indent=2)
    print(f"\n  Results saved to: {results_path}")

    # Generate plots
    if HAS_MATPLOTLIB:
        generate_decay_curves(sweep_results, config.drift_doses, results_dir)
    else:
        print("\n  [Plots skipped - matplotlib not installed]")

    print("\n" + "=" * 70)
    print("  DRIFT SWEEP COMPLETE")
    print("=" * 70)

    return experiment_results


def generate_decay_curves(results: Dict, doses: Tuple, output_dir: Path):
    """Generate decay curve plots."""
    print("\n  Generating decay curves...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Success Rate Decay Curves
    ax = axes[0, 0]

    for system_name, color, marker in [
        ("vanilla_rag", "#e74c3c", "o"),
        ("iterative_rag", "#3498db", "s"),
    ]:
        data = results[system_name]
        doses_pct = [d * 100 for d in data["doses"]]
        rates_pct = [r * 100 for r in data["success_rates"]]

        ax.plot(doses_pct, rates_pct, f'{marker}-', color=color,
                label=system_name.replace("_", " ").title(), linewidth=2, markersize=8)

        # Add half-life marker if exists
        if data.get("half_life") is not None:
            hl = data["half_life"] * 100
            baseline = data["success_rates"][0] * 100
            ax.axvline(x=hl, color=color, linestyle='--', alpha=0.5)
            ax.scatter([hl], [baseline/2], color=color, s=100, marker='*', zorder=5)

    ax.set_xlabel('Drift Dose (%)')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Reliability Decay Under Drift')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 105)
    ax.set_ylim(0, 100)

    # Plot 2: Silent Failure Rate
    ax = axes[0, 1]

    for system_name, color, marker in [
        ("vanilla_rag", "#e74c3c", "o"),
        ("iterative_rag", "#3498db", "s"),
    ]:
        data = results[system_name]
        doses_pct = [d * 100 for d in data["doses"]]
        sfr_pct = [r * 100 for r in data["sfr"]]

        ax.plot(doses_pct, sfr_pct, f'{marker}-', color=color,
                label=system_name.replace("_", " ").title(), linewidth=2, markersize=8)

    ax.set_xlabel('Drift Dose (%)')
    ax.set_ylabel('Silent Failure Rate (%)')
    ax.set_title('Silent Failures (Confident + Wrong)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 105)

    # Plot 3: Confident Error Rate
    ax = axes[1, 0]

    for system_name, color, marker in [
        ("vanilla_rag", "#e74c3c", "o"),
        ("iterative_rag", "#3498db", "s"),
    ]:
        data = results[system_name]
        doses_pct = [d * 100 for d in data["doses"]]
        cer_pct = [r * 100 for r in data["cer"]]

        ax.plot(doses_pct, cer_pct, f'{marker}-', color=color,
                label=system_name.replace("_", " ").title(), linewidth=2, markersize=8)

    ax.set_xlabel('Drift Dose (%)')
    ax.set_ylabel('Confident Error Rate (%)')
    ax.set_title('CER: P(wrong AND confident)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 105)

    # Plot 4: ECE (Calibration)
    ax = axes[1, 1]

    for system_name, color, marker in [
        ("vanilla_rag", "#e74c3c", "o"),
        ("iterative_rag", "#3498db", "s"),
    ]:
        data = results[system_name]
        doses_pct = [d * 100 for d in data["doses"]]
        ece = data["ece"]

        ax.plot(doses_pct, ece, f'{marker}-', color=color,
                label=system_name.replace("_", " ").title(), linewidth=2, markersize=8)

    ax.set_xlabel('Drift Dose (%)')
    ax.set_ylabel('Expected Calibration Error')
    ax.set_title('Calibration Degradation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 105)
    ax.set_ylim(0, 0.5)

    plt.tight_layout()

    plot_path = output_dir / "drift_decay_curves.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.savefig(str(plot_path).replace('.png', '.pdf'), bbox_inches='tight')
    print(f"    Saved: {plot_path}")

    plt.close()


if __name__ == "__main__":
    results = run_drift_sweep()
