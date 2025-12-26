"""
DRIFTBENCH Validation Experiment

Runs the core validation experiment:
1. Test RAG on v1 corpus (baseline)
2. Test RAG on v2 corpus (drifted)
3. Compare Oracle-Doc performance
4. Generate validation plots
"""

import os
import json
import sys
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
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Plots will be saved as JSON.")


@dataclass
class ExperimentConfig:
    """Configuration for validation experiment."""
    name: str = "DRIFTBENCH Validation"
    use_openai: bool = False
    top_k: int = 3
    confidence_threshold: float = 0.8


def evaluate_system(
    system,
    tasks: List[Dict],
    corpus: Dict[str, str],
    version: str,
    use_oracle: bool = False
) -> List[EvaluationResult]:
    """Evaluate a RAG system on tasks."""
    results = []

    # Index corpus (if not oracle)
    if not use_oracle and hasattr(system, 'index'):
        system.index(corpus)

    for task in tasks:
        task_id = task["task_id"]
        question = task["question"]

        # Get expected answer for this version
        expected = task[f"answer_{version}"]
        evidence = task[f"evidence_{version}"]

        # Get response
        if use_oracle:
            response = system.answer(question, evidence)
        else:
            response = system.answer(question)

        # Check correctness (flexible matching)
        predicted = response.answer.lower()
        expected_lower = expected.lower()

        # Check if expected answer appears in prediction
        correct = (
            expected_lower in predicted or
            predicted in expected_lower or
            # Handle True/False matching
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

    return results


def run_validation_experiment(config: ExperimentConfig = None) -> Dict:
    """Run the full validation experiment."""
    if config is None:
        config = ExperimentConfig()

    print("=" * 70)
    print("  DRIFTBENCH VALIDATION EXPERIMENT")
    print("=" * 70)
    print(f"  Config: {config.name}")
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

    oracle_doc = OracleDoc(generator=MockGenerator())

    # Run evaluations
    all_results = {}

    # 1. Vanilla RAG on v1
    print("\n[1/6] Evaluating Vanilla RAG on v1 corpus...")
    results_vanilla_v1 = evaluate_system(vanilla_rag, tasks, corpus_v1, "v1")
    metrics_vanilla_v1 = DriftBenchMetrics(results_vanilla_v1)
    all_results["vanilla_rag_v1"] = {
        "success_rate": metrics_vanilla_v1.success_rate(),
        "sfr": metrics_vanilla_v1.silent_failure_rate(),
        "cer": metrics_vanilla_v1.confident_error_rate(),
        "ece": metrics_vanilla_v1.expected_calibration_error(),
        "retrieval_fail": metrics_vanilla_v1.retrieval_failure_rate(),
    }
    print(f"    Success: {all_results['vanilla_rag_v1']['success_rate']*100:.1f}%")

    # 2. Vanilla RAG on v2
    print("[2/6] Evaluating Vanilla RAG on v2 corpus...")
    results_vanilla_v2 = evaluate_system(vanilla_rag, tasks, corpus_v2, "v2")
    metrics_vanilla_v2 = DriftBenchMetrics(results_vanilla_v2)
    all_results["vanilla_rag_v2"] = {
        "success_rate": metrics_vanilla_v2.success_rate(),
        "sfr": metrics_vanilla_v2.silent_failure_rate(),
        "cer": metrics_vanilla_v2.confident_error_rate(),
        "ece": metrics_vanilla_v2.expected_calibration_error(),
        "retrieval_fail": metrics_vanilla_v2.retrieval_failure_rate(),
    }
    print(f"    Success: {all_results['vanilla_rag_v2']['success_rate']*100:.1f}%")

    # 3. Iterative RAG on v1
    print("[3/6] Evaluating Iterative RAG on v1 corpus...")
    results_iter_v1 = evaluate_system(iterative_rag, tasks, corpus_v1, "v1")
    metrics_iter_v1 = DriftBenchMetrics(results_iter_v1)
    all_results["iterative_rag_v1"] = {
        "success_rate": metrics_iter_v1.success_rate(),
        "sfr": metrics_iter_v1.silent_failure_rate(),
        "cer": metrics_iter_v1.confident_error_rate(),
        "ece": metrics_iter_v1.expected_calibration_error(),
        "retrieval_fail": metrics_iter_v1.retrieval_failure_rate(),
    }
    print(f"    Success: {all_results['iterative_rag_v1']['success_rate']*100:.1f}%")

    # 4. Iterative RAG on v2
    print("[4/6] Evaluating Iterative RAG on v2 corpus...")
    results_iter_v2 = evaluate_system(iterative_rag, tasks, corpus_v2, "v2")
    metrics_iter_v2 = DriftBenchMetrics(results_iter_v2)
    all_results["iterative_rag_v2"] = {
        "success_rate": metrics_iter_v2.success_rate(),
        "sfr": metrics_iter_v2.silent_failure_rate(),
        "cer": metrics_iter_v2.confident_error_rate(),
        "ece": metrics_iter_v2.expected_calibration_error(),
        "retrieval_fail": metrics_iter_v2.retrieval_failure_rate(),
    }
    print(f"    Success: {all_results['iterative_rag_v2']['success_rate']*100:.1f}%")

    # 5. Oracle-Doc on v1
    print("[5/6] Evaluating Oracle-Doc on v1...")
    results_oracle_v1 = evaluate_system(oracle_doc, tasks, corpus_v1, "v1", use_oracle=True)
    metrics_oracle_v1 = DriftBenchMetrics(results_oracle_v1)
    all_results["oracle_doc_v1"] = {
        "success_rate": metrics_oracle_v1.success_rate(),
        "sfr": metrics_oracle_v1.silent_failure_rate(),
        "cer": metrics_oracle_v1.confident_error_rate(),
        "ece": metrics_oracle_v1.expected_calibration_error(),
        "retrieval_fail": 0.0,  # Oracle always has gold retrieval
    }
    print(f"    Success: {all_results['oracle_doc_v1']['success_rate']*100:.1f}%")

    # 6. Oracle-Doc on v2
    print("[6/6] Evaluating Oracle-Doc on v2...")
    results_oracle_v2 = evaluate_system(oracle_doc, tasks, corpus_v2, "v2", use_oracle=True)
    metrics_oracle_v2 = DriftBenchMetrics(results_oracle_v2)
    all_results["oracle_doc_v2"] = {
        "success_rate": metrics_oracle_v2.success_rate(),
        "sfr": metrics_oracle_v2.silent_failure_rate(),
        "cer": metrics_oracle_v2.confident_error_rate(),
        "ece": metrics_oracle_v2.expected_calibration_error(),
        "retrieval_fail": 0.0,
    }
    print(f"    Success: {all_results['oracle_doc_v2']['success_rate']*100:.1f}%")

    # Compute key insights
    print("\n" + "=" * 70)
    print("  KEY FINDINGS")
    print("=" * 70)

    # Accuracy drop under drift
    vanilla_drop = all_results["vanilla_rag_v1"]["success_rate"] - all_results["vanilla_rag_v2"]["success_rate"]
    print(f"\n  [1] Accuracy Drop Under Drift:")
    print(f"      Vanilla RAG: {all_results['vanilla_rag_v1']['success_rate']*100:.1f}% -> {all_results['vanilla_rag_v2']['success_rate']*100:.1f}% (delta = {vanilla_drop*100:+.1f}%)")

    iter_drop = all_results["iterative_rag_v1"]["success_rate"] - all_results["iterative_rag_v2"]["success_rate"]
    print(f"      Iterative RAG: {all_results['iterative_rag_v1']['success_rate']*100:.1f}% -> {all_results['iterative_rag_v2']['success_rate']*100:.1f}% (delta = {iter_drop*100:+.1f}%)")

    # Silent failure rate
    print(f"\n  [2] Silent Failure Rate (confident but wrong):")
    print(f"      Vanilla RAG v2: {all_results['vanilla_rag_v2']['sfr']*100:.1f}%")
    print(f"      Iterative RAG v2: {all_results['iterative_rag_v2']['sfr']*100:.1f}%")

    # Oracle gap (retrieval vs reasoning)
    oracle_gap = all_results["oracle_doc_v2"]["success_rate"] - all_results["vanilla_rag_v2"]["success_rate"]
    print(f"\n  [3] Oracle Gap (retrieval-caused failures):")
    print(f"      Gap: {oracle_gap*100:.1f}%")
    print(f"      (If Oracle >> Full, failures are retrieval-dominated)")

    # Calibration
    print(f"\n  [4] Calibration (ECE):")
    print(f"      Vanilla RAG v1: {all_results['vanilla_rag_v1']['ece']:.3f}")
    print(f"      Vanilla RAG v2: {all_results['vanilla_rag_v2']['ece']:.3f}")

    # Save results
    experiment_results = {
        "config": asdict(config),
        "timestamp": datetime.now().isoformat(),
        "n_tasks": len(tasks),
        "results": all_results,
        "key_findings": {
            "vanilla_accuracy_drop": vanilla_drop,
            "iterative_accuracy_drop": iter_drop,
            "oracle_gap": oracle_gap,
            "sfr_vanilla_v2": all_results["vanilla_rag_v2"]["sfr"],
        }
    }

    results_path = results_dir / "validation_results.json"
    with open(results_path, 'w') as f:
        json.dump(experiment_results, f, indent=2)
    print(f"\n  Results saved to: {results_path}")

    # Generate plots
    if HAS_MATPLOTLIB:
        generate_validation_plots(all_results, results_dir)
    else:
        print("\n  [Plots skipped - matplotlib not installed]")

    print("\n" + "=" * 70)
    print("  VALIDATION COMPLETE")
    print("=" * 70)

    return experiment_results


def generate_validation_plots(results: Dict, output_dir: Path):
    """Generate validation plots."""
    print("\n  Generating plots...")

    # Plot 1: Accuracy v1 vs v2
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Bar chart: v1 vs v2 accuracy
    ax = axes[0]
    systems = ["Vanilla RAG", "Iterative RAG", "Oracle-Doc"]
    v1_acc = [
        results["vanilla_rag_v1"]["success_rate"],
        results["iterative_rag_v1"]["success_rate"],
        results["oracle_doc_v1"]["success_rate"],
    ]
    v2_acc = [
        results["vanilla_rag_v2"]["success_rate"],
        results["iterative_rag_v2"]["success_rate"],
        results["oracle_doc_v2"]["success_rate"],
    ]

    x = range(len(systems))
    width = 0.35

    bars1 = ax.bar([i - width/2 for i in x], v1_acc, width, label='v1 (baseline)', color='#3498db')
    bars2 = ax.bar([i + width/2 for i in x], v2_acc, width, label='v2 (drifted)', color='#e74c3c')

    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Under Version Drift')
    ax.set_xticks(x)
    ax.set_xticklabels(systems)
    ax.legend()
    ax.set_ylim(0, 1.1)

    # Add value labels
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1%}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    # Plot 2: Silent Failure Rate
    ax = axes[1]
    sfr_v1 = [
        results["vanilla_rag_v1"]["sfr"],
        results["iterative_rag_v1"]["sfr"],
        results["oracle_doc_v1"]["sfr"],
    ]
    sfr_v2 = [
        results["vanilla_rag_v2"]["sfr"],
        results["iterative_rag_v2"]["sfr"],
        results["oracle_doc_v2"]["sfr"],
    ]

    bars1 = ax.bar([i - width/2 for i in x], sfr_v1, width, label='v1', color='#3498db')
    bars2 = ax.bar([i + width/2 for i in x], sfr_v2, width, label='v2', color='#e74c3c')

    ax.set_ylabel('Silent Failure Rate')
    ax.set_title('Silent Failures (Confident + Wrong)')
    ax.set_xticks(x)
    ax.set_xticklabels(systems)
    ax.legend()
    ax.set_ylim(0, 1.0)

    # Plot 3: Oracle Gap Analysis
    ax = axes[2]
    gaps = [
        results["oracle_doc_v2"]["success_rate"] - results["vanilla_rag_v2"]["success_rate"],
        results["oracle_doc_v2"]["success_rate"] - results["iterative_rag_v2"]["success_rate"],
    ]
    gap_labels = ["Vanilla RAG", "Iterative RAG"]

    colors = ['#e74c3c' if g > 0 else '#2ecc71' for g in gaps]
    ax.barh(gap_labels, gaps, color=colors)
    ax.set_xlabel('Oracle Gap (retrieval-caused failures)')
    ax.set_title('Failure Attribution')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()

    plot_path = output_dir / "validation_plots.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.savefig(str(plot_path).replace('.png', '.pdf'), bbox_inches='tight')
    print(f"    Saved: {plot_path}")

    plt.close()


if __name__ == "__main__":
    results = run_validation_experiment()
