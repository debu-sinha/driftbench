"""
DRIFTBENCH Real LLM Experiment

Runs the full validation experiment with GPT-4o-mini to observe actual drift effects.
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
    compute_reliability_half_life
)
from rag_baselines import (
    VanillaRAG,
    OracleDoc,
    IterativeRAG,
    SimpleRetriever,
    EmbeddingRetriever,
    OpenAIGenerator,
    load_corpus
)

# Try matplotlib
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class LLMExperimentConfig:
    """Configuration for LLM experiment."""
    name: str = "DRIFTBENCH LLM Validation"
    model: str = "gpt-4o-mini"
    top_k: int = 3
    max_tasks: int = 20  # Limit for cost control


def evaluate_with_llm(
    system,
    tasks: List[Dict],
    corpus: Dict[str, str],
    version: str,
    use_oracle: bool = False,
    max_tasks: int = None
) -> List[EvaluationResult]:
    """Evaluate a RAG system with real LLM."""
    results = []

    # Limit tasks if specified
    eval_tasks = tasks[:max_tasks] if max_tasks else tasks

    # Index corpus (if not oracle)
    if not use_oracle and hasattr(system, 'index'):
        system.index(corpus)

    for i, task in enumerate(eval_tasks):
        task_id = task["task_id"]
        question = task["question"]

        # Get expected answer for this version
        expected = task[f"answer_{version}"]
        evidence = task[f"evidence_{version}"]

        print(f"    [{i+1}/{len(eval_tasks)}] {task_id[:20]}...", end=" ", flush=True)

        try:
            # Get response
            if use_oracle:
                response = system.answer(question, evidence)
            else:
                response = system.answer(question)

            # Check correctness (flexible matching)
            predicted = response.answer.lower()
            expected_lower = expected.lower()

            # Extract key terms for matching
            correct = (
                expected_lower in predicted or
                predicted in expected_lower or
                (expected_lower == "true" and "true" in predicted and "false" not in predicted) or
                (expected_lower == "false" and "false" in predicted and "true" not in predicted) or
                # Check for key method/class names
                any(term in predicted for term in expected_lower.split() if len(term) > 4)
            )

            # Check retrieval success
            retrieval_success = False
            for doc in response.retrieved_docs:
                if any(term in doc.content.lower() for term in expected_lower.split() if len(term) > 4):
                    retrieval_success = True
                    break

            # Determine silent failure
            silent_failure = (
                not correct and
                response.confidence >= 0.8 and
                not response.expressed_uncertainty
            )

            print(f"{'OK' if correct else 'WRONG'} (conf={response.confidence:.2f})")

            results.append(EvaluationResult(
                task_id=task_id,
                correct=correct,
                confidence=response.confidence,
                silent_failure=silent_failure,
                retrieval_success=retrieval_success,
                version_tested=version
            ))

        except Exception as e:
            print(f"ERROR: {e}")
            results.append(EvaluationResult(
                task_id=task_id,
                correct=False,
                confidence=0.0,
                silent_failure=False,
                retrieval_success=False,
                version_tested=version
            ))

    return results


def run_llm_experiment(config: LLMExperimentConfig = None) -> Dict:
    """Run the LLM validation experiment."""
    if config is None:
        config = LLMExperimentConfig()

    print("=" * 70)
    print("  DRIFTBENCH LLM EXPERIMENT")
    print("=" * 70)
    print(f"  Model: {config.model}")
    print(f"  Max tasks: {config.max_tasks}")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    # Load data
    data_dir = Path(__file__).parent.parent / "data"
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Use combined dataset
    corpus_v1 = load_corpus(data_dir / "combined_corpus_v1.json")
    corpus_v2 = load_corpus(data_dir / "combined_corpus_v2.json")

    with open(data_dir / "driftbench_combined.json", 'r') as f:
        dataset = json.load(f)

    # Filter to factoid tasks with standard format
    tasks = [t for t in dataset["tasks"] if "answer_v1" in t and "answer_v2" in t]
    print(f"\n  Loaded {len(tasks)} tasks (using {min(config.max_tasks, len(tasks))})")

    # Initialize generator
    generator = OpenAIGenerator(model=config.model)

    # Initialize systems
    vanilla_rag = VanillaRAG(
        retriever=SimpleRetriever(),
        generator=generator,
        top_k=config.top_k
    )

    oracle_doc = OracleDoc(generator=generator)

    # Results storage
    all_results = {}

    # 1. Vanilla RAG on v1
    print("\n[1/4] Vanilla RAG on v1 corpus...")
    results_v1 = evaluate_with_llm(vanilla_rag, tasks, corpus_v1, "v1", max_tasks=config.max_tasks)
    metrics_v1 = DriftBenchMetrics(results_v1)
    all_results["vanilla_rag_v1"] = {
        "success_rate": metrics_v1.success_rate(),
        "sfr": metrics_v1.silent_failure_rate(),
        "cer": metrics_v1.confident_error_rate(),
        "ece": metrics_v1.expected_calibration_error(),
    }
    print(f"    -> Success: {all_results['vanilla_rag_v1']['success_rate']*100:.1f}%")
    print(f"    -> SFR: {all_results['vanilla_rag_v1']['sfr']*100:.1f}%")

    # 2. Vanilla RAG on v2
    print("\n[2/4] Vanilla RAG on v2 corpus...")
    results_v2 = evaluate_with_llm(vanilla_rag, tasks, corpus_v2, "v2", max_tasks=config.max_tasks)
    metrics_v2 = DriftBenchMetrics(results_v2)
    all_results["vanilla_rag_v2"] = {
        "success_rate": metrics_v2.success_rate(),
        "sfr": metrics_v2.silent_failure_rate(),
        "cer": metrics_v2.confident_error_rate(),
        "ece": metrics_v2.expected_calibration_error(),
    }
    print(f"    -> Success: {all_results['vanilla_rag_v2']['success_rate']*100:.1f}%")
    print(f"    -> SFR: {all_results['vanilla_rag_v2']['sfr']*100:.1f}%")

    # 3. Oracle-Doc on v1
    print("\n[3/4] Oracle-Doc on v1...")
    results_oracle_v1 = evaluate_with_llm(oracle_doc, tasks, corpus_v1, "v1", use_oracle=True, max_tasks=config.max_tasks)
    metrics_oracle_v1 = DriftBenchMetrics(results_oracle_v1)
    all_results["oracle_doc_v1"] = {
        "success_rate": metrics_oracle_v1.success_rate(),
        "sfr": metrics_oracle_v1.silent_failure_rate(),
        "cer": metrics_oracle_v1.confident_error_rate(),
        "ece": metrics_oracle_v1.expected_calibration_error(),
    }
    print(f"    -> Success: {all_results['oracle_doc_v1']['success_rate']*100:.1f}%")

    # 4. Oracle-Doc on v2
    print("\n[4/4] Oracle-Doc on v2...")
    results_oracle_v2 = evaluate_with_llm(oracle_doc, tasks, corpus_v2, "v2", use_oracle=True, max_tasks=config.max_tasks)
    metrics_oracle_v2 = DriftBenchMetrics(results_oracle_v2)
    all_results["oracle_doc_v2"] = {
        "success_rate": metrics_oracle_v2.success_rate(),
        "sfr": metrics_oracle_v2.silent_failure_rate(),
        "cer": metrics_oracle_v2.confident_error_rate(),
        "ece": metrics_oracle_v2.expected_calibration_error(),
    }
    print(f"    -> Success: {all_results['oracle_doc_v2']['success_rate']*100:.1f}%")

    # Analysis
    print("\n" + "=" * 70)
    print("  KEY FINDINGS")
    print("=" * 70)

    # Accuracy drop
    acc_drop = all_results["vanilla_rag_v1"]["success_rate"] - all_results["vanilla_rag_v2"]["success_rate"]
    print(f"\n  [1] Accuracy Drop Under Drift:")
    print(f"      V1: {all_results['vanilla_rag_v1']['success_rate']*100:.1f}%")
    print(f"      V2: {all_results['vanilla_rag_v2']['success_rate']*100:.1f}%")
    print(f"      Drop: {acc_drop*100:+.1f}%")

    # Silent failures
    print(f"\n  [2] Silent Failure Rate:")
    print(f"      V1: {all_results['vanilla_rag_v1']['sfr']*100:.1f}%")
    print(f"      V2: {all_results['vanilla_rag_v2']['sfr']*100:.1f}%")

    # Oracle gap
    oracle_gap = all_results["oracle_doc_v2"]["success_rate"] - all_results["vanilla_rag_v2"]["success_rate"]
    print(f"\n  [3] Oracle Gap (retrieval bottleneck):")
    print(f"      Vanilla RAG v2: {all_results['vanilla_rag_v2']['success_rate']*100:.1f}%")
    print(f"      Oracle-Doc v2: {all_results['oracle_doc_v2']['success_rate']*100:.1f}%")
    print(f"      Gap: {oracle_gap*100:.1f}%")

    # Save results
    experiment_results = {
        "config": asdict(config),
        "timestamp": datetime.now().isoformat(),
        "n_tasks_evaluated": min(config.max_tasks, len(tasks)),
        "results": all_results,
        "key_findings": {
            "accuracy_drop": acc_drop,
            "sfr_v1": all_results["vanilla_rag_v1"]["sfr"],
            "sfr_v2": all_results["vanilla_rag_v2"]["sfr"],
            "oracle_gap": oracle_gap,
        }
    }

    results_path = results_dir / "llm_experiment_results.json"
    with open(results_path, 'w') as f:
        json.dump(experiment_results, f, indent=2)
    print(f"\n  Results saved to: {results_path}")

    # Generate plots
    if HAS_MATPLOTLIB:
        generate_llm_plots(all_results, results_dir)

    print("\n" + "=" * 70)
    print("  LLM EXPERIMENT COMPLETE")
    print("=" * 70)

    return experiment_results


def generate_llm_plots(results: Dict, output_dir: Path):
    """Generate LLM experiment plots."""
    print("\n  Generating plots...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: V1 vs V2 comparison
    ax = axes[0]
    systems = ["Vanilla RAG", "Oracle-Doc"]
    v1_acc = [
        results["vanilla_rag_v1"]["success_rate"] * 100,
        results["oracle_doc_v1"]["success_rate"] * 100,
    ]
    v2_acc = [
        results["vanilla_rag_v2"]["success_rate"] * 100,
        results["oracle_doc_v2"]["success_rate"] * 100,
    ]

    x = range(len(systems))
    width = 0.35

    bars1 = ax.bar([i - width/2 for i in x], v1_acc, width, label='V1 (baseline)', color='#3498db')
    bars2 = ax.bar([i + width/2 for i in x], v2_acc, width, label='V2 (drifted)', color='#e74c3c')

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy Under Version Drift (Real LLM)')
    ax.set_xticks(x)
    ax.set_xticklabels(systems)
    ax.legend()
    ax.set_ylim(0, 100)

    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    # Plot 2: Silent Failure Rate
    ax = axes[1]
    sfr_v1 = [
        results["vanilla_rag_v1"]["sfr"] * 100,
        results["oracle_doc_v1"]["sfr"] * 100,
    ]
    sfr_v2 = [
        results["vanilla_rag_v2"]["sfr"] * 100,
        results["oracle_doc_v2"]["sfr"] * 100,
    ]

    bars1 = ax.bar([i - width/2 for i in x], sfr_v1, width, label='V1', color='#3498db')
    bars2 = ax.bar([i + width/2 for i in x], sfr_v2, width, label='V2', color='#e74c3c')

    ax.set_ylabel('Silent Failure Rate (%)')
    ax.set_title('Silent Failures: Confident + Wrong')
    ax.set_xticks(x)
    ax.set_xticklabels(systems)
    ax.legend()
    ax.set_ylim(0, max(max(sfr_v1), max(sfr_v2)) * 1.5 + 5)

    plt.tight_layout()

    plot_path = output_dir / "llm_experiment_plots.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.savefig(str(plot_path).replace('.png', '.pdf'), bbox_inches='tight')
    print(f"    Saved: {plot_path}")

    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tasks", type=int, default=20, help="Max tasks to evaluate")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model")
    args = parser.parse_args()

    config = LLMExperimentConfig(
        max_tasks=args.max_tasks,
        model=args.model
    )
    run_llm_experiment(config)
