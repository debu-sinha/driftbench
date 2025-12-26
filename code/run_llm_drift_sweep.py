"""
DRIFTBENCH LLM Drift Dose Sweep

Runs drift dose sweep with real LLM to observe reliability half-life.
"""

import os
import json
import sys
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict

sys.path.insert(0, str(Path(__file__).parent))

from driftbench_core import (
    DriftBenchMetrics,
    EvaluationResult,
    compute_reliability_half_life
)
from rag_baselines import (
    VanillaRAG,
    SimpleRetriever,
    OpenAIGenerator,
    load_corpus
)

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class LLMSweepConfig:
    name: str = "DRIFTBENCH LLM Drift Sweep"
    model: str = "gpt-4o-mini"
    drift_doses: Tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 1.0)  # Fewer doses to save cost
    top_k: int = 3
    max_tasks: int = 15
    seed: int = 42


def create_drifted_corpus(
    corpus_v1: Dict[str, str],
    corpus_v2: Dict[str, str],
    drift_dose: float,
    seed: int = 42
) -> Dict[str, str]:
    """Create corpus with controlled drift percentage."""
    random.seed(seed)
    drifted = {}

    task_doc_ids = [k for k in corpus_v1.keys() if not k.startswith("shared_")]
    shared_doc_ids = [k for k in corpus_v1.keys() if k.startswith("shared_")]

    for doc_id in task_doc_ids:
        if random.random() < drift_dose:
            drifted[doc_id] = corpus_v2.get(doc_id, corpus_v1[doc_id])
        else:
            drifted[doc_id] = corpus_v1[doc_id]

    for doc_id in shared_doc_ids:
        drifted[doc_id] = corpus_v2.get(doc_id, corpus_v1[doc_id])

    return drifted


def evaluate_at_dose(
    system,
    tasks: List[Dict],
    corpus_v1: Dict[str, str],
    corpus_v2: Dict[str, str],
    drift_dose: float,
    max_tasks: int,
    seed: int = 42
) -> List[EvaluationResult]:
    """Evaluate at specific drift dose."""
    drifted_corpus = create_drifted_corpus(corpus_v1, corpus_v2, drift_dose, seed)

    if hasattr(system, 'index'):
        system.index(drifted_corpus)

    results = []
    eval_tasks = tasks[:max_tasks]

    for task in eval_tasks:
        task_id = task["task_id"]
        question = task["question"]

        # Determine expected answer based on corpus version
        doc_content = drifted_corpus.get(task_id, "")
        v1_content = corpus_v1.get(task_id, "")

        if doc_content == v1_content:
            expected = task["answer_v1"]
        else:
            expected = task["answer_v2"]

        try:
            response = system.answer(question)

            predicted = response.answer.lower()
            expected_lower = expected.lower()

            correct = (
                expected_lower in predicted or
                predicted in expected_lower or
                any(term in predicted for term in expected_lower.split() if len(term) > 4)
            )

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
                retrieval_success=True,
                version_tested="mixed"
            ))

        except Exception as e:
            results.append(EvaluationResult(
                task_id=task_id,
                correct=False,
                confidence=0.0,
                silent_failure=False,
                retrieval_success=False,
                version_tested="mixed"
            ))

    return results


def run_llm_sweep(config: LLMSweepConfig = None) -> Dict:
    """Run drift sweep with real LLM."""
    if config is None:
        config = LLMSweepConfig()

    print("=" * 70)
    print("  DRIFTBENCH LLM DRIFT SWEEP")
    print("=" * 70)
    print(f"  Model: {config.model}")
    print(f"  Drift doses: {config.drift_doses}")
    print(f"  Max tasks per dose: {config.max_tasks}")
    print("=" * 70)

    data_dir = Path(__file__).parent.parent / "data"
    results_dir = Path(__file__).parent.parent / "results"

    # Load combined corpus
    corpus_v1 = load_corpus(data_dir / "combined_corpus_v1.json")
    corpus_v2 = load_corpus(data_dir / "combined_corpus_v2.json")

    with open(data_dir / "driftbench_combined.json", 'r') as f:
        dataset = json.load(f)

    tasks = [t for t in dataset["tasks"] if "answer_v1" in t and "answer_v2" in t]
    print(f"\n  Using {config.max_tasks} of {len(tasks)} tasks per dose")

    # Initialize system
    generator = OpenAIGenerator(model=config.model)
    vanilla_rag = VanillaRAG(
        retriever=SimpleRetriever(),
        generator=generator,
        top_k=config.top_k
    )

    sweep_results = {
        "doses": [],
        "success_rates": [],
        "sfr": [],
        "cer": [],
    }

    for drift_dose in config.drift_doses:
        print(f"\n  Drift dose {drift_dose*100:.0f}%...")

        results = evaluate_at_dose(
            vanilla_rag, tasks, corpus_v1, corpus_v2,
            drift_dose, config.max_tasks, config.seed
        )

        metrics = DriftBenchMetrics(results)

        sweep_results["doses"].append(drift_dose)
        sweep_results["success_rates"].append(metrics.success_rate())
        sweep_results["sfr"].append(metrics.silent_failure_rate())
        sweep_results["cer"].append(metrics.confident_error_rate())

        print(f"    Success: {metrics.success_rate()*100:.1f}%")
        print(f"    SFR: {metrics.silent_failure_rate()*100:.1f}%")

    # Compute half-life
    half_life = compute_reliability_half_life(
        sweep_results["doses"],
        sweep_results["success_rates"]
    )
    sweep_results["half_life"] = half_life

    print("\n" + "=" * 70)
    print("  RELIABILITY HALF-LIFE")
    print("=" * 70)
    if half_life is not None:
        print(f"  d_1/2 = {half_life*100:.1f}% drift")
    else:
        print("  d_1/2 = N/A (never drops to 50%)")

    # Save results
    experiment_results = {
        "config": asdict(config),
        "timestamp": datetime.now().isoformat(),
        "sweep_results": sweep_results,
    }

    results_path = results_dir / "llm_drift_sweep_results.json"
    with open(results_path, 'w') as f:
        json.dump(experiment_results, f, indent=2)
    print(f"\n  Results saved to: {results_path}")

    if HAS_MATPLOTLIB:
        generate_decay_plot(sweep_results, results_dir)

    print("\n" + "=" * 70)
    print("  LLM DRIFT SWEEP COMPLETE")
    print("=" * 70)

    return experiment_results


def generate_decay_plot(results: Dict, output_dir: Path):
    """Generate decay curve plot."""
    print("\n  Generating decay curve...")

    fig, ax = plt.subplots(figsize=(8, 6))

    doses_pct = [d * 100 for d in results["doses"]]
    rates_pct = [r * 100 for r in results["success_rates"]]

    ax.plot(doses_pct, rates_pct, 'o-', color='#e74c3c', linewidth=2, markersize=10, label='Vanilla RAG')

    # Add baseline and half-life markers
    baseline = rates_pct[0]
    ax.axhline(y=baseline, color='#3498db', linestyle='--', alpha=0.5, label=f'Baseline ({baseline:.1f}%)')
    ax.axhline(y=baseline/2, color='#2ecc71', linestyle='--', alpha=0.5, label=f'Half-life threshold ({baseline/2:.1f}%)')

    if results.get("half_life") is not None:
        hl = results["half_life"] * 100
        ax.axvline(x=hl, color='#9b59b6', linestyle=':', alpha=0.7)
        ax.scatter([hl], [baseline/2], color='#9b59b6', s=150, marker='*', zorder=5, label=f'd_1/2 = {hl:.1f}%')

    ax.set_xlabel('Drift Dose (%)', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Reliability Decay Under Drift (GPT-4o-mini)', fontsize=14)
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 105)
    ax.set_ylim(0, 100)

    plt.tight_layout()

    plot_path = output_dir / "llm_decay_curve.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.savefig(str(plot_path).replace('.png', '.pdf'), bbox_inches='tight')
    print(f"    Saved: {plot_path}")

    plt.close()


if __name__ == "__main__":
    run_llm_sweep()
