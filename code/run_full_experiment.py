"""
DRIFTBENCH Full Experiment with Bootstrap CI

Runs on ALL 77 tasks with confidence intervals for publication.
"""

import os
import json
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict

sys.path.insert(0, str(Path(__file__).parent))

from driftbench_core import DriftBenchMetrics, EvaluationResult
from rag_baselines import (
    VanillaRAG, OracleDoc, SimpleRetriever, EmbeddingRetriever,
    OpenAIGenerator, load_corpus
)


def bootstrap_ci(values: List[bool], n_bootstrap: int = 1000, ci: float = 0.95) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval for a rate."""
    if len(values) == 0:
        return 0.0, 0.0, 0.0

    values = np.array(values, dtype=float)
    point_estimate = np.mean(values)

    # Bootstrap
    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=len(values), replace=True)
        boot_means.append(np.mean(sample))

    alpha = 1 - ci
    lower = np.percentile(boot_means, alpha/2 * 100)
    upper = np.percentile(boot_means, (1 - alpha/2) * 100)

    return point_estimate, lower, upper


def evaluate_system(system, tasks, corpus, version, use_oracle=False):
    """Evaluate system on all tasks, return per-task results."""
    results = []

    if not use_oracle and hasattr(system, 'index'):
        system.index(corpus)

    for i, task in enumerate(tasks):
        print(f".", end="", flush=True)
        if (i + 1) % 20 == 0:
            print(f" [{i+1}/{len(tasks)}]")

        try:
            if use_oracle:
                response = system.answer(task["question"], task[f"evidence_{version}"])
            else:
                response = system.answer(task["question"])

            expected = task[f"answer_{version}"].lower()
            predicted = response.answer.lower()

            correct = (
                expected in predicted or
                predicted in expected or
                any(term in predicted for term in expected.split() if len(term) > 4)
            )

            silent_failure = not correct and response.confidence >= 0.8 and not response.expressed_uncertainty

            results.append({
                "task_id": task["task_id"],
                "correct": correct,
                "confidence": response.confidence,
                "silent_failure": silent_failure,
                "version": version
            })

        except Exception as e:
            results.append({
                "task_id": task["task_id"],
                "correct": False,
                "confidence": 0.0,
                "silent_failure": False,
                "version": version,
                "error": str(e)
            })

    print()
    return results


def compute_metrics_with_ci(results: List[Dict]) -> Dict:
    """Compute metrics with bootstrap CI."""
    correct = [r["correct"] for r in results]
    silent_failures = [r["silent_failure"] for r in results]

    acc, acc_lo, acc_hi = bootstrap_ci(correct)
    sfr, sfr_lo, sfr_hi = bootstrap_ci(silent_failures)

    return {
        "n": len(results),
        "accuracy": {"mean": acc, "ci_lower": acc_lo, "ci_upper": acc_hi},
        "sfr": {"mean": sfr, "ci_lower": sfr_lo, "ci_upper": sfr_hi}
    }


def run_full_experiment():
    """Run full experiment on all 77 tasks."""
    print("=" * 70)
    print("  DRIFTBENCH FULL EXPERIMENT (ALL TASKS + BOOTSTRAP CI)")
    print("=" * 70)

    data_dir = Path(__file__).parent.parent / "data"
    results_dir = Path(__file__).parent.parent / "results"

    corpus_v1 = load_corpus(data_dir / "combined_corpus_v1.json")
    corpus_v2 = load_corpus(data_dir / "combined_corpus_v2.json")

    with open(data_dir / "driftbench_combined.json") as f:
        dataset = json.load(f)

    tasks = [t for t in dataset["tasks"] if "answer_v1" in t and "answer_v2" in t]
    print(f"\n  Total tasks: {len(tasks)}")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    generator = OpenAIGenerator("gpt-4o-mini")

    # Systems
    term_rag = VanillaRAG(retriever=SimpleRetriever(), generator=generator, top_k=3)
    dense_rag = VanillaRAG(retriever=EmbeddingRetriever("all-MiniLM-L6-v2"), generator=generator, top_k=3)
    oracle = OracleDoc(generator=generator)

    all_results = {}

    # 1. Term Overlap RAG
    print("\n[1/6] Term Overlap RAG on V1...")
    r = evaluate_system(term_rag, tasks, corpus_v1, "v1")
    all_results["term_rag_v1"] = {"raw": r, **compute_metrics_with_ci(r)}

    print("[2/6] Term Overlap RAG on V2...")
    r = evaluate_system(term_rag, tasks, corpus_v2, "v2")
    all_results["term_rag_v2"] = {"raw": r, **compute_metrics_with_ci(r)}

    # 2. Dense RAG
    print("[3/6] Dense RAG (MiniLM) on V1...")
    r = evaluate_system(dense_rag, tasks, corpus_v1, "v1")
    all_results["dense_rag_v1"] = {"raw": r, **compute_metrics_with_ci(r)}

    print("[4/6] Dense RAG (MiniLM) on V2...")
    r = evaluate_system(dense_rag, tasks, corpus_v2, "v2")
    all_results["dense_rag_v2"] = {"raw": r, **compute_metrics_with_ci(r)}

    # 3. Oracle-Doc
    print("[5/6] Oracle-Doc on V1...")
    r = evaluate_system(oracle, tasks, corpus_v1, "v1", use_oracle=True)
    all_results["oracle_v1"] = {"raw": r, **compute_metrics_with_ci(r)}

    print("[6/6] Oracle-Doc on V2...")
    r = evaluate_system(oracle, tasks, corpus_v2, "v2", use_oracle=True)
    all_results["oracle_v2"] = {"raw": r, **compute_metrics_with_ci(r)}

    # Summary
    print("\n" + "=" * 70)
    print("  RESULTS (N={}, 95% CI)".format(len(tasks)))
    print("=" * 70)

    def fmt(m):
        return f"{m['mean']*100:.1f}% [{m['ci_lower']*100:.1f}-{m['ci_upper']*100:.1f}]"

    print(f"\n  Term Overlap RAG:")
    print(f"    V1 Acc: {fmt(all_results['term_rag_v1']['accuracy'])}")
    print(f"    V2 Acc: {fmt(all_results['term_rag_v2']['accuracy'])}")
    print(f"    V1 SFR: {fmt(all_results['term_rag_v1']['sfr'])}")
    print(f"    V2 SFR: {fmt(all_results['term_rag_v2']['sfr'])}")

    print(f"\n  Dense RAG (MiniLM):")
    print(f"    V1 Acc: {fmt(all_results['dense_rag_v1']['accuracy'])}")
    print(f"    V2 Acc: {fmt(all_results['dense_rag_v2']['accuracy'])}")
    print(f"    V1 SFR: {fmt(all_results['dense_rag_v1']['sfr'])}")
    print(f"    V2 SFR: {fmt(all_results['dense_rag_v2']['sfr'])}")

    print(f"\n  Oracle-Doc:")
    print(f"    V1 Acc: {fmt(all_results['oracle_v1']['accuracy'])}")
    print(f"    V2 Acc: {fmt(all_results['oracle_v2']['accuracy'])}")

    # Key findings
    acc_drop = all_results['term_rag_v1']['accuracy']['mean'] - all_results['term_rag_v2']['accuracy']['mean']
    oracle_gap = all_results['oracle_v2']['accuracy']['mean'] - all_results['term_rag_v2']['accuracy']['mean']

    print(f"\n  Key Findings:")
    print(f"    Accuracy drop (V1->V2): {acc_drop*100:.1f}%")
    print(f"    Oracle gap (V2): {oracle_gap*100:.1f}%")

    # Save
    output = {
        "experiment": "DRIFTBENCH Full Experiment",
        "timestamp": datetime.now().isoformat(),
        "n_tasks": len(tasks),
        "model": "gpt-4o-mini",
        "results": {k: {kk: vv for kk, vv in v.items() if kk != "raw"} for k, v in all_results.items()},
        "raw_results": {k: v["raw"] for k, v in all_results.items()},
        "key_findings": {
            "accuracy_drop": acc_drop,
            "oracle_gap": oracle_gap
        }
    }

    with open(results_dir / "full_experiment_results.json", 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n  Saved: results/full_experiment_results.json")
    print("=" * 70)

    return output


if __name__ == "__main__":
    run_full_experiment()
