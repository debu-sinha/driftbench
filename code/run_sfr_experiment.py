"""
SFR Forcing Experiment - Make model answer without hedging to reveal silent failures.
"""

import os
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent))

from driftbench_core import DriftBenchMetrics, EvaluationResult
from rag_baselines import VanillaRAG, SimpleRetriever, load_corpus, RAGResponse, RetrievalResult

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class NoHedgingGenerator:
    """Generator that forces confident answers without hedging."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = openai.OpenAI()
        self.model = model

    def generate(self, question: str, context: str) -> Tuple[str, float, bool]:
        prompt = f"""You are a confident technical assistant. Answer the question directly and definitively.

IMPORTANT RULES:
- Give a direct, confident answer
- Do NOT say "I'm not sure", "might be", "could be", "it depends"
- Do NOT hedge or qualify your answer
- State the answer as fact, even if unsure
- Be brief and definitive

Documentation:
{context}

Question: {question}

Provide your answer in JSON format:
{{"answer": "your definitive answer", "confidence": 0.95}}

Remember: Be CONFIDENT. No hedging. No uncertainty language."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=300
            )

            raw = response.choices[0].message.content
            import re
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return (
                    result.get("answer", ""),
                    float(result.get("confidence", 0.95)),
                    False  # Never express uncertainty
                )
            return raw.strip(), 0.95, False

        except Exception as e:
            return f"Error: {e}", 0.9, False


def run_sfr_experiment():
    """Run experiment to reveal silent failures."""
    print("=" * 70)
    print("  DRIFTBENCH SFR FORCING EXPERIMENT")
    print("=" * 70)
    print("  Goal: Force confident answers to reveal silent failures")
    print("=" * 70)

    data_dir = Path(__file__).parent.parent / "data"
    results_dir = Path(__file__).parent.parent / "results"

    corpus_v1 = load_corpus(data_dir / "combined_corpus_v1.json")
    corpus_v2 = load_corpus(data_dir / "combined_corpus_v2.json")

    with open(data_dir / "driftbench_combined.json", 'r') as f:
        dataset = json.load(f)

    tasks = [t for t in dataset["tasks"] if "answer_v1" in t and "answer_v2" in t][:20]
    print(f"\n  Using {len(tasks)} tasks")

    # Initialize no-hedging RAG
    generator = NoHedgingGenerator()
    rag = VanillaRAG(retriever=SimpleRetriever(), generator=generator, top_k=3)

    results_v1 = []
    results_v2 = []

    # Evaluate on V1 corpus (answer should match v1)
    print("\n[1/2] No-hedging RAG on V1...")
    rag.index(corpus_v1)
    for i, task in enumerate(tasks):
        print(f"  [{i+1}/{len(tasks)}]", end=" ", flush=True)
        response = rag.answer(task["question"])
        expected = task["answer_v1"].lower()
        predicted = response.answer.lower()

        correct = expected in predicted or predicted in expected
        silent_failure = not correct and response.confidence >= 0.8 and not response.expressed_uncertainty

        print("SFR!" if silent_failure else ("OK" if correct else "wrong"))

        results_v1.append(EvaluationResult(
            task_id=task["task_id"], correct=correct, confidence=response.confidence,
            silent_failure=silent_failure, retrieval_success=True, version_tested="v1"
        ))

    # Evaluate on V2 corpus (answer should match v2)
    print("\n[2/2] No-hedging RAG on V2...")
    rag.index(corpus_v2)
    for i, task in enumerate(tasks):
        print(f"  [{i+1}/{len(tasks)}]", end=" ", flush=True)
        response = rag.answer(task["question"])
        expected = task["answer_v2"].lower()
        predicted = response.answer.lower()

        correct = expected in predicted or predicted in expected
        silent_failure = not correct and response.confidence >= 0.8 and not response.expressed_uncertainty

        print("SFR!" if silent_failure else ("OK" if correct else "wrong"))

        results_v2.append(EvaluationResult(
            task_id=task["task_id"], correct=correct, confidence=response.confidence,
            silent_failure=silent_failure, retrieval_success=True, version_tested="v2"
        ))

    # Compute metrics
    metrics_v1 = DriftBenchMetrics(results_v1)
    metrics_v2 = DriftBenchMetrics(results_v2)

    print("\n" + "=" * 70)
    print("  RESULTS: NO-HEDGING GENERATOR")
    print("=" * 70)
    print(f"\n  V1 Corpus:")
    print(f"    Success Rate: {metrics_v1.success_rate()*100:.1f}%")
    print(f"    Silent Failure Rate: {metrics_v1.silent_failure_rate()*100:.1f}%")
    print(f"    CER: {metrics_v1.confident_error_rate()*100:.1f}%")

    print(f"\n  V2 Corpus:")
    print(f"    Success Rate: {metrics_v2.success_rate()*100:.1f}%")
    print(f"    Silent Failure Rate: {metrics_v2.silent_failure_rate()*100:.1f}%")
    print(f"    CER: {metrics_v2.confident_error_rate()*100:.1f}%")

    # Save results
    experiment_results = {
        "experiment": "SFR Forcing (No-Hedging Generator)",
        "timestamp": datetime.now().isoformat(),
        "n_tasks": len(tasks),
        "v1": {
            "success_rate": metrics_v1.success_rate(),
            "sfr": metrics_v1.silent_failure_rate(),
            "cer": metrics_v1.confident_error_rate(),
        },
        "v2": {
            "success_rate": metrics_v2.success_rate(),
            "sfr": metrics_v2.silent_failure_rate(),
            "cer": metrics_v2.confident_error_rate(),
        }
    }

    with open(results_dir / "sfr_experiment_results.json", 'w') as f:
        json.dump(experiment_results, f, indent=2)

    # Generate plot
    if HAS_MATPLOTLIB:
        fig, ax = plt.subplots(figsize=(8, 5))

        metrics = ['Success Rate', 'Silent Failure Rate', 'Confident Error Rate']
        v1_vals = [metrics_v1.success_rate()*100, metrics_v1.silent_failure_rate()*100, metrics_v1.confident_error_rate()*100]
        v2_vals = [metrics_v2.success_rate()*100, metrics_v2.silent_failure_rate()*100, metrics_v2.confident_error_rate()*100]

        x = range(len(metrics))
        width = 0.35

        bars1 = ax.bar([i - width/2 for i in x], v1_vals, width, label='V1', color='#3498db', edgecolor='black')
        bars2 = ax.bar([i + width/2 for i in x], v2_vals, width, label='V2', color='#e74c3c', edgecolor='black')

        ax.set_ylabel('Percentage (%)')
        ax.set_title('No-Hedging Generator: Silent Failures Revealed', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.set_ylim(0, 100)

        for bar in bars1 + bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(results_dir / "sfr_experiment_plot.png", dpi=150)
        plt.savefig(results_dir / "sfr_experiment_plot.pdf")
        print(f"\n  Saved: sfr_experiment_plot.png/pdf")

    print("\n" + "=" * 70)

    return experiment_results


if __name__ == "__main__":
    run_sfr_experiment()
