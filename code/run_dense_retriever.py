"""Dense retriever baseline experiment."""
import json, sys
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent))

from driftbench_core import DriftBenchMetrics, EvaluationResult
from rag_baselines import VanillaRAG, EmbeddingRetriever, OpenAIGenerator, load_corpus

def run():
    data_dir = Path(__file__).parent.parent / "data"
    results_dir = Path(__file__).parent.parent / "results"

    corpus_v1 = load_corpus(data_dir / "combined_corpus_v1.json")
    corpus_v2 = load_corpus(data_dir / "combined_corpus_v2.json")

    with open(data_dir / "driftbench_combined.json") as f:
        tasks = [t for t in json.load(f)["tasks"] if "answer_v1" in t][:15]

    print("DENSE RETRIEVER EXPERIMENT (all-MiniLM-L6-v2 + GPT-4o-mini)")
    print("=" * 60)

    dense_rag = VanillaRAG(
        retriever=EmbeddingRetriever("all-MiniLM-L6-v2"),
        generator=OpenAIGenerator("gpt-4o-mini"),
        top_k=3
    )

    all_results = {}

    for version, corpus, answer_key in [("v1", corpus_v1, "answer_v1"), ("v2", corpus_v2, "answer_v2")]:
        print(f"\n[{version.upper()}]")
        dense_rag.index(corpus)
        results = []
        for t in tasks:
            r = dense_rag.answer(t["question"])
            correct = t[answer_key].lower() in r.answer.lower() or r.answer.lower() in t[answer_key].lower()
            silent_failure = not correct and r.confidence >= 0.8 and not r.expressed_uncertainty
            results.append(EvaluationResult(t["task_id"], correct, r.confidence,
                silent_failure, True, version))
            print("." if correct else "X", end="", flush=True)

        m = DriftBenchMetrics(results)
        print(f"\n  Acc: {m.success_rate()*100:.1f}%  SFR: {m.silent_failure_rate()*100:.1f}%")

        all_results[f"dense_rag_{version}"] = {
            "success_rate": m.success_rate(),
            "sfr": m.silent_failure_rate(),
            "cer": m.confident_error_rate(),
            "n_tasks": len(tasks)
        }

    # Save results to JSON
    experiment_output = {
        "experiment": "Dense Retriever Baseline",
        "config": {
            "retriever": "all-MiniLM-L6-v2",
            "generator": "gpt-4o-mini",
            "top_k": 3,
            "n_tasks": len(tasks)
        },
        "timestamp": datetime.now().isoformat(),
        "results": all_results,
        "key_findings": {
            "v1_accuracy": all_results["dense_rag_v1"]["success_rate"],
            "v2_accuracy": all_results["dense_rag_v2"]["success_rate"],
            "v1_sfr": all_results["dense_rag_v1"]["sfr"],
            "v2_sfr": all_results["dense_rag_v2"]["sfr"],
            "accuracy_drop": all_results["dense_rag_v1"]["success_rate"] - all_results["dense_rag_v2"]["success_rate"]
        }
    }

    output_path = results_dir / "dense_retriever_results.json"
    with open(output_path, 'w') as f:
        json.dump(experiment_output, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print("\n" + "=" * 60)

    return experiment_output

if __name__ == "__main__":
    run()
