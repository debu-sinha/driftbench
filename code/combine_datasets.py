"""
Combine all DRIFTBENCH datasets into a unified benchmark.
"""

import json
from pathlib import Path
from datetime import datetime


def combine_datasets():
    """Combine FastAPI, LangChain, and Tool drift datasets."""
    data_dir = Path(__file__).parent.parent / "data"

    all_tasks = []
    all_corpus_v1 = {}
    all_corpus_v2 = {}

    # Load FastAPI tasks
    with open(data_dir / "fastapi_drift_tasks.json", 'r') as f:
        fastapi_data = json.load(f)
    fastapi_tasks = fastapi_data["tasks"]
    print(f"FastAPI: {len(fastapi_tasks)} tasks")

    # Load FastAPI corpus
    with open(data_dir / "corpus_v1.json", 'r') as f:
        fastapi_v1 = json.load(f)
    with open(data_dir / "corpus_v2.json", 'r') as f:
        fastapi_v2 = json.load(f)

    for task in fastapi_tasks:
        task["source"] = "fastapi"
        all_tasks.append(task)

    all_corpus_v1.update({f"fastapi_{k}": v for k, v in fastapi_v1.items()})
    all_corpus_v2.update({f"fastapi_{k}": v for k, v in fastapi_v2.items()})

    # Load LangChain tasks
    with open(data_dir / "langchain_drift_tasks.json", 'r') as f:
        langchain_data = json.load(f)
    langchain_tasks = langchain_data["tasks"]
    print(f"LangChain: {len(langchain_tasks)} tasks")

    with open(data_dir / "langchain_corpus_v1.json", 'r') as f:
        langchain_v1 = json.load(f)
    with open(data_dir / "langchain_corpus_v2.json", 'r') as f:
        langchain_v2 = json.load(f)

    for task in langchain_tasks:
        task["source"] = "langchain"
        all_tasks.append(task)

    all_corpus_v1.update({f"langchain_{k}": v for k, v in langchain_v1.items()})
    all_corpus_v2.update({f"langchain_{k}": v for k, v in langchain_v2.items()})

    # Load Tool drift tasks
    with open(data_dir / "tool_drift_tasks.json", 'r') as f:
        tool_data = json.load(f)
    tool_tasks = tool_data["tasks"]
    print(f"Tool Drift: {len(tool_tasks)} tasks")

    for task in tool_tasks:
        task["source"] = "tool_api"
        # Adapt to common format
        task["answer_v1"] = json.dumps(task.get("correct_call_v1", {}))
        task["answer_v2"] = json.dumps(task.get("correct_call_v2", {}))
        task["evidence_v1"] = task.get("context_v1", "")
        task["evidence_v2"] = task.get("context_v2", "")
        all_tasks.append(task)

        # Add to corpus
        doc_id = task["task_id"]
        all_corpus_v1[f"tool_{doc_id}"] = task["context_v1"]
        all_corpus_v2[f"tool_{doc_id}"] = task["context_v2"]

    # Create combined dataset
    combined = {
        "name": "DRIFTBENCH-Combined",
        "version": "0.1",
        "created": datetime.now().isoformat(),
        "n_tasks": len(all_tasks),
        "sources": {
            "fastapi": len(fastapi_tasks),
            "langchain": len(langchain_tasks),
            "tool_api": len(tool_tasks)
        },
        "tasks": all_tasks
    }

    # Save combined dataset
    with open(data_dir / "driftbench_combined.json", 'w') as f:
        json.dump(combined, f, indent=2)
    print(f"\nCombined: {len(all_tasks)} total tasks")

    # Save combined corpora
    with open(data_dir / "combined_corpus_v1.json", 'w') as f:
        json.dump(all_corpus_v1, f, indent=2)
    with open(data_dir / "combined_corpus_v2.json", 'w') as f:
        json.dump(all_corpus_v2, f, indent=2)

    print(f"Combined corpus v1: {len(all_corpus_v1)} docs")
    print(f"Combined corpus v2: {len(all_corpus_v2)} docs")

    return combined


def main():
    print("=" * 60)
    print("  DRIFTBENCH - Combining All Datasets")
    print("=" * 60)
    print()

    combined = combine_datasets()

    print()
    print("=" * 60)
    print("  FINAL DRIFTBENCH SUMMARY")
    print("=" * 60)
    print(f"  Total tasks: {combined['n_tasks']}")
    print(f"  Sources:")
    for source, count in combined["sources"].items():
        print(f"    - {source}: {count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
