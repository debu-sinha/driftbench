---
license: mit
task_categories:
- question-answering
- text-generation
language:
- en
tags:
- rag
- knowledge-drift
- retrieval-augmented-generation
- distribution-shift
- ai-safety
- benchmark
- nlp
size_categories:
- n<1K
pretty_name: DRIFTBENCH
---

# DRIFTBENCH: Measuring Reliability Half-Life of RAG Systems Under Knowledge Drift

**The first benchmark treating knowledge drift as a first-class experimental variable.**

## The Knowledge Drift Problem

```
Time T0                              Time T1
┌──────────────┐                    ┌──────────────┐
│   Docs V1    │    Knowledge      │   Docs V2    │
│      |       │      Drift        │      |       │
│      v       │   ==========>     │      v       │
│  RAG Index   │                   │  Old Index   │ (outdated!)
│      |       │                   │      |       │
│      v       │                   │      v       │
│   Correct    │                   │    Wrong     │
└──────────────┘                    └──────────────┘
```

**The Problem:** Documentation changes, but your RAG index stays stale. Answers become wrong silently.

## Drift Taxonomy

| Type | Description | Effect | Safety |
|------|-------------|--------|--------|
| **Corrective** | V2 clarifies V1 ambiguities | Accuracy up | Improves reliability |
| **Breaking** | V2 invalidates V1 patterns | Silent failures up | Dangerous |
| **Masking** | Accuracy up but SFR persists | Hidden risk | Deceptive |

## Dataset Description

DRIFTBENCH contains **77 organically-derived drift tasks** mined from real version changes in FastAPI, Pydantic, and LangChain documentation. Each task has:

- A question that has different correct answers depending on documentation version
- V1 answer (based on older documentation)
- V2 answer (based on newer documentation)
- Evidence from both versions
- Metadata about the type of change

### Key Finding

> **Drift effects are heterogeneous.** Accuracy can *improve* under drift while Silent Failure Rate persists at 12%—revealing safety risks invisible to aggregate metrics.

## Dataset Structure

```json
{
  "task_id": "fastapi_organic_0000",
  "question": "What is the default value for response_model_exclude_unset in FastAPI?",
  "answer_v1": "False",
  "answer_v2": "True",
  "evidence_v1": "[FastAPI 0.100.0] Response model serialization...",
  "evidence_v2": "[FastAPI 0.109.0] Response model serialization...",
  "category": "factoid",
  "difficulty": "easy",
  "source": "fastapi",
  "source_change": {
    "file_path": "docs/response_model_exclude_unset.md",
    "change_type": "default_changed",
    "old_value": "False",
    "new_value": "True",
    "version_old": "0.100.0",
    "version_new": "0.109.0"
  }
}
```

## Data Sources

| Source | Tasks | Examples |
|--------|-------|----------|
| FastAPI/Pydantic | 41 | `orm_mode` → `from_attributes`, `.dict()` → `.model_dump()` |
| LangChain | 26 | Package restructuring, `.run()` → `.invoke()` |
| Tool APIs | 10 | Parameter renames, unit changes |

## Drift Taxonomy

| Regime | Description | Safety Implication |
|--------|-------------|-------------------|
| **Corrective** | V2 clarifies V1 ambiguities | Improves reliability |
| **Breaking** | V2 invalidates V1 patterns | Causes silent failures |
| **Masking** | Accuracy improves but SFR persists | Hidden safety risk |

## Evaluation Metrics

**Four key metrics for RAG reliability under drift:**

```
+----------------------+-------------------------------------+
| Success Rate         | Standard accuracy                   |
+----------------------+-------------------------------------+
| Silent Failure Rate  | Wrong + Confident (hidden danger)   |
+----------------------+-------------------------------------+
| Reliability          | Time until 50% accuracy drop        |
| Half-Life            |                                     |
+----------------------+-------------------------------------+
| Oracle Gap           | Retrieval vs reasoning failures     |
+----------------------+-------------------------------------+
```

| Metric | Definition |
|--------|------------|
| **Success Rate** | P(correct answer given docs) |
| **Silent Failure Rate** | P(wrong ∧ confident) — confident errors |
| **Reliability Half-Life** | Drift amount before 50% accuracy drop |
| **Oracle Gap** | Oracle - RAG accuracy (retrieval vs reasoning) |

## Usage

### Load with Datasets Library

```python
from datasets import load_dataset

dataset = load_dataset("dsinha/driftbench")

# Access tasks
for task in dataset["train"]:
    print(f"Q: {task['question']}")
    print(f"V1: {task['answer_v1']}")
    print(f"V2: {task['answer_v2']}")
```

### Evaluate a RAG System

```python
from datasets import load_dataset

dataset = load_dataset("dsinha/driftbench")

def evaluate_rag(rag_system, corpus_version="v1"):
    correct = 0
    silent_failures = 0

    for task in dataset["train"]:
        # Get RAG answer
        answer, confidence = rag_system.query(task["question"])

        # Check correctness based on corpus version
        expected = task[f"answer_{corpus_version}"]
        is_correct = answer_matches(answer, expected)

        if is_correct:
            correct += 1
        elif confidence > 0.8:  # High confidence but wrong
            silent_failures += 1

    accuracy = correct / len(dataset["train"])
    sfr = silent_failures / len(dataset["train"])

    return {"accuracy": accuracy, "sfr": sfr}
```

## Corpus Files

The dataset includes two documentation corpora:

- **corpus_v1.json**: Older documentation versions
- **corpus_v2.json**: Newer documentation versions

These can be used to build RAG indices for testing drift effects.

## Citation

```bibtex
@article{sinha2025driftbench,
  title={DRIFTBENCH: Measuring Reliability Half-Life of RAG Systems Under Knowledge Drift},
  author={Sinha, Debu},
  journal={arXiv preprint},
  year={2025}
}
```

## Related Work

This dataset is part of a research program on **AI reliability under distribution shift**:

| Paper | Focus | Link |
|-------|-------|------|
| **The Semantic Illusion** | Embedding-based detection fails on RLHF | [arXiv:2512.15068](https://arxiv.org/abs/2512.15068) |
| **ATCB** | Agents don't know when they'll fail | [GitHub](https://github.com/debu-sinha/atcb-benchmark) |
| **ConformalDrift** | Conformal guarantees collapse under shift | [GitHub](https://github.com/debu-sinha/conformaldrift) |
| **DRIFTBENCH** | RAG reliability degrades over time | This dataset |

## License

MIT License

## Author

[Debu Sinha](https://github.com/debu-sinha)
