# DRIFTBENCH

**Measuring Reliability Half-Life of RAG Systems Under Knowledge Drift**

[![arXiv](https://img.shields.io/badge/arXiv-2512.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2512.XXXXX)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Systems that rely on external knowledge—RAG pipelines, tool-using agents, and cached memory systems—face an unmeasured vulnerability: **knowledge drift**, the divergence between indexed documentation and current ground truth.

DRIFTBENCH is the first benchmark treating knowledge drift as a first-class experimental variable. Through **77 organically-derived drift tasks** from real version changes in FastAPI, Pydantic, and LangChain, we uncover a surprising finding:

> **Drift effects are heterogeneous.** Accuracy can *improve* under drift while Silent Failure Rate persists—revealing safety risks invisible to aggregate metrics.

## Key Findings

| Metric | V1 (Baseline) | V2 (Drifted) | Insight |
|--------|---------------|--------------|---------|
| Term-based Accuracy | 64.9% | 70.1% (+5.2%) | Drift can *improve* accuracy |
| Dense RAG Accuracy | 80.5% | 85.7% (+5.2%) | Semantic retrieval benefits from clearer docs |
| Silent Failure Rate | 11.7% | 11.7% | **SFR persists regardless of accuracy** |
| Oracle Gap | — | 13% | Reasoning failures even with perfect retrieval |

**Core insight**: Accuracy alone is insufficient for monitoring RAG reliability under drift.

## Drift Taxonomy

We identify three drift regimes with distinct safety implications:

| Regime | Description | Safety Implication |
|--------|-------------|-------------------|
| **Corrective** | V2 clarifies V1 ambiguities | Improves reliability |
| **Breaking** | V2 invalidates V1 patterns | Causes silent failures |
| **Masking** | Accuracy improves but SFR persists | Hidden safety risk |

## Installation

```bash
git clone https://github.com/debu-sinha/driftbench.git
cd driftbench
pip install -r requirements.txt
```

## Quick Start

```bash
# Set API key
export OPENAI_API_KEY=your_key

# Run full experiment
python code/run_full_experiment.py

# Generate visualizations
python code/regenerate_v4_plots.py
```

## Metrics

| Metric | Definition |
|--------|------------|
| **Success Rate** | $S(d) = P(\hat{y} = y \mid d)$ — standard accuracy |
| **Silent Failure Rate** | $\text{SFR}_\tau = P(\hat{y} \neq y \land c \geq \tau)$ — confident errors |
| **Reliability Half-Life** | $d_{1/2} = \inf\{d : S(d) \leq 0.5 \cdot S(0)\}$ — drift robustness |
| **Oracle Gap** | $S_{\text{Oracle}} - S_{\text{RAG}}$ — retrieval vs reasoning failures |

## Data Sources

77 tasks mined from organic breaking changes:

| Source | Tasks | Examples |
|--------|-------|----------|
| FastAPI/Pydantic | 41 | `orm_mode` → `from_attributes`, `.dict()` → `.model_dump()` |
| LangChain | 26 | Package restructuring, `.run()` → `.invoke()` |
| Tool APIs | 10 | Parameter renames, unit changes |

## Repository Structure

```
driftbench/
├── code/
│   ├── driftbench_core.py          # Core benchmark logic
│   ├── run_full_experiment.py      # Main experiment runner
│   └── regenerate_v4_plots.py      # Visualization generation
├── data/
│   ├── driftbench_combined.json    # 77 drift tasks
│   ├── combined_corpus_v1.json     # V1 documentation
│   └── combined_corpus_v2.json     # V2 documentation
├── results/
│   └── full_experiment_results.json
└── figures/
```

## Related Work

This work is part of a research program on **AI reliability under distribution shift**:

| Paper | Focus | Link |
|-------|-------|------|
| **The Semantic Illusion** | Embedding-based detection fails on RLHF hallucinations | [arXiv:2512.15068](https://arxiv.org/abs/2512.15068) |
| **ATCB** | Agents don't know when they'll fail (calibration gap) | [GitHub](https://github.com/debu-sinha/atcb-benchmark) |
| **ConformalDrift** | Conformal guarantees collapse under shift | [GitHub](https://github.com/debu-sinha/conformaldrift) |
| **DRIFTBENCH** | RAG reliability degrades over time | This repo |

## Citation

```bibtex
@article{sinha2025driftbench,
  title={DRIFTBENCH: Measuring Reliability Half-Life of RAG Systems Under Knowledge Drift},
  author={Sinha, Debu},
  journal={arXiv preprint},
  year={2025}
}
```

## License

MIT License
