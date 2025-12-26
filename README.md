# DRIFTBENCH: Measuring Reliability Half-Life of RAG Systems Under Knowledge Drift

**Do RAG Systems Fail Silently When Documentation Goes Stale?**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Author:** Debu Sinha

## Overview

DRIFTBENCH is the first benchmark treating knowledge drift as a first-class experimental variable. Through **77 organically-derived drift tasks** from real version changes in FastAPI, Pydantic, and LangChain, we uncover a surprising finding: **drift effects are heterogeneous**.

**Key Insight**: Accuracy alone is insufficient for monitoring RAG reliability. Average accuracy can *improve* under drift (64.9% -> 70.1%), but **Silent Failure Rate persists at ~12%** regardless of accuracy direction.

## Key Results

| Retriever | V1 Accuracy | V2 Accuracy | Silent Failure Rate |
|-----------|-------------|-------------|---------------------|
| Term-based RAG | 64.9% | 70.1% | 11.7% |
| Dense RAG | 80.5% | 85.7% | 10.4-14.3% |
| Oracle-Doc | 80.5% | 87.0% | 7.8-15.6% |

**Key Findings**:
1. **Drift is not uniformly harmful**: Updated documentation can clarify ambiguities
2. **Silent failures persist**: 12% SFR regardless of accuracy improvement
3. **Three drift regimes**: Corrective, breaking, and masking drift have distinct safety implications

## Drift Taxonomy

| Drift Type | Count | Description |
|------------|-------|-------------|
| default_changed | 15 | Default parameter values changed |
| behavior_changed | 28 | Function/method behavior changed |
| param_renamed | 18 | Parameters or methods renamed |
| import_changed | 16 | Import paths restructured |

## Project Structure

```
driftbench/
├── code/
│   ├── driftbench_core.py          # Core benchmark logic
│   ├── run_full_experiment.py      # Main experiment runner
│   ├── fastapi_diff_miner.py       # FastAPI drift task mining
│   ├── langchain_diff_miner.py     # LangChain drift task mining
│   └── regenerate_v4_plots.py      # Visualization generation
├── paper/
│   ├── driftbench.tex              # LaTeX source
│   ├── driftbench_final.pdf        # Final paper
│   └── references.bib              # Bibliography
├── figures/
│   ├── drift_decay_curves.png      # Accuracy decay over time
│   ├── llm_experiment_plots.png    # LLM comparison
│   └── sfr_experiment_plot.png     # Silent failure analysis
├── data/
│   ├── driftbench_combined.json    # 77 drift tasks
│   ├── combined_corpus_v1.json     # V1 documentation corpus
│   └── combined_corpus_v2.json     # V2 documentation corpus
├── results/
│   ├── full_experiment_results.json # Main experiment data (84KB)
│   └── llm_drift_sweep_results.json # Drift decay data
├── CHANGELOG.md                     # Version history
└── README.md
```

## Installation

```bash
git clone https://github.com/debu-sinha/driftbench.git
cd driftbench
pip install openai sentence-transformers scikit-learn matplotlib
```

## Running Experiments

```bash
# Set API key
export OPENAI_API_KEY=your_key

# Run full experiment
python code/run_full_experiment.py

# Generate plots
python code/regenerate_v4_plots.py
```

## Metrics

- **Accuracy**: Fraction of correct answers (standard RAG metric)
- **Silent Failure Rate (SFR)**: Fraction of wrong answers with confidence >= 0.7
- **Reliability Half-Life**: Time until accuracy drops below 50% (for decay analysis)

## Data Sources

Tasks mined from organic breaking changes:
- **FastAPI/Pydantic** (41 tasks): v1->v2 migration (orm_mode, @validator, .dict())
- **LangChain** (26 tasks): Package restructuring (v0.0->v0.2), LCEL adoption
- **Tool APIs** (10 tasks): Abstract API versioning scenarios

## Citation

```bibtex
@article{sinha2025driftbench,
  title={DRIFTBENCH: Measuring Reliability Half-Life of RAG Systems Under Knowledge Drift},
  author={Sinha, Debu},
  year={2025}
}
```

## License

MIT License
