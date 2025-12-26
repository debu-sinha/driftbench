# DriftBench Leaderboard

**Measuring Reliability Half-Life of RAG Systems Under Knowledge Drift**

Last updated: 2025-12-26

## Main Results (77 tasks)

| Rank | Retriever | V1 Accuracy | V2 Accuracy | Delta | SFR (V1) | SFR (V2) |
|------|-----------|-------------|-------------|-------|----------|----------|
| 1 | Oracle-Doc | 80.5% | **87.0%** | +6.5% | 15.6% | 7.8% |
| 2 | Dense RAG | 80.5% | 85.7% | +5.2% | 14.3% | 10.4% |
| 3 | Term RAG | 64.9% | 70.1% | +5.2% | 11.7% | 11.7% |

## Key Finding

**Drift is not uniformly harmful**: Accuracy *improves* under drift (V1→V2) as updated documentation clarifies ambiguities. However, **Silent Failure Rate persists at ~12%** regardless of accuracy direction.

## Drift Type Breakdown

| Drift Type | Count | Description |
|------------|-------|-------------|
| behavior_changed | 28 | Function behavior modified |
| param_renamed | 18 | Parameters or methods renamed |
| import_changed | 16 | Import paths restructured |
| default_changed | 15 | Default values changed |

## How to Submit

To add your retriever to the leaderboard:

1. Run the benchmark:
```bash
python code/run_full_experiment.py --retriever your-retriever
```

2. Open an issue with your results JSON

## Metrics

- **Accuracy**: Fraction of correct answers
- **SFR**: Silent Failure Rate (wrong answer with confidence >= 0.7)
- **Delta**: Accuracy change from V1 to V2 corpus
