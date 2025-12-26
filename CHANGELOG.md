# DriftBench Changelog

## v1.0 (2025-12-25)
- 77 organically-derived drift tasks from real software version changes
- Data sources: FastAPI (41), LangChain (26), Tool APIs (10)
- Paired v1/v2 corpora with evidence
- Key metrics: Accuracy, Silent Failure Rate (SFR), Reliability Half-Life
- Evaluation with Term-based RAG, Dense RAG, and Oracle-Doc baselines
- Key finding: Accuracy improves under drift (64.9% -> 70.1%) but SFR persists at 12%

## Final Paper
- `paper/driftbench_final.pdf` - Current version
