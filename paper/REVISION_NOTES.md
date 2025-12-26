# DRIFTBENCH Revision Notes

## Review History

### Review 1: EB-1A Assessment (2025-12-25)
**Verdict**: Top 5-10% of EB-1A-useful research

**Positives**:
- Reliability Half-Life (d½) is a named primitive USCIS understands
- Organic drift from real libraries is credible
- Oracle-Doc diagnostic is reviewer-proof

**Issues Identified**:
1. ❌ Framed too narrowly as "RAG systems" → Should include agents, tools, planners
2. ❌ SFR = 0% underutilizes the metric → Need non-zero SFR experiment
3. ❌ Related Work too polite → Need sharper differentiation

**Fixes Applied**:
- [x] Broadened abstract to "systems that rely on external knowledge"
- [x] Added no-hedging experiment → SFR = 90%
- [x] Fixed SFR metric calculation bug
- [x] Updated author to Debu Sinha, Independent Researcher

---

### Review 2: Area Chair Review (2025-12-25)
**Verdict**: Weak Accept → Borderline Accept

**Strengths (S1-S5)**:
- Novel evaluation axis (drift as continuous variable)
- Realistic task construction from real libraries
- Clean Oracle-Doc diagnostic
- Intuitive d½ metric design
- Important SFR insight

**Weaknesses (W1-W4)**:
1. ❌ W1: Scope too narrow - only one retriever, one model
2. ❌ W2: Scale borderline - 77 tasks light for flagship benchmark
3. ❌ W3: SFR experiment feels artificial - needs framing as stress test
4. ❌ W4: Related work positioning too polite

**Upgrade Path** (any 2 for Strong Accept):
1. Add dense retriever baseline
2. Evaluate additional model family
3. Expand to non-Python domain
4. Clarify SFR as deployment-stress metric
5. Sharpen Related Work

**Fixes Applied**:
- [x] Added dense retriever experiment (all-MiniLM-L6-v2)
  - Result: V1=33.3%, V2=20.0%, SFR=60-73%
  - Dense actually worse than term overlap on this corpus (33% vs 93%)
  - But shows SFR persists across retrieval methods
  - Results saved to: results/dense_retriever_results.json
- [x] Framed SFR as "deployment stress test" in paper
  - Added explanation of production systems suppressing hedging
  - Connected dense retrieval's high baseline SFR to stress test narrative
- [x] Sharpened Related Work section
  - Explicit differentiation from CRUD-RAG (synthetic vs organic drift)
  - Clarified agent benchmarks assume static schemas
  - Distinguished from temporal knowledge (world events vs documentation)
  - Positioned SFR as calibration under distribution shift

---

## Experiment Results Summary

| Experiment | V1 Acc | V2 Acc | SFR V1 | SFR V2 | Notes |
|------------|--------|--------|--------|--------|-------|
| Vanilla RAG (term overlap) | 93.3% | 80.0% | 0% | 0% | Default hedging |
| Oracle-Doc | 93.3% | 100% | 0% | 0% | Gold retrieval |
| No-Hedging RAG | 10.0% | 10.0% | 90% | 90% | Forced confidence |
| Dense Retriever | 33.3% | 20.0% | 60.0% | 73.3% | MiniLM-L6-v2 |

**Key Findings**:
- 13.3% accuracy drop under drift (V1→V2)
- 20% Oracle gap proves retrieval bottleneck
- SFR reaches 90% when hedging suppressed
- Dense retrieval shows even higher SFR (60-73%)

---

## Paper Versions

| Version | Date | Changes |
|---------|------|---------|
| v1 | 2025-12-25 | Initial draft, 6 pages |
| v2 | 2025-12-25 | Added SFR=90% result, broadened framing, fixed author |
| v3 | 2025-12-25 | Added dense retriever results, SFR as stress test, sharpened Related Work |
| v4/final | 2025-12-25 | **MAJOR REVISION**: Full 77-task results, reframed thesis, style guide compliance |

---

## Review 3: Full-Scale Validation (2025-12-25)

**Finding**: 15-task subset was NOT representative of full 77-task dataset.

### Results Comparison

| Metric | 15 Tasks (old) | 77 Tasks (new) | Change |
|--------|----------------|----------------|--------|
| Term V1 Acc | 93.3% | 64.9% [53-74] | Different |
| Term V2 Acc | 80.0% | 70.1% [60-79] | Different |
| **Drift Effect** | -13.3% drop | +5.2% improvement | **REVERSED** |
| Dense V1 | 33.3% | 80.5% [71-88] | Different |
| Dense V2 | 20.0% | 85.7% [77-94] | Different |
| Oracle V2 | 100% | 87.0% [79-94] | Different |
| SFR | 0% | ~12% [5-19] | Non-zero |

### New Thesis (v4)

> "Drift effects are heterogeneous: accuracy can IMPROVE while SFR persists. Accuracy alone is insufficient for monitoring RAG reliability."

### Changes Made

1. **Abstract**: Rewrote to reflect heterogeneous drift effects
2. **Contributions**: 5 new contributions based on actual findings
3. **Results Table**: N=77 with 95% bootstrap CI
4. **Key Finding**: "Accuracy improves, SFR persists" (was "accuracy drops")
5. **Analysis**: New sections explaining why accuracy improves, why SFR persists
6. **Limitations**: Proper section (was paragraph)
7. **Broader Impact**: Proper section (was paragraph)
8. **Figures**: Regenerated for new data
9. **Style Guide**: Applied NeurIPS linter rules (U-1 through U-9)

### Files Updated
- `paper/driftbench.tex` - Complete rewrite
- `paper/driftbench_final.pdf` - Camera-ready version
- `paper/fig_v1v2.pdf` - New figures
- `results/full_experiment_results.json` - 77-task results with CI
- `code/run_full_experiment.py` - Full experiment script
- `code/regenerate_v4_plots.py` - Plot generation

---

## Target Venues

| Venue | Track | Deadline | Fit |
|-------|-------|----------|-----|
| NeurIPS 2026 | Datasets & Benchmarks | May 2026 | Primary |
| TMLR | Rolling | Anytime | Secondary |
| ICML 2026 | Main | Jan 2026 | Tertiary |

---

## Final Submission (2025-12-25)

**AC Verdict: 8/10 - Clear Accept**

### Submission Files
| File | Description |
|------|-------------|
| `driftbench_submission.pdf` | Camera-ready PDF (7 pages) |
| `driftbench.tex` | LaTeX source |
| `references.bib` | Bibliography (13 citations) |
| `fig_v1v2.pdf` | Main results figure |

### Final Fixes Applied
- [x] lmodern + microtype + textcomp (clean fonts, no encoding artifacts)
- [x] MiniLM qualifier added to dense-retrieval claim
- [x] Endpoint comparison clarification (already present)
- [x] Figure reference added
- [x] En-dash usage fixed

### Status: READY TO SUBMIT
