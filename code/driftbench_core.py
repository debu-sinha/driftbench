"""
DRIFTBENCH Core - Measuring Reliability Half-Life of RAG Agents Under Drift

Core classes and metrics for the DRIFTBENCH evaluation framework.
"""

import json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime


@dataclass
class DriftTask:
    """A single DRIFTBENCH evaluation task."""
    task_id: str
    question: str
    answer_v1: str  # Correct answer under version 1
    answer_v2: str  # Correct answer under version 2
    evidence_v1: str  # Supporting text from v1 docs
    evidence_v2: str  # Supporting text from v2 docs
    category: str  # factoid, multi_hop, tool_use
    drift_type: str  # knowledge, embedding, tool
    source_file: str  # Original doc file
    difficulty: str  # easy, medium, hard


@dataclass
class ModelResponse:
    """Response from a model on a DRIFTBENCH task."""
    task_id: str
    predicted_answer: str
    confidence: float  # 0-1, model's self-reported confidence
    evidence_cited: str  # What evidence the model cited
    expressed_uncertainty: bool  # Did model flag potential staleness?
    latency_ms: float
    raw_response: str


@dataclass
class EvaluationResult:
    """Evaluation result for a single task."""
    task_id: str
    correct: bool
    confidence: float
    silent_failure: bool  # Wrong but confident
    retrieval_success: bool  # Did retrieval find correct doc?
    version_tested: str  # v1 or v2


class DriftBenchMetrics:
    """Compute DRIFTBENCH metrics from evaluation results."""

    def __init__(self, results: List[EvaluationResult]):
        self.results = results
        self.n = len(results)

    def success_rate(self) -> float:
        """Overall success rate S(d)."""
        if self.n == 0:
            return 0.0
        return sum(1 for r in self.results if r.correct) / self.n

    def silent_failure_rate(self, tau: float = 0.8) -> float:
        """
        Silent Failure Rate (SFR): P(wrong AND confident AND no uncertainty flag)

        SFR_tau(d) = P(y_hat != y AND c >= tau AND u = 0)
        """
        if self.n == 0:
            return 0.0
        silent_failures = sum(1 for r in self.results if r.silent_failure)
        return silent_failures / self.n

    def confident_error_rate(self, tau: float = 0.8) -> float:
        """
        Confident Error Rate (CER): P(wrong AND confident)

        CER_tau(d) = P(y_hat != y AND c >= tau)
        """
        if self.n == 0:
            return 0.0
        confident_errors = sum(
            1 for r in self.results
            if not r.correct and r.confidence >= tau
        )
        return confident_errors / self.n

    def retrieval_failure_rate(self) -> float:
        """Rate at which retrieval failed to find correct documents."""
        if self.n == 0:
            return 0.0
        return sum(1 for r in self.results if not r.retrieval_success) / self.n

    def reasoning_failure_rate(self, oracle_results: List[EvaluationResult]) -> float:
        """
        Rate at which reasoning failed despite correct retrieval.
        Computed as: P(Oracle-Doc correct AND Full incorrect)
        """
        if self.n == 0 or len(oracle_results) != self.n:
            return 0.0

        reasoning_failures = sum(
            1 for r, o in zip(self.results, oracle_results)
            if o.correct and not r.correct
        )
        return reasoning_failures / self.n

    def expected_calibration_error(self, n_bins: int = 10) -> float:
        """
        Expected Calibration Error (ECE).

        Measures how well confidence aligns with accuracy.
        """
        if self.n == 0:
            return 0.0

        bins = [[] for _ in range(n_bins)]
        for r in self.results:
            bin_idx = min(int(r.confidence * n_bins), n_bins - 1)
            bins[bin_idx].append(r)

        ece = 0.0
        for i, bin_results in enumerate(bins):
            if len(bin_results) == 0:
                continue
            bin_conf = (i + 0.5) / n_bins
            bin_acc = sum(1 for r in bin_results if r.correct) / len(bin_results)
            ece += len(bin_results) / self.n * abs(bin_acc - bin_conf)

        return ece

    def reliability_diagram_data(self, n_bins: int = 10) -> Dict[str, List[float]]:
        """Get data for reliability diagram."""
        bins = [[] for _ in range(n_bins)]
        for r in self.results:
            bin_idx = min(int(r.confidence * n_bins), n_bins - 1)
            bins[bin_idx].append(r)

        confidences = []
        accuracies = []
        counts = []

        for i, bin_results in enumerate(bins):
            conf = (i + 0.5) / n_bins
            if len(bin_results) > 0:
                acc = sum(1 for r in bin_results if r.correct) / len(bin_results)
            else:
                acc = 0.0
            confidences.append(conf)
            accuracies.append(acc)
            counts.append(len(bin_results))

        return {
            "confidences": confidences,
            "accuracies": accuracies,
            "counts": counts
        }


def compute_reliability_half_life(
    drift_doses: List[float],
    success_rates: List[float]
) -> Optional[float]:
    """
    Compute Reliability Half-Life (RHL).

    d_1/2 = inf{d : S(d) <= 0.5 * S(0)}

    Args:
        drift_doses: List of drift dose values (e.g., [0, 0.01, 0.02, 0.05, 0.1])
        success_rates: Corresponding success rates at each dose

    Returns:
        Half-life drift dose, or None if never drops below half
    """
    if len(drift_doses) != len(success_rates) or len(drift_doses) == 0:
        return None

    s0 = success_rates[0]
    threshold = 0.5 * s0

    for d, s in zip(drift_doses, success_rates):
        if s <= threshold:
            return d

    # Extrapolate if not reached
    # Fit exponential decay: S(d) = S(0) * exp(-lambda * d)
    if len(drift_doses) >= 2 and success_rates[-1] < s0:
        # Simple linear interpolation to find crossing
        for i in range(1, len(drift_doses)):
            if success_rates[i] <= threshold:
                # Linear interpolation between i-1 and i
                d_prev, s_prev = drift_doses[i-1], success_rates[i-1]
                d_curr, s_curr = drift_doses[i], success_rates[i]

                if s_prev != s_curr:
                    d_half = d_prev + (threshold - s_prev) * (d_curr - d_prev) / (s_curr - s_prev)
                    return d_half

    return None  # Never drops below half


def compute_plan_divergence(
    actions_v1: List[str],
    actions_v2: List[str]
) -> float:
    """
    Compute Plan Divergence (PD) for tool-use tasks.

    PD = EditDist(a, a') / max(|a|, |a'|)

    Args:
        actions_v1: Action sequence under v1
        actions_v2: Action sequence under v2

    Returns:
        Normalized edit distance (0-1)
    """
    m, n = len(actions_v1), len(actions_v2)
    if m == 0 and n == 0:
        return 0.0
    if m == 0 or n == 0:
        return 1.0

    # Dynamic programming for Levenshtein distance
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if actions_v1[i-1] == actions_v2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    return dp[m][n] / max(m, n)


@dataclass
class DriftBenchReport:
    """Complete evaluation report for a system under drift."""
    system_name: str
    drift_type: str
    drift_doses: List[float]
    success_rates: List[float]
    sfr_rates: List[float]
    cer_rates: List[float]
    ece_values: List[float]
    retrieval_failure_rates: List[float]
    reasoning_failure_rates: List[float]
    half_life: Optional[float]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return {
            "system_name": self.system_name,
            "drift_type": self.drift_type,
            "drift_doses": self.drift_doses,
            "success_rates": self.success_rates,
            "sfr_rates": self.sfr_rates,
            "cer_rates": self.cer_rates,
            "ece_values": self.ece_values,
            "retrieval_failure_rates": self.retrieval_failure_rates,
            "reasoning_failure_rates": self.reasoning_failure_rates,
            "half_life": self.half_life,
            "timestamp": self.timestamp
        }

    def save(self, path: Path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


if __name__ == "__main__":
    # Quick test
    print("DRIFTBENCH Core loaded successfully")

    # Test half-life computation
    doses = [0, 0.01, 0.02, 0.05, 0.1]
    rates = [0.9, 0.85, 0.7, 0.5, 0.3]
    hl = compute_reliability_half_life(doses, rates)
    print(f"Test half-life: {hl}")

    # Test plan divergence
    a1 = ["search", "filter", "sort"]
    a2 = ["search", "sort"]
    pd = compute_plan_divergence(a1, a2)
    print(f"Test plan divergence: {pd}")
