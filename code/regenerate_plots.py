"""
Regenerate all DRIFTBENCH plots with improved visualization.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')

results_dir = Path(__file__).parent.parent / "results"


def plot_validation_results():
    """Regenerate validation plots with all 4 metrics."""
    with open(results_dir / "validation_results.json", 'r') as f:
        data = json.load(f)

    results = data["results"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    systems = ["Vanilla RAG", "Iterative RAG", "Oracle-Doc"]
    x = np.arange(len(systems))
    width = 0.35

    # Plot 1: Success Rate
    ax = axes[0, 0]
    v1_acc = [results["vanilla_rag_v1"]["success_rate"] * 100,
              results["iterative_rag_v1"]["success_rate"] * 100,
              results["oracle_doc_v1"]["success_rate"] * 100]
    v2_acc = [results["vanilla_rag_v2"]["success_rate"] * 100,
              results["iterative_rag_v2"]["success_rate"] * 100,
              results["oracle_doc_v2"]["success_rate"] * 100]

    bars1 = ax.bar(x - width/2, v1_acc, width, label='V1 (baseline)', color='#3498db', edgecolor='black')
    bars2 = ax.bar(x + width/2, v2_acc, width, label='V2 (drifted)', color='#e74c3c', edgecolor='black')

    ax.set_ylabel('Success Rate (%)', fontsize=11)
    ax.set_title('Accuracy Under Version Drift', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(systems)
    ax.legend()
    ax.set_ylim(0, 100)

    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    # Plot 2: Confident Error Rate (CER)
    ax = axes[0, 1]
    v1_cer = [results["vanilla_rag_v1"]["cer"] * 100,
              results["iterative_rag_v1"]["cer"] * 100,
              results["oracle_doc_v1"]["cer"] * 100]
    v2_cer = [results["vanilla_rag_v2"]["cer"] * 100,
              results["iterative_rag_v2"]["cer"] * 100,
              results["oracle_doc_v2"]["cer"] * 100]

    bars1 = ax.bar(x - width/2, v1_cer, width, label='V1', color='#3498db', edgecolor='black')
    bars2 = ax.bar(x + width/2, v2_cer, width, label='V2', color='#e74c3c', edgecolor='black')

    ax.set_ylabel('Confident Error Rate (%)', fontsize=11)
    ax.set_title('CER: P(wrong AND confident >= 0.8)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(systems)
    ax.legend()
    ax.set_ylim(0, max(max(v1_cer), max(v2_cer)) * 1.2 + 5)

    for bar in bars1 + bars2:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    # Plot 3: ECE (Calibration Error)
    ax = axes[1, 0]
    v1_ece = [results["vanilla_rag_v1"]["ece"],
              results["iterative_rag_v1"]["ece"],
              results["oracle_doc_v1"]["ece"]]
    v2_ece = [results["vanilla_rag_v2"]["ece"],
              results["iterative_rag_v2"]["ece"],
              results["oracle_doc_v2"]["ece"]]

    bars1 = ax.bar(x - width/2, v1_ece, width, label='V1', color='#3498db', edgecolor='black')
    bars2 = ax.bar(x + width/2, v2_ece, width, label='V2', color='#e74c3c', edgecolor='black')

    ax.set_ylabel('Expected Calibration Error', fontsize=11)
    ax.set_title('Calibration (lower = better)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(systems)
    ax.legend()
    ax.set_ylim(0, 0.5)

    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    # Plot 4: Oracle Gap (Retrieval vs Reasoning)
    ax = axes[1, 1]

    # Oracle gap = Oracle accuracy - RAG accuracy
    gap_v1_vanilla = (results["oracle_doc_v1"]["success_rate"] - results["vanilla_rag_v1"]["success_rate"]) * 100
    gap_v2_vanilla = (results["oracle_doc_v2"]["success_rate"] - results["vanilla_rag_v2"]["success_rate"]) * 100
    gap_v1_iter = (results["oracle_doc_v1"]["success_rate"] - results["iterative_rag_v1"]["success_rate"]) * 100
    gap_v2_iter = (results["oracle_doc_v2"]["success_rate"] - results["iterative_rag_v2"]["success_rate"]) * 100

    gap_systems = ["Vanilla RAG", "Iterative RAG"]
    gap_x = np.arange(len(gap_systems))

    v1_gaps = [gap_v1_vanilla, gap_v1_iter]
    v2_gaps = [gap_v2_vanilla, gap_v2_iter]

    bars1 = ax.bar(gap_x - width/2, v1_gaps, width, label='V1 Gap', color='#9b59b6', edgecolor='black')
    bars2 = ax.bar(gap_x + width/2, v2_gaps, width, label='V2 Gap', color='#f39c12', edgecolor='black')

    ax.set_ylabel('Oracle Gap (%)', fontsize=11)
    ax.set_title('Retrieval Bottleneck (Oracle - RAG)', fontsize=12, fontweight='bold')
    ax.set_xticks(gap_x)
    ax.set_xticklabels(gap_systems)
    ax.legend()
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(results_dir / "validation_plots.png", dpi=150, bbox_inches='tight')
    plt.savefig(results_dir / "validation_plots.pdf", bbox_inches='tight')
    print("Saved validation_plots.png/pdf")
    plt.close()


def plot_drift_sweep():
    """Regenerate drift sweep plots."""
    with open(results_dir / "drift_sweep_results.json", 'r') as f:
        data = json.load(f)

    sweep = data["sweep_results"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Success Rate Decay
    ax = axes[0, 0]
    for system, color, marker in [("vanilla_rag", "#e74c3c", "o"), ("iterative_rag", "#3498db", "s")]:
        doses = [d * 100 for d in sweep[system]["doses"]]
        rates = [r * 100 for r in sweep[system]["success_rates"]]
        ax.plot(doses, rates, f'{marker}-', color=color, label=system.replace("_", " ").title(),
               linewidth=2, markersize=8, markeredgecolor='black')

    ax.set_xlabel('Drift Dose (%)', fontsize=11)
    ax.set_ylabel('Success Rate (%)', fontsize=11)
    ax.set_title('Accuracy Decay Under Drift (MockGenerator)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.set_xlim(-5, 105)
    ax.set_ylim(0, 100)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% threshold')

    # Plot 2: Confident Error Rate
    ax = axes[0, 1]
    for system, color, marker in [("vanilla_rag", "#e74c3c", "o"), ("iterative_rag", "#3498db", "s")]:
        doses = [d * 100 for d in sweep[system]["doses"]]
        cer = [r * 100 for r in sweep[system]["cer"]]
        ax.plot(doses, cer, f'{marker}-', color=color, label=system.replace("_", " ").title(),
               linewidth=2, markersize=8, markeredgecolor='black')

    ax.set_xlabel('Drift Dose (%)', fontsize=11)
    ax.set_ylabel('Confident Error Rate (%)', fontsize=11)
    ax.set_title('CER vs Drift Dose', fontsize=12, fontweight='bold')
    ax.legend()
    ax.set_xlim(-5, 105)

    # Plot 3: ECE
    ax = axes[1, 0]
    for system, color, marker in [("vanilla_rag", "#e74c3c", "o"), ("iterative_rag", "#3498db", "s")]:
        doses = [d * 100 for d in sweep[system]["doses"]]
        ece = sweep[system]["ece"]
        ax.plot(doses, ece, f'{marker}-', color=color, label=system.replace("_", " ").title(),
               linewidth=2, markersize=8, markeredgecolor='black')

    ax.set_xlabel('Drift Dose (%)', fontsize=11)
    ax.set_ylabel('Expected Calibration Error', fontsize=11)
    ax.set_title('Calibration vs Drift Dose', fontsize=12, fontweight='bold')
    ax.legend()
    ax.set_xlim(-5, 105)
    ax.set_ylim(0, 0.5)

    # Plot 4: Summary annotation
    ax = axes[1, 1]
    ax.axis('off')

    # Summary text
    summary = """
    DRIFT SWEEP SUMMARY (MockGenerator)
    ════════════════════════════════════

    Tasks: 41
    Drift doses: 0%, 1%, 2%, 5%, 10%, 20%, 50%, 100%

    KEY FINDINGS:

    • Accuracy is flat from 0-20% drift (53.7%)
    • Jumps to 63-66% at 50-100% drift
      (MockGenerator artifact - v2 has cleaner patterns)

    • CER decreases with drift (46% → 34%)
      (More correct answers → fewer confident errors)

    • ECE improves with drift (0.31 → 0.19)
      (Better calibration as accuracy improves)

    • SFR = 0% at all doses
      (MockGenerator appropriately flags uncertainty)

    NOTE: Real LLM results show opposite pattern
    (accuracy DROPS with drift - see llm_decay_curve.png)
    """
    ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(results_dir / "drift_decay_curves.png", dpi=150, bbox_inches='tight')
    plt.savefig(results_dir / "drift_decay_curves.pdf", bbox_inches='tight')
    print("Saved drift_decay_curves.png/pdf")
    plt.close()


def plot_llm_results():
    """Regenerate LLM experiment plots."""
    with open(results_dir / "llm_experiment_results.json", 'r') as f:
        data = json.load(f)

    results = data["results"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    systems = ["Vanilla RAG", "Oracle-Doc"]
    x = np.arange(len(systems))
    width = 0.35

    # Plot 1: Success Rate
    ax = axes[0]
    v1_acc = [results["vanilla_rag_v1"]["success_rate"] * 100,
              results["oracle_doc_v1"]["success_rate"] * 100]
    v2_acc = [results["vanilla_rag_v2"]["success_rate"] * 100,
              results["oracle_doc_v2"]["success_rate"] * 100]

    bars1 = ax.bar(x - width/2, v1_acc, width, label='V1 (baseline)', color='#3498db', edgecolor='black')
    bars2 = ax.bar(x + width/2, v2_acc, width, label='V2 (drifted)', color='#e74c3c', edgecolor='black')

    ax.set_ylabel('Success Rate (%)', fontsize=11)
    ax.set_title('Accuracy: GPT-4o-mini', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(systems)
    ax.legend()
    ax.set_ylim(0, 110)

    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add arrow showing drop
    ax.annotate('', xy=(0.175, v2_acc[0]), xytext=(0.175, v1_acc[0]),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text(0.3, (v1_acc[0] + v2_acc[0])/2, f'-13.3%', fontsize=10, fontweight='bold', color='red')

    # Plot 2: CER comparison
    ax = axes[1]
    v1_cer = [results["vanilla_rag_v1"]["cer"] * 100,
              results["oracle_doc_v1"]["cer"] * 100]
    v2_cer = [results["vanilla_rag_v2"]["cer"] * 100,
              results["oracle_doc_v2"]["cer"] * 100]

    bars1 = ax.bar(x - width/2, v1_cer, width, label='V1', color='#3498db', edgecolor='black')
    bars2 = ax.bar(x + width/2, v2_cer, width, label='V2', color='#e74c3c', edgecolor='black')

    ax.set_ylabel('Confident Error Rate (%)', fontsize=11)
    ax.set_title('CER: Confident but Wrong', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(systems)
    ax.legend()
    ax.set_ylim(0, 20)

    for bar in bars1 + bars2:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)

    # Plot 3: Oracle Gap
    ax = axes[2]

    oracle_gap_v1 = (results["oracle_doc_v1"]["success_rate"] - results["vanilla_rag_v1"]["success_rate"]) * 100
    oracle_gap_v2 = (results["oracle_doc_v2"]["success_rate"] - results["vanilla_rag_v2"]["success_rate"]) * 100

    bars = ax.bar(['V1 Gap', 'V2 Gap'], [oracle_gap_v1, oracle_gap_v2],
                  color=['#9b59b6', '#f39c12'], edgecolor='black', width=0.5)

    ax.set_ylabel('Oracle Gap (%)', fontsize=11)
    ax.set_title('Retrieval Bottleneck', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 30)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add annotation
    ax.text(0.5, -0.15, 'Gap = Oracle_Acc - RAG_Acc\n(Higher = more retrieval failures)',
           transform=ax.transAxes, ha='center', fontsize=9, style='italic')

    plt.tight_layout()
    plt.savefig(results_dir / "llm_experiment_plots.png", dpi=150, bbox_inches='tight')
    plt.savefig(results_dir / "llm_experiment_plots.pdf", bbox_inches='tight')
    print("Saved llm_experiment_plots.png/pdf")
    plt.close()


def main():
    print("Regenerating all DRIFTBENCH plots...")
    print()

    plot_validation_results()
    plot_drift_sweep()
    plot_llm_results()

    print()
    print("Done! All plots regenerated with improved visualization.")


if __name__ == "__main__":
    main()
