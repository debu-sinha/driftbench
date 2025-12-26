"""Regenerate plots for DRIFTBENCH v4 (77-task results)."""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Results from full_experiment_results.json
results = {
    "term_rag": {"v1_acc": 64.9, "v2_acc": 70.1, "v1_sfr": 11.7, "v2_sfr": 11.7},
    "dense_rag": {"v1_acc": 80.5, "v2_acc": 85.7, "v1_sfr": 14.3, "v2_sfr": 10.4},
    "oracle": {"v1_acc": 80.5, "v2_acc": 87.0, "v1_sfr": 15.6, "v2_sfr": 7.8},
}

output_dir = Path(__file__).parent.parent / "paper"

# Figure 1: Accuracy + SFR comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

systems = ["Term Overlap", "Dense (MiniLM)", "Oracle-Doc"]
v1_acc = [results["term_rag"]["v1_acc"], results["dense_rag"]["v1_acc"], results["oracle"]["v1_acc"]]
v2_acc = [results["term_rag"]["v2_acc"], results["dense_rag"]["v2_acc"], results["oracle"]["v2_acc"]]

x = np.arange(len(systems))
width = 0.35

# Left: Accuracy
ax = axes[0]
bars1 = ax.bar(x - width/2, v1_acc, width, label='V1 (baseline)', color='#3498db', edgecolor='black')
bars2 = ax.bar(x + width/2, v2_acc, width, label='V2 (drifted)', color='#2ecc71', edgecolor='black')
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Accuracy Improves Under Drift', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(systems)
ax.legend()
ax.set_ylim(0, 100)
for bar in bars1:
    ax.annotate(f'{bar.get_height():.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
for bar in bars2:
    ax.annotate(f'{bar.get_height():.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

# Right: SFR
ax = axes[1]
v1_sfr = [results["term_rag"]["v1_sfr"], results["dense_rag"]["v1_sfr"], results["oracle"]["v1_sfr"]]
v2_sfr = [results["term_rag"]["v2_sfr"], results["dense_rag"]["v2_sfr"], results["oracle"]["v2_sfr"]]
bars1 = ax.bar(x - width/2, v1_sfr, width, label='V1', color='#e74c3c', edgecolor='black')
bars2 = ax.bar(x + width/2, v2_sfr, width, label='V2', color='#c0392b', edgecolor='black')
ax.set_ylabel('Silent Failure Rate (%)', fontsize=12)
ax.set_title('SFR Persists Regardless of Accuracy', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(systems)
ax.legend()
ax.set_ylim(0, 25)
ax.axhline(y=12, color='gray', linestyle='--', alpha=0.7, label='~12% baseline')
for bar in bars1:
    ax.annotate(f'{bar.get_height():.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
for bar in bars2:
    ax.annotate(f'{bar.get_height():.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / "fig_v1v2.pdf", bbox_inches='tight')
plt.savefig(output_dir / "fig_v1v2.png", dpi=150, bbox_inches='tight')
print(f"Saved: {output_dir / 'fig_v1v2.pdf'}")

plt.close()
print("Done.")
