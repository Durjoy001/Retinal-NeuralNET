#!/usr/bin/env python3
"""
Bar chart comparing Binary Classification (AUC) vs Multi-level Classification (AUCmacro)
for all 12 models.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load the comprehensive summary data
base_dir = Path(__file__).parent.parent
data_file = base_dir / "results" / "all_models_comprehensive_summary.csv"

# Read the data
df = pd.read_csv(data_file)

# Sort by binary classification AUC in ascending order (lowest to highest)
df_sorted = df.sort_values(by="AUC (%)", ascending=True).reset_index(drop=True)

# Extract data from sorted dataframe
models = df_sorted["Model"].values
binary_auc = df_sorted["AUC (%)"].values
multilevel_auc = df_sorted["AUCmacro"].values

# Set up the plot
fig, ax = plt.subplots(figsize=(16, 8))

# Set up bar positions
x = np.arange(len(models))
width = 0.35  # Width of the bars

# Create bars
bars1 = ax.bar(x - width/2, binary_auc, width, label='Any abnormal vs normal binary screening (AUC)', 
               color='#1E88E5', alpha=1.0, edgecolor='#0D47A1', linewidth=1.2)
bars2 = ax.bar(x + width/2, multilevel_auc, width, label='Multi label classification (AUCmacro)', 
               color='#43A047', alpha=1.0, edgecolor='#1B5E20', linewidth=1.2)

# Customize the plot
ax.set_xlabel('', fontsize=14, fontweight='bold')
ax.set_ylabel('Model Performance, AUC (%)', fontsize=14, fontweight='bold')
# Title removed as requested
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right', fontsize=11)
legend = ax.legend(loc='upper left', fontsize=12, framealpha=0.9)
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_ylim([70, 100])  # Set y-axis range to better show differences

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add dataset label right below the legend
plt.draw()  # Draw to get legend position
legend_bbox = legend.get_window_extent().transformed(ax.transAxes.inverted())
ax.text(legend_bbox.x0, legend_bbox.y0 - 0.03, 'Dataset: RFMiD', transform=ax.transAxes,
        ha='left', va='top', fontsize=12, fontweight='bold')

# Add value labels on bars
def add_value_labels(bars):
    """Add value labels on top of bars."""
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

add_value_labels(bars1)
add_value_labels(bars2)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the figure
output_dir = base_dir / "Graph"
output_dir.mkdir(exist_ok=True)
output_file = output_dir / "model_performance_comparison.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✅ Chart saved to: {output_file}")

# Also save as PDF for high quality
output_file_pdf = output_dir / "model_performance_comparison.pdf"
plt.savefig(output_file_pdf, bbox_inches='tight')
print(f"✅ Chart saved to: {output_file_pdf}")

# Show the plot
plt.show()

