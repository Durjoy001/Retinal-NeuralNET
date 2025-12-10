#!/usr/bin/env python3
"""
Bar chart comparing Sensitivity at 80% Specificity threshold
for Binary Classification vs Multi-class Classification for all 12 models.
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

# Extract binary sensitivity at 80% spec
binary_sensitivity = df["Sensitivity@80%Spec (%)"].values

# Extract multi-class sensitivity at 80% spec from overall_test_results.csv files
MODELS = {
    "Densenet121": "results/CNN/Densenet121",
    "EfficientNetB3": "results/CNN/EfficientNetB3",
    "InceptionV3": "results/CNN/InceptionV3",
    "ResNet50": "results/CNN/ResNet50",
    "CrossViTSmall": "results/ViT/CrossViTSmall",
    "DeiTSmall": "results/ViT/DeiTSmall",
    "SwinTiny": "results/ViT/SwinTiny",
    "ViTSmall": "results/ViT/ViTSmall",
    "CoAtNet0": "results/Hybrid/CoAtNet0",
    "MaxViTTiny": "results/Hybrid/MaxViTTiny",
    "CLIPViTB16": "results/VLM/CLIPViTB16",
    "SigLIPBase384": "results/VLM/SigLIPBase384",
}

multiclass_sensitivity = []
for model_name in df["Model"].values:
    model_path = base_dir / MODELS[model_name]
    test_results_file = model_path / "overall_test_results.csv"
    
    if test_results_file.exists():
        try:
            df_test = pd.read_csv(test_results_file)
            test_dict = dict(zip(df_test.iloc[:, 0], df_test.iloc[:, 1]))
            # test_sensitivity is calculated with thresholds calibrated at 80% specificity
            sens = float(test_dict.get("test_sensitivity", 0)) * 100  # Convert to percentage
            multiclass_sensitivity.append(sens)
        except Exception as e:
            print(f"Warning: Could not load multi-class sensitivity for {model_name}: {e}")
            multiclass_sensitivity.append(0)
    else:
        print(f"Warning: {test_results_file} not found for {model_name}")
        multiclass_sensitivity.append(0)

multiclass_sensitivity = np.array(multiclass_sensitivity)

# Sort by binary sensitivity in ascending order
sort_idx = np.argsort(binary_sensitivity)
models = df["Model"].values[sort_idx]
binary_sensitivity = binary_sensitivity[sort_idx]
multiclass_sensitivity = multiclass_sensitivity[sort_idx]

# Set up the plot
fig, ax = plt.subplots(figsize=(16, 8.5))

# Set up bar positions
x = np.arange(len(models))
width = 0.35  # Width of the bars

# Create bars
bars1 = ax.bar(
    x - width/2,
    binary_sensitivity,
    width,
    label='Binary screening task',
    color='#B0B0B0',          # lighter grey
    alpha=1.0,
    edgecolor='#909090',      # darker grey edge for contrast
    linewidth=1.2
)

bars2 = ax.bar(
    x + width/2,
    multiclass_sensitivity,
    width,
    label='Multi label classification',
    color='#505050',          # dark grey
    alpha=1.0,
    edgecolor='#303030',      # darker grey edge
    linewidth=1.2
)

# Customize the plot
ax.set_xlabel('', fontsize=14, fontweight='bold')
ax.set_ylabel('Model Performance, Sensitivity (%)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right', fontsize=11)
legend = ax.legend(loc='upper left', fontsize=12, framealpha=0.9)
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_ylim([60, 100])  # Set y-axis range to better show differences

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
output_file = output_dir / "sensitivity_comparison.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✅ Chart saved to: {output_file}")

# Also save as PDF for high quality
output_file_pdf = output_dir / "sensitivity_comparison.pdf"
plt.savefig(output_file_pdf, bbox_inches='tight')
print(f"✅ Chart saved to: {output_file_pdf}")

# Show the plot
plt.show()

