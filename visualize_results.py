"""
Visualize Power Encoding Results
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from our experiments
models = ['Traditional', 'Power\n(seed=32)', 'Power\n(seed=64)']
parameters = [109386, 1555, 5587]
accuracy = [96.97, 68.63, 78.88]
compression = [1, 70.3, 19.6]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Parameters
axes[0].bar(models, parameters, color=['red', 'lightblue', 'blue'])
axes[0].set_ylabel('Parameters', fontsize=12)
axes[0].set_title('Model Size Comparison', fontsize=14, fontweight='bold')
axes[0].set_yscale('log')
for i, v in enumerate(parameters):
    axes[0].text(i, v, f'{v:,}', ha='center', va='bottom')

# Plot 2: Accuracy
axes[1].bar(models, accuracy, color=['red', 'lightblue', 'blue'])
axes[1].set_ylabel('Accuracy (%)', fontsize=12)
axes[1].set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
axes[1].set_ylim([0, 100])
for i, v in enumerate(accuracy):
    axes[1].text(i, v+2, f'{v:.2f}%', ha='center', va='bottom')

# Plot 3: Compression vs Accuracy
axes[2].scatter([compression[1], compression[2]], 
                [accuracy[1], accuracy[2]], 
                s=200, c=['lightblue', 'blue'], alpha=0.6)
axes[2].scatter([1], [96.97], s=200, c=['red'], alpha=0.6)
axes[2].set_xlabel('Compression Ratio', fontsize=12)
axes[2].set_ylabel('Accuracy (%)', fontsize=12)
axes[2].set_title('Compression vs Accuracy Trade-off', fontsize=14, fontweight='bold')
axes[2].set_xscale('log')
axes[2].grid(True, alpha=0.3)

# Annotations
axes[2].annotate('Traditional', (1, 96.97), xytext=(1.5, 95), 
                arrowprops=dict(arrowstyle='->', color='red'))
axes[2].annotate('Power (32)', (70.3, 68.63), xytext=(40, 65), 
                arrowprops=dict(arrowstyle='->', color='lightblue'))
axes[2].annotate('Power (64)\n⭐ Best', (19.6, 78.88), xytext=(30, 82), 
                arrowprops=dict(arrowstyle='->', color='blue'))

plt.tight_layout()
plt.savefig('power_encoding_results.png', dpi=150, bbox_inches='tight')
print("📊 Visualization saved: power_encoding_results.png")
plt.show()