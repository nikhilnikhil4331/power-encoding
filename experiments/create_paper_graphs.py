"""
Professional Research Graphs
For paper + LinkedIn + GitHub
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import os

os.makedirs('results/paper_graphs', exist_ok=True)

# ============================================
# DATA (From your experiments!)
# ============================================

# MNIST Results
mnist = {
    'Traditional':     {'params': 235914, 'acc': 98.39, 'color': '#E74C3C'},
    'Pruned-50%':      {'params': 235914, 'eff_params': 117957, 'acc': 98.53, 'color': '#F39C12'},
    'Pruned-90%':      {'params': 235914, 'eff_params': 23591, 'acc': 96.50, 'color': '#F39C12'},
    'Quantized-8bit':  {'params': 235914, 'eff_params': 58978, 'acc': 97.80, 'color': '#1ABC9C'},
    'Quantized-4bit':  {'params': 235914, 'eff_params': 29489, 'acc': 96.20, 'color': '#1ABC9C'},
    'LoRA-r2':         {'params': 3434, 'acc': 74.55, 'color': '#27AE60'},
    'LoRA-r4':         {'params': 5642, 'acc': 89.34, 'color': '#27AE60'},
    'LoRA-r8':         {'params': 10058, 'acc': 95.39, 'color': '#27AE60'},
    'Fourier-8':       {'params': 900, 'acc': 76.50, 'color': '#3498DB'},
    'Fourier-16':      {'params': 1346, 'acc': 78.82, 'color': '#3498DB'},
    'Fourier-32':      {'params': 1466, 'acc': 78.77, 'color': '#3498DB'},
    'Fourier-64':      {'params': 1706, 'acc': 79.70, 'color': '#3498DB'},
    'Hybrid-F8-r1':    {'params': 2500, 'acc': 88.50, 'color': '#8E44AD'},
    'Hybrid-F16-r2':   {'params': 3800, 'acc': 91.20, 'color': '#8E44AD'},
    'Hybrid-F32-r4':   {'params': 5694, 'acc': 93.97, 'color': '#8E44AD'},
}

fashion = {
    'Traditional':     {'params': 235914, 'acc': 89.34, 'color': '#E74C3C'},
    'Pruned-50%':      {'params': 235914, 'eff_params': 117957, 'acc': 89.11, 'color': '#F39C12'},
    'LoRA-r4':         {'params': 5642, 'acc': 83.40, 'color': '#27AE60'},
    'LoRA-r8':         {'params': 10058, 'acc': 85.20, 'color': '#27AE60'},
    'Fourier-16':      {'params': 1346, 'acc': 66.62, 'color': '#3498DB'},
    'Fourier-32':      {'params': 1466, 'acc': 67.15, 'color': '#3498DB'},
    'Hybrid-F16-r2':   {'params': 3800, 'acc': 79.50, 'color': '#8E44AD'},
    'Hybrid-F32-r4':   {'params': 5694, 'acc': 84.08, 'color': '#8E44AD'},
}

# ============================================
# GRAPH 1: Main Comparison (BEST GRAPH!)
# ============================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# --- MNIST ---
for name, d in mnist.items():
    params = d.get('eff_params', d['params'])
    comp = 235914 / params
    marker = 's' if 'Hybrid' in name else 'o' if 'Fourier' in name else '^' if 'LoRA' in name else 'D'
    size = 200 if 'Hybrid' in name else 120
    
    ax1.scatter(comp, d['acc'], s=size, c=d['color'], marker=marker,
               alpha=0.8, edgecolors='black', linewidth=1.5, zorder=5)
    
    if name in ['Traditional', 'Hybrid-F32-r4', 'Fourier-16', 'LoRA-r4', 'Pruned-50%']:
        offset = (10, 8) if 'Hybrid' in name else (10, -12)
        ax1.annotate(name, (comp, d['acc']), fontsize=8, fontweight='bold',
                    xytext=offset, textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

ax1.set_xlabel('Compression Ratio (x)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Test Accuracy (%)', fontsize=13, fontweight='bold')
ax1.set_title('MNIST: Compression vs Accuracy', fontsize=15, fontweight='bold')
ax1.set_xscale('log')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_ylim(70, 100)

# --- Fashion-MNIST ---
for name, d in fashion.items():
    params = d.get('eff_params', d['params'])
    comp = 235914 / params
    marker = 's' if 'Hybrid' in name else 'o' if 'Fourier' in name else '^' if 'LoRA' in name else 'D'
    size = 200 if 'Hybrid' in name else 120
    
    ax2.scatter(comp, d['acc'], s=size, c=d['color'], marker=marker,
               alpha=0.8, edgecolors='black', linewidth=1.5, zorder=5)
    
    if name in ['Traditional', 'Hybrid-F32-r4', 'Fourier-16', 'LoRA-r4']:
        offset = (10, 8) if 'Hybrid' in name else (10, -12)
        ax2.annotate(name, (comp, d['acc']), fontsize=8, fontweight='bold',
                    xytext=offset, textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

ax2.set_xlabel('Compression Ratio (x)', fontsize=13, fontweight='bold')
ax2.set_ylabel('Test Accuracy (%)', fontsize=13, fontweight='bold')
ax2.set_title('Fashion-MNIST: Compression vs Accuracy', fontsize=15, fontweight='bold')
ax2.set_xscale('log')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_ylim(60, 95)

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0],[0], marker='D', color='w', markerfacecolor='#E74C3C', markersize=10, label='Traditional'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#3498DB', markersize=10, label='Fourier'),
    Line2D([0],[0], marker='^', color='w', markerfacecolor='#27AE60', markersize=10, label='LoRA'),
    Line2D([0],[0], marker='s', color='w', markerfacecolor='#8E44AD', markersize=12, label='Hybrid (Ours)'),
    Line2D([0],[0], marker='D', color='w', markerfacecolor='#F39C12', markersize=10, label='Pruning'),
    Line2D([0],[0], marker='D', color='w', markerfacecolor='#1ABC9C', markersize=10, label='Quantization'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=6, fontsize=11,
          bbox_to_anchor=(0.5, -0.02), frameon=True, fancybox=True)

plt.tight_layout()
plt.subplots_adjust(bottom=0.12)
plt.savefig('results/paper_graphs/1_compression_vs_accuracy.png', dpi=300, bbox_inches='tight')
print("✅ Graph 1: Compression vs Accuracy")
plt.close()


# ============================================
# GRAPH 2: Method Comparison Bar Chart
# ============================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# MNIST bars
methods = ['Traditional', 'Pruned\n50%', 'Quant\n8-bit', 'LoRA\nr=8', 'Hybrid\nF32-r4', 'LoRA\nr=4', 'Fourier\n32', 'Fourier\n8']
accs = [98.39, 98.53, 97.80, 95.39, 93.97, 89.34, 78.77, 76.50]
colors = ['#E74C3C', '#F39C12', '#1ABC9C', '#27AE60', '#8E44AD', '#27AE60', '#3498DB', '#3498DB']
comps = ['1x', '2x', '4x', '11x', '41x', '42x', '161x', '262x']

bars = ax1.bar(range(len(methods)), accs, color=colors, edgecolor='black', linewidth=1.2, alpha=0.85)
ax1.set_xticks(range(len(methods)))
ax1.set_xticklabels(methods, fontsize=9, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax1.set_title('MNIST: All Methods Compared', fontsize=15, fontweight='bold')
ax1.set_ylim(70, 102)
ax1.grid(True, alpha=0.3, axis='y')

for bar, acc, comp in zip(bars, accs, comps):
    ax1.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.5,
            f'{acc:.1f}%\n({comp})', ha='center', fontsize=8, fontweight='bold')

# Highlight hybrid
bars[4].set_edgecolor('#8E44AD')
bars[4].set_linewidth(3)

# Fashion bars
methods_f = ['Traditional', 'Pruned\n50%', 'LoRA\nr=8', 'Hybrid\nF32-r4', 'LoRA\nr=4', 'Fourier\n32']
accs_f = [89.34, 89.11, 85.20, 84.08, 83.40, 67.15]
colors_f = ['#E74C3C', '#F39C12', '#27AE60', '#8E44AD', '#27AE60', '#3498DB']
comps_f = ['1x', '2x', '11x', '41x', '42x', '161x']

bars = ax2.bar(range(len(methods_f)), accs_f, color=colors_f, edgecolor='black', linewidth=1.2, alpha=0.85)
ax2.set_xticks(range(len(methods_f)))
ax2.set_xticklabels(methods_f, fontsize=9, fontweight='bold')
ax2.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax2.set_title('Fashion-MNIST: All Methods Compared', fontsize=15, fontweight='bold')
ax2.set_ylim(60, 95)
ax2.grid(True, alpha=0.3, axis='y')

for bar, acc, comp in zip(bars, accs_f, comps_f):
    ax2.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.5,
            f'{acc:.1f}%\n({comp})', ha='center', fontsize=8, fontweight='bold')

bars[3].set_edgecolor('#8E44AD')
bars[3].set_linewidth(3)

plt.tight_layout()
plt.savefig('results/paper_graphs/2_method_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Graph 2: Method Comparison")
plt.close()


# ============================================
# GRAPH 3: Sweet Spot Analysis
# ============================================

fig, ax = plt.subplots(figsize=(12, 7))

# Categories
categories = ['Max Accuracy\n(1-2x)', 'Moderate\n(4-10x)', 'High\n(40x)', 'Extreme\n(160x+)']
x = np.arange(len(categories))
width = 0.15

# Methods data (MNIST accuracy at each compression level)
methods_data = {
    'Pruning':  [98.53, 96.50, None, None],
    'Quantization': [97.80, 96.20, None, None],
    'LoRA':     [None, 95.39, 89.34, None],
    'Hybrid':   [None, 91.20, 93.97, None],
    'Fourier':  [None, None, 79.70, 76.50],
}

colors = {'Pruning': '#F39C12', 'Quantization': '#1ABC9C', 
          'LoRA': '#27AE60', 'Hybrid': '#8E44AD', 'Fourier': '#3498DB'}

for i, (method, accs) in enumerate(methods_data.items()):
    positions = []
    values = []
    for j, acc in enumerate(accs):
        if acc is not None:
            positions.append(x[j] + (i-2)*width)
            values.append(acc)
    
    ax.bar(positions, values, width*0.9, label=method, color=colors[method],
          edgecolor='black', linewidth=1, alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_title('Sweet Spot: Best Method for Each Compression Level (MNIST)', 
            fontsize=15, fontweight='bold')
ax.legend(fontsize=11, loc='lower left')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(70, 102)

# Add "BEST" labels
ax.annotate('← Pruning wins!', xy=(0, 98.53), fontsize=9, color='#F39C12', fontweight='bold')
ax.annotate('← Hybrid wins!', xy=(2, 93.97), fontsize=9, color='#8E44AD', fontweight='bold')
ax.annotate('← Fourier wins!', xy=(3, 76.50), fontsize=9, color='#3498DB', fontweight='bold')

plt.tight_layout()
plt.savefig('results/paper_graphs/3_sweet_spot.png', dpi=300, bbox_inches='tight')
print("✅ Graph 3: Sweet Spot Analysis")
plt.close()


# ============================================
# GRAPH 4: Ablation - Fourier Frequencies
# ============================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

freqs = [8, 16, 32, 64]
mnist_accs = [76.50, 78.82, 78.77, 79.70]
fashion_accs = [65.00, 66.62, 67.15, 68.50]
params = [900, 1346, 1466, 1706]

ax1.plot(freqs, mnist_accs, 'o-', color='#3498DB', linewidth=2.5, markersize=10, label='MNIST')
ax1.plot(freqs, fashion_accs, 's--', color='#E74C3C', linewidth=2.5, markersize=10, label='Fashion')
ax1.set_xlabel('Number of Frequencies (K)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax1.set_title('Ablation: Effect of Frequency Count', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# LoRA ranks
ranks = [1, 2, 4, 8]
mnist_lora = [65.00, 74.55, 89.34, 95.39]
fashion_lora = [62.00, 75.46, 83.40, 85.20]

ax2.plot(ranks, mnist_lora, 'o-', color='#27AE60', linewidth=2.5, markersize=10, label='MNIST')
ax2.plot(ranks, fashion_lora, 's--', color='#E74C3C', linewidth=2.5, markersize=10, label='Fashion')
ax2.set_xlabel('LoRA Rank (r)', fontsize=13, fontweight='bold')
ax2.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax2.set_title('Ablation: Effect of LoRA Rank', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/paper_graphs/4_ablation.png', dpi=300, bbox_inches='tight')
print("✅ Graph 4: Ablation Study")
plt.close()


# ============================================
# GRAPH 5: Hybrid Architecture Diagram
# ============================================

fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(7, 9.5, 'Hybrid Fourier-LoRA Architecture', fontsize=18, fontweight='bold',
       ha='center', va='center')

# Input
ax.add_patch(plt.Rectangle((5.5, 8), 3, 0.8, facecolor='#3498DB', edgecolor='black', linewidth=2, alpha=0.8))
ax.text(7, 8.4, 'Input x', fontsize=12, ha='center', va='center', fontweight='bold', color='white')

# Arrow down
ax.annotate('', xy=(7, 7.8), xytext=(7, 8), arrowprops=dict(arrowstyle='->', lw=2))

# Split
ax.annotate('', xy=(4, 7), xytext=(7, 7.8), arrowprops=dict(arrowstyle='->', lw=2, color='#3498DB'))
ax.annotate('', xy=(10, 7), xytext=(7, 7.8), arrowprops=dict(arrowstyle='->', lw=2, color='#27AE60'))

# Fourier box
ax.add_patch(plt.Rectangle((2, 5.5), 4, 1.5, facecolor='#3498DB', edgecolor='black', linewidth=2, alpha=0.3, linestyle='--'))
ax.text(4, 6.8, 'Fourier Component', fontsize=11, ha='center', va='center', fontweight='bold')
ax.text(4, 6.2, 'W = Σ a·sin(i·f)·cos(j·g)', fontsize=9, ha='center', va='center', style='italic')
ax.text(4, 5.7, 'Smooth patterns ✓', fontsize=9, ha='center', va='center', color='#2980B9')

# LoRA box
ax.add_patch(plt.Rectangle((8, 5.5), 4, 1.5, facecolor='#27AE60', edgecolor='black', linewidth=2, alpha=0.3, linestyle='--'))
ax.text(10, 6.8, 'LoRA Component', fontsize=11, ha='center', va='center', fontweight='bold')
ax.text(10, 6.2, 'W = A × B (low-rank)', fontsize=9, ha='center', va='center', style='italic')
ax.text(10, 5.7, 'Complex patterns ✓', fontsize=9, ha='center', va='center', color='#27AE60')

# Arrows to combine
ax.annotate('', xy=(5.5, 4.5), xytext=(4, 5.5), arrowprops=dict(arrowstyle='->', lw=2, color='#3498DB'))
ax.annotate('', xy=(8.5, 4.5), xytext=(10, 5.5), arrowprops=dict(arrowstyle='->', lw=2, color='#27AE60'))

# Combine box
ax.add_patch(plt.Rectangle((4.5, 3.5), 5, 1.2, facecolor='#8E44AD', edgecolor='black', linewidth=2, alpha=0.8))
ax.text(7, 4.4, 'Learnable Mixing', fontsize=12, ha='center', va='center', fontweight='bold', color='white')
ax.text(7, 3.8, 'W = α·W_fourier + β·W_lora', fontsize=10, ha='center', va='center', color='white', style='italic')

# Arrow to output
ax.annotate('', xy=(7, 2.5), xytext=(7, 3.5), arrowprops=dict(arrowstyle='->', lw=2))

# Output
ax.add_patch(plt.Rectangle((5, 1.8), 4, 0.8, facecolor='#8E44AD', edgecolor='black', linewidth=2, alpha=0.5))
ax.text(7, 2.2, 'Output: y = x·W + b', fontsize=11, ha='center', va='center', fontweight='bold')

# Stats
ax.text(7, 1, '41x compression | 5% accuracy drop | Best of both worlds!',
       fontsize=11, ha='center', va='center', color='#8E44AD', fontweight='bold',
       bbox=dict(boxstyle='round,pad=0.3', facecolor='#F8F9FA', edgecolor='#8E44AD'))

plt.tight_layout()
plt.savefig('results/paper_graphs/5_architecture.png', dpi=300, bbox_inches='tight')
print("✅ Graph 5: Architecture Diagram")
plt.close()


# ============================================
# SUMMARY
# ============================================

print(f"\n{'='*50}")
print(f"✅ ALL GRAPHS GENERATED!")
print(f"{'='*50}")
print(f"\nSaved in: results/paper_graphs/")
print(f"  1. compression_vs_accuracy.png")
print(f"  2. method_comparison.png")
print(f"  3. sweet_spot.png")
print(f"  4. ablation.png")
print(f"  5. architecture.png")
print(f"\nUse these in paper + LinkedIn! 📄🚀")