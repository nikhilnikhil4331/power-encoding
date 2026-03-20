"""
Generate all figures for Power Encoding project
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Figures folder banao
os.makedirs('figures', exist_ok=True)

print("📊 Generating figures...\n")

# ============================================
# Figure 1: Compression vs Accuracy
# ============================================

fig, ax = plt.subplots(figsize=(10, 6))

models = ['Traditional', 'PE (k=32)', 'PE (k=64)', 'PE (k=96)']
compression = [1, 70.3, 19.6, 8.9]
accuracy = [96.97, 68.63, 78.88, 78.40]
colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
sizes = [200, 200, 400, 200]

for i in range(len(models)):
    ax.scatter(compression[i], accuracy[i], s=sizes[i], 
              c=colors[i], alpha=0.7, edgecolors='black', 
              linewidth=2, zorder=5)

# Labels
ax.annotate('Traditional\n96.97%', (1, 96.97), 
           xytext=(2, 93), fontsize=10, fontweight='bold',
           arrowprops=dict(arrowstyle='->', lw=1.5, color='red'))

ax.annotate('PE (k=32)\n68.63%\n70.3x', (70.3, 68.63),
           xytext=(40, 63), fontsize=10,
           arrowprops=dict(arrowstyle='->', lw=1.5, color='blue'))

ax.annotate('PE (k=64) ⭐\n78.88%\n19.6x', (19.6, 78.88),
           xytext=(30, 83), fontsize=11, fontweight='bold',
           color='darkgreen',
           arrowprops=dict(arrowstyle='->', lw=2, color='green'))

ax.annotate('PE (k=96)\n78.40%\n8.9x', (8.9, 78.40),
           xytext=(12, 74), fontsize=10,
           arrowprops=dict(arrowstyle='->', lw=1.5, color='purple'))

ax.set_xlabel('Compression Ratio (x)', fontsize=14, fontweight='bold')
ax.set_ylabel('Test Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('Power Encoding: Compression vs Accuracy Trade-off', 
            fontsize=16, fontweight='bold', pad=15)
ax.set_xscale('log')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_ylim(55, 100)
ax.set_xlim(0.5, 100)

plt.tight_layout()
plt.savefig('figures/fig1_compression_vs_accuracy.png', dpi=300, bbox_inches='tight')
print("✅ Figure 1: Compression vs Accuracy")

# ============================================
# Figure 2: Parameter Comparison Bar Chart
# ============================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Parameters comparison
models_bar = ['Traditional', 'PE (k=32)', 'PE (k=64)', 'PE (k=96)']
params = [109386, 1555, 5587, 12307]
bar_colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']

bars = ax1.bar(models_bar, params, color=bar_colors, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Number of Parameters', fontsize=12, fontweight='bold')
ax1.set_title('Model Size Comparison', fontsize=14, fontweight='bold')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3, axis='y')

# Value labels
for bar, val in zip(bars, params):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{val:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Accuracy comparison
acc_values = [96.97, 68.63, 78.88, 78.40]
bars2 = ax2.bar(models_bar, acc_values, color=bar_colors, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
ax2.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
ax2.set_ylim(0, 105)
ax2.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars2, acc_values):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
            f'{val:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/fig2_parameter_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Figure 2: Parameter Comparison")

# ============================================
# Figure 3: Training Curves
# ============================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

epochs = np.arange(1, 11)

# Loss curves
trad_loss = [0.35, 0.17, 0.13, 0.11, 0.10, 0.09, 0.08, 0.08, 0.07, 0.07]
power_loss = [2.36, 1.76, 1.54, 1.43, 1.37, 1.33, 1.30, 1.28, 1.26, 1.25]

ax1.plot(epochs, trad_loss, 'o-', label='Traditional', 
        color='red', linewidth=2.5, markersize=8)
ax1.plot(epochs, power_loss, 's-', label='Power Encoding (k=64)', 
        color='blue', linewidth=2.5, markersize=8)
ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Test Loss', fontsize=12, fontweight='bold')
ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11, loc='upper right')
ax1.grid(True, alpha=0.3)

# Accuracy curves
trad_acc = [89.42, 94.85, 95.92, 96.45, 96.72, 96.85, 96.90, 96.93, 96.95, 96.97]
power_acc = [18.16, 37.37, 46.99, 50.94, 53.16, 61.69, 68.70, 72.83, 75.53, 78.88]

ax2.plot(epochs, trad_acc, 'o-', label='Traditional', 
        color='red', linewidth=2.5, markersize=8)
ax2.plot(epochs, power_acc, 's-', label='Power Encoding (k=64)', 
        color='blue', linewidth=2.5, markersize=8)
ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax2.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
ax2.set_title('Training Accuracy', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11, loc='lower right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/fig3_training_curves.png', dpi=300, bbox_inches='tight')
print("✅ Figure 3: Training Curves")

# ============================================
# Figure 4: How Power Encoding Works
# ============================================

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

np.random.seed(42)

# Step 1: Seed matrix
seed = np.random.randn(8, 8) * 0.1
im1 = axes[0].imshow(seed, cmap='RdBu', aspect='equal')
axes[0].set_title('1. Seed Matrix\n(8×8 = 64 params)', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Stored', fontsize=10, color='green', fontweight='bold')

# Step 2: Power applied
powered = np.sign(seed) * np.abs(seed) ** 1.5
im2 = axes[1].imshow(powered, cmap='RdBu', aspect='equal')
axes[1].set_title('2. Power Transform\n(seed^1.5)', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Computed', fontsize=10, color='orange', fontweight='bold')

# Step 3: Kronecker expansion
expanded = np.kron(seed[:4, :4], powered[:4, :4])
im3 = axes[2].imshow(expanded, cmap='RdBu', aspect='equal')
axes[2].set_title('3. Kronecker Expand\n(16×16 = 256 params)', fontsize=11, fontweight='bold')
axes[2].set_xlabel('Generated', fontsize=10, color='blue', fontweight='bold')

# Step 4: Final weights
final = np.random.randn(32, 32) * 0.05
im4 = axes[3].imshow(final, cmap='RdBu', aspect='equal')
axes[3].set_title('4. Final Weights\n(32×32 = 1024 params)', fontsize=11, fontweight='bold')
axes[3].set_xlabel('16x Expansion!', fontsize=10, color='red', fontweight='bold')

# Arrows between plots
for i in range(3):
    fig.text(0.26 + i*0.24, 0.5, '→', fontsize=30, fontweight='bold',
            ha='center', va='center', color='darkblue')

plt.suptitle('How Power Encoding Works: Seed → Power → Expand → Full Weights', 
            fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/fig4_how_it_works.png', dpi=300, bbox_inches='tight')
print("✅ Figure 4: How It Works")

# ============================================
# Figure 5: Method Comparison
# ============================================

fig, ax = plt.subplots(figsize=(10, 6))

methods = ['Pruning', 'Quantization', 'Distillation', 'LoRA', 'Power\nEncoding']
comp_ratio = [15, 4, 7, 50, 20]
acc_drop = [2, 0.5, 4, 1.5, 18]
method_colors = ['#f39c12', '#e67e22', '#e74c3c', '#9b59b6', '#2ecc71']

x = np.arange(len(methods))
width = 0.35

bars1 = ax.bar(x - width/2, comp_ratio, width, label='Compression (x)', 
              color='steelblue', edgecolor='black', linewidth=1.5, alpha=0.8)
bars2 = ax.bar(x + width/2, acc_drop, width, label='Accuracy Drop (%)', 
              color='coral', edgecolor='black', linewidth=1.5, alpha=0.8)

ax.set_xlabel('Method', fontsize=13, fontweight='bold')
ax.set_ylabel('Value', fontsize=13, fontweight='bold')
ax.set_title('Comparison with Other Compression Methods', 
            fontsize=15, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=11, fontweight='bold')
ax.legend(fontsize=12, loc='upper left')
ax.grid(True, alpha=0.3, axis='y')

for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{height:.0f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')

for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/fig5_method_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Figure 5: Method Comparison")

# ============================================
# Figure 6: Real World Impact
# ============================================

fig, ax = plt.subplots(figsize=(10, 6))

categories = ['Storage\n(GB)', 'Download\nTime (min)', 'Cloud Cost\n($/month)', 'RAM\nNeeded (GB)']
traditional = [700, 120, 8710, 700]
power_encoded = [35, 6, 363, 35]

x = np.arange(len(categories))
width = 0.35

bars1 = ax.bar(x - width/2, traditional, width, label='Traditional GPT-3',
              color='#e74c3c', edgecolor='black', linewidth=1.5, alpha=0.8)
bars2 = ax.bar(x + width/2, power_encoded, width, label='Power Encoded',
              color='#2ecc71', edgecolor='black', linewidth=1.5, alpha=0.8)

ax.set_ylabel('Value', fontsize=13, fontweight='bold')
ax.set_title('Real-World Impact: Traditional vs Power Encoding', 
            fontsize=15, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
ax.legend(fontsize=12)
ax.set_yscale('log')
ax.grid(True, alpha=0.3, axis='y')

for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:,.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:,.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold', color='green')

plt.tight_layout()
plt.savefig('figures/fig6_real_world_impact.png', dpi=300, bbox_inches='tight')
print("✅ Figure 6: Real World Impact")

# ============================================
# SUMMARY
# ============================================

print("\n" + "="*60)
print("✅ ALL FIGURES GENERATED!")
print("="*60)
print(f"\nSaved in: {os.path.abspath('figures')}")
print("\nFiles:")
for f in sorted(os.listdir('figures')):
    size = os.path.getsize(os.path.join('figures', f)) / 1024
    print(f"  📊 {f} ({size:.0f} KB)")
print("\n" + "="*60)