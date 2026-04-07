"""
COMPLETE ABLATION STUDY
+ All Baselines (Fourier, LoRA, Hybrid, Pruning, Quantization)
+ Visual Graphs
+ Multiple Datasets

This is the KILLER experiment for the paper!
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import time
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# ============================================
# ALL COMPRESSION METHODS
# ============================================

class FourierLayer(nn.Module):
    def __init__(self, in_f, out_f, num_freq=32):
        super().__init__()
        self.in_f, self.out_f, self.nf = in_f, out_f, num_freq
        self.amp = nn.Parameter(torch.randn(num_freq) * 0.1)
        self.rf = nn.Parameter(torch.randn(num_freq) * 2.0)
        self.cf = nn.Parameter(torch.randn(num_freq) * 2.0)
        self.rp = nn.Parameter(torch.randn(num_freq) * 0.5)
        self.cp = nn.Parameter(torch.randn(num_freq) * 0.5)
        self.bias = nn.Parameter(torch.zeros(out_f))

    def forward(self, x):
        d = self.amp.device
        r = torch.linspace(0, math.pi*2, self.out_f, device=d)
        c = torch.linspace(0, math.pi*2, self.in_f, device=d)
        W = sum(self.amp[k] * torch.outer(
            torch.sin(r*self.rf[k]+self.rp[k]),
            torch.cos(c*self.cf[k]+self.cp[k])
        ) for k in range(self.nf))
        std = math.sqrt(2.0/(self.in_f+self.out_f))
        return x @ (torch.tanh(W)*std).T + self.bias


class LoRALayer(nn.Module):
    def __init__(self, in_f, out_f, rank=4):
        super().__init__()
        self.A = nn.Parameter(torch.randn(out_f, rank)*0.01)
        self.B = nn.Parameter(torch.randn(rank, in_f)*0.01)
        self.bias = nn.Parameter(torch.zeros(out_f))

    def forward(self, x):
        W = self.A @ self.B
        std = math.sqrt(2.0/(x.shape[1]+self.A.shape[0]))
        return x @ (torch.tanh(W)*std).T + self.bias


class HybridLayer(nn.Module):
    def __init__(self, in_f, out_f, num_freq=16, rank=2):
        super().__init__()
        self.fourier = FourierLayer(in_f, out_f, num_freq)
        self.lora_A = nn.Parameter(torch.randn(out_f, rank)*0.01)
        self.lora_B = nn.Parameter(torch.randn(rank, in_f)*0.01)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.bias = nn.Parameter(torch.zeros(out_f))
        self.in_f, self.out_f = in_f, out_f

    def forward(self, x):
        W_f = self.fourier.forward.__wrapped__(self.fourier, x) if hasattr(self.fourier.forward, '__wrapped__') else None
        
        # Generate Fourier weights
        d = self.fourier.amp.device
        r = torch.linspace(0, math.pi*2, self.out_f, device=d)
        c = torch.linspace(0, math.pi*2, self.in_f, device=d)
        W_fourier = sum(self.fourier.amp[k] * torch.outer(
            torch.sin(r*self.fourier.rf[k]+self.fourier.rp[k]),
            torch.cos(c*self.fourier.cf[k]+self.fourier.cp[k])
        ) for k in range(self.fourier.nf))
        
        # LoRA weights
        W_lora = self.lora_A @ self.lora_B
        
        std = math.sqrt(2.0/(self.in_f+self.out_f))
        a = torch.sigmoid(self.alpha)
        b = torch.sigmoid(self.beta)
        W = a * torch.tanh(W_fourier)*std + b * torch.tanh(W_lora)*std
        
        return x @ W.T + self.bias


# Pruned Layer (magnitude pruning)
class PrunedLayer(nn.Module):
    def __init__(self, in_f, out_f, sparsity=0.9):
        super().__init__()
        self.linear = nn.Linear(in_f, out_f)
        self.sparsity = sparsity
        self.mask = None

    def forward(self, x):
        if self.mask is None or self.mask.shape != self.linear.weight.shape:
            self._update_mask()
        return nn.functional.linear(x, self.linear.weight * self.mask, self.linear.bias)

    def _update_mask(self):
        with torch.no_grad():
            w = self.linear.weight.abs()
            threshold = torch.quantile(w, self.sparsity)
            self.mask = (w >= threshold).float().to(w.device)


# Quantized Layer (simulated 8-bit)
class QuantizedLayer(nn.Module):
    def __init__(self, in_f, out_f, bits=8):
        super().__init__()
        self.linear = nn.Linear(in_f, out_f)
        self.bits = bits

    def forward(self, x):
        w = self.linear.weight
        w_min, w_max = w.min(), w.max()
        scale = (w_max - w_min) / (2**self.bits - 1)
        w_q = torch.round((w - w_min) / (scale + 1e-8)) * scale + w_min
        return nn.functional.linear(x, w_q, self.linear.bias)


# ============================================
# NETWORK BUILDERS
# ============================================

def build_net(method, in_size, classes, **kw):
    if method == 'traditional':
        return nn.Sequential(
            nn.Linear(in_size, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.2),
            nn.Linear(128, classes))

    elif method == 'fourier':
        f = kw.get('freq', 32)
        return nn.Sequential(
            FourierLayer(in_size, 128, f), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.2),
            FourierLayer(128, 64, f//2), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.2),
            nn.Linear(64, classes))

    elif method == 'lora':
        r = kw.get('rank', 4)
        return nn.Sequential(
            LoRALayer(in_size, 128, r), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.2),
            LoRALayer(128, 64, r//2 or 1), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.2),
            nn.Linear(64, classes))

    elif method == 'hybrid':
        f, r = kw.get('freq', 16), kw.get('rank', 2)
        return nn.Sequential(
            HybridLayer(in_size, 128, f, r), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.2),
            HybridLayer(128, 64, f//2, max(r//2,1)), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.2),
            nn.Linear(64, classes))

    elif method == 'pruned':
        s = kw.get('sparsity', 0.9)
        return nn.Sequential(
            PrunedLayer(in_size, 256, s), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
            PrunedLayer(256, 128, s), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.2),
            nn.Linear(128, classes))

    elif method == 'quantized':
        b = kw.get('bits', 8)
        return nn.Sequential(
            QuantizedLayer(in_size, 256, b), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
            QuantizedLayer(256, 128, b), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.2),
            nn.Linear(128, classes))


# ============================================
# TRAIN + EVAL
# ============================================

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def effective_params(model):
    """Count effective (non-zero) params for pruned models"""
    total = 0
    for m in model.modules():
        if isinstance(m, PrunedLayer):
            if m.mask is not None:
                total += m.mask.sum().item() + m.linear.bias.numel()
            else:
                total += sum(p.numel() for p in m.parameters())
        elif isinstance(m, (nn.Linear, FourierLayer, LoRALayer, HybridLayer, QuantizedLayer)):
            total += sum(p.numel() for p in m.parameters())
    return int(total) if total > 0 else count_params(model)

def train_eval(model, name, train_l, test_l, epochs=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best = 0
    history = []
    start = time.time()

    for epoch in range(epochs):
        model.train()
        for data, target in train_l:
            data = data.view(data.size(0), -1).to(device)
            target = target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Update pruning masks
        for m in model.modules():
            if isinstance(m, PrunedLayer):
                m._update_mask()

        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_l:
                data = data.view(data.size(0), -1).to(device)
                target = target.to(device)
                correct += model(data).argmax(1).eq(target).sum().item()

        acc = 100.*correct/len(test_l.dataset)
        if acc > best: best = acc
        history.append(acc)

    elapsed = time.time() - start
    eff_p = effective_params(model)

    return {'name': name, 'params': count_params(model), 'effective_params': eff_p,
            'accuracy': best, 'time': round(elapsed,1), 'history': history}


# ============================================
# GRAPHS
# ============================================

def plot_results(all_results, save_dir='results/graphs'):
    os.makedirs(save_dir, exist_ok=True)

    for dataset_name, results in all_results.items():
        trad = results[0]

        # Graph 1: Compression vs Accuracy
        fig, ax = plt.subplots(figsize=(10, 6))
        for r in results:
            comp = trad['params']/r['effective_params'] if r['effective_params'] < trad['params'] else 1
            color = {'Traditional':'red','Fourier':'blue','LoRA':'green',
                     'Hybrid':'purple','Pruned':'orange','Quantized':'cyan'}
            c = 'gray'
            for k,v in color.items():
                if k.lower() in r['name'].lower(): c = v
            ax.scatter(comp, r['accuracy'], s=150, c=c, alpha=0.7, edgecolors='black', zorder=5)
            ax.annotate(r['name'], (comp, r['accuracy']), fontsize=7,
                       xytext=(5,5), textcoords='offset points')

        ax.set_xlabel('Compression Ratio', fontsize=12, fontweight='bold')
        ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'{dataset_name}: Compression vs Accuracy', fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{dataset_name}_compression_vs_accuracy.png', dpi=200)
        plt.close()
        print(f"  📊 Saved: {dataset_name}_compression_vs_accuracy.png")

        # Graph 2: Params vs Accuracy
        fig, ax = plt.subplots(figsize=(10, 6))
        names = [r['name'] for r in results]
        accs = [r['accuracy'] for r in results]
        colors = []
        for r in results:
            c = 'gray'
            for k,v in {'Traditional':'red','Fourier':'blue','LoRA':'green',
                        'Hybrid':'purple','Pruned':'orange','Quantized':'cyan'}.items():
                if k.lower() in r['name'].lower(): c = v
            colors.append(c)

        bars = ax.bar(range(len(names)), accs, color=colors, edgecolor='black', alpha=0.8)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'{dataset_name}: Method Comparison', fontsize=14, fontweight='bold')

        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.5,
                   f'{acc:.1f}%', ha='center', fontsize=8, fontweight='bold')

        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{dataset_name}_method_comparison.png', dpi=200)
        plt.close()
        print(f"  📊 Saved: {dataset_name}_method_comparison.png")

        # Graph 3: Training curves
        fig, ax = plt.subplots(figsize=(10, 6))
        for r in results:
            if r['history']:
                c = 'gray'
                for k,v in {'Traditional':'red','Fourier':'blue','LoRA':'green',
                            'Hybrid':'purple','Pruned':'orange','Quantized':'cyan'}.items():
                    if k.lower() in r['name'].lower(): c = v
                ax.plot(r['history'], label=r['name'], linewidth=2, color=c)

        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'{dataset_name}: Training Curves', fontsize=14, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{dataset_name}_training_curves.png', dpi=200)
        plt.close()
        print(f"  📊 Saved: {dataset_name}_training_curves.png")


# ============================================
# MAIN EXPERIMENT
# ============================================

def main():
    print(f"\n{'🔬'*20}")
    print(f"  COMPLETE ABLATION STUDY")
    print(f"  All Methods + All Datasets + Graphs")
    print(f"{'🔬'*20}\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Datasets
    configs = [
        ('MNIST', datasets.MNIST, (0.1307,), (0.3081,), 784, 10),
        ('Fashion-MNIST', datasets.FashionMNIST, (0.2860,), (0.3530,), 784, 10),
    ]

    all_results = {}

    for ds_name, ds_class, mean, std_v, inp, cls in configs:
        print(f"\n{'='*60}")
        print(f"📊 {ds_name}")
        print(f"{'='*60}")

        t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std_v)])
        train_l = DataLoader(ds_class('./data', True, download=True, transform=t), 64, shuffle=True)
        test_l = DataLoader(ds_class('./data', False, download=True, transform=t), 64)

        results = []

        # All methods
        experiments = [
            ('Traditional', 'traditional', {}),
            ('Fourier-8', 'fourier', {'freq': 8}),
            ('Fourier-16', 'fourier', {'freq': 16}),
            ('Fourier-32', 'fourier', {'freq': 32}),
            ('Fourier-64', 'fourier', {'freq': 64}),
            ('LoRA-r1', 'lora', {'rank': 1}),
            ('LoRA-r2', 'lora', {'rank': 2}),
            ('LoRA-r4', 'lora', {'rank': 4}),
            ('LoRA-r8', 'lora', {'rank': 8}),
            ('Hybrid-F8-r1', 'hybrid', {'freq': 8, 'rank': 1}),
            ('Hybrid-F16-r2', 'hybrid', {'freq': 16, 'rank': 2}),
            ('Hybrid-F32-r4', 'hybrid', {'freq': 32, 'rank': 4}),
            ('Pruned-50%', 'pruned', {'sparsity': 0.5}),
            ('Pruned-90%', 'pruned', {'sparsity': 0.9}),
            ('Pruned-95%', 'pruned', {'sparsity': 0.95}),
            ('Quantized-8bit', 'quantized', {'bits': 8}),
            ('Quantized-4bit', 'quantized', {'bits': 4}),
        ]

        for name, method, kwargs in experiments:
            print(f"  Training {name}...", end=' ')
            model = build_net(method, inp, cls, **kwargs)
            result = train_eval(model, name, train_l, test_l, epochs=15)
            results.append(result)
            print(f"✅ {result['accuracy']:.2f}% ({result['effective_params']:,} params)")

            if device.type == 'cuda':
                torch.cuda.empty_cache()

        all_results[ds_name] = results

        # Print table
        trad = results[0]
        print(f"\n  {'Method':<20} | {'Params':>10} | {'Eff.Params':>10} | {'Acc':>7} | {'Comp':>6}")
        print(f"  {'-'*60}")
        for r in results:
            comp = f"{trad['params']/r['effective_params']:.0f}x" if r['effective_params'] < trad['params'] else "1x"
            print(f"  {r['name']:<20} | {r['params']:>10,} | {r['effective_params']:>10,} | {r['accuracy']:>5.2f}% | {comp:>6}")

    # Generate graphs
    print(f"\n📊 Generating graphs...")
    plot_results(all_results)

    # Save results
    os.makedirs('results', exist_ok=True)
    save_data = {}
    for ds, results in all_results.items():
        save_data[ds] = [{k:v for k,v in r.items() if k != 'history'} for r in results]

    with open('results/complete_ablation.json', 'w') as f:
        json.dump(save_data, f, indent=2)

    # FINAL SUMMARY
    print(f"\n{'🏆'*20}")
    print(f"  COMPLETE RESULTS")
    print(f"{'🏆'*20}\n")

    for ds, results in all_results.items():
        trad = results[0]
        compressed = [r for r in results if r['name'] != 'Traditional']
        best = max(compressed, key=lambda x: x['accuracy'])
        best_comp = max(compressed, key=lambda x: trad['params']/max(x['effective_params'],1))

        print(f"📊 {ds}:")
        print(f"  Baseline: {trad['accuracy']:.2f}%")
        print(f"  Best accuracy: {best['name']} → {best['accuracy']:.2f}%")
        print(f"  Best compression: {best_comp['name']} → {trad['params']/best_comp['effective_params']:.0f}x")

        # Find hybrid rank
        hybrids = [r for r in results if 'Hybrid' in r['name']]
        if hybrids:
            best_hybrid = max(hybrids, key=lambda x: x['accuracy'])
            print(f"  Best hybrid: {best_hybrid['name']} → {best_hybrid['accuracy']:.2f}%")
        print()

    print(f"💾 Results: results/complete_ablation.json")
    print(f"📊 Graphs: results/graphs/")
    print(f"\n✅ Complete Ablation Study Done!\n")


if __name__ == "__main__":
    main()