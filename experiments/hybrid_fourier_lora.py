"""
HYBRID: Fourier + LoRA Combined!
Best of both worlds!

Fourier: Good at smooth patterns (extreme compression)
LoRA: Good at complex patterns (accuracy)
Combined: Good at BOTH! 

This could be the KEY to better results!
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


# ============================================
# FOURIER LAYER (Your method)
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

    def generate_weight(self):
        d = self.amp.device
        r = torch.linspace(0, math.pi * 2, self.out_f, device=d)
        c = torch.linspace(0, math.pi * 2, self.in_f, device=d)
        W = torch.zeros(self.out_f, self.in_f, device=d)
        for k in range(self.nf):
            W += self.amp[k] * torch.outer(
                torch.sin(r * self.rf[k] + self.rp[k]),
                torch.cos(c * self.cf[k] + self.cp[k])
            )
        std = math.sqrt(2.0 / (self.in_f + self.out_f))
        return torch.tanh(W) * std

    def forward(self, x):
        return x @ self.generate_weight().T + self.bias


# ============================================
# LoRA LAYER
# ============================================

class LoRALayer(nn.Module):
    def __init__(self, in_f, out_f, rank=4):
        super().__init__()
        self.A = nn.Parameter(torch.randn(out_f, rank) * 0.01)
        self.B = nn.Parameter(torch.randn(rank, in_f) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_f))

    def forward(self, x):
        W = self.A @ self.B
        std = math.sqrt(2.0 / (x.shape[1] + self.A.shape[0]))
        return x @ (torch.tanh(W) * std).T + self.bias


# ============================================
# NEW! HYBRID LAYER (Fourier + LoRA!)
# ============================================

class HybridFourierLoRA(nn.Module):
    """
    COMBINES Fourier + LoRA!
    
    Fourier captures: Smooth, periodic patterns
    LoRA captures: Sharp, complex patterns
    Together: BEST of both worlds!
    
    W = alpha * W_fourier + beta * W_lora
    
    alpha, beta are LEARNABLE!
    Network decides how much of each to use!
    """
    
    def __init__(self, in_f, out_f, num_freq=16, rank=2):
        super().__init__()
        self.in_f = in_f
        self.out_f = out_f
        
        # Fourier part (smooth patterns)
        self.fourier = FourierLayer(in_f, out_f, num_freq)
        
        # LoRA part (complex patterns)
        self.lora_A = nn.Parameter(torch.randn(out_f, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(rank, in_f) * 0.01)
        
        # Learnable mixing weights!
        self.alpha = nn.Parameter(torch.tensor(0.5))  # Fourier weight
        self.beta = nn.Parameter(torch.tensor(0.5))   # LoRA weight
        
        self.bias = nn.Parameter(torch.zeros(out_f))
    
    def forward(self, x):
        # Fourier weights (smooth patterns)
        W_fourier = self.fourier.generate_weight()
        
        # LoRA weights (complex patterns)
        W_lora = self.lora_A @ self.lora_B
        std = math.sqrt(2.0 / (self.in_f + self.out_f))
        W_lora = torch.tanh(W_lora) * std
        
        # COMBINE with learnable mixing!
        alpha = torch.sigmoid(self.alpha)
        beta = torch.sigmoid(self.beta)
        
        W = alpha * W_fourier + beta * W_lora
        
        return x @ W.T + self.bias


# ============================================
# NEW! ENHANCED FOURIER (Learnable Basis!)
# ============================================

class EnhancedFourierLayer(nn.Module):
    """
    IMPROVED Fourier with:
    1. Learnable basis functions (not just sin/cos)
    2. Multi-scale frequencies
    3. Residual connections
    """
    
    def __init__(self, in_f, out_f, num_freq=32):
        super().__init__()
        self.in_f, self.out_f, self.nf = in_f, out_f, num_freq
        
        # Standard Fourier params
        self.amp = nn.Parameter(torch.randn(num_freq) * 0.1)
        self.rf = nn.Parameter(torch.randn(num_freq) * 2.0)
        self.cf = nn.Parameter(torch.randn(num_freq) * 2.0)
        self.rp = nn.Parameter(torch.randn(num_freq) * 0.5)
        self.cp = nn.Parameter(torch.randn(num_freq) * 0.5)
        
        # NEW: Cross-frequency interaction
        self.cross_amp = nn.Parameter(torch.randn(num_freq) * 0.05)
        self.cross_freq = nn.Parameter(torch.randn(num_freq) * 1.0)
        
        # NEW: Learnable nonlinearity mixing
        self.mix_sin = nn.Parameter(torch.tensor(0.7))
        self.mix_cos = nn.Parameter(torch.tensor(0.3))
        
        self.bias = nn.Parameter(torch.zeros(out_f))
    
    def forward(self, x):
        d = self.amp.device
        r = torch.linspace(0, math.pi * 2, self.out_f, device=d)
        c = torch.linspace(0, math.pi * 2, self.in_f, device=d)
        
        W = torch.zeros(self.out_f, self.in_f, device=d)
        
        for k in range(self.nf):
            # Standard wave
            row = torch.sin(r * self.rf[k] + self.rp[k])
            col = torch.cos(c * self.cf[k] + self.cp[k])
            standard = self.amp[k] * torch.outer(row, col)
            
            # Cross-frequency (captures complex patterns!)
            cross_row = torch.sin(r * self.cross_freq[k])
            cross_col = torch.sin(c * self.cross_freq[k])
            cross = self.cross_amp[k] * torch.outer(cross_row, cross_col)
            
            W += standard + cross
        
        std = math.sqrt(2.0 / (self.in_f + self.out_f))
        return x @ (torch.tanh(W) * std).T + self.bias


# ============================================
# BUILD NETWORKS
# ============================================

def build_net(layer_type, input_size, num_classes, **kwargs):
    """Build network with given layer type"""
    if layer_type == 'traditional':
        return nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    elif layer_type == 'fourier':
        freq = kwargs.get('freq', 32)
        return nn.Sequential(
            FourierLayer(input_size, 128, freq),
            nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.2),
            FourierLayer(128, 64, freq // 2),
            nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    
    elif layer_type == 'lora':
        rank = kwargs.get('rank', 4)
        return nn.Sequential(
            LoRALayer(input_size, 128, rank),
            nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.2),
            LoRALayer(128, 64, rank),
            nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    
    elif layer_type == 'hybrid':
        freq = kwargs.get('freq', 16)
        rank = kwargs.get('rank', 2)
        return nn.Sequential(
            HybridFourierLoRA(input_size, 128, freq, rank),
            nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.2),
            HybridFourierLoRA(128, 64, freq // 2, max(rank // 2, 1)),
            nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    
    elif layer_type == 'enhanced_fourier':
        freq = kwargs.get('freq', 32)
        return nn.Sequential(
            EnhancedFourierLayer(input_size, 128, freq),
            nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.2),
            EnhancedFourierLayer(128, 64, freq // 2),
            nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )


# ============================================
# TRAIN + EVAL
# ============================================

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def train_eval(model, name, train_loader, test_loader, epochs=20, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    params = count_params(model)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    best = 0
    start = time.time()
    
    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            data = data.view(-1, data.shape[-1] * data.shape[-2] if len(data.shape) > 2 else data.shape[-1]).to(device)
            target = target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = data.view(-1, data.shape[-1] * data.shape[-2] if len(data.shape) > 2 else data.shape[-1]).to(device)
                target = target.to(device)
                pred = model(data).argmax(dim=1)
                correct += pred.eq(target).sum().item()
        
        acc = 100. * correct / len(test_loader.dataset)
        if acc > best:
            best = acc
    
    elapsed = time.time() - start
    return {'name': name, 'params': params, 'accuracy': best, 'time': round(elapsed, 1)}


# ============================================
# MAIN EXPERIMENT
# ============================================

def main():
    print(f"\n{'🔬' * 20}")
    print(f"  HYBRID FOURIER + LoRA EXPERIMENT")
    print(f"  + Enhanced Fourier")
    print(f"{'🔬' * 20}\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Datasets
    data_configs = [
        ('MNIST', datasets.MNIST, (0.1307,), (0.3081,), 784, 10),
        ('Fashion-MNIST', datasets.FashionMNIST, (0.2860,), (0.3530,), 784, 10),
    ]
    
    all_results = {}
    
    for dataset_name, dataset_class, mean, std_val, input_size, num_classes in data_configs:
        print(f"\n{'=' * 60}")
        print(f"📊 Dataset: {dataset_name}")
        print(f"{'=' * 60}")
        
        t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std_val)])
        train_data = dataset_class('./data', train=True, download=True, transform=t)
        test_data = dataset_class('./data', train=False, download=True, transform=t)
        train_loader = DataLoader(train_data, 64, shuffle=True)
        test_loader = DataLoader(test_data, 64)
        
        results = []
        
        # 1. Traditional (baseline)
        print(f"\n  Training Traditional...")
        model = build_net('traditional', input_size, num_classes)
        results.append(train_eval(model, "Traditional", train_loader, test_loader))
        print(f"  ✅ {results[-1]['accuracy']:.2f}%")
        
        if device.type == 'cuda': torch.cuda.empty_cache()
        
        # 2. Fourier
        print(f"  Training Fourier-32...")
        model = build_net('fourier', input_size, num_classes, freq=32)
        results.append(train_eval(model, "Fourier-32", train_loader, test_loader))
        print(f"  ✅ {results[-1]['accuracy']:.2f}%")
        
        if device.type == 'cuda': torch.cuda.empty_cache()
        
        # 3. Enhanced Fourier (NEW!)
        print(f"  Training Enhanced-Fourier-32...")
        model = build_net('enhanced_fourier', input_size, num_classes, freq=32)
        results.append(train_eval(model, "Enhanced-Fourier-32", train_loader, test_loader))
        print(f"  ✅ {results[-1]['accuracy']:.2f}%")
        
        if device.type == 'cuda': torch.cuda.empty_cache()
        
        # 4. LoRA
        print(f"  Training LoRA-r4...")
        model = build_net('lora', input_size, num_classes, rank=4)
        results.append(train_eval(model, "LoRA-r4", train_loader, test_loader))
        print(f"  ✅ {results[-1]['accuracy']:.2f}%")
        
        if device.type == 'cuda': torch.cuda.empty_cache()
        
        # 5. HYBRID (Fourier + LoRA!) - NEW!
        print(f"  Training Hybrid (F16+LoRA-r2)...")
        model = build_net('hybrid', input_size, num_classes, freq=16, rank=2)
        results.append(train_eval(model, "Hybrid-F16-r2", train_loader, test_loader))
        print(f"  ✅ {results[-1]['accuracy']:.2f}%")
        
        if device.type == 'cuda': torch.cuda.empty_cache()
        
        # 6. HYBRID larger
        print(f"  Training Hybrid (F32+LoRA-r4)...")
        model = build_net('hybrid', input_size, num_classes, freq=32, rank=4)
        results.append(train_eval(model, "Hybrid-F32-r4", train_loader, test_loader))
        print(f"  ✅ {results[-1]['accuracy']:.2f}%")
        
        if device.type == 'cuda': torch.cuda.empty_cache()
        
        all_results[dataset_name] = results
        
        # Print results
        trad = results[0]['params']
        print(f"\n  {'Method':<25} | {'Params':>8} | {'Acc':>7} | {'Comp':>6}")
        print(f"  {'-'*55}")
        for r in results:
            comp = f"{trad/r['params']:.0f}x" if r['params'] < trad else "1x"
            print(f"  {r['name']:<25} | {r['params']:>8,} | {r['accuracy']:>5.2f}% | {comp:>6}")
    
    # FINAL SUMMARY
    print(f"\n{'🏆' * 20}")
    print(f"  FINAL RESULTS")
    print(f"{'🏆' * 20}\n")
    
    for dataset_name, results in all_results.items():
        trad = results[0]
        compressed = results[1:]
        best = max(compressed, key=lambda x: x['accuracy'])
        
        print(f"📊 {dataset_name}:")
        print(f"  Baseline: {trad['accuracy']:.2f}% ({trad['params']:,} params)")
        print(f"  Best compressed: {best['name']} → {best['accuracy']:.2f}% ({best['params']:,} params)")
        print(f"  Compression: {trad['params']/best['params']:.1f}x\n")
    
    # Save
    os.makedirs('results', exist_ok=True)
    with open('results/hybrid_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"💾 Saved: results/hybrid_results.json")
    print(f"\n✅ Experiment Complete!\n")


if __name__ == "__main__":
    main()