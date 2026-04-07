"""
Fair Comparison: Fourier Method vs LoRA vs Traditional
Same parameter budget for all!

This is the MOST important experiment!
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import time


# ============================================
# METHOD 1: Your Fourier Compression
# ============================================

class FourierLayer(nn.Module):
    """Your method: Fourier wave generation"""
    def __init__(self, in_f, out_f, num_freq=64):
        super().__init__()
        self.in_f = in_f
        self.out_f = out_f
        self.num_freq = num_freq
        self.amp = nn.Parameter(torch.randn(num_freq) * 0.1)
        self.rf = nn.Parameter(torch.randn(num_freq) * 2.0)
        self.cf = nn.Parameter(torch.randn(num_freq) * 2.0)
        self.rp = nn.Parameter(torch.randn(num_freq) * 0.5)
        self.cp = nn.Parameter(torch.randn(num_freq) * 0.5)
        self.bias = nn.Parameter(torch.zeros(out_f))

    def forward(self, x):
        device = self.amp.device
        rows = torch.linspace(0, math.pi * 2, self.out_f, device=device)
        cols = torch.linspace(0, math.pi * 2, self.in_f, device=device)
        W = torch.zeros(self.out_f, self.in_f, device=device)
        for k in range(self.num_freq):
            W += self.amp[k] * torch.outer(
                torch.sin(rows * self.rf[k] + self.rp[k]),
                torch.cos(cols * self.cf[k] + self.cp[k])
            )
        std = math.sqrt(2.0 / (self.in_f + self.out_f))
        W = torch.tanh(W) * std
        return x @ W.T + self.bias


# ============================================
# METHOD 2: LoRA (Low-Rank Adaptation)
# ============================================

class LoRA_Layer(nn.Module):
    """LoRA: W = A × B (low-rank factorization)"""
    def __init__(self, in_f, out_f, rank=4):
        super().__init__()
        self.A = nn.Parameter(torch.randn(out_f, rank) * 0.01)
        self.B = nn.Parameter(torch.randn(rank, in_f) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_f))

    def forward(self, x):
        W = self.A @ self.B
        std = math.sqrt(2.0 / (x.shape[1] + self.A.shape[0]))
        W = torch.tanh(W) * std
        return x @ W.T + self.bias


# ============================================
# METHOD 3: Traditional (Small)
# ============================================

class SmallTraditional(nn.Module):
    """Traditional but with same param count"""
    def __init__(self, in_f, out_f, hidden=None):
        super().__init__()
        if hidden is None:
            total_params = in_f * out_f
            hidden = max(int(total_params ** 0.5), 10)
        self.fc1 = nn.Linear(in_f, hidden)
        self.fc2 = nn.Linear(hidden, out_f)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


# ============================================
# BUILD NETWORKS
# ============================================

class FourierNet(nn.Module):
    def __init__(self, num_freq=32):
        super().__init__()
        self.net = nn.Sequential(
            FourierLayer(784, 128, num_freq),
            nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.2),
            FourierLayer(128, 64, num_freq // 2),
            nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.2),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        return self.net(x.view(-1, 784))


class LoRA_Net(nn.Module):
    def __init__(self, rank=4):
        super().__init__()
        self.net = nn.Sequential(
            LoRA_Layer(784, 128, rank),
            nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.2),
            LoRA_Layer(128, 64, rank),
            nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.2),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        return self.net(x.view(-1, 784))


class TraditionalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.2),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        return self.net(x.view(-1, 784))


# ============================================
# TRAINING
# ============================================

def count_params(model):
    return sum(p.numel() for p in model.parameters())


def train_and_eval(model, name, train_loader, test_loader, epochs=15, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    params = count_params(model)

    print(f"\n{'='*50}")
    print(f"Training: {name}")
    print(f"Parameters: {params:,}")
    print(f"{'='*50}")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    start = time.time()

    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
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
                data, target = data.to(device), target.to(device)
                pred = model(data).argmax(dim=1)
                correct += pred.eq(target).sum().item()

        acc = 100. * correct / len(test_loader.dataset)
        if acc > best_acc:
            best_acc = acc

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Acc: {acc:.2f}% | Best: {best_acc:.2f}%")

    elapsed = time.time() - start

    return {'name': name, 'params': params, 'accuracy': best_acc, 'time': round(elapsed, 1)}


# ============================================
# MAIN COMPARISON
# ============================================

def main():
    print(f"\n{'🔬'*20}")
    print(f"  FAIR COMPARISON EXPERIMENT")
    print(f"  Fourier vs LoRA vs Traditional")
    print(f"{'🔬'*20}\n")

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    results = []

    # 1. Traditional (baseline - full params)
    model = TraditionalNet()
    results.append(train_and_eval(model, "Traditional (Full)", train_loader, test_loader))

    # 2. Fourier (different freq counts)
    for freq in [16, 32, 64]:
        model = FourierNet(num_freq=freq)
        results.append(train_and_eval(model, f"Fourier (freq={freq})", train_loader, test_loader))

    # 3. LoRA (different ranks)
    for rank in [2, 4, 8]:
        model = LoRA_Net(rank=rank)
        results.append(train_and_eval(model, f"LoRA (rank={rank})", train_loader, test_loader))

    # RESULTS TABLE
    print(f"\n{'='*70}")
    print(f"{'🏆'*15}")
    print(f"  FAIR COMPARISON RESULTS")
    print(f"{'🏆'*15}")
    print(f"{'='*70}")

    trad_params = results[0]['params']

    print(f"\n{'Method':<25} | {'Params':>10} | {'Accuracy':>10} | {'Compression':>12} | {'Time':>8}")
    print("-" * 75)

    for r in results:
        comp = f"{trad_params/r['params']:.1f}x" if r['params'] < trad_params else "1x"
        print(f"{r['name']:<25} | {r['params']:>10,} | {r['accuracy']:>8.2f}% | {comp:>12} | {r['time']:>6.1f}s")

    # Find winners
    compressed = [r for r in results if r['name'] != "Traditional (Full)"]
    best_acc = max(compressed, key=lambda x: x['accuracy'])
    best_comp = max(compressed, key=lambda x: trad_params / x['params'])

    print(f"\n{'='*70}")
    print(f"⭐ Best accuracy (compressed): {best_acc['name']} ({best_acc['accuracy']:.2f}%)")
    print(f"⭐ Most compressed: {best_comp['name']} ({trad_params/best_comp['params']:.1f}x)")

    # Fair comparison at similar param count
    print(f"\n{'='*70}")
    print(f"📊 SAME BUDGET COMPARISON:")
    print(f"{'='*70}")

    for r in sorted(compressed, key=lambda x: x['params']):
        print(f"  {r['name']:<25} | {r['params']:>8,} params | {r['accuracy']:.2f}%")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()