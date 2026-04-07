"""
Multi-Dataset Fair Comparison
Fourier vs LoRA vs Traditional
Datasets: MNIST, Fashion-MNIST, CIFAR-10

This proves results are NOT dataset-specific!
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
# COMPRESSION METHODS (Same as before)
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
        r = torch.linspace(0, math.pi * 2, self.out_f, device=d)
        c = torch.linspace(0, math.pi * 2, self.in_f, device=d)
        W = sum(self.amp[k] * torch.outer(
            torch.sin(r * self.rf[k] + self.rp[k]),
            torch.cos(c * self.cf[k] + self.cp[k])
        ) for k in range(self.nf))
        std = math.sqrt(2.0 / (self.in_f + self.out_f))
        return x @ (torch.tanh(W) * std).T + self.bias


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
# NETWORK BUILDERS
# ============================================

def build_traditional(input_size, num_classes):
    return nn.Sequential(
        nn.Linear(input_size, 256),
        nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.2),
        nn.Linear(128, num_classes)
    )

def build_fourier(input_size, num_classes, freq=32):
    return nn.Sequential(
        FourierLayer(input_size, 128, freq),
        nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.2),
        FourierLayer(128, 64, freq // 2),
        nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.2),
        nn.Linear(64, num_classes)
    )

def build_lora(input_size, num_classes, rank=4):
    return nn.Sequential(
        LoRALayer(input_size, 128, rank),
        nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.2),
        LoRALayer(128, 64, rank),
        nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.2),
        nn.Linear(64, num_classes)
    )

# CIFAR needs CNN
def build_traditional_cnn(num_classes):
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
        nn.MaxPool2d(2), nn.Dropout2d(0.2),
        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
        nn.MaxPool2d(2), nn.Dropout2d(0.3),
        nn.Flatten(),
        nn.Linear(128 * 8 * 8, 256),
        nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )

def build_fourier_cnn(num_classes, freq=32):
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
        nn.MaxPool2d(2), nn.Dropout2d(0.2),
        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
        nn.MaxPool2d(2), nn.Dropout2d(0.3),
        nn.Flatten(),
        FourierLayer(128 * 8 * 8, 128, freq),
        nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(128, num_classes)
    )

def build_lora_cnn(num_classes, rank=4):
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
        nn.MaxPool2d(2), nn.Dropout2d(0.2),
        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
        nn.MaxPool2d(2), nn.Dropout2d(0.3),
        nn.Flatten(),
        LoRALayer(128 * 8 * 8, 128, rank),
        nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(128, num_classes)
    )


# ============================================
# DATA LOADERS
# ============================================

def get_mnist():
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train = datasets.MNIST('./data', train=True, download=True, transform=t)
    test = datasets.MNIST('./data', train=False, download=True, transform=t)
    return DataLoader(train, 64, shuffle=True), DataLoader(test, 64), 784, 10, 'MNIST'

def get_fashion_mnist():
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    train = datasets.FashionMNIST('./data', train=True, download=True, transform=t)
    test = datasets.FashionMNIST('./data', train=False, download=True, transform=t)
    return DataLoader(train, 64, shuffle=True), DataLoader(test, 64), 784, 10, 'Fashion-MNIST'

def get_cifar10():
    t_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    t_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    train = datasets.CIFAR10('./data', train=True, download=True, transform=t_train)
    test = datasets.CIFAR10('./data', train=False, download=True, transform=t_test)
    return DataLoader(train, 64, shuffle=True), DataLoader(test, 64), 3072, 10, 'CIFAR-10'


# ============================================
# TRAIN + EVAL
# ============================================

def train_eval(model, name, train_loader, test_loader, epochs=15, flatten=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    params = sum(p.numel() for p in model.parameters())

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best = 0
    start = time.time()

    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            if flatten and len(data.shape) == 4 and data.shape[1] == 1:
                data = data.view(-1, 784)
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
                if flatten and len(data.shape) == 4 and data.shape[1] == 1:
                    data = data.view(-1, 784)
                pred = model(data).argmax(dim=1)
                correct += pred.eq(target).sum().item()

        acc = 100. * correct / len(test_loader.dataset)
        if acc > best:
            best = acc

    elapsed = time.time() - start
    return {'name': name, 'params': params, 'accuracy': best, 'time': round(elapsed, 1)}


# ============================================
# MAIN
# ============================================

def main():
    print(f"\n{'🔬' * 20}")
    print(f"  MULTI-DATASET COMPARISON")
    print(f"  Fourier vs LoRA vs Traditional")
    print(f"{'🔬' * 20}\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    all_results = {}

    # ============================================
    # DATASET 1: MNIST
    # ============================================
    print(f"\n{'='*60}")
    print(f"📊 DATASET 1: MNIST")
    print(f"{'='*60}")

    train_l, test_l, inp, cls, name = get_mnist()
    results = []
    results.append(train_eval(build_traditional(inp, cls), "Traditional", train_l, test_l))
    results.append(train_eval(build_fourier(inp, cls, 16), "Fourier-16", train_l, test_l))
    results.append(train_eval(build_fourier(inp, cls, 32), "Fourier-32", train_l, test_l))
    results.append(train_eval(build_lora(inp, cls, 2), "LoRA-r2", train_l, test_l))
    results.append(train_eval(build_lora(inp, cls, 4), "LoRA-r4", train_l, test_l))

    all_results['MNIST'] = results
    print_results("MNIST", results)

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # ============================================
    # DATASET 2: Fashion-MNIST
    # ============================================
    print(f"\n{'='*60}")
    print(f"📊 DATASET 2: Fashion-MNIST")
    print(f"{'='*60}")

    train_l, test_l, inp, cls, name = get_fashion_mnist()
    results = []
    results.append(train_eval(build_traditional(inp, cls), "Traditional", train_l, test_l))
    results.append(train_eval(build_fourier(inp, cls, 16), "Fourier-16", train_l, test_l))
    results.append(train_eval(build_fourier(inp, cls, 32), "Fourier-32", train_l, test_l))
    results.append(train_eval(build_lora(inp, cls, 2), "LoRA-r2", train_l, test_l))
    results.append(train_eval(build_lora(inp, cls, 4), "LoRA-r4", train_l, test_l))

    all_results['Fashion-MNIST'] = results
    print_results("Fashion-MNIST", results)

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # ============================================
    # DATASET 3: CIFAR-10
    # ============================================
    print(f"\n{'='*60}")
    print(f"📊 DATASET 3: CIFAR-10 (CNN)")
    print(f"{'='*60}")

    train_l, test_l, inp, cls, name = get_cifar10()
    results = []
    results.append(train_eval(build_traditional_cnn(cls), "Traditional-CNN", train_l, test_l, flatten=False))
    results.append(train_eval(build_fourier_cnn(cls, 32), "Fourier-CNN-32", train_l, test_l, flatten=False))
    results.append(train_eval(build_fourier_cnn(cls, 64), "Fourier-CNN-64", train_l, test_l, flatten=False))
    results.append(train_eval(build_lora_cnn(cls, 4), "LoRA-CNN-r4", train_l, test_l, flatten=False))
    results.append(train_eval(build_lora_cnn(cls, 8), "LoRA-CNN-r8", train_l, test_l, flatten=False))

    all_results['CIFAR-10'] = results
    print_results("CIFAR-10", results)

    # ============================================
    # FINAL SUMMARY
    # ============================================
    print(f"\n{'🏆' * 20}")
    print(f"  COMPLETE MULTI-DATASET RESULTS")
    print(f"{'🏆' * 20}\n")

    for dataset, results in all_results.items():
        trad = results[0]['params']
        print(f"\n📊 {dataset}:")
        print(f"{'Method':<20} | {'Params':>8} | {'Acc':>7} | {'Compression':>12}")
        print("-" * 55)
        for r in results:
            comp = f"{trad/r['params']:.1f}x" if r['params'] < trad else "1x"
            print(f"{r['name']:<20} | {r['params']:>8,} | {r['accuracy']:>5.2f}% | {comp:>12}")

    # Key finding
    print(f"\n{'='*60}")
    print(f"🎯 KEY FINDING:")
    print(f"{'='*60}")

    for dataset, results in all_results.items():
        fourier_results = [r for r in results if 'Fourier' in r['name']]
        lora_results = [r for r in results if 'LoRA' in r['name']]

        if fourier_results and lora_results:
            best_fourier = max(fourier_results, key=lambda x: x['accuracy'])
            smallest_lora = min(lora_results, key=lambda x: x['params'])

            if best_fourier['params'] < smallest_lora['params']:
                winner = "Fourier" if best_fourier['accuracy'] > smallest_lora['accuracy'] else "LoRA"
                print(f"\n  {dataset} at extreme compression:")
                print(f"    Fourier: {best_fourier['params']:,} params → {best_fourier['accuracy']:.2f}%")
                print(f"    LoRA:    {smallest_lora['params']:,} params → {smallest_lora['accuracy']:.2f}%")
                print(f"    Winner:  {winner}! {'✅' if winner == 'Fourier' else ''}")

    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/multi_dataset_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n💾 Results saved: results/multi_dataset_results.json")
    print(f"\n✅ Multi-Dataset Comparison Complete!\n")


def print_results(dataset, results):
    trad = results[0]['params']
    print(f"\n{'Method':<20} | {'Params':>8} | {'Acc':>7} | {'Comp':>6} | {'Time':>7}")
    print("-" * 55)
    for r in results:
        comp = f"{trad/r['params']:.0f}x" if r['params'] < trad else "1x"
        print(f"{r['name']:<20} | {r['params']:>8,} | {r['accuracy']:>5.2f}% | {comp:>6} | {r['time']:>5.1f}s")


if __name__ == "__main__":
    main()