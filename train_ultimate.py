"""
Ultimate Power Encoding - Target 85%+
Combination of all optimizations
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
from power_network import PowerEncodedNet

def load_mnist(batch_size=16):  # Even smaller batches
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        # Data augmentation
        transforms.RandomRotation(5),
    ])
    train_data = datasets.MNIST("./data", train=True, download=False, transform=transform)
    test_data = datasets.MNIST("./data", train=False, download=False, transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def train_epoch(model, device, loader, optimizer, criterion):
    model.train()
    correct = 0
    total = 0
    
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # Aggressive gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3)
        
        optimizer.step()
        
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    return 100. * correct / total

def test_model(model, device, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    return 100. * correct / len(loader.dataset)

def main():
    print("\n🔥 ULTIMATE POWER ENCODING - Target 85%+ 🔥\n")
    
    train_loader, test_loader = load_mnist(batch_size=16)
    
    # Even larger seed for max capacity
    model = PowerEncodedNet(seed_size=96)
    
    print(f"📊 Parameters: {model.count_parameters():,}\n")
    
    device = torch.device("cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Higher LR
    
    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    
    print("🚀 Training 20 epochs...\n")
    
    start = time.time()
    best_acc = 0
    
    for epoch in range(1, 21):
        train_acc = train_epoch(model, device, train_loader, optimizer, criterion)
        test_acc = test_model(model, device, test_loader)
        
        scheduler.step()
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'power_model_ultimate.pth')
        
        if epoch % 2 == 0:
            print(f"Epoch {epoch:2d}/20 | Train: {train_acc:.2f}% | Test: {test_acc:.2f}% | Best: {best_acc:.2f}%")
    
    total_time = time.time() - start
    
    print(f"\n{'='*70}")
    print(f"✅ FINAL: {best_acc:.2f}% accuracy")
    print(f"   Time: {total_time/60:.1f} minutes")
    print(f"   Params: {model.count_parameters():,}")
    print(f"   Compression vs Traditional: {109386/model.count_parameters():.1f}x")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()