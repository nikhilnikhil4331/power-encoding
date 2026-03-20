"""
Improved Power Encoding Training
Target: 80%+ accuracy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
from power_network import PowerEncodedNet

def load_mnist(batch_size=32):  # Smaller batch for better gradients
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_data = datasets.MNIST("./data", train=True, download=False, transform=transform)
    test_data = datasets.MNIST("./data", train=False, download=False, transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def train_epoch(model, device, loader, optimizer, criterion, scheduler):
    model.train()
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        
        optimizer.step()
        
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        if batch_idx % 300 == 0:
            print(f'   [{batch_idx:4d}] Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}%')
    
    scheduler.step()
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
    print("\n" + "🚀" * 35)
    print("   IMPROVED POWER ENCODING TRAINING")
    print("🚀" * 35 + "\n")
    
    train_loader, test_loader = load_mnist(batch_size=32)
    
    # Larger seed for better capacity
    model = PowerEncodedNet(seed_size=64)  # 64 instead of 32
    
    print(f"📊 Parameters: {model.count_parameters():,}")
    
    device = torch.device("cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Higher LR
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    print("\n🚀 TRAINING (10 epochs)...\n")
    
    start = time.time()
    best_acc = 0
    
    for epoch in range(1, 11):
        print(f"📚 Epoch {epoch}/10")
        print("-" * 70)
        
        train_acc = train_epoch(model, device, train_loader, optimizer, criterion, scheduler)
        test_acc = test_model(model, device, test_loader)
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'power_model_best.pth')
        
        print(f"📊 Train: {train_acc:.2f}% | Test: {test_acc:.2f}% | Best: {best_acc:.2f}%\n")
    
    total_time = time.time() - start
    
    print("=" * 70)
    print(f"✅ TRAINING COMPLETE!")
    print(f"   Best accuracy: {best_acc:.2f}%")
    print(f"   Total time: {total_time:.1f}s")
    print(f"   Parameters: {model.count_parameters():,}")
    print("=" * 70)

if __name__ == "__main__":
    main()