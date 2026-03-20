"""
Train Power Encoded Network - FIXED VERSION
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
from power_network import PowerEncodedNet, TraditionalNet


def load_mnist(batch_size=64):
    print("📥 Loading MNIST...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_data = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_data = datasets.MNIST("./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    print(f"✅ Loaded: {len(train_data):,} train, {len(test_data):,} test\n")
    return train_loader, test_loader


def train_epoch(model, device, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        if batch_idx % 200 == 0:
            print(f"   [{batch_idx:3d}/{len(loader)}] Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}%")
    
    return total_loss / len(loader), 100. * correct / total


def test_model(model, device, loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    
    return test_loss / len(loader), 100. * correct / len(loader.dataset)


def train_full(model, name, train_loader, test_loader, epochs=5):
    print("\n" + "=" * 75)
    print(f"🚀 TRAINING: {name}")
    print("=" * 75)
    
    device = torch.device("cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Lower learning rate for Power Encoded
    lr = 0.0001 if "Power" in name else 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    start = time.time()
    
    for epoch in range(1, epochs + 1):
        print(f"\n📚 Epoch {epoch}/{epochs}")
        print("-" * 75)
        
        train_loss, train_acc = train_epoch(model, device, train_loader, optimizer, criterion)
        test_loss, test_acc = test_model(model, device, test_loader, criterion)
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        
        print(f"\n📊 Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}% | Test: Loss={test_loss:.4f}, Acc={test_acc:.2f}%")
    
    total_time = time.time() - start
    print("\n" + "=" * 75)
    print(f"✅ Done! Time: {total_time:.1f}s | Final: {history['test_acc'][-1]:.2f}%")
    print("=" * 75)
    
    return model, history, total_time


def main():
    print("\n" + "🎯" * 37)
    print("    POWER ENCODING - FIXED VERSION")
    print("🎯" * 37 + "\n")
    
    train_loader, test_loader = load_mnist()
    
    print("🔨 Creating models...")
    power_model = PowerEncodedNet(seed_size=32)
    trad_model = TraditionalNet()
    
    power_params = power_model.count_parameters()
    trad_params = trad_model.count_parameters()
    
    print(f"\n📊 PARAMETERS:")
    print(f"   Power:      {power_params:,}")
    print(f"   Traditional: {trad_params:,}")
    print(f"   Compression: {trad_params/power_params:.1f}x")
    
    # Train with more epochs
    power_model, power_hist, power_time = train_full(power_model, "Power Encoded", train_loader, test_loader, epochs=5)
    trad_model, trad_hist, trad_time = train_full(trad_model, "Traditional", train_loader, test_loader, epochs=3)
    
    print("\n🏆 FINAL RESULTS")
    print(f"🎯 Accuracy: Power={power_hist['test_acc'][-1]:.2f}%, Trad={trad_hist['test_acc'][-1]:.2f}%")
    print(f"⚡ Time: Power={power_time:.1f}s, Trad={trad_time:.1f}s")
    
    torch.save(power_model.state_dict(), "power_model_fixed.pth")
    torch.save(trad_model.state_dict(), "trad_model_fixed.pth")
    print("\n💾 Models saved!\n✅ COMPLETE!\n")


if __name__ == "__main__":
    main()