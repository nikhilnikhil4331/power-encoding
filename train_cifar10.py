"""
Test Power Encoding on CIFAR-10
(Harder than MNIST - color images!)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from power_layer import PowerEncodedLinear
import time

class PowerEncodedCIFAR(nn.Module):
    def __init__(self, seed_size=64):
        super().__init__()
        # CIFAR-10: 32x32x3 = 3072 input
        self.fc1 = PowerEncodedLinear(3072, 512, seed_size=seed_size)
        self.fc2 = PowerEncodedLinear(512, 256, seed_size=seed_size//2)
        self.fc3 = PowerEncodedLinear(256, 10, seed_size=seed_size//4)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = x.view(-1, 3072)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

def main():
    print("🎯 CIFAR-10 Power Encoding Test\n")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    
    model = PowerEncodedCIFAR(seed_size=64)
    params = model.count_parameters() if hasattr(model, 'count_parameters') else sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}\n")
    
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(5):
        model.train()
        correct = 0
        total = 0
        
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
        
        # Test
        model.eval()
        test_correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                pred = output.argmax(dim=1)
                test_correct += pred.eq(target).sum().item()
        
        test_acc = 100. * test_correct / len(test_loader.dataset)
        print(f"Epoch {epoch+1}: Train={100.*correct/total:.2f}%, Test={test_acc:.2f}%")
    
    print(f"\n✅ Final: {test_acc:.2f}% on CIFAR-10")
    print("Push results to GitHub!")

if __name__ == "__main__":
    main()