"""
CIFAR-10 Advanced Training
Target: 90%+ Accuracy
Strategy: CNN + Power Encoding Hybrid
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

# Import our Power Encoding
from power_layer import PowerEncodedLinear


class PowerEncodedCIFAR_CNN(nn.Module):
    """
    Hybrid Model: CNN (feature extraction) + Power Encoding (classification)
    
    CNN detects: edges, shapes, textures, colors
    Power Encoding: compresses classification layers
    
    Best of both worlds!
    """
    
    def __init__(self, seed_size=64):
        super().__init__()
        
        # === CNN Feature Extractor (Traditional - needed for images!) ===
        
        # Block 1: 3 → 64 channels
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)  # 32×32 → 16×16
        self.drop1 = nn.Dropout2d(0.2)
        
        # Block 2: 64 → 128 channels
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)  # 16×16 → 8×8
        self.drop2 = nn.Dropout2d(0.3)
        
        # Block 3: 128 → 256 channels
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)  # 8×8 → 4×4
        self.drop3 = nn.Dropout2d(0.4)
        
        # === Power Encoded Classifier (Compressed!) ===
        # After CNN: 256 channels × 4 × 4 = 4096 features
        self.fc1 = PowerEncodedLinear(4096, 512, seed_size=seed_size)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.drop_fc1 = nn.Dropout(0.5)
        
        self.fc2 = PowerEncodedLinear(512, 256, seed_size=seed_size // 2)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.drop_fc2 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(256, 10)  # Final output (small, no need to compress)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # CNN Block 1
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.drop1(self.pool1(x))
        
        # CNN Block 2
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.drop2(self.pool2(x))
        
        # CNN Block 3
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.drop3(self.pool3(x))
        
        # Flatten
        x = x.view(x.size(0), -1)  # [batch, 4096]
        
        # Power Encoded Classifier
        x = self.drop_fc1(self.relu(self.bn_fc1(self.fc1(x))))
        x = self.drop_fc2(self.relu(self.bn_fc2(self.fc2(x))))
        x = self.fc3(x)
        
        return x
    
    def count_cnn_params(self):
        """CNN params (traditional)"""
        cnn_params = 0
        for name, param in self.named_parameters():
            if 'conv' in name or 'bn' in name:
                cnn_params += param.numel()
        return cnn_params
    
    def count_pe_params(self):
        """Power Encoded params (compressed)"""
        pe_params = 0
        for name, param in self.named_parameters():
            if 'fc1' in name or 'fc2' in name:
                pe_params += param.numel()
        return pe_params
    
    def count_total_params(self):
        return sum(p.numel() for p in self.parameters())


def get_cifar10_loaders(batch_size=128):
    """CIFAR-10 with strong data augmentation"""
    
    # Training: heavy augmentation for better generalization
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    # Test: no augmentation
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    train_data = datasets.CIFAR10('./data', train=True, download=True, transform=train_transform)
    test_data = datasets.CIFAR10('./data', train=False, download=True, transform=test_transform)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, test_loader


def train_one_epoch(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        if batch_idx % 100 == 0:
            print(f"   [{batch_idx:3d}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f} | "
                  f"Acc: {100.*correct/total:.2f}%")
    
    return total_loss / len(train_loader), 100. * correct / total


def test(model, device, test_loader):
    model.eval()
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    
    return 100. * correct / len(test_loader.dataset)


def main():
    print("\n" + "🎯" * 35)
    print("   CIFAR-10 ADVANCED TRAINING")
    print("   Target: 90%+ Accuracy!")
    print("🎯" * 35)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Data
    print("\n📥 Loading CIFAR-10...")
    train_loader, test_loader = get_cifar10_loaders(batch_size=128)
    print("✅ Loaded: 50,000 train, 10,000 test")
    
    # Model
    print("\n🔨 Building CNN + Power Encoding model...")
    model = PowerEncodedCIFAR_CNN(seed_size=64)
    model = model.to(device)
    
    total_params = model.count_total_params()
    cnn_params = model.count_cnn_params()
    pe_params = model.count_pe_params()
    
    print(f"\n📊 Model Statistics:")
    print(f"   Total Parameters:    {total_params:,}")
    print(f"   CNN Parameters:      {cnn_params:,} (feature extraction)")
    print(f"   PE Parameters:       {pe_params:,} (compressed classifier)")
    print(f"   Other Parameters:    {total_params - cnn_params - pe_params:,}")
    
    # Traditional equivalent
    trad_fc_params = 4096 * 512 + 512 * 256 + 256 * 10  # If all FC layers were traditional
    print(f"\n✨ Classifier Compression:")
    print(f"   Traditional FC:     {trad_fc_params:,} params")
    print(f"   Power Encoded FC:   {pe_params:,} params")
    print(f"   Compression:        {trad_fc_params / max(pe_params, 1):.1f}x")
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    
    criterion = nn.CrossEntropyLoss()
    
    # Training
    epochs = 50
    print(f"\n🚀 Training for {epochs} epochs...")
    print("=" * 70)
    
    best_acc = 0
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        print(f"\n📚 Epoch {epoch}/{epochs} (LR: {scheduler.get_last_lr()[0]:.6f})")
        print("-" * 70)
        
        train_loss, train_acc = train_one_epoch(
            model, device, train_loader, optimizer, criterion, epoch
        )
        
        test_acc = test(model, device, test_loader)
        
        scheduler.step()
        
        # Save best
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'cifar10_best_model.pth')
        
        print(f"\n📊 Epoch {epoch}: Train={train_acc:.2f}% | Test={test_acc:.2f}% | Best={best_acc:.2f}%")
        
        # Early stopping check
        if best_acc >= 90:
            print("\n🎉 TARGET REACHED: 90%+ Accuracy!")
            break
    
    total_time = time.time() - start_time
    
    # Final Results
    print("\n" + "=" * 70)
    print("🏆 FINAL RESULTS")
    print("=" * 70)
    print(f"   Best Accuracy:    {best_acc:.2f}%")
    print(f"   Total Time:       {total_time/60:.1f} minutes")
    print(f"   Total Parameters: {total_params:,}")
    print(f"   FC Compression:   {trad_fc_params / max(pe_params, 1):.1f}x")
    print(f"   Model saved:      cifar10_best_model.pth")
    
    # Class-wise accuracy
    print("\n📊 Class-wise Results:")
    classes = ['Airplane', 'Car', 'Bird', 'Cat', 'Deer', 
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    
    model.load_state_dict(torch.load('cifar10_best_model.pth'))
    model.eval()
    
    class_correct = [0] * 10
    class_total = [0] * 10
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            
            for i in range(len(target)):
                label = target[i].item()
                class_total[label] += 1
                if pred[i].item() == label:
                    class_correct[label] += 1
    
    for i in range(10):
        acc = 100 * class_correct[i] / max(class_total[i], 1)
        emoji = "✅" if acc >= 85 else "⚠️" if acc >= 70 else "❌"
        print(f"   {emoji} {classes[i]:>10}: {acc:.1f}%")
    
    print("\n" + "=" * 70)
    print(f"✅ TRAINING COMPLETE! Best: {best_acc:.2f}%")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()