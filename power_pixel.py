"""
Power Pixel v2 - Better Accuracy!
More bases + better architecture
"""

import torch
import torch.nn as nn
import math
import time
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class PowerPixelLayer_v2(nn.Module):
    """
    Improved: More bases + position encoding + residual mixing
    """
    
    def __init__(self, in_features, out_features, num_bases=64):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_bases = num_bases
        
        # More bases for better expressivity
        self.bases = nn.Parameter(torch.randn(num_bases) * 0.5)
        
        # Richer power patterns (more params per base)
        self.power_a = nn.Parameter(torch.randn(num_bases) * 0.1)
        self.power_b = nn.Parameter(torch.randn(num_bases) * 0.1)
        self.power_c = nn.Parameter(torch.randn(num_bases) * 0.1)
        self.power_d = nn.Parameter(torch.randn(num_bases) * 0.1)
        
        # Row and column specific patterns
        self.row_pattern = nn.Parameter(torch.randn(num_bases) * 0.1)
        self.col_pattern = nn.Parameter(torch.randn(num_bases) * 0.1)
        
        # Mixing weights (learnable combination)
        self.mix = nn.Parameter(torch.randn(num_bases) * 0.1)
        
        # Bias
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    def generate_weight(self):
        """Generate full weight matrix from bases + powers"""
        
        # Row and column positions
        rows = torch.linspace(0, 1, self.out_features, device=self.bases.device)
        cols = torch.linspace(0, 1, self.in_features, device=self.bases.device)
        
        # Create position grid
        row_grid = rows.unsqueeze(1).expand(self.out_features, self.in_features)
        col_grid = cols.unsqueeze(0).expand(self.out_features, self.in_features)
        
        # Generate weight matrix
        W = torch.zeros(self.out_features, self.in_features, device=self.bases.device)
        
        for i in range(self.num_bases):
            base = torch.sigmoid(self.bases[i]) * 2 + 0.1
            
            # Unique power for EVERY position using row + col
            power = (
                self.power_a[i] * torch.sin(row_grid * self.row_pattern[i] * 10) +
                self.power_b[i] * torch.cos(col_grid * self.col_pattern[i] * 10) +
                self.power_c[i] * torch.sin(row_grid * col_grid * self.power_d[i] * 10)
            )
            
            # Base ^ Power
            generated = torch.pow(base, power)
            
            # Add to weight with mixing
            W = W + self.mix[i] * generated
        
        # Normalize
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        W = torch.tanh(W) * std
        
        return W
    
    def forward(self, x):
        W = self.generate_weight()
        return x @ W.T + self.bias
    
    def count_params(self):
        stored = sum(p.numel() for p in self.parameters())
        generated = self.in_features * self.out_features + self.out_features
        return stored, generated, generated / stored


def main():
    print("\n" + "🔥" * 35)
    print("   POWER PIXEL v2 - MNIST TRAINING")
    print("🔥" * 35)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    
    # Model with MORE bases
    model = nn.Sequential(
        PowerPixelLayer_v2(784, 256, num_bases=128),
        nn.ReLU(),
        nn.Dropout(0.2),
        PowerPixelLayer_v2(256, 128, num_bases=64),
        nn.ReLU(),
        nn.Dropout(0.2),
        PowerPixelLayer_v2(128, 10, num_bases=32)
    ).to(device)
    
    total_stored = sum(p.numel() for p in model.parameters())
    trad_params = 784*256+256 + 256*128+128 + 128*10+10
    
    print(f"\n📊 Model:")
    print(f"   PowerPixel v2 params: {total_stored:,}")
    print(f"   Traditional equiv:    {trad_params:,}")
    print(f"   Compression:          {trad_params/total_stored:.1f}x")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\n🚀 Training 15 epochs...\n")
    
    best_acc = 0
    start = time.time()
    
    for epoch in range(15):
        # Train
        model.train()
        correct = 0
        total = 0
        
        for data, target in train_loader:
            data = data.view(-1, 784).to(device)
            target = target.to(device)
            
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
                data = data.view(-1, 784).to(device)
                target = target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                test_correct += pred.eq(target).sum().item()
        
        test_acc = 100. * test_correct / len(test_loader.dataset)
        train_acc = 100. * correct / total
        
        if test_acc > best_acc:
            best_acc = test_acc
            marker = " ⭐"
        else:
            marker = ""
        
        print(f"Epoch {epoch+1:2d}/15 | Train: {train_acc:.2f}% | Test: {test_acc:.2f}% | Best: {best_acc:.2f}%{marker}")
    
    total_time = time.time() - start
    
    print(f"\n{'='*60}")
    print(f"🏆 RESULTS:")
    print(f"{'='*60}")
    print(f"   Best Accuracy:  {best_acc:.2f}%")
    print(f"   Time:           {total_time:.1f}s")
    print(f"   Stored Params:  {total_stored:,}")
    print(f"   Traditional:    {trad_params:,}")
    print(f"   Compression:    {trad_params/total_stored:.1f}x")
    print(f"\n   Comparison:")
    print(f"   Original Power Encoding: 78.88% with 5,587 params (19.6x)")
    print(f"   Power Pixel v2:          {best_acc:.2f}% with {total_stored:,} params ({trad_params/total_stored:.1f}x)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()