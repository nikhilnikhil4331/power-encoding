"""
Power Encoded Neural Network - FIXED
"""

import torch
import torch.nn as nn
from power_layer import PowerEncodedLinear

class PowerEncodedNet(nn.Module):
    def __init__(self, seed_size=32):
        super().__init__()
        
        self.fc1 = PowerEncodedLinear(784, 128, seed_size=seed_size)
        self.fc2 = PowerEncodedLinear(128, 64, seed_size=seed_size//2)
        self.fc3 = PowerEncodedLinear(64, 10, seed_size=seed_size//4)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


class TraditionalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


def compare_models():
    print("\n" + "=" * 70)
    print("📊 MODEL COMPARISON (FIXED VERSION)")
    print("=" * 70)
    
    power_model = PowerEncodedNet(seed_size=32)
    trad_model = TraditionalNet()
    
    power_params = power_model.count_parameters()
    trad_params = trad_model.count_parameters()
    
    print(f"\n🔴 Traditional:  {trad_params:,} parameters")
    print(f"🟢 Power Encoded: {power_params:,} parameters")
    print(f"\n✨ Compression: {trad_params/power_params:.1f}x")
    
    test_input = torch.randn(4, 1, 28, 28)
    with torch.no_grad():
        power_out = power_model(test_input)
        trad_out = trad_model(test_input)
    
    print(f"\n🧪 Test: Input {test_input.shape} → Output {power_out.shape}")
    print(f"   ✅ Both models working!\n")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    compare_models()