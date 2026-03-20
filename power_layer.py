"""
Power Encoding Layer - FIXED VERSION
"""

import torch
import torch.nn as nn
import math

class PowerEncodedLinear(nn.Module):
    def __init__(self, in_features, out_features, seed_size=32, num_powers=3):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.seed_size = seed_size
        
        # Initialize with better values
        self.seed = nn.Parameter(torch.randn(seed_size, seed_size) * 0.1)
        self.powers = nn.Parameter(torch.tensor([1.0, 1.5, 2.0])[:num_powers])
        self.bias = nn.Parameter(torch.zeros(out_features))
        
    def generate_weight(self):
        """Generate weight matrix from seed"""
        W = self.seed
        
        # Simpler expansion for better gradients
        for i, power in enumerate(self.powers):
            # Power transformation
            powered = torch.sign(W) * torch.pow(torch.abs(W) + 1e-8, power)
            
            # Expand using Kronecker product
            expand_size = min(8, self.seed_size)
            W = torch.kron(W, powered[:expand_size, :expand_size])
            
            # Check if we have enough size
            if W.shape[0] >= self.out_features and W.shape[1] >= self.in_features:
                break
        
        # Trim to exact size
        W = W[:self.out_features, :self.in_features]
        
        # Better normalization
        fan_in = self.in_features
        fan_out = self.out_features
        std = math.sqrt(2.0 / (fan_in + fan_out))
        
        # Normalize
        W = W / (torch.std(W) + 1e-8) * std
        
        return W
    
    def forward(self, x):
        """
        FIXED: Generate weights fresh every time
        This allows gradients to flow back to seed
        """
        W = self.generate_weight()  # Always fresh!
        output = torch.matmul(x, W.T) + self.bias
        return output
    
    def count_parameters(self):
        seed_params = self.seed.numel()
        power_params = self.powers.numel()
        bias_params = self.bias.numel()
        total_stored = seed_params + power_params + bias_params
        total_generated = self.in_features * self.out_features + bias_params
        
        return {
            'seed': seed_params,
            'powers': power_params,
            'bias': bias_params,
            'total_stored': total_stored,
            'total_generated': total_generated,
            'compression_ratio': total_generated / total_stored
        }


def test_power_layer():
    print("\n" + "=" * 70)
    print("🧪 TESTING POWER ENCODED LAYER (FIXED VERSION)")
    print("=" * 70)
    
    layer = PowerEncodedLinear(in_features=784, out_features=128, seed_size=32)
    params = layer.count_parameters()
    
    print(f"\n📊 PARAMETER COMPARISON:")
    print(f"   Traditional: {params['total_generated']:,} parameters")
    print(f"   Power Encoded: {params['total_stored']:,} parameters")
    print(f"   Compression: {params['compression_ratio']:.1f}x")
    
    print(f"\n🚀 FORWARD PASS TEST:")
    test_input = torch.randn(16, 784)
    output = layer(test_input)
    print(f"   Input:  {test_input.shape}")
    print(f"   Output: {output.shape}")
    print(f"   ✅ SUCCESS!")
    
    print(f"\n🔄 GRADIENT FLOW TEST:")
    loss = output.sum()
    loss.backward()
    
    print(f"   Seed gradient:   {layer.seed.grad.abs().mean():.6f}")
    print(f"   Powers gradient: {layer.powers.grad.abs().mean():.6f}")
    print(f"   Bias gradient:   {layer.bias.grad.abs().mean():.6f}")
    
    if layer.seed.grad.abs().mean() > 1e-6:
        print(f"   ✅ Gradients flowing properly!")
    else:
        print(f"   ❌ Gradient flow issue!")
    
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    test_power_layer()