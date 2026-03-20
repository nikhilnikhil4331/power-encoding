import torch
import torchvision
import numpy as np
import matplotlib

print("=" * 60)
print("✅ LIBRARY INSTALLATION CHECK")
print("=" * 60)

print(f"\n📦 PyTorch:     {torch.__version__}")
print(f"📦 TorchVision: {torchvision.__version__}")
print(f"📦 NumPy:       {np.__version__}")
print(f"📦 Matplotlib:  {matplotlib.__version__}")

print(f"\n🖥️  CUDA:        {torch.cuda.is_available()}")

# Quick test
x = torch.randn(3, 3)
print(f"\n🧪 Tensor Test:")
print(x)

print("\n" + "=" * 60)
print("✅ ALL LIBRARIES WORKING PERFECTLY!")
print("=" * 60)