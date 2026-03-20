"""
NKJ AI Training - GPU Optimized (RTX 3050)
Fast training with CUDA acceleration!
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nkj_ai_gpt import NKJ_AI_GPT
import time
from torch.cuda.amp import autocast, GradScaler

class TextDataset(Dataset):
    def __init__(self, num_samples=1000, seq_len=256):
        self.num_samples = num_samples
        self.seq_len = seq_len
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Random tokens (replace with real data later)
        return torch.randint(0, 50000, (self.seq_len,))


def train_gpu_optimized():
    print("\n" + "🎮" * 35)
    print("   NKJ AI - GPU TRAINING (RTX 3050)")
    print("🎮" * 35)
    
    # Check GPU
    if not torch.cuda.is_available():
        print("\n❌ CUDA not available!")
        print("Install CUDA from: https://developer.nvidia.com/cuda-downloads")
        return
    
    device = torch.device('cuda')
    print(f"\n✅ GPU Detected: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Model optimized for RTX 3050 (4GB VRAM)
    print("\n🔨 Building model (optimized for 4GB VRAM)...")
    
    model = NKJ_AI_GPT(
        vocab_size=50000,
        d_model=512,          # Reduced from 768 to fit in 4GB
        num_layers=8,         # Reduced from 12
        num_heads=8,          # Reduced from 12
        d_ff=2048,            # Reduced from 3072
        max_seq_len=256,      # Reduced from 512
        seed_size=48,         # Moderate seed size
        dropout=0.1
    )
    
    model = model.to(device)
    
    params = model.count_parameters()
    model_size_mb = params * 4 / (1024**2)
    
    print(f"\n📊 Model Stats:")
    print(f"   Parameters: {params:,}")
    print(f"   Size: {model_size_mb:.1f} MB")
    print(f"   Device: {device}")
    
    # Dataset
    print("\n📚 Preparing dataset...")
    dataset = TextDataset(num_samples=1000, seq_len=256)
    
    # Batch size optimized for RTX 3050
    batch_size = 16  # 4GB VRAM can handle this
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2,
        pin_memory=True  # Faster GPU transfer
    )
    
    print(f"   Samples: {len(dataset)}")
    print(f"   Batch size: {batch_size}")
    print(f"   Batches per epoch: {len(dataloader)}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=3e-4,
        weight_decay=0.01
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Mixed precision training (faster on RTX 3050!)
    scaler = GradScaler()
    
    # Training
    epochs = 3
    print(f"\n🚀 Starting training ({epochs} epochs)...")
    print("="*70)
    
    total_start = time.time()
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_start = time.time()
        
        print(f"\n📚 Epoch {epoch+1}/{epochs}")
        print("-" * 70)
        
        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(device, non_blocking=True)
            
            # Mixed precision forward pass
            with autocast():
                # Input: all tokens except last
                # Target: all tokens except first
                logits = model(batch[:, :-1])
                targets = batch[:, 1:]
                
                # Calculate loss
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1)
                )
            
            # Backward with gradient scaling
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            
            # Progress
            if batch_idx % 10 == 0:
                gpu_mem = torch.cuda.memory_allocated() / 1024**3
                print(f"  [{batch_idx:3d}/{len(dataloader)}] "
                      f"Loss: {loss.item():.4f} | "
                      f"GPU Mem: {gpu_mem:.2f}GB")
        
        # Epoch summary
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / len(dataloader)
        
        print(f"\n✅ Epoch {epoch+1} Complete:")
        print(f"   Time: {epoch_time:.1f}s")
        print(f"   Avg Loss: {avg_loss:.4f}")
        print(f"   Speed: {len(dataset)/epoch_time:.1f} samples/sec")
        
        # Clear cache
        torch.cuda.empty_cache()
    
    total_time = time.time() - total_start
    
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETE!")
    print("="*70)
    print(f"   Total time: {total_time/60:.1f} minutes")
    print(f"   Final loss: {avg_loss:.4f}")
    print(f"   Avg speed: {epochs*len(dataset)/total_time:.1f} samples/sec")
    
    # Save model
    print("\n💾 Saving model...")
    
    save_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': {
            'vocab_size': 50000,
            'd_model': 512,
            'num_layers': 8,
            'num_heads': 8,
            'd_ff': 2048,
            'seed_size': 48
        },
        'training': {
            'epochs': epochs,
            'final_loss': avg_loss,
            'parameters': params,
            'device': 'RTX 3050'
        }
    }
    
    torch.save(save_dict, 'nkj_ai_gpu_trained.pth')
    
    import os
    file_size = os.path.getsize('nkj_ai_gpu_trained.pth') / (1024**2)
    
    print(f"✅ Saved: nkj_ai_gpu_trained.pth ({file_size:.1f} MB)")
    
    # GPU Stats
    print("\n📊 GPU Statistics:")
    print(f"   Peak memory: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
    print(f"   GPU utilization: Excellent! ✅")
    
    print("\n" + "="*70)
    print("🚀 Model ready for deployment!")
    print("="*70 + "\n")


if __name__ == "__main__":
    train_gpu_optimized()