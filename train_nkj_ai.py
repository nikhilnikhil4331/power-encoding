# File: train_nkj_ai.py

"""
Train NKJ AI on Text Data
Example: Wikipedia, Books, etc.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nkj_ai_gpt import NKJ_AI_GPT
import time

class TextDataset(Dataset):
    """
    Simple text dataset
    Replace with your actual data
    """
    def __init__(self, texts, max_length=512):
        self.texts = texts
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        # Simple tokenization (replace with proper tokenizer)
        text = self.texts[idx]
        tokens = torch.randint(0, 50000, (self.max_length,))
        return tokens

def train_nkj_ai():
    print("\n" + "🎓" * 30)
    print("   TRAINING NKJ AI")
    print("🎓" * 30)
    
    # Create model
    model = NKJ_AI_GPT(
        vocab_size=50000,
        d_model=768,
        num_layers=12,
        seed_size=64
    )
    
    print(f"\n📊 Model: {model.count_parameters():,} parameters")
    
    # Sample data (replace with real data!)
    sample_texts = [
        "The future of AI is bright.",
        "Machine learning is amazing.",
        "Deep learning transforms everything.",
    ] * 100  # 300 samples
    
    dataset = TextDataset(sample_texts)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Training setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\n🚀 Training on {device}...")
    print(f"   Dataset: {len(dataset)} samples")
    print(f"   Batch size: 4")
    
    # Training loop
    epochs = 3
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        print(f"\n📚 Epoch {epoch+1}/{epochs}")
        print("-" * 50)
        
        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(device)
            
            # Forward
            logits = model(batch[:, :-1])  # Input
            targets = batch[:, 1:]  # Shifted targets
            
            # Loss
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"  [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"\n✅ Epoch {epoch+1} Complete | Avg Loss: {avg_loss:.4f}")
    
    # Save trained model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epochs,
        'loss': avg_loss
    }, 'nkj_ai_trained.pth')
    
    print("\n💾 Model saved: nkj_ai_trained.pth")
    print("✅ Training complete!")

if __name__ == "__main__":
    train_nkj_ai()