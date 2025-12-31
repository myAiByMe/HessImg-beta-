#!/usr/bin/env python3
"""
📝 Text Encoder Training - Contrastive Learning (CLIP-style)
Entraîne le text encoder avec image-text pairs
Phase 2 du projet Latent Diffusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms
import os
from tqdm import tqdm
import json

from text_tokenizer import SimpleTokenizer
from text_encoder import TextEncoder
from vae_encoder import VAEEncoder

# ============================================
# CONTRASTIVE LOSS (CLIP-style)
# ============================================
class ContrastiveLoss(nn.Module):
    """
    Contrastive loss pour aligner texte et image
    Similaire à CLIP
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, image_features, text_features):
        """
        image_features: [batch, embed_dim]
        text_features: [batch, embed_dim]
        """
        # Normalize
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Compute similarity matrix
        logits = (image_features @ text_features.T) / self.temperature
        
        # Labels: diagonal elements are positive pairs
        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size, device=image_features.device)
        
        # Symmetric loss (image→text + text→image)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        
        loss = (loss_i2t + loss_t2i) / 2
        
        # Accuracy
        with torch.no_grad():
            pred_i2t = logits.argmax(dim=1)
            pred_t2i = logits.T.argmax(dim=1)
            acc_i2t = (pred_i2t == labels).float().mean()
            acc_t2i = (pred_t2i == labels).float().mean()
            acc = (acc_i2t + acc_t2i) / 2
        
        return loss, acc.item()

# ============================================
# DATASET
# ============================================
class ImageTextPairDataset(Dataset):
    """
    Dataset pour paires image-texte
    Format:
    - images/: images
    - captions.json: {"image.jpg": "caption...", ...}
    """
    def __init__(self, image_dir, captions_file, tokenizer, image_size=512):
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        
        with open(captions_file, 'r') as f:
            self.captions = json.load(f)
        
        self.image_files = list(self.captions.keys())
        
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_file)
        
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            
            caption = self.captions[img_file]
            tokens = self.tokenizer.encode(caption)
            tokens = torch.tensor(tokens, dtype=torch.long)
            
            return image, tokens, caption
        except Exception as e:
            return self.__getitem__((idx + 1) % len(self))

# ============================================
# PROJECTION HEAD
# ============================================
class ProjectionHead(nn.Module):
    """
    Project features to shared embedding space
    """
    def __init__(self, input_dim, output_dim=512):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, x):
        return self.projection(x)

# ============================================
# TRAINER
# ============================================
class TextEncoderTrainer:
    def __init__(
        self,
        text_encoder,
        vae_encoder,
        tokenizer,
        device='cuda',
        learning_rate=1e-4,
        projection_dim=512,
        use_fp16=True
    ):
        self.device = device
        self.tokenizer = tokenizer
        self.use_fp16 = use_fp16
        
        # Models
        self.text_encoder = text_encoder.to(device)
        self.vae_encoder = vae_encoder.to(device).eval()
        
        # Freeze VAE
        for param in self.vae_encoder.parameters():
            param.requires_grad = False
        
        # Projection heads
        self.text_projection = ProjectionHead(768, projection_dim).to(device)
        self.image_projection = ProjectionHead(4 * 64 * 64, projection_dim).to(device)
        
        # Loss
        self.contrastive_loss = ContrastiveLoss(temperature=0.07)
        
        # Optimizer (text encoder + projections)
        self.optimizer = torch.optim.AdamW(
            list(text_encoder.parameters()) + 
            list(self.text_projection.parameters()) + 
            list(self.image_projection.parameters()),
            lr=learning_rate,
            betas=(0.9, 0.98),
            weight_decay=0.2
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=50000,
            eta_min=1e-6
        )
        
        # Scaler
        self.scaler = torch.amp.GradScaler('cuda') if use_fp16 else None
        
        self.step = 0
    
    def train_step(self, batch):
        """Un step de training"""
        images, tokens, captions = batch
        images = images.to(self.device)
        tokens = tokens.to(self.device)
        
        # Forward
        with torch.amp.autocast('cuda', enabled=self.use_fp16):
            # Encode images (frozen VAE)
            with torch.no_grad():
                latent, _, _ = self.vae_encoder(images)
                # Flatten latent
                image_features = latent.flatten(1)  # [batch, 4*64*64]
            
            # Project image features
            image_embed = self.image_projection(image_features)
            
            # Encode text
            text_features = self.text_encoder(tokens)
            # Use [CLS] token (first token) or mean pooling
            text_features = text_features[:, 0, :]  # [batch, 768]
            text_embed = self.text_projection(text_features)
            
            # Contrastive loss
            loss, acc = self.contrastive_loss(image_embed, text_embed)
        
        # Backward
        self.optimizer.zero_grad(set_to_none=True)
        
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        
        self.scheduler.step()
        self.step += 1
        
        return {
            'loss': loss.item(),
            'accuracy': acc,
            'lr': self.optimizer.param_groups[0]['lr']
        }
    
    @torch.no_grad()
    def validate(self, val_loader, num_batches=10):
        """Validation"""
        self.text_encoder.eval()
        self.text_projection.eval()
        self.image_projection.eval()
        
        total_loss = 0
        total_acc = 0
        
        for i, batch in enumerate(val_loader):
            if i >= num_batches:
                break
            
            images, tokens, _ = batch
            images = images.to(self.device)
            tokens = tokens.to(self.device)
            
            # Forward
            latent, _, _ = self.vae_encoder(images)
            image_features = latent.flatten(1)
            image_embed = self.image_projection(image_features)
            
            text_features = self.text_encoder(tokens)
            text_features = text_features[:, 0, :]
            text_embed = self.text_projection(text_features)
            
            loss, acc = self.contrastive_loss(image_embed, text_embed)
            
            total_loss += loss.item()
            total_acc += acc
        
        self.text_encoder.train()
        self.text_projection.train()
        self.image_projection.train()
        
        return {
            'loss': total_loss / num_batches,
            'accuracy': total_acc / num_batches
        }
    
    def save_checkpoint(self, path):
        """Sauvegarder checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'text_encoder_state_dict': self.text_encoder.state_dict(),
            'text_projection_state_dict': self.text_projection.state_dict(),
            'image_projection_state_dict': self.image_projection.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'step': self.step,
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
        print(f"✅ Text encoder checkpoint saved: {path}")
    
    def load_checkpoint(self, path):
        """Charger checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])
        self.text_projection.load_state_dict(checkpoint['text_projection_state_dict'])
        self.image_projection.load_state_dict(checkpoint['image_projection_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.step = checkpoint['step']
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"✅ Text encoder loaded: {path} (step {self.step})")

# ============================================
# MAIN TRAINING
# ============================================
def train_text_encoder(
    image_dir,
    captions_file,
    vae_checkpoint,
    checkpoint_dir='./checkpoints/text_encoder',
    num_epochs=10,
    batch_size=128,
    save_every=1000,
    val_every=500,
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Device: {device}")
    
    # Load VAE (frozen)
    print("📦 Loading frozen VAE...")
    vae_encoder = VAEEncoder()
    vae_checkpoint_data = torch.load(vae_checkpoint, map_location=device)
    vae_encoder.load_state_dict(vae_checkpoint_data['encoder_state_dict'])
    print("✅ VAE loaded and frozen")
    
    # Create text encoder
    print("📦 Creating text encoder...")
    tokenizer = SimpleTokenizer(max_length=77)
    text_encoder = TextEncoder(vocab_size=tokenizer.vocab_size)
    
    params = sum(p.numel() for p in text_encoder.parameters()) / 1e6
    print(f"   Text Encoder: {params:.1f}M params")
    
    # Dataset
    print("📂 Loading dataset...")
    dataset = ImageTextPairDataset(image_dir, captions_file, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Trainer
    trainer = TextEncoderTrainer(
        text_encoder=text_encoder,
        vae_encoder=vae_encoder,
        tokenizer=tokenizer,
        device=device,
        learning_rate=1e-4,
        use_fp16=True
    )
    
    # Training loop
    print("🚀 Starting text encoder training...")
    
    for epoch in range(num_epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in pbar:
            metrics = trainer.train_step(batch)
            
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'acc': f"{metrics['accuracy']:.2%}",
                'lr': f"{metrics['lr']:.2e}"
            })
            
            # Validation
            if trainer.step % val_every == 0:
                val_metrics = trainer.validate(dataloader)
                print(f"\n📊 Step {trainer.step}")
                print(f"   Val Loss: {val_metrics['loss']:.4f}")
                print(f"   Val Acc: {val_metrics['accuracy']:.2%}\n")
            
            # Save
            if trainer.step % save_every == 0:
                path = f"{checkpoint_dir}/text_encoder_step_{trainer.step}.pt"
                trainer.save_checkpoint(path)
        
        print(f"\n✅ Epoch {epoch+1} done!")
    
    # Final save
    final_path = f"{checkpoint_dir}/text_encoder_final.pt"
    trainer.save_checkpoint(final_path)
    
    print("\n🎉 Text Encoder Training complete!")

# ============================================
# CLI
# ============================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--captions_file', type=str, required=True)
    parser.add_argument('--vae_checkpoint', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/text_encoder')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    
    args = parser.parse_args()
    
    train_text_encoder(
        image_dir=args.image_dir,
        captions_file=args.captions_file,
        vae_checkpoint=args.vae_checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size
    )