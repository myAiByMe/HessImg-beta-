#!/usr/bin/env python3
"""
🎨 VAE Training - From Scratch
Entraîne le VAE sur des images publiques
Phase 1 du projet Latent Diffusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
from tqdm import tqdm
from pathlib import Path
import json

from vae_encoder import VAEEncoder
from vae_decoder import VAEDecoder

# ============================================
# DATASET
# ============================================
class ImageDataset(Dataset):
    """
    Dataset simple pour images
    Supporte plusieurs sources :
    - Dossier local
    - ImageNet
    - COCO
    - LAION
    """
    def __init__(self, image_dir, image_size=512, max_images=None):
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        
        # Trouver toutes les images
        extensions = ['.jpg', '.jpeg', '.png', '.webp']
        self.image_paths = []
        
        for ext in extensions:
            self.image_paths.extend(list(self.image_dir.rglob(f'*{ext}')))
        
        if max_images:
            self.image_paths = self.image_paths[:max_images]
        
        print(f"✅ Found {len(self.image_paths)} images")
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1]
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            return image
        except Exception as e:
            # Si erreur, retourner l'image suivante
            print(f"⚠️ Error loading {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))

class LAIONDataset(Dataset):
    """
    Dataset pour LAION (streaming)
    """
    def __init__(self, subset='laion400m', image_size=512, max_samples=100000):
        from datasets import load_dataset
        
        print(f"📥 Loading LAION {subset}...")
        self.dataset = load_dataset(
            "laion/laion400m",
            split='train',
            streaming=True
        )
        
        self.max_samples = max_samples
        self.image_size = image_size
        
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return self.max_samples
    
    def __iter__(self):
        count = 0
        for item in self.dataset:
            if count >= self.max_samples:
                break
            
            try:
                # LAION a un champ 'image' (URL) ou 'image_bytes'
                if 'image' in item:
                    image = item['image']
                    if isinstance(image, str):
                        # Download from URL
                        import requests
                        from io import BytesIO
                        response = requests.get(image, timeout=5)
                        image = Image.open(BytesIO(response.content)).convert('RGB')
                    else:
                        image = image.convert('RGB')
                    
                    image = self.transform(image)
                    yield image
                    count += 1
            except Exception as e:
                continue

# ============================================
# VAE LOSS
# ============================================
def vae_loss(recon_x, x, mu, logvar, kl_weight=0.00025):
    """
    VAE Loss = Reconstruction Loss + KL Divergence
    
    recon_x: [batch, 3, 512, 512] reconstructed image
    x: [batch, 3, 512, 512] original image
    mu: [batch, 4, 64, 64] mean of latent distribution
    logvar: [batch, 4, 64, 64] log variance
    """
    # Reconstruction loss (L1 ou L2)
    # L1 est souvent meilleur pour les images
    recon_loss = F.l1_loss(recon_x, x, reduction='mean')
    
    # KL divergence: KL(q(z|x) || p(z))
    # où p(z) = N(0, 1)
    # KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_loss / (mu.shape[0] * mu.shape[1] * mu.shape[2] * mu.shape[3])
    
    # Total loss
    total_loss = recon_loss + kl_weight * kl_loss
    
    return total_loss, recon_loss, kl_loss

# ============================================
# TRAINER
# ============================================
class VAETrainer:
    def __init__(
        self,
        encoder,
        decoder,
        device='cuda',
        learning_rate=1e-4,
        kl_weight=0.00025,
        use_fp16=True
    ):
        self.device = device
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.kl_weight = kl_weight
        self.use_fp16 = use_fp16
        
        # Optimizer pour encoder + decoder
        self.optimizer = torch.optim.AdamW(
            list(encoder.parameters()) + list(decoder.parameters()),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100000,
            eta_min=1e-6
        )
        
        # Scaler pour FP16
        self.scaler = torch.amp.GradScaler('cuda') if use_fp16 else None
        
        self.step = 0
    
    def train_step(self, images):
        """Un step de training"""
        images = images.to(self.device)
        
        # Forward
        with torch.amp.autocast('cuda', enabled=self.use_fp16):
            # Encode
            latent, mu, logvar = self.encoder(images)
            
            # Decode
            recon_images = self.decoder(latent)
            
            # Loss
            loss, recon_loss, kl_loss = vae_loss(
                recon_images, images, mu, logvar, self.kl_weight
            )
        
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
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),
            'lr': self.optimizer.param_groups[0]['lr']
        }
    
    @torch.no_grad()
    def validate(self, val_loader, num_batches=10):
        """Validation"""
        self.encoder.eval()
        self.decoder.eval()
        
        total_loss = 0
        total_recon = 0
        total_kl = 0
        
        for i, images in enumerate(val_loader):
            if i >= num_batches:
                break
            
            images = images.to(self.device)
            
            latent, mu, logvar = self.encoder(images)
            recon = self.decoder(latent)
            
            loss, recon_loss, kl_loss = vae_loss(
                recon, images, mu, logvar, self.kl_weight
            )
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
        
        self.encoder.train()
        self.decoder.train()
        
        return {
            'loss': total_loss / num_batches,
            'recon_loss': total_recon / num_batches,
            'kl_loss': total_kl / num_batches
        }
    
    @torch.no_grad()
    def save_samples(self, images, save_path):
        """Sauvegarder des exemples de reconstruction"""
        self.encoder.eval()
        self.decoder.eval()
        
        images = images.to(self.device)
        
        latent, _, _ = self.encoder(images)
        recon = self.decoder(latent)
        
        # Denormalize
        images = (images + 1.0) / 2.0
        recon = (recon + 1.0) / 2.0
        
        # Créer grille
        import torchvision.utils as vutils
        
        grid = torch.cat([images[:4], recon[:4]], dim=0)
        vutils.save_image(grid, save_path, nrow=4)
        
        self.encoder.train()
        self.decoder.train()
    
    def save_checkpoint(self, path):
        """Sauvegarder checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'step': self.step,
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
        print(f"✅ VAE checkpoint saved: {path}")
    
    def load_checkpoint(self, path):
        """Charger checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.step = checkpoint['step']
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"✅ VAE checkpoint loaded: {path} (step {self.step})")

# ============================================
# MAIN TRAINING
# ============================================
def train_vae(
    image_dir,
    checkpoint_dir='./checkpoints/vae',
    num_epochs=10,
    batch_size=16,
    save_every=1000,
    val_every=500,
    dataset_type='local',  # 'local' ou 'laion'
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Device: {device}")
    
    # Créer modèles
    print("📦 Creating VAE models...")
    encoder = VAEEncoder(in_channels=3, latent_channels=4, base_channels=128)
    decoder = VAEDecoder(latent_channels=4, out_channels=3, base_channels=128)
    
    # Compter params
    enc_params = sum(p.numel() for p in encoder.parameters()) / 1e6
    dec_params = sum(p.numel() for p in decoder.parameters()) / 1e6
    print(f"   Encoder: {enc_params:.1f}M params")
    print(f"   Decoder: {dec_params:.1f}M params")
    print(f"   Total: {enc_params + dec_params:.1f}M params")
    
    # Dataset
    print("📂 Loading dataset...")
    if dataset_type == 'local':
        dataset = ImageDataset(image_dir, image_size=512)
    elif dataset_type == 'laion':
        dataset = LAIONDataset(image_size=512, max_samples=100000)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(dataset_type == 'local'),
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    # Trainer
    trainer = VAETrainer(
        encoder=encoder,
        decoder=decoder,
        device=device,
        learning_rate=1e-4,
        kl_weight=0.00025,
        use_fp16=True
    )
    
    # Training loop
    print("🚀 Starting VAE training...")
    os.makedirs(f"{checkpoint_dir}/samples", exist_ok=True)
    
    for epoch in range(num_epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for images in pbar:
            metrics = trainer.train_step(images)
            
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'recon': f"{metrics['recon_loss']:.4f}",
                'kl': f"{metrics['kl_loss']:.6f}",
                'lr': f"{metrics['lr']:.2e}"
            })
            
            # Validation
            if trainer.step % val_every == 0:
                val_metrics = trainer.validate(dataloader, num_batches=10)
                print(f"\n📊 Step {trainer.step} - Val Loss: {val_metrics['loss']:.4f}")
                
                # Sauvegarder samples
                sample_path = f"{checkpoint_dir}/samples/step_{trainer.step}.png"
                trainer.save_samples(images, sample_path)
            
            # Save checkpoint
            if trainer.step % save_every == 0:
                checkpoint_path = f"{checkpoint_dir}/vae_step_{trainer.step}.pt"
                trainer.save_checkpoint(checkpoint_path)
        
        print(f"\n✅ Epoch {epoch+1} done!")
    
    # Final save
    final_path = f"{checkpoint_dir}/vae_final.pt"
    trainer.save_checkpoint(final_path)
    
    print("\n🎉 VAE Training complete!")

# ============================================
# CLI
# ============================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train VAE from scratch")
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory with images')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/vae',
                        help='Checkpoint directory')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--dataset_type', type=str, default='local',
                        choices=['local', 'laion'],
                        help='Dataset type')
    
    args = parser.parse_args()
    
    train_vae(
        image_dir=args.image_dir,
        checkpoint_dir=args.checkpoint_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        dataset_type=args.dataset_type
    )