#!/usr/bin/env python3
"""
🚀 Training Pipeline - Latent Diffusion Model
Pipeline complet optimisé pour GPU T4/L4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
from datetime import datetime
import json

# Imports des composants
from vae_encoder import VAEEncoder
from vae_decoder import VAEDecoder
from text_tokenizer import SimpleTokenizer
from text_encoder import TextEncoder
from diffusion_unet import DiffusionUNet
from noise_scheduler import DDPMScheduler, DDIMScheduler

# ============================================
# DATASET
# ============================================
class ImageTextDataset(Dataset):
    """
    Dataset simple pour images + captions
    Format attendu:
    - images/: dossier avec images
    - captions.json: {"image1.jpg": "caption...", ...}
    """
    def __init__(self, image_dir, captions_file, tokenizer, image_size=512):
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        
        # Load captions
        with open(captions_file, 'r') as f:
            self.captions = json.load(f)
        
        self.image_files = list(self.captions.keys())
        
        # Transform pour images
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # [-1, 1]
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_file = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_file)
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # Get caption
        caption = self.captions[img_file]
        
        # Tokenize
        tokens = self.tokenizer.encode(caption)
        tokens = torch.tensor(tokens, dtype=torch.long)
        
        return image, tokens, caption

# ============================================
# TRAINING LOOP
# ============================================
class LatentDiffusionTrainer:
    def __init__(
        self,
        vae_encoder,
        vae_decoder,
        text_encoder,
        unet,
        scheduler,
        device='cuda',
        learning_rate=1e-4,
        use_fp16=True,
    ):
        self.device = device
        self.use_fp16 = use_fp16
        
        # Models
        self.vae_encoder = vae_encoder.to(device).eval()
        self.vae_decoder = vae_decoder.to(device).eval()
        self.text_encoder = text_encoder.to(device).eval()
        self.unet = unet.to(device)
        
        # Freeze VAE et text encoder
        for param in self.vae_encoder.parameters():
            param.requires_grad = False
        for param in self.vae_decoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        # Scheduler
        self.scheduler = scheduler
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        
        # Scaler pour FP16
        self.scaler = torch.amp.GradScaler('cuda') if use_fp16 else None
        
        # Stats
        self.step = 0
    
    def train_step(self, batch):
        """Un step de training"""
        images, tokens, captions = batch
        images = images.to(self.device)
        tokens = tokens.to(self.device)
        
        batch_size = images.shape[0]
        
        # 1. Encoder images en latent
        with torch.no_grad():
            latent_clean, _, _ = self.vae_encoder(images)
            # Scale latent (important!)
            latent_clean = latent_clean * 0.18215
        
        # 2. Random timesteps
        timesteps = torch.randint(
            0, self.scheduler.num_steps,
            (batch_size,),
            device=self.device
        )
        
        # 3. Random noise
        noise = torch.randn_like(latent_clean)
        
        # 4. Ajouter bruit
        latent_noisy = self.scheduler.add_noise(latent_clean, timesteps, noise)
        
        # 5. Encoder texte
        with torch.no_grad():
            text_embeddings = self.text_encoder(tokens)
        
        # 6. Forward avec mixed precision
        with torch.amp.autocast('cuda', enabled=self.use_fp16):
            # Prédire le bruit
            noise_pred = self.unet(latent_noisy, timesteps, text_embeddings)
            
            # Loss MSE
            loss = F.mse_loss(noise_pred, noise)
        
        # 7. Backward
        self.optimizer.zero_grad(set_to_none=True)
        
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        
        self.step += 1
        
        return loss.item()
    
    @torch.no_grad()
    def validate(self, val_loader, num_batches=10):
        """Validation"""
        self.unet.eval()
        total_loss = 0
        
        for i, batch in enumerate(val_loader):
            if i >= num_batches:
                break
            
            images, tokens, _ = batch
            images = images.to(self.device)
            tokens = tokens.to(self.device)
            
            batch_size = images.shape[0]
            
            # Encoder
            latent_clean, _, _ = self.vae_encoder(images)
            latent_clean = latent_clean * 0.18215
            
            # Random timesteps
            timesteps = torch.randint(0, self.scheduler.num_steps, (batch_size,), device=self.device)
            noise = torch.randn_like(latent_clean)
            latent_noisy = self.scheduler.add_noise(latent_clean, timesteps, noise)
            
            # Text
            text_embeddings = self.text_encoder(tokens)
            
            # Predict
            noise_pred = self.unet(latent_noisy, timesteps, text_embeddings)
            loss = F.mse_loss(noise_pred, noise)
            
            total_loss += loss.item()
        
        self.unet.train()
        return total_loss / min(num_batches, len(val_loader))
    
    @torch.no_grad()
    def generate_sample(self, prompt, tokenizer, num_steps=50, guidance_scale=7.5):
        """Génère une image depuis un prompt"""
        self.unet.eval()
        
        # Tokenize
        tokens = tokenizer.encode(prompt)
        tokens = torch.tensor([tokens], dtype=torch.long).to(self.device)
        
        # Text embeddings
        text_emb = self.text_encoder(tokens)
        
        # Unconditional embedding (pour CFG)
        uncond_tokens = tokenizer.encode("")
        uncond_tokens = torch.tensor([uncond_tokens], dtype=torch.long).to(self.device)
        uncond_emb = self.text_encoder(uncond_tokens)
        
        # DDIM scheduler pour sampling
        ddim = DDIMScheduler(num_train_steps=self.scheduler.num_steps)
        timesteps = ddim.get_sampling_timesteps(num_steps)
        
        # Start from noise
        latent = torch.randn(1, 4, 64, 64, device=self.device)
        
        # Denoising loop
        for i, t in enumerate(tqdm(timesteps, desc="Generating")):
            t_tensor = torch.tensor([t], device=self.device)
            
            # Classifier-free guidance
            noise_pred_text = self.unet(latent, t_tensor, text_emb)
            noise_pred_uncond = self.unet(latent, t_tensor, uncond_emb)
            
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # DDIM step
            if i < len(timesteps) - 1:
                next_t = timesteps[i + 1].item()
            else:
                next_t = -1
            
            latent, _ = ddim.step(latent, t.item(), next_t, noise_pred, eta=0.0)
        
        # Decode
        latent = latent / 0.18215
        image = self.vae_decoder(latent)
        
        # [-1, 1] → [0, 1]
        image = (image + 1.0) / 2.0
        image = torch.clamp(image, 0.0, 1.0)
        
        self.unet.train()
        
        return image
    
    def save_checkpoint(self, path):
        """Sauvegarder checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'unet_state_dict': self.unet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': self.step,
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
        print(f"✅ Checkpoint saved: {path}")
    
    def load_checkpoint(self, path):
        """Charger checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.unet.load_state_dict(checkpoint['unet_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step = checkpoint['step']
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"✅ Checkpoint loaded: {path} (step {self.step})")

# ============================================
# MAIN TRAINING
# ============================================
def train(
    image_dir,
    captions_file,
    checkpoint_dir='./checkpoints',
    num_epochs=10,
    batch_size=8,
    save_every=1000,
    val_every=500,
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Device: {device}")
    
    # 1. Créer les modèles
    print("📦 Loading models...")
    
    vae_encoder = VAEEncoder(in_channels=3, latent_channels=4, base_channels=128)
    vae_decoder = VAEDecoder(latent_channels=4, out_channels=3, base_channels=128)
    
    tokenizer = SimpleTokenizer(max_length=77)
    text_encoder = TextEncoder(vocab_size=tokenizer.vocab_size, embed_dim=768)
    
    unet = DiffusionUNet(
        in_channels=4,
        out_channels=4,
        model_channels=320,
        num_heads=8,
        context_dim=768
    )
    
    scheduler = DDPMScheduler(num_steps=1000, schedule_type="cosine")
    
    # 2. Créer trainer
    trainer = LatentDiffusionTrainer(
        vae_encoder=vae_encoder,
        vae_decoder=vae_decoder,
        text_encoder=text_encoder,
        unet=unet,
        scheduler=scheduler,
        device=device,
        learning_rate=1e-4,
        use_fp16=True
    )
    
    # 3. Dataset
    print("📂 Loading dataset...")
    dataset = ImageTextDataset(image_dir, captions_file, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # 4. Training loop
    print("🚀 Starting training...")
    
    for epoch in range(num_epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in pbar:
            loss = trainer.train_step(batch)
            
            pbar.set_postfix({'loss': f'{loss:.4f}', 'step': trainer.step})
            
            # Validation
            if trainer.step % val_every == 0:
                val_loss = trainer.validate(dataloader, num_batches=10)
                print(f"\n📊 Step {trainer.step} - Val Loss: {val_loss:.4f}\n")
            
            # Save checkpoint
            if trainer.step % save_every == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"model_step_{trainer.step}.pt")
                trainer.save_checkpoint(checkpoint_path)
        
        # Epoch terminée
        print(f"\n✅ Epoch {epoch+1} done!")
    
    print("\n🎉 Training complete!")

# ============================================
# TEST
# ============================================
if __name__ == "__main__":
    print("="*60)
    print("🚀 TRAINING PIPELINE TEST")
    print("="*60)
    
    # Test simple sans dataset
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Models
    vae_encoder = VAEEncoder().to(device)
    vae_decoder = VAEDecoder().to(device)
    tokenizer = SimpleTokenizer()
    text_encoder = TextEncoder(vocab_size=tokenizer.vocab_size).to(device)
    unet = DiffusionUNet().to(device)
    scheduler = DDPMScheduler()
    
    # Trainer
    trainer = LatentDiffusionTrainer(
        vae_encoder, vae_decoder, text_encoder, unet, scheduler, device
    )
    
    print("✅ Trainer initialized!")
    print(f"📊 UNet params: {sum(p.numel() for p in unet.parameters())/1e6:.1f}M")
    
    print("\n📝 To train, run:")
    print("   python 7_train_pipeline.py")
    print("\nOr use the train() function:")
    print("   train(image_dir='./data/images', captions_file='./data/captions.json')")