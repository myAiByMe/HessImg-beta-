#!/usr/bin/env python3
"""
🎨 Inference Pipeline - Génération d'images
Charge un modèle trained et génère des images depuis des prompts
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from tqdm import tqdm
import argparse

# Imports des composants
from vae_encoder import VAEEncoder
from vae_decoder import VAEDecoder
from text_tokenizer import SimpleTokenizer
from text_encoder import TextEncoder
from diffusion_unet import DiffusionUNet
from noise_scheduler import DDIMScheduler

class ImageGenerator:
    """
    Pipeline de génération d'images
    """
    def __init__(
        self,
        checkpoint_path,
        device='cuda',
        num_inference_steps=50,
    ):
        self.device = device
        self.num_inference_steps = num_inference_steps
        
        print(f"🚀 Initializing generator on {device}...")
        
        # 1. Load models
        self.vae_encoder = VAEEncoder(
            in_channels=3,
            latent_channels=4,
            base_channels=128
        ).to(device).eval()
        
        self.vae_decoder = VAEDecoder(
            latent_channels=4,
            out_channels=3,
            base_channels=128
        ).to(device).eval()
        
        self.tokenizer = SimpleTokenizer(max_length=77)
        
        self.text_encoder = TextEncoder(
            vocab_size=self.tokenizer.vocab_size,
            embed_dim=768,
            max_length=77,
            num_layers=6,
            num_heads=12,
            hidden_dim=3072
        ).to(device).eval()
        
        self.unet = DiffusionUNet(
            in_channels=4,
            out_channels=4,
            model_channels=320,
            num_heads=8,
            context_dim=768
        ).to(device).eval()
        
        # 2. Load checkpoint
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"📂 Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            self.unet.load_state_dict(checkpoint['unet_state_dict'])
            print(f"✅ Loaded step {checkpoint.get('step', 'unknown')}")
        else:
            print("⚠️  No checkpoint loaded, using random weights")
        
        # 3. Scheduler pour inference (DDIM)
        self.scheduler = DDIMScheduler(num_train_steps=1000)
        
        print("✅ Generator ready!")
    
    @torch.no_grad()
    def generate(
        self,
        prompt,
        negative_prompt="",
        guidance_scale=7.5,
        num_images=1,
        seed=None,
        height=512,
        width=512,
    ):
        """
        Génère des images depuis un prompt
        
        Args:
            prompt: Description de l'image
            negative_prompt: Ce qu'on ne veut PAS
            guidance_scale: Force du guidance (7.5 = bon compromis)
            num_images: Nombre d'images à générer
            seed: Random seed (pour reproductibilité)
            height, width: Taille finale (doit être multiple de 64)
        
        Returns:
            List de PIL Images
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        # Vérifier dimensions
        assert height % 64 == 0 and width % 64 == 0, "Height/Width must be multiple of 64"
        latent_h, latent_w = height // 8, width // 8
        
        print(f"🎨 Generating {num_images} image(s)...")
        print(f"   Prompt: {prompt}")
        if negative_prompt:
            print(f"   Negative: {negative_prompt}")
        print(f"   Size: {width}×{height}")
        print(f"   Guidance: {guidance_scale}")
        print(f"   Steps: {self.num_inference_steps}")
        
        # 1. Encode prompts
        positive_tokens = self.tokenizer.encode(prompt)
        positive_tokens = torch.tensor([positive_tokens] * num_images, dtype=torch.long).to(self.device)
        positive_emb = self.text_encoder(positive_tokens)
        
        negative_tokens = self.tokenizer.encode(negative_prompt)
        negative_tokens = torch.tensor([negative_tokens] * num_images, dtype=torch.long).to(self.device)
        negative_emb = self.text_encoder(negative_tokens)
        
        # 2. Get sampling timesteps
        timesteps = self.scheduler.get_sampling_timesteps(self.num_inference_steps)
        
        # 3. Start from noise
        latent = torch.randn(num_images, 4, latent_h, latent_w, device=self.device)
        
        # 4. Denoising loop
        for i, t in enumerate(tqdm(timesteps, desc="Denoising")):
            t_tensor = torch.tensor([t] * num_images, device=self.device)
            
            # Classifier-Free Guidance (CFG)
            # 2 forward passes: avec et sans condition
            noise_pred_cond = self.unet(latent, t_tensor, positive_emb)
            noise_pred_uncond = self.unet(latent, t_tensor, negative_emb)
            
            # Guidance
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            # DDIM step
            if i < len(timesteps) - 1:
                next_t = timesteps[i + 1].item()
            else:
                next_t = -1
            
            latent, _ = self.scheduler.step(
                latent, t.item(), next_t, noise_pred, eta=0.0
            )
        
        # 5. Decode latents
        latent = latent / 0.18215  # Unscale
        images = self.vae_decoder(latent)
        
        # 6. Post-process
        images = (images + 1.0) / 2.0  # [-1, 1] → [0, 1]
        images = torch.clamp(images, 0.0, 1.0)
        
        # 7. Convert to PIL
        pil_images = []
        for i in range(num_images):
            img_tensor = images[i].cpu()
            img_array = (img_tensor.permute(1, 2, 0).numpy() * 255).astype('uint8')
            pil_img = Image.fromarray(img_array)
            pil_images.append(pil_img)
        
        return pil_images
    
    def generate_grid(
        self,
        prompts,
        output_path="output_grid.png",
        guidance_scale=7.5,
        images_per_prompt=4,
    ):
        """
        Génère une grille d'images pour plusieurs prompts
        """
        all_images = []
        
        for prompt in prompts:
            print(f"\n📝 Generating: {prompt}")
            images = self.generate(
                prompt=prompt,
                guidance_scale=guidance_scale,
                num_images=images_per_prompt
            )
            all_images.extend(images)
        
        # Créer grille
        from PIL import Image
        
        num_cols = images_per_prompt
        num_rows = len(prompts)
        
        img_w, img_h = all_images[0].size
        grid_w = img_w * num_cols
        grid_h = img_h * num_rows
        
        grid = Image.new('RGB', (grid_w, grid_h))
        
        for i, img in enumerate(all_images):
            row = i // num_cols
            col = i % num_cols
            grid.paste(img, (col * img_w, row * img_h))
        
        grid.save(output_path)
        print(f"\n✅ Grid saved: {output_path}")
        
        return grid

# ============================================
# CLI
# ============================================
def main():
    parser = argparse.ArgumentParser(description="Generate images with Latent Diffusion")
    
    parser.add_argument('--prompt', type=str, required=True,
                        help='Text prompt describing the image')
    parser.add_argument('--negative', type=str, default="",
                        help='Negative prompt (what to avoid)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='output.png',
                        help='Output image path')
    parser.add_argument('--num-images', type=int, default=1,
                        help='Number of images to generate')
    parser.add_argument('--guidance', type=float, default=7.5,
                        help='Guidance scale (7.5 recommended)')
    parser.add_argument('--steps', type=int, default=50,
                        help='Number of inference steps')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--height', type=int, default=512,
                        help='Image height (must be multiple of 64)')
    parser.add_argument('--width', type=int, default=512,
                        help='Image width (must be multiple of 64)')
    
    args = parser.parse_args()
    
    # Create generator
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator = ImageGenerator(
        checkpoint_path=args.checkpoint,
        device=device,
        num_inference_steps=args.steps
    )
    
    # Generate
    images = generator.generate(
        prompt=args.prompt,
        negative_prompt=args.negative,
        guidance_scale=args.guidance,
        num_images=args.num_images,
        seed=args.seed,
        height=args.height,
        width=args.width
    )
    
    # Save
    if args.num_images == 1:
        images[0].save(args.output)
        print(f"\n✅ Image saved: {args.output}")
    else:
        for i, img in enumerate(images):
            output_path = args.output.replace('.png', f'_{i+1}.png')
            img.save(output_path)
            print(f"✅ Image saved: {output_path}")

# ============================================
# EXAMPLES
# ============================================
if __name__ == "__main__":
    print("="*60)
    print("🎨 INFERENCE PIPELINE")
    print("="*60)
    
    # Example usage
    print("\n📝 Example commands:")
    print("\n1. Simple generation:")
    print("   python 8_inference_pipeline.py \\")
    print('     --prompt "a beautiful cat sitting on a table" \\')
    print('     --checkpoint ./checkpoints/model_step_10000.pt \\')
    print('     --output output.png')
    
    print("\n2. Multiple images:")
    print("   python 8_inference_pipeline.py \\")
    print('     --prompt "cyberpunk city at night" \\')
    print('     --negative "blurry, low quality" \\')
    print('     --num-images 4 \\')
    print('     --guidance 7.5')
    
    print("\n3. Custom size:")
    print("   python 8_inference_pipeline.py \\")
    print('     --prompt "sunset over mountains" \\')
    print('     --height 768 \\')
    print('     --width 768 \\')
    print('     --steps 100')
    
    print("\n4. Reproducible:")
    print("   python 8_inference_pipeline.py \\")
    print('     --prompt "magical forest" \\')
    print('     --seed 42')
    
    print("\n" + "="*60)
    print("\n📦 Or use as Python module:")
    print("""
from inference_pipeline import ImageGenerator

generator = ImageGenerator(
    checkpoint_path='./checkpoints/model.pt',
    num_inference_steps=50
)

images = generator.generate(
    prompt="a beautiful landscape",
    guidance_scale=7.5,
    num_images=4
)

for i, img in enumerate(images):
    img.save(f'output_{i}.png')
    """)
    
    # Run CLI if arguments provided
    import sys
    if len(sys.argv) > 1:
        main()