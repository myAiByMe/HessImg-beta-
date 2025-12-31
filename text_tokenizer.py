#!/usr/bin/env python3
"""
📝 Text Tokenizer - Simple BPE
Convertit texte → tokens pour le diffusion model
"""

import torch
import re
import json
from pathlib import Path

class SimpleTokenizer:
    """
    Tokenizer simple basé sur GPT-2 tokenizer
    Utilise le vocab de GPT-2 pour éviter de retrain
    """
    def __init__(self, max_length=77):
        self.max_length = max_length
        
        # Utiliser le tokenizer GPT-2 (déjà disponible)
        try:
            from transformers import GPT2Tokenizer
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.vocab_size = len(self.tokenizer)
            print(f"✅ Loaded GPT-2 tokenizer (vocab_size={self.vocab_size})")
        except:
            print("⚠️  transformers not installed, using basic tokenizer")
            self._init_basic_tokenizer()
    
    def _init_basic_tokenizer(self):
        """Fallback: tokenizer basique par mots"""
        self.vocab_size = 50000
        self.word_to_id = {}
        self.id_to_word = {}
        self.next_id = 0
        
        # Tokens spéciaux
        self.pad_token_id = self._add_token("<PAD>")
        self.bos_token_id = self._add_token("<BOS>")
        self.eos_token_id = self._add_token("<EOS>")
        self.unk_token_id = self._add_token("<UNK>")
    
    def _add_token(self, token):
        if token not in self.word_to_id:
            self.word_to_id[token] = self.next_id
            self.id_to_word[self.next_id] = token
            self.next_id += 1
        return self.word_to_id[token]
    
    def encode(self, text):
        """
        Encode texte → liste de token IDs
        """
        if hasattr(self, 'tokenizer'):
            # GPT-2 tokenizer
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            
            # Truncate ou pad à max_length
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]
            else:
                # Pad avec EOS
                tokens = tokens + [self.tokenizer.eos_token_id] * (self.max_length - len(tokens))
            
            return tokens
        else:
            # Basic tokenizer
            words = text.lower().split()
            tokens = [self.bos_token_id]
            
            for word in words:
                if word in self.word_to_id:
                    tokens.append(self.word_to_id[word])
                else:
                    # Nouveau mot
                    if self.next_id < self.vocab_size:
                        tokens.append(self._add_token(word))
                    else:
                        tokens.append(self.unk_token_id)
            
            tokens.append(self.eos_token_id)
            
            # Truncate/pad
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]
            else:
                tokens = tokens + [self.pad_token_id] * (self.max_length - len(tokens))
            
            return tokens
    
    def encode_batch(self, texts):
        """
        Encode batch de textes
        texts: list of strings
        returns: [batch_size, max_length]
        """
        token_ids = [self.encode(text) for text in texts]
        return torch.tensor(token_ids, dtype=torch.long)
    
    def decode(self, token_ids):
        """
        Decode token IDs → texte
        """
        if hasattr(self, 'tokenizer'):
            return self.tokenizer.decode(token_ids, skip_special_tokens=True)
        else:
            words = []
            for token_id in token_ids:
                if token_id in self.id_to_word:
                    word = self.id_to_word[token_id]
                    if word not in ["<PAD>", "<BOS>", "<EOS>"]:
                        words.append(word)
            return " ".join(words)

# ============================================
# TEST
# ============================================
if __name__ == "__main__":
    # Créer tokenizer
    tokenizer = SimpleTokenizer(max_length=77)
    
    # Test prompts
    prompts = [
        "a beautiful cat sitting on a table",
        "sunset over mountains with dramatic clouds",
        "cyberpunk city at night, neon lights, rain",
    ]
    
    print("="*60)
    print("📝 TEXT TOKENIZER TEST")
    print("="*60)
    
    for prompt in prompts:
        print(f"\n📄 Prompt: {prompt}")
        
        # Encode
        tokens = tokenizer.encode(prompt)
        print(f"   Tokens: {len(tokens)} → {tokens[:10]}...")
        
        # Decode
        decoded = tokenizer.decode(tokens)
        print(f"   Decoded: {decoded[:50]}...")
    
    # Batch encoding
    print(f"\n📦 Batch encoding...")
    token_batch = tokenizer.encode_batch(prompts)
    print(f"   Shape: {token_batch.shape}")
    print(f"   Dtype: {token_batch.dtype}")
    
    print("\n✅ Tokenizer OK!")