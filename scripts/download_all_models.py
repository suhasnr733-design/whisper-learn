# scripts/download_all_models.py
#!/usr/bin/env python3
"""
Download all required models for Whisper Learn 8GB
"""
import os
import sys
import argparse
from pathlib import Path
import subprocess

class ModelDownloader:
    def __init__(self, models_dir="models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
    def download_whisper(self, model_size="medium"):
        """Download Whisper model"""
        print(f"\n📥 Downloading Whisper {model_size}...")
        import whisper
        
        try:
            model = whisper.load_model(model_size)
            print(f"✅ Whisper {model_size} downloaded")
            return True
        except Exception as e:
            print(f"❌ Failed: {e}")
            return False
    
    def download_embeddings(self, model_name="all-MiniLM-L6-v2"):
        """Download sentence transformer model"""
        print(f"\n📥 Downloading embeddings model {model_name}...")
        from sentence_transformers import SentenceTransformer
        
        try:
            model = SentenceTransformer(model_name)
            print(f"✅ Embeddings model downloaded")
            return True
        except Exception as e:
            print(f"❌ Failed: {e}")
            return False
    
    def download_summarization(self, model_name="facebook/bart-large-cnn"):
        """Download summarization model"""
        print(f"\n📥 Downloading summarization model {model_name}...")
        from transformers import pipeline
        
        try:
            summarizer = pipeline("summarization", model=model_name)
            print(f"✅ Summarization model downloaded")
            return True
        except Exception as e:
            print(f"❌ Failed: {e}")
            return False
    
    def download_llm(self, model_name="microsoft/phi-2", quantized=True):
        """Download LLM for Q&A"""
        print(f"\n📥 Downloading LLM {model_name}...")
        
        if quantized:
            print("   Using 4-bit quantization")
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            import torch
            
            if quantized:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
            
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            print(f"✅ LLM model downloaded")
            return True
        except Exception as e:
            print(f"❌ Failed: {e}")
            return False
    
    def download_translation(self, source="en", target="es"):
        """Download translation model"""
        print(f"\n📥 Downloading translation model {source}->{target}...")
        from transformers import MarianMTModel, MarianTokenizer
        
        model_name = f"Helsinki-NLP/opus-mt-{source}-{target}"
        
        try:
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            print(f"✅ Translation model downloaded")
            return True
        except Exception as e:
            print(f"❌ Failed: {e}")
            return False
    
    def download_all_recommended(self):
        """Download recommended models for 8GB RAM"""
        print("=" * 50)
        print("Downloading Recommended Models for 8GB RAM")
        print("=" * 50)
        
        results = {}
        
        # Core models (must have)
        results['whisper'] = self.download_whisper("medium")
        results['embeddings'] = self.download_embeddings("all-MiniLM-L6-v2")
        
        # Optional based on RAM
        import psutil
        available_ram = psutil.virtual_memory().available / (1024**3)
        
        if available_ram > 6:
            print("\n✅ Enough RAM for additional models")
            results['summarization'] = self.download_summarization()
            results['llm'] = self.download_llm(quantized=True)
        else:
            print("\n⚠️ Limited RAM, skipping heavy models")
            results['summarization'] = False
            results['llm'] = False
        
        # Print summary
        print("\n" + "=" * 50)
        print("Download Summary:")
        for model, success in results.items():
            status = "✅" if success else "❌"
            print(f"  {status} {model}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Download models for Whisper Learn")
    parser.add_argument("--whisper", choices=["tiny", "base", "small", "medium", "large"],
                       default="medium", help="Whisper model size")
    parser.add_argument("--embeddings", default="all-MiniLM-L6-v2",
                       help="Embeddings model name")
    parser.add_argument("--summarization", action="store_true",
                       help="Download summarization model")
    parser.add_argument("--llm", action="store_true",
                       help="Download LLM model")
    parser.add_argument("--all", action="store_true",
                       help="Download all models")
    
    args = parser.parse_args()
    
    downloader = ModelDownloader()
    
    if args.all:
        downloader.download_all_recommended()
    else:
        if args.whisper:
            downloader.download_whisper(args.whisper)
        if args.embeddings:
            downloader.download_embeddings(args.embeddings)
        if args.summarization:
            downloader.download_summarization()
        if args.llm:
            downloader.download_llm()

if __name__ == "__main__":
    main()