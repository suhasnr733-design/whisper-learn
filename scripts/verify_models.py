# scripts/verify_models.py
#!/usr/bin/env python3
"""
Verify all models are correctly downloaded and working
"""
import sys
from pathlib import Path

def verify_whisper():
    """Verify Whisper model"""
    print("\n🔍 Verifying Whisper...")
    try:
        import whisper
        for size in ['tiny', 'base', 'small', 'medium']:
            try:
                model = whisper.load_model(size)
                print(f"  ✅ Whisper {size} - OK")
                del model
            except Exception as e:
                print(f"  ❌ Whisper {size} - Failed: {e}")
        return True
    except ImportError:
        print("  ❌ Whisper not installed")
        return False

def verify_embeddings():
    """Verify embeddings model"""
    print("\n🔍 Verifying Embeddings...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        # Test encoding
        embedding = model.encode(["test"])
        print(f"  ✅ Embeddings model OK (dim={len(embedding[0])})")
        return True
    except Exception as e:
        print(f"  ❌ Embeddings failed: {e}")
        return False

def verify_summarization():
    """Verify summarization model"""
    print("\n🔍 Verifying Summarization...")
    try:
        from transformers import pipeline
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
        test_text = "This is a test sentence for summarization."
        result = summarizer(test_text, max_length=10)
        print(f"  ✅ Summarization OK")
        return True
    except Exception as e:
        print(f"  ⚠️ Summarization not available: {e}")
        return False

def verify_llm():
    """Verify LLM model"""
    print("\n🔍 Verifying LLM...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Try loading quantized version
        try:
            from transformers import BitsAndBytesConfig
            import torch
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                "microsoft/phi-2",
                quantization_config=quantization_config,
                device_map="auto"
            )
            print(f"  ✅ LLM (quantized) OK")
            return True
        except:
            print(f"  ⚠️ LLM not available (requires more RAM)")
            return False
    except Exception as e:
        print(f"  ⚠️ LLM not available: {e}")
        return False

def main():
    print("=" * 50)
    print("Model Verification for Whisper Learn")
    print("=" * 50)
    
    results = {
        'whisper': verify_whisper(),
        'embeddings': verify_embeddings(),
        'summarization': verify_summarization(),
        'llm': verify_llm()
    }
    
    print("\n" + "=" * 50)
    print("Summary:")
    for model, status in results.items():
        status_str = "✅" if status else "⚠️"
        print(f"  {status_str} {model}")
    
    # Recommendations
    print("\n📋 Recommendations:")
    if not results['summarization']:
        print("  - Run: pip install transformers torch")
    if not results['llm']:
        print("  - For Q&A: pip install bitsandbytes accelerate")
        print("  - Or use smaller model: TinyLlama")
    
    return all(results.values())

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)