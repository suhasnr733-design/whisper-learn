# test_setup.py
"""Test if all imports work"""

print("Testing imports...")

try:
    import whisper
    print("✅ Whisper imported")
except Exception as e:
    print(f"❌ Whisper: {e}")

try:
    import torch
    print(f"✅ PyTorch imported (CUDA: {torch.cuda.is_available()})")
except Exception as e:
    print(f"❌ PyTorch: {e}")

try:
    from fastapi import FastAPI
    print("✅ FastAPI imported")
except Exception as e:
    print(f"❌ FastAPI: {e}")

try:
    import streamlit as st
    print("✅ Streamlit imported")
except Exception as e:
    print(f"❌ Streamlit: {e}")

try:
    from sentence_transformers import SentenceTransformer
    print("✅ Sentence Transformers imported")
except Exception as e:
    print(f"❌ Sentence Transformers: {e}")

print("\n" + "="*40)
print("Testing local imports...")

# Test src imports
import sys
sys.path.insert(0, '.')

try:
    from src.audio.recorder import AudioRecorder
    print("✅ Audio Recorder imported")
except Exception as e:
    print(f"❌ Audio Recorder: {e}")

try:
    from src.transcription.whisper_model import WhisperTranscriber
    print("✅ Whisper Transcriber imported")
except Exception as e:
    print(f"❌ Whisper Transcriber: {e}")

try:
    from src.summarization.hybrid_summarizer import HybridSummarizer
    print("✅ Hybrid Summarizer imported")
except Exception as e:
    print(f"❌ Hybrid Summarizer: {e}")

print("\n✅ Setup test complete!")