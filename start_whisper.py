#!/usr/bin/env python3
"""
Start Whisper Learn - Main Entry Point
"""
import subprocess
import sys
import os
import time
import threading
import webbrowser

def run_api():
    """Start FastAPI server"""
    os.environ["WHISPER_MODEL"] = "small"
    
    print("🚀 Starting API Server...")
    subprocess.run([
        sys.executable, "-c",
        "import uvicorn; from api.main_fixed import app; uvicorn.run(app, host='0.0.0.0', port=8000)"
    ])

def run_frontend():
    """Start Streamlit frontend"""
    time.sleep(3)  # Wait for API to start
    
    print("🎨 Starting Frontend...")
    subprocess.run([
        "streamlit", "run", "frontend/app_simple.py",
        "--server.port", "8501",
        "--server.address", "localhost"
    ])

def open_browser():
    """Open browser automatically"""
    time.sleep(5)
    webbrowser.open("http://localhost:8501")
    print("\n" + "="*50)
    print("✅ Whisper Learn is running!")
    print("📍 Frontend: http://localhost:8501")
    print("📍 API Docs: http://localhost:8000/docs")
    print("="*50 + "\n")

if __name__ == "__main__":
    print("="*50)
    print("🎙️ Whisper Learn - Starting Application")
    print("="*50)
    
    # Start API in background thread
    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()
    
    # Open browser
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    # Run frontend (this blocks)
    run_frontend()