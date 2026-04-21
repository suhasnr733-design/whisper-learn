"""
Fixed API for Whisper Learn
"""
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import whisper
import tempfile
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

app = FastAPI(title="Whisper Learn API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
print("🔄 Loading Whisper model...")
model = whisper.load_model("small")
print("✅ Model loaded!")

@app.get("/")
async def root():
    return {
        "message": "Whisper Learn API",
        "status": "running",
        "model": "whisper-small"
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """Transcribe audio file"""
    
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        content = await file.read()
        tmp.write(content)
        temp_path = tmp.name
    
    try:
        # Transcribe
        result = model.transcribe(temp_path)
        
        return {
            "success": True,
            "transcript": result["text"],
            "language": result["language"],
            "segments": result.get("segments", [])
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )
    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)