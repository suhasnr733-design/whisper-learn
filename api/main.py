"""
FastAPI main application
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uuid
import os
from pathlib import Path

# Import modules
import sys
sys.path.append('.')

from src.transcription.whisper_model import WhisperTranscriber
from src.audio.preprocessor import AudioPreprocessor
from src.slides.pdf_extractor import PDFExtractor
from src.summarization.hybrid_summarizer import HybridSummarizer
from src.utils.memory_manager import MemoryManager

# Initialize app
app = FastAPI(title="Whisper Learn API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
transcriber = None
summarizer = None
memory_manager = MemoryManager()

# Models
class TranscribeRequest(BaseModel):
    language: Optional[str] = None
    task: str = "transcribe"
    summarize: bool = False
    summary_style: str = "hybrid"

class QuestionRequest(BaseModel):
    question: str
    context: Optional[str] = None

# Helper functions
def get_transcriber():
    global transcriber
    if transcriber is None:
        import os
        model = os.environ.get('WHISPER_MODEL', 'medium')
        transcriber = WhisperTranscriber(model_size=model)
    return transcriber

def get_summarizer():
    global summarizer
    if summarizer is None:
        use_abs = os.environ.get('USE_SUMMARIZATION', 'true').lower() == 'true'
        summarizer = HybridSummarizer(use_abstractive=use_abs)
    return summarizer

# Routes
@app.get("/")
async def root():
    return {
        "message": "Whisper Learn API",
        "version": "1.0.0",
        "endpoints": ["/transcribe", "/health", "/memory"]
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/memory")
async def memory_status():
    return memory_manager.get_memory_usage()

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    language: Optional[str] = None,
    summarize: bool = False,
    summary_style: str = "hybrid"
):
    """Transcribe audio file"""
    
    # Save uploaded file
    task_id = str(uuid.uuid4())
    file_path = f"data/uploads/{task_id}_{file.filename}"
    os.makedirs("data/uploads", exist_ok=True)
    
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    try:
        # Transcribe
        transcriber = get_transcriber()
        
        # Check file length (simple heuristic)
        import librosa
        duration = librosa.get_duration(path=file_path)
        
        if duration > 3600:  # > 1 hour
            result = transcriber.transcribe_long(file_path)
        else:
            result = transcriber.transcribe(file_path, language=language)
        
        response = {
            "task_id": task_id,
            "transcript": result['text'],
            "segments": result.get('segments', []),
            "language": result.get('language', 'en'),
            "duration_seconds": duration
        }
        
        # Summarize if requested
        if summarize:
            summarizer = get_summarizer()
            summary = summarizer.summarize(result['text'], style=summary_style)
            response['summary'] = summary
        
        return JSONResponse(response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/transcribe-url")
async def transcribe_url(
    url: str,
    summarize: bool = False
):
    """Transcribe audio from URL"""
    import requests
    import tempfile
    
    # Download audio
    response = requests.get(url, stream=True)
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        for chunk in response.iter_content(chunk_size=8192):
            tmp.write(chunk)
        tmp_path = tmp.name
    
    try:
        transcriber = get_transcriber()
        result = transcriber.transcribe(tmp_path)
        
        response = {
            "transcript": result['text'],
            "segments": result.get('segments', []),
            "language": result.get('language', 'en')
        }
        
        if summarize:
            summarizer = get_summarizer()
            response['summary'] = summarizer.summarize(result['text'])
        
        return JSONResponse(response)
        
    finally:
        os.unlink(tmp_path)

@app.post("/align-slides")
async def align_slides(
    transcript: str,
    slide_file: UploadFile = File(...)
):
    """Align transcript with slides"""
    
    # Save slide file
    slide_path = f"data/temp/{uuid.uuid4()}_{slide_file.filename}"
    os.makedirs("data/temp", exist_ok=True)
    
    with open(slide_path, "wb") as f:
        f.write(await slide_file.read())
    
    try:
        # Extract slide text
        if slide_file.filename.endswith('.pdf'):
            extractor = PDFExtractor()
            extractor.load_pdf(slide_path)
            slides = extractor.extract_text()
            slide_texts = [slide['text'] for slide in slides.values()]
        else:
            from src.slides.ppt_extractor import PPTExtractor
            extractor = PPTExtractor()
            extractor.load_ppt(slide_path)
            slides = extractor.extract_text()
            slide_texts = [slide['text'] for slide in slides]
        
        # Align
        from src.slides.slide_aligner import SlideAligner
        aligner = SlideAligner()
        aligner.embed_slides(slide_texts)
        
        # Split transcript into segments
        sentences = transcript.split('. ')
        segments = [{'text': s + '.', 'start': i, 'end': i+1} 
                   for i, s in enumerate(sentences[:50])]  # Limit
        
        aligned = aligner.align_transcript(segments)
        
        return JSONResponse({
            "aligned_segments": aligned,
            "num_slides": len(slide_texts)
        })
        
    finally:
        if os.path.exists(slide_path):
            os.remove(slide_path)

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """Ask a question about transcribed content"""
    
    use_llm = os.environ.get('USE_LLM', 'false').lower() == 'true'
    
    if not use_llm:
        # Simple keyword matching fallback
        import re
        words = re.findall(r'\b\w+\b', request.question.lower())
        matches = []
        
        for word in words:
            if word in request.context.lower():
                matches.append(word)
        
        return {
            "question": request.question,
            "answer": f"Found {len(matches)} relevant terms: {', '.join(matches[:5])}",
            "confidence": len(matches) / max(len(words), 1)
        }
    
    try:
        from src.llm.phi_model import Phi2QA
        llm = Phi2QA(use_quantization=True)
        llm.load_model()
        
        answer = llm.generate_answer(request.question, request.context)
        
        return {
            "question": request.question,
            "answer": answer,
            "source": "phi-2"
        }
        
    except Exception as e:
        return {
            "question": request.question,
            "answer": f"Error generating answer: {str(e)}",
            "error": True
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)