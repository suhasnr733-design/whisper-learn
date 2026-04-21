"""
Whisper model integration for speech-to-text
"""
import whisper
import torch
import numpy as np
from typing import Dict, List, Optional, Union
from pathlib import Path
import gc
import time

class WhisperTranscriber:
    def __init__(self, model_size: str = "medium", device: str = None):
        """
        model_size: "tiny", "base", "small", "medium", "large"
        """
        self.model_size = model_size
        self.model_sizes_mb = {
            "tiny": 75, "base": 145, "small": 488, "medium": 1500, "large": 2900
        }
        
        # Auto-select device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Loading Whisper {model_size} on {self.device}...")
        start_time = time.time()
        
        self.model = whisper.load_model(model_size, device=self.device)
        
        print(f"Loaded in {time.time() - start_time:.1f}s (~{self.model_sizes_mb[model_size]}MB)")
    
    def transcribe(self, 
                   audio_path: Union[str, np.ndarray],
                   language: Optional[str] = None,
                   task: str = "transcribe",
                   verbose: bool = False,
                   word_timestamps: bool = True) -> Dict:
        """
        Transcribe audio file or numpy array
        """
        # Handle numpy array input
        if isinstance(audio_path, np.ndarray):
            # Save temporarily
            import soundfile as sf
            temp_path = "/tmp/whisper_temp.wav"
            sf.write(temp_path, audio_path, 16000)
            audio_path = temp_path
        
        result = self.model.transcribe(
            audio_path,
            language=language,
            task=task,
            verbose=verbose,
            word_timestamps=word_timestamps,
            fp16=(self.device == "cuda")
        )
        
        return result
    
    def transcribe_long(self, audio_path: str, chunk_duration: int = 60, **kwargs) -> Dict:
        """
        Transcribe long audio files by chunking
        """
        import librosa
        
        # Load full audio
        audio, sr = librosa.load(audio_path, sr=16000)
        chunk_samples = chunk_duration * sr
        
        all_segments = []
        full_text = []
        
        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i:i+chunk_samples]
            
            # Save chunk temporarily
            temp_path = f"/tmp/whisper_chunk_{i}.wav"
            import soundfile as sf
            sf.write(temp_path, chunk, sr)
            
            # Transcribe chunk
            result = self.model.transcribe(temp_path, **kwargs)
            full_text.append(result['text'])
            
            # Adjust timestamps
            for segment in result.get('segments', []):
                segment['start'] += i / sr
                segment['end'] += i / sr
                all_segments.append(segment)
            
            # Cleanup
            import os
            os.remove(temp_path)
            gc.collect()
        
        return {
            'text': ' '.join(full_text),
            'segments': all_segments,
            'language': result.get('language', 'en')
        }
    
    def transcribe_batch(self, audio_paths: List[str], **kwargs) -> List[Dict]:
        """Transcribe multiple files"""
        results = []
        for path in audio_paths:
            results.append(self.transcribe(path, **kwargs))
            gc.collect()
        return results
    
    def get_available_memory(self) -> Dict:
        """Check memory usage"""
        if self.device == "cuda":
            memory = torch.cuda.memory_allocated() / 1024**2
            total = torch.cuda.get_device_properties(0).total_memory / 1024**2
            return {"used_mb": memory, "total_mb": total}
        else:
            import psutil
            return {"used_mb": psutil.Process().memory_info().rss / 1024**2}
    
    def unload(self):
        """Unload model to free memory"""
        del self.model
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        print("Model unloaded")


class WhisperBatchProcessor:
    """Process audio in batches for better throughput"""
    
    def __init__(self, transcriber: WhisperTranscriber, batch_size: int = 4):
        self.transcriber = transcriber
        self.batch_size = batch_size
    
    def process_batch(self, audio_batch: List[np.ndarray]) -> List[str]:
        """Process batch of audio arrays"""
        results = []
        for audio in audio_batch:
            result = self.transcriber.transcribe(audio)
            results.append(result['text'])
        return results