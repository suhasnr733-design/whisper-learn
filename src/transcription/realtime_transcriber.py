"""
Real-time transcription using Whisper
"""
import asyncio
import numpy as np
from typing import Callable, Optional
from collections import deque
import time
import threading
import queue

class RealtimeTranscriber:
    def __init__(self, model_size: str = "small", sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.audio_buffer = deque(maxlen=sample_rate * 30)  # 30 seconds buffer
        self.transcriber = None
        self.is_running = False
        self.model_size = model_size
        
    def _init_model(self):
        """Lazy load model"""
        if self.transcriber is None:
            from .whisper_model import WhisperTranscriber
            self.transcriber = WhisperTranscriber(self.model_size)
    
    async def stream_transcribe(self, 
                                audio_stream: np.ndarray,
                                callback: Callable,
                                chunk_duration: float = 3.0):
        """
        Stream transcription from audio chunks
        """
        self._init_model()
        self.is_running = True
        
        accumulated_audio = []
        last_process_time = time.time()
        
        for chunk in audio_stream:
            if not self.is_running:
                break
            
            accumulated_audio.extend(chunk)
            
            # Process every N seconds
            if time.time() - last_process_time >= chunk_duration:
                if len(accumulated_audio) > 0:
                    audio_array = np.array(accumulated_audio)
                    
                    # Transcribe in background
                    result = self.transcriber.transcribe(audio_array)
                    
                    await callback({
                        'text': result['text'],
                        'timestamp': time.time(),
                        'is_partial': True
                    })
                    
                    # Keep last 2 seconds for context
                    keep_samples = int(2 * self.sample_rate)
                    accumulated_audio = accumulated_audio[-keep_samples:]
                    last_process_time = time.time()
            
            await asyncio.sleep(0.01)
    
    def stop(self):
        self.is_running = False


class StreamingBuffer:
    """Buffer for streaming audio"""
    
    def __init__(self, sample_rate: int = 16000, buffer_seconds: int = 10):
        self.sample_rate = sample_rate
        self.buffer_size = sample_rate * buffer_seconds
        self.buffer = deque(maxlen=self.buffer_size)
        self.lock = threading.Lock()
    
    def add_audio(self, audio_chunk: np.ndarray):
        """Add audio chunk to buffer"""
        with self.lock:
            self.buffer.extend(audio_chunk)
    
    def get_audio(self, seconds: float) -> np.ndarray:
        """Get last N seconds of audio"""
        samples = int(seconds * self.sample_rate)
        with self.lock:
            buffer_list = list(self.buffer)
            if len(buffer_list) >= samples:
                return np.array(buffer_list[-samples:])
            else:
                return np.array(buffer_list)
    
    def clear(self):
        """Clear buffer"""
        with self.lock:
            self.buffer.clear()


class AudioStreamSimulator:
    """Simulate audio stream from file for testing"""
    
    def __init__(self, audio_path: str, chunk_size: int = 16000):
        import librosa
        self.audio, self.sr = librosa.load(audio_path, sr=16000)
        self.chunk_size = chunk_size
        self.position = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.position >= len(self.audio):
            raise StopIteration
        
        chunk = self.audio[self.position:self.position + self.chunk_size]
        self.position += self.chunk_size
        return chunk