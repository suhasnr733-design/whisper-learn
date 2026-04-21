"""
Audio recorder module for capturing from microphone and system audio
"""
import pyaudio
import wave
import threading
import time
import numpy as np
from typing import Optional, Callable
from pathlib import Path
import os

class AudioRecorder:
    def __init__(self, sample_rate: int = 16000, channels: int = 1, chunk_size: int = 1024):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.format = pyaudio.paInt16
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        self.is_recording = False
        self.thread = None
        
    def start_recording(self, output_file: Optional[str] = None):
        """Start recording from microphone"""
        self.is_recording = True
        self.frames = []
        
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        if output_file:
            self.thread = threading.Thread(target=self._record_to_file, args=(output_file,))
        else:
            self.thread = threading.Thread(target=self._record_to_memory)
        
        self.thread.start()
        return self
    
    def _record_to_memory(self):
        """Record audio to memory buffer"""
        while self.is_recording:
            data = self.stream.read(self.chunk_size, exception_on_overflow=False)
            self.frames.append(data)
    
    def _record_to_file(self, output_file: str):
        """Record audio directly to file"""
        wf = wave.open(output_file, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.audio.get_sample_size(self.format))
        wf.setframerate(self.sample_rate)
        
        while self.is_recording:
            data = self.stream.read(self.chunk_size, exception_on_overflow=False)
            wf.writeframes(data)
        
        wf.close()
    
    def stop_recording(self, output_file: Optional[str] = None) -> Optional[np.ndarray]:
        """Stop recording and return audio data"""
        self.is_recording = False
        
        if self.thread:
            self.thread.join()
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        if output_file and not self.frames:
            # Already saved to file
            return None
        
        # Convert frames to numpy array
        if self.frames:
            audio_data = np.frombuffer(b''.join(self.frames), dtype=np.int16)
            audio_float = audio_data.astype(np.float32) / 32768.0
            
            if output_file:
                wf = wave.open(output_file, 'wb')
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(self.format))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(self.frames))
                wf.close()
            
            return audio_float
        
        return None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.stream:
            self.stream.close()
        self.audio.terminate()


class SystemAudioRecorder:
    """Record system audio (for meeting bots)"""
    
    def __init__(self):
        try:
            import sounddevice as sd
            self.sd = sd
        except ImportError:
            raise ImportError("pip install sounddevice for system audio recording")
    
    def list_devices(self):
        """List available audio devices"""
        return self.sd.query_devices()
    
    def record_system_audio(self, duration: int, output_file: str, device_id: Optional[int] = None):
        """Record system audio for specified duration"""
        if device_id is None:
            # Try to find loopback device
            devices = self.list_devices()
            for i, dev in enumerate(devices):
                if 'loopback' in dev['name'].lower() or 'stereo mix' in dev['name'].lower():
                    device_id = i
                    break
        
        if device_id is None:
            raise RuntimeError("No loopback device found. Install VB-Cable or BlackHole")
        
        recording = self.sd.rec(
            int(duration * 16000),
            samplerate=16000,
            channels=1,
            dtype='float32',
            device=device_id
        )
        self.sd.wait()
        
        # Save to file
        import soundfile as sf
        sf.write(output_file, recording, 16000)
        
        return output_file