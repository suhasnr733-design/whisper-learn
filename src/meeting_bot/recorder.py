"""
Meeting audio recorder
"""
import wave
import pyaudio
import threading
import numpy as np
from typing import Optional

class MeetingRecorder:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.channels = 1
        self.chunk_size = 1024
        self.format = pyaudio.paInt16
        
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        self.is_recording = False
        self.thread = None
        
    def start_recording(self, output_file: str):
        """Start recording meeting audio"""
        self.is_recording = True
        self.frames = []
        
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        self.thread = threading.Thread(target=self._record, args=(output_file,))
        self.thread.start()
        
    def _record(self, output_file: str):
        """Recording loop"""
        wf = wave.open(output_file, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.audio.get_sample_size(self.format))
        wf.setframerate(self.sample_rate)
        
        while self.is_recording:
            data = self.stream.read(self.chunk_size, exception_on_overflow=False)
            wf.writeframes(data)
            self.frames.append(data)
        
        wf.close()
    
    def stop_recording(self):
        """Stop recording"""
        self.is_recording = False
        
        if self.thread:
            self.thread.join()
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
    
    def get_audio_numpy(self) -> Optional[np.ndarray]:
        """Get recorded audio as numpy array"""
        if self.frames:
            audio_data = np.frombuffer(b''.join(self.frames), dtype=np.int16)
            return audio_data.astype(np.float32) / 32768.0
        return None
    
    def __del__(self):
        self.audio.terminate()