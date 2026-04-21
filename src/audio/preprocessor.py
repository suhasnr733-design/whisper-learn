"""
Audio preprocessing: noise reduction, normalization, resampling
"""
import numpy as np
import librosa
from scipy import signal
from typing import Tuple, Optional
import warnings

class AudioPreprocessor:
    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr
        
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file with error handling"""
        try:
            audio, sr = librosa.load(audio_path, sr=None)
            return audio, sr
        except Exception as e:
            raise RuntimeError(f"Failed to load audio: {e}")
    
    def resample(self, audio: np.ndarray, original_sr: int) -> np.ndarray:
        """Resample audio to target sample rate"""
        if original_sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=original_sr, target_sr=self.target_sr)
        return audio
    
    def normalize(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range"""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        return audio
    
    def remove_silence(self, audio: np.ndarray, top_db: int = 20) -> np.ndarray:
        """Remove silence from beginning and end"""
        audio_trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        return audio_trimmed
    
    def reduce_noise(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Simple noise reduction using spectral gating"""
        try:
            import noisereduce as nr
            # Estimate noise from first 0.5 seconds
            noise_sample = audio[:int(0.5 * sample_rate)]
            audio_clean = nr.reduce_noise(
                y=audio,
                sr=sample_rate,
                y_noise=noise_sample,
                prop_decrease=0.8
            )
            return audio_clean
        except ImportError:
            # Fallback: simple high-pass filter
            warnings.warn("noisereduce not installed, using simple filter")
            b, a = signal.butter(5, 100, btype='high', fs=sample_rate)
            return signal.filtfilt(b, a, audio)
    
    def apply_preemphasis(self, audio: np.ndarray, coef: float = 0.97) -> np.ndarray:
        """Apply pre-emphasis filter to boost high frequencies"""
        return np.append(audio[0], audio[1:] - coef * audio[:-1])
    
    def process(self, audio_path: str, remove_silence: bool = True, 
                denoise: bool = True, normalize: bool = True) -> np.ndarray:
        """Full preprocessing pipeline"""
        audio, sr = self.load_audio(audio_path)
        audio = self.resample(audio, sr)
        
        if denoise:
            audio = self.reduce_noise(audio, self.target_sr)
        
        if remove_silence:
            audio = self.remove_silence(audio)
        
        if normalize:
            audio = self.normalize(audio)
        
        return audio
    
    def save_audio(self, audio: np.ndarray, output_path: str, sample_rate: int = 16000):
        """Save processed audio to file"""
        import soundfile as sf
        sf.write(output_path, audio, sample_rate)


class AudioChunker:
    """Split long audio into manageable chunks"""
    
    def __init__(self, chunk_duration: int = 30, overlap_duration: int = 2):
        self.chunk_duration = chunk_duration  # seconds
        self.overlap_duration = overlap_duration  # seconds
    
    def chunk_audio(self, audio: np.ndarray, sample_rate: int) -> list:
        """Split audio into overlapping chunks"""
        chunk_samples = self.chunk_duration * sample_rate
        overlap_samples = self.overlap_duration * sample_rate
        step_samples = chunk_samples - overlap_samples
        
        chunks = []
        for start in range(0, len(audio), step_samples):
            end = min(start + chunk_samples, len(audio))
            chunk = audio[start:end]
            
            # Pad last chunk if needed
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
            
            chunks.append({
                'audio': chunk,
                'start_time': start / sample_rate,
                'end_time': end / sample_rate
            })
            
            if end == len(audio):
                break
        
        return chunks