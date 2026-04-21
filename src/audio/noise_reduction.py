"""
Advanced noise reduction techniques
"""
import numpy as np
from scipy import signal
from scipy.fft import rfft, irfft
from typing import Tuple, Optional

class NoiseReducer:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        
    def spectral_gating(self, audio: np.ndarray, 
                        noise_floor: Optional[np.ndarray] = None,
                        reduction_factor: float = 0.8) -> np.ndarray:
        """
        Spectral gating noise reduction
        """
        # Compute STFT
        f, t, Zxx = signal.stft(audio, fs=self.sample_rate, nperseg=1024)
        
        # Estimate noise floor if not provided
        if noise_floor is None:
            # Assume first 0.5 seconds is noise
            noise_samples = audio[:int(0.5 * self.sample_rate)]
            _, _, noise_stft = signal.stft(noise_samples, fs=self.sample_rate, nperseg=1024)
            noise_floor = np.mean(np.abs(noise_stft), axis=1, keepdims=True)
        
        # Apply spectral gating
        magnitude = np.abs(Zxx)
        mask = magnitude > (noise_floor * reduction_factor)
        Zxx_clean = Zxx * mask
        
        # Inverse STFT
        _, audio_clean = signal.istft(Zxx_clean, fs=self.sample_rate)
        
        return audio_clean
    
    def wiener_filter(self, audio: np.ndarray, noise_variance: float = 0.01) -> np.ndarray:
        """
        Wiener filter for noise reduction
        """
        from scipy.signal import wiener
        return wiener(audio, mysize=55)
    
    def moving_average_filter(self, audio: np.ndarray, window_size: int = 5) -> np.ndarray:
        """Simple moving average filter"""
        window = np.ones(window_size) / window_size
        return np.convolve(audio, window, mode='same')
    
    def adaptive_filter(self, audio: np.ndarray, noise_reference: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Adaptive noise cancellation (requires noise reference)
        """
        if noise_reference is None:
            return audio
        
        from scipy.signal import lms
        # Simple LMS adaptive filter
        filtered = lms(audio, noise_reference, 0.01)
        return filtered
    
    def automatic_gain_control(self, audio: np.ndarray, target_level: float = 0.3) -> np.ndarray:
        """Automatic gain control to maintain consistent volume"""
        rms = np.sqrt(np.mean(audio**2))
        if rms > 0:
            gain = target_level / rms
            gain = np.clip(gain, 0.1, 5.0)  # Limit gain range
            return audio * gain
        return audio


class BandpassFilter:
    """Bandpass filter for speech enhancement"""
    
    def __init__(self, sample_rate: int = 16000, lowcut: int = 80, highcut: int = 4000):
        self.sample_rate = sample_rate
        self.lowcut = lowcut
        self.highcut = highcut
        self._design_filter()
    
    def _design_filter(self):
        """Design bandpass filter"""
        nyquist = self.sample_rate / 2
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        self.b, self.a = signal.butter(6, [low, high], btype='band')
    
    def apply(self, audio: np.ndarray) -> np.ndarray:
        """Apply bandpass filter"""
        return signal.filtfilt(self.b, self.a, audio)