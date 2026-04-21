"""
Voice Activity Detection for utterance segmentation
"""
import numpy as np
import webrtcvad
from collections import deque
from typing import List, Tuple, Generator

class VoiceActivityDetector:
    def __init__(self, mode: int = 3, sample_rate: int = 16000, frame_duration_ms: int = 30):
        """
        mode: 0-3, higher = more aggressive
        frame_duration_ms: 10, 20, or 30 ms
        """
        self.vad = webrtcvad.Vad(mode)
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_bytes = int(sample_rate * frame_duration_ms / 1000) * 2  # 16-bit = 2 bytes
        
    def is_speech(self, audio_chunk: bytes) -> bool:
        """Check if audio chunk contains speech"""
        try:
            return self.vad.is_speech(audio_chunk, self.sample_rate)
        except:
            return False
    
    def pcm_to_bytes(self, audio: np.ndarray) -> bytes:
        """Convert float32 PCM to int16 bytes"""
        int_audio = (audio * 32767).astype(np.int16)
        return int_audio.tobytes()
    
    def detect_utterances(self, audio: np.ndarray, 
                          padding_ms: int = 300) -> Generator[Tuple[float, float], None, None]:
        """
        Detect speech utterances in audio
        Yields (start_time, end_time) for each utterance
        """
        audio_bytes = self.pcm_to_bytes(audio)
        
        # Calculate number of frames
        num_frames = len(audio_bytes) // self.frame_bytes
        frame_duration_sec = self.frame_duration_ms / 1000.0
        
        speech_flags = []
        for i in range(num_frames):
            start_byte = i * self.frame_bytes
            end_byte = start_byte + self.frame_bytes
            frame = audio_bytes[start_byte:end_byte]
            speech_flags.append(self.is_speech(frame))
        
        # Add padding frames
        padding_frames = int(padding_ms / self.frame_duration_ms)
        
        # Merge speech segments
        utterances = []
        in_utterance = False
        utterance_start = 0
        
        for i, is_speech in enumerate(speech_flags):
            if is_speech and not in_utterance:
                in_utterance = True
                utterance_start = max(0, i - padding_frames) * frame_duration_sec
            elif not is_speech and in_utterance:
                # Check if we're just in a short pause
                pause_frames = 0
                for j in range(i, min(i + padding_frames, len(speech_flags))):
                    if speech_flags[j]:
                        pause_frames = 0
                        break
                    pause_frames += 1
                
                if pause_frames >= padding_frames:
                    in_utterance = False
                    utterance_end = min(len(speech_flags), i + padding_frames) * frame_duration_sec
                    utterances.append((utterance_start, utterance_end))
        
        # Handle last utterance
        if in_utterance:
            utterance_end = len(speech_flags) * frame_duration_sec
            utterances.append((utterance_start, utterance_end))
        
        return utterances
    
    def split_audio_by_utterances(self, audio: np.ndarray, sample_rate: int) -> List[np.ndarray]:
        """Split audio into separate utterances"""
        utterances = self.detect_utterances(audio)
        
        chunks = []
        for start, end in utterances:
            start_sample = int(start * sample_rate)
            end_sample = int(end * sample_rate)
            chunks.append(audio[start_sample:end_sample])
        
        return chunks


class StreamingVAD:
    """Real-time VAD for live transcription"""
    
    def __init__(self, mode: int = 3, sample_rate: int = 16000):
        self.vad = webrtcvad.Vad(mode)
        self.sample_rate = sample_rate
        self.buffer = deque()
        self.speech_buffer = deque()
        self.silence_counter = 0
        self.is_speaking = False
        self.frame_duration_ms = 30
        self.frame_size = int(sample_rate * self.frame_duration_ms / 1000)
        
    def process_chunk(self, audio_chunk: np.ndarray) -> dict:
        """Process audio chunk and return status"""
        # Convert to bytes
        int_audio = (audio_chunk * 32767).astype(np.int16)
        audio_bytes = int_audio.tobytes()
        
        # Check if chunk contains speech
        is_speech = self.vad.is_speech(audio_bytes, self.sample_rate)
        
        result = {
            'is_speech': is_speech,
            'utterance_started': False,
            'utterance_ended': False,
            'speech_buffer': None
        }
        
        if is_speech:
            self.speech_buffer.extend(audio_chunk)
            self.silence_counter = 0
            
            if not self.is_speaking:
                self.is_speaking = True
                result['utterance_started'] = True
        else:
            self.silence_counter += 1
            
            if self.is_speaking and self.silence_counter > 10:  # 300ms silence
                self.is_speaking = False
                result['utterance_ended'] = True
                result['speech_buffer'] = np.array(self.speech_buffer)
                self.speech_buffer.clear()
        
        return result