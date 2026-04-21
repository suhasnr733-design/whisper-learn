"""
Speaker diarization using pyannote
"""
import numpy as np
from typing import List, Dict, Tuple
import torch

class SpeakerDiarizer:
    def __init__(self, use_auth_token: str = None):
        """
        Requires huggingface token for pyannote models
        """
        self.use_auth_token = use_auth_token
        self.pipeline = None
        
    def _load_model(self):
        """Lazy load diarization model"""
        if self.pipeline is None:
            try:
                from pyannote.audio import Pipeline
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization",
                    use_auth_token=self.use_auth_token
                )
                # Move to GPU if available
                if torch.cuda.is_available():
                    self.pipeline.to(torch.device("cuda"))
            except ImportError:
                raise ImportError("pip install pyannote.audio for speaker diarization")
    
    def diarize(self, audio_path: str) -> List[Dict]:
        """
        Perform speaker diarization
        Returns list of segments with speaker labels
        """
        self._load_model()
        
        # Run diarization
        diarization = self.pipeline(audio_path)
        
        # Convert to list of segments
        segments = []
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                'start': segment.start,
                'end': segment.end,
                'speaker': speaker
            })
        
        return segments
    
    def assign_speakers_to_transcript(self, 
                                       transcript_segments: List[Dict],
                                       diarization_segments: List[Dict]) -> List[Dict]:
        """
        Assign speaker labels to transcript segments
        """
        for trans_seg in transcript_segments:
            trans_start = trans_seg.get('start', 0)
            trans_end = trans_seg.get('end', 0)
            
            # Find overlapping diarization segment
            for diar_seg in diarization_segments:
                if (diar_seg['start'] <= trans_end and 
                    diar_seg['end'] >= trans_start):
                    trans_seg['speaker'] = diar_seg['speaker']
                    break
            else:
                trans_seg['speaker'] = 'UNKNOWN'
        
        return transcript_segments


class SimpleSpeakerClustering:
    """
    Lightweight speaker clustering using voice characteristics
    (No external API required)
    """
    
    def __init__(self, num_speakers: int = 2):
        self.num_speakers = num_speakers
        
    def extract_voice_features(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract MFCC features for voice characterization"""
        import librosa
        
        mfcc = librosa.feature.mfcc(
            y=audio, 
            sr=sample_rate, 
            n_mfcc=13,
            n_fft=2048,
            hop_length=512
        )
        
        # Aggregate features
        features = np.mean(mfcc, axis=1)
        return features
    
    def cluster_utterances(self, utterances: List[np.ndarray], sample_rate: int) -> List[int]:
        """
        Cluster utterances by speaker
        Returns list of speaker IDs for each utterance
        """
        from sklearn.cluster import KMeans
        
        # Extract features for each utterance
        features = []
        for utt in utterances:
            feat = self.extract_voice_features(utt, sample_rate)
            features.append(feat)
        
        features = np.array(features)
        
        if len(features) < self.num_speakers:
            return [0] * len(features)
        
        # Cluster
        kmeans = KMeans(n_clusters=min(self.num_speakers, len(features)))
        labels = kmeans.fit_predict(features)
        
        return labels.tolist()