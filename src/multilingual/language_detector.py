"""
Language detection for multilingual support
"""
from typing import Tuple, Optional
import numpy as np

class LanguageDetector:
    def __init__(self):
        self.model = None
        
    def _load_fasttext(self):
        """Load fasttext language detection model"""
        import fasttext
        
        model_path = "/tmp/lid.176.bin"
        import os
        if not os.path.exists(model_path):
            import urllib.request
            url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
            urllib.request.urlretrieve(url, model_path)
        
        self.model = fasttext.load_model(model_path)
    
    def detect(self, text: str) -> Tuple[str, float]:
        """
        Detect language of text
        Returns (language_code, confidence)
        """
        if self.model is None:
            self._load_fasttext()
        
        # Clean text
        text = text[:1000]  # Limit length for speed
        
        predictions = self.model.predict(text, k=1)
        lang = predictions[0][0].replace('__label__', '')
        confidence = float(predictions[1][0])
        
        return lang, confidence
    
    def detect_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Detect language for multiple texts"""
        results = []
        for text in texts:
            results.append(self.detect(text))
        return results


class WhisperLanguageDetector:
    """Use Whisper's language detection capability"""
    
    def __init__(self):
        import whisper
        self.model = whisper.load_model("tiny")
    
    def detect_from_audio(self, audio_path: str) -> Tuple[str, float]:
        """Detect language directly from audio"""
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
        
        # Detect language
        _, probs = self.model.detect_language(mel)
        detected_lang = max(probs, key=probs.get)
        
        return detected_lang, probs[detected_lang]


# Language codes mapping
LANGUAGE_CODES = {
    'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
    'it': 'Italian', 'pt': 'Portuguese', 'nl': 'Dutch', 'ru': 'Russian',
    'zh': 'Chinese', 'ja': 'Japanese', 'ko': 'Korean', 'ar': 'Arabic',
    'hi': 'Hindi', 'tr': 'Turkish', 'pl': 'Polish', 'uk': 'Ukrainian'
}

def get_language_name(code: str) -> str:
    return LANGUAGE_CODES.get(code, code)