"""
Multilingual support module
"""
from .language_detector import LanguageDetector, WhisperLanguageDetector, get_language_name
from .translator import OPUSTranslator, LightweightTranslator
from .cross_lingual import CrossLingualSummarizer

__all__ = [
    'LanguageDetector',
    'WhisperLanguageDetector',
    'get_language_name',
    'OPUSTranslator',
    'LightweightTranslator',
    'CrossLingualSummarizer'
]