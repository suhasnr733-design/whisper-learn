"""
Slide processing module
"""
from .pdf_extractor import PDFExtractor, PDFTextExtractor
from .ppt_extractor import PPTExtractor, LightweightPPTExtractor
from .slide_aligner import SlideAligner, TFIDFSlideAligner
from .image_processor import ImageProcessor, OCRProcessor

__all__ = [
    'PDFExtractor',
    'PDFTextExtractor',
    'PPTExtractor',
    'LightweightPPTExtractor',
    'SlideAligner',
    'TFIDFSlideAligner',
    'ImageProcessor',
    'OCRProcessor'
]