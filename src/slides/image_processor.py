"""
Image processing for slides (OCR, feature extraction)
"""
from typing import List, Dict, Optional
import base64
from io import BytesIO

class ImageProcessor:
    def __init__(self):
        self.ocr = None
        
    def _load_ocr(self):
        """Lazy load OCR engine"""
        if self.ocr is None:
            try:
                import pytesseract
                from PIL import Image
                self.ocr = pytesseract
                self.pil_image = Image
            except ImportError:
                print("OCR not available. Install pytesseract and PIL")
                self.ocr = False
    
    def extract_text_from_image(self, image_data: bytes) -> str:
        """Extract text from image using OCR"""
        self._load_ocr()
        
        if not self.ocr:
            return ""
        
        from PIL import Image
        import io
        
        image = Image.open(io.BytesIO(image_data))
        text = self.ocr.image_to_string(image)
        
        return text.strip()
    
    def extract_text_from_base64(self, base64_str: str) -> str:
        """Extract text from base64 encoded image"""
        image_data = base64.b64decode(base64_str)
        return self.extract_text_from_image(image_data)
    
    def get_image_features(self, image_data: bytes) -> Dict:
        """Extract image features for alignment"""
        from PIL import Image
        import numpy as np
        import io
        
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to grayscale
        gray = image.convert('L')
        
        # Get basic features
        width, height = image.size
        pixels = np.array(gray)
        
        return {
            'width': width,
            'height': height,
            'mean_brightness': float(np.mean(pixels)),
            'std_brightness': float(np.std(pixels)),
            'has_text': self.extract_text_from_image(image_data) != ""
        }


class OCRProcessor:
    """Batch OCR processing for slides"""
    
    def __init__(self):
        self.processor = ImageProcessor()
    
    def process_slide_batch(self, slide_images: List[bytes]) -> List[str]:
        """Process multiple slides"""
        results = []
        for img in slide_images:
            text = self.processor.extract_text_from_image(img)
            results.append(text)
        return results
    
    def should_process_image(self, image_data: bytes, min_text_length: int = 10) -> bool:
        """Determine if image likely contains text"""
        features = self.processor.get_image_features(image_data)
        
        # Simple heuristic: if image is mostly dark/light, likely has text
        if features['mean_brightness'] < 50 or features['mean_brightness'] > 200:
            return True
        
        return False