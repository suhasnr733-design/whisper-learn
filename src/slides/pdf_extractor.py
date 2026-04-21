"""
Extract text and images from PDF files
"""
import fitz  # PyMuPDF
from typing import List, Dict, Optional
from pathlib import Path
import base64
from io import BytesIO

class PDFExtractor:
    def __init__(self):
        self.doc = None
    
    def load_pdf(self, pdf_path: str):
        """Load PDF document"""
        self.doc = fitz.open(pdf_path)
        return self
    
    def extract_text(self, page_num: Optional[int] = None) -> Dict:
        """Extract text from PDF"""
        if self.doc is None:
            raise ValueError("Load PDF first with load_pdf()")
        
        result = {}
        
        if page_num is not None:
            page = self.doc[page_num]
            result[page_num] = {
                'text': page.get_text(),
                'page_num': page_num + 1
            }
        else:
            for page_num, page in enumerate(self.doc):
                result[page_num] = {
                    'text': page.get_text(),
                    'page_num': page_num + 1
                }
        
        return result
    
    def extract_images(self, page_num: Optional[int] = None) -> List[Dict]:
        """Extract images from PDF"""
        if self.doc is None:
            raise ValueError("Load PDF first with load_pdf()")
        
        images = []
        
        pages_to_process = [page_num] if page_num is not None else range(len(self.doc))
        
        for page_idx in pages_to_process:
            page = self.doc[page_idx]
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                pix = fitz.Pixmap(self.doc, xref)
                
                if pix.n - pix.alpha < 4:  # Can save as PNG
                    img_data = pix.tobytes("png")
                else:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                    img_data = pix.tobytes("png")
                
                # Convert to base64 for JSON
                img_b64 = base64.b64encode(img_data).decode('utf-8')
                
                images.append({
                    'page_num': page_idx + 1,
                    'image_index': img_index,
                    'data': img_b64,
                    'size': len(img_data)
                })
                
                pix = None  # Free memory
        
        return images
    
    def extract_metadata(self) -> Dict:
        """Extract PDF metadata"""
        if self.doc is None:
            raise ValueError("Load PDF first with load_pdf()")
        
        metadata = self.doc.metadata
        return {
            'title': metadata.get('title', ''),
            'author': metadata.get('author', ''),
            'subject': metadata.get('subject', ''),
            'keywords': metadata.get('keywords', ''),
            'page_count': len(self.doc)
        }
    
    def close(self):
        """Close PDF document"""
        if self.doc:
            self.doc.close()


class PDFTextExtractor:
    """Lightweight text-only extraction"""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
    
    def extract_all_text(self) -> str:
        """Extract all text from PDF"""
        import PyPDF2
        
        with open(self.pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = []
            for page in reader.pages:
                text.append(page.extract_text())
        
        return '\n'.join(text)
    
    def extract_page_text(self, page_number: int) -> str:
        """Extract text from specific page"""
        import PyPDF2
        
        with open(self.pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            if page_number < len(reader.pages):
                return reader.pages[page_number].extract_text()
        return ""