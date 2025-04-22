# backend/app/services/ocr_service.py

import logging
import io
import os
from PIL import Image
import pytesseract
from typing import Optional

log = logging.getLogger(__name__)

# Check if TESSERACT_CMD is set in environment variables
if tesseract_cmd := os.getenv("TESSERACT_CMD"):
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    log.info(f"Using custom Tesseract path: {tesseract_cmd}")
else:
    log.info("Using system Tesseract installation")

def perform_ocr(image_bytes: bytes) -> str:
    """
    Extract text from an image using OCR.
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        Extracted text as string
    """
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Perform OCR
        log.info("Performing OCR on image...")
        text = pytesseract.image_to_string(image)
        
        # Check if any text was extracted
        if not text or text.isspace():
            log.warning("OCR returned empty or whitespace-only text")
            return "No text detected in image"
        
        log.info(f"OCR successful, extracted {len(text)} characters")
        return text.strip()
    
    except Exception as e:
        log.error(f"OCR error: {str(e)}", exc_info=True)
        return f"OCR error: {str(e)}"
