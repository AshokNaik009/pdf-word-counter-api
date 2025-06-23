from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import io
import re
import os
import logging
import time
from typing import List
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PDF Word Counter API - pdf2image + pytesseract",
    description="Extract text from PDFs using pdf2image and pytesseract for reliable OCR",
    version="4.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class WordCountResponse(BaseModel):
    total_words: int
    text_extracted: bool
    processing_method: str
    languages_detected: List[str]
    pages_processed: int
    confidence_score: float = None
    extracted_text_preview: str = None
    processing_time: float = None
    error: str = None

class PDF2ImageProcessor:
    def __init__(self):
        # Configure Tesseract
        self.tesseract_config = '--oem 3 --psm 6'
        self.supported_languages = 'eng+ara'  # English + Arabic
        
        # Test if tesseract is available
        try:
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract version: {version}")
            self.tesseract_available = True
        except Exception as e:
            logger.error(f"Tesseract not available: {e}")
            self.tesseract_available = False
    
    def pdf_to_images_pdf2image(self, pdf_bytes: bytes) -> List[Image.Image]:
        """Convert PDF to images using pdf2image"""
        try:
            logger.info("Converting PDF to images using pdf2image...")
            start_time = time.time()
            
            # Convert PDF to images
            # Using poppler_path for Windows - adjust if needed
            images = convert_from_bytes(
                pdf_bytes, 
                dpi=300,  # High DPI for better OCR
                fmt='PNG',
                thread_count=2  # Parallel processing
            )
            
            conversion_time = time.time() - start_time
            logger.info(f"PDF converted to {len(images)} images in {conversion_time:.2f} seconds")
            
            return images
            
        except Exception as e:
            logger.error(f"Error converting PDF to images with pdf2image: {str(e)}")
            raise e
    
    def extract_text_with_tesseract(self, images: List[Image.Image]) -> tuple:
        """Extract text from images using pytesseract"""
        if not self.tesseract_available:
            raise Exception("Tesseract OCR not available")
        
        try:
            all_text = ""
            total_confidence = 0
            confidence_count = 0
            
            for i, img in enumerate(images):
                logger.info(f"Processing page {i+1}/{len(images)} with Tesseract OCR...")
                
                try:
                    start_time = time.time()
                    
                    # Optimize image for OCR
                    # Convert to RGB if needed
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Resize if too large (for performance)
                    width, height = img.size
                    if width > 2500:
                        ratio = 2500 / width
                        new_height = int(height * ratio)
                        img = img.resize((2500, new_height), Image.Resampling.LANCZOS)
                        logger.info(f"Resized image to {2500}x{new_height}")
                    
                    # Extract text with multilingual support
                    text = pytesseract.image_to_string(
                        img, 
                        lang=self.supported_languages,
                        config=self.tesseract_config
                    )
                    
                    # Get confidence data
                    try:
                        data = pytesseract.image_to_data(
                            img, 
                            lang=self.supported_languages,
                            config=self.tesseract_config,
                            output_type=pytesseract.Output.DICT
                        )
                        
                        # Calculate average confidence for this page
                        confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                        if confidences:
                            page_confidence = sum(confidences) / len(confidences)
                            total_confidence += page_confidence
                            confidence_count += 1
                            logger.info(f"Page {i+1} confidence: {page_confidence:.1f}%")
                    except Exception as conf_error:
                        logger.warning(f"Could not get confidence for page {i+1}: {conf_error}")
                    
                    processing_time = time.time() - start_time
                    word_count = len(text.split())
                    
                    logger.info(f"Page {i+1} processed in {processing_time:.2f}s, extracted {word_count} words")
                    
                    all_text += text + "\n\n"
                    
                except Exception as page_error:
                    logger.warning(f"OCR failed for page {i+1}: {page_error}")
                    
                    # Fallback to English-only
                    try:
                        logger.info(f"Trying English-only OCR for page {i+1}...")
                        text = pytesseract.image_to_string(
                            img, 
                            lang='eng',
                            config='--oem 3 --psm 6'
                        )
                        all_text += text + "\n\n"
                        logger.info(f"Page {i+1} fallback successful")
                    except Exception as fallback_error:
                        logger.error(f"Page {i+1} OCR completely failed: {fallback_error}")
                        continue
            
            avg_confidence = total_confidence / confidence_count if confidence_count > 0 else 0
            logger.info(f"OCR completed. Average confidence: {avg_confidence:.1f}%")
            
            return all_text.strip(), avg_confidence
            
        except Exception as e:
            logger.error(f"Error during OCR processing: {str(e)}")
            raise e
    
    def count_words(self, text: str) -> int:
        """Count words in text"""
        if not text or text.strip() == "":
            return 0
        
        # Remove extra whitespace and clean text
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Split by whitespace and filter meaningful words
        words = [word for word in text.split() if len(word.strip()) > 1]
        
        return len(words)
    
    def detect_languages(self, text: str) -> List[str]:
        """Detect languages in text"""
        languages = []
        
        # Check for English characters
        if re.search(r'[a-zA-Z]', text):
            languages.append("English")
        
        # Check for Arabic characters
        if re.search(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]', text):
            languages.append("Arabic")
        
        # Check for numbers
        if re.search(r'\d', text):
            languages.append("Numbers")
        
        return languages if languages else ["Unknown"]
    
    def get_text_preview(self, text: str, max_length: int = 300) -> str:
        """Get a preview of extracted text"""
        if not text:
            return ""
        
        # Clean text for preview
        preview = re.sub(r'\s+', ' ', text.strip())
        
        if len(preview) <= max_length:
            return preview
        
        return preview[:max_length] + "..."

# Initialize processor
pdf_processor = PDF2ImageProcessor()

@app.get("/test")
async def test_endpoint():
    """Test endpoint"""
    return {
        "status": "API is working",
        "timestamp": time.time(),
        "tesseract_available": pdf_processor.tesseract_available,
        "message": "PDF Word Counter with pdf2image + pytesseract"
    }

@app.post("/count-words", response_model=WordCountResponse)
async def count_pdf_words(file: UploadFile = File(...)):
    """
    Count words in PDF using pdf2image + pytesseract
    """
    
    start_time = time.time()
    
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400, 
            detail="Only PDF files are supported"
        )
    
    # Read file content
    content = await file.read()
    
    # Check file size
    if len(content) > 15 * 1024 * 1024:  # 15MB limit
        raise HTTPException(
            status_code=413,
            detail="File too large. Maximum size is 15MB."
        )
    
    if len(content) == 0:
        return WordCountResponse(
            total_words=0,
            text_extracted=False,
            processing_method="none",
            languages_detected=[],
            pages_processed=0,
            error="Empty file provided"
        )
    
    if not pdf_processor.tesseract_available:
        return WordCountResponse(
            total_words=0,
            text_extracted=False,
            processing_method="tesseract_unavailable",
            languages_detected=[],
            pages_processed=0,
            error="Tesseract OCR not available on this system"
        )
    
    try:
        logger.info(f"Processing PDF: {file.filename}")
        
        # Convert PDF to images using pdf2image
        try:
            images = pdf_processor.pdf_to_images_pdf2image(content)
            
            if not images:
                return WordCountResponse(
                    total_words=0,
                    text_extracted=False,
                    processing_method="conversion_failed",
                    languages_detected=[],
                    pages_processed=0,
                    error="Could not convert PDF to images"
                )
            
            # Limit pages for performance
            original_page_count = len(images)
            if len(images) > 10:
                logger.warning(f"Limiting to first 10 pages (found {len(images)} pages)")
                images = images[:10]
            
            # Extract text using Tesseract OCR
            logger.info(f"Starting OCR processing for {len(images)} pages...")
            extracted_text, confidence = pdf_processor.extract_text_with_tesseract(images)
            
            # Process results
            word_count = pdf_processor.count_words(extracted_text)
            languages = pdf_processor.detect_languages(extracted_text)
            preview = pdf_processor.get_text_preview(extracted_text)
            
            total_time = time.time() - start_time
            
            logger.info(f"Processing completed: {word_count} words, {languages}, {total_time:.2f}s")
            
            result = WordCountResponse(
                total_words=word_count,
                text_extracted=True,
                processing_method="pdf2image_tesseract",
                languages_detected=languages,
                pages_processed=len(images),
                confidence_score=round(confidence, 2) if confidence > 0 else None,
                extracted_text_preview=preview,
                processing_time=round(total_time, 2)
            )
            
            if original_page_count > len(images):
                result.error = f"Only processed first {len(images)} of {original_page_count} pages for performance"
            
            return result
            
        except Exception as e:
            logger.error(f"PDF processing failed: {str(e)}")
            return WordCountResponse(
                total_words=0,
                text_extracted=False,
                processing_method="failed",
                languages_detected=[],
                pages_processed=0,
                error=f"Processing failed: {str(e)}",
                processing_time=round(time.time() - start_time, 2)
            )
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy" if pdf_processor.tesseract_available else "limited",
        "tesseract_available": pdf_processor.tesseract_available,
        "tesseract_version": str(pytesseract.get_tesseract_version()) if pdf_processor.tesseract_available else "Not available",
        "supported_languages": pdf_processor.supported_languages if pdf_processor.tesseract_available else "None",
        "processing_method": "pdf2image + pytesseract"
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "PDF Word Counter API - pdf2image + pytesseract",
        "version": "4.0.0",
        "description": "Extract text from PDFs using pdf2image and pytesseract",
        "approach": "pdf2image converts PDF to images, pytesseract performs OCR",
        "advantages": [
            "Reliable PDF to image conversion",
            "Specialized for OCR workflows", 
            "Better handling of complex PDFs",
            "Optimized image preprocessing"
        ],
        "endpoints": {
            "POST /count-words": "Count words in PDF file",
            "GET /health": "Health check",
            "GET /test": "Simple test",
            "GET /docs": "API documentation"
        },
        "requirements": {
            "tesseract_ocr": "Required for OCR processing",
            "poppler": "Required for pdf2image conversion"
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)