from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import io
import re
import os
from typing import Dict, Any
import logging
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="PDF Word Counter API",
    description="Extract text from PDFs (regular and scanned) and count words using OCR",
    version="1.0.0"
)

# Add CORS middleware for web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class WordCountResponse(BaseModel):
    total_words: int
    text_extracted: bool
    processing_method: str
    languages_detected: list
    error: str = None

class PDFProcessor:
    def __init__(self):
        # Configure Tesseract for English and Arabic
        self.tesseract_config = '--oem 3 --psm 6'
        self.supported_languages = 'eng+ara'  # English + Arabic
        
        # Set Tesseract path if needed (Render should have it in PATH)
        try:
            pytesseract.get_tesseract_version()
            logger.info("Tesseract found and working")
        except Exception as e:
            logger.error(f"Tesseract not found: {e}")
        
    def extract_text_from_regular_pdf(self, pdf_bytes: bytes) -> str:
        """Extract text from regular PDF using PyMuPDF"""
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = ""
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                page_text = page.get_text()
                text += page_text + "\n"
            
            doc.close()
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from regular PDF: {str(e)}")
            raise e
    
    def pdf_to_images(self, pdf_bytes: bytes) -> list:
        """Convert PDF pages to images for OCR"""
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            images = []
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                # Convert page to image (200 DPI for better performance on cloud)
                mat = fitz.Matrix(200/72, 200/72)  # Reduced from 300 for cloud performance
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                images.append(img)
            
            doc.close()
            return images
        except Exception as e:
            logger.error(f"Error converting PDF to images: {str(e)}")
            raise e
    
    def extract_text_with_ocr(self, images: list) -> str:
        """Extract text from images using Tesseract OCR"""
        try:
            all_text = ""
            
            for img in images:
                # Use Tesseract with English and Arabic language support
                text = pytesseract.image_to_string(
                    img, 
                    lang=self.supported_languages,
                    config=self.tesseract_config
                )
                all_text += text + "\n"
            
            return all_text.strip()
        except Exception as e:
            logger.error(f"Error during OCR processing: {str(e)}")
            raise e
    
    def count_words(self, text: str) -> int:
        """Count words in text (supports English and Arabic)"""
        if not text or text.strip() == "":
            return 0
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Split by whitespace and filter out empty strings
        words = [word for word in text.split() if word.strip()]
        
        return len(words)
    
    def detect_languages(self, text: str) -> list:
        """Detect if text contains English or Arabic characters"""
        languages = []
        
        # Check for English characters
        if re.search(r'[a-zA-Z]', text):
            languages.append("English")
        
        # Check for Arabic characters
        if re.search(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]', text):
            languages.append("Arabic")
        
        return languages if languages else ["Unknown"]

# Initialize processor
pdf_processor = PDFProcessor()

@app.post("/count-words", response_model=WordCountResponse)
async def count_pdf_words(file: UploadFile = File(...)):
    """
    Count words in a PDF file (supports both regular and scanned PDFs)
    
    - **file**: PDF file as byte array
    - Returns: Word count, processing method, and detected languages
    """
    
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400, 
            detail="Only PDF files are supported"
        )
    
    # Check file size (limit to 10MB for cloud deployment)
    file_size = 0
    content = await file.read()
    file_size = len(content)
    
    if file_size > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(
            status_code=413,
            detail="File too large. Maximum size is 10MB."
        )
    
    try:
        if len(content) == 0:
            return WordCountResponse(
                total_words=0,
                text_extracted=False,
                processing_method="none",
                languages_detected=[],
                error="Empty file provided"
            )
        
        # First, try to extract text directly from PDF
        try:
            extracted_text = pdf_processor.extract_text_from_regular_pdf(content)
            
            # Check if meaningful text was extracted
            word_count = pdf_processor.count_words(extracted_text)
            
            if word_count > 0:
                languages = pdf_processor.detect_languages(extracted_text)
                
                return WordCountResponse(
                    total_words=word_count,
                    text_extracted=True,
                    processing_method="direct_extraction",
                    languages_detected=languages
                )
        
        except Exception as e:
            logger.warning(f"Direct text extraction failed: {str(e)}")
        
        # If direct extraction fails or returns no text, use OCR
        try:
            logger.info("Attempting OCR processing...")
            
            # Convert PDF to images
            images = pdf_processor.pdf_to_images(content)
            
            if not images:
                return WordCountResponse(
                    total_words=0,
                    text_extracted=False,
                    processing_method="ocr",
                    languages_detected=[],
                    error="Could not convert PDF to images"
                )
            
            # Extract text using OCR
            ocr_text = pdf_processor.extract_text_with_ocr(images)
            word_count = pdf_processor.count_words(ocr_text)
            languages = pdf_processor.detect_languages(ocr_text)
            
            return WordCountResponse(
                total_words=word_count,
                text_extracted=True,
                processing_method="ocr",
                languages_detected=languages
            )
            
        except Exception as e:
            logger.error(f"OCR processing failed: {str(e)}")
            return WordCountResponse(
                total_words=0,
                text_extracted=False,
                processing_method="failed",
                languages_detected=[],
                error=f"OCR processing failed: {str(e)}"
            )
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test Tesseract installation
        version = pytesseract.get_tesseract_version()
        return {
            "status": "healthy",
            "tesseract_version": str(version),
            "supported_languages": pdf_processor.supported_languages,
            "environment": "render"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "PDF Word Counter API - Deployed on Render",
        "version": "1.0.0",
        "description": "Upload PDF files to count words using OCR for scanned documents",
        "endpoints": {
            "POST /count-words": "Count words in PDF file",
            "GET /health": "Health check",
            "GET /docs": "API documentation"
        },
        "limits": {
            "max_file_size": "10MB",
            "supported_formats": ["PDF"],
            "supported_languages": ["English", "Arabic"]
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)