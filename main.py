from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import io
import re
import os
import logging
from typing import List
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PDF Word Counter API with OCR",
    description="Extract text from PDFs (regular and scanned) and count words using OCR",
    version="2.0.0"
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
    error: str = None

class PDFProcessor:
    def __init__(self):
        # Configure Tesseract for multiple languages
        self.tesseract_config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF .,:-/()<>'
        self.supported_languages = 'eng+ara'  # English + Arabic
        
        # Test Tesseract installation
        try:
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract version: {version}")
        except Exception as e:
            logger.error(f"Tesseract not available: {e}")
    
    def extract_text_from_regular_pdf(self, pdf_bytes: bytes) -> tuple:
        """Extract text from regular PDF using PyMuPDF"""
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = ""
            pages_count = doc.page_count
            
            for page_num in range(pages_count):
                page = doc[page_num]
                page_text = page.get_text()
                text += page_text + "\n"
            
            doc.close()
            return text.strip(), pages_count
        except Exception as e:
            logger.error(f"Error extracting text from regular PDF: {str(e)}")
            raise e
    
    def pdf_to_images(self, pdf_bytes: bytes) -> List[Image.Image]:
        """Convert PDF pages to images for OCR"""
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            images = []
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                # High resolution for better OCR (300 DPI)
                mat = fitz.Matrix(300/72, 300/72)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                images.append(img)
            
            doc.close()
            return images
        except Exception as e:
            logger.error(f"Error converting PDF to images: {str(e)}")
            raise e
    
    def extract_text_with_ocr(self, images: List[Image.Image]) -> tuple:
        """Extract text from images using Tesseract OCR"""
        try:
            all_text = ""
            total_confidence = 0
            confidence_count = 0
            
            for i, img in enumerate(images):
                logger.info(f"Processing page {i+1} with OCR...")
                
                # Extract text with confidence data
                try:
                    # First try with English and Arabic
                    text = pytesseract.image_to_string(
                        img, 
                        lang=self.supported_languages,
                        config=self.tesseract_config
                    )
                    
                    # Get confidence score
                    try:
                        data = pytesseract.image_to_data(
                            img, 
                            lang=self.supported_languages,
                            config=self.tesseract_config,
                            output_type=pytesseract.Output.DICT
                        )
                        
                        # Calculate average confidence
                        confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                        if confidences:
                            page_confidence = sum(confidences) / len(confidences)
                            total_confidence += page_confidence
                            confidence_count += 1
                    except:
                        pass
                    
                    all_text += text + "\n"
                    
                except Exception as page_error:
                    logger.warning(f"OCR failed for page {i+1}: {page_error}")
                    # Try with just English if Arabic fails
                    try:
                        text = pytesseract.image_to_string(
                            img, 
                            lang='eng',
                            config='--oem 3 --psm 6'
                        )
                        all_text += text + "\n"
                    except:
                        logger.error(f"OCR completely failed for page {i+1}")
                        continue
            
            avg_confidence = total_confidence / confidence_count if confidence_count > 0 else 0
            return all_text.strip(), avg_confidence
            
        except Exception as e:
            logger.error(f"Error during OCR processing: {str(e)}")
            raise e
    
    def count_words(self, text: str) -> int:
        """Count words in text (supports multiple languages)"""
        if not text or text.strip() == "":
            return 0
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Split by whitespace and filter out empty strings and single characters
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
    
    def get_text_preview(self, text: str, max_length: int = 200) -> str:
        """Get a preview of extracted text"""
        if not text:
            return ""
        
        # Clean text for preview
        preview = re.sub(r'\s+', ' ', text.strip())
        
        if len(preview) <= max_length:
            return preview
        
        return preview[:max_length] + "..."

pdf_processor = PDFProcessor()

@app.post("/count-words", response_model=WordCountResponse)
async def count_pdf_words(file: UploadFile = File(...)):
    """
    Count words in PDF file (supports both regular and scanned PDFs)
    
    - **file**: PDF file as byte array
    - Returns: Word count, processing method, detected languages, and confidence score
    """
    
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400, 
            detail="Only PDF files are supported"
        )
    
    # Read file content
    content = await file.read()
    
    # Check file size (limit to 15MB for OCR processing)
    if len(content) > 15 * 1024 * 1024:
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
    
    try:
        # Step 1: Try direct text extraction first
        logger.info("Attempting direct text extraction...")
        
        try:
            extracted_text, pages_count = pdf_processor.extract_text_from_regular_pdf(content)
            word_count = pdf_processor.count_words(extracted_text)
            
            # If we got meaningful text, return it
            if word_count > 5:  # Threshold for meaningful content
                languages = pdf_processor.detect_languages(extracted_text)
                preview = pdf_processor.get_text_preview(extracted_text)
                
                return WordCountResponse(
                    total_words=word_count,
                    text_extracted=True,
                    processing_method="direct_extraction",
                    languages_detected=languages,
                    pages_processed=pages_count,
                    extracted_text_preview=preview
                )
        
        except Exception as e:
            logger.warning(f"Direct text extraction failed: {str(e)}")
        
        # Step 2: Use OCR for scanned documents
        logger.info("Direct extraction yielded little text. Attempting OCR...")
        
        try:
            # Convert PDF to images
            images = pdf_processor.pdf_to_images(content)
            
            if not images:
                return WordCountResponse(
                    total_words=0,
                    text_extracted=False,
                    processing_method="ocr_failed",
                    languages_detected=[],
                    pages_processed=0,
                    error="Could not convert PDF to images for OCR"
                )
            
            # Extract text using OCR
            ocr_text, confidence = pdf_processor.extract_text_with_ocr(images)
            word_count = pdf_processor.count_words(ocr_text)
            languages = pdf_processor.detect_languages(ocr_text)
            preview = pdf_processor.get_text_preview(ocr_text)
            
            return WordCountResponse(
                total_words=word_count,
                text_extracted=True,
                processing_method="ocr",
                languages_detected=languages,
                pages_processed=len(images),
                confidence_score=round(confidence, 2),
                extracted_text_preview=preview
            )
            
        except Exception as e:
            logger.error(f"OCR processing failed: {str(e)}")
            return WordCountResponse(
                total_words=0,
                text_extracted=False,
                processing_method="failed",
                languages_detected=[],
                pages_processed=0,
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
        version = pytesseract.get_tesseract_version()
        return {
            "status": "healthy",
            "tesseract_version": str(version),
            "supported_languages": pdf_processor.supported_languages,
            "capabilities": ["text_extraction", "ocr", "multilingual"],
            "environment": "docker"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "capabilities": ["text_extraction_only"]
        }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "PDF Word Counter API with OCR",
        "version": "2.0.0",
        "description": "Extract text from both regular and scanned PDFs using OCR",
        "endpoints": {
            "POST /count-words": "Count words in PDF file (with OCR support)",
            "GET /health": "Health check",
            "GET /docs": "API documentation"
        },
        "features": [
            "Direct text extraction for regular PDFs",
            "OCR for scanned documents and images",
            "English and Arabic language support",
            "Confidence scoring for OCR results",
            "Text preview in response"
        ],
        "limits": {
            "max_file_size": "15MB",
            "supported_formats": ["PDF"],
            "supported_languages": ["English", "Arabic"]
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)