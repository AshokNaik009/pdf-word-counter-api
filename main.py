import os
import platform
import pytesseract

# Configure Tesseract path for Windows FIRST
if platform.system() == "Windows":
    tesseract_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe".format(os.getenv('USERNAME'))
    ]
    
    for path in tesseract_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            print(f"✅ Found Tesseract at: {path}")
            break
    else:
        print("❌ Tesseract not found. Please install Tesseract-OCR.")

# Now import other modules
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pdf2image import convert_from_bytes
from PIL import Image
import io
import re
import logging
import time
from typing import List
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PDF Word Counter API - Windows OCR",
    description="Extract text from PDFs using pdf2image and pytesseract on Windows",
    version="4.1.0"
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

class WindowsOCRProcessor:
    def __init__(self):
        # Configure Tesseract
        self.tesseract_config = '--oem 3 --psm 6'
        self.supported_languages = 'eng+ara'  # English + Arabic
        
        # Test if tesseract is available
        try:
            version = pytesseract.get_tesseract_version()
            logger.info(f"✅ Tesseract version: {version}")
            self.tesseract_available = True
        except Exception as e:
            logger.error(f"❌ Tesseract not available: {e}")
            self.tesseract_available = False
    
    def pdf_to_images(self, pdf_bytes: bytes) -> List[Image.Image]:
        """Convert PDF to images using pdf2image"""
        try:
            logger.info("Converting PDF to images using pdf2image...")
            start_time = time.time()
            
            # Convert PDF to images
            images = convert_from_bytes(
                pdf_bytes, 
                dpi=300,  # High DPI for better OCR
                fmt='PNG',
                thread_count=1  # Single thread for Windows stability
            )
            
            conversion_time = time.time() - start_time
            logger.info(f"PDF converted to {len(images)} images in {conversion_time:.2f} seconds")
            
            return images
            
        except Exception as e:
            logger.error(f"Error converting PDF to images: {str(e)}")
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
                    
                    # Convert to RGB if needed
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Resize if too large (for Windows performance)
                    width, height = img.size
                    if width > 2000:
                        ratio = 2000 / width
                        new_height = int(height * ratio)
                        img = img.resize((2000, new_height), Image.Resampling.LANCZOS)
                        logger.info(f"Resized image to {2000}x{new_height} for performance")
                    
                    # Extract text with multilingual support
                    text = pytesseract.image_to_string(
                        img, 
                        lang=self.supported_languages,
                        config=self.tesseract_config
                    )
                    
                    # Get confidence data (simplified for Windows)
                    try:
                        data = pytesseract.image_to_data(
                            img, 
                            lang='eng',  # Use English for confidence calculation
                            config='--oem 3 --psm 6',
                            output_type=pytesseract.Output.DICT
                        )
                        
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
        
        text = re.sub(r'\s+', ' ', text.strip())
        words = [word for word in text.split() if len(word.strip()) > 1]
        return len(words)
    
    def detect_languages(self, text: str) -> List[str]:
        """Detect languages in text"""
        languages = []
        
        if re.search(r'[a-zA-Z]', text):
            languages.append("English")
        
        if re.search(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]', text):
            languages.append("Arabic")
        
        if re.search(r'\d', text):
            languages.append("Numbers")
        
        return languages if languages else ["Unknown"]
    
    def get_text_preview(self, text: str, max_length: int = 300) -> str:
        """Get a preview of extracted text"""
        if not text:
            return ""
        
        preview = re.sub(r'\s+', ' ', text.strip())
        if len(preview) <= max_length:
            return preview
        return preview[:max_length] + "..."

# Initialize processor
pdf_processor = WindowsOCRProcessor()

@app.get("/test")
async def test_endpoint():
    """Test endpoint"""
    return {
        "status": "API is working on Windows",
        "timestamp": time.time(),
        "tesseract_available": pdf_processor.tesseract_available,
        "tesseract_path": pytesseract.pytesseract.tesseract_cmd if hasattr(pytesseract.pytesseract, 'tesseract_cmd') else "Default",
        "platform": platform.system(),
        "message": "PDF Word Counter with Windows OCR support"
    }

@app.post("/count-words", response_model=WordCountResponse)
async def count_pdf_words(file: UploadFile = File(...)):
    """Count words in PDF using OCR on Windows"""
    
    start_time = time.time()
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files supported")
    
    content = await file.read()
    
    if len(content) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=413, detail="File too large. Maximum 10MB.")
    
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
            error="Tesseract OCR not available. Please check installation."
        )
    
    try:
        logger.info(f"Processing PDF: {file.filename}")
        
        # Convert PDF to images
        try:
            images = pdf_processor.pdf_to_images(content)
            
            if not images:
                return WordCountResponse(
                    total_words=0,
                    text_extracted=False,
                    processing_method="conversion_failed",
                    languages_detected=[],
                    pages_processed=0,
                    error="Could not convert PDF to images"
                )
            
            # Limit pages for Windows performance
            original_page_count = len(images)
            if len(images) > 5:
                logger.warning(f"Limiting to first 5 pages for Windows performance (found {len(images)} pages)")
                images = images[:5]
            
            # Extract text using OCR
            logger.info(f"Starting OCR processing for {len(images)} pages...")
            extracted_text, confidence = pdf_processor.extract_text_with_tesseract(images)
            
            # Process results
            word_count = pdf_processor.count_words(extracted_text)
            languages = pdf_processor.detect_languages(extracted_text)
            preview = pdf_processor.get_text_preview(extracted_text)
            
            total_time = time.time() - start_time
            
            logger.info(f"✅ Processing completed: {word_count} words, {languages}, {total_time:.2f}s")
            
            result = WordCountResponse(
                total_words=word_count,
                text_extracted=True,
                processing_method="pdf2image_tesseract_windows",
                languages_detected=languages,
                pages_processed=len(images),
                confidence_score=round(confidence, 2) if confidence > 0 else None,
                extracted_text_preview=preview,
                processing_time=round(total_time, 2)
            )
            
            if original_page_count > len(images):
                result.error = f"Processed {len(images)} of {original_page_count} pages for performance"
            
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
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check"""
    tesseract_info = "Not available"
    if pdf_processor.tesseract_available:
        try:
            tesseract_info = str(pytesseract.get_tesseract_version())
        except:
            tesseract_info = "Available but version unknown"
    
    return {
        "status": "healthy" if pdf_processor.tesseract_available else "limited",
        "tesseract_available": pdf_processor.tesseract_available,
        "tesseract_version": tesseract_info,
        "tesseract_path": getattr(pytesseract.pytesseract, 'tesseract_cmd', 'Default'),
        "platform": platform.system(),
        "supported_languages": pdf_processor.supported_languages if pdf_processor.tesseract_available else "None",
        "processing_method": "pdf2image + pytesseract on Windows"
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "PDF Word Counter API - Windows OCR Version",
        "version": "4.1.0",
        "platform": platform.system(),
        "tesseract_configured": pdf_processor.tesseract_available,
        "description": "Extract text from PDFs using pdf2image and pytesseract on Windows",
        "capabilities": [
            "Text-based PDF processing",
            "OCR for scanned documents" if pdf_processor.tesseract_available else "OCR unavailable",
            "English and Arabic support" if pdf_processor.tesseract_available else "Limited language support"
        ],
        "endpoints": {
            "POST /count-words": "Count words in PDF file",
            "GET /health": "Health check with Tesseract status",
            "GET /test": "Test endpoint with system info",
            "GET /docs": "API documentation"
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting PDF Word Counter API on Windows...")
    print(f"✅ Tesseract available: {pdf_processor.tesseract_available}")
    if pdf_processor.tesseract_available:
        print(f"✅ Tesseract path: {getattr(pytesseract.pytesseract, 'tesseract_cmd', 'Default')}")
    print("🌐 Access at: http://localhost:8000")
    print("📚 API docs at: http://localhost:8000/docs")
    uvicorn.run(app, host="127.0.0.1", port=8000)