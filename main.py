import os
import platform
import logging
import cv2
import numpy as np
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import threading

# Configure logging first
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import pytesseract and configure paths
try:
    import pytesseract
    
    # Configure Tesseract path based on platform
    if platform.system() == "Windows":
        # Windows paths
        tesseract_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            r"C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe".format(os.getenv('USERNAME', 'User'))
        ]
        
        for path in tesseract_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                logger.info(f"✅ Found Tesseract at: {path}")
                break
        else:
            logger.warning("❌ Tesseract not found in standard Windows locations")
    
    elif platform.system() == "Linux":
        # Linux/Azure paths
        linux_paths = [
            "/usr/bin/tesseract",
            "/usr/local/bin/tesseract",
            "/opt/tesseract/bin/tesseract"
        ]
        
        # Check environment variable first
        tesseract_cmd = os.getenv('TESSERACT_CMD')
        if tesseract_cmd and os.path.exists(tesseract_cmd):
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            logger.info(f"✅ Using Tesseract from env var: {tesseract_cmd}")
        else:
            for path in linux_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    logger.info(f"✅ Found Tesseract at: {path}")
                    break
            else:
                logger.warning("❌ Tesseract not found in standard Linux locations")
    
    # Test Tesseract availability
    try:
        version = pytesseract.get_tesseract_version()
        logger.info(f"✅ Tesseract version: {version}")
        TESSERACT_AVAILABLE = True
    except Exception as e:
        logger.error(f"❌ Tesseract test failed: {e}")
        TESSERACT_AVAILABLE = False

except ImportError as e:
    logger.error(f"❌ Failed to import pytesseract: {e}")
    TESSERACT_AVAILABLE = False

# Import other required modules
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageEnhance, ImageFilter
import io
import re
import time
from typing import List, Optional, Tuple
from pydantic import BaseModel

# Try to import pdf2image and PyMuPDF
try:
    from pdf2image import convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
    logger.info("✅ pdf2image available")
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    logger.warning("❌ pdf2image not available")

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
    logger.info("✅ PyMuPDF available")
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("❌ PyMuPDF not available")

# Detect environment
IS_AZURE = os.getenv('WEBSITE_SITE_NAME') is not None
IS_DOCKER = os.path.exists('/.dockerenv')
IS_CLOUD = IS_AZURE or IS_DOCKER

logger.info(f"🌍 Environment: {'Azure' if IS_AZURE else 'Docker' if IS_DOCKER else 'Local'}")

# Initialize FastAPI app
app = FastAPI(
    title="Ultra-Fast PDF Word Counter API",
    description="Extract text from PDFs with multithreaded OCR support",
    version="5.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response models
class WordCountResponse(BaseModel):
    total_words: int
    text_extracted: bool
    processing_method: str
    languages_detected: List[str]
    pages_processed: int
    confidence_score: Optional[float] = None
    extracted_text_preview: Optional[str] = None
    processing_time: Optional[float] = None
    error: Optional[str] = None

class SystemInfo(BaseModel):
    platform: str
    environment: str
    tesseract_available: bool
    tesseract_version: Optional[str] = None
    tesseract_path: Optional[str] = None
    pdf2image_available: bool
    pymupdf_available: bool

class UltraFastPDFProcessor:
    def __init__(self):
        self.tesseract_available = TESSERACT_AVAILABLE
        self.pdf2image_available = PDF2IMAGE_AVAILABLE
        self.pymupdf_available = PYMUPDF_AVAILABLE
        
        # Enhanced settings for large document support
        if IS_CLOUD:
            self.dpi = 300
            self.scanned_dpi = 350  # Higher DPI for scanned docs
            self.max_pages = 10     # Allow more pages
            self.max_dimension = 1500
            self.max_workers = 2
        else:
            self.dpi = 400          # Higher DPI for text docs
            self.scanned_dpi = 400  # Higher DPI for scanned docs
            self.max_pages = 25     # Allow more pages
            self.max_dimension = 2000
            self.max_workers = 4
        
        # Simple language support for speed
        self.supported_languages = 'eng'
        
        logger.info(f"🚀 Ultra-fast processor - Text: {self.dpi}DPI, Scanned: {self.scanned_dpi}DPI")
        logger.info(f"⚡ Max workers: {self.max_workers}, Max pages: {self.max_pages}")
    
    def get_optimal_dpi_for_large_docs(self, page_count: int) -> int:
        """Get optimal DPI based on document size"""
        if page_count > 20:
            return max(250, self.scanned_dpi - 100)  # Much lower DPI for large docs
        elif page_count > 10:
            return max(300, self.scanned_dpi - 50)   # Slightly lower DPI
        else:
            return self.scanned_dpi  # Use full DPI for small docs
    
    def extract_text_pymupdf(self, pdf_bytes: bytes) -> tuple:
        """Extract text using PyMuPDF (fastest method)"""
        if not self.pymupdf_available:
            raise Exception("PyMuPDF not available")
        
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = ""
            pages_count = doc.page_count
            
            # Process more pages for large documents
            max_pages_to_process = min(pages_count, self.max_pages * 2) if pages_count > 15 else self.max_pages
            
            for page_num in range(min(pages_count, max_pages_to_process)):
                page = doc[page_num]
                page_text = page.get_text()
                text += page_text + "\n"
            
            doc.close()
            return text.strip(), pages_count
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed: {e}")
            raise e
    
    def pdf_to_images_fast(self, pdf_bytes: bytes) -> List[Image.Image]:
        """Ultra-fast PDF to image conversion with dynamic DPI for large documents"""
        if not self.pdf2image_available:
            raise Exception("pdf2image not available")
        
        try:
            # Get page count first
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            page_count = doc.page_count
            doc.close()
            
            # Use lower DPI for large documents
            if page_count > 15:
                dpi_to_use = 250  # Much lower DPI for large documents
                max_pages_to_process = min(page_count, 30)  # Process more pages
                logger.info(f"Large document detected ({page_count} pages) - using {dpi_to_use} DPI")
            elif page_count > 10:
                dpi_to_use = 300  # Medium DPI
                max_pages_to_process = min(page_count, 20)
                logger.info(f"Medium document detected ({page_count} pages) - using {dpi_to_use} DPI")
            else:
                dpi_to_use = self.scanned_dpi  # Full DPI for small docs
                max_pages_to_process = self.max_pages
                logger.info(f"Small document detected ({page_count} pages) - using {dpi_to_use} DPI")
            
            logger.info(f"Converting PDF: {page_count} pages, processing {max_pages_to_process} pages...")
            start_time = time.time()
            
            images = convert_from_bytes(
                pdf_bytes, 
                dpi=dpi_to_use,
                fmt='JPEG',
                thread_count=self.max_workers,
                first_page=1,
                last_page=max_pages_to_process,
                grayscale=True,
                transparent=False
            )
            
            # Less aggressive resizing for large documents
            optimized_images = []
            for img in images:
                width, height = img.size
                
                # Be less aggressive with large documents
                if page_count > 15:
                    max_dimension = self.max_dimension * 1.3  # Allow larger images
                else:
                    max_dimension = self.max_dimension
                
                if width > max_dimension:
                    ratio = max_dimension / width
                    new_height = int(height * ratio)
                    img = img.resize((int(max_dimension), new_height), Image.Resampling.LANCZOS)
                
                optimized_images.append(img)
            
            conversion_time = time.time() - start_time
            logger.info(f"⚡ Converted {len(optimized_images)} images in {conversion_time:.2f}s")
            
            return optimized_images
            
        except Exception as e:
            logger.error(f"Fast PDF conversion failed: {e}")
            raise e
    
    def process_single_image_ocr(self, args) -> tuple:
        """Process a single image with OCR - designed for multithreading"""
        img, page_num = args
        
        try:
            logger.info(f"⚡ Processing page {page_num} in thread...")
            
            # Single fast preprocessing
            cv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Simple threshold - fastest preprocessing
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed_img = Image.fromarray(thresh)
            
            # Single OCR attempt - fastest config
            text = pytesseract.image_to_string(
                processed_img, 
                config='--oem 3 --psm 6 -l eng'  # Fastest config
            )
            
            word_count = len(text.split())
            logger.info(f"⚡ Page {page_num}: {word_count} words")
            
            return text, page_num, word_count
            
        except Exception as e:
            logger.error(f"❌ Page {page_num} failed: {e}")
            return "", page_num, 0
    
    def extract_text_multithreaded_ocr(self, images: List[Image.Image]) -> tuple:
        """Ultra-fast multithreaded OCR processing"""
        all_text = ""
        total_words = 0
        
        try:
            logger.info(f"🚀 Starting MULTITHREADED OCR for {len(images)} pages...")
            start_time = time.time()
            
            # Prepare arguments for multithreading
            args_list = [(img, i+1) for i, img in enumerate(images)]
            
            # Process images in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                results = list(executor.map(self.process_single_image_ocr, args_list))
            
            # Combine results in order
            results.sort(key=lambda x: x[1])  # Sort by page number
            
            for text, page_num, word_count in results:
                if text.strip():
                    cleaned_text = self.clean_text_fast(text)
                    all_text += cleaned_text + "\n\n"
                    total_words += len(cleaned_text.split())
            
            processing_time = time.time() - start_time
            logger.info(f"⚡ MULTITHREADED OCR completed in {processing_time:.2f}s")
            
            # Simple confidence calculation
            confidence = min(total_words * 2, 100)  # Simple heuristic
            
            return all_text.strip(), confidence
            
        except Exception as e:
            logger.error(f"Multithreaded OCR failed: {e}")
            raise e
    
    def clean_text_fast(self, text: str) -> str:
        """Ultra-fast text cleaning"""
        if not text:
            return ""
        
        # Minimal cleaning for speed
        text = re.sub(r'[^\w\s\-.,!?]', ' ', text)
        text = re.sub(r'\s+', ' ', text.strip())
        
        return text
    
    def is_scanned_document_fast(self, text: str) -> bool:
        """Fast scanned document detection"""
        if not text or text.strip() == "":
            return True
        
        word_count = len(text.split())
        return word_count < 5  # Simple threshold
    
    def extract_text_ocr_ultra_fast(self, images: List[Image.Image]) -> tuple:
        """Ultra-fast OCR with automatic fallback"""
        if not self.tesseract_available:
            raise Exception("Tesseract OCR not available")
        
        try:
            # Quick test on first image
            if images:
                quick_text = pytesseract.image_to_string(images[0], config='--oem 3 --psm 6 -l eng')
                is_scanned = self.is_scanned_document_fast(quick_text)
                
                if is_scanned:
                    logger.info("📄 Using MULTITHREADED fast OCR...")
                    return self.extract_text_multithreaded_ocr(images)
                else:
                    logger.info("📄 Simple OCR sufficient...")
                    # For non-scanned, use simple single-threaded
                    all_text = ""
                    for i, img in enumerate(images):
                        text = pytesseract.image_to_string(img, config='--oem 3 --psm 6 -l eng')
                        all_text += text + "\n\n"
                    return all_text.strip(), 85
            
            return "", 0
                
        except Exception as e:
            logger.error(f"Ultra-fast OCR failed: {e}")
            raise e
    
    def count_words_advanced(self, text: str) -> int:
        """Advanced word counting - less aggressive for large documents"""
        if not text or text.strip() == "":
            return 0
        
        # Estimate document size
        raw_word_count = len(text.split())
        is_large_document = raw_word_count > 3000
        
        if is_large_document:
            logger.info(f"Large document detected ({raw_word_count} raw words) - using gentle cleaning")
            
            # GENTLE cleaning for large documents
            # Only remove the most obvious headers
            text = re.sub(r'Test Document - \d+ Words \(Font: \d+pt\)', '', text, flags=re.IGNORECASE)
            
            # Convert page numbers to spaces (don't remove completely)
            text = re.sub(r'Page \d+ of \d+', ' ', text, flags=re.IGNORECASE)
            
            # Very minimal punctuation cleaning
            text = re.sub(r'[^\w\s\-\'.,!?;:\n/]', ' ', text)
            text = re.sub(r'\s+', ' ', text.strip())
            
            # Minimal word filtering - keep almost everything
            words = text.split()
            valid_words = []
            
            for word in words:
                clean_word = word.strip('.,!?;:')
                
                # Very lenient - keep any word with letters or numbers
                if clean_word and len(clean_word) >= 1:
                    if re.search(r'[a-zA-Z0-9]', clean_word):
                        valid_words.append(clean_word)
            
            result = len(valid_words)
            logger.info(f"Large document: {raw_word_count} raw -> {result} cleaned words")
            return result
            
        else:
            # NORMAL cleaning for smaller documents
            text = re.sub(r'Test Document.*?pt\)', '', text, flags=re.IGNORECASE)
            text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
            text = re.sub(r'\d+/\d+', '', text)
            
            text = re.sub(r'[^\w\s\-\'.,!?;:]', ' ', text)
            text = re.sub(r'\s+', ' ', text.strip())
            
            words = text.split()
            valid_words = []
            
            for word in words:
                clean_word = re.sub(r'^[^\w]+|[^\w]+$', '', word)
                
                if clean_word and (re.search(r'[a-zA-Z]', clean_word) or clean_word.isdigit()):
                    if len(clean_word) == 1 and clean_word.lower() in ['a', 'i']:
                        valid_words.append(clean_word)
                    elif len(clean_word) == 1 and not clean_word.isdigit():
                        continue
                    else:
                        valid_words.append(clean_word)
            
            return len(valid_words)
    
    def count_words(self, text: str) -> int:
        """Use the advanced counting method"""
        return self.count_words_advanced(text)
    
    def detect_languages(self, text: str) -> List[str]:
        """Fast language detection"""
        languages = []
        
        if re.search(r'[a-zA-Z]', text):
            languages.append("English")
        if re.search(r'\d', text):
            languages.append("Numbers")
        
        return languages if languages else ["Unknown"]
    
    def get_text_preview(self, text: str, max_length: int = 200) -> str:
        """Fast text preview"""
        if not text:
            return ""
        
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."

# Initialize processor
pdf_processor = UltraFastPDFProcessor()

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "ULTRA-FAST PDF Word Counter API with Large Document Support",
        "version": "5.1.0",
        "environment": "Azure App Service" if IS_AZURE else "Docker" if IS_DOCKER else "Local",
        "platform": platform.system(),
        "capabilities": {
            "multithreaded_ocr": True,
            "ultra_fast_processing": True,
            "large_document_support": True,
            "dynamic_dpi_scaling": True,
            "target_speed": "5-15 seconds for scanned documents",
            "max_workers": pdf_processor.max_workers
        },
        "speed_optimizations": [
            f"Multithreaded OCR ({pdf_processor.max_workers} workers)",
            "Dynamic DPI (250-400 based on document size)",
            "Large document detection and optimization",
            "Gentle word cleaning for large documents",
            "Extended page processing for large documents"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    tesseract_version = None
    
    if pdf_processor.tesseract_available:
        try:
            tesseract_version = str(pytesseract.get_tesseract_version())
        except Exception:
            pass
    
    return {
        "status": "ultra_fast_with_large_doc_support",
        "tesseract_available": pdf_processor.tesseract_available,
        "tesseract_version": tesseract_version,
        "max_workers": pdf_processor.max_workers,
        "target_speed": "5-15 seconds for scanned documents",
        "large_document_support": True
    }

@app.get("/system-info", response_model=SystemInfo)
async def get_system_info():
    """Detailed system information"""
    tesseract_version = None
    tesseract_path = None
    
    if pdf_processor.tesseract_available:
        try:
            tesseract_version = str(pytesseract.get_tesseract_version())
            tesseract_path = getattr(pytesseract.pytesseract, 'tesseract_cmd', 'system')
        except Exception:
            pass
    
    return SystemInfo(
        platform=platform.system(),
        environment="Azure" if IS_AZURE else "Docker" if IS_DOCKER else "Local",
        tesseract_available=pdf_processor.tesseract_available,
        tesseract_version=tesseract_version,
        tesseract_path=tesseract_path,
        pdf2image_available=pdf_processor.pdf2image_available,
        pymupdf_available=pdf_processor.pymupdf_available
    )

@app.post("/count-words", response_model=WordCountResponse)
async def count_pdf_words(file: UploadFile = File(...)):
    """ULTRA-FAST endpoint with multithreading and large document support"""
    start_time = time.time()
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")
    
    file_size = len(content)
    max_size = 10 * 1024 * 1024
    if file_size > max_size:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum size: {max_size // (1024*1024)}MB")
    
    if file_size == 0:
        return WordCountResponse(
            total_words=0,
            text_extracted=False,
            processing_method="none",
            languages_detected=[],
            pages_processed=0,
            error="Empty file provided"
        )
    
    logger.info(f"🚀 ULTRA-FAST processing with large doc support: {file.filename} ({file_size:,} bytes)")
    
    try:
        # Method 1: Direct text extraction (still fastest)
        if pdf_processor.pymupdf_available:
            try:
                logger.info("⚡ Attempting ultra-fast direct extraction...")
                extracted_text, pages_count = pdf_processor.extract_text_pymupdf(content)
                word_count = pdf_processor.count_words(extracted_text)
                
                if word_count > 10:
                    languages = pdf_processor.detect_languages(extracted_text)
                    preview = pdf_processor.get_text_preview(extracted_text)
                    processing_time = time.time() - start_time
                    
                    logger.info(f"⚡ ULTRA-FAST direct extraction: {word_count} words in {processing_time:.2f}s")
                    
                    return WordCountResponse(
                        total_words=word_count,
                        text_extracted=True,
                        processing_method="ultra_fast_direct_extraction_large_doc_support",
                        languages_detected=languages,
                        pages_processed=pages_count,
                        extracted_text_preview=preview,
                        processing_time=round(processing_time, 2)
                    )
            except Exception as e:
                logger.warning(f"❌ Direct extraction failed: {e}")
        
        # Method 2: ULTRA-FAST Multithreaded OCR with large document support
        if not pdf_processor.tesseract_available:
            return WordCountResponse(
                total_words=0,
                text_extracted=False,
                processing_method="ocr_unavailable",
                languages_detected=[],
                pages_processed=0,
                error="OCR not available"
            )
        
        if not pdf_processor.pdf2image_available:
            return WordCountResponse(
                total_words=0,
                text_extracted=False,
                processing_method="pdf2image_unavailable",
                languages_detected=[],
                pages_processed=0,
                error="PDF conversion not available"
            )
        
        logger.info("🚀 Starting ULTRA-FAST multithreaded OCR with large document support...")
        
        try:
            images = pdf_processor.pdf_to_images_fast(content)
            
            if not images:
                return WordCountResponse(
                    total_words=0,
                    text_extracted=False,
                    processing_method="pdf_conversion_failed",
                    languages_detected=[],
                    pages_processed=0,
                    error="Could not convert PDF pages"
                )
            
            extracted_text, confidence = pdf_processor.extract_text_ocr_ultra_fast(images)
            
            word_count = pdf_processor.count_words(extracted_text)
            languages = pdf_processor.detect_languages(extracted_text)
            preview = pdf_processor.get_text_preview(extracted_text)
            processing_time = time.time() - start_time
            
            logger.info(f"🚀 ULTRA-FAST OCR with large doc support completed: {word_count} words in {processing_time:.2f}s")
            
            return WordCountResponse(
                total_words=word_count,
                text_extracted=True,
                processing_method="ultra_fast_multithreaded_ocr_large_doc_support",
                languages_detected=languages,
                pages_processed=len(images),
                confidence_score=round(confidence, 2) if confidence > 0 else None,
                extracted_text_preview=preview,
                processing_time=round(processing_time, 2)
            )
            
        except Exception as e:
            logger.error(f"❌ Ultra-fast OCR failed: {e}")
            return WordCountResponse(
                total_words=0,
                text_extracted=False,
                processing_method="ultra_fast_ocr_failed",
                languages_detected=[],
                pages_processed=0,
                error=f"Ultra-fast OCR failed: {str(e)}",
                processing_time=round(time.time() - start_time, 2)
            )
    
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.exception_handler(413)
async def file_too_large_handler(request, exc):
    return JSONResponse(
        status_code=413,
        content={"detail": "File too large. Maximum size is 10MB."}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again."}
    )

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting ULTRA-FAST PDF Word Counter API with Large Document Support...")
    logger.info(f"⚡ Multithreaded OCR with {pdf_processor.max_workers} workers")
    logger.info(f"📄 Large document support with dynamic DPI scaling")
    logger.info(f"🎯 Target: Text<1s, Scanned<15s, Large docs optimized")

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", os.getenv("WEBSITES_PORT", 8000)))
    host = "0.0.0.0" if IS_CLOUD else "127.0.0.1"
    
    print("🚀 Starting ULTRA-FAST PDF Word Counter API with Large Document Support...")
    print(f"⚡ Multithreaded processing: {pdf_processor.max_workers} workers")
    print(f"📄 Large document support: Dynamic DPI (250-400)")
    print(f"🎯 Speed target: 5-15 seconds for scanned documents")
    print(f"🌐 Server: http://{host}:{port}")
    
    uvicorn.run(app, host=host, port=port, access_log=True, log_level="info")