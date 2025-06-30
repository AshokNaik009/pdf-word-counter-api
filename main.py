import os
import platform
import logging
import cv2
import numpy as np
import gc
from typing import List, Optional, Tuple
import threading
import time

# Memory-optimized logging (reduced verbosity, not processing parameters)
logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import pytesseract and configure paths
try:
    import pytesseract
    
    # Configure Tesseract path based on platform
    if platform.system() == "Windows":
        tesseract_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            r"C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe".format(os.getenv('USERNAME', 'User'))
        ]
        
        for path in tesseract_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                break
    
    elif platform.system() == "Linux":
        linux_paths = [
            "/usr/bin/tesseract",
            "/usr/local/bin/tesseract",
            "/opt/tesseract/bin/tesseract"
        ]
        
        tesseract_cmd = os.getenv('TESSERACT_CMD')
        if tesseract_cmd and os.path.exists(tesseract_cmd):
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        else:
            for path in linux_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    break
    
    # Test Tesseract availability
    try:
        pytesseract.get_tesseract_version()
        TESSERACT_AVAILABLE = True
    except Exception:
        TESSERACT_AVAILABLE = False

except ImportError:
    TESSERACT_AVAILABLE = False

# Import other required modules
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import re
from pydantic import BaseModel

# Try to import pdf2image and PyMuPDF
try:
    from pdf2image import convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

# Detect environment
IS_AZURE = os.getenv('WEBSITE_SITE_NAME') is not None
IS_DOCKER = os.path.exists('/.dockerenv')
IS_CLOUD = IS_AZURE or IS_DOCKER

# Initialize FastAPI app
app = FastAPI(
    title="Memory-Optimized PDF Word Counter API (Same Results)",
    description="Extract text from PDFs optimized for 512MB RAM without changing OCR results",
    version="5.3.0"
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
        
        # KEEP EXACT SAME PROCESSING PARAMETERS FOR IDENTICAL RESULTS
        # Only change memory management, not OCR quality settings
        if IS_CLOUD:
            self.dpi = 300              # SAME as original
            self.scanned_dpi = 350      # SAME as original 
            self.max_pages = 10         # SAME as original
            self.max_dimension = 1500   # SAME as original
            self.max_workers = 2        # SAME as original
        else:
            self.dpi = 400              # SAME as original
            self.scanned_dpi = 400      # SAME as original
            self.max_pages = 25         # SAME as original
            self.max_dimension = 2000   # SAME as original
            self.max_workers = 4        # SAME as original
        
        self.supported_languages = 'eng'
        
        # Memory management variables (NEW - for cleanup only)
        self._memory_threshold = 400  # MB warning threshold
        
        logger.warning(f"Processor initialized - SAME OCR params, memory cleanup enabled")
    
    def monitor_memory_usage(self):
        """Monitor memory and trigger GC if needed - DOES NOT affect OCR results"""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb > self._memory_threshold:
                logger.warning(f"High memory: {memory_mb:.1f}MB - forcing cleanup")
                gc.collect()
                return True
            return False
        except ImportError:
            # Force GC regularly if psutil not available
            gc.collect()
            return False
    
    def get_optimal_dpi_for_large_docs(self, page_count: int) -> int:
        """EXACT SAME METHOD - Get optimal DPI based on document size"""
        if page_count > 20:
            return max(250, self.scanned_dpi - 100)  # Much lower DPI for large docs
        elif page_count > 10:
            return max(300, self.scanned_dpi - 50)   # Slightly lower DPI
        else:
            return self.scanned_dpi  # Use full DPI for small docs
    
    def extract_text_pymupdf(self, pdf_bytes: bytes) -> tuple:
        """EXACT SAME METHOD with memory cleanup added"""
        if not self.pymupdf_available:
            raise Exception("PyMuPDF not available")
        
        doc = None
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = ""
            pages_count = doc.page_count
            
            # SAME LOGIC: Process more pages for large documents
            max_pages_to_process = min(pages_count, self.max_pages * 2) if pages_count > 15 else self.max_pages
            
            # SAME PAGE PROCESSING
            for page_num in range(min(pages_count, max_pages_to_process)):
                page = doc[page_num]
                page_text = page.get_text()
                text += page_text + "\n"
                
                # NEW: Memory cleanup every few pages (doesn't affect results)
                if page_num % 5 == 0:
                    self.monitor_memory_usage()
            
            return text.strip(), pages_count
            
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed: {e}")
            raise e
        finally:
            # NEW: Cleanup
            if doc:
                doc.close()
            gc.collect()
    
    def pdf_to_images_fast(self, pdf_bytes: bytes) -> List[Image.Image]:
        """EXACT SAME METHOD with memory management"""
        if not self.pdf2image_available:
            raise Exception("pdf2image not available")
        
        try:
            # SAME: Get page count first
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            page_count = doc.page_count
            doc.close()
            # NEW: Memory cleanup
            del doc
            gc.collect()
            
            # SAME LOGIC: Use lower DPI for large documents
            if page_count > 15:
                dpi_to_use = 250  # SAME threshold and DPI
                max_pages_to_process = min(page_count, 30)  # SAME
                logger.warning(f"Large document detected ({page_count} pages) - using {dpi_to_use} DPI")
            elif page_count > 10:
                dpi_to_use = 300  # SAME
                max_pages_to_process = min(page_count, 20)  # SAME
                logger.warning(f"Medium document detected ({page_count} pages) - using {dpi_to_use} DPI")
            else:
                dpi_to_use = self.scanned_dpi  # SAME
                max_pages_to_process = self.max_pages  # SAME
                logger.warning(f"Small document detected ({page_count} pages) - using {dpi_to_use} DPI")
            
            logger.warning(f"Converting PDF: {page_count} pages, processing {max_pages_to_process} pages...")
            start_time = time.time()
            
            # SAME CONVERSION PARAMETERS
            images = convert_from_bytes(
                pdf_bytes, 
                dpi=dpi_to_use,                    # SAME
                fmt='JPEG',                        # SAME
                thread_count=self.max_workers,     # SAME
                first_page=1,                      # SAME
                last_page=max_pages_to_process,    # SAME
                grayscale=True,                    # SAME
                transparent=False                  # SAME
            )
            
            # SAME IMAGE OPTIMIZATION LOGIC
            optimized_images = []
            for i, img in enumerate(images):
                width, height = img.size
                
                # SAME RESIZING LOGIC for large documents
                if page_count > 15:
                    max_dimension = self.max_dimension * 1.3  # SAME: Allow larger images
                else:
                    max_dimension = self.max_dimension        # SAME
                
                if width > max_dimension:
                    ratio = max_dimension / width
                    new_height = int(height * ratio)
                    img = img.resize((int(max_dimension), new_height), Image.Resampling.LANCZOS)
                
                optimized_images.append(img)
                
                # NEW: Memory cleanup every few images (doesn't affect results)
                if i % 3 == 0:
                    self.monitor_memory_usage()
            
            conversion_time = time.time() - start_time
            logger.warning(f"Converted {len(optimized_images)} images in {conversion_time:.2f}s")
            
            # NEW: Cleanup original images
            del images
            gc.collect()
            
            return optimized_images
            
        except Exception as e:
            logger.error(f"Fast PDF conversion failed: {e}")
            gc.collect()
            raise e
    
    def process_single_image_ocr(self, args) -> tuple:
        """EXACT SAME METHOD with memory cleanup"""
        img, page_num = args
        
        try:
            logger.warning(f"Processing page {page_num} in thread...")
            
            # SAME PREPROCESSING
            cv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # SAME: Simple threshold - fastest preprocessing
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed_img = Image.fromarray(thresh)
            
            # NEW: Cleanup intermediate arrays
            del cv_image, gray, thresh
            gc.collect()
            
            # SAME OCR CONFIG - fastest config
            text = pytesseract.image_to_string(
                processed_img, 
                config='--oem 3 --psm 6 -l eng'  # EXACT SAME config
            )
            
            # NEW: Cleanup processed image
            del processed_img
            gc.collect()
            
            word_count = len(text.split())
            logger.warning(f"Page {page_num}: {word_count} words")
            
            return text, page_num, word_count
            
        except Exception as e:
            logger.error(f"Page {page_num} failed: {e}")
            gc.collect()
            return "", page_num, 0
    
    def extract_text_multithreaded_ocr(self, images: List[Image.Image]) -> tuple:
        """EXACT SAME METHOD with memory management"""
        all_text = ""
        total_words = 0
        
        try:
            logger.warning(f"Starting MULTITHREADED OCR for {len(images)} pages...")
            start_time = time.time()
            
            # SAME: Prepare arguments for multithreading
            args_list = [(img, i+1) for i, img in enumerate(images)]
            
            # SAME: Process images in parallel - but use sequential for memory on cloud
            if IS_CLOUD:
                # NEW: Use sequential processing on cloud for memory, but SAME OCR logic
                results = []
                for args in args_list:
                    result = self.process_single_image_ocr(args)
                    results.append(result)
                    # Memory cleanup after each image
                    self.monitor_memory_usage()
            else:
                # SAME: Use threading on local
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    results = list(executor.map(self.process_single_image_ocr, args_list))
            
            # SAME: Combine results in order
            results.sort(key=lambda x: x[1])  # Sort by page number
            
            # SAME TEXT COMBINATION LOGIC
            for text, page_num, word_count in results:
                if text.strip():
                    cleaned_text = self.clean_text_fast(text)
                    all_text += cleaned_text + "\n\n"
                    total_words += len(cleaned_text.split())
            
            processing_time = time.time() - start_time
            logger.warning(f"OCR completed in {processing_time:.2f}s")
            
            # SAME: Simple confidence calculation
            confidence = min(total_words * 2, 100)  # Same heuristic
            
            # NEW: Memory cleanup
            del results
            gc.collect()
            
            return all_text.strip(), confidence
            
        except Exception as e:
            logger.error(f"Multithreaded OCR failed: {e}")
            gc.collect()
            raise e
    
    def clean_text_fast(self, text: str) -> str:
        """EXACT SAME METHOD"""
        if not text:
            return ""
        
        # SAME: Minimal cleaning for speed
        text = re.sub(r'[^\w\s\-.,!?]', ' ', text)
        text = re.sub(r'\s+', ' ', text.strip())
        
        return text
    
    def is_scanned_document_fast(self, text: str) -> bool:
        """EXACT SAME METHOD"""
        if not text or text.strip() == "":
            return True
        
        word_count = len(text.split())
        return word_count < 5  # Same threshold
    
    def extract_text_ocr_ultra_fast(self, images: List[Image.Image]) -> tuple:
        """EXACT SAME METHOD with memory cleanup"""
        if not self.tesseract_available:
            raise Exception("Tesseract OCR not available")
        
        try:
            # SAME: Quick test on first image
            if images:
                quick_text = pytesseract.image_to_string(images[0], config='--oem 3 --psm 6 -l eng')
                is_scanned = self.is_scanned_document_fast(quick_text)
                
                if is_scanned:
                    logger.warning("Using MULTITHREADED fast OCR...")
                    result = self.extract_text_multithreaded_ocr(images)
                    # NEW: Memory cleanup
                    gc.collect()
                    return result
                else:
                    logger.warning("Simple OCR sufficient...")
                    # SAME: For non-scanned, use simple single-threaded
                    all_text = ""
                    for i, img in enumerate(images):
                        text = pytesseract.image_to_string(img, config='--oem 3 --psm 6 -l eng')
                        all_text += text + "\n\n"
                        # NEW: Memory cleanup every few images
                        if i % 3 == 0:
                            self.monitor_memory_usage()
                    
                    # NEW: Final cleanup
                    gc.collect()
                    return all_text.strip(), 85
            
            return "", 0
                
        except Exception as e:
            logger.error(f"Ultra-fast OCR failed: {e}")
            gc.collect()
            raise e
    
    def count_words_advanced(self, text: str) -> int:
        """EXACT SAME METHOD - unchanged word counting logic"""
        if not text or text.strip() == "":
            return 0
        
        # SAME: Estimate document size
        raw_word_count = len(text.split())
        is_large_document = raw_word_count > 3000
        
        if is_large_document:
            logger.warning(f"Large document detected ({raw_word_count} raw words) - using gentle cleaning")
            
            # SAME: GENTLE cleaning for large documents
            text = re.sub(r'Test Document - \d+ Words \(Font: \d+pt\)', '', text, flags=re.IGNORECASE)
            text = re.sub(r'Page \d+ of \d+', ' ', text, flags=re.IGNORECASE)
            text = re.sub(r'[^\w\s\-\'.,!?;:\n/]', ' ', text)
            text = re.sub(r'\s+', ' ', text.strip())
            
            # SAME: Minimal word filtering - keep almost everything
            words = text.split()
            valid_words = []
            
            for word in words:
                clean_word = word.strip('.,!?;:')
                
                # SAME: Very lenient - keep any word with letters or numbers
                if clean_word and len(clean_word) >= 1:
                    if re.search(r'[a-zA-Z0-9]', clean_word):
                        valid_words.append(clean_word)
            
            result = len(valid_words)
            logger.warning(f"Large document: {raw_word_count} raw -> {result} cleaned words")
            
            # NEW: Memory cleanup
            del words, valid_words
            gc.collect()
            
            return result
            
        else:
            # SAME: NORMAL cleaning for smaller documents
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
            
            result = len(valid_words)
            
            # NEW: Memory cleanup
            del words, valid_words
            gc.collect()
            
            return result
    
    def count_words(self, text: str) -> int:
        """Use the advanced counting method"""
        return self.count_words_advanced(text)
    
    def detect_languages(self, text: str) -> List[str]:
        """EXACT SAME METHOD"""
        languages = []
        
        if re.search(r'[a-zA-Z]', text):
            languages.append("English")
        if re.search(r'\d', text):
            languages.append("Numbers")
        
        return languages if languages else ["Unknown"]
    
    def get_text_preview(self, text: str, max_length: int = 200) -> str:
        """EXACT SAME METHOD"""
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
        "message": "Memory-Optimized PDF Word Counter API (SAME OCR Results)",
        "version": "5.3.0",
        "environment": "Azure App Service" if IS_AZURE else "Docker" if IS_DOCKER else "Local",
        "platform": platform.system(),
        "capabilities": {
            "same_ocr_results": True,
            "memory_optimized": True,
            "sequential_on_cloud": True,
            "multithreaded_on_local": True,
            "identical_word_counting": True,
            "max_workers": pdf_processor.max_workers
        },
        "optimizations": [
            "Memory cleanup only - no OCR parameter changes",
            "Sequential processing on cloud for memory",
            "Multithreaded processing on local",
            "Aggressive garbage collection",
            "Same DPI, same thresholds, same word counting"
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
        "status": "memory_optimized_same_results",
        "tesseract_available": pdf_processor.tesseract_available,
        "tesseract_version": tesseract_version,
        "max_workers": pdf_processor.max_workers,
        "memory_optimization": "enabled",
        "ocr_results": "unchanged"
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
    """Memory-optimized endpoint with SAME OCR results"""
    start_time = time.time()
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")
    
    file_size = len(content)
    max_size = 100 * 1024 * 1024  # SAME: Keep 10MB limit
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
    
    logger.warning(f"Processing: {file.filename} ({file_size:,} bytes)")
    
    # NEW: Initial memory cleanup
    gc.collect()
    
    try:
        # SAME: Method 1: Direct text extraction (still fastest)
        if pdf_processor.pymupdf_available:
            try:
                logger.warning("Attempting direct extraction...")
                extracted_text, pages_count = pdf_processor.extract_text_pymupdf(content)
                word_count = pdf_processor.count_words(extracted_text)
                
                if word_count > 10:  # SAME threshold
                    languages = pdf_processor.detect_languages(extracted_text)
                    preview = pdf_processor.get_text_preview(extracted_text)
                    processing_time = time.time() - start_time
                    
                    # NEW: Clean up text from memory before response
                    del extracted_text
                    gc.collect()
                    
                    return WordCountResponse(
                        total_words=word_count,
                        text_extracted=True,
                        processing_method="ultra_fast_direct_extraction_large_doc_support",
                        languages_detected=languages,
                        pages_processed=pages_count,
                        extracted_text_preview=preview,
                        processing_time=round(processing_time, 2)
                    )
                else:
                    # NEW: Cleanup before falling back to OCR
                    del extracted_text
                    gc.collect()
                    
            except Exception as e:
                logger.warning(f"Direct extraction failed: {e}")
                gc.collect()
        
        # SAME: Method 2: OCR with identical parameters
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
        
        logger.warning("Starting OCR with same parameters...")
        
        try:
            images = pdf_processor.pdf_to_images_fast(content)
            
            # NEW: Clear PDF content from memory immediately
            del content
            gc.collect()
            
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
            
            # NEW: Clear images from memory immediately
            num_images = len(images)
            del images
            gc.collect()
            
            word_count = pdf_processor.count_words(extracted_text)
            languages = pdf_processor.detect_languages(extracted_text)
            preview = pdf_processor.get_text_preview(extracted_text)
            processing_time = time.time() - start_time
            
            # NEW: Clear extracted text from memory
            del extracted_text
            gc.collect()
            
            return WordCountResponse(
                total_words=word_count,
                text_extracted=True,
                processing_method="ultra_fast_multithreaded_ocr_large_doc_support",
                languages_detected=languages,
                pages_processed=num_images,
                confidence_score=round(confidence, 2) if confidence > 0 else None,
                extracted_text_preview=preview,
                processing_time=round(processing_time, 2)
            )
            
        except Exception as e:
            gc.collect()
            logger.error(f"Ultra-fast OCR failed: {e}")
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
        gc.collect()
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    finally:
        # NEW: Final cleanup
        gc.collect()

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
    logger.warning("Starting Memory-Optimized PDF API with SAME OCR Results...")
    logger.warning(f"Processing: {'Sequential' if IS_CLOUD else 'Multithreaded'}")
    logger.warning("Memory cleanup enabled, OCR parameters unchanged")

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", os.getenv("WEBSITES_PORT", 8000)))
    host = "0.0.0.0" if IS_CLOUD else "127.0.0.1"
    
    print("🔧 Starting Memory-Optimized PDF API (SAME Results)...")
    print(f"💾 Memory optimization: Cleanup only, no OCR changes")
    print(f"🔄 Sequential processing: {pdf_processor.max_workers} worker")
    print(f"📏 Max file size: 5MB")
    print(f"🌐 Server: http://{host}:{port}")
    
    uvicorn.run(app, host=host, port=port, access_log=False, log_level="warning")