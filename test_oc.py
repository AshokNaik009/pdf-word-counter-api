# Add this to the top of your main.py after installing Tesseract

import pytesseract
import os
import platform

# Configure Tesseract path for Windows
if platform.system() == "Windows":
    # Common Tesseract installation paths
    tesseract_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe".format(os.getenv('USERNAME'))
    ]
    
    for path in tesseract_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            print(f"Found Tesseract at: {path}")
            break
    else:
        print("Tesseract not found. Please install Tesseract-OCR.")

# Test Tesseract
try:
    version = pytesseract.get_tesseract_version()
    print(f"Tesseract version: {version}")
except Exception as e:
    print(f"Tesseract error: {e}")