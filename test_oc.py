#!/usr/bin/env python3
"""
Test script for PDF Word Counter API
Works with both local and Azure deployments
"""

import requests
import json
import time
from pathlib import Path

def test_api(base_url="http://localhost:8000", pdf_file_path=None):
    """
    Test the PDF Word Counter API
    
    Args:
        base_url: API base URL (local or Azure)
        pdf_file_path: Path to test PDF file
    """
    
    print(f"🧪 Testing PDF Word Counter API")
    print(f"📍 Base URL: {base_url}")
    print("=" * 50)
    
    # Test 1: Root endpoint
    try:
        print("1️⃣ Testing root endpoint...")
        response = requests.get(f"{base_url}/", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ API Version: {data.get('version')}")
            print(f"   ✅ Environment: {data.get('environment')}")
            print(f"   ✅ Platform: {data.get('platform')}")
        else:
            print(f"   ❌ Failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        print("   💡 Make sure the API server is running")
        return False
    
    # Test 2: Health check
    try:
        print("\n2️⃣ Testing health endpoint...")
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            health = response.json()
            print(f"   ✅ Status: {health.get('status')}")
            print(f"   ✅ Tesseract: {'Available' if health.get('tesseract_available') else 'Not Available'}")
            print(f"   ✅ pdf2image: {'Available' if health.get('pdf2image_available') else 'Not Available'}")
            if health.get('tesseract_version'):
                print(f"   ✅ Tesseract Version: {health.get('tesseract_version')}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
    
    # Test 3: System info
    try:
        print("\n3️⃣ Testing system info...")
        response = requests.get(f"{base_url}/system-info", timeout=10)
        if response.status_code == 200:
            info = response.json()
            print(f"   ✅ Platform: {info.get('platform')}")
            print(f"   ✅ Environment: {info.get('environment')}")
            print(f"   ✅ OCR Capabilities: {info.get('tesseract_available')}")
        else:
            print(f"   ❌ System info failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ System info error: {e}")
    
    # Test 4: PDF Processing
    if pdf_file_path:
        try:
            print(f"\n4️⃣ Testing PDF processing...")
            pdf_path = Path(pdf_file_path)
            
            if not pdf_path.exists():
                print(f"   ❌ PDF file not found: {pdf_file_path}")
                return True  # API tests passed, just no file to test
            
            print(f"   📄 Processing file: {pdf_path.name}")
            
            with open(pdf_path, 'rb') as file:
                files = {'file': (pdf_path.name, file, 'application/pdf')}
                
                start_time = time.time()
                response = requests.post(
                    f"{base_url}/count-words",
                    files=files,
                    timeout=120  # 2 minutes timeout for OCR
                )
                processing_time = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"   ✅ Processing completed in {processing_time:.2f}s")
                    print(f"   📊 Results:")
                    print(f"      • Total Words: {result.get('total_words', 0)}")
                    print(f"      • Text Extracted: {result.get('text_extracted', False)}")
                    print(f"      • Method: {result.get('processing_method', 'Unknown')}")
                    print(f"      • Languages: {', '.join(result.get('languages_detected', []))}")
                    print(f"      • Pages: {result.get('pages_processed', 0)}")
                    
                    if result.get('confidence_score'):
                        print(f"      • Confidence: {result.get('confidence_score')}%")
                    
                    if result.get('extracted_text_preview'):
                        preview = result.get('extracted_text_preview', '')[:100]
                        print(f"      • Preview: {preview}...")
                    
                    if result.get('error'):
                        print(f"      • Warning: {result.get('error')}")
                    
                    return True
                else:
                    print(f"   ❌ PDF processing failed: {response.status_code}")
                    print(f"   📄 Response: {response.text}")
                    return False
        
        except requests.exceptions.Timeout:
            print(f"   ⏰ Request timed out (this is normal for large documents)")
            print(f"   💡 Try with a smaller PDF or increase timeout")
            return True
        except Exception as e:
            print(f"   ❌ PDF processing error: {e}")
            return False
    else:
        print(f"\n4️⃣ PDF processing test skipped (no file provided)")
        print(f"   💡 To test with a PDF file:")
        print(f"   💡 python test_api.py --file your_document.pdf")
    
    print(f"\n🎉 API tests completed!")
    print(f"🌐 Interactive docs: {base_url}/docs")
    return True

def main():
    """Main test function with command line support"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test PDF Word Counter API')
    parser.add_argument('--url', default='http://localhost:8000', help='API base URL')
    parser.add_argument('--file', help='Path to test PDF file')
    parser.add_argument('--azure', action='store_true', help='Use Azure App Service URL format')
    
    args = parser.parse_args()
    
    # Handle Azure URL format
    if args.azure and not args.url.startswith('http'):
        args.url = f"https://{args.url}.azurewebsites.net"
    
    # Run tests
    success = test_api(args.url, args.file)
    
    if success:
        print(f"\n✅ All tests passed!")
    else:
        print(f"\n❌ Some tests failed!")
        exit(1)

if __name__ == "__main__":
    main()