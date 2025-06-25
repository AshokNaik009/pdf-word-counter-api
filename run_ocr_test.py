#!/usr/bin/env python3
"""
OCR Accuracy Test Runner
Run this script to test OCR accuracy on your test documents
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the tester (save the previous script as 'ocr_accuracy_tester.py')
from ocr_accuracy_tester import OCRAccuracyTester

def main():
    print("🔍 OCR Accuracy Test Runner")
    print("=" * 50)
    
    # Check if test folder exists
    test_folder = "test_document_5"
    if not os.path.exists(test_folder):
        print(f"❌ Error: Test folder '{test_folder}' not found!")
        print(f"Please make sure the folder exists in: {os.path.abspath('.')}")
        return
    
    # Count PDF files
    pdf_files = [f for f in os.listdir(test_folder) if f.lower().endswith('.pdf')]
    print(f"📁 Found {len(pdf_files)} PDF files in '{test_folder}'")
    
    if len(pdf_files) == 0:
        print("❌ No PDF files found in the test folder!")
        return
    
    # List the files
    print("\n📄 Files to be processed:")
    for i, filename in enumerate(sorted(pdf_files), 1):
        print(f"  {i:2d}. {filename}")
    
    # Ask for confirmation
    print(f"\n🚀 Ready to test OCR accuracy on {len(pdf_files)} files.")
    response = input("Continue? (y/N): ").strip().lower()
    
    if response not in ['y', 'yes']:
        print("Test cancelled.")
        return
    
    # Run the test
    try:
        tester = OCRAccuracyTester(test_folder)
        tester.run_accuracy_test()
        
        print("\n✅ Test completed successfully!")
        print(f"📊 Check the 'ocr_test_results' folder for detailed reports.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print("Please check the logs above for more details.")

if __name__ == "__main__":
    main()