import os
import re
import time
import logging
from datetime import datetime
from typing import List, Dict, Tuple
import json

# Import the existing processor from main.py
try:
    from main import UniversalPDFProcessor
except ImportError:
    print("❌ Could not import UniversalPDFProcessor from main.py")
    print("Make sure main.py is in the same directory and contains the UniversalPDFProcessor class.")
    print("Current directory:", os.path.abspath('.'))
    raise

class OCRAccuracyTester:
    def __init__(self, test_folder_path: str = "test_document_5"):
        self.test_folder_path = test_folder_path
        self.pdf_processor = UniversalPDFProcessor()
        self.results = []
        self.summary_stats = {
            'total_files': 0,
            'successful_ocr': 0,
            'failed_ocr': 0,
            'average_accuracy': 0.0,
            'total_processing_time': 0.0
        }
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Create results directory
        self.results_dir = "ocr_test_results"
        os.makedirs(self.results_dir, exist_ok=True)
    
    def extract_expected_word_count(self, filename: str) -> int:
        """
        Extract expected word count from filename
        Expected format: test_document_50words_expected0.2p_actual1p_f16.pdf
        """
        try:
            # Look for pattern like "50words" in the specific format
            match = re.search(r'test_document_(\d+)words', filename.lower())
            if match:
                return int(match.group(1))
            else:
                # Fallback: Look for any pattern like "50words" or "50_words"
                match = re.search(r'(\d+)_?words?', filename.lower())
                if match:
                    return int(match.group(1))
                else:
                    self.logger.warning(f"Could not extract word count from filename: {filename}")
                    return 0
        except Exception as e:
            self.logger.error(f"Error extracting word count from {filename}: {e}")
            return 0
    
    def extract_file_metadata(self, filename: str) -> Dict:
        """
        Extract additional metadata from filename
        Format: test_document_50words_expected0.2p_actual1p_f16.pdf
        """
        metadata = {
            'expected_pages': None,
            'actual_pages': None,
            'font_size': None
        }
        
        try:
            # Extract expected pages (e.g., "expected0.2p")
            expected_match = re.search(r'expected(\d+(?:\.\d+)?)p', filename.lower())
            if expected_match:
                metadata['expected_pages'] = float(expected_match.group(1))
            
            # Extract actual pages (e.g., "actual1p")
            actual_match = re.search(r'actual(\d+(?:\.\d+)?)p', filename.lower())
            if actual_match:
                metadata['actual_pages'] = float(actual_match.group(1))
            
            # Extract font size (e.g., "f16")
            font_match = re.search(r'f(\d+)', filename.lower())
            if font_match:
                metadata['font_size'] = int(font_match.group(1))
                
        except Exception as e:
            self.logger.warning(f"Error extracting metadata from {filename}: {e}")
        
        return metadata
    
    def calculate_accuracy(self, expected: int, actual: int) -> float:
        """Calculate accuracy percentage"""
        if expected == 0:
            return 0.0
        
        # Calculate accuracy as percentage
        accuracy = (min(expected, actual) / max(expected, actual)) * 100
        return round(accuracy, 2)
    
    def process_single_file(self, file_path: str, filename: str) -> Dict:
        """Process a single PDF file and return results"""
        self.logger.info(f"Processing: {filename}")
        
        start_time = time.time()
        
        # Extract metadata from filename
        file_metadata = self.extract_file_metadata(filename)
        
        result = {
            'filename': filename,
            'file_path': file_path,
            'expected_word_count': self.extract_expected_word_count(filename),
            'expected_pages': file_metadata['expected_pages'],
            'actual_pages_from_filename': file_metadata['actual_pages'],
            'font_size': file_metadata['font_size'],
            'ocr_word_count': 0,
            'actual_pages_processed': 0,
            'accuracy_percentage': 0.0,
            'processing_time': 0.0,
            'processing_method': '',
            'languages_detected': [],
            'confidence_score': None,
            'error': None,
            'success': False,
            'text_preview': ''
        }
        
        try:
            # Read the PDF file
            with open(file_path, 'rb') as file:
                pdf_content = file.read()
            
            file_size = len(pdf_content)
            self.logger.info(f"File size: {file_size:,} bytes")
            
            # Try direct text extraction first
            if self.pdf_processor.pymupdf_available:
                try:
                    self.logger.info("Attempting direct text extraction...")
                    extracted_text, pages_count = self.pdf_processor.extract_text_pymupdf(pdf_content)
                    word_count = self.pdf_processor.count_words(extracted_text)
                    
                    if word_count > 5:  # Meaningful content threshold
                        result.update({
                            'ocr_word_count': word_count,
                            'actual_pages_processed': pages_count,
                            'processing_method': 'direct_text_extraction',
                            'languages_detected': self.pdf_processor.detect_languages(extracted_text),
                            'text_preview': self.pdf_processor.get_text_preview(extracted_text, 200),
                            'success': True
                        })
                        self.logger.info(f"Direct extraction successful: {word_count} words from {pages_count} pages")
                    else:
                        raise Exception("Insufficient text content, trying OCR")
                        
                except Exception as e:
                    self.logger.warning(f"Direct extraction failed: {e}")
                    # Continue to OCR method
                    pass
            
            # If direct extraction didn't work, try OCR
            if not result['success'] and self.pdf_processor.tesseract_available and self.pdf_processor.pdf2image_available:
                try:
                    self.logger.info("Attempting OCR processing...")
                    
                    # Convert PDF to images
                    images = self.pdf_processor.pdf_to_images(pdf_content)
                    
                    if images:
                        # Extract text using OCR
                        extracted_text, confidence = self.pdf_processor.extract_text_ocr(images)
                        word_count = self.pdf_processor.count_words(extracted_text)
                        
                        result.update({
                            'ocr_word_count': word_count,
                            'actual_pages_processed': len(images),
                            'processing_method': 'ocr_pdf2image_tesseract',
                            'languages_detected': self.pdf_processor.detect_languages(extracted_text),
                            'confidence_score': round(confidence, 2) if confidence > 0 else None,
                            'text_preview': self.pdf_processor.get_text_preview(extracted_text, 200),
                            'success': True
                        })
                        self.logger.info(f"OCR successful: {word_count} words from {len(images)} pages")
                    else:
                        raise Exception("Failed to convert PDF to images")
                        
                except Exception as e:
                    result['error'] = f"OCR processing failed: {str(e)}"
                    self.logger.error(f"OCR failed for {filename}: {e}")
            
            # Calculate accuracy if we have results
            if result['success']:
                result['accuracy_percentage'] = self.calculate_accuracy(
                    result['expected_word_count'], 
                    result['ocr_word_count']
                )
            
        except Exception as e:
            result['error'] = f"File processing failed: {str(e)}"
            self.logger.error(f"Failed to process {filename}: {e}")
        
        finally:
            result['processing_time'] = round(time.time() - start_time, 2)
        
        return result
    
    def get_pdf_files(self) -> List[str]:
        """Get all PDF files from the test folder"""
        if not os.path.exists(self.test_folder_path):
            raise FileNotFoundError(f"Test folder not found: {self.test_folder_path}")
        
        pdf_files = []
        for filename in os.listdir(self.test_folder_path):
            if filename.lower().endswith('.pdf'):
                pdf_files.append(filename)
        
        pdf_files.sort()  # Sort for consistent processing order
        return pdf_files
    
    def run_accuracy_test(self):
        """Run the complete accuracy test on all PDF files"""
        self.logger.info("=" * 60)
        self.logger.info("STARTING OCR ACCURACY TEST")
        self.logger.info("=" * 60)
        
        try:
            pdf_files = self.get_pdf_files()
            self.logger.info(f"Found {len(pdf_files)} PDF files in {self.test_folder_path}")
            
            if not pdf_files:
                self.logger.warning("No PDF files found!")
                return
            
            self.summary_stats['total_files'] = len(pdf_files)
            total_start_time = time.time()
            
            # Process each file
            for i, filename in enumerate(pdf_files, 1):
                self.logger.info(f"\n[{i}/{len(pdf_files)}] Processing: {filename}")
                
                file_path = os.path.join(self.test_folder_path, filename)
                result = self.process_single_file(file_path, filename)
                self.results.append(result)
                
                if result['success']:
                    self.summary_stats['successful_ocr'] += 1
                    self.logger.info(f"✅ Success - Expected: {result['expected_word_count']}, "
                                   f"OCR: {result['ocr_word_count']}, "
                                   f"Accuracy: {result['accuracy_percentage']}%, "
                                   f"Pages: {result['actual_pages_processed']}")
                else:
                    self.summary_stats['failed_ocr'] += 1
                    self.logger.error(f"❌ Failed - {result['error']}")
            
            # Calculate summary statistics
            self.summary_stats['total_processing_time'] = round(time.time() - total_start_time, 2)
            
            successful_results = [r for r in self.results if r['success']]
            if successful_results:
                total_accuracy = sum([r['accuracy_percentage'] for r in successful_results])
                self.summary_stats['average_accuracy'] = round(total_accuracy / len(successful_results), 2)
            
            # Generate reports
            self.generate_detailed_report()
            self.generate_summary_report()
            self.generate_csv_report()
            
            self.logger.info("\n" + "=" * 60)
            self.logger.info("OCR ACCURACY TEST COMPLETED")
            self.logger.info("=" * 60)
            self.print_summary()
            
        except Exception as e:
            self.logger.error(f"Test execution failed: {e}")
            raise
    
    def generate_detailed_report(self):
        """Generate detailed text report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.results_dir, f"ocr_accuracy_detailed_{timestamp}.txt")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("OCR ACCURACY TEST - DETAILED REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Test Folder: {self.test_folder_path}\n")
            f.write(f"Total Files: {self.summary_stats['total_files']}\n")
            f.write(f"Successful OCR: {self.summary_stats['successful_ocr']}\n")
            f.write(f"Failed OCR: {self.summary_stats['failed_ocr']}\n")
            f.write(f"Average Accuracy: {self.summary_stats['average_accuracy']}%\n")
            f.write(f"Total Processing Time: {self.summary_stats['total_processing_time']}s\n")
            f.write("\n" + "=" * 80 + "\n\n")
            
            for i, result in enumerate(self.results, 1):
                f.write(f"FILE #{i}: {result['filename']}\n")
                f.write("-" * 60 + "\n")
                f.write(f"Expected Word Count: {result['expected_word_count']}\n")
                f.write(f"OCR Word Count: {result['ocr_word_count']}\n")
                f.write(f"Accuracy: {result['accuracy_percentage']}%\n")
                f.write(f"Expected Pages (from filename): {result['expected_pages']}\n")
                f.write(f"Actual Pages (from filename): {result['actual_pages_from_filename']}\n")
                f.write(f"Pages Processed: {result['actual_pages_processed']}\n")
                f.write(f"Font Size (from filename): {result['font_size']}\n")
                f.write(f"Processing Method: {result['processing_method']}\n")
                f.write(f"Processing Time: {result['processing_time']}s\n")
                f.write(f"Languages Detected: {', '.join(result['languages_detected'])}\n")
                f.write(f"Confidence Score: {result['confidence_score']}\n")
                f.write(f"Success: {'✅' if result['success'] else '❌'}\n")
                
                if result['error']:
                    f.write(f"Error: {result['error']}\n")
                
                if result['text_preview']:
                    f.write(f"Text Preview: {result['text_preview'][:150]}...\n")
                
                f.write("\n" + "-" * 60 + "\n\n")
        
        self.logger.info(f"Detailed report saved: {report_file}")
        return report_file
    
    def generate_summary_report(self):
        """Generate summary JSON report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = os.path.join(self.results_dir, f"ocr_accuracy_summary_{timestamp}.json")
        
        summary_data = {
            'test_info': {
                'timestamp': datetime.now().isoformat(),
                'test_folder': self.test_folder_path,
                'processor_capabilities': {
                    'tesseract_available': self.pdf_processor.tesseract_available,
                    'pdf2image_available': self.pdf_processor.pdf2image_available,
                    'pymupdf_available': self.pdf_processor.pymupdf_available
                }
            },
            'summary_statistics': self.summary_stats,
            'detailed_results': self.results
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Summary JSON saved: {summary_file}")
        return summary_file
    
    def generate_csv_report(self):
        """Generate CSV report for easy analysis"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = os.path.join(self.results_dir, f"ocr_accuracy_results_{timestamp}.csv")
        
        with open(csv_file, 'w', encoding='utf-8') as f:
            # Write header
            f.write("Filename,Expected_Words,OCR_Words,Accuracy_%,Expected_Pages,Actual_Pages_Filename,")
            f.write("Pages_Processed,Font_Size,Processing_Method,Processing_Time_s,")
            f.write("Languages,Confidence,Success,Error\n")
            
            # Write data
            for result in self.results:
                f.write(f'"{result["filename"]}",')
                f.write(f'{result["expected_word_count"]},')
                f.write(f'{result["ocr_word_count"]},')
                f.write(f'{result["accuracy_percentage"]},')
                f.write(f'{result["expected_pages"] or ""},')
                f.write(f'{result["actual_pages_from_filename"] or ""},')
                f.write(f'{result["actual_pages_processed"]},')
                f.write(f'{result["font_size"] or ""},')
                f.write(f'"{result["processing_method"]}",')
                f.write(f'{result["processing_time"]},')
                f.write(f'"{"; ".join(result["languages_detected"])}",')
                f.write(f'{result["confidence_score"] or ""},')
                f.write(f'{"Yes" if result["success"] else "No"},')
                f.write(f'"{result["error"] or ""}"\n')
        
        self.logger.info(f"CSV report saved: {csv_file}")
        return csv_file
    
    def print_summary(self):
        """Print summary to console"""
        print("\n" + "=" * 80)
        print("OCR ACCURACY TEST SUMMARY")
        print("=" * 80)
        print(f"Total Files Processed: {self.summary_stats['total_files']}")
        print(f"Successful OCR: {self.summary_stats['successful_ocr']}")
        print(f"Failed OCR: {self.summary_stats['failed_ocr']}")
        print(f"Success Rate: {(self.summary_stats['successful_ocr']/self.summary_stats['total_files']*100):.1f}%")
        print(f"Average Accuracy: {self.summary_stats['average_accuracy']}%")
        print(f"Total Processing Time: {self.summary_stats['total_processing_time']}s")
        print(f"Average Time per File: {self.summary_stats['total_processing_time']/self.summary_stats['total_files']:.2f}s")
        
        print("\nAccuracy Breakdown:")
        accuracy_ranges = {'90-100%': 0, '80-89%': 0, '70-79%': 0, '60-69%': 0, 'Below 60%': 0}
        
        for result in self.results:
            if result['success']:
                acc = result['accuracy_percentage']
                if acc >= 90:
                    accuracy_ranges['90-100%'] += 1
                elif acc >= 80:
                    accuracy_ranges['80-89%'] += 1
                elif acc >= 70:
                    accuracy_ranges['70-79%'] += 1
                elif acc >= 60:
                    accuracy_ranges['60-69%'] += 1
                else:
                    accuracy_ranges['Below 60%'] += 1
        
        for range_name, count in accuracy_ranges.items():
            print(f"  {range_name}: {count} files")
        
        print(f"\nResults saved in: {self.results_dir}/")
        print("=" * 80)


def main():
    """Main function to run the OCR accuracy test"""
    # Initialize the tester
    tester = OCRAccuracyTester("test_document_5")
    
    # Run the test
    try:
        tester.run_accuracy_test()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure the 'test_document_5' folder exists in the current directory.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()