# PDF Word Counter API with OCR

A FastAPI application that extracts text from PDF documents and counts words using OCR (Optical Character Recognition). Supports both regular text-based PDFs and scanned documents with multilingual text (English + Arabic).

## 🚀 Features

- **PDF Text Extraction**: Direct text extraction from text-based PDFs
- **OCR Processing**: Advanced OCR for scanned documents using Tesseract
- **Multilingual Support**: English and Arabic text recognition
- **Word Counting**: Accurate word count with language detection
- **Performance Optimized**: Fast processing with configurable limits
- **Cross-Platform**: Works on Windows, Linux, and cloud platforms
- **RESTful API**: Easy integration with web applications
- **Interactive Documentation**: Built-in Swagger UI

## 📋 Requirements

### System Requirements
- Python 3.11 or higher
- Tesseract OCR engine
- Poppler utilities (for PDF to image conversion)

### Python Dependencies
- FastAPI
- Uvicorn
- pytesseract
- pdf2image
- Pillow
- PyMuPDF (fallback)
- pydantic

## 🛠️ Installation

### Option 1: Local Development (Windows)

#### Step 1: Install Python
1. Download Python 3.11 from [python.org](https://www.python.org/downloads/)
2. During installation, check "Add Python to PATH"
3. Verify installation:
   ```cmd
   python --version
   pip --version
   ```

#### Step 2: Install System Dependencies

**Install Tesseract OCR:**
1. Download from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
2. Download: `tesseract-ocr-w64-setup-5.3.3.20231005.exe`
3. Install to default location: `C:\Program Files\Tesseract-OCR`
4. Add to PATH: `C:\Program Files\Tesseract-OCR`

**Install Poppler (for pdf2image):**
1. Download from [poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases/)
2. Extract to `C:\poppler`
3. Add to PATH: `C:\poppler\Library\bin`

#### Step 3: Setup Project
```cmd
# Clone or create project directory
mkdir pdf-word-counter
cd pdf-word-counter

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install Python packages
pip install -r requirements.txt
```

#### Step 4: Run Application
```cmd
# Method 1: Direct run
python main.py

# Method 2: Using uvicorn (recommended for development)
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### Option 2: Linux/Ubuntu Setup

#### Install System Dependencies
```bash
# Update system
sudo apt update

# Install Tesseract and Poppler
sudo apt install -y tesseract-ocr tesseract-ocr-eng tesseract-ocr-ara
sudo apt install -y poppler-utils
sudo apt install -y python3-pip python3-venv

# Install additional libraries
sudo apt install -y libgl1-mesa-glx libglib2.0-0
```

#### Setup Application
```bash
# Create project directory
mkdir pdf-word-counter && cd pdf-word-counter

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python packages
pip install -r requirements.txt

# Run application
python main.py
```

### Option 3: Docker Deployment

#### Create Dockerfile
```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-ara \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
USER app

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=10)" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Build and Run
```bash
# Build image
docker build -t pdf-word-counter .

# Run container
docker run -p 8000:8000 pdf-word-counter
```

## ☁️ Azure App Service Deployment

### Prerequisites
- Azure subscription
- Azure CLI installed

### Step 1: Prepare for Azure
1. Ensure your `main.py` includes Azure-specific configurations
2. Create `startup.sh` for Linux App Service
3. Set environment variables in Azure portal

### Step 2: Deploy to Azure App Service

#### Using Azure CLI
```bash
# Login to Azure
az login

# Create resource group
az group create --name pdf-ocr-rg --location "East US"

# Create App Service plan (Linux)
az appservice plan create \
    --name pdf-ocr-plan \
    --resource-group pdf-ocr-rg \
    --sku B1 \
    --is-linux

# Create web app
az webapp create \
    --resource-group pdf-ocr-rg \
    --plan pdf-ocr-plan \
    --name your-pdf-ocr-app \
    --runtime "PYTHON|3.11" \
    --deployment-source-url https://github.com/yourusername/pdf-word-counter

# Configure startup command
az webapp config set \
    --resource-group pdf-ocr-rg \
    --name your-pdf-ocr-app \
    --startup-file "startup.sh"
```

#### Using Azure Portal
1. Go to [Azure Portal](https://portal.azure.com)
2. Create new "Web App"
3. Choose:
   - **OS**: Linux
   - **Runtime**: Python 3.11
   - **Plan**: Basic B1 or higher (recommended)
4. Deploy code via GitHub integration or ZIP upload
5. Set startup command: `startup.sh`

### Step 3: Configure Environment Variables
In Azure Portal → Configuration → Application Settings:
```
WEBSITES_PORT=8000
SCM_DO_BUILD_DURING_DEPLOYMENT=true
```

### Step 4: Create startup.sh
```bash
#!/bin/bash
# Install system dependencies
apt-get update
apt-get install -y tesseract-ocr tesseract-ocr-eng tesseract-ocr-ara poppler-utils

# Start the application
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

## 📁 Project Structure

```
pdf-word-counter/
├── main.py                 # Main application file
├── requirements.txt        # Python dependencies
├── startup.sh             # Azure startup script
├── Dockerfile             # Docker configuration
├── README.md              # This file
├── .gitignore             # Git ignore file
└── tests/                 # Test files (optional)
    └── test_api.py
```

## 🔧 Configuration

### Environment Variables
```bash
# Optional: Custom Tesseract path
TESSERACT_CMD=/usr/bin/tesseract

# Azure specific
WEBSITES_PORT=8000
SCM_DO_BUILD_DURING_DEPLOYMENT=true
```

### Application Settings
- **Max file size**: 10MB (configurable in code)
- **Supported formats**: PDF
- **Languages**: English, Arabic
- **OCR Engine**: Tesseract 5.x
- **Image DPI**: 300 (local), 200 (cloud)

## 🧪 Testing

### Test Endpoints
```bash
# Health check
curl http://localhost:8000/health

# Test endpoint
curl http://localhost:8000/test

# Upload PDF for processing
curl -X POST "http://localhost:8000/count-words" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@sample.pdf"
```

### Using Interactive Documentation
1. Start the application
2. Open browser: `http://localhost:8000/docs`
3. Use the interactive Swagger UI to test endpoints

## 📊 API Endpoints

### GET /
- **Description**: Root endpoint with API information
- **Response**: API details and capabilities

### GET /health
- **Description**: Health check with system status
- **Response**: Tesseract availability, version, and platform info

### GET /test
- **Description**: Simple test endpoint
- **Response**: Basic status and configuration

### POST /count-words
- **Description**: Main endpoint for PDF word counting
- **Input**: PDF file (multipart/form-data)
- **Response**: Word count, languages detected, processing time, text preview

#### Sample Response
```json
{
  "total_words": 245,
  "text_extracted": true,
  "processing_method": "pdf2image_tesseract",
  "languages_detected": ["English", "Arabic", "Numbers"],
  "pages_processed": 2,
  "confidence_score": 87.5,
  "extracted_text_preview": "UNITED ARAB EMIRATES FEDERAL AUTHORITY...",
  "processing_time": 15.3,
  "error": null
}
```

## 🚀 Performance

### Local Performance
- **Text-based PDF**: ~1 second
- **Scanned PDF**: ~1 second per page
- **Memory usage**: ~200-500MB during processing

### Cloud Performance (Azure B1)
- **Text-based PDF**: ~2-3 seconds
- **Scanned PDF**: ~2-3 seconds per page
- **Memory usage**: ~300-800MB during processing

### Optimization Tips
1. **Reduce image DPI** for cloud deployment (200 instead of 300)
2. **Limit page count** for large documents
3. **Use async processing** for very large files
4. **Implement caching** for repeated requests

## 🛠️ Troubleshooting

### Common Issues

#### Tesseract Not Found
```
Error: tesseract is not installed or it's not in your PATH
```
**Solution**: 
1. Install Tesseract OCR
2. Add to system PATH
3. Restart terminal/IDE

#### pdf2image Conversion Failed
```
Error: Unable to get page count. Is poppler installed and in PATH?
```
**Solution**:
1. Install Poppler utilities
2. Add to system PATH
3. Restart application

#### Memory Issues on Cloud
```
Error: Out of memory during OCR processing
```
**Solution**:
1. Reduce image DPI in cloud configuration
2. Limit pages processed per request
3. Upgrade to higher tier plan

#### Timeout on Large Files
```
Error: Request timeout
```
**Solution**:
1. Reduce file size or page count
2. Implement async processing
3. Use higher timeout values

### Debug Mode
Enable detailed logging by setting:
```python
logging.basicConfig(level=logging.DEBUG)
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review Azure App Service logs
3. Create an issue on GitHub

## 🔄 Updates

### Version 4.1.0
- Added Windows OCR support
- Improved Azure App Service compatibility
- Enhanced error handling
- Performance optimizations

### Version 4.0.0
- pdf2image integration
- Multilingual OCR support
- Cloud deployment optimizations
- Comprehensive documentation