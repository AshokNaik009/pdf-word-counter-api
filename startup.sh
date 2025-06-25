#!/bin/bash

echo "🚀 Starting Azure App Service deployment..."

# Update package list
echo "📦 Updating package list..."
apt-get update -y

# Install system dependencies
echo "📦 Installing Tesseract OCR..."
apt-get install -y tesseract-ocr tesseract-ocr-eng tesseract-ocr-ara libtesseract-dev

echo "📦 Installing Poppler utilities..."
apt-get install -y poppler-utils

echo "📦 Installing additional libraries..."
apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev

# Verify installations
echo "🔍 Verifying Tesseract installation..."
tesseract --version || echo "❌ Tesseract installation failed"

echo "🔍 Verifying Poppler installation..."
pdftoppm -h > /dev/null 2>&1 && echo "✅ Poppler installed" || echo "❌ Poppler installation failed"

# Set environment variables
export TESSERACT_CMD=/usr/bin/tesseract

# Install Python dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "📦 Installing Python dependencies..."
    pip install --no-cache-dir -r requirements.txt
fi

# Start the application
echo "🚀 Starting PDF Word Counter API..."
python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1