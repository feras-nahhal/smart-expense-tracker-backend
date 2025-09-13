# ---- Base Image ----
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (Tesseract OCR + build tools)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    build-essential \
    libpq-dev \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
