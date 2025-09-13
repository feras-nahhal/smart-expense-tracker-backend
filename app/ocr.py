# app/ocr.py
from PIL import Image, ImageOps, ImageFilter
import pytesseract
import io
import os

# ✅ Auto-detect environment (Windows vs Linux/Docker)
if os.path.exists("/usr/bin/tesseract"):  
    # Inside Docker/Linux
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
else:
    # Local Windows dev environment
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def preprocess_image(img):
    """Preprocess the image for better OCR accuracy."""
    # Convert to grayscale
    img = img.convert("L")
    # Increase contrast
    img = ImageOps.autocontrast(img)
    # Apply threshold (binarize)
    img = img.point(lambda x: 0 if x < 140 else 255, "1")
    # Optional: sharpen
    img = img.filter(ImageFilter.SHARPEN)
    return img


def extract_text_from_bytes(file_bytes: bytes) -> str:
    """Extract text from image bytes using Tesseract OCR."""
    img = Image.open(io.BytesIO(file_bytes))
    img = preprocess_image(img)
    text = pytesseract.image_to_string(img)
    print("OCR Extracted Text:", text)
    return text
