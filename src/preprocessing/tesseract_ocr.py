"""
Tesseract OCR wrapper.
Reads the Tesseract binary path from the TESSERACT_CMD env var (set in
a local .env file), so deployment to other machines just needs a different
.env without code changes.
"""

import os
from PIL import Image
import pytesseract
from dotenv import load_dotenv

# Load .env once at module import
load_dotenv()

_tess_cmd = os.getenv("TESSERACT_CMD")
if _tess_cmd:
    pytesseract.pytesseract.tesseract_cmd = _tess_cmd
# Otherwise rely on system PATH (Linux servers, Docker, etc.)


def ocr_image(image: Image.Image, lang: str = "eng") -> str:
    """Run Tesseract OCR on a PIL image. Returns extracted text."""
    try:
        return pytesseract.image_to_string(image, lang=lang).strip()
    except pytesseract.TesseractNotFoundError:
        raise RuntimeError(
            "Tesseract binary not found. Set TESSERACT_CMD in .env or "
            "install Tesseract and add it to PATH."
        )
