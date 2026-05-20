import io
import subprocess
import tempfile
import os
from dataclasses import dataclass

import pdfplumber
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image

from config import settings
from models.schemas import ValidationFlag

# Hardcoded — do not make configurable (supply chain risk)
_MIN_TEXT_CHARS_PER_PAGE = 50


@dataclass
class ExtractionResult:
    text: str
    page_count: int
    flags: list[ValidationFlag]
    used_ocr: bool


def extract_text(pdf_bytes: bytes, timeout: int = 30) -> ExtractionResult:
    """
    Attempt digital text extraction first (pdfplumber).
    Fall back to OCR (pytesseract) if the PDF appears to be a scan.
    Raises ValueError on corrupt/unreadable PDF.
    """
    if settings.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

    flags: list[ValidationFlag] = []

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        page_count = len(pdf.pages)
        pages_to_process = min(page_count, settings.max_pages_ocr)

        text_parts = []
        for page in pdf.pages[:pages_to_process]:
            extracted = page.extract_text() or ""
            text_parts.append(extracted)

    digital_text = "\n".join(text_parts).strip()
    chars_per_page = len(digital_text) / max(pages_to_process, 1)

    # Digital extraction succeeded — enough text found
    if chars_per_page >= _MIN_TEXT_CHARS_PER_PAGE:
        return ExtractionResult(
            text=digital_text,
            page_count=page_count,
            flags=flags,
            used_ocr=False,
        )

    # Looks like a scanned PDF — try OCR
    flags.append(ValidationFlag.possible_scan)
    ocr_text = _ocr_pdf(pdf_bytes, pages_to_process, timeout)

    if ocr_text is None:
        flags.append(ValidationFlag.ocr_failed)
        return ExtractionResult(
            text="",
            page_count=page_count,
            flags=flags,
            used_ocr=True,
        )

    if not ocr_text.strip():
        flags.append(ValidationFlag.no_extractable_text)

    return ExtractionResult(
        text=ocr_text,
        page_count=page_count,
        flags=flags,
        used_ocr=True,
    )


def _ocr_pdf(pdf_bytes: bytes, max_pages: int, timeout: int) -> str | None:
    """
    Convert PDF pages to images and run Tesseract OCR.
    Uses explicit subprocess args — never shell=True.
    Returns None if OCR fails or times out.
    """
    try:
        images: list[Image.Image] = convert_from_bytes(
            pdf_bytes,
            first_page=1,
            last_page=max_pages,
            dpi=200,
        )
    except Exception:
        return None

    text_parts = []
    for image in images:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
            image.save(tmp_path, format="PNG")
        try:
            result = subprocess.run(
                ["tesseract", tmp_path, "stdout", "--psm", "3", "-l", "eng"],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            text_parts.append(result.stdout)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None
        finally:
            os.unlink(tmp_path)

    return "\n".join(text_parts)
