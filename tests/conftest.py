import pytest
import fitz  # PyMuPDF


def make_pdf(text: str = "") -> bytes:
    """Generate a minimal valid PDF containing the given text."""
    doc = fitz.open()
    page = doc.new_page()
    if text:
        page.insert_text((50, 750), text, fontsize=11)
    return doc.tobytes()


@pytest.fixture()
def sample_pdf_bytes() -> bytes:
    body = "The quick brown fox jumps over the lazy dog. " * 200
    return make_pdf(body)


@pytest.fixture()
def empty_pdf_bytes() -> bytes:
    return make_pdf("")
