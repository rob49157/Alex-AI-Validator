from unittest.mock import patch

import pytest
from httpx import AsyncClient, ASGITransport

from main import app
from models.schemas import ValidationFlag
from services.text_extraction import ExtractionResult

_HEADERS = {"Authorization": "Bearer change_me"}


async def _post_pdf(client: AsyncClient, data: bytes, filename: str = "book.pdf") -> dict:
    response = await client.post(
        "/api/validate/upload",
        headers=_HEADERS,
        files={"file": (filename, data, "application/pdf")},
    )
    return response


@pytest.fixture()
def client():
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def test_rejects_non_pdf_magic_bytes(client):
    async with client as c:
        r = await _post_pdf(c, b"NOT A PDF CONTENT AT ALL")
    assert r.status_code == 400


async def test_rejects_missing_auth():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post(
            "/api/validate/upload",
            files={"file": ("book.pdf", b"%PDF-fake", "application/pdf")},
        )
    assert r.status_code == 401


async def test_valid_pdf_returns_valid_true(client, sample_pdf_bytes):
    mock_result = ExtractionResult(
        text="This is a well-written book with coherent sentences. " * 60,
        page_count=10,
        flags=[],
        used_ocr=False,
    )
    async with client as c:
        with patch("api.routes.tier1.extract_text", return_value=mock_result):
            r = await _post_pdf(c, sample_pdf_bytes)
    assert r.status_code == 200
    data = r.json()
    assert data["valid"] is True
    assert data["page_count"] == 10
    assert data["extracted_text"] is not None


async def test_no_text_returns_invalid(client, sample_pdf_bytes):
    mock_result = ExtractionResult(
        text="",
        page_count=5,
        flags=[ValidationFlag.no_extractable_text],
        used_ocr=False,
    )
    async with client as c:
        with patch("api.routes.tier1.extract_text", return_value=mock_result):
            r = await _post_pdf(c, sample_pdf_bytes)
    assert r.status_code == 200
    data = r.json()
    assert data["valid"] is False
    assert "no_extractable_text" in data["flags"]


async def test_ocr_failed_returns_invalid(client, sample_pdf_bytes):
    mock_result = ExtractionResult(
        text="",
        page_count=3,
        flags=[ValidationFlag.possible_scan, ValidationFlag.ocr_failed],
        used_ocr=True,
    )
    async with client as c:
        with patch("api.routes.tier1.extract_text", return_value=mock_result):
            r = await _post_pdf(c, sample_pdf_bytes)
    assert r.status_code == 200
    data = r.json()
    assert data["valid"] is False
    assert "ocr_failed" in data["flags"]


async def test_timeout_returns_invalid(client, sample_pdf_bytes):
    import asyncio

    async def slow(*_):
        raise asyncio.TimeoutError()

    async with client as c:
        with patch("api.routes.tier1.asyncio.wait_for", side_effect=slow):
            r = await _post_pdf(c, sample_pdf_bytes)
    assert r.status_code == 200
    data = r.json()
    assert data["valid"] is False
    assert "parse_timeout" in data["flags"]


async def test_health_endpoint_no_auth():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
