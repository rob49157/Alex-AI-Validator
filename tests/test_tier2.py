import asyncio
from unittest.mock import patch

import pytest
from httpx import AsyncClient, ASGITransport

from main import app
from models.schemas import JobStatus

_HEADERS = {"Authorization": "Bearer change_me"}

_BOOK_TEXT = (
    "The expedition set off at dawn, navigating dense jungle terrain. "
    "Rivers carved deep gorges into the limestone plateau, making progress slow. "
    "Each evening the team catalogued specimens and updated their field notes. "
) * 80


@pytest.fixture(autouse=True)
def clear_jobs():
    import api.routes.tier2 as t2
    t2._jobs.clear()
    yield
    t2._jobs.clear()


@pytest.fixture()
def client():
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


# ── route-level ────────────────────────────────────────────────────────────────

async def test_start_job_returns_job_id(client):
    async with client as c:
        r = await c.post(
            "/api/validate/deep-analysis",
            headers=_HEADERS,
            json={"arweave_hash": "hash-001", "mongo_upload_id": "mongo-001"},
        )
    assert r.status_code == 200
    assert "job_id" in r.json()


async def test_unknown_job_returns_404(client):
    async with client as c:
        r = await c.get("/api/validate/deep-analysis/does-not-exist", headers=_HEADERS)
    assert r.status_code == 404


# ── background task flows ──────────────────────────────────────────────────────

async def test_no_text_completes_with_flag(client):
    async with client as c:
        r = await c.post(
            "/api/validate/deep-analysis",
            headers=_HEADERS,
            json={"arweave_hash": "hash-notext", "mongo_upload_id": "mongo-002"},
        )
        job_id = r.json()["job_id"]
        await asyncio.sleep(0.1)
        r2 = await c.get(f"/api/validate/deep-analysis/{job_id}", headers=_HEADERS)

    assert r2.json()["status"] == JobStatus.complete
    assert "no_text_provided" in r2.json()["results"]["flags"]


async def test_full_analysis_completes(client):
    with (
        patch("api.routes.tier2.find_semantic_duplicates", return_value=[]),
        patch("api.routes.tier2.store_embedding"),
        patch("api.routes.tier2.detect_language", return_value="en"),
        patch("api.routes.tier2.score_content_quality", return_value=0.78),
        patch("api.routes.tier2.estimate_completeness", return_value="full"),
    ):
        async with client as c:
            r = await c.post(
                "/api/validate/deep-analysis",
                headers=_HEADERS,
                json={
                    "arweave_hash": "hash-002",
                    "mongo_upload_id": "mongo-003",
                    "extracted_text": _BOOK_TEXT,
                    "page_count": 60,
                },
            )
            job_id = r.json()["job_id"]
            await asyncio.sleep(0.1)
            r2 = await c.get(f"/api/validate/deep-analysis/{job_id}", headers=_HEADERS)

    data = r2.json()
    assert data["status"] == JobStatus.complete
    results = data["results"]
    assert results["language"] == "en"
    assert results["quality_score"] == 0.78
    assert results["completeness"] == "full"
    assert results["semantic_duplicates"] == []
    assert "semantic_duplicate_found" not in results["flags"]


async def test_duplicate_found_sets_flag(client):
    from models.schemas import SemanticDuplicate
    duplicate = SemanticDuplicate(arweave_hash="hash-original", similarity=0.96)

    with (
        patch("api.routes.tier2.find_semantic_duplicates", return_value=[duplicate]),
        patch("api.routes.tier2.store_embedding"),
        patch("api.routes.tier2.detect_language", return_value="en"),
        patch("api.routes.tier2.score_content_quality", return_value=0.7),
        patch("api.routes.tier2.estimate_completeness", return_value="full"),
    ):
        async with client as c:
            r = await c.post(
                "/api/validate/deep-analysis",
                headers=_HEADERS,
                json={
                    "arweave_hash": "hash-copy",
                    "mongo_upload_id": "mongo-004",
                    "extracted_text": _BOOK_TEXT,
                },
            )
            job_id = r.json()["job_id"]
            await asyncio.sleep(0.1)
            r2 = await c.get(f"/api/validate/deep-analysis/{job_id}", headers=_HEADERS)

    data = r2.json()
    assert data["status"] == JobStatus.complete
    assert data["results"]["semantic_duplicates"][0]["arweave_hash"] == "hash-original"
    assert "semantic_duplicate_found" in data["results"]["flags"]


async def test_service_error_marks_job_failed(client):
    with patch("api.routes.tier2.find_semantic_duplicates", side_effect=RuntimeError("chroma down")):
        async with client as c:
            r = await c.post(
                "/api/validate/deep-analysis",
                headers=_HEADERS,
                json={
                    "arweave_hash": "hash-err",
                    "mongo_upload_id": "mongo-005",
                    "extracted_text": _BOOK_TEXT,
                },
            )
            job_id = r.json()["job_id"]
            await asyncio.sleep(0.1)
            r2 = await c.get(f"/api/validate/deep-analysis/{job_id}", headers=_HEADERS)

    data = r2.json()
    assert data["status"] == JobStatus.failed
    assert "chroma down" in data["error"]
