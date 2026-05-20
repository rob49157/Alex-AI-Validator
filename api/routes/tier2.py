import asyncio
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException, status

from config import settings
from models.schemas import (
    DeepAnalysisRequest,
    DeepAnalysisJobResponse,
    DeepAnalysisStatusResponse,
    DeepAnalysisResults,
    JobStatus,
)
from services.semantic_similarity import find_semantic_duplicates, store_embedding
from services.language_detection import detect_language
from services.deep_quality import score_content_quality, estimate_completeness

router = APIRouter()

# In-memory job store — replace with Redis for production
_jobs: dict[str, DeepAnalysisStatusResponse] = {}

# Limits how many analysis jobs run concurrently; excess stay "pending"
_semaphore = asyncio.Semaphore(settings.max_concurrent_jobs)


@router.post("/deep-analysis", response_model=DeepAnalysisJobResponse)
async def start_deep_analysis(body: DeepAnalysisRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    _jobs[job_id] = DeepAnalysisStatusResponse(status=JobStatus.pending)
    background_tasks.add_task(_run_deep_analysis, job_id, body)
    return DeepAnalysisJobResponse(job_id=job_id)


@router.get("/deep-analysis/{job_id}", response_model=DeepAnalysisStatusResponse)
async def get_deep_analysis_status(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return job


async def _run_deep_analysis(job_id: str, body: DeepAnalysisRequest) -> None:
    async with _semaphore:  # job stays "pending" while limit is reached
        _jobs[job_id] = DeepAnalysisStatusResponse(status=JobStatus.processing)
        results = DeepAnalysisResults()

        try:
            text = body.extracted_text or ""
            page_count = body.page_count or 1

            if not text:
                results.flags.append("no_text_provided")
                _jobs[job_id] = DeepAnalysisStatusResponse(status=JobStatus.complete, results=results)
                return

            # Query for semantic duplicates before storing so this doc can't match itself
            duplicates = await asyncio.to_thread(find_semantic_duplicates, text, body.arweave_hash)
            results.semantic_duplicates = duplicates
            if duplicates:
                results.flags.append("semantic_duplicate_found")

            # Persist this doc's embedding for future comparisons
            await asyncio.to_thread(store_embedding, text, body.arweave_hash)

            results.language = await asyncio.to_thread(detect_language, text)
            results.quality_score = await asyncio.to_thread(score_content_quality, text, page_count)
            results.completeness = await asyncio.to_thread(estimate_completeness, text, page_count)

            _jobs[job_id] = DeepAnalysisStatusResponse(status=JobStatus.complete, results=results)

        except Exception as exc:
            _jobs[job_id] = DeepAnalysisStatusResponse(status=JobStatus.failed, error=str(exc))
