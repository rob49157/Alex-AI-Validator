import asyncio
from fastapi import APIRouter, UploadFile, File, HTTPException, status

from config import settings
from models.schemas import Tier1Response, ValidationFlag
from services.text_extraction import extract_text
from services.content_quality import check_basic_quality

router = APIRouter()

_MAX_BYTES = settings.max_file_size_mb * 1024 * 1024


@router.post("/upload", response_model=Tier1Response)
async def validate_upload(file: UploadFile = File(...)):
    # Size guard before doing any work
    pdf_bytes = await file.read()
    if len(pdf_bytes) > _MAX_BYTES:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="File exceeds size limit")

    # Magic bytes check — must start with %PDF-
    if not pdf_bytes[:5] == b"%PDF-":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File is not a valid PDF")

    # Run extraction in a thread so the async event loop isn't blocked
    try:
        extraction = await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(
                None, extract_text, pdf_bytes, settings.job_timeout_seconds
            ),
            timeout=settings.job_timeout_seconds,
        )
    except asyncio.TimeoutError:
        return Tier1Response(
            valid=False,
            flags=[ValidationFlag.parse_timeout],
            message="PDF processing timed out",
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    flags = list(extraction.flags)

    # If OCR also failed to get text, reject
    if ValidationFlag.no_extractable_text in flags or ValidationFlag.ocr_failed in flags:
        return Tier1Response(
            valid=False,
            flags=flags,
            page_count=extraction.page_count,
            message="PDF contains no extractable text",
        )

    quality = check_basic_quality(extraction.text, extraction.page_count)
    flags.extend(ValidationFlag(f) for f in quality.flags if f in ValidationFlag._value2member_map_)

    if not quality.has_readable_text:
        return Tier1Response(
            valid=False,
            flags=flags,
            page_count=extraction.page_count,
            message="PDF does not contain readable text",
        )

    return Tier1Response(
        valid=True,
        flags=flags,
        extracted_text=extraction.text,
        page_count=extraction.page_count,
    )
