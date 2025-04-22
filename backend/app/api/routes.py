# backend/app/api/routes.py

import logging
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query
from app.models.schemas import AnalyzeResponse
from app.services import analysis_service

log = logging.getLogger(__name__)
router = APIRouter()

@router.post(
    "/analyze_question",
    response_model=AnalyzeResponse,
    summary="Analyze an uploaded exam question image",
    description="Upload an image file containing an exam question snippet...", # Shortened for brevity
    tags=["Analysis"]
)
async def analyze_question_endpoint(
    file: UploadFile = File(..., description="Image file...")
    # subject_hint: Optional[str] = Query(None, ...)
    ):
    log.info(f"Received file upload: {file.filename}, content type: {file.content_type}")

    # 1. Validate File Type
    if not file.content_type or not file.content_type.startswith("image/"):
        # --- CORRECTED LOGGING ---
        log.warning(f"Invalid file type uploaded: {file.content_type}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Please upload an image."
        )
        # --- END CORRECTION ---

    image_bytes : bytes | None = None # Initialize to handle finally block correctly
    try:
        # 2. Read Image Data
        image_bytes = await file.read()
        if not image_bytes:
            log.error("Uploaded file is empty.")
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        log.debug(f"Read {len(image_bytes)} bytes from uploaded file.")

        # 3. Delegate to Analysis Service
        log.info("Calling analysis service to process image...")
        analysis_result: Optional[AnalyzeResponse] = await analysis_service.process_question_image(
            image_bytes=image_bytes
            # subject_hint=subject_hint
        )

        # 4. Handle Service Result
        if analysis_result is None:
            log.error("Analysis service returned None, indicating processing failure.")
            raise HTTPException(
                status_code=500,
                detail="Analysis failed. Could not process the question due to an internal error."
            )

        log.info("Analysis successful. Returning results.")
        return analysis_result

    except HTTPException as http_exc:
        log.warning(f"HTTP Exception during analysis: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        log.error(f"Unexpected error analyzing question file '{file.filename}': {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred during analysis."
        )
    finally:
        # Ensure the file is closed
        try:
            await file.close()
            log.debug(f"Closed uploaded file: {file.filename}")
        except Exception as close_err:
            log.warning(f"Error closing uploaded file '{file.filename}': {close_err}")

# Root endpoint
@router.get("/")
async def read_root():
    log.info("Root endpoint '/' accessed.")
    return {"message": "Welcome to the ExamSage API! Visit /docs for documentation."}
