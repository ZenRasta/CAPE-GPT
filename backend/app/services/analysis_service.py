# backend/app/services/analysis_service.py
'''

Purpose: Acts as the orchestrator or coordinator for the core analysis workflow. It ties the other services together.

Input: The initial request data (in this case, the image bytes from the API route).

Output: A structured data object (likely the Pydantic AnalyzeResponse model) containing the complete result to be sent back to the API layer.

Responsibilities: Calling the other services in the correct sequence (OCR -> Vision LLM -> Embedding -> Supabase Query -> Mapping LLM), passing data between them, performing intermediate calculations (like frequency analysis), handling errors from the services it calls, and assembling the final response structure.
'''
# backend/app/services/analysis_service.py

import logging
from typing import List, Dict, Any, Optional

# Import functions from other services
from . import ocr_service
from . import embedding_service
from . import supabase_service
from . import llm_service

# Import Pydantic models
from app.models.schemas import AnalyzeResponse, AnalysisResult, SyllabusMapping, SimilarQuestion, FrequencyAnalysisData

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')
log = logging.getLogger(__name__)


def calculate_frequency(similar_questions: List[Dict[str, Any]]) -> FrequencyAnalysisData:
    """Calculates frequency distribution from retrieved similar questions."""
    years_dist: Dict[str, int] = {}
    papers_dist: Dict[str, int] = {}
    min_year = 9999
    max_year = 0

    if not similar_questions:
        return FrequencyAnalysisData(
            total_similar_found=0,
            years_distribution={},
            papers_distribution=None,
            summary_statement="No similar past questions found in the database."
        )

    for q in similar_questions:
        year = q.get("year")
        paper = q.get("paper")
        if year:
            year_str = str(year)
            years_dist[year_str] = years_dist.get(year_str, 0) + 1
            min_year = min(min_year, year)
            max_year = max(max_year, year)
        if paper:
            papers_dist[paper] = papers_dist.get(paper, 0) + 1

    total_found = len(similar_questions)
    year_range = max_year - min_year + 1 if max_year >= min_year else 1
    years_analyzed_str = f"{year_range} year{'s' if year_range != 1 else ''}" if max_year >= min_year else "analyzed period"

    summary = f"Found {total_found} similar question{'s' if total_found != 1 else ''} from the {years_analyzed_str} ({min_year}-{max_year})." if total_found > 0 else "No similar past questions found."

    # Fill missing years in the distribution for a contiguous chart
    if max_year >= min_year:
        full_years_dist = {str(y): years_dist.get(str(y), 0) for y in range(min_year, max_year + 1)}
    else:
        full_years_dist = years_dist


    return FrequencyAnalysisData(
        total_similar_found=total_found,
        years_distribution=full_years_dist,
        papers_distribution=papers_dist if papers_dist else None,
        summary_statement=summary
    )

async def process_question_image(image_bytes: bytes, subject_hint: Optional[str] = None) -> Optional[AnalyzeResponse]:
    """
    Orchestrates the analysis of an uploaded question image.
    """
    log.info("Starting question image analysis process...")

    # 1. OCR (or Vision LLM)
    # For now, default to OCR, but keep Vision function available
    log.info("Attempting OCR...")
    recognized_text = ocr_service.perform_ocr(image_bytes)

    # --- Placeholder for potential Vision LLM call if OCR fails badly ---
    # if not recognized_text or len(recognized_text) < 20: # Heuristic for poor OCR
    #     log.warning("OCR result seems poor or missing. Attempting Vision LLM.")
    #     img_b64 = base64.b64encode(image_bytes).decode('utf-8') # Assumes base64 import
    #     vision_text = await llm_service.get_content_from_image_llm(img_b64)
    #     if vision_text:
    #         log.info("Using text extracted by Vision LLM.")
    #         recognized_text = vision_text
    #     else:
    #         log.error("Both OCR and Vision LLM failed to extract text.")
    #         return None # Cannot proceed without text
    # --- End Placeholder ---

    if not recognized_text:
        log.error("OCR failed to extract any text.")
        # Return a minimal response indicating failure
        return AnalyzeResponse(
             recognized_question_text=None,
             analysis=AnalysisResult(detailed_approach="Failed to read question from image.", key_concepts=[], common_pitfalls=[]),
             syllabus_mapping=None,
             similar_past_questions=[],
             frequency_analysis=calculate_frequency([])
         )

    log.info(f"OCR Result (first 200 chars): {recognized_text[:200]}")

    # 2. Embedding
    log.info("Generating embedding for recognized text...")
    embedding = await embedding_service.generate_text_embedding(recognized_text)
    if not embedding:
        log.error("Failed to generate embedding.")
        # Can still proceed with LLM analysis but without similar questions context
        similar_objectives = []
        frequency_data = calculate_frequency([])
    else:
        # 3. Database Query
        # TODO: Determine subject more robustly (e.g., from user input, LLM classification)
        # For now, using hint or trying without subject filter
        # If removing subject filter from find_similar_objectives, pass None or modify call
        if subject_hint:
             log.info(f"Searching database with subject hint: {subject_hint}")
             similar_objectives = await supabase_service.find_similar_objectives(embedding, subject_hint)
        else:
             log.warning("No subject hint provided. Querying may be less effective or disabled.")
             # Decide if you want to query without subject (potentially slow/irrelevant)
             # similar_objectives = await supabase_service.find_similar_objectives(embedding, None) # If function allows None subject
             similar_objectives = [] # Default to no results if no subject hint

        # 4. Frequency Analysis
        log.info("Calculating frequency analysis...")
        frequency_data = calculate_frequency(similar_objectives)

    # 5. LLM Analysis Generation
    log.info("Generating detailed analysis using LLM...")
    llm_analysis_result = await llm_service.generate_detailed_analysis(
        query_text=recognized_text,
        similar_objectives=similar_objectives, # Pass potentially empty list
        frequency_data=frequency_data.dict() # Pass frequency data as dict
    )

    if not llm_analysis_result:
        log.error("LLM failed to generate detailed analysis.")
        # Return partial results if LLM fails
        return AnalyzeResponse(
             recognized_question_text=recognized_text,
             analysis=AnalysisResult(detailed_approach="Failed to generate AI analysis.", key_concepts=[], common_pitfalls=[]),
             syllabus_mapping=None, # Cannot determine mapping without LLM
             similar_past_questions=[SimilarQuestion(**q) for q in similar_objectives], # Convert raw dicts
             frequency_analysis=frequency_data
         )

    # 6. Structure Final Response
    log.info("Structuring final analysis response...")
    final_response = AnalyzeResponse(
        recognized_question_text=recognized_text,
        analysis=AnalysisResult(
            detailed_approach=llm_analysis_result.get("detailed_approach", "N/A"),
            key_concepts=llm_analysis_result.get("key_concepts", []),
            common_pitfalls=llm_analysis_result.get("common_pitfalls", [])
        ),
        syllabus_mapping=SyllabusMapping(**llm_analysis_result["syllabus_mapping"]) if llm_analysis_result.get("syllabus_mapping") else None,
        similar_past_questions=[SimilarQuestion(**q) for q in similar_objectives], # Convert raw dicts
        frequency_analysis=frequency_data
    )

    log.info("Analysis process completed successfully.")
    return final_response