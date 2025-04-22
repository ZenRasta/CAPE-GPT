# llm_service.py
'''
Purpose: Handles all interactions with the primary Large Language Model (OpenAI GPT-4o in this case) for complex reasoning, generation, and potentially vision tasks.

Input: Depends on the task. Could be text prompts, image data, context (like similar questions).

Output: Processed results from the LLM, often structured text or JSON.

Responsibilities: Formatting prompts (prompt engineering), calling the OpenAI Chat Completions API (potentially with image data), handling LLM API errors, parsing the LLM's response (e.g., extracting JSON), potentially handling refusals or irrelevant answers. We'll put both the image-to-text extraction and the analysis/mapping generation here.
'''

# backend/app/services/llm_service.py

import logging
import json
from typing import Optional, List, Dict, Any, Tuple

from openai import OpenAI, APIError, RateLimitError
from app.core.config import OPENAI_API_KEY, LLM_MODEL # Use config

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')
log = logging.getLogger(__name__)

# Initialize OpenAI client
try:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    log.info("OpenAI client initialized for llm_service.")
except Exception as e:
    log.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
    openai_client = None

# --- Vision Processing ---
async def get_content_from_image_llm(image_base64: str) -> Optional[str]:
    """
    Uses Vision LLM (GPT-4o/Turbo) to extract question content from an image.

    Args:
        image_base64: Base64 encoded string of the image.

    Returns:
        Extracted text content string or None on failure/refusal.
    """
    if not openai_client: log.error("Vision LLM: OpenAI client not initialized."); return None
    if not image_base64: log.warning("Vision LLM: No base64 image data provided."); return None

    log.info(f"Sending image to Vision LLM ({LLM_MODEL})...")
    try:
        # Ensure the model specified supports vision
        response = await openai_client.chat.completions.create( # Use await for async call
            model=LLM_MODEL, # e.g., "gpt-4-turbo", "gpt-4o"
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract the main exam question text visible in this image. Ignore headers, footers, page numbers, margin notes, instructions, and marks allocations unless they are essential to understanding the question itself. If there are multiple distinct questions (e.g., 1a, 1b), extract all of them, clearly separated. Briefly describe diagrams/figures/code only if they are directly part of the question content."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}", # Adjust mime type if necessary (png?)
                                "detail": "high" # Use high detail for better accuracy on exam text
                            },
                        },
                    ],
                }
            ],
            max_tokens=800 # Allow more tokens for potentially complex questions/descriptions
        )
        if response.choices and response.choices[0].message.content:
            extracted_content = response.choices[0].message.content.strip()
            log.info(f"Vision LLM extracted content: {extracted_content[:150]}...")
            # Basic check for refusals
            if "can't assist" in extracted_content.lower() or "cannot provide" in extracted_content.lower() or "unable to" in extracted_content.lower():
                log.warning(f"Vision LLM refused or could not extract content: {extracted_content}")
                return None
            return extracted_content
        else:
            log.warning("Vision LLM response was empty or invalid.")
            return None

    except APIError as api_err: log.error(f"OpenAI API Error (Vision): {api_err}", exc_info=True); return None
    except RateLimitError as rate_err: log.error(f"OpenAI Rate Limit Error (Vision): {rate_err}", exc_info=True); return None
    except Exception as e: log.error(f"Unexpected EXCEPTION during Vision LLM call: {e}", exc_info=True); return None


# --- Text Analysis & Mapping ---
async def generate_detailed_analysis(
    query_text: str,
    similar_objectives: List[Dict[str, Any]],
    frequency_data: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Uses LLM to generate detailed analysis, approach, pitfalls, and confirm mapping.

    Args:
        query_text: The text extracted from the user's question (OCR or Vision).
        similar_objectives: List of similar objectives found in the database.
        frequency_data: Dictionary containing frequency analysis results.

    Returns:
        A dictionary containing 'detailed_approach', 'key_concepts',
        'common_pitfalls', and 'syllabus_mapping' (with objective_id/text), or None.
    """
    if not openai_client: log.error("Analysis LLM: OpenAI client not initialized."); return None
    if not query_text: log.warning("Analysis LLM: No query text provided."); return None

    # Select the top 1-3 most similar objectives to provide as context
    # Assuming objectives are sorted by similarity by the RPC function
    context_objectives = similar_objectives[:3] # Limit context to top 3

    if not context_objectives:
        log.warning("Analysis LLM: No similar objectives provided for context. Cannot generate detailed analysis.")
        # Return a structure indicating mapping failed if no context
        return {
             "detailed_approach": "Could not determine approach without matching syllabus objectives.",
             "key_concepts": [],
             "common_pitfalls": ["Could not determine common pitfalls."],
             "syllabus_mapping": None
         }

    # Format context for the prompt
    objective_context = "\n---\n".join([
        f"Similar Objective Option {i+1} (ID: {s.get('objective_id', 'N/A')} | Sim: {s.get('similarity', -1):.3f}):\n{s.get('content', '')}"
        for i, s in enumerate(context_objectives)
    ])
    frequency_summary = frequency_data.get("summary_statement", "Frequency data unavailable.")

    prompt = f"""
You are ExamSage, an expert exam preparation assistant for STEM subjects (Math, Physics, CompSci, etc.).
A student has provided the following question snippet:

**Student's Question:**
---
{query_text}
---

Based on a vector search, the following syllabus objectives seem most relevant:
**Relevant Syllabus Objectives:**
---
{objective_context}
---

**Past Frequency:** {frequency_summary}

**Your Task:**
1.  **Confirm Mapping:** Identify the SINGLE best matching syllabus objective from the options provided above that directly corresponds to the student's question.
2.  **Generate Analysis:** Provide helpful guidance based *only* on the student's question and the single best matching objective identified in step 1. Do NOT invent information not present in the provided context.

**Response Format:**
Respond ONLY with a valid JSON object containing the following keys:
- "detailed_approach": A concise, step-by-step methodology (as a string with newlines) for how a student should approach solving the student's question, referencing the core concept from the best matching objective.
- "key_concepts": A list of essential keywords, topics, formulas, or concepts (list of strings) directly required by the best matching objective and relevant to the student's question.
- "common_pitfalls": A list of 2-3 common mistakes or misunderstandings (list of strings) students typically make related to the concepts involved in the best matching objective.
- "syllabus_mapping": An object containing the details of the *single best* matching objective identified in step 1: {{"objective_id": "The ID or null", "objective_text": "The full objective text"}}. If NO objective is a good match, return null for this entire "syllabus_mapping" field.

Example Response:
{{
  "detailed_approach": "1. Identify the type of logic gate(s).\n2. Construct the truth table inputs.\n3. Evaluate the output for each input row based on the gate's function.",
  "key_concepts": ["Logic Gates", "Truth Tables", "AND gate", "NOT gate", "Boolean Algebra"],
  "common_pitfalls": ["Incorrectly evaluating the intermediate gate output.", "Missing input combinations in the truth table.", "Confusing AND with OR logic."],
  "syllabus_mapping": {{ "objective_id": "CS-U1-M1-SO1", "objective_text": "Construct the truth table for a given logic circuit." }}
}}

Ensure the output is ONLY the valid JSON object.
"""

    log.info(f"Sending request to LLM ({LLM_MODEL}) for detailed analysis...")
    log.debug(f"Analysis prompt context length approx: {len(prompt)}")
    try:
        response = await openai_client.chat.completions.create( # Use await for async call
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are ExamSage generating helpful, structured exam question analysis based ONLY on provided context. Respond ONLY in JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3, # Slightly higher temp for more descriptive generation
            max_tokens=1000, # Allow ample space for detailed response
            response_format={"type": "json_object"}
        )
        log.debug(f"LLM analysis response received.")

        if response.choices and response.choices[0].message.content:
            llm_output_text = response.choices[0].message.content.strip()
            log.debug(f"LLM Raw Analysis Output: {llm_output_text}")
            try:
                # Parse the JSON response
                analysis_data = json.loads(llm_output_text)
                # Basic validation of expected keys
                if all(k in analysis_data for k in ["detailed_approach", "key_concepts", "common_pitfalls", "syllabus_mapping"]):
                    log.info("LLM analysis generated successfully.")
                    # Ensure syllabus_mapping is None if it was returned as empty/null object by LLM
                    if not analysis_data.get("syllabus_mapping"):
                         analysis_data["syllabus_mapping"] = None
                    elif not analysis_data["syllabus_mapping"].get("objective_text"): # If mapping object exists but text is missing
                         log.warning("LLM returned syllabus_mapping object without objective_text, setting mapping to None.")
                         analysis_data["syllabus_mapping"] = None

                    return analysis_data
                else:
                    log.error(f"LLM analysis response missing required keys. Response: {llm_output_text}")
                    return None
            except json.JSONDecodeError as json_e:
                log.error(f"Failed to parse JSON from LLM analysis response: {json_e}. Response: {llm_output_text}")
                return None
            except Exception as parse_ex:
                log.error(f"Error processing LLM analysis JSON response: {parse_ex}", exc_info=True)
                return None
        else:
            log.warning("LLM analysis response was empty or invalid.")
            return None

    except APIError as api_err: log.error(f"OpenAI API Error (Analysis): {api_err}", exc_info=True); return None
    except RateLimitError as rate_err: log.error(f"OpenAI Rate Limit Error (Analysis): {rate_err}", exc_info=True); return None
    except Exception as e: log.error(f"Unexpected EXCEPTION during LLM analysis call: {e}", exc_info=True); return None