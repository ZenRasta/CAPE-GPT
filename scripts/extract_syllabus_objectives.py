# extract_syllabus_objectives.py
import os
import re
import json
import logging
from typing import List, Dict, Optional, Any

import fitz  # PyMuPDF
from dotenv import load_dotenv
from supabase import create_client, Client
from openai import OpenAI

# --- Configuration & Initialization ---
# Use DEBUG for detailed tracing, INFO for less verbosity
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')
logging.getLogger("httpx").setLevel(logging.WARNING)  # Reduce noise from http library
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.INFO) # Can increase openai logging if needed

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")  # Use service_role key for backend scripts
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL_NAME = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
LLM_MODEL_NAME = os.getenv("OPENAI_LLM_MODEL", "gpt-4-turbo") # Use a capable model like gpt-4-turbo or gpt-4o for better JSON and structure understanding
EXPECTED_EMBEDDING_DIM = 1536

SYLLABUS_DIR = '/Users/anthonyzamore/Projects/CAPE-GPT/syallabus'  # <-- Your Syllabus Directory

# --- Environment Variable Validation ---
if not SUPABASE_URL: logging.error("Missing environment variable: SUPABASE_URL"); exit(1)
if not SUPABASE_KEY: logging.error("Missing environment variable: SUPABASE_KEY"); exit(1)
if not OPENAI_API_KEY: logging.error("Missing environment variable: OPENAI_API_KEY"); exit(1)
logging.info("Successfully loaded environment variables.")

# --- Initialize Clients ---
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    logging.info("Supabase client initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize Supabase client: {e}", exc_info=True); exit(1)

try:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    logging.info(f"OpenAI client initialized. Using embedding model: {EMBEDDING_MODEL_NAME}. Using LLM: {LLM_MODEL_NAME}")
except Exception as e:
    logging.error(f"Failed to initialize OpenAI client: {e}", exc_info=True); exit(1)


# --- Helper Functions ---

def standardize_subject_name(filename: str) -> Optional[str]:
    """
    Converts a syllabus filename into a standardized subject name.
    **REQUIRES CUSTOMIZATION** based on your filenames.
    """
    # Option A: Dictionary Mapping (Recommended for irregular names)
    filename_to_subject = {
        "CAPESociologySyllabuswithSpecimenPaperaqndMarkScheme1.pdf": "Sociology",
        "CAPE Management of Business.pdf": "Management of Business",
        "CAPE_Communication_Studies.pdf": "Communication Studies",
        "CAPE Accounting.pdf": "Accounting",
        "CAPE Physics.pdf": "Physics",
        "CAPE Pure Mathematics.pdf": "Pure Mathematics",
        "CAPE Environmental Science.pdf": "Environmental Science",
        "CAPE Computer Science.pdf": "Computer Science",
        "Economics.pdf": "Economics",  # Assuming CAPE Economics
        "CAPEChemistry.pdf": "Chemistry",
        "CAPE History.pdf": "History",
        "CAPE Physical Education & Sport.pdf": "Physical Education and Sport",
        "CAPEBiology.pdf": "Biology",
        "CAPE Applied Mathematics Syllabus.pdf": "Applied Mathematics",
        # Add all your syllabus filenames and their desired subject names here
    }
    subject = filename_to_subject.get(filename)
    if subject:
        logging.debug(f"Standardized subject for '{filename}' to '{subject}' using mapping.")
        return subject

    # Option B: Regex (Fallback or alternative if names are very consistent)
    # match = re.match(r"CAPE_?(.+?)(?:_?Syllabus)?\.pdf", filename, re.IGNORECASE)
    # if match:
    #     subject = match.group(1).replace('_', ' ').strip()
    #     logging.debug(f"Standardized subject for '{filename}' to '{subject}' using regex.")
    #     return subject

    logging.warning(f"Could not standardize subject name for filename: '{filename}'. Please update the mapping or regex in 'standardize_subject_name'.")
    return None  # Indicate failure to standardize


def generate_embedding(text: str, model_name: str = EMBEDDING_MODEL_NAME) -> Optional[List[float]]:
    """Generates sentence embedding using OpenAI API."""
    if not text or not text.strip():
        logging.warning("Attempted to generate embedding for empty text.")
        return None
    try:
        text_to_embed = text.replace("\n", " ").strip()
        if not text_to_embed:
            logging.warning("Text became empty after cleaning, skipping embedding.")
            return None
        logging.debug(f"Requesting embedding for objective text (first 100 chars): {text_to_embed[:100]}...")
        response = openai_client.embeddings.create(input=[text_to_embed], model=model_name)
        logging.debug(f"Received embedding response.")
        if not response.data or not response.data[0].embedding:
            logging.error(f"OpenAI embedding response data or embedding is missing! Response: {response}")
            return None
        embedding = response.data[0].embedding
        if len(embedding) != EXPECTED_EMBEDDING_DIM:
            logging.warning(
                f"Embedding dimension mismatch for model {model_name}. Expected {EXPECTED_EMBEDDING_DIM}, got {len(embedding)}.")
        logging.debug(f"Embedding generated successfully (dim: {len(embedding)}).")
        return embedding
    except Exception as e:
        logging.error(f"EXCEPTION during OpenAI embedding generation for objective text '{text[:100]}...': {e}",
                      exc_info=True)
        return None


def get_objectives_from_llm(page_text: str, subject_name: str, filename: str, page_num: int) -> List[Dict[str, Any]]:
    """
    Uses LLM to extract learning objectives from the text of a single page.
    Handles responses that are lists OR single objective dictionaries.
    """
    if not page_text.strip():
        logging.debug(f"Skipping LLM call for page {page_num} of '{filename}' due to empty text.")
        return []

    prompt = f"""
You are an expert educational assistant analyzing syllabus documents. Given the following text from the {subject_name} syllabus PDF (filename: {filename}, page: {page_num}), identify and extract all specific learning objectives. For each objective, determine its corresponding Unit and Module if possible from the surrounding text or headings on this page or commonly associated with objectives listed like this.

Present the output as a JSON list, where each object has the following keys:
- "unit": The Unit identifier (e.g., "Unit 1", "Unit 2") or null if not found/inferable.
- "module": The Module identifier (e.g., "Module 1", "Module 3") or null if not found/inferable.
- "objective_id": The specific ID of the objective (e.g., "1.a.i", "3.", "(b)", "SO-1.1") or null if none is explicitly listed next to the objective.
- "objective_text": The full, exact text of the specific learning objective (e.g., "explain the concepts associated with stratification").
- "source_document": "{filename}"
- "approx_page": {page_num}

Ensure you only extract specific, actionable learning objectives, typically starting with verbs like 'explain', 'calculate', 'define', 'analyze', 'discuss', 'evaluate', 'describe', 'identify', 'compare', 'contrast', etc. Ignore general section headings, introductory paragraphs, rationales, aims, content summaries (unless they clearly state an objective), resource lists, assessment details, and examples unless they are part of the objective itself. Focus on what the student should be *able* to do.

Syllabus Text from Page {page_num}:
---
{page_text}
---

Respond ONLY with the JSON list. Even if only one objective is found, return it inside a JSON list like `[ {{"objective_text": "..." , ...}} ]`. If no objectives are found on this page, return an empty JSON list `[]`.
"""
    logging.debug(
        f"Sending request to LLM ({LLM_MODEL_NAME}) for objective extraction from page {page_num} of '{filename}'. Prompt length approx: {len(prompt)}")

    try:
        response = openai_client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an AI assistant extracting structured data from educational documents."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Lower temperature for more factual extraction
            response_format={"type": "json_object"}  # Request JSON output
        )
        logging.debug(f"Received LLM response for page {page_num}.")

        raw_response_content = response.choices[0].message.content
        logging.debug(f"LLM Raw JSON Output (Page {page_num}): {raw_response_content}")

        extracted_objectives = []
        try:
            parsed_data = json.loads(raw_response_content)

            potential_list = None
            if isinstance(parsed_data, list):
                potential_list = parsed_data
                logging.debug(f"LLM returned a JSON list directly for page {page_num}.")
            elif isinstance(parsed_data, dict):
                objective_text_val = parsed_data.get("objective_text")
                # Check if the dictionary itself represents a single valid objective
                if objective_text_val and isinstance(objective_text_val, str):
                    logging.debug(
                        f"LLM returned a single JSON object that looks like an objective for page {page_num}. Wrapping it in a list.")
                    potential_list = [parsed_data]  # Treat the single object as a list of one
                else:
                    # Check if the dictionary contains a list under some key
                    found_list_in_dict = None
                    for key, value in parsed_data.items():
                        if isinstance(value, list):
                            found_list_in_dict = value
                            logging.debug(f"Found list under key '{key}' in LLM dictionary response for page {page_num}.")
                            break
                    if found_list_in_dict is not None:
                        potential_list = found_list_in_dict
                    else:
                        logging.warning(
                            f"LLM response for page {page_num} was a JSON dict but contained no list and didn't look like a single objective: {raw_response_content}")

            # Validate items in the potential_list
            if potential_list is not None:
                for item in potential_list:
                    item_objective_text = item.get("objective_text")
                    if isinstance(item, dict) and item_objective_text and isinstance(item_objective_text, str) and item_objective_text.strip():
                        extracted_objectives.append(item)
                    else:
                        logging.warning(f"LLM response for page {page_num} contained invalid/empty item in list: {item}")

            if not potential_list and not extracted_objectives:
                logging.info(f"No valid objectives parsed from LLM response for page {page_num}.")
            else:
                logging.info(f"Successfully parsed {len(extracted_objectives)} objectives from LLM for page {page_num}.")

            return extracted_objectives

        except json.JSONDecodeError as json_e:
            logging.error(
                f"JSONDecodeError parsing LLM response for page {page_num}: {json_e}. Response: {raw_response_content}",
                exc_info=False) # Avoid duplicate stack trace if error is just bad JSON
            return []

    except Exception as e:
        logging.error(
            f"EXCEPTION during OpenAI LLM call for objective extraction (Page {page_num}, File '{filename}'): {e}",
            exc_info=True)
        return []


def store_objectives_in_supabase(objective_data: List[Dict[str, Any]]):
    """Generates embeddings and stores extracted objectives in Supabase, clearing old data first."""
    if not objective_data:
        logging.warning("No objective data provided to store in Supabase.")
        return

    records_to_insert = []
    skipped_embedding_failure = 0
    valid_objectives_count = len(objective_data)
    logging.info(f"Preparing {valid_objectives_count} extracted objectives for Supabase insertion...")

    for i, obj in enumerate(objective_data):
        objective_text = obj.get("objective_text")
        if not objective_text or not objective_text.strip():
            logging.warning(
                f"Skipping objective {i + 1}/{valid_objectives_count} due to missing/empty 'objective_text'. Data: {obj}")
            continue

        embedding = generate_embedding(objective_text)
        if embedding is None:
            logging.warning(
                f"Skipping objective {i + 1}/{valid_objectives_count} (Page: {obj.get('approx_page')}) due to embedding generation failure.")
            skipped_embedding_failure += 1
            continue

        record = {
            "subject": obj.get("subject"),
            "unit": obj.get("unit"),
            "module": obj.get("module"),
            "section_title": None, # Or derive from surrounding text if added to LLM prompt
            "objective_id": obj.get("objective_id"),
            "content": objective_text, # Store objective text here
            "embedding": embedding,
            "source_document": obj.get("source_document"),
            "page_number": obj.get("approx_page")
        }
        records_to_insert.append(record)

    final_records_count = len(records_to_insert)
    logging.info(
        f"Finished preparing records. Total valid records for insertion: {final_records_count}. Skipped (Embedding Failure): {skipped_embedding_failure}")

    if not records_to_insert:
        logging.warning("No valid records remaining after embedding process. Nothing to insert.")
        return

    # --- Clear existing table data ---
    try:
        logging.warning("Attempting to DELETE existing data from 'syllabus_sections' table...")
        delete_response = supabase.table("syllabus_sections").delete().neq("id", -1).execute() # Filter matches all rows
        logging.info(f"Delete operation response: {delete_response}")
        if hasattr(delete_response, 'error') and delete_response.error:
            logging.error(f"Failed to delete existing data: {delete_response.error}.")
        else:
             count = len(getattr(delete_response, 'data', [])) # Check how many rows were returned (optional)
             logging.info(f"Successfully deleted existing data from 'syllabus_sections' (approx {count} rows affected).")
    except Exception as e:
        logging.error(f"EXCEPTION during table clearing: {e}.", exc_info=True)
        logging.warning("Proceeding with insert despite potential clearing failure. Duplicates may occur if run multiple times.")

    # --- Insert new data in batches ---
    batch_size = 100
    total_inserted = 0
    logging.info(f"Attempting to insert {final_records_count} records into 'syllabus_sections' in batches of {batch_size}...")

    for i in range(0, final_records_count, batch_size):
        batch = records_to_insert[i:i + batch_size]
        current_batch_num = i // batch_size + 1
        total_batches = (final_records_count + batch_size - 1) // batch_size
        logging.info(f"Preparing Batch {current_batch_num}/{total_batches} ({len(batch)} records)...")

        try:
            logging.info(f"Attempting Supabase insert for batch {current_batch_num}...")
            response = supabase.table("syllabus_sections").insert(batch).execute()
            logging.info(f"Supabase insert call executed for batch {current_batch_num}.")
            logging.debug(f"Supabase response object for batch {current_batch_num}: {response}")

            has_error = hasattr(response, 'error') and response.error is not None
            has_data = hasattr(response, 'data') and response.data is not None
            status = getattr(response, 'status_code', None)

            if has_error:
                logging.error(f"SUPABASE RETURNED AN ERROR for batch {current_batch_num}: {response.error}")
            elif status is not None and 200 <= status < 300:
                 logging.info(f"Successfully inserted batch {current_batch_num} (Status: {status}).")
                 total_inserted += len(batch)
            elif has_data: # Fallback check if status code isn't available but data is
                 logging.info(f"Successfully inserted batch {current_batch_num} (Data returned).")
                 total_inserted += len(batch)
            else:
                 logging.warning(f"Supabase insert for batch {current_batch_num} likely failed (Status: {status}, No error/data). Response: {response}")

        except Exception as e:
            logging.error(f"EXCEPTION occurred during Supabase insert for batch {current_batch_num}: {e}", exc_info=True)
            if batch:
                try:
                    first_record_preview = {k: v for k, v in batch[0].items() if k != 'embedding'}
                    logging.error(f"First record of failed batch (preview): {json.dumps(first_record_preview, indent=2, default=str)}")
                except Exception as log_ex:
                    logging.error(f"Could not serialize the first record of the failed batch for logging: {log_ex}")

    logging.info(f"--- Supabase Insertion Summary ---")
    logging.info(f"Total records prepared for insertion: {final_records_count}")
    logging.info(f"Total records successfully inserted (based on response checks): {total_inserted}")


# --- Main Execution ---
if __name__ == "__main__":
    logging.info(f"Starting syllabus objective extraction process from directory: {SYLLABUS_DIR}")

    if not os.path.isdir(SYLLABUS_DIR):
        logging.error(f"Syllabus directory not found: {SYLLABUS_DIR}"); exit(1)

    all_extracted_objectives = [] # Initialize list to store objectives from ALL files
    processed_files_count = 0
    skipped_files_count = 0

    for filename in os.listdir(SYLLABUS_DIR):
        # Process only PDF files, ignore hidden/temp files
        if filename.lower().endswith(".pdf") and not filename.startswith('.') and not filename.startswith('~'):
            filepath = os.path.join(SYLLABUS_DIR, filename)
            logging.info(f"--- Processing Syllabus File: {filename} ---")
            processed_files_count += 1

            # 1. Standardize Subject Name
            subject = standardize_subject_name(filename)
            if not subject:
                logging.error(f"Skipping file '{filename}' due to inability to standardize subject name.")
                skipped_files_count += 1
                continue

            # 2. Extract Objectives using LLM page by page
            objectives_from_this_file = [] # Reset for each file
            try:
                doc = fitz.open(filepath)
                logging.info(f"Opened '{filename}', Pages: {len(doc)}")
                for page_num_zero_based, page in enumerate(doc):
                    page_num_actual = page_num_zero_based + 1
                    try:
                        logging.debug(f"Extracting text from page {page_num_actual}...")
                        page_text = page.get_text("text")
                        # Skip pages with very little text
                        if not page_text or len(page_text.strip()) < 75:
                             logging.debug(f"Skipping page {page_num_actual}, short/empty content.")
                             continue

                        # Call LLM to get objectives from this page's text
                        page_objectives = get_objectives_from_llm(page_text, subject, filename, page_num_actual)

                        if page_objectives: # Only add subject if objectives were found
                            for obj in page_objectives:
                                obj['subject'] = subject # Add standardized subject
                            objectives_from_this_file.extend(page_objectives)
                            # Log objectives found per page
                            logging.debug(f"Added {len(page_objectives)} objectives from page {page_num_actual} to list for '{filename}'.")

                    except Exception as page_ex:
                        logging.error(f"Error processing page {page_num_actual} of {filename}: {page_ex}", exc_info=True)
                doc.close()
                # Log total objectives for the current file
                logging.info(f"Finished processing '{filename}'. Extracted total {len(objectives_from_this_file)} objectives from this file.")
                all_extracted_objectives.extend(objectives_from_this_file)
                # Log cumulative total
                logging.info(f"Cumulative objectives extracted so far: {len(all_extracted_objectives)}")

            except Exception as file_ex:
                 logging.error(f"Failed to process PDF file {filename}: {file_ex}", exc_info=True)
                 skipped_files_count += 1

    # --- Final Summary Log ---
    logging.info(f"--- Syllabus File Processing Summary ---")
    logging.info(f"Files processed: {processed_files_count}")
    logging.info(f"Files skipped: {skipped_files_count}")
    logging.info(f"FINAL Total objectives extracted across all files: {len(all_extracted_objectives)}") # Log before check

    # 3. Store extracted objectives in Supabase
    if all_extracted_objectives: # Check the final accumulated list
        logging.info("--- Starting Supabase Storage Process ---")
        store_objectives_in_supabase(all_extracted_objectives)
    else:
        # Make sure this warning stands out if the list is unexpectedly empty
        logging.warning("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        logging.warning("FINAL CHECK: No objectives were accumulated. Nothing to store in Supabase.")
        logging.warning("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    logging.info("--- Syllabus Objective Extraction Script Complete ---")
    