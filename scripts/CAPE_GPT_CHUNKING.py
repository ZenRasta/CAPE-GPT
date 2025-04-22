# Full CAPE_GPT_CHUNKING.py script with Subject Standardization, BBox Fix, Vision LLM, AND Explicit JSON Conversion for Insert (FINAL)

import os
import re
import io
import json # Still needed for dumps and logging
import base64
import logging
import datetime # Added for serializer
from typing import List, Dict, Optional, Tuple, Any

import fitz  # PyMuPDF
import sympy
from PIL import Image
import pytesseract
from dotenv import load_dotenv
from supabase import create_client, Client
from openai import OpenAI

# --- Configuration & Initialization ---
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.INFO)

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY") # Use service_role key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL_NAME = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
EXPECTED_EMBEDDING_DIM = 1536
# *** USE A VISION-CAPABLE MODEL ***
LLM_MODEL_NAME = os.getenv("OPENAI_LLM_MODEL", "gpt-4o") # Using vision model

# --- Environment Variable Validation ---
if not SUPABASE_URL: logging.error("Missing environment variable: SUPABASE_URL"); exit(1)
if not SUPABASE_KEY: logging.error("Missing environment variable: SUPABASE_KEY"); exit(1)
if not OPENAI_API_KEY: logging.error("Missing environment variable: OPENAI_API_KEY"); exit(1)
logging.info("Successfully loaded environment variables.")


# Initialize Supabase client
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    logging.info("Supabase client initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize Supabase client: {e}", exc_info=True); exit(1)

# Initialize OpenAI client
try:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    logging.info(f"OpenAI client initialized. Using embedding model: {EMBEDDING_MODEL_NAME}. Using LLM: {LLM_MODEL_NAME}")
except Exception as e:
    logging.error(f"Failed to initialize OpenAI client: {e}", exc_info=True); exit(1)


# --- Helper Functions ---

def extract_equations(text: str) -> List[Dict[str, str]]:
    """Extracts LaTeX equations and attempts to parse them with SymPy."""
    equations_data = []
    latex_matches = re.findall(r'\$\$(.*?)\$\$|\$(.*?)\$', text, re.DOTALL)
    for match in latex_matches:
        latex_str = match[0] if match[0] else match[1]; latex_str = latex_str.strip()
        if not latex_str: continue
        parsed_text = f"Equation: {latex_str}"
        try: sympy_expr = sympy.parsing.latex.parse_latex(latex_str); parsed_text = f"Equation represents: {str(sympy_expr)}"
        except Exception as e: logging.warning(f"SymPy parse failed for '{latex_str}': {e}")
        equations_data.append({"latex": latex_str, "text": parsed_text})
    return equations_data

tesseract_available = True
def ocr_image(image_bytes: bytes) -> Optional[str]:
    """Performs OCR on image bytes using Tesseract."""
    global tesseract_available;
    if not tesseract_available: return None
    try: img = Image.open(io.BytesIO(image_bytes)); text = pytesseract.image_to_string(img); logging.debug(f"OCR success."); return text.strip()
    except pytesseract.TesseractNotFoundError: logging.error("Tesseract not found."); tesseract_available = False; return None
    except Exception as e: logging.error(f"OCR Error: {e}", exc_info=True); return None

def encode_image_base64(image_bytes: bytes) -> str:
     """Encodes image bytes to Base64 string."""
     return base64.b64encode(image_bytes).decode('utf-8')

def generate_embedding(text: str, model_name: str = EMBEDDING_MODEL_NAME) -> Optional[List[float]]:
    """Generates sentence embedding using OpenAI API."""
    if not text or not text.strip(): logging.warning("Embed: Empty text."); return None
    try:
        text_to_embed = text.replace("\n", " ").strip();
        if not text_to_embed: logging.warning("Embed: Text empty after clean."); return None
        logging.debug(f"Embed request (start): {text_to_embed[:100]}...")
        response = openai_client.embeddings.create(input=[text_to_embed], model=model_name);
        logging.debug(f"Embed response received.")
        if not response.data or not response.data[0].embedding: logging.error(f"Embed invalid response: {response}"); return None
        embedding = response.data[0].embedding
        if len(embedding) != EXPECTED_EMBEDDING_DIM: logging.warning(f"Embed dim mismatch: {len(embedding)} vs {EXPECTED_EMBEDDING_DIM}.")
        logging.debug(f"Embed success (dim: {len(embedding)}).")
        return embedding
    except Exception as e: logging.error(f"EXCEPTION during OpenAI embedding: {e}", exc_info=True); return None

# --- Revised extract_text_images_equations_from_pdf (Includes BBox Fix) ---
def extract_text_images_equations_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """Extracts text blocks, images (with OCR), and equations from each page of a PDF."""
    all_extracted_data = []
    try:
        doc = fitz.open(pdf_path)
        logging.info(f"Opened PDF: {pdf_path}, Pages: {len(doc)}")
    except Exception as e:
        logging.error(f"Failed to open PDF: {pdf_path}. Error: {e}", exc_info=True)
        return []

    for page_num, page in enumerate(doc):
        page_num_actual = page_num + 1
        logging.debug(f"Processing page {page_num_actual}/{len(doc)}...")
        page_items = []
        has_actual_text_on_page = False # Flag for this specific page

        # 1. Extract Text Blocks
        try:
            blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT | fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
            for block_idx, block in enumerate(blocks):
                 if block.get('type') == 0 and block.get('lines'):
                     block_text = "".join(span["text"] for line in block["lines"] for span in line["spans"]).strip()
                     if block_text:
                         equations_in_block = extract_equations(block_text)
                         page_items.append({ "type": "text", "content": block_text, "equations": equations_in_block,
                                            "page": page_num_actual, "bbox": block.get("bbox", None) });
                         logging.debug(f"Page {page_num_actual}: Extracted text block {block_idx}.")
                         if len(block_text) > 20: # Heuristic: assume meaningful text if > 20 chars
                             has_actual_text_on_page = True
        except Exception as text_ex: logging.error(f"Error extracting text page {page_num_actual}: {text_ex}", exc_info=True)

        # 2. Extract Images
        try:
            img_list = page.get_images(full=True)
            if img_list: logging.debug(f"Page {page_num_actual}: Found {len(img_list)} images.")
            for img_index, img_info in enumerate(img_list):
                 xref = img_info[0]; logging.debug(f"Page {page_num_actual}: Processing image xref {xref}...")
                 bbox_to_store = None
                 try:
                     base_image = doc.extract_image(xref)
                     if not base_image or not base_image.get("image"): logging.warning(f"No image data for xref {xref} p{page_num_actual}."); continue
                     image_bytes = base_image["image"]
                     try: # Robust BBOX Handling
                         img_rects = page.get_image_rects(xref, transform=True)
                         if img_rects:
                             first_item = img_rects[0]
                             if isinstance(first_item, fitz.Rect): bbox_to_store = first_item.irect; logging.debug(f"Using .irect: {bbox_to_store}")
                             elif isinstance(first_item, tuple) and len(first_item) == 4:
                                 try: bbox_to_store = tuple(map(int, first_item)); logging.debug(f"Using tuple converted to int: {bbox_to_store}")
                                 except ValueError: logging.warning(f"Could not convert bbox tuple {first_item}"); bbox_to_store = None
                             else: logging.warning(f"Unexpected bbox type: {type(first_item)}.")
                         else: logging.warning(f"Empty img_rects xref {xref} p{page_num_actual}.")
                     except Exception as bbox_ex: logging.error(f"Error getting/processing bbox for xref {xref} p{page_num_actual}: {bbox_ex}"); bbox_to_store = None
                     ocr_text = ocr_image(image_bytes) or ""
                     page_items.append({ "type": "image", "base64_data": encode_image_base64(image_bytes), "ocr_text": ocr_text,
                                        "extension": base_image.get("ext", "unknown"), "page": page_num_actual, "bbox": bbox_to_store }); logging.debug(f"Page {page_num_actual}: Processed image xref {xref}.")
                 except Exception as img_ex: logging.error(f"Error processing image xref {xref} p{page_num_actual}: {img_ex}", exc_info=True)
        except Exception as image_list_ex: logging.error(f"Error getting image list page {page_num_actual}: {image_list_ex}", exc_info=True)

        # Add page-level flag indicating if text blocks were found
        for item in page_items:
            item['has_text_blocks'] = has_actual_text_on_page

        if page_items: logging.debug(f"Adding {len(page_items)} items from page {page_num_actual}."); all_extracted_data.extend(page_items)
        else: logging.debug(f"No items extracted page {page_num_actual}.")

    doc.close(); logging.info(f"Finished extraction {pdf_path}. Items extracted: {len(all_extracted_data)}"); return all_extracted_data
# --- End Revised Function ---

# --- Revised chunk_extracted_data function ---
def chunk_extracted_data(extracted_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Chunks extracted PDF data, grouping items by page. Each page becomes one chunk."""
    logging.info(f"Starting page-based chunking for {len(extracted_data)} extracted items...")
    chunks = []; items_by_page: Dict[int, List[Dict[str, Any]]] = {}

    for item in extracted_data: # Group by page
        page_num = item.get("page")
        if page_num:
            if page_num not in items_by_page: items_by_page[page_num] = []
            items_by_page[page_num].append(item)
    logging.debug(f"Grouped items into {len(items_by_page)} pages.")

    for page_num in sorted(items_by_page.keys()): # Create chunk per page
        page_items = items_by_page[page_num]
        logging.debug(f"Creating chunk for page {page_num} with {len(page_items)} items.")
        page_content_list = []; page_equations = []; page_images = []
        combined_text = ""; has_text_blocks = False # Default flag for the page

        page_items.sort(key=lambda item: (item["bbox"][1] if item.get("bbox") and len(item["bbox"]) >= 2 else 0))

        for item in page_items: # Consolidate content for the page
            if item["type"] == "text":
                if item["content"] and item["content"].strip():
                     page_content_list.append(item["content"])
                     combined_text += item["content"] + "\n"
                     # Use flag directly from extraction - more reliable
                     if item.get('has_text_blocks'):
                         has_text_blocks = True
                page_equations.extend(item["equations"])
                for eq in item["equations"]: combined_text += eq["text"] + "\n"
            elif item["type"] == "image":
                img_data = {"base64_data": item["base64_data"], "ocr_text": item["ocr_text"], "extension": item["extension"]}
                page_images.append(img_data)
                if item["ocr_text"]: combined_text += "Image Content: " + item["ocr_text"] + "\n"

        # Determine if the page is primarily image-based
        is_image_dominant = (not has_text_blocks and page_images)

        # Assemble the chunk dictionary
        chunk = { "page": page_num, "page_text_content": "\n".join(page_content_list).strip(), "equations": page_equations,
                  "images": page_images, "combined_text_for_embedding": combined_text.strip(), "is_image_dominant": is_image_dominant }
        chunks.append(chunk)
        logging.debug(f"Created chunk for page {page_num}. Image dominant: {is_image_dominant}")

    logging.info(f"Page-based chunking complete. Generated {len(chunks)} chunks.")
    return chunks
# --- End Revised Function ---

def query_syllabus_db(chunk_embedding: List[float], subject: str, match_threshold: float = 0.7, match_count: int = 5) -> List[Dict[str, Any]]:
    """Queries the syllabus_sections table for similar objective chunks based on embedding."""
    if chunk_embedding is None: logging.warning("Cannot query syllabus DB: Input embedding is None."); return []
    query_subject = subject.strip(); logging.info(f"Querying syllabus DB for subject '{query_subject}' (Thresh {match_threshold})...")
    try:
        response = supabase.rpc('match_syllabus_sections', { 'query_embedding': chunk_embedding, 'query_subject': query_subject, 'match_threshold': match_threshold, 'match_count': match_count }).execute()
        logging.debug(f"RPC 'match_syllabus_sections' response: {response}")
        if hasattr(response, 'error') and response.error: logging.error(f"Supabase RPC error: {response.error}"); return []
        if hasattr(response, 'data') and response.data:
            if isinstance(response.data, list): logging.info(f"Retrieved {len(response.data)} objectives."); return response.data
            else: logging.warning(f"Unexpected RPC data type: {type(response.data)}"); return []
        else: logging.info(f"No relevant objectives found."); return []
    except Exception as e:
        logging.error(f"EXCEPTION during RPC: {e}", exc_info=True)
        if "Could not find the function" in str(e) or ('code' in str(e) and 'PGRST202' in str(e)): logging.critical("!!! FATAL: SQL function 'match_syllabus_sections' missing !!!")
        else: logging.error("Ensure 'match_syllabus_sections' function exists.")
        return []

def get_mapping_from_llm(past_paper_chunk_text: str, relevant_syllabus_chunks: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    """Uses an LLM to map the past paper chunk to the most relevant syllabus objective."""
    if not relevant_syllabus_chunks: logging.info("LLM Map: No relevant objectives."); return None, None
    logging.debug(f"Preparing LLM prompt with {len(relevant_syllabus_chunks)} objectives...")
    syllabus_context = "\n\n".join([ f"Opt {i+1}: ID: {s.get('objective_id', 'N/A')} | Unit: {s.get('unit','?')} Mod: {s.get('module','?')} | Sim: {s.get('similarity', -1):.3f}\nText: {s.get('content', '')[:400]}..." for i, s in enumerate(relevant_syllabus_chunks) ])
    prompt = f"""Analyze the exam question chunk and select the single MOST relevant syllabus objective from the options provided.

**Exam Question Chunk:**
---
{past_paper_chunk_text}
---

**Potentially Relevant Syllabus Objective Options (ranked by similarity):**
---
{syllabus_context}
---

**Task:** Compare the exam chunk with each objective. Choose the ONE best match.

**Response Format:** Respond ONLY with JSON: {{"syllabus_section": "Exact text of best matching objective", "specific_objective_id": "ID of best matching objective or null"}}"""
    logging.debug(f"Sending request to LLM ({LLM_MODEL_NAME}) for mapping.")
    try:
        response = openai_client.chat.completions.create( model=LLM_MODEL_NAME, messages=[ {"role": "system", "content": "Map exam questions to the single most relevant syllabus objective provided."}, {"role": "user", "content": prompt} ], temperature=0.1, max_tokens=400, response_format={"type": "json_object"} )
        logging.debug(f"LLM map response received.")
        if not response.choices: logging.error("LLM map response missing choices."); return None, None
        llm_output_text = response.choices[0].message.content.strip(); logging.debug(f"LLM Raw Map Output: {llm_output_text}")
        try:
            parsed_json = json.loads(llm_output_text)
            syllabus_section = parsed_json.get("syllabus_section"); specific_objective_id = parsed_json.get("specific_objective_id")
            if isinstance(syllabus_section, str) and syllabus_section.strip(): logging.info(f"LLM mapped to Obj ID: '{specific_objective_id}' Text: '{syllabus_section[:80]}...'"); return syllabus_section.strip(), specific_objective_id
            else: logging.warning(f"LLM map JSON missing/empty 'syllabus_section'. Resp: {llm_output_text}"); return None, None
        except json.JSONDecodeError as json_e: logging.error(f"LLM map JSON parse failed: {json_e}. Resp: {llm_output_text}"); return None, None
        except Exception as parse_ex: logging.error(f"LLM map JSON processing error: {parse_ex}", exc_info=True); return None, None
    except Exception as e: logging.error(f"EXCEPTION during LLM mapping call: {e}", exc_info=True); return None, None

# --- NEW: Helper function for LLM Vision processing ---
def get_content_from_image_llm(image_base64: str, page_num: int, filename: str) -> Optional[str]:
    """Uses Vision LLM (GPT-4o/Turbo) to extract question content from an image."""
    logging.info(f"Sending image from page {page_num} of '{filename}' to Vision LLM ({LLM_MODEL_NAME})...")
    try:
        response = openai_client.chat.completions.create(
            model=LLM_MODEL_NAME, # Ensure this model supports vision
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"This image is page {page_num} from the exam paper '{filename}'. Extract ONLY the main exam question text visible. Ignore headers, footers, page numbers, margin notes, and instructions unless they are part of the question itself. If there are multiple distinct questions (e.g., 1a, 1b), extract all of them clearly separated. Describe any essential diagrams, figures, or code snippets briefly if they are part of the question."},
                        {
                            "type": "image_url",
                            "image_url": { "url": f"data:image/jpeg;base64,{image_base64}", "detail": "low" }, # Assuming JPEG, adjust if needed
                        },
                    ],
                }
            ],
            max_tokens=600
        )
        extracted_content = response.choices[0].message.content
        logging.info(f"Vision LLM extracted content page {page_num}: {extracted_content[:100]}...")
        return extracted_content.strip() if extracted_content else None
    except Exception as e:
        logging.error(f"EXCEPTION during Vision LLM call page {page_num}: {e}", exc_info=True)
        return None

# --- Helper function for JSON serialization ---
def default_serializer(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (datetime.date, datetime.datetime)): return obj.isoformat()
    try: return str(obj)
    except Exception: logging.error(f"Object type {type(obj)} not serializable"); return f"<unserializable: {type(obj).__name__}>"

# --- Storage Function (Integrates Vision LLM Call & uses json.dumps) ---
def store_chunks_in_supabase(chunks: List[Dict[str, Any]], metadata: Dict[str, Any]):
    """Processes chunks (using Vision LLM for image chunks) and stores results."""
    records_to_insert = []; processed_chunk_count = 0
    skipped_embedding_failure = 0; skipped_empty_content = 0; skipped_vision_failure = 0
    mapping_failures = 0; syllabus_query_skipped = 0
    total_initial_chunks = len(chunks)
    logging.info(f"Starting record preparation for '{metadata.get('filename', 'N/A')}' ({total_initial_chunks} chunks)...")

    for i, chunk in enumerate(chunks):
        chunk_number = i + 1; logging.debug(f"Prep record {chunk_number}/{total_initial_chunks} page {chunk['page']}...")
        processed_chunk_count += 1

        main_content_from_chunk = chunk.get("page_text_content", "").strip()
        combined_text = chunk.get("combined_text_for_embedding", "").strip()
        is_image_dominant = chunk.get("is_image_dominant", False)
        final_content_for_db = main_content_from_chunk
        text_to_embed = combined_text
        llm_generated_content = False # Initialize flag

        if is_image_dominant:
            logging.info(f"Chunk {chunk_number} (Page {chunk['page']}) image-dominant. Using Vision LLM.")
            if chunk.get("images"):
                img_b64 = chunk["images"][0].get("base64_data")
                if img_b64:
                    vision_content = get_content_from_image_llm(img_b64, chunk['page'], metadata.get('filename', 'N/A'))
                    if vision_content and "can’t assist" not in vision_content and "cannot provide" not in vision_content:
                        final_content_for_db = vision_content; text_to_embed = vision_content
                        llm_generated_content = True; logging.info(f"Using Vision LLM content chunk {chunk_number}.")
                    else:
                        logging.warning(f"Vision LLM failed/refused chunk {chunk_number}. Falling back."); skipped_vision_failure += 1
                        ocr_text = chunk["images"][0].get("ocr_text", "")
                        final_content_for_db = ocr_text if ocr_text else f"[Image Vision Fail p{chunk['page']}]"
                        text_to_embed = combined_text if combined_text else final_content_for_db
                        logging.debug(f"Fallback DB content: '{final_content_for_db[:50]}...', Embed text: '{text_to_embed[:50]}...'")
                else: logging.warning(f"No b64 data chunk {chunk_number}."); final_content_for_db = f"[Image Data Missing p{chunk['page']}]"; text_to_embed = final_content_for_db
            else: logging.warning(f"No image array chunk {chunk_number}."); final_content_for_db = f"[Image List Empty p{chunk['page']}]"; text_to_embed = final_content_for_db
        else:
             final_content_for_db = main_content_from_chunk; text_to_embed = combined_text
             if not text_to_embed and final_content_for_db: text_to_embed = final_content_for_db

        if not text_to_embed: logging.warning(f"Skip chunk {chunk_number}: empty text for embedding."); skipped_empty_content += 1; continue

        past_paper_embedding = generate_embedding(text_to_embed)
        if past_paper_embedding is None: logging.warning(f"Skip chunk {chunk_number}: Embed fail."); skipped_embedding_failure += 1; continue

        # Determine content for mapping step
        main_content_for_llm = final_content_for_db if llm_generated_content else main_content_from_chunk
        if not main_content_for_llm: main_content_for_llm = text_to_embed
        logging.debug(f"Using for LLM mapping (chunk {chunk_number}): '{main_content_for_llm[:100]}...'")

        subject = metadata.get("subject"); relevant_syllabus_objectives = []
        if not subject:
             logging.warning(f"Chunk {chunk_number}: Skip syllabus query (No Subject)."); syllabus_query_skipped += 1
             final_syllabus_section = "Mapping Skipped - No Subject"; final_objective_id = None
        else:
             relevant_syllabus_objectives = query_syllabus_db(past_paper_embedding, subject)
             syllabus_objective_text, objective_id_from_llm = get_mapping_from_llm(main_content_for_llm, relevant_syllabus_objectives)
             if syllabus_objective_text: final_syllabus_section = syllabus_objective_text; final_objective_id = objective_id_from_llm
             else:
                 if relevant_syllabus_objectives: logging.warning(f"Chunk {chunk_number}: LLM mapping failed."); mapping_failures += 1
                 else: logging.info(f"Chunk {chunk_number}: LLM map skipped (No objectives).")
                 final_syllabus_section = "Mapping Failed"; final_objective_id = None

        q_num_match = re.match(r'^\s*(\d+[a-z]*\.?|\([a-z]+\))\s+', final_content_for_db, re.IGNORECASE)
        question_number = q_num_match.group(1).strip('.() ') if q_num_match else f"Chunk_{chunk_number}"

        # --- Use json.dumps() with default handler for JSONB columns ---
        images_json_str = None; equations_json_str = None
        try: images_json_str = json.dumps(chunk.get("images"), default=default_serializer) if chunk.get("images") else None
        except Exception as json_e: logging.error(f"JSON serialize error (images) chunk {chunk_number}: {json_e}", exc_info=True); images_json_str = None
        try: equations_json_str = json.dumps(chunk.get("equations"), default=default_serializer) if chunk.get("equations") else None
        except Exception as json_e: logging.error(f"JSON serialize error (equations) chunk {chunk_number}: {json_e}", exc_info=True); equations_json_str = None
        # --- End Change ---

        record = {
            "content": final_content_for_db, # Store meaningful content
            "embedding": past_paper_embedding,
            "source": metadata.get("source", "CAPE"), "year": metadata.get("year"),
            "paper": metadata.get("paper"), "question_number": question_number,
            "subject": subject, "topic": metadata.get("topic"), "sub_topic": metadata.get("sub_topic"),
            "syllabus_section": final_syllabus_section, "specific_objective_id": final_objective_id,
            "images": images_json_str, # Use JSON string
            "equations": equations_json_str # Use JSON string
        }
        records_to_insert.append(record)
        logging.debug(f"Record {chunk_number} prepared.")

    logging.info(f"--- Prep Summary for {metadata.get('filename', 'N/A')} ---")
    logging.info(f"Chunks processed: {processed_chunk_count}, Skipped (Empty): {skipped_empty_content}, Skipped (Embed Fail): {skipped_embedding_failure}, Skipped (Vision Fail): {skipped_vision_failure}")
    logging.info(f"Syllabus Query Skipped: {syllabus_query_skipped}, LLM Map Fails: {mapping_failures}, Records to Insert: {len(records_to_insert)}")
    if not records_to_insert: logging.warning(f"No records to insert for {metadata.get('filename', 'N/A')}."); return

    batch_size = 50; total_inserted = 0
    logging.info(f"Attempting insert {len(records_to_insert)} records into 'exam_questions'...")
    for i in range(0, len(records_to_insert), batch_size):
        batch = records_to_insert[i:i + batch_size]; current_batch_num = i // batch_size + 1
        total_batches = (len(records_to_insert) + batch_size - 1) // batch_size
        logging.info(f"Prep Batch {current_batch_num}/{total_batches} ({len(batch)})...")
        try:
            logging.info(f"Attempting Supabase insert batch {current_batch_num}...")
            response = supabase.table("exam_questions").insert(batch).execute(); logging.info(f"Insert call executed batch {current_batch_num}."); logging.debug(f"Response batch {current_batch_num}: {response}")
            has_error = hasattr(response, 'error') and response.error is not None; has_data = hasattr(response, 'data') and response.data is not None
            status = getattr(response, 'status_code', None)
            if has_error: logging.error(f"SUPABASE ERROR batch {current_batch_num}: {response.error}")
            elif status is not None and 200 <= status < 300: logging.info(f"Success batch {current_batch_num} (Status: {status})."); total_inserted += len(batch)
            elif has_data: logging.info(f"Success batch {current_batch_num} (Data returned)."); total_inserted += len(batch)
            else: logging.warning(f"Insert batch {current_batch_num} likely failed (Status: {status}). Resp: {response}")
        except Exception as e:
            logging.error(f"EXCEPTION during Supabase insert batch {current_batch_num}: {e}", exc_info=True)
            if batch:
                try: first_record_preview = {k: v for k, v in batch[0].items() if k not in ['embedding', 'images', 'equations']}; logging.error(f"Failed batch {current_batch_num} preview: {json.dumps(first_record_preview, indent=2, default=str)}")
                except Exception as log_ex: logging.error(f"Could not serialize failed batch preview: {log_ex}")
            # break # Optional: Stop on first batch error
    logging.info(f"--- Insert Summary for {metadata.get('filename', 'N/A')} ---")
    logging.info(f"Records prepared: {len(records_to_insert)} | Records inserted: {total_inserted}")

# --- Main Processing Function ---
def process_past_papers(pdf_path: str, metadata: Dict[str, Any]):
    filename = os.path.basename(pdf_path); metadata['filename'] = filename
    logging.info(f"--- Start processing: {filename} ---"); logging.info(f"Metadata: {metadata}")
    if not metadata.get("subject"): logging.error(f"No subject for {filename}. Skip."); return False
    extracted_data = extract_text_images_equations_from_pdf(pdf_path)
    if not extracted_data: logging.warning(f"No data successfully extracted from {filename}."); return True
    chunks = chunk_extracted_data(extracted_data)
    if not chunks: logging.warning(f"No chunks created from {filename}."); return True
    store_chunks_in_supabase(chunks, metadata)
    logging.info(f"--- Finish processing: {filename} ---"); return True

# --- Main Execution Block ---
if __name__ == "__main__":
    base_past_paper_dir = "/Users/anthonyzamore/Projects/CAPE-GPT/Past_Papers"
    logging.info(f"Starting past paper processing walk: {base_past_paper_dir}")
    if not os.path.isdir(base_past_paper_dir): logging.error(f"Base directory not found: {base_past_paper_dir}"); exit(1)

    total_files_processed = 0; total_files_skipped = 0

    for dirpath, dirnames, filenames in os.walk(base_past_paper_dir):
        logging.debug(f"Entering directory: {dirpath}")
        relative_path = os.path.relpath(dirpath, base_past_paper_dir)
        path_parts = relative_path.split(os.sep)

        standardized_subject = None; current_unit = None
        if len(path_parts) > 0 and path_parts[0] != '.':
            raw_subject_dir = path_parts[0]; logging.debug(f"Raw dir part: '{raw_subject_dir}'")
            unit_match = re.search(r'Unit\s*(\d+)', raw_subject_dir, re.IGNORECASE)
            if unit_match:
                current_unit = f"Unit {unit_match.group(1)}"
                subject_base = re.sub(r'\s*Unit\s*\d+', '', raw_subject_dir, flags=re.IGNORECASE).strip()
            else: subject_base = raw_subject_dir.strip()
            standardized_subject = subject_base.replace('_', ' ')
            subject_mapping = {"Physical Education & Sport": "Physical Education and Sport"} # Customize!
            standardized_subject = subject_mapping.get(standardized_subject, standardized_subject)
            logging.info(f"Derived Subject: '{standardized_subject}', Unit: '{current_unit}' from path '{relative_path}'")

        for filename in filenames:
            if filename.lower().endswith(".pdf") and not filename.startswith('.') and not filename.startswith('~'):
                filepath = os.path.join(dirpath, filename)
                logging.info(f"Found PDF: {filepath}.")
                if not standardized_subject: logging.warning(f"Skipping '{filepath}', subject undetermined."); total_files_skipped += 1; continue

                file_metadata = { "source": "CAPE", "subject": standardized_subject, "unit": current_unit }
                try:
                    base_name = filename.replace(".pdf", ""); year = None; paper = None
                    year_match = re.search(r'\b(19[89]\d|20\d{2})\b', base_name)
                    if year_match: year = int(year_match.group(1)); file_metadata["year"] = year; logging.debug(f"Year: {year}")
                    else: logging.warning(f"No YEAR in '{filename}'.")
                    paper_match = re.search(r'\b[Pp](?:aper)?[\s_]*(\d+)\b', base_name)
                    if paper_match: paper_num = int(paper_match.group(1)); paper = f"Paper {paper_num:02}"; file_metadata["paper"] = paper; logging.debug(f"Paper: {paper}")
                    else: logging.warning(f"No PAPER in '{filename}'.")

                    if process_past_papers(filepath, file_metadata): total_files_processed += 1
                    else: total_files_skipped += 1
                except Exception as e: logging.error(f"Error parsing/processing file '{filepath}': {e}", exc_info=True); total_files_skipped += 1

    logging.info(f"--- Overall Summary ---")
    logging.info(f"Files Processed: {total_files_processed}")
    logging.info(f"Files Skipped: {total_files_skipped}")
    logging.info(f"--- Script Complete ---")
