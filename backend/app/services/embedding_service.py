# embedding_service.py

'''
embedding_service.py (Implemented in previous prompt)

Purpose: Handles generating vector embeddings from text using the chosen OpenAI model (e.g., text-embedding-ada-002).

Input: Text string.

Output: A list of floats representing the vector embedding, or None on failure.

Responsibilities: Cleaning input text, calling the OpenAI Embeddings API, handling API errors (rate limits, authentication, etc.), potentially validating embedding dimensions.git 
'''


# backend/app/services/embedding_service.py

import logging
from typing import Optional, List

from openai import OpenAI, APIError, RateLimitError # Import specific errors
from app.core.config import OPENAI_API_KEY, EMBEDDING_MODEL # Use config

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')
log = logging.getLogger(__name__)

# Initialize OpenAI client here or ideally inject/share a single instance
# For simplicity now, initialize here
try:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    log.info("OpenAI client initialized for embedding service.")
except Exception as e:
    log.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
    openai_client = None # Ensure client is None if init fails

async def generate_text_embedding(text: str) -> Optional[List[float]]:
    """
    Generates sentence embedding for the given text using OpenAI API.
    Uses async approach suitable for FastAPI.

    Args:
        text: The text content to embed.

    Returns:
        A list of floats representing the embedding, or None if generation fails.
    """
    if not openai_client:
        log.error("OpenAI client not initialized. Cannot generate embedding.")
        return None

    if not text or not text.strip():
        log.warning("Attempted to generate embedding for empty/whitespace text.")
        return None

    try:
        text_to_embed = text.replace("\n", " ").strip()
        if not text_to_embed:
             log.warning("Text became empty after cleaning, skipping embedding.")
             return None

        log.debug(f"Requesting embedding for text (first 100 chars): {text_to_embed[:100]}...")

        # Assuming the openai client library handles async calls appropriately
        # If using an older version or direct http, ensure async handling
        response = await openai_client.embeddings.create(
            input=[text_to_embed],
            model=EMBEDDING_MODEL
            # dimensions=EXPECTED_EMBEDDING_DIM # Specify if using v3 models and need specific dim
            )

        log.debug(f"Received embedding response.")

        if not response.data or not response.data[0].embedding:
            log.error(f"OpenAI response data or embedding is missing! Response: {response}")
            return None

        embedding = response.data[0].embedding
        # Optional dimension check
        # if len(embedding) != EXPECTED_EMBEDDING_DIM:
        #      log.warning(f"Embedding dimension mismatch. Expected {EXPECTED_EMBEDDING_DIM}, got {len(embedding)}.")

        log.info(f"Embedding generated successfully (dim: {len(embedding)}).")
        return embedding

    except APIError as api_err:
         log.error(f"OpenAI API Error during embedding: {api_err}", exc_info=True)
         return None
    except RateLimitError as rate_err:
        log.error(f"OpenAI Rate Limit Error during embedding: {rate_err}", exc_info=True)
        return None
    except Exception as e:
        log.error(f"Unexpected EXCEPTION during OpenAI embedding generation: {e}", exc_info=True)
        return None