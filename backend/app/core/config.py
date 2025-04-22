# config.py
# backend/app/core/config.py

import os
import logging
from dotenv import load_dotenv

# Determine the path to the .env file (assuming it's in the backend/ directory)
# If your .env is in the project root, you might need to adjust the path:
# load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '..', '.env'))
env_path = os.path.join(os.path.dirname(__file__), '..', '.env') # Path relative to this config.py file
load_dotenv(dotenv_path=env_path)

log = logging.getLogger(__name__)

# --- Database Configuration ---
SUPABASE_URL: str | None = os.getenv("SUPABASE_URL", "https://example.supabase.co")  # Added default for demo
SUPABASE_KEY: str | None = os.getenv("SUPABASE_KEY", "demo_key")  # Added default for demo

# --- OpenAI Configuration ---
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY", "demo_key")  # Added default for demo
LLM_MODEL: str = os.getenv("OPENAI_LLM_MODEL", "gpt-4o") # Default to gpt-4o
EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")

# --- Optional: Tesseract Configuration ---
# If Tesseract isn't in your system PATH, uncomment and set the path in .env
TESSERACT_CMD: str | None = os.getenv("TESSERACT_CMD")

# --- DEMO MODE ---
# For demonstration purposes, we'll recognize when using placeholder values
DEMO_MODE = (
    SUPABASE_URL == "https://example.supabase.co" or 
    SUPABASE_KEY == "demo_key" or 
    OPENAI_API_KEY == "demo_key"
)

# --- Validation ---
# For demo purposes, we'll log warnings instead of raising errors
missing_vars = []
if not SUPABASE_URL or SUPABASE_URL == "https://example.supabase.co": 
    missing_vars.append("SUPABASE_URL")
if not SUPABASE_KEY or SUPABASE_KEY == "demo_key": 
    missing_vars.append("SUPABASE_KEY")
if not OPENAI_API_KEY or OPENAI_API_KEY == "demo_key": 
    missing_vars.append("OPENAI_API_KEY")

if missing_vars:
    if DEMO_MODE:
        log.warning(f"Running in DEMO MODE with placeholder values for: {', '.join(missing_vars)}")
        log.warning("Some functionality will be limited to static demo data")
    else:
        error_message = f"Missing critical environment variables: {', '.join(missing_vars)}"
        log.critical(error_message)
        # In non-demo mode, we'd raise an error
        # raise ValueError(error_message)
else:
    log.info("Core configuration loaded successfully with all required variables.")

# You can add other configuration variables here as needed
# Example:
# MAX_SIMILAR_QUESTIONS = 5
# SIMILARITY_THRESHOLD = 0.75
