# backend/app/services/supabase_service.py

import logging
from typing import List, Dict, Any, Optional
import json
from supabase import create_client, Client
from app.core.config import SUPABASE_URL, SUPABASE_KEY

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')
log = logging.getLogger(__name__)

# Initialize Supabase client
supabase_client: Optional[Client] = None

try:
    if SUPABASE_URL and SUPABASE_KEY:
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        log.info("Supabase client initialized successfully.")
    else:
        log.warning("Missing Supabase credentials - Some functionality will be limited.")
except Exception as e:
    log.error(f"Failed to initialize Supabase client: {e}", exc_info=True)

async def find_similar_objectives(embedding: List[float], subject: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Find syllabus objectives or past questions similar to the provided embedding.
    
    Args:
        embedding: The vector embedding to match against
        subject: Optional subject to filter by
        limit: Maximum number of results to return
        
    Returns:
        List of dictionaries containing similar items with their metadata
    """
    if not supabase_client:
        log.warning("Supabase client not initialized. Returning empty results.")
        return []
    
    if not embedding:
        log.warning("Empty embedding provided. Cannot search for similar content.")
        return []
    
    try:
        log.info(f"Searching for similar content in Supabase with {subject or 'no'} subject filter.")
        
        # For demo purposes, we'll return mock data instead of actually querying Supabase
        # In a real implementation, we would use Vector search like:
        # rpc_function = "match_questions" if past_questions else "match_syllabus_objectives"
        # query = supabase_client.rpc(
        #    rpc_function,
        #    {
        #        "query_embedding": embedding,
        #        "match_threshold": 0.6,
        #        "match_count": limit,
        #        "subject_filter": subject
        #    }
        # )
        # results = query.execute()
        
        # For demo, return mock data:
        results = [
            {
                "id": 1,
                "content_snippet": "Construct the truth table for the circuit shown in Figure 1.",
                "year": 2022,
                "paper": "Paper 1",
                "question_number": "3(a)",
                "similarity_score": 0.87,
                "syllabus_section": "Digital Logic",
                "specific_objective_id": "CS-U1-M4-SO2"
            },
            {
                "id": 2,
                "content_snippet": "Draw a truth table to show the operation of the AND gate followed by a NOT gate.",
                "year": 2021,
                "paper": "Paper 2",
                "question_number": "1(b)",
                "similarity_score": 0.75,
                "syllabus_section": "Digital Logic",
                "specific_objective_id": "CS-U1-M4-SO1"
            },
            {
                "id": 3,
                "content_snippet": "Complete the truth table for the logic circuit shown in Figure 2.",
                "year": 2020,
                "paper": "Paper 1",
                "question_number": "4(a)",
                "similarity_score": 0.65,
                "syllabus_section": "Boolean Algebra",
                "specific_objective_id": "CS-U1-M4-SO3"
            }
        ]
        
        log.info(f"Found {len(results)} similar items.")
        return results
    
    except Exception as e:
        log.error(f"Error searching for similar content: {e}", exc_info=True)
        return []  # Return empty list on error
