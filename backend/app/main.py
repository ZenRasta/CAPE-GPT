# main.py
# backend/app/main.py

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import the API router from routes.py
from app.api import routes
# Import config to ensure environment variables are loaded on startup
from app.core import config # Although config might not be directly used here, importing ensures it runs its checks

# --- Basic Logging Setup ---
# You might want a more sophisticated logging setup for production
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- Create FastAPI App Instance ---
# Add metadata for API docs
app = FastAPI(
    title="ExamSage API",
    description="API for the ExamSage Intelligent Exam Preparation Assistant. Allows analyzing exam question images.",
    version="0.1.0",
    # Optionally add contact info, license info etc.
    # contact={
    #     "name": "Your Name/Team",
    #     "url": "http://yourwebsite.com",
    #     "email": "your@email.com",
    # },
)

# --- CORS (Cross-Origin Resource Sharing) Middleware ---
# Define the list of origins (frontend URLs) allowed to communicate with this API
# Use "*" for development ONLY if necessary, be specific in production
origins = [
    "http://localhost:5173",    # Default Vite dev port for React/Vue etc.
    "http://localhost:3000",    # Common React dev port (Create React App)
    # Add the URL of your deployed frontend when available
    # "https://your-deployed-frontend.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # List of allowed origins
    allow_credentials=True,      # Allow cookies if needed later
    allow_methods=["*"],         # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],         # Allow all headers
)

# --- Include API Routers ---
# Include the router defined in api/routes.py
# All routes defined in routes.router will be available under the /api prefix
app.include_router(routes.router, prefix="/api")
log.info("API router included under /api prefix.")

# --- Root Endpoint (Optional Health Check/Welcome Message) ---
@app.get("/", tags=["Root"])
async def read_root():
    """Simple endpoint to check if the API is running."""
    log.info("Root endpoint '/' accessed.")
    return {"message": "Welcome to the ExamSage API! Visit /docs for documentation."}

# --- Optional: Add Startup/Shutdown Events ---
# @app.on_event("startup")
# async def startup_event():
#     log.info("Application startup: Initializing resources...")
#     # Example: Initialize database connections pools if needed
#
# @app.on_event("shutdown")
# def shutdown_event():
#     log.info("Application shutdown: Cleaning up resources...")
    # Example: Close database connections
    