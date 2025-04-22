# ExamSage: Intelligent Exam Preparation Assistant

ExamSage is a sophisticated application designed to help students prepare for STEM exams. Users can upload images or snippets of exam questions and receive detailed feedback, step-by-step solution approaches, relevant syllabus objectives, common pitfalls, and analysis of similar past questions, all powered by AI and a vector database of historical exam data.

![ExamSage Concept](placeholder.png) // Optional: Add a screenshot or diagram later

## Core Features

1.  **Question Analysis via Image Upload:**
    *   Upload a screenshot or photo of an exam question.
    *   AI-powered OCR and Vision models extract the question text and context.

2.  **Similar Question Retrieval:**
    *   Leverages a vector database (Supabase with pgvector) populated with past exam papers.
    *   Identifies and displays questions from past papers that are semantically similar to the user's uploaded question.

3.  **Syllabus Mapping:**
    *   Maps the user's question and similar past questions to specific learning objectives from the official syllabus (stored in a separate vector database).
    *   Helps students understand the exact curriculum point being tested.

4.  **AI-Generated Guidance:**
    *   Provides a detailed, step-by-step approach to solving the user's question.
    *   Lists key concepts, formulas, and definitions required.
    *   Warns about common pitfalls, calculation errors, and reasoning mistakes associated with the question type or topic.

5.  **Frequency Analysis & Visualization:**
    *   Analyzes how frequently similar questions have appeared in past papers over recent years.
    *   Visualizes this trend using a bar chart to help students gauge topic recurrence.
    *   Provides a summary statement about historical frequency.

6.  **STEM Subject Optimized:**
    *   Designed primarily for Science, Technology, Engineering, and Mathematics (STEM) subjects.
    *   Includes basic equation parsing (LaTeX detection via regex, semantic parsing via SymPy during data ingestion).
    *   Handles diagrams via image extraction and Vision LLM analysis (during backend processing).

## Application Architecture

ExamSage uses a modern web architecture with distinct components:

*   **Frontend:** A React-based chatbot interface for user interaction, image uploads, and displaying results including visualizations.
*   **Backend:** A FastAPI (Python) application providing the core API endpoints. It orchestrates the analysis process by interacting with various services.
*   **Database:** Supabase (PostgreSQL) utilizing the `pgvector` extension for efficient similarity searches on embedded text data. Contains two main tables: `syllabus_sections` and `exam_questions`.
*   **AI Models:**
    *   **Embeddings:** OpenAI `text-embedding-ada-002` (or newer) for converting text to vectors.
    *   **LLM (Text/Mapping):** OpenAI `gpt-4o` (or equivalent) for generating analysis, solution approaches, pitfalls, and mapping questions to syllabus objectives based on retrieved context.
    *   **LLM (Vision):** OpenAI `gpt-4o` (or equivalent) for extracting question text and context directly from images when OCR is insufficient.
*   **Data Processing Scripts:** Offline Python scripts used to parse syllabus PDFs and past paper PDFs, extract content, generate embeddings, perform initial mapping, and populate the Supabase database.

**High-Level Data Flow (User Interaction):**

1.  User uploads question image via React Frontend.
2.  Frontend sends image to FastAPI Backend (`/api/analyze_question`).
3.  Backend `analysis_service`:
    *   Uses `ocr_service` or `llm_service` (vision) to get text.
    *   Uses `embedding_service` to vectorize the text.
    *   Uses `supabase_service` to query `syllabus_sections` (using `match_syllabus_sections` RPC) for relevant objectives.
    *   Uses `supabase_service` (potentially another function needed) to query `exam_questions` for similar past questions and frequency data (or calculates frequency within the service).
    *   Uses `llm_service` to generate detailed analysis, approach, pitfalls, and confirm the best syllabus mapping based on user question, similar objectives, and frequency data.
    *   Constructs the `AnalyzeResponse`.
4.  Backend returns the structured JSON response to the Frontend.
5.  Frontend parses the response and displays the information in the chat interface, including rendering the frequency chart.

## Technical Stack

*   **Frontend:**
    *   Framework: React.js (with TypeScript)
    *   UI Library: Material UI (MUI)
    *   State Management: React Hooks (`useState`, `useCallback`, etc.) / Zustand (Optional)
    *   Charting: Recharts
    *   API Client: Axios
    *   Build Tool: ViteOkay
*   **Backend:**
    *   Framework: FastAPI (Python)
    *   Server: Uvicorn
    *   Data, let's create a comprehensive `README.md` and an `install.sh` script.

**1. File Locations:**

*   **`README.md`:** Should be placed in the **root directory** of your project Validation: Pydantic
    *   API Interaction: OpenAI Python Library (`openai`)
    *   Database Interaction (`examsage/`). This is the standard location where platforms like GitHub expect to find it.
*   **`install.sh`: Supabase Python Library (`supabase-py`)
    *   OCR: Tesseract (`pytesseract`, requires:** Can also be placed in the **root directory** (`examsage/`) for ease of use. Alternatively, you separate Tesseract install)
*   **Database:** Supabase (PostgreSQL + pgvector)
*   **AI could have separate install instructions within the `backend/` and `frontend/` directories if you prefer users to install dependencies individually Models:** OpenAI API (Embeddings, GPT-4o / GPT-4 Turbo)
*   **Data Processing:**. Placing it in the root is convenient for a single setup command.

**2. `README.md` Content Python (`PyMuPDF`, `Pillow`, `sympy`, `openai`, `supabase-py`)

## Database:**

```markdown
# ExamSage: Intelligent Exam Preparation Assistant

ExamSage is a sophisticated application designed to assist students in their Schema

*(Refer to `database/schema.sql` for full details)*

*   **`syllabus_sections`:** exam preparation, particularly for STEM subjects. By uploading snippets (images) of exam questions, students receive detailed feedback, potential solution approaches, common pitfall warnings, and insights into how similar questions have appeared in past papers, all powered by AI Stores chunked syllabus objectives with metadata and vector embeddings.
    *   Key columns: `id`, `subject`, `unit`, `module`, `objective_id`, `content` (objective text), `embedding`, `source_document`, and a vector database of syllabus objectives and past questions.

## Core Features

*   **Question Image Upload:** Simple `page_number`.
*   **`exam_questions`:** Stores chunked past paper questions with metadata, vector embeddings, image/equation data, and mappings to syllabus objectives.
    *   Key columns: `id`, `content` interface to upload screenshots or photos of exam questions.
*   **Content Extraction:** Utilizes OCR and potentially Vision AI (question text/OCR/Vision output), `embedding`, `source`, `year`, `paper`, `question_number`, `subject`, `syllabus_section` (mapped objective text), `specific_objective_id`, `images` (jsonb), `equations` (jsonb).

## Setup and Installation

**Prerequisites:**

*   Python (3.9+) and (GPT-4o) to accurately extract text and understand diagrams from uploaded images.
*   **Similarity Analysis:** Embeds the extracted question content and performs a vector similarity search against a database of past exam questions (chunked by page/question).
*   **Syllabus Mapping:** Identifies the most relevant syllabus objective(s) for the submitted Pip
*   Node.js (LTS version recommended) and npm (or yarn)
*   Tesseract OCR Engine question based on semantic similarity and LLM reasoning, referencing a pre-processed database of syllabus objectives.
*   **Frequency Insights: Must be installed separately on the system running the backend/scripts. Follow Tesseract installation guides for your OS (mac:** Analyzes the frequency and distribution (by year, paper type) of similar questions found in the past papers database.
*   **AIOS: `brew install tesseract`, Ubuntu: `sudo apt-get install tesseract-ocr`). Ensure it's in the system PATH or set `TESSERACT_CMD` in `.env`.
*   Supabase Account & Project:-Generated Guidance:** Leverages a Large Language Model (GPT-4o) to synthesize information and provide:
    *   **Detailed Solution Create a project on [Supabase](https://supabase.com/).
*   OpenAI Account & API Key: Obtain an API key from Approach:** Step-by-step methodology tailored to the question type and relevant objective.
    *   **Key Concepts:** Highlights [OpenAI](https://platform.openai.com/). Ensure you have access to the required models (embedding, GPT-4o/Turbo essential topics, formulas, and definitions.
    *   **Common Pitfalls:** Warns about frequent mistakes and misunderstand) and sufficient credits/billing setup.

**Installation Steps:**

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd examsage
    ```

2.  **Set up Environment Variables:**
    ings related to the question's topic.
*   **Interactive Chat Interface:** Presents the analysis and guidance in a user-friendly chatbot*   Navigate to the `backend/` directory. Copy `.env.example` (if provided) to `.env` format.
*   **Visualizations:** Includes charts (e.g., bar chart showing frequency over years) to help or create a new `.env` file.
    *   Fill in your specific values:
        ```dotenv
        SUPABASE_URL=your_supabase_project_url
        SUPABASE_KEY=your_supabase_service_role_key
        OPENAI students understand trends.
*   **STEM Subject Optimized:** Includes specific handling for mathematical equations (parsing with SymPy during_API_KEY=your_openai_sk_key
        OPENAI_LLM_MODEL=gpt-4o # Or data ingestion) and diagram understanding (using Vision AI).

## Technical Architecture

*   **Frontend:** React.js (TypeScript gpt-4-turbo if preferred/available
        # Optional: TESSERACT_CMD=/path/to/your/tesseract/executable
        ```
    *   Create a similar `.env` file in the `scripts) with Material UI (MUI) for components and Recharts for visualizations. Built using Vite.
*   **Backend:**/` directory (or copy/symlink from `backend/.env`).

3.  **Run Installation Script:**
    *   Make Python with FastAPI, providing a RESTful API for the frontend.
*   **Database:** Supabase (PostgreSQL) the script executable: `chmod +x install.sh`
    *   Run the script from the project root directory with the `pgvector` extension for efficient vector similarity search.
*   **AI Models:**
    *   **Embed: `./install.sh`
    *   This script will install Python dependencies for the backend and scripts, and Node.js dependencies for the frontend.

4.  **Database Setup:**
    *   Go to your Supabase projectdings:** OpenAI `text-embedding-ada-002` (or newer) for creating vector representations of text dashboard.
    *   Navigate to the "SQL Editor".
    *   Run the contents of `database/schema.sql`.
    *   **Core LLM:** OpenAI `gpt-4o` (or `gpt-4-turbo`) for vision to create the tables and indexes. Make sure the `vector` extension is enabled (check under Database -> Extensions).
    *    processing (image understanding), analysis generation, and syllabus mapping confirmation.
*   **Data Processing (Offline Scripts):** PythonRun the contents of `database/functions.sql` to create the `match_syllabus_sections` function.

 scripts using libraries like `PyMuPDF` (fitz), `pytesseract`, `Pillow`, `openai`, `supabase5.  **Data Ingestion (Populate Database):**
    *   **Important:** Customize the `standardize_subject_name` function in `scripts/extract_syllabus_objectives.py` and the `subject`, and `sympy` to process syllabus PDFs and past paper PDFs into structured data for the database.

### Data Flow_mapping` logic in `scripts/CAPE_GPT_CHUNKING.py` to match your syllabus filenames and past (User Query)

1.  **Upload:** User uploads question image via React frontend.
2.  **API paper directory structure.
    *   Navigate to the `scripts/` directory: `cd scripts`
    *   Run the syllabus Call:** Frontend sends image to FastAPI `/api/analyze_question` endpoint.
3.  **Backend Processing ( processing script: `python extract_syllabus_objectives.py`
    *   Verify the `syllabus_sections` table is populated in Supabase.
    *   Run the past paper processing script: `python CAPE_GPT_CHUNKING.py`
    *   Verify the `exam_questions` table is populated in Supabase.
    *   `FastAPI `analysis_service`):**
    *   Extracts text/content from image (OCR or Vision LLM).
    cd ..` to return to the project root.

## Running the Application

1.  **Start the Backend Server:**
    ```bash
    cd backend
    uvicorn app.main:app --reload --host 0.0.0*   Generates embedding for the extracted text (`embedding_service`).
    *   Determines subject (currently basic.0 --port 8000
    # Use --host 0.0.0.0 if/hint-based - TODO: Improve).
    *   Queries Supabase `syllabus_sections` using vector search running in Docker or need access from other devices
    ```
    The API should be available at `http://localhost:80 (`match_syllabus_sections` RPC function) for relevant objectives (`supabase_service`).
    *   (Future00`. Check `http://localhost:8000/docs` for documentation.

2.  **Start the enhancement: Query `exam_questions` table for similar past *questions* and calculate frequency).
    *   Sends Frontend Development Server:**
    *   Open a **new terminal** in the project root.
    ```bash
    cd frontend
    npm extracted text, relevant objectives, and frequency data to `llm_service`.
    *   Generates detailed analysis, approach, pitfalls run dev
    ```
    The frontend should be available at `http://localhost:5173` (or, and confirms best syllabus mapping using GPT-4o (`llm_service`).
    *   Constructs ` another port Vite indicates).

3.  **Access the Application:** Open your web browser and navigate to the frontend URLAnalyzeResponse` Pydantic model.
4.  **Response:** FastAPI sends structured JSON response back to the frontend.
 (e.g., `http://localhost:5173`).

## Future Enhancements

*   User5.  **Display:** React frontend parses the response and displays the formatted information, including text analysis and charts, in the chat interface.

## Project authentication and personalized history.
*   More sophisticated chunking strategies for past papers.
*   Fine-tuning embedding Structure
# CAPE-GPT
