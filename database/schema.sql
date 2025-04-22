# schema.sql
-- Enable the pgvector extension (if not already enabled)
-- Run this once per database:
-- create extension if not exists vector;

-- Table structure for storing processed past exam questions
create table if not exists exam_questions (
    id bigserial primary key,
    content text,                     -- Text of the question chunk
    embedding vector(1536),           -- Vector embedding (using text-embedding-ada-002 dimension)
    source text,                      -- e.g., "CAPE"
    year integer,                     -- e.g., 2023
    paper text,                       -- e.g., "Paper 01", "Paper 02"
    question_number text,             -- e.g., "1", "2a", "Chunk_5" (extracted or generated)
    subject text,                     -- e.g., "Biology", "Chemistry", "Physics", "Computer Science"
    topic text,                       -- Placeholder: e.g., "Cell Structure", "Organic Chemistry" (Can be populated later)
    sub_topic text,                   -- Placeholder: e.g., "Mitochondria", "Alkanes" (Can be populated later)
    syllabus_section text,            -- Mapped section/objective text from LLM or "Mapping Failed"
    specific_objective_id text,       -- Mapped objective ID from LLM (can be null)
    images jsonb,                     -- Stores array of image data as a single JSONB object [{base64_data: "...", ocr_text: "..."}, ...]
    equations jsonb                   -- Stores array of equation data as a single JSONB object [{latex: "...", text: "..."}, ...]
);

-- Note: The initial create table statement above includes the `specific_objective_id` column.
-- If your table was created *without* it based on the original README,
-- you would need to run the ALTER TABLE command below. Otherwise, it's not needed if running the CREATE TABLE above.
-- Run this *only* if the 'exam_questions' table exists and is missing the 'specific_objective_id' column:
/*
ALTER TABLE exam_questions
ADD COLUMN IF NOT EXISTS specific_objective_id text DEFAULT NULL;
*/

-- Optional: Add indexes for faster metadata filtering (Recommended)
CREATE INDEX IF NOT EXISTS idx_exam_subject ON exam_questions (subject);
CREATE INDEX IF NOT EXISTS idx_exam_year ON exam_questions (year);
CREATE INDEX IF NOT EXISTS idx_exam_paper ON exam_questions (paper);
CREATE INDEX IF NOT EXISTS idx_exam_topic ON exam_questions (topic); -- If used for filtering later
CREATE INDEX IF NOT EXISTS idx_exam_syllabus_section ON exam_questions (syllabus_section); -- If filtering by mapped section

-- Optional: Add an index for vector similarity search (Highly Recommended for performance)
-- Choose ONE of the following based on your data size and query patterns. HNSW is often preferred.
-- Option A: IVFFlat
-- CREATE INDEX ON exam_questions USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100); -- Adjust lists count

-- Option B: HNSW
-- CREATE INDEX ON exam_questions USING hnsw (embedding vector_cosine_ops); -- WITH (m = 16, ef_construction = 64);

-- Enable the pgvector extension (if not already enabled)
-- Run this once per database:
-- create extension if not exists vector;

-- Table structure for storing syllabus sections/objectives
create table if not exists syllabus_sections (
    id bigserial primary key,
    subject text not null,              -- e.g., "Applied Mathematics", "Computer Science"
    unit text,                          -- e.g., "Unit 1", "Unit 2"
    module text,                        -- e.g., "Module 1", "Module 2: Problem-Solving"
    section_title text,                 -- e.g., "Collecting and Describing Data", "Specific Objectives"
    objective_id text,                  -- e.g., "1.a.i", "Specific Objective 3", "SO-5.3" (Can be null if chunk is broader)
    content text not null,              -- The actual text chunk from the syllabus
    embedding vector(1536) not null,    -- Vector embedding (using text-embedding-ada-002 dimension)
    source_document text,               -- Filename of the syllabus PDF, e.g., "CAPE Applied Mathematics Syllabus.pdf"
    page_number integer                 -- Page number in the source PDF where the chunk starts/ends (based on script logic)
);

-- Optional: Add indexes for faster metadata filtering (Recommended)
-- These help speed up queries that filter by subject, unit, or module before vector search.
CREATE INDEX IF NOT EXISTS idx_syllabus_subject ON syllabus_sections (subject);
CREATE INDEX IF NOT EXISTS idx_syllabus_unit ON syllabus_sections (unit);
CREATE INDEX IF NOT EXISTS idx_syllabus_module ON syllabus_sections (module);

-- Optional: Add an index for vector similarity search (Highly Recommended for performance)
-- Choose ONE of the following based on your data size and query patterns. HNSW is often preferred for high-dimensional data.
-- Option A: IVFFlat (Good for moderate sized datasets)
-- The number of lists should be chosen carefully, e.g., sqrt(N) where N is the number of rows. Start with 100-1000.
-- CREATE INDEX ON syllabus_sections USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Option B: HNSW (Hierarchical Navigable Small Worlds - often faster for large datasets and high dimensions)
-- Adjust m and ef_construction based on performance needs and available memory.
-- CREATE INDEX ON syllabus_sections USING hnsw (embedding vector_cosine_ops); -- WITH (m = 16, ef_construction = 64);

-- Ensure you have the match_syllabus_sections function created as shown in the CAPE_GPT_CHUNKING.py comments!

