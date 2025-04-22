# functions.sql
-- Function to find similar syllabus sections based on embedding & subject
CREATE OR REPLACE FUNCTION match_syllabus_sections (
  query_embedding vector(1536), -- Dimension must match your embeddings
  query_subject text,
  match_threshold float,
  match_count int
)
RETURNS TABLE (
  id bigint,
  subject text,
  unit text,
  module text,
  section_title text,
  objective_id text,
  content text,
  source_document text,
  page_number integer,
  similarity float
)
LANGUAGE sql STABLE
AS $$ -- Start of function body quoted with dollar signs
  SELECT
    ss.id,
    ss.subject,
    ss.unit,
    ss.module,
    ss.section_title,
    ss.objective_id,
    ss.content,
    ss.source_document,
    ss.page_number,
    1 - (ss.embedding <=> query_embedding) AS similarity -- Cosine Similarity
  FROM syllabus_sections ss
  WHERE
    ss.subject = query_subject
    AND 1 - (ss.embedding <=> query_embedding) > match_threshold
  ORDER BY
    similarity DESC
  LIMIT match_count;
$$; -- <<<< IMPORTANT: Ensure this semicolon is present right after the closing $$

-- Optional: Grant execute permission (run separately if needed)
-- GRANT EXECUTE ON FUNCTION match_syllabus_sections TO service_role; --<< Also needs its own semicolon

