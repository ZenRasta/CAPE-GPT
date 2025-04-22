// frontend/src/services/api.ts
import axios from 'axios';
import { IAnalyzeResponse, IFrequencyData, ISyllabusObjective, ISimilarQuestion } from '../types';

// Create axios instance with base URL and common config
const api = axios.create({
  baseURL: '/api', // Proxy setup in vite.config.ts will forward requests to the backend
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * Function to transform backend response to frontend format
 */
const transformAnalysisResponse = (backendResponse: any): IAnalyzeResponse => {
  // Extract recognized question text
  const questionText = backendResponse.recognized_question_text || 'Unable to extract question text';
  
  // Extract approach from analysis
  const approach = backendResponse.analysis.detailed_approach;
  
  // Extract key concepts and pitfalls
  const keyConcepts = backendResponse.analysis.key_concepts;
  const commonPitfalls = backendResponse.analysis.common_pitfalls;
  
  // Transform syllabus mapping to syllabusObjectives array
  const syllabusObjectives: ISyllabusObjective[] = [];
  if (backendResponse.syllabus_mapping && backendResponse.syllabus_mapping.objective_text) {
    syllabusObjectives.push({
      id: backendResponse.syllabus_mapping.objective_id || 'unknown',
      subject: '', // Not provided in backend response
      unit: '', // Not provided in backend response
      content: backendResponse.syllabus_mapping.objective_text,
      objectiveId: backendResponse.syllabus_mapping.objective_id,
      similarity: 1.0, // Default to highest similarity since this is the best match
    });
  }
  
  // Transform similar questions
  const similarQuestions: ISimilarQuestion[] = backendResponse.similar_past_questions.map((q: any) => ({
    id: q.id.toString(),
    content: q.content_snippet,
    year: q.year || 0,
    paper: q.paper,
    questionNumber: q.question_number,
    subject: '', // Not provided directly in backend response
    similarity: q.similarity_score || 0,
  }));
  
  // Transform frequency data
  const frequencyData: IFrequencyData[] = Object.entries(backendResponse.frequency_analysis.years_distribution)
    .map(([yearStr, count]) => ({
      year: parseInt(yearStr, 10),
      count: count as number,
    }))
    .sort((a, b) => a.year - b.year);
  
  // Extract frequency summary
  const frequencySummary = backendResponse.frequency_analysis.summary_statement;
  
  return {
    questionText,
    approach,
    keyConcepts,
    commonPitfalls,
    syllabusObjectives,
    similarQuestions,
    frequencyData,
    frequencySummary,
  };
};

/**
 * Send a question image to the backend for analysis
 */
export const analyzeQuestion = async (file: File): Promise<IAnalyzeResponse> => {
  try {
    // Create form data with file
    const formData = new FormData();
    formData.append('file', file);
    
    // Set headers for form data
    const config = {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    };
    
    // Send request to backend
    const response = await api.post('/analyze_question', formData, config);
    
    // Transform and return data
    return transformAnalysisResponse(response.data);
  } catch (error) {
    console.error('Error analyzing question:', error);
    throw new Error(
      error instanceof Error 
        ? error.message 
        : 'Failed to analyze question. Please try again.'
    );
  }
};

// Export the API instance in case we need to use it elsewhere
export default api;
