// Types for ExamSage application

// Message sender types
export type MessageSender = 'user' | 'bot' | 'system';

// Message content types
export type MessageType = 
  | 'text' 
  | 'image_preview' 
  | 'loading' 
  | 'error' 
  | 'analysis';

// Chat message interface
export interface IChatMessage {
  id: string;
  timestamp: number;
  sender: MessageSender;
  type: MessageType;
  content: string | IAnalyzeResponse;
  imagePreviewUrl?: string;
}

// Frequency data for visualization
export interface IFrequencyData {
  year: number;
  count: number;
}

// Similar question interface
export interface ISimilarQuestion {
  id: string;
  content: string;
  year: number;
  paper?: string;
  questionNumber?: string;
  subject: string;
  similarity: number;
}

// Syllabus objective interface
export interface ISyllabusObjective {
  id: string;
  subject: string;
  unit: string;
  module?: string;
  content: string;
  objectiveId?: string;
  similarity: number;
}

// Analysis response from the API
export interface IAnalyzeResponse {
  questionText: string;
  approach: string;
  keyConcepts: string[];
  commonPitfalls: string[];
  syllabusObjectives: ISyllabusObjective[];
  similarQuestions: ISimilarQuestion[];
  frequencyData: IFrequencyData[];
  frequencySummary: string;
}

// Props for chart component
export interface IFrequencyChartProps {
  data: IFrequencyData[];
  summary: string;
}

// Props for file uploader component
export interface IFileUploaderProps {
  onFileUpload: (file: File) => void;
  disabled?: boolean;
}

// Props for chat input component
export interface IChatInputProps {
  onFileUpload: (file: File) => void;
  disabled?: boolean;
}

// Props for chat message component
export interface IChatMessageProps {
  message: IChatMessage;
}
