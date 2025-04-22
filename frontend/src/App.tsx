// frontend/src/App.tsx
import React, { useState, useCallback, useRef, useEffect } from 'react';
import { Box, Container, Typography, Paper, CircularProgress, Alert } from '@mui/material';
import ChatInput from './components/ChatInput';
import ChatMessage from './components/ChatMessage';
import { IChatMessage, IAnalyzeResponse } from './types';
import { analyzeQuestion } from './services/api';

function App() {
  const [messages, setMessages] = useState<IChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const chatEndRef = useRef<null | HTMLDivElement>(null); // To auto-scroll

  // Function to add a new message to the state
  const addMessage = (message: Omit<IChatMessage, 'id' | 'timestamp'>) => {
    setMessages((prev) => [
      ...prev,
      { ...message, id: crypto.randomUUID(), timestamp: Date.now() },
    ]);
  };

   // Auto-scroll to bottom
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);


  // Handler for file upload triggered by ChatInput -> FileUploader
  const handleFileUpload = useCallback(async (file: File) => {
    setError(null); // Clear previous errors
    setIsLoading(true);

    // Create a temporary URL for image preview
    const previewUrl = URL.createObjectURL(file);

    // Add user message (image preview)
    addMessage({
      sender: 'user',
      type: 'image_preview',
      content: `Analyzing image: ${file.name}`,
      imagePreviewUrl: previewUrl,
    });

    // Add bot loading message
    addMessage({
        sender: 'bot',
        type: 'loading',
        content: 'Analyzing your question, please wait...'
    });

    try {
      const analysisResult = await analyzeQuestion(file);

      // Remove loading message
      setMessages(prev => prev.filter(msg => msg.type !== 'loading'));

      // Add bot analysis message
      addMessage({
        sender: 'bot',
        type: 'analysis',
        content: analysisResult, // Pass the full analysis object
      });

    } catch (err: any) {
      console.error("Analysis failed:", err);
       // Remove loading message
      setMessages(prev => prev.filter(msg => msg.type !== 'loading'));
      // Add error message
      const errorMsg = err instanceof Error ? err.message : "An unknown error occurred.";
      setError(errorMsg); // Set error state to display persistent error
      addMessage({
        sender: 'system',
        type: 'error',
        content: `Error: ${errorMsg}`,
      });
    } finally {
      setIsLoading(false);
      // Revoke the object URL to free memory after a short delay
      // (ensure image has had time to render if needed)
      setTimeout(() => URL.revokeObjectURL(previewUrl), 1000);
    }
  }, []);

  return (
    <Container maxWidth="md" sx={{ height: '100vh', display: 'flex', flexDirection: 'column', pt: 2, pb: 2 }}>
       <Typography variant="h4" component="h1" gutterBottom align="center">
         ExamSage Chat
       </Typography>
       <Paper elevation={2} sx={{ flexGrow: 1, overflowY: 'auto', p: 2, display: 'flex', flexDirection: 'column' }}>
          {messages.map((msg) => (
            <ChatMessage key={msg.id} message={msg} />
          ))}
          {/* Element to scroll to */}
          <div ref={chatEndRef} />
       </Paper>
        {/* Display persistent error above input if needed */}
       {error && !isLoading && <Alert severity="error" sx={{ mt: 1 }}>{error}</Alert>}
       <ChatInput onFileUpload={handleFileUpload} disabled={isLoading} />
    </Container>
  );
}

export default App;
