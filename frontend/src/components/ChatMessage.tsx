// frontend/src/components/ChatMessage.tsx
import React from 'react';
import { Paper, Typography, Box, List, ListItem, ListItemText, Divider, CircularProgress, Chip } from '@mui/material';
import { IChatMessageProps, IAnalyzeResponse, ISyllabusObjective, ISimilarQuestion } from '../types';
import FrequencyChart from './FrequencyChart';

const ChatMessage: React.FC<IChatMessageProps> = ({ message }) => {
  // Helper to determine message alignment based on sender
  const isUserMessage = message.sender === 'user';
  
  // Format a timestamp (Unix milliseconds) to a readable time string
  const formatTime = (timestamp: number): string => {
    return new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  // Render analysis content
  const renderAnalysisContent = () => {
    const analysisData = message.content as IAnalyzeResponse;
    
    return (
      <Box>
        <Typography variant="h6" gutterBottom>Question Analysis</Typography>
        
        {/* Extracted Question Text */}
        <Typography variant="subtitle1" fontWeight="bold">Recognized Question:</Typography>
        <Typography paragraph>{analysisData.questionText}</Typography>
        
        {/* Approach */}
        <Typography variant="subtitle1" fontWeight="bold">Recommended Approach:</Typography>
        <Typography paragraph>{analysisData.approach}</Typography>
        
        {/* Key Concepts */}
        <Typography variant="subtitle1" fontWeight="bold">Key Concepts:</Typography>
        <List dense>
          {analysisData.keyConcepts.map((concept, index) => (
            <ListItem key={index}>
              <ListItemText primary={concept} />
            </ListItem>
          ))}
        </List>
        
        {/* Common Pitfalls */}
        <Typography variant="subtitle1" fontWeight="bold">Common Pitfalls:</Typography>
        <List dense>
          {analysisData.commonPitfalls.map((pitfall, index) => (
            <ListItem key={index}>
              <ListItemText primary={pitfall} />
            </ListItem>
          ))}
        </List>
        
        <Divider sx={{ my: 2 }} />
        
        {/* Syllabus Objectives */}
        <Typography variant="h6" gutterBottom>Syllabus Mapping</Typography>
        {analysisData.syllabusObjectives.length > 0 ? (
          <List>
            {analysisData.syllabusObjectives.map((obj: ISyllabusObjective) => (
              <Paper key={obj.id} elevation={1} sx={{ p: 1, mb: 1 }}>
                <Box display="flex" justifyContent="space-between" alignItems="center">
                  <Typography variant="body2" color="primary">
                    {obj.subject} • {obj.unit} {obj.module && `• ${obj.module}`}
                  </Typography>
                  <Chip size="small" label={`${Math.round(obj.similarity * 100)}% match`} color="primary" />
                </Box>
                <Typography variant="body1">{obj.content}</Typography>
                {obj.objectiveId && <Typography variant="caption">Objective ID: {obj.objectiveId}</Typography>}
              </Paper>
            ))}
          </List>
        ) : (
          <Typography color="text.secondary">No direct syllabus matches found.</Typography>
        )}
        
        <Divider sx={{ my: 2 }} />
        
        {/* Similar Questions */}
        <Typography variant="h6" gutterBottom>Similar Past Questions</Typography>
        {analysisData.similarQuestions.length > 0 ? (
          <List>
            {analysisData.similarQuestions.map((q: ISimilarQuestion) => (
              <Paper key={q.id} elevation={1} sx={{ p: 1, mb: 1 }}>
                <Box display="flex" justifyContent="space-between" alignItems="center">
                  <Typography variant="body2" color="primary">
                    {q.year} • {q.subject} {q.paper && `• ${q.paper}`} {q.questionNumber && `• Q${q.questionNumber}`}
                  </Typography>
                  <Chip size="small" label={`${Math.round(q.similarity * 100)}% similar`} color="secondary" />
                </Box>
                <Typography variant="body1">{q.content}</Typography>
              </Paper>
            ))}
          </List>
        ) : (
          <Typography color="text.secondary">No similar past questions found.</Typography>
        )}
        
        <Divider sx={{ my: 2 }} />
        
        {/* Frequency Analysis */}
        {analysisData.frequencyData.length > 0 && (
          <>
            <Typography variant="h6" gutterBottom>Frequency Analysis</Typography>
            <FrequencyChart data={analysisData.frequencyData} summary={analysisData.frequencySummary} />
          </>
        )}
      </Box>
    );
  };

  // Render message content based on message type
  const renderContent = () => {
    switch (message.type) {
      case 'text':
        return <Typography>{message.content as string}</Typography>;
        
      case 'image_preview':
        return (
          <Box>
            <Typography paragraph>{message.content as string}</Typography>
            {message.imagePreviewUrl && (
              <Box sx={{ maxWidth: '100%', maxHeight: 300, overflow: 'hidden', borderRadius: 1 }}>
                <img 
                  src={message.imagePreviewUrl} 
                  alt="Uploaded question" 
                  style={{ maxWidth: '100%', maxHeight: 300, objectFit: 'contain' }} 
                />
              </Box>
            )}
          </Box>
        );
        
      case 'loading':
        return (
          <Box display="flex" alignItems="center">
            <CircularProgress size={20} sx={{ mr: 2 }} />
            <Typography>{message.content as string}</Typography>
          </Box>
        );
        
      case 'error':
        return <Typography color="error">{message.content as string}</Typography>;
        
      case 'analysis':
        return renderAnalysisContent();
        
      default:
        return <Typography>{message.content as string}</Typography>;
    }
  };

  return (
    <Box 
      sx={{ 
        display: 'flex', 
        justifyContent: isUserMessage ? 'flex-end' : 'flex-start',
        mb: 2
      }}
    >
      <Box 
        sx={{ 
          maxWidth: '75%', 
          minWidth: message.type === 'analysis' ? '90%' : 'auto'
        }}
      >
        <Paper 
          elevation={1}
          sx={{ 
            p: 2, 
            bgcolor: isUserMessage ? 'primary.light' : 'background.paper',
            color: isUserMessage ? 'primary.contrastText' : 'text.primary'
          }}
        >
          {renderContent()}
        </Paper>
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5 }}>
          {formatTime(message.timestamp)}
        </Typography>
      </Box>
    </Box>
  );
};

export default ChatMessage;
