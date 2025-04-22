// frontend/src/components/ChatInput.tsx
import React, { useState } from 'react';
import { Box, Grid } from '@mui/material';
import FileUploader from './FileUploader';
import { IChatInputProps } from '../types';

const ChatInput: React.FC<IChatInputProps> = ({ onFileUpload, disabled = false }) => {
  return (
    <Box sx={{ mt: 2 }}>
      <Grid container spacing={2}>
        <Grid item xs={12}>
          <FileUploader 
            onFileUpload={onFileUpload} 
            disabled={disabled} 
          />
        </Grid>
      </Grid>
    </Box>
  );
};

export default ChatInput;
