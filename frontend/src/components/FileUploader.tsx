// frontend/src/components/FileUploader.tsx
import React, { useState, useRef, useCallback } from 'react';
import { Box, Typography, Button, Paper, IconButton } from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import ImageIcon from '@mui/icons-material/Image';
import { IFileUploaderProps } from '../types';

const FileUploader: React.FC<IFileUploaderProps> = ({ onFileUpload, disabled = false }) => {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Handle file selection
  const handleFileSelected = useCallback(
    (files: FileList | null) => {
      if (!files || files.length === 0 || disabled) return;

      const file = files[0];
      
      // Check if file is an image
      if (!file.type.startsWith('image/')) {
        alert('Please upload an image file (JPEG, PNG, etc.)');
        return;
      }

      // Check file size (limit to 5MB)
      if (file.size > 5 * 1024 * 1024) {
        alert('File size exceeds 5MB. Please upload a smaller file.');
        return;
      }

      onFileUpload(file);
      
      // Reset file input after upload
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    },
    [onFileUpload, disabled]
  );

  // Handle drag events
  const handleDragEnter = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (!disabled) setIsDragging(true);
  }, [disabled]);

  const handleDragLeave = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (!disabled) setIsDragging(true);
  }, [disabled]);

  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    
    if (disabled) return;
    
    const { files } = e.dataTransfer;
    handleFileSelected(files);
  }, [handleFileSelected, disabled]);

  // Handle manual file selection
  const handleBrowseClick = useCallback(() => {
    if (disabled) return;
    fileInputRef.current?.click();
  }, [disabled]);

  const handleFileInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const { files } = e.target;
    handleFileSelected(files);
  }, [handleFileSelected]);

  return (
    <Paper
      elevation={0}
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
      sx={{
        border: '2px dashed',
        borderColor: isDragging ? 'primary.main' : 'grey.400',
        borderRadius: 2,
        p: 3,
        textAlign: 'center',
        backgroundColor: isDragging ? 'action.hover' : 'background.paper',
        transition: 'all 0.2s ease',
        opacity: disabled ? 0.6 : 1,
        cursor: disabled ? 'not-allowed' : 'pointer',
      }}
      onClick={handleBrowseClick}
    >
      <input
        type="file"
        accept="image/*"
        hidden
        ref={fileInputRef}
        onChange={handleFileInputChange}
        disabled={disabled}
      />
      
      <IconButton 
        color="primary" 
        sx={{ mb: 1, pointerEvents: 'none' }}
        disabled={disabled}
      >
        <CloudUploadIcon fontSize="large" />
      </IconButton>
      
      <Typography variant="body1" gutterBottom>
        {isDragging ? 'Drop your image here' : 'Drag & drop an image here'}
      </Typography>
      
      <Typography variant="body2" color="text.secondary" gutterBottom>
        or
      </Typography>
      
      <Button
        variant="outlined"
        startIcon={<ImageIcon />}
        size="small"
        disabled={disabled}
        sx={{ pointerEvents: 'none' }}
      >
        Browse for files
      </Button>
      
      <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 1 }}>
        Supported formats: JPEG, PNG, BMP, WEBP
      </Typography>
    </Paper>
  );
};

export default FileUploader;
