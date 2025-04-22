// frontend/src/main.tsx
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App'; // Import the main App component
import { ThemeProvider, createTheme, CssBaseline } from '@mui/material';

// Optional: Define a basic MUI theme (customize colors, typography etc. here)
const theme = createTheme({
  palette: {
    // mode: 'light', // or 'dark'
    primary: {
      main: '#1976d2', // Example primary color (MUI blue)
    },
    secondary: {
      main: '#dc004e', // Example secondary color (MUI pink)
    },
  },
  // You can customize typography, breakpoints, etc. here
});

// Find the root element in your index.html
const rootElement = document.getElementById('root');

// Ensure the root element exists before rendering
if (!rootElement) {
  throw new Error("Failed to find the root element. Make sure your index.html has <div id='root'></div>.");
}

// Create the React root
const root = ReactDOM.createRoot(rootElement);

// Render the application
root.render(
  <React.StrictMode>
    {/* Apply the MUI theme and baseline CSS reset */}
    <ThemeProvider theme={theme}>
      <CssBaseline /> {/* Ensures consistent baseline styling */}
      <App /> {/* Render your main application component */}
    </ThemeProvider>
  </React.StrictMode>
);
