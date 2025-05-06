import React from 'react';
import { CircularProgress, Box, Typography, Paper } from '@mui/material';

function LoadingSpinner_dashboard() { // Changed from LoadingSpinner to LoadingSpinner_dashboard
  return (
    <Box
      sx={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        backgroundColor: 'rgba(255, 255, 255, 0.6)', // Subtle translucency
        backdropFilter: 'blur(8px)', // Modern blur effect
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        zIndex: 1300, // Above other elements
      }}
    >
      <Paper
        elevation={6}
        sx={{
          padding: 4,
          borderRadius: 2,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          backgroundColor: 'rgba(255, 255, 255, 0.9)', // Slightly translucent white
          boxShadow: '0 8px 16px rgba(0, 0, 0, 0.2)', // Modern shadow effect
        }}
      >
        <CircularProgress size={70} thickness={5} color="primary" />
        <Box sx={{ mt: 2 }}>
          <Typography variant="h6" color="text.primary" align="center">
            Fetching data...
          </Typography>
        </Box>
      </Paper>
    </Box>
  );
}

export default LoadingSpinner_dashboard; // Matches the function name