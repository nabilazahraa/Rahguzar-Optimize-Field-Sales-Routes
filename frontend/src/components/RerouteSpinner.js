
import React, { useState, useEffect, useRef } from 'react';
import { CircularProgress, Box, Typography, Paper, Fade } from '@mui/material';

// Note: Replace this import with your actual logo import path

const loadingStages = [
  {
    message: "Dynamic Rerouting in Progress...",
    duration: 3500, // 3 seconds
  },
  { message: 'Rerouting Schedule...', duration: 30000 },

  
  {
    message: 'Creating schedule...',
    duration: 30000, // 5 seconds
  },
  {
    message: 'Optimizing routes...',
    duration: 30000, // 6 seconds
  },
];

const finalMessage = 'Sorry for the wait, we are getting the best results for you!';

function LoadingSpinner() {
  const [stageIndex, setStageIndex] = useState(0);
  const [fadeIn, setFadeIn] = useState(true);
  const [isFinalMessage, setIsFinalMessage] = useState(false);
  const [progress, setProgress] = useState(0);
  const progressIntervalRef = useRef(null);
  const timeoutRef = useRef(null);

  // Clean up function to clear all timers
  const cleanupTimers = () => {
    if (progressIntervalRef.current) clearInterval(progressIntervalRef.current);
    if (timeoutRef.current) clearTimeout(timeoutRef.current);
  };

  useEffect(() => {
    if (isFinalMessage) return;

    const currentStage = loadingStages[stageIndex];
    const stageDuration = currentStage.duration;
    
    // Reset progress at the start of each stage
    setProgress(0);
    
    // Create a smooth progress animation for this stage
    progressIntervalRef.current = setInterval(() => {
      setProgress(prev => {
        const increment = 100 / (stageDuration / 100); // Calculate increment to reach 100% at end of duration
        return Math.min(prev + increment, 100);
      });
    }, 100);

    // Schedule the next stage transition
    timeoutRef.current = setTimeout(() => {
      // Clear the progress interval
      clearInterval(progressIntervalRef.current);
      
      // Start fade out
      setFadeIn(false);
      
      // After fade out completes, move to next stage or final message
      setTimeout(() => {
        if (stageIndex < loadingStages.length - 1) {
          setStageIndex(prevIndex => prevIndex + 1);
        } else {
          setIsFinalMessage(true);
        }
        setFadeIn(true); // Start fade in
      }, 500); // This should match the Fade timeout
      
    }, stageDuration);

    // Cleanup on unmount or when stage changes
    return cleanupTimers;
  }, [stageIndex, isFinalMessage]);

  // Cleanup on unmount
  useEffect(() => {
    return cleanupTimers;
  }, []);

  return (
    <Box
      sx={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        backgroundColor: 'rgba(255, 255, 255, 0.6)',
        backdropFilter: 'blur(5px)',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        zIndex: 1300,
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
          backgroundColor: 'rgba(255, 255, 255, 0.9)',
          boxShadow: '0 8px 16px rgba(0, 0, 0, 0.2)',
          maxWidth: '400px',
          width: '90%',
        }}
      >
        {/* Logo above spinner */}
        <Box sx={{ mb: 3, width: '60%', maxWidth: '150px' }}>
          <img 
            src={"./3.png"} 
            alt="Company Logo" 
            style={{ width: '100%', height: 'auto' }} 
          />
        </Box>
        
        {/* Progress spinner */}
        <Box sx={{ position: 'relative', display: 'inline-flex' }}>
          <CircularProgress 
            size={70} 
            thickness={5} 
            color="primary" 
          />
          {!isFinalMessage && (
            <Box
              sx={{
                top: 0,
                left: 0,
                bottom: 0,
                right: 0,
                position: 'absolute',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <Typography variant="caption" component="div" color="text.secondary">
                {`${Math.round(progress)}%`}
              </Typography>
            </Box>
          )}
        </Box>
        
        {/* Loading message */}
        <Box sx={{ mt: 3, minHeight: '60px', width: '100%' }}>
          <Fade in={fadeIn} timeout={500}>
            <Typography variant="h6" color="text.primary" align="center">
              {isFinalMessage ? finalMessage : loadingStages[stageIndex].message}
            </Typography>
          </Fade>
        </Box>

        {/* Stage indicators */}
        <Box sx={{ display: 'flex', mt: 2, gap: 1 }}>
          {loadingStages.map((_, idx) => (
            <Box
              key={idx}
              sx={{
                width: 8,
                height: 8,
                borderRadius: '50%',
                backgroundColor: idx === stageIndex && !isFinalMessage 
                  ? 'primary.main' 
                  : idx < stageIndex || isFinalMessage
                    ? 'success.main'
                    : 'grey.300',
                transition: 'all 0.3s ease'
              }}
            />
          ))}
        </Box>
      </Paper>
    </Box>
  );
}

export default LoadingSpinner;