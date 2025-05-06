import React from 'react';
import AuthHeader from './AuthHeader';
import { Box } from '@mui/material';
import Footer from './Footer';

function AuthLayout({ children }) {
  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        height: '100vh', // Use fixed height instead of minHeight
        overflow: 'hidden' // Prevent outer scrolling
      }}
    >
      <AuthHeader />
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          backgroundColor: '#f9f9fc',
          overflow: 'hidden', // Changed from 'auto' to 'hidden' to prevent scrolling
          padding: '24px 48px',
        }}
      >
        {children}
      </Box>
      <Footer />
    </Box>
  );
}

export default AuthLayout;