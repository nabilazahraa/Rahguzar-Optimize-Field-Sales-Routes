import React from 'react';
import Header from './components/Header';
import Footer from './components/Footer';
import { Box } from '@mui/material';

function Layout({ children }) {
  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        height: '100vh', // Use fixed height instead of minHeight
        overflow: 'hidden' // Prevent outer scrolling
      }}
    >
      <Header />
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          backgroundColor: '#f9f9fc',
          overflow: 'auto', // This allows scrolling inside the content area only
          padding: '24px 48px',
        }}
      >
        {children}
      </Box>
      <Footer />
    </Box>
  );
}

export default Layout;