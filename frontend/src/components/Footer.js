import React from 'react';
import { Box, Typography } from '@mui/material';

function Footer() {
  return (
    <Box
      component="footer"
      sx={{
        textAlign: 'center',
        py: 2,
        backgroundColor: '#e9ecef',
        position: 'sticky',
        bottom: 0,
        width: '100%',
        zIndex: 1000,
      }}
    >
      <Typography variant="body2" color="textSecondary">
        © {new Date().getFullYear()} RAHGUZAR. All Rights Reserved.
      </Typography>
    </Box>
  );
}

export default Footer;