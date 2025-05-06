import React from 'react';
import { AppBar, Toolbar, Box, Typography } from '@mui/material';
import { styled } from '@mui/material/styles';

// Styled Components
const Logo = styled('img')(({ theme }) => ({
  height: 70,
  marginRight: theme.spacing(2),
}));

const SmallLogo = styled('img')(({ theme }) => ({
  height: 40,
  width: 40,
  marginLeft: theme.spacing(1),
  borderRadius: '50%',
}));

const SalesfloText = styled(Typography)(({ theme }) => ({
  color: '#f0f0f0',
  // fontStyle: 'italic',
  fontWeight: 600,
  fontSize: '1.2rem',
  display: 'flex',
  alignItems: 'center',
  transition: 'all 0.3s ease',
  '&:hover': {
    color: '#ffffff',
    textShadow: '0 0 10px rgba(255,255,255,0.5)',
  },
}));
const StyledToolbar = styled(Toolbar)({
    minHeight: '50px',
    padding: '0 0',
    '@media (min-width: 600px)': {
      minHeight: '70px',
    },
  });

function AuthHeader() {
  return (
    <AppBar
      position="sticky"
      sx={{
        background: 'linear-gradient(90deg, #161181 0%, #3F3A9D 100%)',
        boxShadow: '0px 4px 10px rgba(0, 0, 0, 0.1)',
      }}
    >
      <StyledToolbar>
        {/* Main Logo */}
        <Box display="flex" alignItems="center">
          <Logo
            src="/LOGO2.png"
            alt="PJP Optimization Logo"
          />
        </Box>

        {/* Spacer to push "Salesflo" to the right */}
        <Box sx={{ flexGrow: 1 }} />
        
        {/* "Salesflo" Text with Small Logo */}
        <SalesfloText variant="h6">
          Salesflo
          <SmallLogo
            src="/salesflologo.png"
            alt="Salesflo Small Logo"
          />
        </SalesfloText>
    </StyledToolbar>
    </AppBar>
  );
}

export default AuthHeader;