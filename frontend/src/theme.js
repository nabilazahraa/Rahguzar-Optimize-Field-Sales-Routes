// theme.js
import { createTheme } from '@mui/material/styles';

const theme = createTheme({
  palette: {
    primary: {
      main: '#161181',
    },
    secondary: {
      main: '#3F3A9D',
    },
    background: {
      default: '#f9f9fc',
      paper: '#ffffff',
    },
    text: {
      primary: '#161181',
      secondary: '#3F3A9D',
    },
  },
  typography: {
    fontFamily: 'Roboto, Arial, sans-serif',
    h5: {
      fontWeight: 700,
      color: '#161181',
    },
    h6: {
      fontWeight: 600,
      color: '#3F3A9D',
    },
    body1: {
      color: '#161181',
    },
    button: {
      textTransform: 'none',
    },
  },
  components: {
    MuiPaper: {
      styleOverrides: {
        root: {
          padding: '8px',
          boxShadow: '0px 4px 20px rgba(0, 0, 0, 0.1)',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: '16px',
          padding: '16px',
          boxShadow: '0px 4px 20px rgba(0, 0, 0, 0.1)',
        },
      },
    },
    MuiTabs: {
      styleOverrides: {
        indicator: {
          height: '3px',
          backgroundColor: '#3F3A9D',
        },
      },
    },
    MuiTab: {
      styleOverrides: {
        root: {
          fontWeight: 600,
          textTransform: 'none',
          color: '#161181',
          '&.Mui-selected': {
            color: '#3F3A9D',
          },
        },
      },
    },
  },
});

export default theme;
