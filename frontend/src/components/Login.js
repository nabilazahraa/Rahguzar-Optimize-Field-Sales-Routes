import AuthLayout from './AuthLayout';
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  TextField,
  Button,
  Typography,
  Paper,
  Grid,
  Link,
  Container,
  InputAdornment,
  IconButton,
  Divider,
  Checkbox,
  FormControlLabel,
} from '@mui/material';
import { styled } from '@mui/material/styles';
import { motion } from 'framer-motion';
import { Link as RouterLink } from 'react-router-dom';
import VisibilityIcon from '@mui/icons-material/Visibility';
import VisibilityOffIcon from '@mui/icons-material/VisibilityOff';
import PersonOutlineIcon from '@mui/icons-material/PersonOutline';
import LockOutlinedIcon from '@mui/icons-material/LockOutlined';

const logoVariants = {
  hidden: { opacity: 0, scale: 0.9 },
  visible: { 
    opacity: 1, 
    scale: 1, 
    transition: { 
      duration: 1.5,
      ease: 'easeOut' 
    } 
  },
  hover: { 
    scale: 1.05, 
    boxShadow: '0 6px 25px rgba(0, 0, 0, 0.15)', 
    transition: { duration: 0.3 } 
  },
};

const formVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      delay: 0.3,
      duration: 0.8
    }
  }
};

const LoginPaper = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(5),
  background: '#ffffff',
  boxShadow: '0px 8px 30px rgba(0, 0, 0, 0.08)',
  borderRadius: '16px',
  marginTop: '-40px',
  position: 'relative',
  overflow: 'hidden',
  '&::before': {
    content: '""',
    position: 'absolute',
    top: 0,
    left: 0,
    width: '6px',
    height: '100%',
    background: 'linear-gradient(180deg, #161181 0%, #3F3A9D 100%)',
  }
}));

const StyledTextField = styled(TextField)(({ theme }) => ({
  marginBottom: theme.spacing(3),
  '& .MuiOutlinedInput-root': {
    borderRadius: '8px',
    transition: 'all 0.3s ease',
    '&:hover fieldset': {
      borderColor: '#3F3A9D',
    },
    '&.Mui-focused fieldset': {
      borderColor: '#161181',
      borderWidth: '2px',
    },
    '& .MuiInputAdornment-root': {
      color: '#666',
    },
  },
  '& .MuiInputLabel-root': {
    color: '#666',
    '&.Mui-focused': {
      color: '#161181',
    },
    '&.MuiInputLabel-shrink': {
      background: '#fff',
      padding: '0 8px',
    }
  },
  '& input': {
    padding: '14px 12px 14px 0',
  },
  '& input:-webkit-autofill': {
    WebkitBoxShadow: '0 0 0px 1000px white inset !important',
    transition: 'background-color 5000s ease-in-out 0s',
  },
}));

const LoginButton = styled(Button)(({ theme }) => ({
  background: 'linear-gradient(90deg, #161181 0%, #3F3A9D 100%)',
  padding: '12px',
  borderRadius: '8px',
  fontSize: '16px',
  fontWeight: 500,
  textTransform: 'none',
  boxShadow: '0 4px 12px rgba(22, 17, 129, 0.3)',
  transition: 'all 0.3s ease',
  '&:hover': {
    background: 'linear-gradient(90deg, #130f6a 0%, #322f7e 100%)',
    boxShadow: '0 6px 16px rgba(22, 17, 129, 0.4)',
    transform: 'translateY(-2px)',
  },
}));

const SocialButton = styled(Button)(({ theme }) => ({
  borderRadius: '8px',
  padding: '10px',
  textTransform: 'none',
  border: '1px solid #e0e0e0',
  color: '#555',
  fontWeight: 500,
  '&:hover': {
    backgroundColor: '#f5f5f5',
  },
}));

function Login() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [rememberMe, setRememberMe] = useState(false);
  const BASE_URL = process.env.REACT_APP_API_BASE_URL;

  const navigate = useNavigate();
  
  const handleLogin = async (event) => {
    event.preventDefault();
  
    try {
      const response = await fetch(`${BASE_URL}/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });
  
      const data = await response.json();
  
      if (response.ok) {
        localStorage.setItem("distributor_id", data.distributor_id);
        navigate("/Optimize");
      } else {
        alert("Login failed: " + data.message);
      }
    } catch (error) {
      console.error("Login error:", error);
      alert("An error occurred while logging in.");
    }
  };
  
  return (
    <AuthLayout>
      <Container 
        maxWidth={false} 
        sx={{ 
          minHeight: 'calc(100vh - 70px)',
          background: 'linear-gradient(135deg, #f9f9fc 0%, #f0f0f8 100%)',
          pt: 12,
          pb: 4,
        }}
      >
        <Grid 
          container 
          spacing={4}
          sx={{ 
            justifyContent: 'space-between',
            alignItems: 'center',
          }}
        >
          {/* Login Box - Left Side */}
          <Grid item xs={12} md={5} lg={4} sx={{ ml: { md: 8, lg: 12 } }}>
            <motion.div
              variants={formVariants}
              initial="hidden"
              animate="visible"
            >
              <LoginPaper elevation={3}>
                <Typography 
                  variant="h4" 
                  gutterBottom 
                  sx={{ 
                    color: '#161181',
                    fontWeight: 600,
                    mb: 1,
                    textAlign: 'left'
                  }}
                >
                  Welcome Back
                </Typography>
                <Typography 
                  variant="body1" 
                  sx={{ 
                    color: '#666',
                    mb: 4,
                  }}
                >
                  Please enter your credentials to continue
                </Typography>
                
                <form onSubmit={handleLogin} autoComplete='off'>
                  <StyledTextField
                    label="Username"
                    id="username"
                    type="text"
                    variant="outlined"
                    autoComplete='off'
                    fullWidth
                    required
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    InputProps={{
                      startAdornment: (
                        <InputAdornment position="start">
                          <PersonOutlineIcon sx={{ color: '#666' }} />
                        </InputAdornment>
                      ),
                    }}
                  />

                  <StyledTextField
                    label="Password"
                    id="password"
                    type={showPassword ? "text" : "password"}
                    variant="outlined"
                    autoComplete="new-password"
                    fullWidth
                    required
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    InputProps={{
                      startAdornment: (
                        <InputAdornment position="start">
                          <LockOutlinedIcon sx={{ color: '#666' }} />
                        </InputAdornment>
                      ),
                      endAdornment: (
                        <InputAdornment position="end">
                          <IconButton
                            onClick={() => setShowPassword(!showPassword)}
                            edge="end"
                          >
                            {showPassword ? <VisibilityOffIcon /> : <VisibilityIcon />}
                          </IconButton>
                        </InputAdornment>
                      )
                    }}
                  />
                  
                  <Box sx={{ 
                    display: 'flex', 
                    justifyContent: 'space-between', 
                    alignItems: 'center', 
                    mb: 3 
                  }}>
                    <FormControlLabel
                      control={
                        <Checkbox 
                          checked={rememberMe}
                          onChange={(e) => setRememberMe(e.target.checked)}
                          sx={{ 
                            color: '#161181',
                            '&.Mui-checked': {
                              color: '#161181',
                            },
                          }}
                        />
                      }
                      label="Remember me"
                      sx={{ '& .MuiTypography-root': { fontSize: '14px', color: '#666' } }}
                    />
                    {/* <Link
                      component={RouterLink}
                      to="/forgot-password"
                      sx={{
                        color: '#161181',
                        textDecoration: 'none',
                        fontSize: '14px',
                        fontWeight: 500,
                        '&:hover': {
                          textDecoration: 'underline',
                        }
                      }}
                    >
                      Forgot password?
                    </Link> */}
                  </Box>
                  
                  <Box sx={{ mb: 4 }}>
                    <LoginButton 
                      type="submit" 
                      variant="contained" 
                      fullWidth
                    >
                      Sign In
                    </LoginButton>
                  </Box>
                </form>
                
               
                
                
              </LoginPaper>
            </motion.div>
          </Grid>

          {/* Image - Right Side */}
          <Grid 
            item 
            xs={12} 
            md={7} 
            sx={{ 
              display: { xs: 'none', md: 'flex' },
              justifyContent: 'center',
              alignItems: 'center',
            }}
          >
            <motion.div
              variants={logoVariants}
              initial="hidden"
              animate="visible"
              whileHover="hover"
            >
              <motion.img
                src="/3.png"
                alt="PJP Optimization Logo"
                style={{
                  width: '550px',
                  height: 'auto',
                  filter: 'drop-shadow(0px 10px 20px rgba(0, 0, 0, 0.15))',
                }}
              />
            </motion.div>
          </Grid>
        </Grid>
      </Container>
    </AuthLayout>
  );
}

export default Login;