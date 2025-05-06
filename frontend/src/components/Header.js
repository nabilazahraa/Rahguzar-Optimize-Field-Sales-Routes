import React, { useState } from 'react';
import { 
  AppBar, 
  Toolbar,
  Button, 
  Box, 
  Typography,
  IconButton,
  Menu,
  MenuItem,
  useMediaQuery,
  Tooltip,
  Divider
} from '@mui/material';
import { styled, useTheme } from '@mui/material/styles';
import { Link, useLocation } from 'react-router-dom';
import LogoutIcon from '@mui/icons-material/Logout';
import MenuIcon from '@mui/icons-material/Menu';
import DashboardIcon from '@mui/icons-material/Dashboard';
import RouteIcon from '@mui/icons-material/Route';
import StorefrontIcon from '@mui/icons-material/Storefront';
import { useNavigate } from "react-router-dom";
import { clearDashboardState } from './Dashboard';

// Styled Components
const Logo = styled('img')(({ theme }) => ({
  height: 70,
  marginRight: theme.spacing(2),
}));


const SmallLogo = styled('img')(({ theme }) => ({
  height: 35,
  width: 35,
  marginLeft: theme.spacing(1),
  borderRadius: '50%',
  boxShadow: '0 2px 4px rgba(0,0,0,0.2)',
  transition: 'transform 0.3s ease',
  '&:hover': {
    transform: 'scale(1.1)',
  },
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

const NavButton = styled(Button)(({ theme, active }) => ({
  margin: theme.spacing(0, 1),
  borderRadius: '8px',
  padding: theme.spacing(0.5, 2),
  position: 'relative',
  fontWeight: 600,
  fontSize: '1rem',
  '&:after': active ? {
    content: '""',
    position: 'absolute',
    bottom: '5px',
    left: '50%',
    transform: 'translateX(-50%)',
    width: '30%',
    height: '3px',
    backgroundColor: '#ffffff',
    borderRadius: '2px',
  } : {},
  transition: 'all 0.3s ease',
  '&:hover': {
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    transform: 'translateY(-2px)',
  },
}));

function Header() {
  const navigate = useNavigate();
  const theme = useTheme();
  const location = useLocation();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [anchorEl, setAnchorEl] = useState(null);

  const handleMenuOpen = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  const handleLogout = () => {
    console.log("Logging out...");
    localStorage.removeItem("distributor_id");
    clearDashboardState();
    handleMenuClose();
    navigate("/");
  };

  const handleRouteClick = () => {
    localStorage.setItem("reset_plan", "true");
    handleMenuClose();
  };

  // Check if current route matches to set active state
  const isActive = (path) => {
    return location.pathname === path;
  };

  return (
    <AppBar
      position="sticky"
      sx={{
        background: 'linear-gradient(90deg, #161181 0%, #3F3A9D 100%)',
        boxShadow: '0px 4px 15px rgba(0, 0, 0, 0.15)',
        zIndex: theme.zIndex.drawer + 1,
      }}
    >
      <Toolbar sx={{ justifyContent: 'space-between' }}>
        {/* Logo and Navigation Combined Section */}
        <Box display="flex" alignItems="center">
          <Link to="/Optimize" style={{ textDecoration: 'none', display: 'flex', alignItems: 'center' }}>
            <Logo
              src="/LOGO2.png"
              alt="PJP Optimization Logo"
            />
          </Link>
          
          {/* Navigation - Desktop View - Now next to logo */}
          {!isMobile && (
            <Box sx={{ display: 'flex', alignItems: 'center', ml: 2 }}>
              <Tooltip title="Plan your routes" arrow>
                <NavButton 
                  component={Link} 
                  to="/Optimize" 
                  color="inherit" 
                  active={isActive('/Optimize') ? 1 : 0}
                  onClick={handleRouteClick}
                  startIcon={<RouteIcon fontSize="small" />}
                >
                  Routes
                </NavButton>
              </Tooltip>
              
              <Tooltip title="Manage stores" arrow>
                <NavButton 
                  component={Link} 
                  to="/store-management" 
                  color="inherit"
                  active={isActive('/store-management') ? 1 : 0}
                  startIcon={<StorefrontIcon fontSize="small" />}
                >
                  Store Management
                </NavButton>
              </Tooltip>
              
              <Tooltip title="View dashboard" arrow>
                <NavButton 
                  component={Link} 
                  to="/dashboard" 
                  color="inherit"
                  active={isActive('/dashboard') ? 1 : 0}
                  startIcon={<DashboardIcon fontSize="small" />}
                >
                  Dashboard
                </NavButton>
              </Tooltip>
            </Box>
          )}
        </Box>

        {/* Empty space filler removed since navigation is no longer centered */}

        {/* Right Section - Logout & Branding */}
        <Box display="flex" alignItems="center">
          {/* Mobile Menu */}
          {isMobile && (
            <>
              <IconButton
                color="inherit"
                aria-label="open menu"
                onClick={handleMenuOpen}
                sx={{ mr: 1 }}
              >
                <MenuIcon />
              </IconButton>
              <Menu
                anchorEl={anchorEl}
                open={Boolean(anchorEl)}
                onClose={handleMenuClose}
                sx={{ mt: 1 }}
              >
                <MenuItem 
                  component={Link} 
                  to="/Optimize" 
                  onClick={handleRouteClick}
                  sx={{ fontSize: '1rem', py: 1.2 }}
                >
                  <RouteIcon sx={{ mr: 1.5 }} /> Routes
                </MenuItem>
                <MenuItem 
                  component={Link} 
                  to="/store-management" 
                  onClick={handleMenuClose}
                  sx={{ fontSize: '1rem', py: 1.2 }}
                >
                  <StorefrontIcon sx={{ mr: 1.5 }} /> Store Management
                </MenuItem>
                <MenuItem 
                  component={Link} 
                  to="/dashboard" 
                  onClick={handleMenuClose}
                  sx={{ fontSize: '1rem', py: 1.2 }}
                >
                  <DashboardIcon sx={{ mr: 1.5 }} /> Dashboard
                </MenuItem>
                <Divider />
                <MenuItem 
                  onClick={handleLogout}
                  sx={{ fontSize: '1rem', py: 1.2 }}
                >
                  <LogoutIcon sx={{ mr: 1.5 }} /> Logout
                </MenuItem>
              </Menu>
            </>
          )}

          {/* Salesflo branding */}
          <SalesfloText variant="h6">
            Salesflo.
            <SmallLogo
              src="/salesflologo.png"
              alt="Salesflo logo"
            />
          </SalesfloText>
          
          {/* Logout Button - Only visible on desktop */}
          {!isMobile && (
            <Tooltip title="Logout from your account" arrow>
             <Button
  variant="contained"
  color="primary"
  onClick={handleLogout}
  sx={{
    ml: 2,
    borderRadius: theme.shape.borderRadius * 2,
    fontWeight: 600,
    textTransform: 'none',
    boxShadow: theme.shadows[2],
    padding: theme.spacing(1, 2),
    transition: 'all 0.3s ease',
    '&:hover': {
      boxShadow: theme.shadows[4],
      transform: 'translateY(-2px)',
      backgroundColor: theme.palette.primary.dark,
    },
  }}
  endIcon={<LogoutIcon />}
/>


            </Tooltip>
          )}
        </Box>
      </Toolbar>
    </AppBar>
  );
}

export default Header;