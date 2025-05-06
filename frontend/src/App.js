import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, CssBaseline } from '@mui/material';
import Layout from './Layout';
import theme from './theme';
import Optimize from './components/Route/OptimizeRoutes';
import StoreManagement from './components/StoresManagement';
import Login from './components/Login';
import Dashboard from './components/Dashboard';

import LandingPage from './components/LandingPage';

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Routes>
          <Route path="/" element={
              
            //   <LandingPage />} />
            // <Route path="/login" element={
              
              <Login />} /> 
          <Route
            path="/Optimize"
            element={
              <Layout>
                <Optimize />
              </Layout>
            }
          />
          <Route
            path="/store-management"
            element={
              <Layout>
                <StoreManagement />
              </Layout>
            }
          />
          <Route
            path="/Dashboard"
            element={
              <Layout>
                <Dashboard />
              </Layout>
            }
          />

         
        </Routes>
      </Router>
    </ThemeProvider>
  );
}

export default App;
