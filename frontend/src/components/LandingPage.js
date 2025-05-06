import React from 'react';
import { 
  Box, 
  Button, 
  Container, 
  Grid, 
  Typography, 
  Card, 
  CardContent,
  CardMedia,
  List,
  ListItem,
  ListItemIcon,
  ListItemText
} from '@mui/material';
import { styled } from '@mui/material/styles';
import { motion } from 'framer-motion';
import { Link as RouterLink } from 'react-router-dom';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import RouteIcon from '@mui/icons-material/Route';
import AutorenewIcon from '@mui/icons-material/Autorenew';
import StorefrontIcon from '@mui/icons-material/Storefront';
import InsightsIcon from '@mui/icons-material/Insights';
import AuthLayout from './AuthLayout';

// Styled components
const HeroSection = styled(Box)(({ theme }) => ({
  background: 'linear-gradient(135deg, #161181 0%, #3F3A9D 100%)',
  color: '#ffffff',
  padding: theme.spacing(12, 0, 8),
  textAlign: 'center',
}));

const FeatureCard = styled(Card)(({ theme }) => ({
  height: '100%',
  boxShadow: '0px 4px 20px rgba(0, 0, 0, 0.08)',
  borderRadius: '12px',
  transition: 'transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out',
  '&:hover': {
    transform: 'translateY(-5px)',
    boxShadow: '0px 12px 30px rgba(0, 0, 0, 0.12)',
  },
}));

const ActionButton = styled(Button)(({ theme }) => ({
  background: 'linear-gradient(90deg, #161181 0%, #3F3A9D 100%)',
  padding: '12px 24px',
  borderRadius: '8px',
  textTransform: 'none',
  fontSize: '1rem',
  marginTop: theme.spacing(3),
  '&:hover': {
    background: 'linear-gradient(90deg, #130f6a 0%, #322f7e 100%)',
  },
}));

const SectionTitle = styled(Typography)(({ theme }) => ({
  position: 'relative',
  marginBottom: theme.spacing(8),
  fontWeight: 600,
  '&:after': {
    content: '""',
    position: 'absolute',
    bottom: '-12px',
    left: '50%',
    transform: 'translateX(-50%)',
    width: '80px',
    height: '4px',
    background: 'linear-gradient(90deg, #161181 0%, #3F3A9D 100%)',
    borderRadius: '2px',
  },
}));

const heroContentVariants = {
  hidden: { opacity: 0, y: 30 },
  visible: { 
    opacity: 1, 
    y: 0, 
    transition: { 
      duration: 0.8, 
      ease: 'easeOut',
      staggerChildren: 0.2,
    } 
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: { 
    opacity: 1, 
    y: 0,
    transition: { duration: 0.5 } 
  }
};

const featureCardVariants = {
  offscreen: { y: 50, opacity: 0 },
  onscreen: {
    y: 0,
    opacity: 1,
    transition: { type: "spring", bounce: 0.4, duration: 0.8 }
  }
};

function LandingPage() {
  return (
    <AuthLayout>
      {/* Hero Section */}
      <HeroSection>
  <Container>
    <motion.div
      variants={heroContentVariants}
      initial="hidden"
      animate="visible"
    >
      <motion.div variants={itemVariants}>
        <Typography 
          variant="h3" 
          component="h1" 
          sx={{ fontWeight: 700, mb: 2, color: '#ffffff' }}
        >
          Rahguzar
        </Typography>
      </motion.div>
      <motion.div variants={itemVariants}>
        <Typography 
          variant="h5" 
          sx={{ fontWeight: 400, mb: 4, color: '#ffffff' }}
        >
          Intelligent Route Planning & Store Management
        </Typography>
      </motion.div>
      <motion.div variants={itemVariants}>
        <Typography 
          variant="body1" 
          sx={{ maxWidth: '800px', mx: 'auto', mb: 5, color: '#ffffff' }}
        >
          Transform your sales team's efficiency with our AI-powered journey planning tool. 
          Create optimized routes, manage stores effectively, and gain actionable insights - all in one platform.
        </Typography>
      </motion.div>
      <motion.div variants={itemVariants}>
        <ActionButton 
          variant="contained" 
          size="large" 
          component={RouterLink} 
          to="/login"
        >
          Get Started
        </ActionButton>
      </motion.div>
    </motion.div>
  </Container>
</HeroSection>


      {/* Key Features Section */}
      <Box sx={{ py: 10, backgroundColor: '#f9f9fc' }}>
        <Container>
          <SectionTitle variant="h4" align="center">
            Key Features
          </SectionTitle>
          
          <Grid container spacing={4}>
            {/* Feature 1 */}
            <Grid item xs={12} sm={6} md={3}>
              
               <motion.div
               initial="offscreen"
               animate="onscreen"
               variants={featureCardVariants}
             >
           
             
                <FeatureCard>
                  <CardContent sx={{ textAlign: 'center', p: 3 }}>
                    <RouteIcon sx={{ fontSize: 60, color: '#161181', mb: 2 }} />
                    <Typography variant="h6" component="h3" sx={{ mb: 2, fontWeight: 600 }}>
                      Route Plan Generation
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Create optimized journey plans that maximize coverage while minimizing travel time and costs.
                    </Typography>
                  </CardContent>
                </FeatureCard>
              </motion.div>
            </Grid>

            {/* Feature 2 */}
            <Grid item xs={12} sm={6} md={3}>
            <motion.div
               initial="offscreen"
               animate="onscreen"
               variants={featureCardVariants}
             >
                <FeatureCard>
                  <CardContent sx={{ textAlign: 'center', p: 3 }}>
                    <AutorenewIcon sx={{ fontSize: 60, color: '#161181', mb: 2 }} />
                    <Typography variant="h6" component="h3" sx={{ mb: 2, fontWeight: 600 }}>
                      Automated Planning
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Let AI handle complex route calculations, saving hours of manual planning work each week.
                    </Typography>
                  </CardContent>
                </FeatureCard>
              </motion.div>
            </Grid>

            {/* Feature 3 */}
            <Grid item xs={12} sm={6} md={3}>
            <motion.div
               initial="offscreen"
               animate="onscreen"
               variants={featureCardVariants}
             >
                <FeatureCard>
                  <CardContent sx={{ textAlign: 'center', p: 3 }}>
                    <StorefrontIcon sx={{ fontSize: 60, color: '#161181', mb: 2 }} />
                    <Typography variant="h6" component="h3" sx={{ mb: 2, fontWeight: 600 }}>
                      Store Management
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Easily organize store details, visit frequencies, and special requirements in one central place.
                    </Typography>
                  </CardContent>
                </FeatureCard>
              </motion.div>
            </Grid>

            {/* Feature 4 */}
            <Grid item xs={12} sm={6} md={3}>
            <motion.div
               initial="offscreen"
               animate="onscreen"
               variants={featureCardVariants}
             >
                <FeatureCard>
                  <CardContent sx={{ textAlign: 'center', p: 3 }}>
                    <InsightsIcon sx={{ fontSize: 60, color: '#161181', mb: 2 }} />
                    <Typography variant="h6" component="h3" sx={{ mb: 2, fontWeight: 600 }}>
                      Insights Dashboard
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Track performance metrics, analyze coverage patterns, and identify optimization opportunities.
                    </Typography>
                  </CardContent>
                </FeatureCard>
              </motion.div>
            </Grid>
          </Grid>
        </Container>
      </Box>

      {/* Benefits Section */}
      <Box sx={{ py: 10, backgroundColor: 'white' }}>
        <Container>
          <SectionTitle variant="h4" align="center">
            Why Choose Rahguzar?
          </SectionTitle>
          
          <Grid container spacing={8} alignItems="center">
            <Grid item xs={12} md={6}>
              <motion.img
                src="/benefits-image.png" // You'll need to replace this with your actual image
                alt="PJP Optimizer Benefits"
                style={{ width: '100%', height: 'auto', borderRadius: '12px' }}
                initial={{ opacity: 0, scale: 0.9 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ duration: 0.8 }}
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <List>
                <ListItem>
                  <ListItemIcon>
                    <CheckCircleIcon sx={{ color: '#161181' }} />
                  </ListItemIcon>
                  <ListItemText 
                    primary="30% More Efficient Than Manual Planning" 
                    secondary="Our algorithms create optimized routes that save time and reduce travel costs."
                  />
                </ListItem>
                
                <ListItem>
                  <ListItemIcon>
                    <CheckCircleIcon sx={{ color: '#161181' }} />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Increase Store Coverage by 25%" 
                    secondary="Visit more stores in less time with intelligent scheduling and route optimization."
                  />
                </ListItem>
                
                <ListItem>
                  <ListItemIcon>
                    <CheckCircleIcon sx={{ color: '#161181' }} />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Save 8+ Hours Per Week on Planning" 
                    secondary="Automated planning frees your team to focus on sales rather than logistics."
                  />
                </ListItem>
                
                <ListItem>
                  <ListItemIcon>
                    <CheckCircleIcon sx={{ color: '#161181' }} />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Data-Driven Decision Making" 
                    secondary="Gain insights into performance patterns to continuously improve your strategy."
                  />
                </ListItem>
                
                <ListItem>
                  <ListItemIcon>
                    <CheckCircleIcon sx={{ color: '#161181' }} />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Quick Implementation & Easy Adoption" 
                    secondary="User-friendly interface requires minimal training for your team to start seeing results."
                  />
                </ListItem>
              </List>
              
              <Box sx={{ mt: 4, textAlign: 'center' }}>
                <ActionButton 
                  variant="contained" 
                  size="large" 
                  component={RouterLink} 
                  to="/login"
                >
                  Start Optimizing Now
                </ActionButton>
              </Box>
            </Grid>
          </Grid>
        </Container>
      </Box>

      {/* Call to Action Section */}
      <Box sx={{ 
        py: 8, 
        backgroundColor: '#161181', 
        color: 'white',
        textAlign: 'center' 
      }}>
        <Container>
          <Typography variant="h4" component="h2" sx={{ mb: 3, fontWeight: 600 }}>
            Ready to Transform Your Sales Journey Planning?
          </Typography>
          <Typography variant="body1" sx={{ mb: 4, maxWidth: '800px', mx: 'auto', opacity: 0.9 }}>
            Join companies that have increased their sales efficiency and store coverage while reducing planning time.
          </Typography>
          <Button 
            variant="contained" 
            size="large"
            component={RouterLink}
            to="/login"
            sx={{ 
              bgcolor: 'white', 
              color: '#161181',
              px: 4,
              '&:hover': {
                bgcolor: '#f2f2f2',
              }
            }}
          >
            Get Started
          </Button>
        </Container>
      </Box>

     
    </AuthLayout>
  );
}

export default LandingPage;