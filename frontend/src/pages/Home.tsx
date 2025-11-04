import { 
  Box, 
  Typography, 
  Card, 
  CardContent, 
  Grid, 
  Button, 
  Container,
  Stack,
  Chip,
  Paper
} from "@mui/material";

import { 
  LocalHospital as HospitalIcon,
  Analytics as AnalyticsIcon,
  Psychology as PsychologyIcon,
  Assessment as AssessmentIcon,
  Security as SecurityIcon,
  Speed as SpeedIcon,
  ArrowForward as ArrowForwardIcon
} from "@mui/icons-material";
import { Link } from "react-router-dom";

export default function Home() {
  const features = [
    {
      icon: <PsychologyIcon sx={{ fontSize: 40, color: 'primary.main' }} />,
      title: "Explainable AI",
      description: "Transparent machine learning with SHAP explanations showing exactly why predictions are made."
    },
    {
      icon: <AssessmentIcon sx={{ fontSize: 40, color: 'secondary.main' }} />,
      title: "Multi-Model Ensemble",
      description: "Combines Random Forest, XGBoost, and Neural Networks for robust predictions."
    },
    {
      icon: <SecurityIcon sx={{ fontSize: 40, color: 'success.main' }} />,
      title: "Clinical Guidelines",
      description: "Evidence-based recommendations tailored to individual patient risk profiles."
    },
    {
      icon: <SpeedIcon sx={{ fontSize: 40, color: 'warning.main' }} />,
      title: "Real-time Analysis",
      description: "Instant risk assessment with comprehensive audit trails for quality assurance."
    }
  ];

    return (
    <Box>
      {/* Hero Section */}
      <Paper
        elevation={0}
        sx={{
          background: 'linear-gradient(135deg, #2E7D32 0%, #4CAF50 100%)',
          color: 'white',
          borderRadius: 4,
          p: 6,
          mb: 6,
          position: 'relative',
          overflow: 'hidden',
          '&::before': {
            content: '""',
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'url("data:image/svg+xml,%3Csvg width="60" height="60" viewBox="0 0 60 60" xmlns="http://www.w3.org/2000/svg"%3E%3Cg fill="none" fill-rule="evenodd"%3E%3Cg fill="%23ffffff" fill-opacity="0.05"%3E%3Ccircle cx="30" cy="30" r="4"/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")',
            opacity: 0.1,
          }
        }}
      >
        <Container maxWidth="lg">
          <Box sx={{ position: 'relative', zIndex: 1 }}>
            <Stack direction="row" alignItems="center" spacing={2} sx={{ mb: 3 }}>
              <HospitalIcon sx={{ fontSize: 48 }} />
              <Typography variant="h2" component="h1" sx={{ fontWeight: 700 }}>
                XAI Heart Risk Assessment
              </Typography>
            </Stack>
            <Typography variant="h5" sx={{ mb: 4, opacity: 0.9, maxWidth: '600px' }}>
              Advanced machine learning models with transparent explanations for clinical decision support in heart disease prediction.
            </Typography>
            <Stack direction="row" spacing={2} flexWrap="wrap" sx={{ mb: 4 }}>
              <Chip 
                label="Multi-Model Ensemble" 
                sx={{ 
                  backgroundColor: 'rgba(255,255,255,0.2)', 
                  color: 'white',
                  fontWeight: 500
                }} 
              />
              <Chip 
                label="SHAP Explanations" 
                sx={{ 
                  backgroundColor: 'rgba(255,255,255,0.2)', 
                  color: 'white',
                  fontWeight: 500
                }} 
              />
              <Chip 
                label="Clinical Guidelines" 
                sx={{ 
                  backgroundColor: 'rgba(255,255,255,0.2)', 
                  color: 'white',
                  fontWeight: 500
                }} 
              />
              <Chip 
                label="Real-time Analysis" 
                sx={{ 
                  backgroundColor: 'rgba(255,255,255,0.2)', 
                  color: 'white',
                  fontWeight: 500
                }} 
              />
            </Stack>
            <Button
              component={Link}
              to="/predict"
              variant="contained"
              size="large"
              endIcon={<ArrowForwardIcon />}
              sx={{
                backgroundColor: 'white',
                color: 'primary.main',
                px: 4,
                py: 2,
                fontSize: '1.1rem',
                fontWeight: 600,
                borderRadius: 3,
                boxShadow: '0px 8px 24px rgba(0,0,0,0.15)',
                '&:hover': {
                  backgroundColor: 'rgba(255,255,255,0.9)',
                  transform: 'translateY(-2px)',
                  boxShadow: '0px 12px 32px rgba(0,0,0,0.2)',
                },
                transition: 'all 0.3s ease-in-out',
              }}
            >
              Start Risk Assessment
            </Button>
          </Box>
        </Container>
      </Paper>

      {/* Features Section */}
      <Container maxWidth="lg">
        <Box sx={{ mb: 6 }}>
          <Typography variant="h3" component="h2" textAlign="center" sx={{ mb: 2, fontWeight: 600 }}>
            Advanced Clinical Decision Support
          </Typography>
          <Typography variant="h6" textAlign="center" color="text.secondary" sx={{ mb: 6, maxWidth: '600px', mx: 'auto' }}>
            Our system combines cutting-edge machine learning with transparent explanations to support healthcare professionals in making informed decisions.
          </Typography>
          
          <Grid container spacing={4}>
            {features.map((feature, index) => (
              <Grid xs={12} md={6} key={index}>
                <Card 
                  sx={{ 
                    height: '100%',
                    p: 3,
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    textAlign: 'center',
                    transition: 'all 0.3s ease-in-out',
                    '&:hover': {
                      transform: 'translateY(-8px)',
                      boxShadow: '0px 20px 40px rgba(0,0,0,0.1)',
                    }
                  }}
                >
                  <CardContent sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                    <Box sx={{ mb: 3 }}>
                      {feature.icon}
                    </Box>
                    <Typography variant="h5" component="h3" sx={{ mb: 2, fontWeight: 600 }}>
                      {feature.title}
                    </Typography>
                    <Typography variant="body1" color="text.secondary" sx={{ lineHeight: 1.6 }}>
                      {feature.description}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Box>

        {/* CTA Section */}
        <Paper
          elevation={0}
          sx={{
            background: 'linear-gradient(135deg, #F8F9FA 0%, #E3F2FD 100%)',
            p: 6,
            borderRadius: 4,
            textAlign: 'center',
            border: '1px solid',
            borderColor: 'divider',
          }}
        >
          <AnalyticsIcon sx={{ fontSize: 64, color: 'primary.main', mb: 3 }} />
          <Typography variant="h4" component="h2" sx={{ mb: 2, fontWeight: 600 }}>
            Ready to Get Started?
          </Typography>
          <Typography variant="h6" color="text.secondary" sx={{ mb: 4, maxWidth: '500px', mx: 'auto' }}>
            Experience the power of explainable AI in healthcare. Start with a comprehensive heart disease risk assessment.
          </Typography>
          <Button
            component={Link}
            to="/predict"
            variant="contained"
            size="large"
            endIcon={<ArrowForwardIcon />}
            sx={{
              px: 6,
              py: 2,
              fontSize: '1.1rem',
              fontWeight: 600,
              borderRadius: 3,
            }}
          >
            Begin Assessment
          </Button>
        </Paper>
      </Container>
    </Box>
    );
  }
  