import { BrowserRouter, Routes, Route, Link } from "react-router-dom";
import { ThemeProvider, createTheme } from "@mui/material/styles";
import CssBaseline from "@mui/material/CssBaseline";
import { Box, AppBar, Toolbar, Typography, Button, Container } from "@mui/material";
import { LocalHospital as HospitalIcon, Analytics as AnalyticsIcon } from "@mui/icons-material";
import Home from "./pages/Home";
import Predict from "./pages/Predict";

const theme = createTheme({
  palette: {
    primary: {
      main: '#2E7D32', // Medical green
      light: '#4CAF50',
      dark: '#1B5E20',
      contrastText: '#FFFFFF',
    },
    secondary: {
      main: '#1976D2', // Professional blue
      light: '#42A5F5',
      dark: '#0D47A1',
    },
    error: {
      main: '#D32F2F',
      light: '#EF5350',
      dark: '#C62828',
    },
    warning: {
      main: '#F57C00',
      light: '#FFB74D',
      dark: '#E65100',
    },
    success: {
      main: '#388E3C',
      light: '#66BB6A',
      dark: '#1B5E20',
    },
    background: {
      default: '#F8F9FA',
      paper: '#FFFFFF',
    },
    text: {
      primary: '#212121',
      secondary: '#757575',
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontWeight: 700,
      fontSize: '2.5rem',
      lineHeight: 1.2,
    },
    h2: {
      fontWeight: 600,
      fontSize: '2rem',
      lineHeight: 1.3,
    },
    h3: {
      fontWeight: 600,
      fontSize: '1.75rem',
      lineHeight: 1.4,
    },
    h4: {
      fontWeight: 600,
      fontSize: '1.5rem',
      lineHeight: 1.4,
    },
    h5: {
      fontWeight: 600,
      fontSize: '1.25rem',
      lineHeight: 1.5,
    },
    h6: {
      fontWeight: 600,
      fontSize: '1.125rem',
      lineHeight: 1.5,
    },
    body1: {
      fontSize: '1rem',
      lineHeight: 1.6,
    },
    body2: {
      fontSize: '0.875rem',
      lineHeight: 1.6,
    },
  },
  shape: {
    borderRadius: 12,
  },
  shadows: [
    'none',
    '0px 2px 4px rgba(0, 0, 0, 0.1)',
    '0px 4px 8px rgba(0, 0, 0, 0.12)',
    '0px 6px 16px rgba(0, 0, 0, 0.15)',
    '0px 8px 24px rgba(0, 0, 0, 0.18)',
    '0px 12px 32px rgba(0, 0, 0, 0.2)',
    '0px 16px 40px rgba(0, 0, 0, 0.22)',
    '0px 20px 48px rgba(0, 0, 0, 0.24)',
    '0px 24px 56px rgba(0, 0, 0, 0.26)',
    '0px 28px 64px rgba(0, 0, 0, 0.28)',
    '0px 32px 72px rgba(0, 0, 0, 0.3)',
    '0px 36px 80px rgba(0, 0, 0, 0.32)',
    '0px 40px 88px rgba(0, 0, 0, 0.34)',
    '0px 44px 96px rgba(0, 0, 0, 0.36)',
    '0px 48px 104px rgba(0, 0, 0, 0.38)',
    '0px 52px 112px rgba(0, 0, 0, 0.4)',
    '0px 56px 120px rgba(0, 0, 0, 0.42)',
    '0px 60px 128px rgba(0, 0, 0, 0.44)',
    '0px 64px 136px rgba(0, 0, 0, 0.46)',
    '0px 68px 144px rgba(0, 0, 0, 0.48)',
    '0px 72px 152px rgba(0, 0, 0, 0.5)',
    '0px 76px 160px rgba(0, 0, 0, 0.52)',
    '0px 80px 168px rgba(0, 0, 0, 0.54)',
    '0px 84px 176px rgba(0, 0, 0, 0.56)',
    '0px 88px 184px rgba(0, 0, 0, 0.58)',
  ],
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 600,
          borderRadius: 8,
          padding: '10px 24px',
          boxShadow: '0px 2px 4px rgba(0, 0, 0, 0.1)',
          '&:hover': {
            boxShadow: '0px 4px 8px rgba(0, 0, 0, 0.15)',
            transform: 'translateY(-1px)',
            transition: 'all 0.2s ease-in-out',
          },
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 16,
          boxShadow: '0px 4px 20px rgba(0, 0, 0, 0.08)',
          border: '1px solid rgba(0, 0, 0, 0.05)',
          transition: 'all 0.3s ease-in-out',
          '&:hover': {
            boxShadow: '0px 8px 30px rgba(0, 0, 0, 0.12)',
            transform: 'translateY(-2px)',
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          borderRadius: 16,
          boxShadow: '0px 4px 20px rgba(0, 0, 0, 0.08)',
        },
      },
    },
  },
});

export default function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <BrowserRouter>
        <Box sx={{ minHeight: '100vh', backgroundColor: 'background.default' }}>
          <AppBar 
            position="sticky" 
            elevation={0}
            sx={{ 
              backgroundColor: 'white',
              borderBottom: '1px solid',
              borderColor: 'divider',
              backdropFilter: 'blur(10px)',
            }}
          >
            <Toolbar sx={{ justifyContent: 'space-between', py: 1 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <HospitalIcon sx={{ color: 'primary.main', fontSize: 32 }} />
                <Typography 
                  variant="h5" 
                  component={Link} 
                  to="/"
                  sx={{ 
                    fontWeight: 700,
                    color: 'primary.main',
                    textDecoration: 'none',
                    '&:hover': { color: 'primary.dark' }
                  }}
                >
                  XAI Heart Risk
                </Typography>
              </Box>
              <Box sx={{ display: 'flex', gap: 2 }}>
                <Button
                  component={Link}
                  to="/"
                  startIcon={<AnalyticsIcon />}
                  sx={{ 
                    color: 'text.primary',
                    fontWeight: 500,
                    '&:hover': { 
                      backgroundColor: 'primary.light',
                      color: 'primary.contrastText'
                    }
                  }}
                >
                  Dashboard
                </Button>
                <Button
                  component={Link}
                  to="/predict"
                  variant="contained"
                  sx={{ 
                    px: 3,
                    py: 1,
                    borderRadius: 2,
                    background: 'linear-gradient(45deg, #2E7D32 30%, #4CAF50 90%)',
                    boxShadow: '0px 4px 12px rgba(46, 125, 50, 0.3)',
                    '&:hover': {
                      background: 'linear-gradient(45deg, #1B5E20 30%, #2E7D32 90%)',
                      boxShadow: '0px 6px 16px rgba(46, 125, 50, 0.4)',
                    }
                  }}
                >
                  Risk Assessment
                </Button>
              </Box>
            </Toolbar>
          </AppBar>
          
          <Container maxWidth="xl" sx={{ py: 4 }}>
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/predict" element={<Predict />} />
            </Routes>
          </Container>
        </Box>
      </BrowserRouter>
    </ThemeProvider>
  );
}
