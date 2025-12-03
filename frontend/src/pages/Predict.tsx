import { useEffect, useState } from "react";
import { predict, explain, fetchPdp, type PredictRequest, type PredictResponse, type ExplainResponse, type PdpIceResponse } from "../services/api";
import {
  Box,
  Paper,
  Typography,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Button,
  Card,
  CardContent,
  Alert,
  FormControlLabel,
  Radio,
  RadioGroup,
  Chip,
  Stack,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Tabs,
  Tab,
  CircularProgress,
} from "@mui/material";
import {
  CheckCircle as CheckCircleIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
  TrendingUp as TrendingUpIcon,
  LocalHospital,
} from "@mui/icons-material";

// Risk Stratification Types and Utilities
type RiskCategory = 'Low' | 'Moderate' | 'High';

interface RiskAssessment {
  level: RiskCategory;
  color: 'success' | 'warning' | 'error';
  recommendation: string;
  icon: React.ReactNode;
  description: string;
}

const getRiskCategory = (probability: number): RiskAssessment => {
  if (probability < 0.3) {
    return {
      level: 'Low',
      color: 'success',
      recommendation: 'Continue routine monitoring',
      icon: <CheckCircleIcon />,
      description: 'Low risk of heart disease. Maintain healthy lifestyle and regular checkups.'
    };
  } else if (probability < 0.6) {
    return {
      level: 'Moderate',
      color: 'warning',
      recommendation: 'Consider additional testing',
      icon: <WarningIcon />,
      description: 'Moderate risk detected. Consider further evaluation and lifestyle modifications.'
    };
  } else {
    return {
      level: 'High',
      color: 'error',
      recommendation: 'Immediate clinical evaluation recommended',
      icon: <ErrorIcon />,
      description: 'High risk of heart disease. Urgent medical attention and comprehensive evaluation needed.'
    };
  }
};



// Clinical Guidelines
const getClinicalGuidelines = (riskCategory: RiskCategory, features: PredictRequest): string[] => {
  const baseGuidelines = {
    'Low': [
      'Continue routine primary care monitoring',
      'Maintain healthy lifestyle (diet, exercise, no smoking)',
      'Annual cardiovascular risk assessment',
      'Monitor blood pressure and cholesterol regularly'
    ],
    'Moderate': [
      'Schedule follow-up within 1-2 weeks',
      'Consider ECG and basic cardiac workup',
      'Discuss lifestyle modifications with patient',
      'Review current medications and family history',
      'Consider stress testing if symptoms persist'
    ],
    'High': [
      'Refer to cardiologist within 24-48 hours',
      'Consider immediate stress testing or cardiac catheterization',
      'Review and optimize current medications',
      'Implement aggressive lifestyle modifications',
      'Consider admission for observation if symptoms severe',
      'Discuss advanced imaging (CT angiography, MRI)'
    ]
  };

  // Add specific guidelines based on patient features
  const specificGuidelines: string[] = [];
  
  if (features.age > 65) {
    specificGuidelines.push('Consider age-appropriate cardiac screening protocols');
  }
  
  if (features.cholesterol > 240) {
    specificGuidelines.push('Discuss lipid-lowering therapy options');
  }
  
  if (features.max_hr < 100) {
    specificGuidelines.push('Evaluate for potential cardiac dysfunction');
  }
  
  if (features.chest_pain_type === 'TA') {
    specificGuidelines.push('Classic angina pattern - high suspicion for CAD');
  }
  
  if (features.exercise_angina === 'Y') {
    specificGuidelines.push('Exercise-induced symptoms require immediate evaluation');
  }

  return [...baseGuidelines[riskCategory], ...specificGuidelines];
};

// Patient Report Generation
interface PatientReport {
  patientId: string;
  timestamp: string;
  riskAssessment: RiskAssessment;
  predictions: PredictResponse['predictions'];
  keyFindings: string[];
  recommendations: string[];
  modelAgreement: number;
  nextSteps: string[];
  patientData: PredictRequest;
}

const generatePatientId = (): string => {
  return `PAT-${Date.now().toString(36).toUpperCase()}`;
};

const extractKeyFindings = (explanations: ExplainResponse | null): string[] => {
  if (!explanations?.contributions) return [];
  
  const findings: string[] = [];
  const rfContributions = explanations.contributions.random_forest;
  
  // Get top 3 contributing factors from SHAP (preferred) or LIME
  const topFactors = (rfContributions?.shap || rfContributions?.lime || []).slice(0, 3);
  
  topFactors.forEach((factor: { feature: string; value: number; impact: string }) => {
    const impact = factor.impact === 'Increases Risk' ? 'increases' : 'decreases';
    findings.push(`${factor.feature} (${factor.value}) ${impact} heart disease risk`);
  });
  
  return findings;
};

const generateNextSteps = (riskCategory: RiskCategory): string[] => {
  const nextSteps = {
    'Low': [
      'Continue current lifestyle and monitoring',
      'Schedule annual checkup',
      'Maintain healthy diet and exercise routine'
    ],
    'Moderate': [
      'Schedule follow-up appointment within 2 weeks',
      'Consider additional cardiac testing',
      'Implement recommended lifestyle changes',
      'Monitor symptoms closely'
    ],
    'High': [
      'Seek immediate medical attention',
      'Contact cardiologist for urgent evaluation',
      'Consider emergency department if symptoms worsen',
      'Prepare for comprehensive cardiac workup'
    ]
  };
  
  return nextSteps[riskCategory];
};

const generatePatientReport = (
  result: PredictResponse,
  explanations: ExplainResponse | null,
  form: PredictRequest
): PatientReport => {
  const avgRisk = result.risk_assessment?.average_probability || 0;
  const riskAssessment = getRiskCategory(avgRisk);
  
  return {
    patientId: generatePatientId(),
    timestamp: new Date().toISOString(),
    riskAssessment,
    predictions: result.predictions!,
    keyFindings: extractKeyFindings(explanations),
    recommendations: getClinicalGuidelines(result.risk_assessment?.category || 'Low', form),
    modelAgreement: result.risk_assessment?.model_agreement || 0,
    nextSteps: generateNextSteps(result.risk_assessment?.category || 'Low'),
    patientData: form
  };
};

const exportPatientReport = (report: PatientReport) => {
  const reportText = `
HEART DISEASE RISK ASSESSMENT REPORT
=====================================

Patient ID: ${report.patientId}
Assessment Date: ${new Date(report.timestamp).toLocaleString()}

RISK ASSESSMENT
---------------
Risk Level: ${report.riskAssessment.level}
Probability: ${(report.riskAssessment.level === 'Low' ? 0.2 : report.riskAssessment.level === 'Moderate' ? 0.45 : 0.7) * 100}%
Recommendation: ${report.riskAssessment.recommendation}

MODEL PREDICTIONS
-----------------
Random Forest: ${report.predictions ? (report.predictions.random_forest.probability * 100).toFixed(1) : 'N/A'}%
XGBoost: ${report.predictions ? (report.predictions.xgboost.probability * 100).toFixed(1) : 'N/A'}%
Neural Network: ${report.predictions ? (report.predictions.neural_net.probability * 100).toFixed(1) : 'N/A'}%
Model Agreement: ${(report.modelAgreement * 100).toFixed(1)}%

KEY FINDINGS
------------
${report.keyFindings.map(finding => `• ${finding}`).join('\n')}

CLINICAL RECOMMENDATIONS
------------------------
${report.recommendations.map(rec => `• ${rec}`).join('\n')}

NEXT STEPS
----------
${report.nextSteps.map(step => `• ${step}`).join('\n')}

PATIENT DATA
------------
Age: ${report.patientData.age} years
Sex: ${report.patientData.sex}
Chest Pain Type: ${report.patientData.chest_pain_type}
Cholesterol: ${report.patientData.cholesterol} mg/dL
Fasting Blood Sugar: ${report.patientData.fasting_bs}
Max Heart Rate: ${report.patientData.max_hr} bpm
Exercise Angina: ${report.patientData.exercise_angina}
Oldpeak: ${report.patientData.oldpeak}
ST Slope: ${report.patientData.st_slope}

---
Generated by XAI Heart Disease Prediction System
This report is for clinical decision support only.
  `.trim();
  
  const blob = new Blob([reportText], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `heart-risk-report-${report.patientId}.txt`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
};

const MODEL_LABELS: Record<"rf" | "xgb" | "nn", string> = {
  rf: "Random Forest",
  xgb: "XGBoost",
  nn: "Neural Network"
};

const DEFAULT_FEATURE_OPTIONS: string[] = ['Age', 'Sex', 'ChestPainType', 'Cholesterol', 'FastingBS', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope'];
const PDP_CATEGORICAL = new Set(['Sex', 'ChestPainType', 'FastingBS', 'ExerciseAngina', 'ST_Slope']);

type PdpChartPayload = {
  grid: number[];
  grid_labels: string[];
  pdp: number[];
  ice: number[][];
  feature: string;
  feature_type: "numeric" | "categorical";
  model: "rf" | "xgb" | "nn";
};

const hasPdpPayload = (payload: PdpIceResponse | null): payload is PdpIceResponse & PdpChartPayload => {
  return Boolean(
    payload &&
    payload.grid &&
    payload.grid_labels &&
    payload.pdp &&
    payload.ice &&
    payload.feature &&
    payload.feature_type &&
    payload.model
  );
};

const PdpIceChart = ({ data }: { data: PdpChartPayload }) => {
  const width = 900;
  const height = 500;
  const leftPadding = 70;
  const rightPadding = 48;
  const topPadding = 48;
  const bottomPadding = 60;
  const plotWidth = width - leftPadding - rightPadding;
  const plotHeight = height - topPadding - bottomPadding;
  const flatIce = data.ice.flat();
  const yValues = [...data.pdp, ...flatIce];
  const minY = Math.min(0, ...yValues);
  const maxY = Math.max(1, ...yValues);
  const yRange = maxY - minY || 1;
  const gridMin = Math.min(...data.grid);
  const gridMax = Math.max(...data.grid);
  const gridRange = gridMax - gridMin || 1;
  const maxIceLines = Math.min(data.ice.length, 12);

  // Generate y-axis ticks
  const numYTicks = 6;
  const yTicks = Array.from({ length: numYTicks }, (_, i) => {
    const value = minY + (yRange * i) / (numYTicks - 1);
    return {
      value: value,
      label: value.toFixed(2),
      y: height - bottomPadding - (plotHeight * i) / (numYTicks - 1)
    };
  });

  const getXCoord = (value: number, index: number) => {
    if (data.feature_type === "categorical") {
      const denom = data.grid.length > 1 ? data.grid.length - 1 : 1;
      return leftPadding + (index / denom) * plotWidth;
    }
    const ratio = gridRange === 0 ? 0.5 : (value - gridMin) / gridRange;
    return leftPadding + ratio * plotWidth;
  };

  const getYCoord = (value: number) => {
    const ratio = (value - minY) / yRange;
    return height - bottomPadding - ratio * plotHeight;
  };

  const seriesToPath = (series: number[]) =>
    series
      .map((value, idx) => {
        const x = getXCoord(data.grid[idx], idx);
        const y = getYCoord(value);
        return `${idx === 0 ? "M" : "L"}${x} ${y}`;
      })
      .join(" ");

  const pdpPath = seriesToPath(data.pdp);
  const icePaths = data.ice.slice(0, maxIceLines).map((curve) => seriesToPath(curve));
  const xTicks = data.grid.map((value, idx) => ({
    x: getXCoord(value, idx),
    label: data.grid_labels[idx] ?? value.toString()
  }));

  return (
    <Box>
      <svg viewBox={`0 0 ${width} ${height}`} width="100%" height="500px">
        <rect
          x={leftPadding}
          y={topPadding}
          width={plotWidth}
          height={plotHeight}
          fill="transparent"
          stroke="#e0e0e0"
        />
        {icePaths.map((path, idx) => (
          <path
            key={`ice-${idx}`}
            d={path}
            fill="none"
            stroke="rgba(33, 150, 243, 0.25)"
            strokeWidth={1}
          />
        ))}
        <path d={pdpPath} fill="none" stroke="#d32f2f" strokeWidth={3} />
        <line
          x1={leftPadding}
          y1={height - bottomPadding}
          x2={leftPadding + plotWidth}
          y2={height - bottomPadding}
          stroke="#9e9e9e"
          strokeWidth={1.5}
        />
        <line 
          x1={leftPadding} 
          y1={topPadding} 
          x2={leftPadding} 
          y2={height - bottomPadding} 
          stroke="#9e9e9e"
          strokeWidth={1.5}
        />
        {yTicks.map((tick, idx) => (
          <g key={`y-tick-${idx}`}>
            <line
              x1={leftPadding - 6}
              x2={leftPadding}
              y1={tick.y}
              y2={tick.y}
              stroke="#9e9e9e"
            />
            <text
              x={leftPadding - 12}
              y={tick.y}
              textAnchor="end"
              fontSize={11}
              fill="#616161"
              dominantBaseline="middle"
            >
              {tick.label}
            </text>
          </g>
        ))}
        {xTicks.map((tick, idx) => (
          <g key={`tick-${idx}`}>
            <line
              x1={tick.x}
              x2={tick.x}
              y1={height - bottomPadding}
              y2={height - bottomPadding + 6}
              stroke="#9e9e9e"
            />
            <text
              x={tick.x}
              y={height - bottomPadding + 22}
              textAnchor="middle"
              fontSize={11}
              fill="#616161"
            >
              {tick.label}
            </text>
          </g>
        ))}
        <text
          x={leftPadding + plotWidth / 2}
          y={height - 12}
          textAnchor="middle"
          fontSize={13}
          fill="#424242"
          fontWeight={500}
        >
          {data.feature}
        </text>
        <text
          transform={`rotate(-90 ${24} ${topPadding + plotHeight / 2})`}
          x={24}
          y={topPadding + plotHeight / 2}
          textAnchor="middle"
          fontSize={13}
          fill="#424242"
          fontWeight={500}
        >
          P(Heart Disease)
        </text>
      </svg>
      <Stack direction="row" spacing={2} sx={{ mt: 1, flexWrap: "wrap" }}>
        <Chip size="small" label="Partial Dependence" sx={{ bgcolor: '#ffebee', color: '#c62828' }} />
        <Chip size="small" label={`ICE samples (${maxIceLines} shown)`} sx={{ bgcolor: '#e3f2fd', color: '#1565c0' }} />
        <Chip size="small" label={MODEL_LABELS[data.model]} />
      </Stack>
      <Typography variant="caption" color="text.secondary" sx={{ display: "block", mt: 1 }}>
        PDP shows the average effect as the red curve. Lighter blue lines show individual conditional
        expectations from representative patients.
      </Typography>
    </Box>
  );
};

export default function PredictionForm() {
  const [form, setForm] = useState<PredictRequest>({
    age: 50,
    sex: "M",
    chest_pain_type: "ATA",
    cholesterol: 200,
    fasting_bs: 0,
    max_hr: 150,
    exercise_angina: "N",
    oldpeak: 1,
    st_slope: "Up",
  });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [explanations, setExplanations] = useState<ExplainResponse | null>(null);
  const [modelForExplain, setModelForExplain] = useState<"rf" | "xgb" | "nn">("rf");
  const [explanationMethod, setExplanationMethod] = useState<"shap" | "lime">("shap");
  const [pdpModel, setPdpModel] = useState<"rf" | "xgb" | "nn">("rf");
  const [pdpFeature, setPdpFeature] = useState<string>("Age");
  const [pdpData, setPdpData] = useState<PdpIceResponse | null>(null);
  const [pdpLoading, setPdpLoading] = useState(false);
  const [pdpError, setPdpError] = useState<string | null>(null);

  function update<K extends keyof PredictRequest>(key: K, value: PredictRequest[K]) {
    setForm((prev) => ({ ...prev, [key]: value }));
  }

  function validate(values: PredictRequest): string | null {
    if (values.age < 1 || values.age > 120) return "Age must be between 1 and 120";
    if (values.cholesterol < 50 || values.cholesterol > 700) return "Cholesterol looks out of range";
    if (values.max_hr < 40 || values.max_hr > 220) return "Max HR looks out of range";
    if (values.oldpeak < 0 || values.oldpeak > 10) return "Oldpeak must be between 0 and 10";
    return null;
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    setExplanations(null);
    setPdpData(null);
    setPdpError(null);
    try {
      const validationError = validate(form);
      if (validationError) throw new Error(validationError);
      const res = await predict(form);
      if (!res.ok) throw new Error(res.error || "Prediction failed");
      setResult(res);
      console.log('result', res);
      // Fire and await explanations
      const exp = await explain(form);
      if (exp.ok) setExplanations(exp);
      console.log('explanations', exp);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Unexpected error";
      setError(message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (explanations?.feature_names && explanations.feature_names.length > 0) {
      if (!explanations.feature_names.includes(pdpFeature)) {
        setPdpFeature(explanations.feature_names[0]);
      }
    }
  }, [explanations, pdpFeature]);

  const featureOptions = explanations?.feature_names ?? DEFAULT_FEATURE_OPTIONS;
  const isFeatureCategorical = PDP_CATEGORICAL.has(pdpFeature);

  const handleFetchPdp = async () => {
    if (!result) {
      setPdpError("Run a prediction first to unlock partial dependence plots.");
      return;
    }
    try {
      setPdpLoading(true);
      setPdpError(null);
      const response = await fetchPdp({
        feature: pdpFeature,
        model: pdpModel,
        grid_resolution: isFeatureCategorical ? undefined : 25,
        ice_count: 20
      });
      if (!response.ok || !hasPdpPayload(response)) {
        throw new Error(response.error || "Unable to generate PDP/ICE data");
      }
      setPdpData(response);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to generate PDP/ICE curves";
      setPdpError(message);
      setPdpData(null);
    } finally {
      setPdpLoading(false);
    }
  };

  return (
    <Box>
      {/* Hero Section */}
      <Paper
        elevation={0}
        sx={{
          background: 'linear-gradient(135deg, #E8F5E8 0%, #F1F8E9 100%)',
          p: 4,
          mb: 4,
          borderRadius: 3,
          border: '1px solid',
          borderColor: 'primary.light',
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
          <LocalHospital sx={{ fontSize: 40, color: 'primary.main' }} />
          <Typography variant="h4" component="h1" sx={{ fontWeight: 700, color: 'primary.main' }}>
            Heart Disease Risk Assessment
          </Typography>
        </Box>
        <Typography variant="h6" color="text.secondary" sx={{ maxWidth: '600px' }}>
          Enter patient information below to receive comprehensive risk assessment with explainable AI insights and clinical recommendations.
        </Typography>
      </Paper>
      
      {/* Form Section */}
      <Paper 
        elevation={0} 
        sx={{ 
          p: 4, 
          mb: 4,
          borderRadius: 3,
          border: '1px solid',
          borderColor: 'divider',
          background: 'linear-gradient(135deg, #FFFFFF 0%, #FAFAFA 100%)',
        }}
      >
        <form onSubmit={handleSubmit}>
          <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr' }, gap: 3 }}>
            <TextField
              fullWidth
              label="Age (years)"
              type="number"
              value={form.age}
              onChange={(e) => update("age", Number(e.target.value))}
              inputProps={{ min: 1, max: 120 }}
              helperText="1 – 120"
              required
            />

            <FormControl fullWidth required>
              <InputLabel>Sex</InputLabel>
              <Select
                value={form.sex}
                onChange={(e) => update("sex", e.target.value as PredictRequest["sex"])}
                label="Sex"
              >
                <MenuItem value="M">Male</MenuItem>
                <MenuItem value="F">Female</MenuItem>
              </Select>
            </FormControl>

            <FormControl fullWidth required>
              <InputLabel>Chest Pain Type</InputLabel>
              <Select
                value={form.chest_pain_type}
                onChange={(e) => update("chest_pain_type", e.target.value as PredictRequest["chest_pain_type"])}
                label="Chest Pain Type"
              >
                <MenuItem value="TA">Typical Angina</MenuItem>
                <MenuItem value="ATA">Atypical Angina</MenuItem>
                <MenuItem value="NAP">Non-Anginal Pain</MenuItem>
                <MenuItem value="ASY">Asymptomatic</MenuItem>
              </Select>
            </FormControl>

            <TextField
              fullWidth
              label="Cholesterol (mg/dL)"
              type="number"
              value={form.cholesterol}
              onChange={(e) => update("cholesterol", Number(e.target.value))}
              inputProps={{ min: 0, max: 700 }}
              helperText="0 – 700 mg/dL"
              required
            />

            <FormControl fullWidth required>
              <InputLabel>Fasting Blood Sugar ≥ 120 mg/dL</InputLabel>
              <Select
                value={form.fasting_bs}
                onChange={(e) => update("fasting_bs", Number(e.target.value) as 0 | 1)}
                label="Fasting Blood Sugar ≥ 120 mg/dL"
              >
                <MenuItem value={0}>No</MenuItem>
                <MenuItem value={1}>Yes</MenuItem>
              </Select>
            </FormControl>

            <TextField
              fullWidth
              label="Max Heart Rate"
              type="number"
              value={form.max_hr}
              onChange={(e) => update("max_hr", Number(e.target.value))}
              inputProps={{ min: 60, max: 202 }}
              helperText="60 – 202"
              required
            />

            <FormControl fullWidth required>
              <InputLabel>Exercise Angina</InputLabel>
              <Select
                value={form.exercise_angina}
                onChange={(e) => update("exercise_angina", e.target.value as PredictRequest["exercise_angina"])}
                label="Exercise Angina"
              >
                <MenuItem value="N">No</MenuItem>
                <MenuItem value="Y">Yes</MenuItem>
              </Select>
            </FormControl>

            <TextField
              fullWidth
              label="Oldpeak (ST depression)"
              type="number"
              value={form.oldpeak}
              onChange={(e) => update("oldpeak", Number(e.target.value))}
              inputProps={{ min: 0, max: 10, step: 0.1 }}
              helperText="0 – 10"
              required
            />

            <FormControl fullWidth required>
              <InputLabel>ST Slope</InputLabel>
              <Select
                value={form.st_slope}
                onChange={(e) => update("st_slope", e.target.value as PredictRequest["st_slope"])}
                label="ST Slope"
              >
                <MenuItem value="Up">Up</MenuItem>
                <MenuItem value="Flat">Flat</MenuItem>
                <MenuItem value="Down">Down</MenuItem>
              </Select>
            </FormControl>
          </Box>

          <Box sx={{ mt: 3, display: 'flex', alignItems: 'center', gap: 2 }}>
            <Button 
              type="submit" 
              variant="contained" 
              disabled={loading}
              size="large"
              sx={{
                background: 'linear-gradient(45deg, #2E7D32 30%, #4CAF50 90%)',
                px: 6,
                py: 2,
                fontSize: '1.1rem',
                fontWeight: 600,
                borderRadius: 3,
                boxShadow: '0px 4px 12px rgba(46, 125, 50, 0.3)',
                '&:hover': {
                  background: 'linear-gradient(45deg, #1B5E20 30%, #2E7D32 90%)',
                  boxShadow: '0px 6px 16px rgba(46, 125, 50, 0.4)',
                  transform: 'translateY(-1px)',
                },
                '&:disabled': {
                  background: 'linear-gradient(45deg, #BDBDBD 30%, #E0E0E0 90%)',
                  color: 'white',
                },
                transition: 'all 0.3s ease-in-out',
              }}
            >
              {loading ? "Analyzing..." : "🔍 Analyze Risk"}
            </Button>
            <Typography variant="body2" color="text.secondary">
              Predictions and explanations will appear below
            </Typography>
          </Box>
        </form>

        {error && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {error}
          </Alert>
        )}
      </Paper>





      {result?.predictions && (
        <Box sx={{ mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom sx={{ mb: 3, fontWeight: 600 }}>
            Model Predictions
          </Typography>
          <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr', md: '1fr 1fr 1fr 1fr' }, gap: 3 }}>
            <Card sx={{ 
              background: 'linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%)',
              border: '1px solid',
              borderColor: 'secondary.light',
            }}>
              <CardContent sx={{ textAlign: 'center' }}>
                <Typography variant="h6" color="secondary.main" gutterBottom sx={{ fontWeight: 600 }}>
                  Random Forest
                </Typography>
                <Typography variant="h3" component="div" sx={{ fontWeight: 700, color: 'secondary.dark', mb: 1 }}>
                  {(result.predictions.random_forest.probability * 100).toFixed(1)}%
                </Typography>
                <Chip 
                  label={result.predictions.random_forest.label === 1 ? 'High Risk' : 'Low Risk'}
                  color={result.predictions.random_forest.label === 1 ? 'error' : 'success'}
                  size="small"
                />
              </CardContent>
            </Card>
            <Card sx={{ 
              background: 'linear-gradient(135deg, #E8F5E8 0%, #C8E6C9 100%)',
              border: '1px solid',
              borderColor: 'success.light',
            }}>
              <CardContent sx={{ textAlign: 'center' }}>
                <Typography variant="h6" color="success.main" gutterBottom sx={{ fontWeight: 600 }}>
                  XGBoost
                </Typography>
                <Typography variant="h3" component="div" sx={{ fontWeight: 700, color: 'success.dark', mb: 1 }}>
                  {(result.predictions.xgboost.probability * 100).toFixed(1)}%
                </Typography>
                <Chip 
                  label={result.predictions.xgboost.label === 1 ? 'High Risk' : 'Low Risk'}
                  color={result.predictions.xgboost.label === 1 ? 'error' : 'success'}
                  size="small"
                />
              </CardContent>
            </Card>
            <Card sx={{ 
              background: 'linear-gradient(135deg, #FFF3E0 0%, #FFE0B2 100%)',
              border: '1px solid',
              borderColor: 'warning.light',
            }}>
              <CardContent sx={{ textAlign: 'center' }}>
                <Typography variant="h6" color="warning.main" gutterBottom sx={{ fontWeight: 600 }}>
                  Neural Network
                </Typography>
                <Typography variant="h3" component="div" sx={{ fontWeight: 700, color: 'warning.dark', mb: 1 }}>
                  {(result.predictions.neural_net.probability * 100).toFixed(1)}%
                </Typography>
                <Chip 
                  label={result.predictions.neural_net.label === 1 ? 'High Risk' : 'Low Risk'}
                  color={result.predictions.neural_net.label === 1 ? 'error' : 'success'}
                  size="small"
                />
              </CardContent>
            </Card>
            <Card sx={{ 
              background: 'linear-gradient(135deg, #2E7D32 0%, #4CAF50 100%)',
              color: 'white',
              position: 'relative',
              overflow: 'hidden',
              '&::before': {
                content: '""',
                position: 'absolute',
                top: 0,
                right: 0,
                width: '100px',
                height: '100px',
                background: 'rgba(255,255,255,0.1)',
                borderRadius: '50%',
                transform: 'translate(30px, -30px)',
              }
            }}>
              <CardContent sx={{ textAlign: 'center', position: 'relative', zIndex: 1 }}>
                <Typography variant="h6" gutterBottom sx={{ fontWeight: 600, opacity: 0.9 }}>
                  Ensemble Average
                </Typography>
                <Typography variant="h3" component="div" sx={{ fontWeight: 700, mb: 1 }}>
                  {(((result.predictions.random_forest.probability + result.predictions.xgboost.probability + result.predictions.neural_net.probability) / 3) * 100).toFixed(1)}%
                </Typography>
                <Chip 
                  label="Combined Prediction"
                  sx={{ 
                    backgroundColor: 'rgba(255,255,255,0.2)',
                    color: 'white',
                    fontWeight: 500
                  }}
                  size="small"
                />
              </CardContent>
            </Card>
          </Box>

          {/* Risk Stratification Section */}
          {result?.predictions && (
            <>
              <Box sx={{ mb: 4 }}>
                <Typography variant="h5" component="h2" gutterBottom sx={{ mb: 3, fontWeight: 600 }}>
                  Risk Assessment
                </Typography>
                
                <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' }, gap: 3 }}>
                  {/* Overall Risk Assessment */}
                  <Card sx={{ 
                    border: 3, 
                    borderColor: `${getRiskCategory(result.risk_assessment?.average_probability || 0).color}.main`,
                    background: `linear-gradient(135deg, ${getRiskCategory(result.risk_assessment?.average_probability || 0).color === 'success' ? '#E8F5E8' : getRiskCategory(result.risk_assessment?.average_probability || 0).color === 'warning' ? '#FFF8E1' : '#FFEBEE'} 0%, ${getRiskCategory(result.risk_assessment?.average_probability || 0).color === 'success' ? '#C8E6C9' : getRiskCategory(result.risk_assessment?.average_probability || 0).color === 'warning' ? '#FFE0B2' : '#FFCDD2'} 100%)`,
                    position: 'relative',
                    overflow: 'hidden',
                    '&::before': {
                      content: '""',
                      position: 'absolute',
                      top: 0,
                      right: 0,
                      width: '120px',
                      height: '120px',
                      background: `linear-gradient(45deg, ${getRiskCategory(result.risk_assessment?.average_probability || 0).color === 'success' ? 'rgba(76, 175, 80, 0.1)' : getRiskCategory(result.risk_assessment?.average_probability || 0).color === 'warning' ? 'rgba(255, 193, 7, 0.1)' : 'rgba(244, 67, 54, 0.1)'}, transparent)`,
                      borderRadius: '50%',
                      transform: 'translate(40px, -40px)',
                    }
                  }}>
                    <CardContent sx={{ position: 'relative', zIndex: 1 }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                        <Box sx={{ 
                          p: 2, 
                          borderRadius: '50%', 
                          backgroundColor: `${getRiskCategory(result.risk_assessment?.average_probability || 0).color}.main`,
                          color: 'white',
                          mr: 2
                        }}>
                          {getRiskCategory(result.risk_assessment?.average_probability || 0).icon}
                        </Box>
                        <Typography variant="h4" component="div" sx={{ fontWeight: 700, color: `${getRiskCategory(result.risk_assessment?.average_probability || 0).color}.dark` }}>
                          {result.risk_assessment?.category || 'Unknown'} Risk
                        </Typography>
                      </Box>
                      <Typography variant="h2" component="div" sx={{ fontWeight: 800, mb: 2, color: `${getRiskCategory(result.risk_assessment?.average_probability || 0).color}.dark` }}>
                        {((result.risk_assessment?.average_probability || 0) * 100).toFixed(1)}%
                      </Typography>
                      <Typography variant="body1" sx={{ mb: 3, color: 'text.secondary', lineHeight: 1.6 }}>
                        {getRiskCategory(result.risk_assessment?.average_probability || 0).description}
                      </Typography>
                      <Alert 
                        severity={getRiskCategory(result.risk_assessment?.average_probability || 0).color} 
                        sx={{ 
                          mt: 2,
                          borderRadius: 2,
                          '& .MuiAlert-message': {
                            fontWeight: 600
                          }
                        }}
                      >
                        <Typography variant="body2" sx={{ fontWeight: 600 }}>
                          Recommendation: {getRiskCategory(result.risk_assessment?.average_probability || 0).recommendation}
                        </Typography>
                      </Alert>
                    </CardContent>
                  </Card>

                  {/* Model Agreement Analysis */}
                  <Card sx={{
                    background: 'linear-gradient(135deg, #F3E5F5 0%, #E1BEE7 100%)',
                    border: '1px solid',
                    borderColor: 'secondary.light',
                  }}>
                    <CardContent>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                        <Box sx={{ 
                          p: 2, 
                          borderRadius: '50%', 
                          backgroundColor: 'secondary.main',
                          color: 'white',
                          mr: 2
                        }}>
                          <TrendingUpIcon />
                        </Box>
                        <Typography variant="h6" component="div" sx={{ fontWeight: 600, color: 'secondary.dark' }}>
                          Model Agreement
                        </Typography>
                      </Box>
                      <Typography variant="h2" component="div" sx={{ fontWeight: 800, mb: 2, color: 'secondary.dark' }}>
                        {((result.risk_assessment?.model_agreement || 0) * 100).toFixed(1)}%
                      </Typography>
                      <Typography variant="body2" color="text.secondary" sx={{ mb: 3, lineHeight: 1.6 }}>
                        {(result.risk_assessment?.model_agreement || 0) > 0.8 
                          ? 'High agreement between models - prediction is reliable'
                          : 'Moderate agreement - consider additional validation'
                        }
                      </Typography>
                      <Chip 
                        label={(result.risk_assessment?.model_agreement || 0) > 0.8 ? 'High Confidence' : 'Moderate Confidence'}
                        color={(result.risk_assessment?.model_agreement || 0) > 0.8 ? 'success' : 'warning'}
                        size="medium"
                        sx={{ fontWeight: 600 }}
                      />
                    </CardContent>
                  </Card>
                </Box>
              </Box>

              {/* Clinical Guidelines Section */}
              <Paper sx={{ 
                p: 4, 
                mb: 4,
                background: 'linear-gradient(135deg, #F8F9FA 0%, #E3F2FD 100%)',
                border: '1px solid',
                borderColor: 'primary.light',
              }}>
                <Typography variant="h5" component="h2" gutterBottom sx={{ mb: 4, fontWeight: 600, color: 'primary.dark' }}>
                  Clinical Recommendations
                </Typography>
                
                <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' }, gap: 3 }}>
                  {/* Evidence-Based Guidelines */}
                  <Box>
                    <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                      <CheckCircleIcon color="primary" sx={{ mr: 1 }} />
                      Evidence-Based Guidelines
                    </Typography>
                    <List dense>
                      {getClinicalGuidelines(
                        result.risk_assessment?.category || 'Low', 
                        form
                      ).map((guideline, index) => (
                        <ListItem key={index} sx={{ py: 0.5 }}>
                          <ListItemIcon sx={{ minWidth: 32 }}>
                            <CheckCircleIcon color="primary" fontSize="small" />
                          </ListItemIcon>
                          <ListItemText 
                            primary={guideline}
                            primaryTypographyProps={{ variant: 'body2' }}
                          />
                        </ListItem>
                      ))}
                    </List>
                  </Box>

                  {/* Key Risk Factors */}
                  <Box>
                    <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                      <WarningIcon color="warning" sx={{ mr: 1 }} />
                      Key Risk Factors
                    </Typography>
                    <Stack spacing={1}>
                      <Chip 
                        label={`Age: ${form.age} years`}
                        color={form.age > 65 ? 'error' : form.age > 50 ? 'warning' : 'success'}
                        size="small"
                      />
                      <Chip 
                        label={`Cholesterol: ${form.cholesterol} mg/dL`}
                        color={form.cholesterol > 240 ? 'error' : form.cholesterol > 200 ? 'warning' : 'success'}
                        size="small"
                      />
                      <Chip 
                        label={`Max HR: ${form.max_hr} bpm`}
                        color={form.max_hr < 100 ? 'error' : form.max_hr < 120 ? 'warning' : 'success'}
                        size="small"
                      />
                      <Chip 
                        label={`Chest Pain: ${form.chest_pain_type}`}
                        color={form.chest_pain_type === 'TA' ? 'error' : form.chest_pain_type === 'ATA' ? 'warning' : 'success'}
                        size="small"
                      />
                      <Chip 
                        label={`Exercise Angina: ${form.exercise_angina}`}
                        color={form.exercise_angina === 'Y' ? 'error' : 'success'}
                        size="small"
                      />
                    </Stack>
                  </Box>
                </Box>
              </Paper>

              {/* Patient Report Export */}
              <Box sx={{ mb: 4, textAlign: 'center' }}>
                <Button
                  variant="contained"
                  size="large"
                  onClick={() => {
                    if (result) {
                      const report = generatePatientReport(result, explanations, form);
                      exportPatientReport(report);
                    }
                  }}
                  sx={{ 
                    background: 'linear-gradient(45deg, #2E7D32 30%, #4CAF50 90%)',
                    px: 6,
                    py: 2,
                    fontSize: '1.1rem',
                    fontWeight: 600,
                    borderRadius: 3,
                    boxShadow: '0px 8px 24px rgba(46, 125, 50, 0.3)',
                    '&:hover': {
                      background: 'linear-gradient(45deg, #1B5E20 30%, #2E7D32 90%)',
                      boxShadow: '0px 12px 32px rgba(46, 125, 50, 0.4)',
                      transform: 'translateY(-2px)',
                    },
                    transition: 'all 0.3s ease-in-out',
                  }}
                >
                  📄 Export Patient Report
                </Button>
                <Typography variant="body2" color="text.secondary" sx={{ mt: 2, maxWidth: '400px', mx: 'auto' }}>
                  Download comprehensive assessment report for clinical records and quality assurance
                </Typography>
              </Box>
            </>
          )}

          {explanations?.ok && explanations.plots && (
            <Box sx={{ mt: 4 }}>
              <Typography variant="h5" component="h3" gutterBottom sx={{ mb: 3 }}>
                Explainability Analysis
              </Typography>
              
              <Box sx={{ mb: 3 }}>
                <Typography variant="body1" sx={{ mb: 2 }}>Model:</Typography>
                <RadioGroup
                  row
                  value={modelForExplain}
                  onChange={(e) => setModelForExplain(e.target.value as "rf" | "xgb" | "nn")}
                >
                  <FormControlLabel value="rf" control={<Radio />} label="Random Forest" />
                  <FormControlLabel value="xgb" control={<Radio />} label="XGBoost" />
                  <FormControlLabel value="nn" control={<Radio />} label="Neural Network" />
                </RadioGroup>
              </Box>

              <Box sx={{ mb: 3, borderBottom: 1, borderColor: 'divider' }}>
                <Tabs 
                  value={explanationMethod} 
                  onChange={(_e, v) => setExplanationMethod(v)}
                  aria-label="explanation method tabs"
                >
                  <Tab label="SHAP" value="shap" />
                  <Tab label="LIME" value="lime" />
                </Tabs>
              </Box>

              <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' }, gap: 2 }}>
                {explanationMethod === "shap" ? (
                  // SHAP plots
                  modelForExplain === "rf" ? (
                    <>
                      <Card>
                        <CardContent>
                          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                            SHAP Waterfall - Random Forest
                          </Typography>
                          <Box
                            component="img"
                            src={`data:image/png;base64,${explanations.plots.shap_rf}`}
                            alt="SHAP RF"
                            sx={{ width: '100%', height: 'auto', borderRadius: 1, border: 1, borderColor: 'divider', bgcolor: 'white' }}
                          />
                        </CardContent>
                      </Card>
                      <Card>
                        <CardContent>
                          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                            Feature Importance - Random Forest
                          </Typography>
                          <Box
                            component="img"
                            src={`data:image/png;base64,${explanations.plots.importance_rf}`}
                            alt="FI RF"
                            sx={{ width: '100%', height: 'auto', borderRadius: 1, border: 1, borderColor: 'divider', bgcolor: 'white' }}
                          />
                        </CardContent>
                      </Card>
                    </>
                  ) : modelForExplain === "xgb" ? (
                    <>
                      <Card>
                        <CardContent>
                          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                            SHAP Waterfall - XGBoost
                          </Typography>
                          <Box
                            component="img"
                            src={`data:image/png;base64,${explanations.plots.shap_xgb}`}
                            alt="SHAP XGB"
                            sx={{ width: '100%', height: 'auto', borderRadius: 1, border: 1, borderColor: 'divider', bgcolor: 'white' }}
                          />
                        </CardContent>
                      </Card>
                      <Card>
                        <CardContent>
                          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                            Feature Importance - XGBoost
                          </Typography>
                          <Box
                            component="img"
                            src={`data:image/png;base64,${explanations.plots.importance_xgb}`}
                            alt="FI XGB"
                            sx={{ width: '100%', height: 'auto', borderRadius: 1, border: 1, borderColor: 'divider', bgcolor: 'white' }}
                          />
                        </CardContent>
                      </Card>
                    </>
                  ) : (
                    <>
                      <Card>
                        <CardContent>
                          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                            SHAP Waterfall - Neural Network
                          </Typography>
                          {explanations.plots.shap_nn ? (
                            <Box
                              component="img"
                              src={`data:image/png;base64,${explanations.plots.shap_nn}`}
                              alt="SHAP NN"
                              sx={{ width: '100%', height: 'auto', borderRadius: 1, border: 1, borderColor: 'divider', bgcolor: 'white' }}
                            />
                          ) : (
                            <Typography variant="body2" color="text.secondary">
                              Neural Network SHAP plot not available
                            </Typography>
                          )}
                        </CardContent>
                      </Card>
                      <Card>
                        <CardContent>
                          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                            SHAP Feature Importance - Neural Network
                          </Typography>
                          {explanations.plots.importance_nn ? (
                            <Box
                              component="img"
                              src={`data:image/png;base64,${explanations.plots.importance_nn}`}
                              alt="FI NN (SHAP)"
                              sx={{ width: '100%', height: 'auto', borderRadius: 1, border: 1, borderColor: 'divider', bgcolor: 'white' }}
                            />
                          ) : (
                            <Typography variant="body2" color="text.secondary">
                              Neural networks use SHAP explanations instead of tree-based feature importance. Run another explanation if this visualization fails to generate.
                            </Typography>
                          )}
                        </CardContent>
                      </Card>
                    </>
                  )
                ) : (
                  // LIME plots
                  modelForExplain === "rf" ? (
                    <>
                      <Card>
                        <CardContent>
                          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                            LIME Explanation - Random Forest
                          </Typography>
                          {explanations.plots.lime_rf ? (
                            <Box
                              component="img"
                              src={`data:image/png;base64,${explanations.plots.lime_rf}`}
                              alt="LIME RF"
                              sx={{ width: '100%', height: 'auto', borderRadius: 1, border: 1, borderColor: 'divider', bgcolor: 'white' }}
                            />
                          ) : (
                            <Typography variant="body2" color="text.secondary">
                              LIME explanation not available
                            </Typography>
                          )}
                        </CardContent>
                      </Card>
                      <Card>
                        <CardContent>
                          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                            Feature Importance - Random Forest
                          </Typography>
                          <Box
                            component="img"
                            src={`data:image/png;base64,${explanations.plots.importance_rf}`}
                            alt="FI RF"
                            sx={{ width: '100%', height: 'auto', borderRadius: 1, border: 1, borderColor: 'divider', bgcolor: 'white' }}
                          />
                        </CardContent>
                      </Card>
                    </>
                  ) : modelForExplain === "xgb" ? (
                    <>
                      <Card>
                        <CardContent>
                          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                            LIME Explanation - XGBoost
                          </Typography>
                          {explanations.plots.lime_xgb ? (
                            <Box
                              component="img"
                              src={`data:image/png;base64,${explanations.plots.lime_xgb}`}
                              alt="LIME XGB"
                              sx={{ width: '100%', height: 'auto', borderRadius: 1, border: 1, borderColor: 'divider', bgcolor: 'white' }}
                            />
                          ) : (
                            <Typography variant="body2" color="text.secondary">
                              LIME explanation not available
                            </Typography>
                          )}
                        </CardContent>
                      </Card>
                      <Card>
                        <CardContent>
                          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                            Feature Importance - XGBoost
                          </Typography>
                          <Box
                            component="img"
                            src={`data:image/png;base64,${explanations.plots.importance_xgb}`}
                            alt="FI XGB"
                            sx={{ width: '100%', height: 'auto', borderRadius: 1, border: 1, borderColor: 'divider', bgcolor: 'white' }}
                          />
                        </CardContent>
                      </Card>
                    </>
                  ) : (
                    <>
                      <Card>
                        <CardContent>
                          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                            LIME Explanation - Neural Network
                          </Typography>
                          {explanations.plots.lime_nn ? (
                            <Box
                              component="img"
                              src={`data:image/png;base64,${explanations.plots.lime_nn}`}
                              alt="LIME NN"
                              sx={{ width: '100%', height: 'auto', borderRadius: 1, border: 1, borderColor: 'divider', bgcolor: 'white' }}
                            />
                          ) : (
                            <Typography variant="body2" color="text.secondary">
                              LIME explanation not available
                            </Typography>
                          )}
                        </CardContent>
                      </Card>
                      <Card>
                        <CardContent>
                          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                            LIME Feature Importance - Neural Network
                          </Typography>
                          {explanations.plots.importance_nn_lime ? (
                            <Box
                              component="img"
                              src={`data:image/png;base64,${explanations.plots.importance_nn_lime}`}
                              alt="FI NN (LIME)"
                              sx={{ width: '100%', height: 'auto', borderRadius: 1, border: 1, borderColor: 'divider', bgcolor: 'white' }}
                            />
                          ) : (
                            <Typography variant="body2" color="text.secondary">
                              LIME importance visualization not available. Try rerunning the explanation to regenerate the plot.
                            </Typography>
                          )}
                        </CardContent>
                      </Card>
                    </>
                  )
                )}
              </Box>

              {explanations.contributions && (
                <Card sx={{ mt: 3 }}>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Top Feature Contributions ({modelForExplain.toUpperCase()} - {explanationMethod.toUpperCase()})
                    </Typography>
                    <Stack spacing={1}>
                      {(() => {
                        let contributions = [];
                        if (modelForExplain === "rf") {
                          const modelContribs = explanations.contributions.random_forest;
                          contributions = explanationMethod === "shap" 
                            ? (modelContribs?.shap || [])
                            : (modelContribs?.lime || []);
                        } else if (modelForExplain === "xgb") {
                          const modelContribs = explanations.contributions.xgboost;
                          contributions = explanationMethod === "shap"
                            ? (modelContribs?.shap || [])
                            : (modelContribs?.lime || []);
                        } else {
                          const modelContribs = explanations.contributions.neural_net;
                          contributions = explanationMethod === "shap"
                            ? (modelContribs?.shap || [])
                            : (modelContribs?.lime || []);
                        }
                        return contributions.length > 0 ? (
                          contributions.slice(0, 5).map((c, i) => (
                            <Box key={i} sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                              <Typography variant="body2">
                                {c.feature} = {c.value}
                              </Typography>
                              <Chip
                                label={c.contribution.toFixed(3)}
                                color={c.impact.includes("Increases") ? "error" : "success"}
                                size="small"
                              />
                            </Box>
                          ))
                        ) : (
                          <Typography variant="body2" color="text.secondary">
                            No {explanationMethod.toUpperCase()} contributions available for this model
                          </Typography>
                        );
                      })()}
                    </Stack>
                  </CardContent>
                </Card>
              )}
            </Box>
          )}

          {result && (
            <Box sx={{ mt: 6 }}>
              <Typography variant="h5" component="h3" gutterBottom>
                Partial Dependence & ICE Explorer
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 3, maxWidth: 640 }}>
                Understand how varying a single feature changes the model&apos;s predicted probability,
                while keeping the rest of the patient context anchored to real records.
              </Typography>
              <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} sx={{ mb: 3 }}>
                <FormControl fullWidth sx={{ minWidth: 200 }}>
                  <InputLabel id="pdp-model-label">Model</InputLabel>
                  <Select
                    labelId="pdp-model-label"
                    value={pdpModel}
                    label="Model"
                    onChange={(e) => setPdpModel(e.target.value as "rf" | "xgb" | "nn")}
                  >
                    <MenuItem value="rf">Random Forest</MenuItem>
                    <MenuItem value="xgb">XGBoost</MenuItem>
                    <MenuItem value="nn">Neural Network</MenuItem>
                  </Select>
                </FormControl>
                <FormControl fullWidth sx={{ minWidth: 200 }}>
                  <InputLabel id="pdp-feature-label">Feature</InputLabel>
                  <Select
                    labelId="pdp-feature-label"
                    value={pdpFeature}
                    label="Feature"
                    onChange={(e) => setPdpFeature(e.target.value)}
                  >
                    {featureOptions.map((feature) => (
                      <MenuItem key={feature} value={feature}>
                        {feature.replace('_', ' ')}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
                <Button
                  variant="outlined"
                  onClick={handleFetchPdp}
                  disabled={pdpLoading}
                  sx={{ alignSelf: { xs: 'stretch', sm: 'center' }, minWidth: 180 }}
                >
                  {pdpLoading ? 'Generating...' : 'Generate Curves'}
                </Button>
              </Stack>
              {pdpError && (
                <Alert severity="error" sx={{ mb: 2 }}>
                  {pdpError}
                </Alert>
              )}
              <Card>
                <CardContent>
                  {pdpLoading && (
                    <Box sx={{ display: 'flex', justifyContent: 'center', py: 6 }}>
                      <CircularProgress />
                    </Box>
                  )}
                  {!pdpLoading && hasPdpPayload(pdpData) && (
                    <PdpIceChart data={pdpData} />
                  )}
                  {!pdpLoading && !hasPdpPayload(pdpData) && (
                    <Typography variant="body2" color="text.secondary">
                      Select a feature and click &quot;Generate Curves&quot; to view the partial dependence and ICE
                      visualization for that model.
                    </Typography>
                  )}
                </CardContent>
              </Card>
            </Box>
          )}
        </Box>
      )}
    </Box>
  );
}
