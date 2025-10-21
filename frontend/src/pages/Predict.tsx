import { useState } from "react";
import { predict, explain, type PredictRequest, type PredictResponse, type ExplainResponse } from "../services/api";
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
} from "@mui/material";
import {
  CheckCircle as CheckCircleIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
  TrendingUp as TrendingUpIcon,
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
  
  // Get top 3 contributing factors
  const topFactors = rfContributions.slice(0, 3);
  
  topFactors.forEach(factor => {
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
  const [modelForExplain, setModelForExplain] = useState<"rf" | "xgb">("rf");

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

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom sx={{ mb: 3 }}>
        Heart Disease Risk Prediction
      </Typography>
      
      <Paper elevation={3} sx={{ p: 4, mb: 4 }}>
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
            >
              {loading ? "Predicting..." : "Predict"}
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
          <Typography variant="h5" component="h2" gutterBottom sx={{ mb: 3 }}>
            Predictions
          </Typography>
          <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr', md: '1fr 1fr 1fr 1fr' }, gap: 2 }}>
            <Card>
              <CardContent>
                <Typography variant="h6" color="primary" gutterBottom>
                  Random Forest
                </Typography>
                <Typography variant="h4" component="div" sx={{ fontWeight: 'bold' }}>
                  {(result.predictions.random_forest.probability * 100).toFixed(1)}%
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Label: {result.predictions.random_forest.label}
                </Typography>
              </CardContent>
            </Card>
            <Card>
              <CardContent>
                <Typography variant="h6" color="primary" gutterBottom>
                  XGBoost
                </Typography>
                <Typography variant="h4" component="div" sx={{ fontWeight: 'bold' }}>
                  {(result.predictions.xgboost.probability * 100).toFixed(1)}%
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Label: {result.predictions.xgboost.label}
                </Typography>
              </CardContent>
            </Card>
            <Card>
              <CardContent>
                <Typography variant="h6" color="primary" gutterBottom>
                  Neural Net
                </Typography>
                <Typography variant="h4" component="div" sx={{ fontWeight: 'bold' }}>
                  {(result.predictions.neural_net.probability * 100).toFixed(1)}%
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Label: {result.predictions.neural_net.label}
                </Typography>
              </CardContent>
            </Card>
            <Card sx={{ bgcolor: 'primary.light', color: 'primary.contrastText' }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Aggregate (avg of 3)
                </Typography>
                <Typography variant="h4" component="div" sx={{ fontWeight: 'bold' }}>
                  {(((result.predictions.random_forest.probability + result.predictions.xgboost.probability + result.predictions.neural_net.probability) / 3) * 100).toFixed(1)}%
                </Typography>
              </CardContent>
            </Card>
          </Box>

          {/* Risk Stratification Section */}
          {result?.predictions && (
            <>
              <Box sx={{ mb: 4 }}>
                <Typography variant="h5" component="h2" gutterBottom sx={{ mb: 3 }}>
                  Risk Assessment
                </Typography>
                
                <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' }, gap: 3 }}>
                  {/* Overall Risk Assessment */}
                  <Card sx={{ 
                    border: 2, 
                    borderColor: `${getRiskCategory(result.risk_assessment?.average_probability || 0).color}.main`,
                    bgcolor: `${getRiskCategory(result.risk_assessment?.average_probability || 0).color}.light`,
                    color: `${getRiskCategory(result.risk_assessment?.average_probability || 0).color}.dark`
                  }}>
                    <CardContent>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                        {getRiskCategory(result.risk_assessment?.average_probability || 0).icon}
                        <Typography variant="h4" component="div" sx={{ ml: 1, fontWeight: 'bold' }}>
                          {result.risk_assessment?.category || 'Unknown'} Risk
                        </Typography>
                      </Box>
                      <Typography variant="h3" component="div" sx={{ fontWeight: 'bold', mb: 1 }}>
                        {((result.risk_assessment?.average_probability || 0) * 100).toFixed(1)}%
                      </Typography>
                      <Typography variant="body1" sx={{ mb: 2 }}>
                        {getRiskCategory(result.risk_assessment?.average_probability || 0).description}
                      </Typography>
                      <Alert severity={getRiskCategory(result.risk_assessment?.average_probability || 0).color} sx={{ mt: 2 }}>
                        <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                          Recommendation: {getRiskCategory(result.risk_assessment?.average_probability || 0).recommendation}
                        </Typography>
                      </Alert>
                    </CardContent>
                  </Card>

                  {/* Model Agreement Analysis */}
                  <Card>
                    <CardContent>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                        <TrendingUpIcon color="primary" />
                        <Typography variant="h6" component="div" sx={{ ml: 1 }}>
                          Model Agreement
                        </Typography>
                      </Box>
                      <Typography variant="h3" component="div" sx={{ fontWeight: 'bold', mb: 1 }}>
                        {((result.risk_assessment?.model_agreement || 0) * 100).toFixed(1)}%
                      </Typography>
                      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                        {(result.risk_assessment?.model_agreement || 0) > 0.8 
                          ? 'High agreement between models - prediction is reliable'
                          : 'Moderate agreement - consider additional validation'
                        }
                      </Typography>
                      <Chip 
                        label={(result.risk_assessment?.model_agreement || 0) > 0.8 ? 'High Confidence' : 'Moderate Confidence'}
                        color={(result.risk_assessment?.model_agreement || 0) > 0.8 ? 'success' : 'warning'}
                        size="small"
                      />
                    </CardContent>
                  </Card>
                </Box>
              </Box>

              {/* Clinical Guidelines Section */}
              <Paper sx={{ p: 3, mb: 4 }}>
                <Typography variant="h5" component="h2" gutterBottom sx={{ mb: 3 }}>
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
                    bgcolor: 'primary.main',
                    '&:hover': { bgcolor: 'primary.dark' },
                    px: 4,
                    py: 1.5
                  }}
                >
                  📄 Export Patient Report
                </Button>
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  Download comprehensive assessment report for clinical records
                </Typography>
              </Box>
            </>
          )}

          {explanations?.ok && explanations.plots && (
            <Box sx={{ mt: 4 }}>
              <Typography variant="h5" component="h3" gutterBottom sx={{ mb: 3 }}>
                Explain
              </Typography>
              
              <Box sx={{ mb: 3 }}>
                <Typography variant="body1" sx={{ mb: 1 }}>Model:</Typography>
                <RadioGroup
                  row
                  value={modelForExplain}
                  onChange={(e) => setModelForExplain(e.target.value as "rf" | "xgb")}
                >
                  <FormControlLabel value="rf" control={<Radio />} label="Random Forest" />
                  <FormControlLabel value="xgb" control={<Radio />} label="XGBoost" />
                </RadioGroup>
              </Box>

              <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' }, gap: 2 }}>
                {modelForExplain === "rf" ? (
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
                ) : (
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
                )}
              </Box>

              {explanations.contributions && (
                <Card sx={{ mt: 3 }}>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Top Feature Contributions ({modelForExplain.toUpperCase()})
                    </Typography>
                    <Stack spacing={1}>
                      {(modelForExplain === "rf" ? explanations.contributions.random_forest : explanations.contributions.xgboost).slice(0, 5).map((c, i) => (
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
                      ))}
                    </Stack>
                  </CardContent>
                </Card>
              )}
            </Box>
          )}
        </Box>
      )}
    </Box>
  );
}
