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
} from "@mui/material";

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
