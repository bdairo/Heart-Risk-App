export type PredictRequest = {
  age: number;
  sex: "M" | "F";
  chest_pain_type: "TA" | "ATA" | "NAP" | "ASY";
  cholesterol: number;
  fasting_bs: 0 | 1;
  max_hr: number;
  exercise_angina: "Y" | "N";
  oldpeak: number;
  st_slope: "Up" | "Flat" | "Down";
};

export type PredictResponse = {
  ok: boolean;
  error?: string;
  prediction_id?: string;
  features?: Record<string, unknown>;
  predictions?: {
    random_forest: { label: number; probability: number };
    xgboost: { label: number; probability: number };
    neural_net: { label: number; probability: number };
  };
  risk_assessment?: {
    category: 'Low' | 'Moderate' | 'High';
    average_probability: number;
    model_agreement: number;
  };
};

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:5000";

export async function predict(req: PredictRequest): Promise<PredictResponse> {
  const res = await fetch(`${API_BASE}/api/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  return res.json();
}

export type ExplainResponse = {
  ok: boolean;
  error?: string;
  feature_names?: string[];
  plots?: {
    shap_rf: string; // base64 png
    shap_xgb: string; // base64 png
    shap_nn?: string; // base64 png
    lime_rf?: string; // base64 png
    lime_xgb?: string; // base64 png
    lime_nn?: string; // base64 png
    importance_rf: string; // base64 png
    importance_xgb: string; // base64 png
    importance_nn?: string; // base64 png
    importance_nn_lime?: string; // base64 png
  };
  contributions?: {
    random_forest: {
      shap: Array<{ feature: string; value: number; contribution: number; impact: string }>;
      lime: Array<{ feature: string; value: number; contribution: number; impact: string }>;
    };
    xgboost: {
      shap: Array<{ feature: string; value: number; contribution: number; impact: string }>;
      lime: Array<{ feature: string; value: number; contribution: number; impact: string }>;
    };
    neural_net?: {
      shap: Array<{ feature: string; value: number; contribution: number; impact: string }>;
      lime: Array<{ feature: string; value: number; contribution: number; impact: string }>;
    };
  };
};

export type PdpIceResponse = {
  ok: boolean;
  error?: string;
  feature?: string;
  model?: "rf" | "xgb" | "nn";
  feature_type?: "numeric" | "categorical";
  grid?: number[];
  grid_labels?: string[];
  pdp?: number[];
  ice?: number[][];
};

export async function explain(req: PredictRequest): Promise<ExplainResponse> {
  const res = await fetch(`${API_BASE}/api/explain`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  return res.json();
}

export async function fetchPdp(body: { feature: string; model: "rf" | "xgb" | "nn"; grid_resolution?: number; ice_count?: number; }): Promise<PdpIceResponse> {
  const res = await fetch(`${API_BASE}/api/pdp`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return res.json();
}

