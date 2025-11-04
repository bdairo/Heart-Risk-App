from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import io
import base64
from keras.models import load_model
import json
import uuid
from datetime import datetime
import os

import shap.explainers._tree as shap_tree
import ast

# Monkey patch TreeEnsemble to fix base_score before it reaches XGBTreeModelLoader
original_tree_ensemble_init = shap_tree.TreeEnsemble.__init__

def patched_tree_ensemble_init(self, model, data=None, data_missing=None, model_output=None):
    """Patched TreeEnsemble that fixes XGBoost base_score issue"""
    # For XGBoost models, fix the base_score in the model config
    if hasattr(model, 'get_booster'):
        try:
            booster = model.get_booster()
            config = booster.save_config()
            config_dict = json.loads(config)
            
            # Fix base_score if it exists and is a string array
            learner_params = config_dict.get("learner", {}).get("learner_train_param", {})
            base_score = learner_params.get("base_score")
            
            if base_score is not None:
                # Parse string array format like '[5.494744E-1]'
                if isinstance(base_score, str):
                    try:
                        base_score = ast.literal_eval(base_score)
                    except Exception:
                        pass
                if isinstance(base_score, (list, tuple, np.ndarray)):
                    base_score = base_score[0]
                # Update config with fixed base_score
                config_dict["learner"]["learner_train_param"]["base_score"] = str(float(base_score))
                booster.load_config(json.dumps(config_dict))
        except Exception as e:
            print(f"Warning: Could not fix XGBoost base_score: {e}")
    
    # Call original init
    original_tree_ensemble_init(self, model, data, data_missing, model_output)

shap_tree.TreeEnsemble.__init__ = patched_tree_ensemble_init


app = Flask(__name__)
CORS(app)

# Load models and encoders
# Use correct relative paths from backend/ directory
best_model = joblib.load('machine_learning/models/best_model.pkl')  # Random Forest
# best_xgb = joblib.load('machine_learning/models/best_xgb.pkl')  # XGBoost
import xgboost as xgb
best_xgb = xgb.XGBClassifier()
best_xgb.load_model('machine_learning/models/best_xgb.json')

# Fix XGBoost base_score issue for SHAP compatibility
try:
    booster = best_xgb.get_booster()
    config = booster.save_config()
    config_dict = json.loads(config)
    
    # Fix base_score if it exists and is a string array
    learner_params = config_dict.get("learner", {}).get("learner_train_param", {})
    base_score = learner_params.get("base_score")
    
    if base_score is not None:
        # Parse string array format like '[5.494744E-1]'
        if isinstance(base_score, str):
            try:
                base_score = ast.literal_eval(base_score)
            except Exception:
                pass
        if isinstance(base_score, (list, tuple, np.ndarray)):
            base_score = base_score[0]
        # Update config with fixed base_score
        config_dict["learner"]["learner_train_param"]["base_score"] = str(float(base_score))
        booster.load_config(json.dumps(config_dict))
        print("✓ Fixed XGBoost base_score for SHAP compatibility")
except Exception as e:
    print(f"Warning: Could not fix XGBoost base_score: {e}")

model_nn = load_model('machine_learning/models/model_nn.keras')  # Neural Network

# Load encoders
sex_encoder = joblib.load('machine_learning/encoders/sex_encoder.pkl')
chestpain_encoder = joblib.load('machine_learning/encoders/chestpain_encoder.pkl')
exercise_encoder = joblib.load('machine_learning/encoders/exercise_encoder.pkl')
slope_encoder = joblib.load('machine_learning/encoders/slope_encoder.pkl')

# Feature names (after dropping RestingBP and RestingECG)
FEATURE_NAMES = ['Age', 'Sex', 'ChestPainType', 'Cholesterol', 'FastingBS', 
                 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']

# Audit logging
AUDIT_LOG_FILE = 'audit_log.jsonl'

def log_prediction(prediction_id, input_data, predictions, risk_category, model_agreement, timestamp):
    """Log prediction for audit trail"""
    audit_entry = {
        'prediction_id': prediction_id,
        'timestamp': timestamp,
        'input_data': input_data,
        'predictions': predictions,
        'risk_category': risk_category,
        'model_agreement': model_agreement,
        'model_versions': {
            'random_forest': '1.0',
            'xgboost': '1.0', 
            'neural_network': '1.0'
        }
    }
    
    # Append to audit log file
    with open(AUDIT_LOG_FILE, 'a') as f:
        f.write(json.dumps(audit_entry) + '\n')

def calculate_risk_category(avg_probability):
    """Calculate risk category based on average probability"""
    if avg_probability < 0.3:
        return 'Low'
    elif avg_probability < 0.6:
        return 'Moderate'
    else:
        return 'High'

def calculate_model_agreement(predictions):
    """Calculate agreement between models"""
    probs = [
        predictions['random_forest']['probability'],
        predictions['xgboost']['probability'],
        predictions['neural_net']['probability']
    ]
    mean = np.mean(probs)
    std_dev = np.std(probs)
    return max(0, 1 - (std_dev * 2))

# Create SHAP explainers
explainer_rf = shap.TreeExplainer(best_model)

# For XGBoost, use KernelExplainer as fallback due to base_score compatibility issues
try:
    # Try TreeExplainer first (fastest for tree-based models)
    explainer_xgb = shap.TreeExplainer(best_xgb)
    print("✓ Successfully created XGBoost explainer using TreeExplainer")
except Exception as e:
    print(f"Warning: TreeExplainer failed for XGBoost: {e}")
    try:
        # Fallback to KernelExplainer with wrapper function
        # Create a callable wrapper for predict_proba
        def xgb_predict_wrapper(X):
            """Wrapper function for XGBoost predict_proba - returns positive class probabilities"""
            return best_xgb.predict_proba(X)[:, 1]  # Return probability of positive class
        
        # Create background data (100 samples with reasonable feature ranges)
        # This is used as a reference for SHAP calculations
        np.random.seed(42)  # For reproducibility
        background_data = np.zeros((100, 9))
        background_data[:, 0] = np.random.uniform(20, 80, 100)  # Age: 20-80
        background_data[:, 1] = np.random.randint(0, 2, 100)  # Sex: 0-1
        background_data[:, 2] = np.random.randint(0, 4, 100)  # ChestPainType: 0-3
        background_data[:, 3] = np.random.uniform(100, 400, 100)  # Cholesterol: 100-400
        background_data[:, 4] = np.random.randint(0, 2, 100)  # FastingBS: 0-1
        background_data[:, 5] = np.random.uniform(60, 200, 100)  # MaxHR: 60-200
        background_data[:, 6] = np.random.randint(0, 2, 100)  # ExerciseAngina: 0-1
        background_data[:, 7] = np.random.uniform(0, 5, 100)  # Oldpeak: 0-5
        background_data[:, 8] = np.random.randint(0, 3, 100)  # ST_Slope: 0-2
        
        explainer_xgb = shap.KernelExplainer(xgb_predict_wrapper, background_data)
        print("✓ Successfully created XGBoost explainer using KernelExplainer (fallback)")
    except Exception as e2:
        print(f"Warning: Could not create XGBoost explainer: {e2}")
        explainer_xgb = None

@app.route("/")
def home():
    """Home page with input form"""
    return render_template('index.html')

# JSON API for frontend consumption
@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Accept JSON payload, return model predictions as JSON (no plots)."""
    try:
        payload = request.get_json(force=True)

        # Extract and encode input features
        age = float(payload["age"])  # numeric
        sex = sex_encoder.transform([payload["sex"]])[0]
        chest_pain = chestpain_encoder.transform([payload["chest_pain_type"]])[0]
        cholesterol = float(payload["cholesterol"])  # numeric
        fasting_bs = int(payload["fasting_bs"])  # 1 if FastingBS >= 120 else 0
        max_hr = float(payload["max_hr"])  # numeric
        exercise_angina = exercise_encoder.transform([payload["exercise_angina"]])[0]
        oldpeak = float(payload["oldpeak"])  # numeric
        st_slope = slope_encoder.transform([payload["st_slope"]])[0]

        features = np.array([[
            age,
            sex,
            chest_pain,
            cholesterol,
            fasting_bs,
            max_hr,
            exercise_angina,
            oldpeak,
            st_slope,
        ]])
        
        # Predictions
        pred_rf = int(best_model.predict(features)[0])
        prob_rf = float(best_model.predict_proba(features)[0][1])

        pred_xgb = int(best_xgb.predict(features)[0])
        prob_xgb = float(best_xgb.predict_proba(features)[0][1])

        pred_nn = int(np.argmax(model_nn.predict(features, verbose=0), axis=1)[0])
        prob_nn = float(model_nn.predict(features, verbose=0)[0][1])

        # Calculate risk metrics
        predictions = {
            "random_forest": {"label": pred_rf, "probability": prob_rf},
            "xgboost": {"label": pred_xgb, "probability": prob_xgb},
            "neural_net": {"label": pred_nn, "probability": prob_nn},
        }
        
        # print('predictions are: ', predictions);
        # Calculate average risk and model agreement
        avg_probability = (prob_rf + prob_xgb + prob_nn) / 3
        risk_category = calculate_risk_category(avg_probability)
        model_agreement = calculate_model_agreement(predictions)
        
        # Generate prediction ID and log for audit trail
        prediction_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        log_prediction(
            prediction_id=prediction_id,
            input_data={
                "age": age,
                "sex": payload["sex"],
                "chest_pain_type": payload["chest_pain_type"],
                "cholesterol": cholesterol,
                "fasting_bs": fasting_bs,
                "max_hr": max_hr,
                "exercise_angina": payload["exercise_angina"],
                "oldpeak": oldpeak,
                "st_slope": payload["st_slope"],
            },
            predictions=predictions,
            risk_category=risk_category,
            model_agreement=model_agreement,
            timestamp=timestamp
        )

        return jsonify({
            "ok": True,
            "prediction_id": prediction_id,
            "features": {
                "Age": age,
                "Sex": payload["sex"],
                "ChestPainType": payload["chest_pain_type"],
                "Cholesterol": cholesterol,
                "FastingBS": fasting_bs,
                "MaxHR": max_hr,
                "ExerciseAngina": payload["exercise_angina"],
                "Oldpeak": oldpeak,
                "ST_Slope": payload["st_slope"],
            },
            "predictions": predictions,
            "risk_assessment": {
                "category": risk_category,
                "average_probability": float(avg_probability),
                "model_agreement": float(model_agreement)
            }
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/explain", methods=["POST"])
def api_explain():
    print('in api_explain');
    """Return SHAP plots and feature contributions for RF and XGB."""
    try:
        payload = request.get_json(force=True)

        age = float(payload["age"])
        sex = sex_encoder.transform([payload["sex"]])[0]
        chest_pain = chestpain_encoder.transform([payload["chest_pain_type"]])[0]
        cholesterol = float(payload["cholesterol"]) 
        fasting_bs = int(payload["fasting_bs"]) 
        max_hr = float(payload["max_hr"]) 
        exercise_angina = exercise_encoder.transform([payload["exercise_angina"]])[0]
        oldpeak = float(payload["oldpeak"]) 
        st_slope = slope_encoder.transform([payload["st_slope"]])[0]

        features = np.array([[
            age,
            sex,
            chest_pain,
            cholesterol,
            fasting_bs,
            max_hr,
            exercise_angina,
            oldpeak,
            st_slope,
        ]])

        shap_plot_rf = generate_shap_plot(explainer_rf, features, 'Random Forest')
        # print('shap_plot_rf', shap_plot_rf);
        shap_plot_xgb = generate_shap_plot(explainer_xgb, features, 'XGBoost') if explainer_xgb else None
        # print('shap_plot_xgb', shap_plot_xgb);
        feature_importance_rf = generate_feature_importance(best_model, 'Random Forest')
        feature_importance_xgb = generate_feature_importance(best_xgb, 'XGBoost')

        feature_contrib_rf = generate_feature_contribution(explainer_rf, features, 1)
        feature_contrib_xgb = generate_feature_contribution(explainer_xgb, features, 1) if explainer_xgb else []

        return jsonify({
            "ok": True,
            "feature_names": FEATURE_NAMES,
            "plots": {
                "shap_rf": shap_plot_rf,
                "shap_xgb": shap_plot_xgb,
                "importance_rf": feature_importance_rf,
                "importance_xgb": feature_importance_xgb
            },
            "contributions": {
                "random_forest": feature_contrib_rf,
                "xgboost": feature_contrib_xgb
            }
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

def generate_shap_plot(explainer, features, model_name):
    """Generate SHAP waterfall plot for individual prediction"""
    # print('in generate_shap_plot');
    # print('explainer', explainer);
    # print('features', features);
    # print('model_name', model_name);
    if explainer is None:
        return None
    shap_values = explainer.shap_values(features)
    # print('shap_values', shap_values);

    # For binary classification, use positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # Handle 3D array (samples, features, classes) - take first sample and positive class
    if len(shap_values.shape) == 3:
        shap_values = shap_values[0, :, 1]  # First sample, all features, positive class
    # Handle 2D array (samples, features) - flatten to 1D for single prediction
    elif len(shap_values.shape) == 2:
        shap_values = shap_values[0]  # Take first row (single prediction)
    # Handle matrix of explanations - take the first row (single prediction)
    elif len(shap_values.shape) > 1 and shap_values.shape[0] > 1:
        shap_values = shap_values[0]

    # print(f"About to create SHAP plot for {model_name}")
    # print(f"shap_values shape: {shap_values.shape}")
    # print(f"shap_values: {shap_values}")
    # print(f"features[0]: {features[0]}")
    # print(f"FEATURE_NAMES: {FEATURE_NAMES}")
    # print(f"explainer.expected_value: {explainer.expected_value}")
    
    # Handle base_values for different explainer types
    if isinstance(explainer.expected_value, np.ndarray):
        if len(explainer.expected_value.shape) > 0 and explainer.expected_value.shape[0] > 1:
            base_value = explainer.expected_value[1]  # Positive class for binary classification
        else:
            base_value = float(explainer.expected_value[0]) if len(explainer.expected_value) > 0 else 0.5
    else:
        base_value = float(explainer.expected_value)
    
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(shap.Explanation(
        values=shap_values,
        base_values=base_value,
        data=features[0],
        feature_names=FEATURE_NAMES
    ), max_display=9, show=False)
    
    plt.title(f'SHAP Waterfall Plot - {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Convert to base64
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.read()).decode()
    plt.close()
    
    # print('returning img_base64', img_base64, 'for', model_name);
    return img_base64

def generate_feature_importance(model, model_name):
    """Generate feature importance bar plot"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances[indices], color='steelblue')
    plt.xticks(range(len(importances)), [FEATURE_NAMES[i] for i in indices], rotation=45, ha='right')
    plt.xlabel('Features', fontsize=12, fontweight='bold')
    plt.ylabel('Importance', fontsize=12, fontweight='bold')
    plt.title(f'Feature Importance - {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.read()).decode()
    plt.close()
    
    return img_base64

def generate_feature_contribution(explainer, features, prediction):
    """Generate feature contribution table data"""
    if explainer is None:
        return []
    shap_values = explainer.shap_values(features)
    
    # For binary classification, use positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # Handle 3D array (samples, features, classes) - take first sample and positive class
    if len(shap_values.shape) == 3:
        shap_values = shap_values[0, :, 1]  # First sample, all features, positive class
    # Handle 2D array (samples, features) - flatten to 1D for single prediction
    elif len(shap_values.shape) == 2:
        shap_values = shap_values[0]  # Take first row (single prediction)
    # Handle matrix of explanations - take the first row (single prediction)
    elif len(shap_values.shape) > 1 and shap_values.shape[0] > 1:
        shap_values = shap_values[0]
    
    contributions = []
    for i, feature_name in enumerate(FEATURE_NAMES):
        contributions.append({
            'feature': feature_name,
            'value': float(features[0][i]),
            'contribution': float(shap_values[i]),
            'impact': 'Increases Risk' if shap_values[i] > 0 else 'Decreases Risk'
        })
    
    # Sort by absolute contribution
    contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
    
    return contributions

@app.route("/api/audit", methods=["GET"])
def api_audit():
    """Return audit trail analytics"""
    try:
        if not os.path.exists(AUDIT_LOG_FILE):
            return jsonify({"ok": True, "audit_data": [], "summary": {}})
        
        # Read audit log
        audit_entries = []
        with open(AUDIT_LOG_FILE, 'r') as f:
            for line in f:
                if line.strip():
                    audit_entries.append(json.loads(line.strip()))
        
        # Calculate summary statistics
        total_predictions = len(audit_entries)
        risk_categories = {}
        avg_agreement = 0
        
        for entry in audit_entries:
            category = entry.get('risk_category', 'Unknown')
            risk_categories[category] = risk_categories.get(category, 0) + 1
            avg_agreement += entry.get('model_agreement', 0)
        
        if total_predictions > 0:
            avg_agreement /= total_predictions
        
        summary = {
            "total_predictions": total_predictions,
            "risk_distribution": risk_categories,
            "average_model_agreement": round(avg_agreement, 3),
            "date_range": {
                "earliest": min([entry.get('timestamp', '') for entry in audit_entries]) if audit_entries else None,
                "latest": max([entry.get('timestamp', '') for entry in audit_entries]) if audit_entries else None
            }
        }
        
        return jsonify({
            "ok": True,
            "audit_data": audit_entries[-10:],  # Last 10 entries
            "summary": summary
        })
        
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

@app.route("/about")
def about():
    """About page explaining the models and features"""
    return render_template('about.html')

if __name__ == "__main__":
    app.run(debug=True, port=5000)
