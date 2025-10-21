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

app = Flask(__name__)
CORS(app)

# Load models and encoders
# Use correct relative paths from backend/ directory
best_model = joblib.load('models/best_model.pkl')  # Random Forest
best_xgb = joblib.load('models/best_xgb.pkl')  # XGBoost
model_nn = load_model('models/model_nn.keras')  # Neural Network

# Load encoders
sex_encoder = joblib.load('models/sex_encoder.pkl')
chestpain_encoder = joblib.load('models/chestpain_encoder.pkl')
exercise_encoder = joblib.load('models/exercise_encoder.pkl')
slope_encoder = joblib.load('models/slope_encoder.pkl')

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
try:
    explainer_xgb = shap.TreeExplainer(best_xgb)
except Exception as e:
    print(f"Warning: Could not create XGBoost explainer: {e}")
    explainer_xgb = None

@app.route("/")
def home():
    """Home page with input form"""
    return render_template('index.html')

# @app.route("/predict", methods=['POST'])
# def predict():
#     """Make prediction and return results with visualizations"""
#     try:
#         # Get form data
#         data = request.form
        
#         # Parse and encode input
#         age = float(data['age'])
#         sex = sex_encoder.transform([data['sex']])[0]
#         chest_pain = chestpain_encoder.transform([data['chest_pain']])[0]
#         cholesterol = float(data['cholesterol'])
#         fasting_bs = int(data['fasting_bs'])
#         max_hr = float(data['max_hr'])
#         exercise_angina = exercise_encoder.transform([data['exercise_angina']])[0]
#         oldpeak = float(data['oldpeak'])
#         st_slope = slope_encoder.transform([data['st_slope']])[0]
        
#         # Create feature array
#         features = np.array([[age, sex, chest_pain, cholesterol, fasting_bs, 
#                             max_hr, exercise_angina, oldpeak, st_slope]])
        
#         # Make predictions
#         pred_rf = best_model.predict(features)[0]
#         pred_prob_rf = best_model.predict_proba(features)[0]
        
#         pred_xgb = best_xgb.predict(features)[0]
#         pred_prob_xgb = best_xgb.predict_proba(features)[0]
        
#         pred_nn = np.argmax(model_nn.predict(features, verbose=0), axis=1)[0]
#         pred_prob_nn = model_nn.predict(features, verbose=0)[0]
        
#         # Generate SHAP visualizations
#         shap_plot_rf = generate_shap_plot(explainer_rf, features, 'Random Forest')
#         shap_plot_xgb = generate_shap_plot(explainer_xgb, features, 'XGBoost') if explainer_xgb else None
        
#         # Generate feature importance plots
#         feature_importance_rf = generate_feature_importance(best_model, 'Random Forest')
#         feature_importance_xgb = generate_feature_importance(best_xgb, 'XGBoost')
        
#         # Generate individual feature contribution
#         feature_contrib_rf = generate_feature_contribution(explainer_rf, features, pred_rf)
#         feature_contrib_xgb = generate_feature_contribution(explainer_xgb, features, pred_xgb) if explainer_xgb else []
        
#         return render_template('results.html',
#                              pred_rf=int(pred_rf),
#                              pred_prob_rf=float(pred_prob_rf[1]),
#                              pred_xgb=int(pred_xgb),
#                              pred_prob_xgb=float(pred_prob_xgb[1]),
#                              pred_nn=int(pred_nn),
#                              pred_prob_nn=float(pred_prob_nn[1]),
#                              shap_plot_rf=shap_plot_rf,
#                              shap_plot_xgb=shap_plot_xgb,
#                              feature_importance_rf=feature_importance_rf,
#                              feature_importance_xgb=feature_importance_xgb,
#                              feature_contrib_rf=feature_contrib_rf,
#                              feature_contrib_xgb=feature_contrib_xgb)
    
#     except Exception as e:
#         return render_template('error.html', error=str(e))


# JSON API for frontend consumption
@app.route("/api/predict", methods=["POST"])
def api_predict():
    print('in api_predict');
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
        print('shap_plot_rf', shap_plot_rf);
        shap_plot_xgb = generate_shap_plot(explainer_xgb, features, 'XGBoost') if explainer_xgb else None
        print('shap_plot_xgb', shap_plot_xgb);
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
    print('in generate_shap_plot');
    print('explainer', explainer);
    print('features', features);
    print('model_name', model_name);
    if explainer is None:
        return None
    shap_values = explainer.shap_values(features)
    print('shap_values', shap_values);

    # For binary classification, use positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # Handle 3D array (samples, features, classes) - take first sample and positive class
    if len(shap_values.shape) == 3:
        shap_values = shap_values[0, :, 1]  # First sample, all features, positive class
    # Handle matrix of explanations - take the first row (single prediction)
    elif len(shap_values.shape) > 1 and shap_values.shape[0] > 1:
        shap_values = shap_values[0]

    print(f"About to create SHAP plot for {model_name}")
    print(f"shap_values shape: {shap_values.shape}")
    print(f"shap_values: {shap_values}")
    print(f"features[0]: {features[0]}")
    print(f"FEATURE_NAMES: {FEATURE_NAMES}")
    print(f"explainer.expected_value: {explainer.expected_value}")
    
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value if not isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value[1],
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
    
    print('returning img_base64', img_base64);
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
