from flask import Flask, render_template, request, jsonify
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

app = Flask(__name__)

# Load models and encoders
best_model = joblib.load('../best_model.pkl')  # Random Forest
best_xgb = joblib.load('../best_xgb.pkl')  # XGBoost
model_nn = load_model('../model_nn.keras')  # Neural Network

# Load encoders
sex_encoder = joblib.load('../sex_encoder.pkl')
chestpain_encoder = joblib.load('../chestpain_encoder.pkl')
exercise_encoder = joblib.load('../exercise_encoder.pkl')
slope_encoder = joblib.load('../slope_encoder.pkl')

# Feature names (after dropping RestingBP and RestingECG)
FEATURE_NAMES = ['Age', 'Sex', 'ChestPainType', 'Cholesterol', 'FastingBS', 
                 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']

# Create SHAP explainers
explainer_rf = shap.TreeExplainer(best_model)
explainer_xgb = shap.TreeExplainer(best_xgb)

@app.route("/")
def home():
    """Home page with input form"""
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    """Make prediction and return results with visualizations"""
    try:
        # Get form data
        data = request.form
        
        # Parse and encode input
        age = float(data['age'])
        sex = sex_encoder.transform([data['sex']])[0]
        chest_pain = chestpain_encoder.transform([data['chest_pain']])[0]
        cholesterol = float(data['cholesterol'])
        fasting_bs = int(data['fasting_bs'])
        max_hr = float(data['max_hr'])
        exercise_angina = exercise_encoder.transform([data['exercise_angina']])[0]
        oldpeak = float(data['oldpeak'])
        st_slope = slope_encoder.transform([data['st_slope']])[0]
        
        # Create feature array
        features = np.array([[age, sex, chest_pain, cholesterol, fasting_bs, 
                            max_hr, exercise_angina, oldpeak, st_slope]])
        
        # Make predictions
        pred_rf = best_model.predict(features)[0]
        pred_prob_rf = best_model.predict_proba(features)[0]
        
        pred_xgb = best_xgb.predict(features)[0]
        pred_prob_xgb = best_xgb.predict_proba(features)[0]
        
        pred_nn = np.argmax(model_nn.predict(features, verbose=0), axis=1)[0]
        pred_prob_nn = model_nn.predict(features, verbose=0)[0]
        
        # Generate SHAP visualizations
        shap_plot_rf = generate_shap_plot(explainer_rf, features, 'Random Forest')
        shap_plot_xgb = generate_shap_plot(explainer_xgb, features, 'XGBoost')
        
        # Generate feature importance plots
        feature_importance_rf = generate_feature_importance(best_model, 'Random Forest')
        feature_importance_xgb = generate_feature_importance(best_xgb, 'XGBoost')
        
        # Generate individual feature contribution
        feature_contrib_rf = generate_feature_contribution(explainer_rf, features, pred_rf)
        feature_contrib_xgb = generate_feature_contribution(explainer_xgb, features, pred_xgb)
        
        return render_template('results.html',
                             pred_rf=int(pred_rf),
                             pred_prob_rf=float(pred_prob_rf[1]),
                             pred_xgb=int(pred_xgb),
                             pred_prob_xgb=float(pred_prob_xgb[1]),
                             pred_nn=int(pred_nn),
                             pred_prob_nn=float(pred_prob_nn[1]),
                             shap_plot_rf=shap_plot_rf,
                             shap_plot_xgb=shap_plot_xgb,
                             feature_importance_rf=feature_importance_rf,
                             feature_importance_xgb=feature_importance_xgb,
                             feature_contrib_rf=feature_contrib_rf,
                             feature_contrib_xgb=feature_contrib_xgb)
    
    except Exception as e:
        return render_template('error.html', error=str(e))

def generate_shap_plot(explainer, features, model_name):
    """Generate SHAP waterfall plot for individual prediction"""
    shap_values = explainer.shap_values(features)
    
    # For binary classification, use positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(shap.Explanation(
        values=shap_values[0],
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
    shap_values = explainer.shap_values(features)
    
    # For binary classification, use positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    contributions = []
    for i, feature_name in enumerate(FEATURE_NAMES):
        contributions.append({
            'feature': feature_name,
            'value': float(features[0][i]),
            'contribution': float(shap_values[0][i]),
            'impact': 'Increases Risk' if shap_values[0][i] > 0 else 'Decreases Risk'
        })
    
    # Sort by absolute contribution
    contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
    
    return contributions

@app.route("/about")
def about():
    """About page explaining the models and features"""
    return render_template('about.html')

if __name__ == "__main__":
    app.run(debug=True, port=5000)
