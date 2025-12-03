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
import re
from lime import lime_tabular
import xgboost as xgb

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
best_model = joblib.load('machine_learning/models/best_model.pkl')  # Random Forest
# best_xgb = joblib.load('machine_learning/models/best_xgb.pkl')  # XGBoost
best_xgb = xgb.XGBClassifier()
best_xgb.load_model('machine_learning/models/best_xgb.json')
model_nn = load_model('machine_learning/models/model_nn.keras')  # Neural Network
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

# Load encoders
sex_encoder = joblib.load('machine_learning/encoders/sex_encoder.pkl')
chestpain_encoder = joblib.load('machine_learning/encoders/chestpain_encoder.pkl')
exercise_encoder = joblib.load('machine_learning/encoders/exercise_encoder.pkl')
slope_encoder = joblib.load('machine_learning/encoders/slope_encoder.pkl')

# Feature names (after dropping RestingBP and RestingECG)
FEATURE_NAMES = ['Age', 'Sex', 'ChestPainType', 'Cholesterol', 'FastingBS', 
                 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']
FEATURE_INDEX_MAP = {name: idx for idx, name in enumerate(FEATURE_NAMES)}
CATEGORICAL_FEATURES = {'Sex', 'ChestPainType', 'FastingBS', 'ExerciseAngina', 'ST_Slope'}
PDP_BACKGROUND_DATA = None
FEATURE_VALUE_STATS = {}
FASTING_BS_LABELS = {
    0: 'Normal (<120 mg/dL)',
    1: 'High (>=120 mg/dL)'
}

# Audit logging
AUDIT_LOG_FILE = 'audit_log.jsonl'


# Create SHAP explainers for all models
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

# Create Neural Network explainer
try:
    # Try DeepExplainer first (optimized for neural networks)
    # Create background data for neural network
    np.random.seed(42)
    nn_background_data = np.zeros((50, 9))  # Smaller sample for faster computation
    nn_background_data[:, 0] = np.random.uniform(20, 80, 50)  # Age: 20-80
    nn_background_data[:, 1] = np.random.randint(0, 2, 50)  # Sex: 0-1
    nn_background_data[:, 2] = np.random.randint(0, 4, 50)  # ChestPainType: 0-3
    nn_background_data[:, 3] = np.random.uniform(100, 400, 50)  # Cholesterol: 100-400
    nn_background_data[:, 4] = np.random.randint(0, 2, 50)  # FastingBS: 0-1
    nn_background_data[:, 5] = np.random.uniform(60, 200, 50)  # MaxHR: 60-200
    nn_background_data[:, 6] = np.random.randint(0, 2, 50)  # ExerciseAngina: 0-1
    nn_background_data[:, 7] = np.random.uniform(0, 5, 50)  # Oldpeak: 0-5
    nn_background_data[:, 8] = np.random.randint(0, 3, 50)  # ST_Slope: 0-2
    
    explainer_nn = shap.DeepExplainer(model_nn, nn_background_data)
    print("✓ Successfully created Neural Network explainer using DeepExplainer")
except Exception as e:
    print(f"Warning: DeepExplainer failed for Neural Network: {e}")
    try:
        # Fallback to KernelExplainer
        def nn_predict_wrapper(X):
            """Wrapper function for Neural Network predict - returns positive class probabilities"""
            return model_nn.predict(X, verbose=0)[:, 1]  # Return probability of positive class
        
        explainer_nn = shap.KernelExplainer(nn_predict_wrapper, nn_background_data)
        print("✓ Successfully created Neural Network explainer using KernelExplainer (fallback)")
    except Exception as e2:
        print(f"Warning: Could not create Neural Network explainer: {e2}")
        explainer_nn = None

# Create LIME explainers for all models
try:
    # Prefer using the real training data as background to keep LIME grounded in actual patients
    heart_data_path = 'machine_learning/heart.csv'
    heart_df = pd.read_csv(heart_data_path)
    lime_background_df = heart_df[['Age', 'Sex', 'ChestPainType', 'Cholesterol', 'FastingBS',
                                   'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']].copy()

    # Encode categorical columns using the same encoders as the models
    lime_background_df['Sex'] = sex_encoder.transform(lime_background_df['Sex'])
    lime_background_df['ChestPainType'] = chestpain_encoder.transform(lime_background_df['ChestPainType'])
    lime_background_df['ExerciseAngina'] = exercise_encoder.transform(lime_background_df['ExerciseAngina'])
    lime_background_df['ST_Slope'] = slope_encoder.transform(lime_background_df['ST_Slope'])

    lime_background_data = lime_background_df.to_numpy(dtype=float)

    # If the dataset is large, sample a subset for performance
    if lime_background_data.shape[0] > 1000:
        np.random.seed(42)
        sample_idx = np.random.choice(lime_background_data.shape[0], 1000, replace=False)
        lime_background_data = lime_background_data[sample_idx]

    # Create LIME explainers
    lime_explainer_rf = lime_tabular.LimeTabularExplainer(
        lime_background_data,
        feature_names=FEATURE_NAMES,
        class_names=['No Heart Disease', 'Heart Disease'],
        mode='classification',
        random_state=42
    )
    
    lime_explainer_xgb = lime_tabular.LimeTabularExplainer(
        lime_background_data,
        feature_names=FEATURE_NAMES,
        class_names=['No Heart Disease', 'Heart Disease'],
        mode='classification',
        random_state=42
    )
    
    lime_explainer_nn = lime_tabular.LimeTabularExplainer(
        lime_background_data,
        feature_names=FEATURE_NAMES,
        class_names=['No Heart Disease', 'Heart Disease'],
        mode='classification',
        random_state=42
    )
    PDP_BACKGROUND_DATA = lime_background_data.copy()
    FEATURE_VALUE_STATS = {
        feature: {
            'min': float(lime_background_data[:, idx].min()),
            'max': float(lime_background_data[:, idx].max()),
            'unique': sorted(list(set(lime_background_data[:, idx].tolist())))
        }
        for idx, feature in enumerate(FEATURE_NAMES)
    }
    print("✓ Successfully created LIME explainers for all models using heart.csv background data")
except Exception as e:
    print(f"Warning: Could not create LIME explainers from heart.csv: {e}")
    try:
        # Fall back to synthetic background if real data loading fails
        np.random.seed(42)
        lime_background_data = np.zeros((100, 9))
        lime_background_data[:, 0] = np.random.uniform(20, 80, 100)  # Age: 20-80
        lime_background_data[:, 1] = np.random.randint(0, 2, 100)  # Sex: 0-1
        lime_background_data[:, 2] = np.random.randint(0, 4, 100)  # ChestPainType: 0-3
        lime_background_data[:, 3] = np.random.uniform(100, 400, 100)  # Cholesterol: 100-400
        lime_background_data[:, 4] = np.random.randint(0, 2, 100)  # FastingBS: 0-1
        lime_background_data[:, 5] = np.random.uniform(60, 200, 100)  # MaxHR: 60-200
        lime_background_data[:, 6] = np.random.randint(0, 2, 100)  # ExerciseAngina: 0-1
        lime_background_data[:, 7] = np.random.uniform(0, 5, 100)  # Oldpeak: 0-5
        lime_background_data[:, 8] = np.random.randint(0, 3, 100)  # ST_Slope: 0-2

        lime_explainer_rf = lime_tabular.LimeTabularExplainer(
            lime_background_data,
            feature_names=FEATURE_NAMES,
            class_names=['No Heart Disease', 'Heart Disease'],
            mode='classification',
            random_state=42
        )
        
        lime_explainer_xgb = lime_tabular.LimeTabularExplainer(
            lime_background_data,
            feature_names=FEATURE_NAMES,
            class_names=['No Heart Disease', 'Heart Disease'],
            mode='classification',
            random_state=42
        )
        
        lime_explainer_nn = lime_tabular.LimeTabularExplainer(
            lime_background_data,
            feature_names=FEATURE_NAMES,
            class_names=['No Heart Disease', 'Heart Disease'],
            mode='classification',
            random_state=42
        )
        print("✓ Successfully created LIME explainers using synthetic fallback data")
        PDP_BACKGROUND_DATA = lime_background_data.copy()
        FEATURE_VALUE_STATS = {
            feature: {
                'min': float(lime_background_data[:, idx].min()),
                'max': float(lime_background_data[:, idx].max()),
                'unique': sorted(list(set(lime_background_data[:, idx].tolist())))
            }
            for idx, feature in enumerate(FEATURE_NAMES)
        }
    except Exception as fallback_error:
        print(f"Warning: Could not create LIME explainers even with fallback data: {fallback_error}")
        lime_explainer_rf = None
        lime_explainer_xgb = None
        lime_explainer_nn = None



# Helper Functions
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


def generate_shap_plot(explainer, features, model_name):
    """Generate SHAP waterfall plot for individual prediction"""
    # print('in generate_shap_plot');
    # print('explainer', explainer);
    # print('features', features);
    # print('model_name', model_name);
    if explainer is None:
        return None
    try:
        shap_values = explainer.shap_values(features)
        # print('shap_values', shap_values);

        # For binary classification, use positive class
        if isinstance(shap_values, list):
            # List of arrays (one per class) - take positive class (index 1)
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        # Convert to numpy array if not already
        if not isinstance(shap_values, np.ndarray):
            shap_values = np.array(shap_values)
        
        # Handle 3D array (samples, features, classes) - take first sample and positive class
        if len(shap_values.shape) == 3:
            shap_values = shap_values[0, :, 1]  # First sample, all features, positive class
        # Handle 2D array (samples, features) - flatten to 1D for single prediction
        elif len(shap_values.shape) == 2:
            shap_values = shap_values[0]  # Take first row (single prediction)
        # Handle matrix of explanations - take the first row (single prediction)
        elif len(shap_values.shape) > 1 and shap_values.shape[0] > 1:
            shap_values = shap_values[0]
        
        # Ensure it's 1D
        if len(shap_values.shape) > 1:
            shap_values = shap_values.flatten()
            
    except Exception as e:
        print(f"Error processing SHAP values for {model_name}: {e}")
        return None

    # print(f"About to create SHAP plot for {model_name}")
    # print(f"shap_values shape: {shap_values.shape}")
    # print(f"shap_values: {shap_values}")
    # print(f"features[0]: {features[0]}")
    # print(f"FEATURE_NAMES: {FEATURE_NAMES}")
    # print(f"explainer.expected_value: {explainer.expected_value}")
    
    # Handle base_values for different explainer types
    try:
        exp_val = explainer.expected_value
        if isinstance(exp_val, np.ndarray):
            # Handle numpy array - use .item() to extract scalar
            if exp_val.ndim == 0:
                # 0-dimensional array (scalar)
                base_value = float(exp_val.item())
            elif exp_val.ndim == 1:
                # 1D array - for binary classification, take positive class if available
                if len(exp_val) > 1:
                    base_value = float(exp_val[1].item() if isinstance(exp_val[1], np.ndarray) else exp_val[1])
                else:
                    base_value = float(exp_val[0].item() if isinstance(exp_val[0], np.ndarray) else exp_val[0])
            else:
                # Multi-dimensional array - flatten and take first
                base_value = float(exp_val.flatten()[0].item() if isinstance(exp_val.flatten()[0], np.ndarray) else exp_val.flatten()[0])
        elif isinstance(exp_val, (list, tuple)):
            # Handle list/tuple
            if len(exp_val) > 1:
                base_value = float(exp_val[1])
            else:
                base_value = float(exp_val[0])
        else:
            # Scalar value
            base_value = float(exp_val)
    except Exception as e:
        # Fallback to default if anything goes wrong
        print(f"Warning: Could not parse expected_value for {model_name}: {e}, using default 0.5")
        base_value = 0.5
    
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

def generate_contribution_importance_plot(contributions, model_name, method_label='SHAP', top_n=9, bar_color='indianred'):
    """Generate feature importance plot from contribution data (SHAP/LIME)"""
    if not contributions:
        return None
    
    try:
        top_features = contributions[:top_n]
        feature_labels = [item['feature'] for item in top_features]
        importance_values = [abs(item['contribution']) for item in top_features]
    except (KeyError, TypeError) as e:
        print(f"Error preparing SHAP importance data for {model_name}: {e}")
        return None
    
    try:
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(feature_labels)), importance_values, color=bar_color)
        plt.xticks(range(len(feature_labels)), feature_labels, rotation=45, ha='right')
        plt.ylabel('Absolute Contribution', fontsize=12, fontweight='bold')
        plt.title(f'{method_label} Feature Importance - {model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        for bar, value in zip(bars, importance_values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f'{value:.3f}',
                ha='center',
                va='bottom',
                fontsize=9
            )
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.read()).decode()
        plt.close()
        return img_base64
    except Exception as e:
        print(f"Error generating {method_label} importance plot for {model_name}: {e}")
        plt.close()
        return None

def get_prediction_function(model_key):
    """Return a callable that outputs positive class probabilities for the specified model."""
    if model_key == 'rf':
        return lambda X: best_model.predict_proba(X)[:, 1]
    if model_key == 'xgb':
        return lambda X: best_xgb.predict_proba(X)[:, 1]
    if model_key == 'nn':
        return lambda X: model_nn.predict(X, verbose=0)[:, 1]
    raise ValueError(f"Unsupported model key: {model_key}")

def format_feature_value(feature_name, value):
    """Convert encoded feature values back to human-readable labels where possible."""
    rounded_value = int(round(value))
    if feature_name == 'Sex':
        try:
            return sex_encoder.inverse_transform([rounded_value])[0]
        except Exception:
            return str(rounded_value)
    if feature_name == 'ChestPainType':
        try:
            return chestpain_encoder.inverse_transform([rounded_value])[0]
        except Exception:
            return str(rounded_value)
    if feature_name == 'ExerciseAngina':
        try:
            return exercise_encoder.inverse_transform([rounded_value])[0]
        except Exception:
            return str(rounded_value)
    if feature_name == 'ST_Slope':
        try:
            return slope_encoder.inverse_transform([rounded_value])[0]
        except Exception:
            return str(rounded_value)
    if feature_name == 'FastingBS':
        return FASTING_BS_LABELS.get(rounded_value, str(rounded_value))
    try:
        return f"{float(value):.2f}"
    except Exception:
        return str(value)

def build_feature_grid(feature_name, grid_resolution=25):
    """Create grid values for PDP/ICE depending on feature type."""
    if feature_name not in FEATURE_VALUE_STATS:
        raise ValueError(f"No statistics available for feature {feature_name}")
    stats = FEATURE_VALUE_STATS[feature_name]
    if feature_name in CATEGORICAL_FEATURES:
        return stats['unique']
    min_val = stats['min']
    max_val = stats['max']
    if min_val == max_val:
        return [min_val]
    return np.linspace(min_val, max_val, grid_resolution).tolist()

def generate_pdp_ice_data(model_key, feature_name, grid_resolution=25, ice_count=20):
    """Generate Partial Dependence and ICE data for the given model and feature."""
    if PDP_BACKGROUND_DATA is None:
        raise ValueError("No background data available for PDP/ICE generation.")
    if feature_name not in FEATURE_INDEX_MAP:
        raise ValueError(f"Unsupported feature: {feature_name}")
    feature_idx = FEATURE_INDEX_MAP[feature_name]
    grid_values = build_feature_grid(feature_name, grid_resolution)
    predictor = get_prediction_function(model_key)
    sample_count = min(ice_count, PDP_BACKGROUND_DATA.shape[0])
    if sample_count <= 0:
        raise ValueError("Insufficient background data for PDP/ICE generation.")
    sample_indices = np.random.choice(PDP_BACKGROUND_DATA.shape[0], sample_count, replace=False)
    baseline_samples = PDP_BACKGROUND_DATA[sample_indices]
    ice_curves = []
    for sample in baseline_samples:
        curve = []
        for grid_val in grid_values:
            perturbed = sample.copy()
            perturbed[feature_idx] = grid_val
            prob = predictor(np.array([perturbed], dtype=float))[0]
            curve.append(float(prob))
        ice_curves.append(curve)
    pdp_curve = np.mean(np.array(ice_curves), axis=0).tolist()
    grid_labels = [format_feature_value(feature_name, val) for val in grid_values]
    return {
        "feature": feature_name,
        "model": model_key,
        "feature_type": "categorical" if feature_name in CATEGORICAL_FEATURES else "numeric",
        "grid": [float(val) for val in grid_values],
        "grid_labels": grid_labels,
        "pdp": pdp_curve,
        "ice": ice_curves
    }

def generate_feature_contribution(explainer, features, prediction):
    """Generate feature contribution table data"""
    if explainer is None:
        return []
    try:
        shap_values = explainer.shap_values(features)
        
        # For binary classification, use positive class
        if isinstance(shap_values, list):
            # List of arrays (one per class) - take positive class (index 1)
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        # Convert to numpy array if not already
        if not isinstance(shap_values, np.ndarray):
            shap_values = np.array(shap_values)
        
        # Handle 3D array (samples, features, classes) - take first sample and positive class
        if len(shap_values.shape) == 3:
            shap_values = shap_values[0, :, 1]  # First sample, all features, positive class
        # Handle 2D array (samples, features) - flatten to 1D for single prediction
        elif len(shap_values.shape) == 2:
            shap_values = shap_values[0]  # Take first row (single prediction)
        # Handle matrix of explanations - take the first row (single prediction)
        elif len(shap_values.shape) > 1 and shap_values.shape[0] > 1:
            shap_values = shap_values[0]
        
        # Ensure it's 1D
        if len(shap_values.shape) > 1:
            shap_values = shap_values.flatten()
    except Exception as e:
        print(f"Error processing SHAP values for feature contributions: {e}")
        return []
    
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

def generate_lime_explanation(lime_explainer, model, features, model_name):
    """Generate LIME explanation for a prediction"""
    if lime_explainer is None:
        print(f"LIME explainer is None for {model_name}")
        return None
    
    try:
        # Convert features to list format for LIME
        instance = features[0].tolist()
        print(f"Generating LIME explanation for {model_name} with instance: {instance[:3]}...")
        
        # Create prediction function wrapper based on model type
        if hasattr(model, 'predict_proba'):
            # For scikit-learn models (RF, XGBoost)
            def predict_fn(x):
                return model.predict_proba(x)
        else:
            # For neural network models
            def predict_fn(x):
                return model.predict(x, verbose=0)
        
        # Test prediction function
        test_pred = predict_fn(np.array([instance]))
        print(f"Test prediction shape for {model_name}: {test_pred.shape}")
        
        # Determine which labels to explain
        if len(test_pred.shape) == 1:
            # Single output probability (e.g., sigmoid) - treat as binary with positive class 1
            positive_label = 1
            predicted_label = 1 if test_pred[0] >= 0.5 else 0
        else:
            positive_label = 1 if test_pred.shape[1] > 1 else 0
            predicted_label = int(np.argmax(test_pred[0]))
        
        labels_to_explain = list({positive_label, predicted_label})
        labels_to_explain = [int(label) for label in labels_to_explain]
        if len(labels_to_explain) == 0:
            labels_to_explain = [positive_label]
        
        # Get explanation
        explanation = lime_explainer.explain_instance(
            np.array(instance),
            predict_fn,
            num_features=9,
            top_labels=max(1, len(labels_to_explain)),
            labels=labels_to_explain
        )
        print(f"LIME explanation generated successfully for {model_name}")
        
        available_labels = getattr(explanation, "available_labels", None)
        if callable(available_labels):
            available_labels = available_labels()
        if available_labels is None:
            available_labels = explanation.local_exp.keys()
        available_labels = list(available_labels)
        if not available_labels:
            available_labels = list(explanation.local_exp.keys())
        label_to_use = positive_label
        if label_to_use not in available_labels:
            if hasattr(explanation, "top_labels") and explanation.top_labels:
                for candidate_label in explanation.top_labels:
                    if candidate_label in available_labels:
                        label_to_use = candidate_label
                        break
            if label_to_use not in available_labels and available_labels:
                label_to_use = available_labels[0]
        print(f"LIME available labels for {model_name}: {available_labels}, using label {label_to_use}")
        
        # Convert to image
        try:
            fig = explanation.as_pyplot_figure(label=label_to_use)
            if fig is None:
                raise ValueError("LIME figure is None")
            
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
            img_buffer.seek(0)
            img_data = img_buffer.read()
            
            if len(img_data) == 0:
                raise ValueError("LIME figure image data is empty")
            
            img_base64 = base64.b64encode(img_data).decode()
            plt.close(fig)
            print(f"LIME plot converted to base64 for {model_name}, length: {len(img_base64)}")
        except Exception as fig_err:
            print(f"Error creating LIME figure for {model_name}: {fig_err}")
            import traceback
            traceback.print_exc()
            raise
        
        # Get feature contributions as list
        try:
            exp_list = explanation.as_list(label=label_to_use)
        except (KeyError, IndexError) as label_err:
            print(f"Label {label_to_use} not available in LIME explanation for {model_name}: {label_err}")
            try:
                exp_list = explanation.as_list()
            except Exception as fallback_err:
                print(f"Fallback label-less LIME explanation failed for {model_name}: {fallback_err}")
                exp_list = []
        contributions = []
        
        # Create mapping: try to match LIME feature names to our feature names
        # LIME might return names like "Age <= 50.00" or "Age > 50.00" or just indices
        feature_name_map = {}
        for i, feat_name in enumerate(FEATURE_NAMES):
            feature_name_map[feat_name.lower()] = (i, feat_name)
            feature_name_map[str(i)] = (i, feat_name)  # Also map by index
        
        # Convert LIME explanation to our format
        for feat_name, contribution in exp_list:
            # Extract base feature name (handle cases like "Age <= 50.00", "Age > 50.00", or just "Age")
            base_feature = feat_name.split(' <=')[0].split(' >')[0].split(' <')[0].split(' >=')[0].strip()
            
            # Try to find matching feature
            feature_idx = None
            feature_display_name = base_feature
            feature_value = 0.0
            
            # Try exact match first
            if base_feature.lower() in feature_name_map:
                feature_idx, feature_display_name = feature_name_map[base_feature.lower()]
            else:
                # Try partial match (e.g., "Age" in "Age <= 50.00")
                for feat_display_name in FEATURE_NAMES:
                    if feat_display_name.lower() in base_feature.lower() or base_feature.lower() in feat_display_name.lower():
                        feature_idx = FEATURE_NAMES.index(feat_display_name)
                        feature_display_name = feat_display_name
                        break
            
            # Get feature value if we found the index
            if feature_idx is not None:
                feature_value = float(instance[feature_idx])
            else:
                # Fallback: try to extract from the feature name string if it contains a number
                numbers = re.findall(r'\d+\.?\d*', base_feature)
                if numbers:
                    feature_value = float(numbers[0])
            
            contributions.append({
                'feature': feature_display_name,
                'value': feature_value,
                'contribution': float(contribution),
                'impact': 'Increases Risk' if contribution > 0 else 'Decreases Risk'
            })
        
        # Sort by absolute contribution
        contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
        
        return {
            'plot': img_base64,
            'contributions': contributions
        }
    except Exception as e:
        print(f"Error generating LIME explanation for {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


# API Routes
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

        # Generate SHAP plots
        shap_plot_rf = generate_shap_plot(explainer_rf, features, 'Random Forest')
        # print('shap_plot_rf', shap_plot_rf);
        shap_plot_xgb = generate_shap_plot(explainer_xgb, features, 'XGBoost') if explainer_xgb else None
        # print('shap_plot_xgb', shap_plot_xgb);
        shap_plot_nn = generate_shap_plot(explainer_nn, features, 'Neural Network') if explainer_nn else None
        
        # Generate feature importance plots
        feature_importance_rf = generate_feature_importance(best_model, 'Random Forest')
        feature_importance_xgb = generate_feature_importance(best_xgb, 'XGBoost')

        # Generate SHAP feature contributions
        feature_contrib_rf = generate_feature_contribution(explainer_rf, features, 1)
        feature_contrib_xgb = generate_feature_contribution(explainer_xgb, features, 1) if explainer_xgb else []
        feature_contrib_nn = generate_feature_contribution(explainer_nn, features, 1) if explainer_nn else []
        shap_importance_nn = generate_contribution_importance_plot(feature_contrib_nn, 'Neural Network', method_label='SHAP', bar_color='indianred') if feature_contrib_nn else None
        lime_contrib_nn = []

        # Generate LIME explanations
        print("Starting LIME explanation generation...")
        lime_explanation_rf = None
        lime_explanation_xgb = None
        lime_explanation_nn = None
        
        try:
            lime_explanation_rf = generate_lime_explanation(lime_explainer_rf, best_model, features, 'Random Forest')
            print(f"LIME RF result: {'Success' if lime_explanation_rf else 'Failed'}")
        except Exception as e:
            print(f"Error generating LIME for RF: {e}")
        
        if lime_explainer_xgb:
            try:
                lime_explanation_xgb = generate_lime_explanation(lime_explainer_xgb, best_xgb, features, 'XGBoost')
                print(f"LIME XGB result: {'Success' if lime_explanation_xgb else 'Failed'}")
            except Exception as e:
                print(f"Error generating LIME for XGB: {e}")
        
        if lime_explainer_nn:
            try:
                lime_explanation_nn = generate_lime_explanation(lime_explainer_nn, model_nn, features, 'Neural Network')
                print(f"LIME NN result: {'Success' if lime_explanation_nn else 'Failed'}")
                lime_contrib_nn = lime_explanation_nn['contributions'] if lime_explanation_nn and 'contributions' in lime_explanation_nn else []
            except Exception as e:
                print(f"Error generating LIME for NN: {e}")
        lime_importance_nn = generate_contribution_importance_plot(lime_contrib_nn, 'Neural Network', method_label='LIME', bar_color='seagreen') if lime_contrib_nn else None

        return jsonify({
            "ok": True,
            "feature_names": FEATURE_NAMES,
            "plots": {
                "shap_rf": shap_plot_rf,
                "shap_xgb": shap_plot_xgb,
                "shap_nn": shap_plot_nn,
                "lime_rf": lime_explanation_rf['plot'] if lime_explanation_rf else None,
                "lime_xgb": lime_explanation_xgb['plot'] if lime_explanation_xgb else None,
                "lime_nn": lime_explanation_nn['plot'] if lime_explanation_nn else None,
                "importance_rf": feature_importance_rf,
                "importance_xgb": feature_importance_xgb,
                "importance_nn": shap_importance_nn,
                "importance_nn_lime": lime_importance_nn
            },
            "contributions": {
                "random_forest": {
                    "shap": feature_contrib_rf,
                    "lime": lime_explanation_rf['contributions'] if lime_explanation_rf else []
                },
                "xgboost": {
                    "shap": feature_contrib_xgb,
                    "lime": lime_explanation_xgb['contributions'] if lime_explanation_xgb else []
                },
                "neural_net": {
                    "shap": feature_contrib_nn,
                    "lime": lime_explanation_nn['contributions'] if lime_explanation_nn else []
                }
            }
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

@app.route("/api/pdp", methods=["POST"])
def api_pdp():
    """Return Partial Dependence and ICE data for the requested feature/model."""
    try:
        payload = request.get_json(force=True)
        feature = payload.get("feature")
        model_key = payload.get("model", "rf")
        grid_resolution = int(payload.get("grid_resolution", 25))
        ice_count = int(payload.get("ice_count", 20))
        
        if feature not in FEATURE_NAMES:
            return jsonify({"ok": False, "error": f"Unsupported feature '{feature}'"}), 400
        if model_key not in {"rf", "xgb", "nn"}:
            return jsonify({"ok": False, "error": f"Unsupported model '{model_key}'"}), 400
        
        pdp_payload = generate_pdp_ice_data(model_key, feature, grid_resolution, ice_count)
        return jsonify({
            "ok": True,
            **pdp_payload
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

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
