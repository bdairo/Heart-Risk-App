import joblib
import xgboost as xgb  # Fixed: correct import

# Load the existing pickled model
print("Loading existing XGBoost model...")
best_xgb = joblib.load('machine_learning/models/best_xgb.pkl')

# Re-save using XGBoost's native format
print("Re-saving model in XGBoost format...")
best_xgb.get_booster().save_model('machine_learning/models/best_xgb.json')
print("✓ Model saved as best_xgb.json")

# Also save as .ubj (binary format, faster to load)
best_xgb.get_booster().save_model('machine_learning/models/best_xgb.ubj')
print("✓ Model saved as best_xgb.ubj")