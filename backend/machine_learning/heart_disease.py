# importing our dependancies

import keras
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, ConfusionMatrixDisplay
from keras.models import Sequential
from keras.layers import Dense
import warnings
warnings.filterwarnings('ignore')

# reading and viewing our data

df = pd.read_csv('heart.csv')

num_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
num_cols


# writting a function to clip outliers instead of dropping them
def clip_outliers(data):
  for col in num_cols:
    q1 = np.percentile(df[col], 25)
    q3 = np.percentile(df[col], 75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)

  return data

# calling our funtion to clip outliers in our data
df = clip_outliers(df)

# encoding each object column separately

sex_encoder = LabelEncoder()
chestpain_encoder = LabelEncoder()
restingecg_encoder = LabelEncoder()
exercise_encoder = LabelEncoder()
slope_encoder = LabelEncoder()

df['Sex'] = sex_encoder.fit_transform(df['Sex'])
df['ChestPainType'] = chestpain_encoder.fit_transform(df['ChestPainType'])
df['RestingECG'] = restingecg_encoder.fit_transform(df['RestingECG'])
df['ExerciseAngina'] = exercise_encoder.fit_transform(df['ExerciseAngina'])
df['ST_Slope'] = slope_encoder.fit_transform(df['ST_Slope'])

# dropping columns with the least correlations
df = df.drop(['RestingBP', 'RestingECG'], axis=1)


# saving our encoders to be used during deployment/ shap analysis

joblib.dump(sex_encoder, 'sex_encoder.pkl')
joblib.dump(chestpain_encoder, 'chestpain_encoder.pkl')
joblib.dump(exercise_encoder, 'exercise_encoder.pkl')
joblib.dump(slope_encoder, 'slope_encoder.pkl')

print("\n------All encoders have been saved------")

# separating features from target column
y = df['HeartDisease']
X = df.drop('HeartDisease', axis=1)

# splitting our data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


# creating a dictionary of our models and model names
#models = {
#    "Logistic Regression": LogisticRegression(),
#    "Decision Tree": DecisionTreeClassifier(),
#    "Random Forest": RandomForestClassifier(),
#    "XGBoost": xgb.XGBClassifier(),
#    "Support Vector Machine": SVC(kernel='linear'),
#    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=15)
#}

# creating an empty list for collection of our trained models
#trained_models = []

#for name, model in models.items():
#  model.fit(X_train, y_train)
#  pred = model.predict(X_test)
#  accuracy = accuracy_score(y_test, pred)
#  print(f"\n{name} Accuracy: {accuracy}")
#  precision = precision_score(y_test, pred)
#  print(f"{name} Precision: {precision}")
#  trained_models.append(model)





# parameter tunnig for Random Forest Classifier
print("\n------Training Random Forest Model------")
model = RandomForestClassifier()

# choosing which parameters to tune
params = {
    "n_estimators": [300, 350, 400],
    "max_depth": [3, 5, 7, 9],
    "min_samples_split": [6, 7, 8]
}

# initializing our grid object
grid = GridSearchCV(estimator=model, param_grid=params, cv=3, scoring='accuracy')
# fitting our grid object
grid.fit(X_train, y_train)


best_model = grid.best_estimator_

print("\nRandom Forest Score: {}".format(best_model.score(X_test, y_test)))


# parameter tuning for XGBClassifier
print("\n------Training XGB Model------")
model = xgb.XGBClassifier()

# choosing which parameters to tune
params = {
    "n_estimators": [300, 350, 400],
    "max_depth": [3, 5, 7, 9],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 0.9, 1.0]
}

# initializing our grid object
grid = GridSearchCV(estimator=model, param_grid=params, cv=3, scoring='accuracy')
# training our grid object
grid.fit(X_train, y_train)

best_xgb = grid.best_estimator_

print("\nXGB Best Score: {}".format(best_xgb.score(X_test, y_test)))


# initializing our deep learning model
model_nn = Sequential(
    [
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(2, activation='softmax')
    ]
)

# compiling our deep learning model
model_nn.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# training our deep learning model
hist = model_nn.fit(X_train, y_train, batch_size=5,epochs=10, validation_data=(X_test, y_test))

print("\nNN best score: {}".format(model_nn.evaluate(X_test, y_test)))



joblib.dump(best_model, 'best_model.pkl')
joblib.dump(best_xgb, 'best_xgb.pkl')
best_xgb.get_booster().save_model('best_xgb.json')
model_nn.save('model_nn.keras')
print("\n------All models have been saved------")
