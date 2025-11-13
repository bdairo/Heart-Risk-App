import joblib
import numpy as np
import pandas as pd
from lime import lime_tabular
import matplotlib.pyplot as plt

FEATURE_NAMES = [
    'Age',
    'Sex',
    'ChestPainType',
    'Cholesterol',
    'FastingBS',
    'MaxHR',
    'ExerciseAngina',
    'Oldpeak',
    'ST_Slope',
]

CLASS_NAMES = ['No Heart Disease', 'Heart Disease']


def load_background_data():
    """Load and encode the training data so LIME sees realistic examples."""
    sex_encoder = joblib.load('encoders/sex_encoder.pkl')
    chestpain_encoder = joblib.load('encoders/chestpain_encoder.pkl')
    exercise_encoder = joblib.load('encoders/exercise_encoder.pkl')
    slope_encoder = joblib.load('encoders/slope_encoder.pkl')

    heart_df = pd.read_csv('heart.csv')
    background_df = heart_df[FEATURE_NAMES].copy()

    background_df['Sex'] = sex_encoder.transform(background_df['Sex'])
    background_df['ChestPainType'] = chestpain_encoder.transform(background_df['ChestPainType'])
    background_df['ExerciseAngina'] = exercise_encoder.transform(background_df['ExerciseAngina'])
    background_df['ST_Slope'] = slope_encoder.transform(background_df['ST_Slope'])

    return background_df.to_numpy(dtype=float)


def main():
    model = joblib.load('models/best_model.pkl')
    background = load_background_data()

    explainer = lime_tabular.LimeTabularExplainer(
        background,
        feature_names=FEATURE_NAMES,
        class_names=CLASS_NAMES,
        mode='classification',
        random_state=42,
    )

    sample = background[0]
    explanation = explainer.explain_instance(
        sample,
        model.predict_proba,
        num_features=len(FEATURE_NAMES),
        top_labels=1,
        labels=[0, 1],  # ask LIME to approximate both classes
    )

    available_labels = getattr(explanation, "available_labels", None)
    if callable(available_labels):
        available_labels = available_labels()
    if available_labels is None:
        available_labels = explanation.local_exp.keys()
    available_labels = list(available_labels)
    print(f"LIME returned explanations for labels: {available_labels}")

    target_label = 1 if 1 in available_labels else available_labels[0]
    print(f"Using label {target_label} for contributions\n")

    for feature, weight in explanation.as_list(label=target_label):
        print(f'{feature}: {weight:.4f}')


    fig = explanation.as_pyplot_figure(label=target_label)
    fig.tight_layout()

    # to view interactively
    plt.show()

    # or to save it
    fig.savefig("lime_explanation.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


if __name__ == '__main__':
    main()