import pandas as pd
import joblib

from advisor.risk_assessment import calculate_risk_score
from advisor.recommendations import get_recommendations


MODEL_PATH = "saved_models/xgboost.pkl"
FEATURE_PATH = "preprocessing/X_test.csv"
MAPPING_PATH = "preprocessing/class_mapping.csv"


def load_model():
    return joblib.load(MODEL_PATH)


def load_class_mapping():
    mapping = pd.read_csv(MAPPING_PATH)

    return dict(
        zip(
            mapping["Encoded"],
            mapping["Attack_Type"]
        )
    )


def predict_sample(row):

    model = load_model()
    class_mapping = load_class_mapping()

    # Features used during model training
    expected_features = model.get_booster().feature_names

    row = row[expected_features]

    # Prediction
    prediction = model.predict(row)[0]

    # Probability of every class
    probabilities = model.predict_proba(row)[0]

    # Highest probability = model confidence
    confidence = float(max(probabilities))

    # Convert encoded class to attack name
    attack_type = class_mapping[int(prediction)]

    return attack_type, confidence


if __name__ == "__main__":

    # Load unseen test data
    X_test = pd.read_csv(FEATURE_PATH)

    # Take one unseen sample
    sample = X_test.iloc[[0]]

    # AI prediction
    attack_type, confidence = predict_sample(sample)

    # Risk assessment
    risk_score, severity = calculate_risk_score(
        attack_type,
        confidence
    )

    # Recommendations
    recommendations = get_recommendations(
        attack_type
    )

    # Display complete advisory
    print("\n========================================")
    print("       AI CYBER THREAT ADVISOR")
    print("========================================")

    print(f"\nAttack Type : {attack_type}")
    print(f"Confidence  : {confidence * 100:.2f}%")

    print(f"\nRisk Score  : {risk_score}/100")
    print(f"Severity    : {severity}")

    print("\nRecommendations:")

    for i, recommendation in enumerate(
        recommendations,
        1
    ):
        print(f"{i}. {recommendation}")

    print("\n========================================")