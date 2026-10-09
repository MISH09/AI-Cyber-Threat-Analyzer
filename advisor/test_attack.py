import pandas as pd

from advisor.predict import predict_sample
from advisor.risk_assessment import calculate_risk_score
from advisor.recommendations import get_recommendations


X_test = pd.read_csv("preprocessing/X_test.csv")
y_test = pd.read_csv("preprocessing/y_test.csv")
mapping = pd.read_csv("preprocessing/class_mapping.csv")

# Find encoded value for DDoS
ddos_code = mapping.loc[
    mapping["Attack_Type"] == "DDoS",
    "Encoded"
].iloc[0]

# Find a DDoS sample
ddos_indices = y_test.index[
    y_test.iloc[:, 0] == ddos_code
]

sample_index = ddos_indices[0]

sample = X_test.iloc[[sample_index]]

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

print("\n========================================")
print("       AI CYBER THREAT ADVISOR")
print("========================================")

print(f"\nActual Attack Type    : DDoS")
print(f"Predicted Attack Type : {attack_type}")
print(f"Confidence             : {confidence * 100:.2f}%")

print(f"\nRisk Score             : {risk_score}/100")
print(f"Severity               : {severity}")

print("\nRecommendations:")

for i, recommendation in enumerate(
    recommendations,
    1
):
    print(f"{i}. {recommendation}")

print("\n========================================")