from risk_assessment import calculate_risk_score
from recommendations import get_recommendations


attack_type = "DDoS"
confidence = 0.998

score, severity = calculate_risk_score(
    attack_type,
    confidence
)

recommendations = get_recommendations(attack_type)

print("\n======================================")
print("CYBER THREAT ADVISOR")
print("======================================")

print(f"Attack Type : {attack_type}")
print(f"Confidence  : {confidence * 100:.2f}%")
print(f"Risk Score  : {score}/100")
print(f"Severity    : {severity}")

print("\nRecommendations:")

for i, recommendation in enumerate(recommendations, 1):
    print(f"{i}. {recommendation}")