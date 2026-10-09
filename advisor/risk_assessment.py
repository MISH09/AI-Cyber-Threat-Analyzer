ATTACK_SEVERITY = {
    "BENIGN": 0,
    "PortScan": 45,
    "Brute Force": 65,
    "Web Attack": 70,
    "Bot": 75,
    "DoS": 85,
    "DDoS": 95
}


def calculate_risk_score(attack_type, confidence):
    """
    Calculate a risk score from 0 to 100.

    attack_type: predicted attack class
    confidence: model confidence between 0 and 1
    """

    base_score = ATTACK_SEVERITY.get(attack_type, 50)

    # Confidence adjustment
    confidence_factor = confidence * 0.20

    score = base_score * (0.80 + confidence_factor)

    score = max(0, min(100, score))

    if score < 25:
        severity = "LOW"
    elif score < 50:
        severity = "MEDIUM"
    elif score < 75:
        severity = "HIGH"
    else:
        severity = "CRITICAL"

    return round(score, 2), severity