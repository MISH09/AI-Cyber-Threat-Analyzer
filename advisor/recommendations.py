RECOMMENDATIONS = {

    "BENIGN": [
        "No immediate malicious activity detected.",
        "Continue monitoring network traffic.",
        "Maintain standard firewall and security controls."
    ],

    "PortScan": [
        "Investigate the source IP address.",
        "Restrict unnecessary exposed ports.",
        "Review firewall and intrusion prevention rules.",
        "Monitor the source for repeated scanning activity."
    ],

    "Brute Force": [
        "Investigate repeated authentication failures.",
        "Enable account lockout or rate limiting.",
        "Enforce strong password policies.",
        "Enable multi-factor authentication where possible."
    ],

    "Web Attack": [
        "Inspect affected web application requests.",
        "Review web server and application logs.",
        "Apply appropriate input validation.",
        "Update and patch the affected application."
    ],

    "Bot": [
        "Investigate the affected host for malware.",
        "Isolate suspicious systems if required.",
        "Review outbound network connections.",
        "Perform endpoint security scanning."
    ],

    "DoS": [
        "Identify the source of excessive traffic.",
        "Apply rate limiting where appropriate.",
        "Review firewall and IDS rules.",
        "Monitor affected services for availability issues."
    ],

    "DDoS": [
        "Activate DDoS protection mechanisms.",
        "Apply traffic filtering and rate limiting.",
        "Identify and block malicious traffic sources.",
        "Contact the network or hosting provider if required."
    ]
}


def get_recommendations(attack_type):
    return RECOMMENDATIONS.get(
        attack_type,
        [
            "Investigate the detected network activity.",
            "Review relevant security logs.",
            "Apply appropriate firewall and access-control rules."
        ]
    )