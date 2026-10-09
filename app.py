import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from advisor.risk_assessment import calculate_risk_score
from advisor.recommendations import get_recommendations


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI-Based Cyber Threat Advisor",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# COLOR PALETTE / CSS
# ============================================================

st.markdown(
    """
<style>

.stApp {
    background-color: #F7F3EC;
    color: #3B3028;
}

.main {
    background-color: #F7F3EC;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* ============================================================
   TOP STREAMLIT HEADER
   ============================================================ */

header[data-testid="stHeader"] {
    background-color: #F7F3EC !important;
    border-bottom: 1px solid #D8CEC2;
}

header[data-testid="stHeader"] button {
    color: #3B3028 !important;
}

div[data-testid="stToolbar"] {
    background-color: transparent !important;
}


/* ============================================================
   SIDEBAR
   ============================================================ */

section[data-testid="stSidebar"] {
    background-color: #E8DED1;
    border-right: 1px solid #D8CEC2;
}

section[data-testid="stSidebar"] * {
    color: #3B3028 !important;
}


/* ============================================================
   HEADINGS
   ============================================================ */

h1 {
    color: #3B3028 !important;
    font-weight: 700 !important;
}

h2 {
    color: #4A3830 !important;
    font-weight: 650 !important;
}

h3 {
    color: #7A3E48 !important;
    font-weight: 650 !important;
}

p {
    color: #6E6259;
}


/* ============================================================
   DIVIDERS
   ============================================================ */

hr {
    border-color: #D8CEC2 !important;
}


/* ============================================================
   METRIC CARDS
   ============================================================ */

div[data-testid="stMetric"] {
    background-color: #FFFDF8;
    border: 1px solid #D8CEC2;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 2px 8px rgba(59, 48, 40, 0.06);
}

div[data-testid="stMetricLabel"] {
    color: #6E6259 !important;
}

div[data-testid="stMetricValue"] {
    color: #3B3028 !important;
}


/* ============================================================
   SELECT BOX
   ============================================================ */

div[data-baseweb="select"] > div {
    background-color: #FFFDF8 !important;
    border: 1px solid #BFAF9F !important;
    border-radius: 8px !important;
}

div[data-baseweb="select"] span {
    color: #3B3028 !important;
}


/* ============================================================
   NORMAL BUTTONS
   ============================================================ */

.stButton > button {
    background-color: #B85C38 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    min-height: 45px;
}

.stButton > button p,
.stButton > button span {
    color: #FFFFFF !important;
}

.stButton > button:hover {
    background-color: #97482D !important;
    color: #FFFFFF !important;
}


/* ============================================================
   DOWNLOAD BUTTONS
   ============================================================ */

.stDownloadButton > button {
    background-color: #7A3E48 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    min-height: 45px;
}

.stDownloadButton > button p,
.stDownloadButton > button span {
    color: #FFFFFF !important;
}

.stDownloadButton > button:hover {
    background-color: #63323B !important;
    color: #FFFFFF !important;
}


/* ============================================================
   FILE UPLOADER
   ============================================================ */

section[data-testid="stFileUploader"] {
    background-color: #FFFDF8;
    border: 1px solid #D8CEC2;
    border-radius: 10px;
    padding: 10px;
}


/* ============================================================
   DATAFRAME
   ============================================================ */

div[data-testid="stDataFrame"] {
    border-radius: 10px;
}


/* ============================================================
   EXPANDER
   ============================================================ */

div[data-testid="stExpander"] {
    background-color: #FFFDF8;
    border: 1px solid #D8CEC2;
    border-radius: 10px;
}


/* ============================================================
   BORDERED CONTAINERS
   ============================================================ */

div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFDF8;
    border-color: #D8CEC2 !important;
    border-radius: 12px;
}


/* ============================================================
   PROGRESS BAR
   ============================================================ */

div[data-testid="stProgress"] > div > div {
    background-color: #B85C38 !important;
}


/* ============================================================
   ALERTS
   ============================================================ */

div[data-testid="stAlert"] {
    border-radius: 10px;
}


/* ============================================================
   RADIO BUTTONS
   ============================================================ */

div[role="radiogroup"] label {
    color: #3B3028 !important;
}


/* ============================================================
   LINKS
   ============================================================ */

a {
    color: #7A3E48 !important;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "saved_models"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

PREPROCESSING_DIR = os.path.join(
    BASE_DIR,
    "preprocessing"
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    model_files = {
        "XGBoost": "xgboost.pkl",
        "Random Forest": "random_forest.pkl",
        "Decision Tree": "decision_tree.pkl",
        "Isolation Forest": "isolation_forest.pkl"
    }

    loaded_models = {}

    for model_name, filename in model_files.items():

        path = os.path.join(
            MODEL_DIR,
            filename
        )

        if os.path.exists(path):

            try:
                loaded_models[model_name] = joblib.load(path)

            except Exception as error:

                st.warning(
                    f"Could not load {model_name}: {error}"
                )

    return loaded_models


models = load_models()


# ============================================================
# LOAD CLASS MAPPING
# ============================================================

@st.cache_data
def load_class_mapping():

    possible_paths = [
        os.path.join(
            PREPROCESSING_DIR,
            "class_mapping.csv"
        ),
        os.path.join(
            RESULTS_DIR,
            "class_mapping.csv"
        )
    ]

    for path in possible_paths:

        if os.path.exists(path):

            mapping = pd.read_csv(path)

            if (
                "Encoded" in mapping.columns
                and
                "Attack_Type" in mapping.columns
            ):

                return dict(
                    zip(
                        mapping["Encoded"].astype(int),
                        mapping["Attack_Type"].astype(str)
                    )
                )

    return {
        0: "BENIGN",
        1: "Bot",
        2: "Brute Force",
        3: "DDoS",
        4: "DoS",
        5: "PortScan",
        6: "Web Attack"
    }


class_mapping = load_class_mapping()


# ============================================================
# LOAD MODEL COMPARISON
# ============================================================

@st.cache_data
def load_model_comparison():

    path = os.path.join(
        RESULTS_DIR,
        "model_comparison.csv"
    )

    if os.path.exists(path):

        return pd.read_csv(path)

    return pd.DataFrame()


model_comparison = load_model_comparison()


# ============================================================
# LOAD CLEAN DATASET
# ============================================================

@st.cache_data
def load_dataset():

    path = os.path.join(
        PREPROCESSING_DIR,
        "clean_dataset.csv"
    )

    if os.path.exists(path):

        return pd.read_csv(path)

    return pd.DataFrame()


dataset = load_dataset()


# ============================================================
# MODEL FEATURE DETECTION
# ============================================================

def get_model_features(model):

    if hasattr(
        model,
        "feature_names_in_"
    ):

        return list(
            model.feature_names_in_
        )

    try:

        booster = model.get_booster()

        if booster.feature_names:

            return list(
                booster.feature_names
            )

    except Exception:

        pass

    return None


# ============================================================
# PREPARE INPUT DATA
# ============================================================

def prepare_input_data(
    dataframe,
    model
):

    expected_features = get_model_features(
        model
    )

    if expected_features is None:

        return dataframe.copy(), []

    missing_features = [
        feature
        for feature in expected_features
        if feature not in dataframe.columns
    ]

    if missing_features:

        return None, missing_features

    X = dataframe[
        expected_features
    ].copy()

    for column in X.columns:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    X = X.replace(
        [
            float("inf"),
            float("-inf")
        ],
        pd.NA
    )

    X = X.fillna(0)

    return X, []


# ============================================================
# ISOLATION FOREST RISK
# ============================================================

def calculate_anomaly_risk(
    decision_scores
):

    if len(decision_scores) == 0:

        return []

    minimum = min(
        decision_scores
    )

    maximum = max(
        decision_scores
    )

    if maximum == minimum:

        return [
            50.0
            for _ in decision_scores
        ]

    risk_scores = []

    for score in decision_scores:

        risk = (
            (maximum - score)
            /
            (maximum - minimum)
        ) * 100

        risk = max(
            0,
            min(
                100,
                risk
            )
        )

        risk_scores.append(
            risk
        )

    return risk_scores


def get_anomaly_severity(
    risk
):

    if risk >= 80:

        return "CRITICAL"

    elif risk >= 60:

        return "HIGH"

    elif risk >= 30:

        return "MEDIUM"

    return "LOW"


# ============================================================
# ANOMALY RECOMMENDATIONS
# ============================================================

def get_anomaly_recommendations():

    return [

        "Investigate the anomalous network activity.",

        "Review source and destination IP addresses.",

        "Check unusual ports, protocols and traffic volumes.",

        "Monitor the affected host for further suspicious activity.",

        "Apply firewall and access-control rules if malicious activity is confirmed."

    ]


# ============================================================
# THREAT SUMMARY
# ============================================================

def get_threat_summary(
    attack_type
):

    summaries = {

        "BENIGN":
        (
            "The selected model did not identify malicious "
            "network activity in this traffic."
        ),

        "ANOMALY":
        (
            "Isolation Forest identified unusual network "
            "behaviour that requires further investigation. "
            "An anomaly does not automatically confirm a "
            "specific attack type."
        ),

        "DDoS":
        (
            "The traffic indicates characteristics associated "
            "with Distributed Denial-of-Service activity. "
            "Such activity can overwhelm network, server or "
            "application resources."
        ),

        "DoS":
        (
            "The traffic indicates characteristics associated "
            "with Denial-of-Service activity that may affect "
            "service availability."
        ),

        "PortScan":
        (
            "The traffic indicates possible network reconnaissance "
            "involving scanning of ports and network services."
        ),

        "Brute Force":
        (
            "The traffic indicates repeated authentication "
            "attempts that may represent credential brute-force activity."
        ),

        "Bot":
        (
            "The traffic may indicate communication or behaviour "
            "associated with a compromised system or botnet."
        ),

        "Web Attack":
        (
            "The traffic may indicate malicious web activity "
            "such as brute force, XSS or SQL injection."
        )
    }

    return summaries.get(
        attack_type,
        "Suspicious network activity was detected."
    )


# ============================================================
# DETAILED THREAT INFORMATION
# ============================================================

def get_detailed_threat_information(
    attack_type
):

    threat_info = {

        "BENIGN": {

            "category":
                "Normal Network Traffic",

            "description":
                (
                    "The selected model did not identify "
                    "characteristics associated with the "
                    "trained malicious traffic classes."
                ),

            "why_it_matters":
                (
                    "Although the current traffic is classified "
                    "as benign, continuous monitoring is recommended "
                    "because network behaviour can change over time."
                ),

            "potential_impact": [
                "No immediate malicious impact was identified.",
                "Normal security monitoring should continue."
            ],

            "immediate_actions": [
                "Continue monitoring network traffic.",
                "Maintain firewall and access-control policies.",
                "Keep security logging enabled."
            ],

            "preventive_measures": [
                "Regularly update firewall rules.",
                "Maintain network monitoring.",
                "Keep systems and applications patched.",
                "Review unusual traffic patterns periodically."
            ],

            "monitoring": [
                "Monitor unusual traffic volumes.",
                "Monitor unexpected connections and ports.",
                "Review security logs for suspicious activity."
            ]
        },


        "DDoS": {

            "category":
                "Distributed Denial-of-Service Attack",

            "description":
                (
                    "The model identified traffic characteristics "
                    "associated with Distributed Denial-of-Service "
                    "activity. DDoS attacks attempt to overwhelm a "
                    "system, service or network with malicious traffic."
                ),

            "why_it_matters":
                (
                    "A successful DDoS attack can reduce service "
                    "availability and consume network, server or "
                    "application resources."
                ),

            "potential_impact": [
                "Service degradation or complete service unavailability.",
                "Network bandwidth exhaustion.",
                "Server resource exhaustion.",
                "Disruption of legitimate user traffic.",
                "Increased recovery and operational costs."
            ],

            "immediate_actions": [
                "Activate available DDoS protection mechanisms.",
                "Apply traffic filtering and rate limiting.",
                "Identify and block confirmed malicious sources.",
                "Monitor inbound traffic volume and connection rates.",
                "Contact the network or hosting provider if required."
            ],

            "preventive_measures": [
                "Deploy network-level DDoS protection.",
                "Configure appropriate traffic-rate thresholds.",
                "Maintain firewall and access-control rules.",
                "Use traffic monitoring and alerting.",
                "Prepare a documented DDoS response procedure."
            ],

            "monitoring": [
                "Monitor sudden increases in traffic volume.",
                "Monitor packet and connection rates.",
                "Monitor bandwidth utilization.",
                "Monitor repeated traffic from suspicious sources."
            ]
        },


        "DoS": {

            "category":
                "Denial-of-Service Attack",

            "description":
                (
                    "The model identified traffic characteristics "
                    "associated with Denial-of-Service activity. "
                    "The traffic may indicate an attempt to exhaust "
                    "resources or disrupt service availability."
                ),

            "why_it_matters":
                (
                    "A DoS attack can prevent legitimate users from "
                    "accessing a network service by consuming available resources."
                ),

            "potential_impact": [
                "Service disruption.",
                "Server resource exhaustion.",
                "Application performance degradation.",
                "Loss of availability for legitimate users."
            ],

            "immediate_actions": [
                "Investigate the source and destination traffic.",
                "Apply traffic filtering where appropriate.",
                "Configure rate limiting.",
                "Block confirmed malicious sources.",
                "Monitor affected services."
            ],

            "preventive_measures": [
                "Maintain firewall rules.",
                "Configure network traffic thresholds.",
                "Use intrusion detection and monitoring.",
                "Keep systems patched and hardened.",
                "Maintain an incident-response plan."
            ],

            "monitoring": [
                "Monitor CPU and memory utilization.",
                "Monitor network bandwidth.",
                "Monitor connection rates.",
                "Monitor service availability."
            ]
        },


        "PortScan": {

            "category":
                "Network Reconnaissance / Port Scanning",

            "description":
                (
                    "The model identified traffic characteristics "
                    "associated with port-scanning activity. Port "
                    "scanning can be used to discover accessible "
                    "services and potential entry points."
                ),

            "why_it_matters":
                (
                    "Reconnaissance can be an early stage of an attack. "
                    "Attackers may use discovered ports and services "
                    "to identify systems that can potentially be targeted."
                ),

            "potential_impact": [
                "Discovery of exposed network services.",
                "Identification of potentially vulnerable systems.",
                "Preparation for subsequent attacks.",
                "Increased exposure of network infrastructure."
            ],

            "immediate_actions": [
                "Identify the scanning source.",
                "Review destination ports being scanned.",
                "Check whether the source is authorized.",
                "Block unauthorized scanning sources where appropriate.",
                "Review exposed services and open ports."
            ],

            "preventive_measures": [
                "Close unnecessary network ports.",
                "Restrict administrative services.",
                "Use network segmentation.",
                "Maintain firewall rules.",
                "Monitor repeated reconnaissance attempts."
            ],

            "monitoring": [
                "Monitor repeated connections to multiple ports.",
                "Monitor connections from unknown sources.",
                "Monitor access to administrative services.",
                "Review firewall logs."
            ]
        },


        "Brute Force": {

            "category":
                "Credential / Authentication Attack",

            "description":
                (
                    "The model identified traffic characteristics "
                    "associated with brute-force activity. Brute-force "
                    "attacks involve repeated authentication attempts "
                    "intended to discover valid credentials."
                ),

            "why_it_matters":
                (
                    "Successful credential attacks can provide "
                    "unauthorized access to accounts, systems and "
                    "sensitive resources."
                ),

            "potential_impact": [
                "Unauthorized account access.",
                "Credential compromise.",
                "Data exposure.",
                "Account takeover.",
                "Potential lateral movement."
            ],

            "immediate_actions": [
                "Investigate repeated authentication attempts.",
                "Review affected accounts.",
                "Block or restrict confirmed malicious sources.",
                "Reset compromised credentials where necessary.",
                "Enable multi-factor authentication."
            ],

            "preventive_measures": [
                "Implement multi-factor authentication.",
                "Use strong password policies.",
                "Apply account lockout or rate limiting.",
                "Restrict remote administrative access.",
                "Monitor authentication failures."
            ],

            "monitoring": [
                "Monitor repeated failed logins.",
                "Monitor unusual login locations.",
                "Monitor unusual authentication times.",
                "Monitor repeated attempts against multiple accounts."
            ]
        },


        "Bot": {

            "category":
                "Potential Bot / Compromised Host Activity",

            "description":
                (
                    "The model identified network characteristics "
                    "associated with bot-related traffic. Such activity "
                    "may indicate a compromised system communicating "
                    "with external infrastructure."
                ),

            "why_it_matters":
                (
                    "Compromised hosts can be controlled remotely and "
                    "may participate in malicious activities such as "
                    "scanning, data transfer or coordinated attacks."
                ),

            "potential_impact": [
                "System compromise.",
                "Unauthorized communication.",
                "Participation in coordinated attacks.",
                "Data theft or suspicious outbound traffic.",
                "Potential lateral movement."
            ],

            "immediate_actions": [
                "Identify the potentially affected host.",
                "Investigate outbound network connections.",
                "Review endpoint and security logs.",
                "Isolate the host if compromise is confirmed.",
                "Perform malware and endpoint security analysis."
            ],

            "preventive_measures": [
                "Maintain endpoint protection.",
                "Keep operating systems and applications patched.",
                "Restrict unnecessary outbound communication.",
                "Use network segmentation.",
                "Monitor unusual DNS and outbound connections."
            ],

            "monitoring": [
                "Monitor unusual outbound traffic.",
                "Monitor repeated communication with unknown destinations.",
                "Monitor abnormal DNS requests.",
                "Monitor unusual process and network activity."
            ]
        },


        "Web Attack": {

            "category":
                "Web Application Attack",

            "description":
                (
                    "The model identified traffic characteristics "
                    "associated with web-based attack activity. "
                    "The trained dataset includes web attack categories "
                    "such as brute force, XSS and SQL injection."
                ),

            "why_it_matters":
                (
                    "Web application attacks may target application "
                    "logic, authentication mechanisms or backend databases "
                    "and can potentially expose sensitive information."
                ),

            "potential_impact": [
                "Unauthorized access.",
                "Data exposure.",
                "Application compromise.",
                "Database manipulation or extraction.",
                "Account compromise."
            ],

            "immediate_actions": [
                "Review suspicious HTTP requests.",
                "Inspect web-server and application logs.",
                "Block confirmed malicious requests.",
                "Check authentication and access logs.",
                "Investigate affected application endpoints."
            ],

            "preventive_measures": [
                "Use secure input validation.",
                "Apply parameterized database queries.",
                "Implement web application firewall controls.",
                "Keep web frameworks and dependencies updated.",
                "Apply secure authentication and authorization."
            ],

            "monitoring": [
                "Monitor unusual HTTP request patterns.",
                "Monitor repeated authentication failures.",
                "Monitor suspicious query parameters.",
                "Review web-server security logs."
            ]
        },


        "ANOMALY": {

            "category":
                "Network Anomaly",

            "description":
                (
                    "Isolation Forest identified network behaviour "
                    "that differs from the normal traffic patterns "
                    "learned during training. An anomaly does not "
                    "automatically confirm a specific attack type."
                ),

            "why_it_matters":
                (
                    "Unusual traffic can represent previously unseen "
                    "attacks, misconfiguration, abnormal application "
                    "behaviour or other network conditions requiring investigation."
                ),

            "potential_impact": [
                "Previously unseen malicious activity.",
                "Compromised host behaviour.",
                "Network misconfiguration.",
                "Unusual resource consumption.",
                "Potential security incident."
            ],

            "immediate_actions": [
                "Investigate the anomalous traffic.",
                "Review source and destination addresses.",
                "Examine unusual ports and protocols.",
                "Review firewall and system logs.",
                "Investigate the affected host if suspicious behaviour continues."
            ],

            "preventive_measures": [
                "Maintain network monitoring.",
                "Establish normal traffic baselines.",
                "Review firewall and access-control policies.",
                "Keep endpoint and network security systems updated.",
                "Investigate recurring anomalies."
            ],

            "monitoring": [
                "Monitor recurrence of the anomaly.",
                "Monitor source and destination addresses.",
                "Monitor unusual traffic volumes.",
                "Monitor changes in network behaviour."
            ]
        }
    }

    return threat_info.get(
        attack_type,
        threat_info["ANOMALY"]
    )


# ============================================================
# RISK INTERPRETATION
# ============================================================

def get_risk_interpretation(
    risk_score
):

    if risk_score >= 80:

        return (
            "The calculated risk is very high. "
            "The detected activity should be treated as a "
            "priority security event and investigated immediately."
        )

    elif risk_score >= 60:

        return (
            "The calculated risk is high. "
            "The activity should be investigated promptly "
            "and appropriate security controls should be applied."
        )

    elif risk_score >= 30:

        return (
            "The calculated risk is moderate. "
            "Further investigation and monitoring are recommended."
        )

    else:

        return (
            "The calculated risk is low. "
            "Normal security monitoring should continue."
        )


# ============================================================
# GET CONFIDENCE
# ============================================================

def get_confidence_from_row(
    row
):

    if "Confidence" in row.index:

        try:

            return float(
                row["Confidence"]
            )

        except Exception:

            return None

    return None


# ============================================================
# SUPERVISED MODEL PREDICTION
# ============================================================

def run_supervised_prediction(
    model,
    X
):

    predictions = model.predict(
        X
    )

    probabilities = model.predict_proba(
        X
    )

    results = []

    for index in range(
        len(predictions)
    ):

        encoded_prediction = int(
            predictions[index]
        )

        attack_type = class_mapping.get(
            encoded_prediction,
            "Unknown"
        )

        confidence = (
            float(
                max(
                    probabilities[index]
                )
            )
            * 100
        )

        risk_score, severity = (
            calculate_risk_score(
                attack_type,
                confidence
            )
        )

        results.append({

            "Attack_Type":
                attack_type,

            "Confidence":
                round(
                    confidence,
                    2
                ),

            "Risk_Score":
                round(
                    risk_score,
                    2
                ),

            "Severity":
                severity
        })

    return pd.DataFrame(
        results
    )


# ============================================================
# ISOLATION FOREST PREDICTION
# ============================================================

def run_isolation_forest(
    model,
    X
):

    predictions = model.predict(
        X
    )

    decision_scores = (
        model.decision_function(
            X
        )
    )

    risk_scores = calculate_anomaly_risk(
        decision_scores
    )

    results = []

    for index in range(
        len(predictions)
    ):

        if predictions[index] == -1:

            attack_type = "ANOMALY"

            risk_score = (
                risk_scores[index]
            )

            severity = (
                get_anomaly_severity(
                    risk_score
                )
            )

        else:

            attack_type = "BENIGN"

            risk_score = 0.0

            severity = "LOW"

        results.append({

            "Attack_Type":
                attack_type,

            "Anomaly_Score":
                round(
                    risk_score,
                    2
                ),

            "Risk_Score":
                round(
                    risk_score,
                    2
                ),

            "Severity":
                severity,

            "Decision_Score":
                round(
                    float(
                        decision_scores[index]
                    ),
                    4
                )
        })

    return pd.DataFrame(
        results
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "AI Cyber Threat Advisor"
)

st.sidebar.write(
    "AI-based network threat detection, "
    "classification, risk assessment and "
    "security recommendations."
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Threat Detection",
        "Dataset Analysis",
        "Model Performance",
        "Threat Advisory"
    ]
)

st.sidebar.divider()

st.sidebar.subheader(
    "Models Available"
)

st.sidebar.write(
    "XGBoost"
)

st.sidebar.write(
    "Random Forest"
)

st.sidebar.write(
    "Decision Tree"
)

st.sidebar.write(
    "Isolation Forest"
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.title(
        "AI-Based Cyber Threat Advisor"
    )

    st.write(
        "An intelligent system for detecting, "
        "classifying and assessing cyber threats."
    )

    st.divider()

    # --------------------------------------------------------
    # DATASET METRICS
    # --------------------------------------------------------

    if not dataset.empty:

        total_flows = len(
            dataset
        )

        if "Attack_Type" in dataset.columns:

            benign_flows = (
                dataset["Attack_Type"]
                .astype(str)
                .str.upper()
                .eq("BENIGN")
                .sum()
            )

            threat_flows = (
                total_flows -
                benign_flows
            )

            attack_classes = (
                dataset["Attack_Type"]
                .nunique()
            )

        else:

            benign_flows = 0
            threat_flows = 0
            attack_classes = 0

    else:

        total_flows = 0
        benign_flows = 0
        threat_flows = 0
        attack_classes = 0

    col1, col2, col3, col4 = st.columns(
        4
    )

    col1.metric(
        "Network Flows",
        f"{total_flows:,}"
    )

    col2.metric(
        "Benign Traffic",
        f"{benign_flows:,}"
    )

    col3.metric(
        "Threat Samples",
        f"{threat_flows:,}"
    )

    col4.metric(
        "Attack Classes",
        attack_classes
    )

    st.write("")

    # --------------------------------------------------------
    # ATTACK DISTRIBUTION
    # --------------------------------------------------------

    col1, col2 = st.columns(
        [1.3, 1]
    )

    with col1:

        st.subheader(
            "Attack Distribution"
        )

        if (
            not dataset.empty
            and
            "Attack_Type" in dataset.columns
        ):

            distribution = (
                dataset[
                    "Attack_Type"
                ].value_counts()
            )

            fig, ax = plt.subplots(
                figsize=(9, 5)
            )

            ax.bar(
                distribution.index,
                distribution.values,
                color="#B85C38"
            )

            ax.set_xlabel(
                "Attack Type"
            )

            ax.set_ylabel(
                "Number of Samples"
            )

            ax.tick_params(
                axis="x",
                rotation=45
            )

            ax.grid(
                axis="y",
                alpha=0.2
            )

            fig.tight_layout()

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)

        else:

            st.info(
                "Dataset not found."
            )

    # --------------------------------------------------------
    # PIPELINE
    # --------------------------------------------------------

    with col2:

        st.subheader(
            "AI Detection Pipeline"
        )

        with st.container(
            border=True
        ):

            st.markdown(
                "**1. Network Data**"
            )

            st.write(
                "Network traffic and security features "
                "are provided to the system."
            )

            st.markdown(
                "**2. Preprocessing**"
            )

            st.write(
                "Missing, infinite and irrelevant values "
                "are handled."
            )

            st.markdown(
                "**3. AI Detection**"
            )

            st.write(
                "The selected machine-learning model "
                "analyzes the traffic."
            )

            st.markdown(
                "**4. Attack Classification**"
            )

            st.write(
                "Supervised models identify attack categories."
            )

            st.markdown(
                "**5. Risk Assessment**"
            )

            st.write(
                "A risk score and severity are calculated."
            )

            st.markdown(
                "**6. Threat Advisory**"
            )

            st.write(
                "Security recommendations are generated."
            )

    # --------------------------------------------------------
    # MODEL SNAPSHOT
    # --------------------------------------------------------

    st.subheader(
        "Model Performance Snapshot"
    )

    if not model_comparison.empty:

        st.dataframe(
            model_comparison,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "Model comparison results are not available."
        )


# ============================================================
# THREAT DETECTION
# ============================================================

elif page == "Threat Detection":

    st.title(
        "Threat Detection"
    )

    st.write(
        "Select an AI model and upload network traffic data for analysis."
    )

    st.divider()

    # --------------------------------------------------------
    # MODEL SELECTION
    # --------------------------------------------------------

    available_models = [
        name
        for name in [
            "XGBoost",
            "Random Forest",
            "Decision Tree",
            "Isolation Forest"
        ]
        if name in models
    ]

    if not available_models:

        st.error(
            "No trained models were found in the saved_models folder."
        )

        st.stop()

    selected_model_name = st.selectbox(
        "Select AI Model",
        available_models
    )

    selected_model = models[
        selected_model_name
    ]

    with st.container(
        border=True
    ):

        st.caption(
            "SELECTED AI MODEL"
        )

        st.subheader(
            selected_model_name
        )

        if selected_model_name == "Isolation Forest":

            st.info(
                "Isolation Forest performs anomaly detection. "
                "It identifies traffic as BENIGN or ANOMALY "
                "instead of predicting a specific attack class."
            )

        else:

            st.info(
                "This supervised model predicts one of the "
                "seven trained attack categories and provides "
                "a model confidence score."
            )

    # --------------------------------------------------------
    # UPLOAD
    # --------------------------------------------------------

    st.subheader(
        "Upload Network Traffic"
    )

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            input_df = pd.read_csv(
                uploaded_file
            )

            st.success(
                f"File loaded successfully: "
                f"{len(input_df):,} rows"
            )

            with st.expander(
                "Preview Uploaded Data"
            ):

                st.dataframe(
                    input_df.head(10),
                    use_container_width=True
                )

            if st.button(
                "Analyze Network Traffic",
                use_container_width=True
            ):

                with st.spinner(
                    f"Analyzing using {selected_model_name}..."
                ):

                    X, missing_features = (
                        prepare_input_data(
                            input_df,
                            selected_model
                        )
                    )

                    if X is None:

                        st.error(
                            "The uploaded CSV is missing "
                            "required features."
                        )

                        with st.expander(
                            "Missing Features"
                        ):

                            st.write(
                                missing_features
                            )

                        st.stop()

                    if (
                        selected_model_name
                        ==
                        "Isolation Forest"
                    ):

                        result_df = (
                            run_isolation_forest(
                                selected_model,
                                X
                            )
                        )

                    else:

                        result_df = (
                            run_supervised_prediction(
                                selected_model,
                                X
                            )
                        )

                    result_df.insert(
                        0,
                        "Model",
                        selected_model_name
                    )

                    st.session_state[
                        "detection_results"
                    ] = result_df

                    st.session_state[
                        "selected_model"
                    ] = selected_model_name

                    st.session_state[
                        "input_data"
                    ] = input_df

                st.success(
                    "Analysis completed successfully."
                )

        except Exception as error:

            st.error(
                f"Analysis failed: {error}"
            )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    if "detection_results" in st.session_state:

        result_df = st.session_state[
            "detection_results"
        ]

        current_model = (
            st.session_state.get(
                "selected_model",
                selected_model_name
            )
        )

        st.divider()

        st.subheader(
            "Detection Results"
        )

        with st.container(
            border=True
        ):

            st.caption(
                "MODEL USED"
            )

            st.subheader(
                current_model
            )

        total = len(
            result_df
        )

        benign_count = (
            result_df[
                "Attack_Type"
            ]
            .eq("BENIGN")
            .sum()
        )

        threat_count = (
            total -
            benign_count
        )

        highest_risk = float(
            result_df[
                "Risk_Score"
            ].max()
        )

        col1, col2, col3, col4 = st.columns(
            4
        )

        col1.metric(
            "Total Flows",
            f"{total:,}"
        )

        col2.metric(
            "Benign",
            f"{benign_count:,}"
        )

        col3.metric(
            "Threat / Anomaly",
            f"{threat_count:,}"
        )

        col4.metric(
            "Highest Risk",
            f"{highest_risk:.1f}/100"
        )

        # ----------------------------------------------------
        # ALERT
        # ----------------------------------------------------

        if threat_count > 0:

            highest_row = result_df.loc[
                result_df[
                    "Risk_Score"
                ].idxmax()
            ]

            highest_attack = (
                highest_row[
                    "Attack_Type"
                ]
            )

            highest_severity = (
                highest_row[
                    "Severity"
                ]
            )

            if highest_attack == "ANOMALY":

                alert_text = (
                    "Isolation Forest identified anomalous "
                    "network behaviour."
                )

            else:

                alert_text = (
                    f"Highest detected threat: "
                    f"{highest_attack}"
                )

            if highest_severity == "CRITICAL":

                st.error(
                    f"{alert_text} | "
                    f"Severity: CRITICAL | "
                    f"Risk: {highest_risk:.2f}/100"
                )

            elif highest_severity == "HIGH":

                st.error(
                    f"{alert_text} | "
                    f"Severity: HIGH | "
                    f"Risk: {highest_risk:.2f}/100"
                )

            elif highest_severity == "MEDIUM":

                st.warning(
                    f"{alert_text} | "
                    f"Severity: MEDIUM | "
                    f"Risk: {highest_risk:.2f}/100"
                )

            else:

                st.info(
                    f"{alert_text} | "
                    f"Severity: LOW"
                )

        else:

            st.success(
                "No malicious activity was detected."
            )

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        st.subheader(
            "Detection Summary"
        )

        attack_counts = (
            result_df[
                "Attack_Type"
            ].value_counts()
        )

        if len(attack_counts) > 0:

            most_common_attack = (
                attack_counts.index[0]
            )

            most_common_count = (
                attack_counts.iloc[0]
            )

            if most_common_attack == "BENIGN":

                st.info(
                    f"Most traffic was classified as "
                    f"BENIGN ({most_common_count:,} flows)."
                )

            else:

                st.warning(
                    f"Most frequent detection: "
                    f"{most_common_attack} "
                    f"({most_common_count:,} flows)."
                )

        # ----------------------------------------------------
        # CHART
        # ----------------------------------------------------

        st.subheader(
            "Detected Attack Distribution"
        )

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        ax.barh(
            attack_counts.index[::-1],
            attack_counts.values[::-1],
            color="#7A3E48"
        )

        ax.set_xlabel(
            "Number of Flows"
        )

        ax.set_ylabel(
            "Detection"
        )

        ax.grid(
            axis="x",
            alpha=0.2
        )

        fig.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

        # ----------------------------------------------------
        # TABLE
        # ----------------------------------------------------

        st.subheader(
            "Detailed Detection Table"
        )

        st.dataframe(
            result_df,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # DOWNLOAD
        # ----------------------------------------------------

        csv_data = result_df.to_csv(
            index=False
        )

        st.download_button(
            "Download Detection Results",
            data=csv_data,
            file_name="threat_detection_results.csv",
            mime="text/csv",
            use_container_width=True
        )


# ============================================================
# DATASET ANALYSIS
# ============================================================

elif page == "Dataset Analysis":

    st.title(
        "Dataset Analysis"
    )

    st.write(
        "Overview of the processed CIC-IDS2017 dataset "
        "used for model development."
    )

    st.divider()

    if dataset.empty:

        st.error(
            "clean_dataset.csv was not found."
        )

    else:

        total_rows = len(
            dataset
        )

        total_columns = len(
            dataset.columns
        )

        if "Attack_Type" in dataset.columns:

            classes = (
                dataset[
                    "Attack_Type"
                ].nunique()
            )

        else:

            classes = 0

        col1, col2, col3 = st.columns(
            3
        )

        col1.metric(
            "Rows",
            f"{total_rows:,}"
        )

        col2.metric(
            "Features / Columns",
            total_columns
        )

        col3.metric(
            "Attack Classes",
            classes
        )

        st.write("")

        if "Attack_Type" in dataset.columns:

            distribution = (
                dataset[
                    "Attack_Type"
                ]
                .value_counts()
                .reset_index()
            )

            distribution.columns = [
                "Attack_Type",
                "Count"
            ]

            st.subheader(
                "Class Distribution"
            )

            st.dataframe(
                distribution,
                use_container_width=True,
                hide_index=True
            )

            fig, ax = plt.subplots(
                figsize=(10, 5)
            )

            ax.bar(
                distribution[
                    "Attack_Type"
                ],
                distribution[
                    "Count"
                ],
                color="#B85C38"
            )

            ax.set_xlabel(
                "Attack Type"
            )

            ax.set_ylabel(
                "Number of Samples"
            )

            ax.tick_params(
                axis="x",
                rotation=45
            )

            ax.grid(
                axis="y",
                alpha=0.2
            )

            fig.tight_layout()

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)

        st.subheader(
            "Dataset Preview"
        )

        st.dataframe(
            dataset.head(20),
            use_container_width=True
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.title(
        "Model Performance"
    )

    st.write(
        "Comparison of the trained AI models."
    )

    st.divider()

    if model_comparison.empty:

        st.warning(
            "Model comparison results are not available."
        )

    else:

        st.subheader(
            "Performance Comparison"
        )

        st.dataframe(
            model_comparison,
            use_container_width=True,
            hide_index=True
        )

        required_columns = [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1_Score"
        ]

        if all(
            column in model_comparison.columns
            for column in required_columns
        ):

            st.subheader(
                "Model Metrics"
            )

            metrics_df = (
                model_comparison
                .set_index("Model")
                [
                    [
                        "Accuracy",
                        "Precision",
                        "Recall",
                        "F1_Score"
                    ]
                ]
            )

            fig, ax = plt.subplots(
                figsize=(11, 6)
            )

            metrics_df.plot(
                kind="barh",
                ax=ax,
                color=[
                    "#B85C38",
                    "#7A3E48",
                    "#8F7465",
                    "#667C68"
                ]
            )

            ax.set_xlabel(
                "Score"
            )

            ax.set_xlim(
                0,
                1
            )

            ax.grid(
                axis="x",
                alpha=0.2
            )

            fig.tight_layout()

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)

        st.subheader(
            "Model Roles"
        )

        col1, col2 = st.columns(
            2
        )

        with col1:

            with st.container(
                border=True
            ):

                st.subheader(
                    "Supervised Models"
                )

                st.markdown(
                    "**XGBoost**"
                )

                st.write(
                    "Gradient-boosted decision-tree model "
                    "used for multiclass attack classification."
                )

                st.markdown(
                    "**Random Forest**"
                )

                st.write(
                    "Ensemble of decision trees used for "
                    "multiclass attack classification."
                )

                st.markdown(
                    "**Decision Tree**"
                )

                st.write(
                    "Tree-based classification model that "
                    "provides interpretable decision rules."
                )

        with col2:

            with st.container(
                border=True
            ):

                st.subheader(
                    "Isolation Forest"
                )

                st.write(
                    "Isolation Forest detects unusual network "
                    "behaviour by identifying observations "
                    "that differ from normal traffic."
                )

                st.write(
                    "It produces a BENIGN or ANOMALY result "
                    "rather than predicting the seven attack classes."
                )

                st.info(
                    "Isolation Forest performs a different task "
                    "from the supervised multiclass classifiers."
                )


# ============================================================
# THREAT ADVISORY
# ============================================================

elif page == "Threat Advisory":

    st.title(
        "Threat Advisory"
    )

    st.write(
        "Detailed security assessment, risk interpretation "
        "and recommended response actions."
    )

    st.divider()

    # --------------------------------------------------------
    # CHECK FOR DETECTION
    # --------------------------------------------------------

    if "detection_results" not in st.session_state:

        st.info(
            "Run a threat detection analysis first "
            "to generate the threat advisory."
        )

    else:

        result_df = st.session_state[
            "detection_results"
        ]

        current_model = st.session_state.get(
            "selected_model",
            "Unknown"
        )

        # ----------------------------------------------------
        # HIGHEST RISK RESULT
        # ----------------------------------------------------

        highest_risk = float(
            result_df[
                "Risk_Score"
            ].max()
        )

        highest_row = result_df.loc[
            result_df[
                "Risk_Score"
            ].idxmax()
        ]

        attack_type = (
            highest_row[
                "Attack_Type"
            ]
        )

        severity = (
            highest_row[
                "Severity"
            ]
        )

        confidence = get_confidence_from_row(
            highest_row
        )

        threat_info = (
            get_detailed_threat_information(
                attack_type
            )
        )

        # ----------------------------------------------------
        # REPORT HEADER
        # ----------------------------------------------------

        with st.container(
            border=True
        ):

            st.caption(
                "THREAT ADVISORY REPORT"
            )

            st.subheader(
                threat_info["category"]
            )

            st.write(
                f"Analysis performed using **{current_model}**."
            )

        # ----------------------------------------------------
        # EXECUTIVE SUMMARY
        # ----------------------------------------------------

        st.subheader(
            "Executive Summary"
        )

        st.write(
            threat_info["description"]
        )

        st.write(
            get_risk_interpretation(
                highest_risk
            )
        )

        # ----------------------------------------------------
        # SECURITY ASSESSMENT
        # ----------------------------------------------------

        st.subheader(
            "Security Assessment"
        )

        col1, col2, col3, col4 = st.columns(
            4
        )

        col1.metric(
            "Detected Threat",
            attack_type
        )

        col2.metric(
            "Risk Score",
            f"{highest_risk:.2f}/100"
        )

        col3.metric(
            "Severity",
            severity
        )

        if confidence is not None:

            col4.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

        else:

            col4.metric(
                "Detection",
                "Anomaly"
            )

        # ----------------------------------------------------
        # RISK STATUS
        # ----------------------------------------------------

        st.subheader(
            "Risk Assessment"
        )

        if severity == "CRITICAL":

            st.error(
                "CRITICAL RISK — Immediate investigation "
                "and mitigation are recommended."
            )

        elif severity == "HIGH":

            st.warning(
                "HIGH RISK — Prompt investigation and "
                "security response are recommended."
            )

        elif severity == "MEDIUM":

            st.warning(
                "MEDIUM RISK — Further investigation and "
                "continuous monitoring are recommended."
            )

        else:

            st.success(
                "LOW RISK — No immediate high-priority "
                "security response is indicated."
            )

        st.progress(
            min(
                int(highest_risk),
                100
            )
        )

        # ----------------------------------------------------
        # THREAT DESCRIPTION
        # ----------------------------------------------------

        st.subheader(
            "Threat Description"
        )

        with st.container(
            border=True
        ):

            st.write(
                threat_info["description"]
            )

        # ----------------------------------------------------
        # WHY IT MATTERS
        # ----------------------------------------------------

        st.subheader(
            "Why This Threat Matters"
        )

        with st.container(
            border=True
        ):

            st.write(
                threat_info["why_it_matters"]
            )

        # ----------------------------------------------------
        # POTENTIAL IMPACT
        # ----------------------------------------------------

        st.subheader(
            "Potential Impact"
        )

        with st.container(
            border=True
        ):

            for impact in threat_info[
                "potential_impact"
            ]:

                st.markdown(
                    f"- {impact}"
                )

        # ----------------------------------------------------
        # DETECTION DETAILS
        # ----------------------------------------------------

        st.subheader(
            "Detection Details"
        )

        col1, col2 = st.columns(
            2
        )

        with col1:

            with st.container(
                border=True
            ):

                st.markdown(
                    "**Model Used**"
                )

                st.write(
                    current_model
                )

                st.markdown(
                    "**Detection Method**"
                )

                if current_model == "Isolation Forest":

                    st.write(
                        "Anomaly Detection"
                    )

                else:

                    st.write(
                        "Multiclass Classification"
                    )

        with col2:

            with st.container(
                border=True
            ):

                st.markdown(
                    "**Detected Activity**"
                )

                st.write(
                    attack_type
                )

                st.markdown(
                    "**Severity**"
                )

                st.write(
                    severity
                )

        # ----------------------------------------------------
        # IMMEDIATE ACTIONS
        # ----------------------------------------------------

        st.subheader(
            "Recommended Immediate Actions"
        )

        for number, action in enumerate(
            threat_info[
                "immediate_actions"
            ],
            start=1
        ):

            with st.container(
                border=True
            ):

                st.markdown(
                    f"**{number}.** {action}"
                )

        # ----------------------------------------------------
        # PREVENTIVE MEASURES
        # ----------------------------------------------------

        st.subheader(
            "Preventive Security Measures"
        )

        for number, measure in enumerate(
            threat_info[
                "preventive_measures"
            ],
            start=1
        ):

            with st.container(
                border=True
            ):

                st.markdown(
                    f"**{number}.** {measure}"
                )

        # ----------------------------------------------------
        # MONITORING
        # ----------------------------------------------------

        st.subheader(
            "Recommended Monitoring"
        )

        for item in threat_info[
            "monitoring"
        ]:

            st.markdown(
                f"- {item}"
            )

        # ----------------------------------------------------
        # INCIDENT RESPONSE WORKFLOW
        # ----------------------------------------------------

        st.subheader(
            "Suggested Incident Response Workflow"
        )

        response_steps = [

            (
                "1. Identify",
                "Identify affected hosts, network flows "
                "and relevant traffic sources."
            ),

            (
                "2. Validate",
                "Review logs and traffic information "
                "to determine whether the activity "
                "is malicious or anomalous."
            ),

            (
                "3. Contain",
                "Apply appropriate network controls "
                "to limit further suspicious activity."
            ),

            (
                "4. Investigate",
                "Examine affected systems, accounts, "
                "services and network connections."
            ),

            (
                "5. Mitigate",
                "Remove or block the underlying threat "
                "and apply appropriate security controls."
            ),

            (
                "6. Monitor",
                "Continue monitoring to determine whether "
                "the suspicious activity returns."
            )
        ]

        for title, description in response_steps:

            with st.container(
                border=True
            ):

                st.markdown(
                    f"**{title}**"
                )

                st.write(
                    description
                )

        # ----------------------------------------------------
        # THREAT-SPECIFIC RECOMMENDATIONS
        # ----------------------------------------------------

        st.subheader(
            "Threat-Specific Recommendations"
        )

        if attack_type == "ANOMALY":

            recommendations = (
                get_anomaly_recommendations()
            )

        else:

            recommendations = (
                get_recommendations(
                    attack_type
                )
            )

        for number, recommendation in enumerate(
            recommendations,
            start=1
        ):

            st.markdown(
                f"**{number}.** {recommendation}"
            )

        # ----------------------------------------------------
        # ANALYSIS SUMMARY TABLE
        # ----------------------------------------------------

        st.subheader(
            "Analysis Summary"
        )

        if confidence is not None:

            confidence_value = (
                f"{confidence:.2f}%"
            )

        else:

            confidence_value = (
                "Not applicable"
            )

        summary_table = pd.DataFrame(
            {
                "Parameter": [
                    "AI Model",
                    "Detection Method",
                    "Detected Threat",
                    "Risk Score",
                    "Severity",
                    "Model Confidence"
                ],

                "Result": [
                    current_model,

                    (
                        "Anomaly Detection"
                        if current_model == "Isolation Forest"
                        else "Multiclass Classification"
                    ),

                    attack_type,

                    f"{highest_risk:.2f}/100",

                    severity,

                    confidence_value
                ]
            }
        )

        st.dataframe(
            summary_table,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # DETAILED TEXT REPORT
        # ----------------------------------------------------

        st.subheader(
            "Detailed Advisory Report"
        )

        report = f"""
AI-BASED CYBER THREAT ADVISOR
============================================================

THREAT ADVISORY REPORT
============================================================

1. DETECTION SUMMARY
------------------------------------------------------------

AI Model:
{current_model}

Detection Method:
{"Anomaly Detection" if current_model == "Isolation Forest" else "Multiclass Classification"}

Detected Threat:
{attack_type}

Risk Score:
{highest_risk:.2f}/100

Severity:
{severity}

Model Confidence:
{confidence_value}


2. EXECUTIVE SUMMARY
------------------------------------------------------------

{threat_info["description"]}

{get_risk_interpretation(highest_risk)}


3. WHY THIS THREAT MATTERS
------------------------------------------------------------

{threat_info["why_it_matters"]}


4. POTENTIAL IMPACT
------------------------------------------------------------

"""

        for impact in threat_info[
            "potential_impact"
        ]:

            report += (
                f"- {impact}\n"
            )

        report += """

5. RECOMMENDED IMMEDIATE ACTIONS
------------------------------------------------------------

"""

        for number, action in enumerate(
            threat_info[
                "immediate_actions"
            ],
            start=1
        ):

            report += (
                f"{number}. {action}\n"
            )

        report += """

6. PREVENTIVE SECURITY MEASURES
------------------------------------------------------------

"""

        for number, measure in enumerate(
            threat_info[
                "preventive_measures"
            ],
            start=1
        ):

            report += (
                f"{number}. {measure}\n"
            )

        report += """

7. RECOMMENDED MONITORING
------------------------------------------------------------

"""

        for item in threat_info[
            "monitoring"
        ]:

            report += (
                f"- {item}\n"
            )

        report += """

8. INCIDENT RESPONSE WORKFLOW
------------------------------------------------------------

"""

        for title, description in response_steps:

            report += (
                f"{title}: {description}\n"
            )

        report += """

9. THREAT-SPECIFIC RECOMMENDATIONS
------------------------------------------------------------

"""

        for number, recommendation in enumerate(
            recommendations,
            start=1
        ):

            report += (
                f"{number}. {recommendation}\n"
            )

        report += """

============================================================
END OF THREAT ADVISORY
============================================================
"""

        st.download_button(
            "Download Detailed Threat Advisory",
            data=report,
            file_name="detailed_threat_advisory.txt",
            mime="text/plain",
            use_container_width=True
        )