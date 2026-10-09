# AI Cyber Threat Advisor

A machine learning-based cybersecurity application that classifies network traffic, assesses potential threats, assigns risk levels, and provides actionable mitigation recommendations.

## Overview

Traditional intrusion detection systems can identify suspicious network activity, but interpreting the severity of an attack and deciding what to do next can require additional analysis.

The **AI Cyber Threat Advisor** aims to bridge this gap by combining machine learning-based attack classification with risk assessment and security recommendations in one application.

## Key Features

- **Attack Classification:** Classifies network traffic into categories such as BENIGN, DoS, DDoS, PortScan, Bot, Brute Force, and Web Attack.
- **Risk Assessment:** Calculates a risk score and assigns a severity level to detected activity.
- **Threat Advisory:** Provides contextual information and recommended mitigation measures.
- **Model Comparison:** Supports comparison of supervised machine learning models.
- **Interactive Dashboard:** Presents detection results, model metrics, and visualizations.
- **Anomaly Detection:** Includes an Isolation Forest model for exploring anomaly-based detection.

## Technology Stack

- **Language:** Python
- **Machine Learning:** Scikit-learn, XGBoost
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib
- **Interface:** Streamlit

## Machine Learning Models

The project explores the following models:

| Model | Approach |
|---|---|
| Random Forest | Supervised classification |
| Decision Tree | Supervised classification |
| XGBoost | Gradient-boosted classification |
| Isolation Forest | Unsupervised anomaly detection |

The supervised models predict attack categories, while Isolation Forest explores the detection of anomalous network activity.

## Dataset and Preprocessing

The project uses network traffic data derived from CICIDS2017 traffic captures and CSV datasets.

Preprocessing includes:

- Combining the relevant traffic datasets
- Handling missing and non-finite values
- Preparing features for model training
- Encoding attack labels
- Splitting data into training and testing sets

The processed dataset and model artifacts are generated or stored locally as required by the implementation.

## Model Evaluation

The supervised models achieved high accuracy in the current experimental evaluation. The project also evaluates precision, recall, and F1-score.

Evaluation metrics should be interpreted alongside class-wise performance, confusion matrices, class imbalance, and checks for data leakage. Results on a random test split may not represent performance on unseen network environments.

## How It Works

1. Network traffic features are supplied to the application.
2. The selected model predicts an attack category or identifies anomalous activity.
3. The risk assessment component evaluates the detected activity.
4. The advisor generates relevant mitigation recommendations.
5. Results are displayed through the application interface.

## Getting Started

### Prerequisites

- Python 3.10 or another Python version supported by the project's dependencies
- pip

### Installation

Clone the repository:

```bash
git clone https://github.com/MISH09/REPOSITORY-NAME.git
cd REPOSITORY-NAME
```

Replace `REPOSITORY-NAME` with the actual GitHub repository name.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies using the repository's dependency file:

```bash
pip install -r requirements.txt
```

### Run the Application

If the project uses Streamlit and the main entry point is `app.py`, run:

```bash
streamlit run app.py
```

Follow the instructions in the repository if model files or datasets need to be downloaded or generated separately.

## Future Improvements

- Evaluate generalization on separate traffic captures and unseen network environments.
- Investigate data leakage and class-wise performance.
- Improve risk scoring through systematic validation.
- Expand threat advisories with attack-specific mitigation guidance.
- Add model explainability to help users understand predictions.
- Explore integration with live or near-real-time traffic monitoring.

## Disclaimer

This project is intended for educational and research purposes. Its predictions and mitigation recommendations should be validated before being used in production security environments.

## Author

**Manasi Joshi**

GitHub: [MISH09](https://github.com/MISH09)

