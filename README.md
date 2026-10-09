# AI Cyber Threat Analyzer

An AI-powered cybersecurity application that uses machine learning to classify network traffic, assess threat severity, and provide actionable mitigation recommendations.

## Overview

Traditional Intrusion Detection Systems (IDS) focus primarily on identifying suspicious network activity. The AI Cyber Threat Analyzer aims to extend this process by combining attack classification with risk assessment and security advisories.

The application provides an interactive dashboard where users can select machine learning models, analyze predictions, review performance metrics, and understand recommended mitigation measures.

## Features

- **Attack Classification:** Classifies network traffic into benign and malicious categories.
- **Multiple ML Models:** Supports Random Forest, Decision Tree, XGBoost, and Isolation Forest.
- **Risk Assessment:** Assigns a risk score to help prioritize potential threats.
- **Threat Severity:** Categorizes detected threats according to the implemented risk assessment logic.
- **Threat Advisory:** Provides recommendations for mitigating identified attack types.
- **Interactive Dashboard:** Uses Streamlit to present predictions and analytical results.
- **Model Evaluation:** Compares models using accuracy, precision, recall, and F1-score.
- **Visual Analytics:** Includes confusion matrices, feature importance plots, and performance comparisons.

## Attack Categories

The project covers the following traffic categories:

| Category | Description |
|---|---|
| BENIGN | Normal network traffic |
| Bot | Traffic associated with bot activity |
| Brute Force | Repeated attempts to guess credentials |
| DDoS | Distributed Denial-of-Service |
| DoS | Denial-of-Service |
| PortScan | Network port scanning |
| Web Attack | Malicious activity targeting web applications |

## Technology Stack

- **Language:** Python
- **Machine Learning:** Scikit-learn, XGBoost
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib
- **Web Application:** Streamlit
- **Model Persistence:** Pickle

## Machine Learning Models

| Model | Purpose |
|---|---|
| Random Forest | Supervised traffic classification |
| Decision Tree | Supervised traffic classification |
| XGBoost | Gradient-boosted traffic classification |
| Isolation Forest | Anomaly detection |

The supervised models predict predefined traffic categories. Isolation Forest is used to identify observations that differ from learned patterns.

## Dataset

The project uses network traffic data derived from the CICIDS2017 dataset.

The preprocessing pipeline includes:

1. Loading and combining selected CSV files.
2. Handling missing and infinite values.
3. Preparing numerical input features.
4. Encoding attack labels.
5. Splitting the data into training and testing sets.
6. Training and evaluating machine learning models.

**Note:** The raw traffic CSV files are excluded from this repository because of their size. Obtain the dataset from its original source and place the files in the expected directory before running the preprocessing pipeline.

Dataset source: [Canadian Institute for Cybersecurity — Intrusion Detection Evaluation Dataset (CICIDS2017)](https://www.unb.ca/cic/datasets/ids-2017.html)

## Project Structure

```text
AI-Cyber-Threat-Analyzer/
├── advisor/
│   ├── predict.py
│   ├── recommendations.py
│   └── risk_assessment.py
├── evaluation/
│   ├── graphs/
│   ├── combined_result.py
│   ├── plot_model_details.py
│   └── plot_results.py
├── models/
│   ├── decision_tree.py
│   ├── isolation_forest.py
│   ├── random_forest.py
│   └── xgboost_model.py
├── preprocessing/
│   ├── class_mapping.csv
│   ├── clean_dataset.py
│   ├── create_ml_dataset.py
│   └── prepare_data.py
├── results/
├── saved_models/
├── app.py
├── main.py
├── .gitignore
└── README.md
```

## Installation

### Prerequisites

- Python 3.10 or a version compatible with the project dependencies
- pip
- Git

### 1. Clone the repository

```bash
git clone https://github.com/MISH09/AI-Cyber-Threat-Analyzer.git
cd AI-Cyber-Threat-Analyzer
```

### 2. Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

If `requirements.txt` is present:

```bash
pip install -r requirements.txt
```

Otherwise, install the dependencies used by the project, including Streamlit, Pandas, NumPy, Scikit-learn, Matplotlib, and XGBoost, with compatible versions.

### 4. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal.

**Note:** The application requires its trained model files and any other configured artifacts to be present. The raw datasets are not included in this repository.

## Evaluation

The project evaluates classification models using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrices
- Feature importance

Performance metrics should be interpreted in the context of the dataset and evaluation methodology. High accuracy on a prepared dataset does not necessarily indicate equivalent performance on unseen or real-world network traffic.

## Research Motivation

The project explores how machine learning can support cybersecurity decision-making by combining network attack classification with risk-based prioritization and actionable recommendations.

The broader research direction is to move beyond detection alone and help users interpret potential threats and determine appropriate mitigation actions.

## Future Enhancements

- Explainable AI for interpreting model predictions.
- Evaluation on unseen network traffic.
- Improved per-class performance analysis.
- More robust risk scoring and validation.
- Context-aware mitigation recommendations.
- Real-time network traffic monitoring.
- Integration with security monitoring tools.
- Testing against previously unseen attack patterns.

## Disclaimer

This project is intended for educational, academic, and research purposes. Its predictions and risk scores should not be treated as definitive security assessments. Validate recommendations before applying them to production systems.

## Author

**Manasi Joshi**

GitHub: [@MISH09](https://github.com/MISH09)
