# PAYGUARD AI — Customer Payment Risk Intelligence Platform

An end-to-end Data Science project that uses Machine Learning to predict customer late-payment risk and provide financial insights through an interactive dashboard.

## 📌 Project Overview

PAYGUARD AI is designed to help finance teams identify invoices at risk of late payment, analyze customer payment behavior, and support data-driven collection decisions.

The project combines data preprocessing, exploratory data analysis, machine learning, model evaluation, and interactive visualization in one platform.

## 🎯 Business Problem

Late customer payments can affect cash flow, working capital, and financial planning. Businesses need a way to identify potential late payments early and prioritize follow-up activities.

PAYGUARD AI explores how historical invoice and payment information can be used to estimate late-payment risk.

## 🚀 Key Features

- Customer late-payment risk prediction
- Invoice and payment behavior analysis
- Interactive executive dashboard
- Key financial and payment-risk indicators
- Machine learning model evaluation
- Data quality checks and missing-value analysis
- Interactive charts and data visualization

## 🛠️ Technology Stack

- **Programming:** Python
- **Data Analysis:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Dashboard:** Streamlit
- **Visualization:** Plotly
- **Model Storage:** Joblib
- **Version Control:** Git and GitHub

## 📂 Project Structure

```text
PayGuard-AI-Customer-Payment-Risk/
│
├── app.py
├── train.py
├── requirements.txt
├── README.md
│
├── data/
│   └── raw/
│       └── customer_payment_risk_dataset.csv
│
├── models/
│   └── payment_risk_model.joblib
│
└── reports/
    └── model_metrics.json
```

*Update this structure if your actual repository contains different files or folders.*

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/PayGuard-AI-Customer-Payment-Risk.git
```

### 2. Open the project folder

```bash
cd PayGuard-AI-Customer-Payment-Risk
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Train the model

```bash
python train.py
```

Run the training command if your `train.py` is configured to train the model and save the required artifacts.

### 6. Launch the dashboard

```bash
streamlit run app.py
```

The dashboard should open in your browser.

## 📊 Model Evaluation

The model can be evaluated using appropriate classification metrics, such as:

- Precision
- Recall
- F1-score
- ROC-AUC

Actual evaluation results should be reported only after running and validating the model.

## 💼 Business Value

PAYGUARD AI demonstrates how machine learning and data visualization can help businesses:

- Identify customers with a higher likelihood of late payment
- Prioritize collection follow-ups
- Understand historical payment patterns
- Support cash-flow and risk-management decisions

Predictions are estimates, not guarantees of future customer behavior.

## 🔮 Future Improvements

- Model explainability using SHAP
- Model monitoring and drift detection
- API deployment using FastAPI
- Cloud deployment
- Automated reporting and collection-priority recommendations

## 👨‍💻 Author

**Kishor Suvarnakar**



This is a portfolio project for educational and demonstration purposes. Predictions should be validated before being used in real financial decisions.
