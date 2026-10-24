# Customer Payment Risk & Revenue Protection System

An end-to-end ML portfolio project that estimates whether an invoice may be paid late.

## Included
- Dataset copied to `data/raw/customer_payment_risk_dataset.csv`
- `train.py`: data preparation, three-model comparison, holdout metrics, saved model
- `app.py`: Streamlit executive overview, prediction page, model performance, data quality
- `models/`: generated model artifact after training
- `reports/`: generated metrics after training

## Setup (Windows PowerShell)
```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train.py
streamlit run app.py
```

If you already use Python 3.12, these commands create and use a project-local virtual environment. If `py -3.12` is unavailable, use a supported Python version installed on your computer.

## Target and leakage prevention
`late_payment` is the target (1 = late, 0 = not late). The training features are invoice amount, payment terms, previous late-payment count, invoice month, and invoice quarter.

`payment_date` and `outstanding_amount` are excluded from training features because they can reveal information that may not be available when a risk prediction is made. `invoice_id` and `customer_id` are identifiers and are also excluded.

## Important limitations
- The current train/test split is random and is suitable for a first portfolio baseline, not final production validation. For a real deployment, evaluate on a later time period (temporal split).
- Check how `late_payment` was defined and whether the data represents real historical records. The model is only as reliable as the labels and data.
- Do not use the model alone to make credit or collections decisions. Validate it with finance/domain experts, monitor performance and fairness, and test on real company data before production.
