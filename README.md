
# Customer Churn Prediction using Classification Algorithms

## Overview

This project predicts which customers are likely to churn (leave a service) using supervised classification algorithms. It provides an end‑to‑end pipeline in one Python file (`customer_churn.py`) that covers:

1. Data loading from a CSV
2. Pre‑processing (imputation, scaling, one‑hot encoding)
3. Model selection (Logistic Regression, Random Forest, Gradient Boosting, or XGBoost)
4. Training/validation with a configurable test split
5. Evaluation with common metrics (Accuracy, Precision, Recall, F1, ROC‑AUC, Confusion Matrix)
6. Saving the trained model (`best_model.pkl`)
7. Single‑record inference from a JSON file

## Quick Start

### 1. Install Requirements

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install pandas numpy scikit-learn imbalanced-learn joblib xgboost
```

*(If you don’t need XGBoost, omit the last package and choose another model.)*

### 2. Train a Model

```bash
python customer_churn.py \
    --data data/telco.csv \
    --target Churn \
    --model xgboost \
    --test_size 0.2
```

The script prints metrics and writes `models/best_model.pkl`.

### 3. Predict for One Customer

```bash
python customer_churn.py \
    --predict sample_customer.json \
    --model_path models/best_model.pkl \
    --target Churn
```

Outputs the churn probability (0‑1) and the binary label (0=retain, 1=churn).

## Directory Layout (suggested)

```
project_root/
├── customer_churn.py        # <— single‑file pipeline
├── data/
│   └── telco.csv            # raw dataset
├── models/
│   └── best_model.pkl       # saved after training
├── sample_customer.json     # one record for inference
└── README.txt               # you are here
```

## Command‑Line Arguments

| Flag           | Default | Description                                 |
| -------------- | ------- | ------------------------------------------- |
| `--data`       | —       | Path to CSV file with data                  |
| `--target`     | Churn   | Name of the target column                   |
| `--model`      | xgboost | Choice: `logreg`, `rf`, `gb`, `xgboost`     |
| `--test_size`  | 0.2     | Fraction of data for test split             |
| `--save_dir`   | models  | Folder to save the trained model            |
| `--predict`    | —       | Path to JSON for single‑row prediction      |
| `--model_path` | —       | Path to a saved `.pkl` model for prediction |

## Notes & Tips

* **Class imbalance:** If churn cases are rare, edit the code to enable class weights or SMOTE.
* **Feature engineering:** Add derived features (e.g., tenure buckets) before training for better performance.
* **Hyper‑parameter tuning:** For simple tuning, wrap the `get_model` call in `GridSearchCV` or use Optuna.
* **Deployment:** The saved pickle can be served via FastAPI or loaded into a Streamlit dashboard.

