# Technical Architecture Document: ML Invoice Classification System

## 1. Executive Summary

The **ML Invoice Classification System (Intermedia V3)** is an intelligent automation solution designed to streamline the accounts payable process. It uses machine learning to automatically categorize invoices into **Approve**, **Review Further**, or **Reject** buckets, significantly reducing manual effort and improving compliance. The system integrates a **LightGBM** classification model with **Google Gemini (GenAI)** to provide explainable AI decision cards.

## 2. Solution Overview

The solution consists of three main stages:

1.  **Synthetic Data Generation**: Creates realistic enterprise data (Suppliers, POs, Invoices) with specific risk scenarios.
2.  **Model Training & Tracking**: Trains a gradient boosting model (LightGBM) using MLOps best practices (MLflow) with isotonic probability calibration.
3.  **Interactive AI Dashboard**: A Streamlit-based web application that serves the model, visualizes risk metrics, and uses GenAI to explain _why_ a decision was made.

## 3. Component Details

### 3.1. Data Generation (`dataset/generate_data.py`)

- **Purpose**: Simulates a real-world ERP environment with balanced risk scenarios.
- **Key Logic**:
  - **Supplier Profiles**: Generates 50 suppliers with random risk scores (1-100) and tenure (30-3650 days).
  - **Purchase Orders (POs)**: Creates 300 POs across IT, Marketing, Finance, and Operations.
  - **Invoices**: Simulates 1500 historical invoices for training and 60 "New" invoices for demoing specific edge cases (Clean, Variance, Currency Flip, Smart Match, Non-PO Risk).

### 3.2. Training Pipeline (`train_model.py`)

- **Rolling Window**: Trains on a 90-day rolling window of historical data.
- **Model**: `LightGBM Classifier` with hyperparameters optimized via trial-based tracking.
- **Calibration**: Uses `Isotonic Regression` (via `CalibratedClassifierCV`) to ensure the predicted probabilities (Confidence) are well-aligned with true risk.
- **Tracking**: MLflow tracks accuracy, F1-score, precision, recall, confusion matrices, and SHAP feature importance plots.

### 3.3. User Interface (`app.py`)

- **Framework**: Streamlit.
- **Key Features**:
  - **Batch Processing**: Predictions for all new invoices are generated on page load.
  - **Risk Legend**: Visual guide for Approve (Green), Review (Orange), and Reject (Red).
  - **Explainable AI**: Integration with **Gemini 2.5 Flash** to generate "Decision Cards" based on the model's features and output.
  - **Inference Monitoring**: Logs every audit action to MLflow for production monitoring.

## 4. Feature Engineering & Calculation Logic

This section details the specific formulas used to process data before it enters the Machine Learning model.

### 4.1. Basic Risk Indicators

| Metric                  | Formula                                                                               | Description                                 |
| :---------------------- | :------------------------------------------------------------------------------------ | :------------------------------------------ | --- | -------------------- |
| **Amount Variance**     | $`\text{Var} = \frac{\text{Invoice\_Amount} - \text{PO\_Amount}}{\text{PO\_Amount}}`$ | Relative difference between Invoice and PO. |
| **Absolute Difference** | $`\text{Diff}\_{abs} =                                                                | \text{Invoice_Amount} - \text{PO_Amount}    | `$  | Raw dollar variance. |
| **Days Since PO**       | $`\Delta\text{Days} = \text{Invoice\_Date} - \text{PO\_Date}`$                        | Time elapsed since authorization.           |
| **Currency Mismatch**   | $`\mathbb{1}(\text{Inv\_Curr} \neq \text{PO\_Curr})`$                                 | Binary flag for currency discrepancies.     |

### 4.2. Supplier Behavior Metrics

These are calculated from the `invoices_history.csv` dataset.

- **Supplier Approval Rate**:
  $`\text{Rate}_{app} = \frac{\sum \mathbb{1}(\text{Status} = \text{'Approve'})}{\text{Count}(\text{Total Invoices by Supplier})}`$
- **Supplier Is New**:
  $`\text{Is\_New} = \mathbb{1}(\text{Tenure\_Days} < 90)`$
- **Average Invoice Amount**:
  $`\text{Avg}_{amt} = \frac{\sum \text{Invoice\_Amount}}{\text{Count}(\text{Total Invoices by Supplier})}`$

### 4.3. Structural & Text Features

The system extracts structure from both the metadata and the `Description` field:

- **Department**: Categorical feature (IT, Marketing, Finance, Operations) encoded as `Dept_Code`.
- **Requestor**: Identity of the requestor encoded as `Requestor_Code`.
- **Word Count**: Length of the description string.
- **Urgent Keywords**: Binary flag if "urgent", "wire", "transfer", "immediate", or "asap" is present.
- **LSA Features**: Top 5 semantic components extracted via TF-IDF + Truncated SVD.

## 5. Decision & Matching Logic

### 5.1. Matching Engine Stages

The system attempts to link invoices to POs using a cascading strategy:

1.  **Strict Match**: Matches normalized `PO_Number` strings exactly.
2.  **Smart Match**: If `PO_Number` is missing, finds the best PO by matching `Supplier_ID` and meeting these thresholds:
    - **Date Variance**: $\leq 14$ days.
    - **Amount Variance**: $\leq 2\%$.
3.  **Non-PO**: If no match is found, treats as a Non-PO invoice (Higher inherent risk).

### 5.2. Confidence Score

The confidence score displayed in the UI is derived from the calibrated prediction probabilities:
$`\text{Confidence} = \max(P(\text{Approve}), P(\text{Review}), P(\text{Reject}))`$

## 6. Technology Stack

- **Language**: Python 3.10+
- **Machine Learning**: LightGBM, Scikit-Learn
- **MLOps**: MLflow
- **GenAI**: Google Gemini 2.5 Flash
- **Web Framework**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Explanation Layer**: SHAP (Static), Gemini (Dynamic/Natural Language)
