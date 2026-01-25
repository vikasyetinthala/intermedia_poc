import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
import pickle
import os

import mlflow
import mlflow.sklearn
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import matplotlib.pyplot as plt
import shap
from mlflow.models import infer_signature

# Ensure models dir exists
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'dataset')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

print("Loading Data...")
suppliers = pd.read_csv(os.path.join(DATA_DIR, "suppliers.csv"))
pos = pd.read_csv(os.path.join(DATA_DIR, "purchase_orders.csv"))
history = pd.read_csv(os.path.join(DATA_DIR, "invoices_history.csv"))

# 1. Pre-calculate Supplier Stats
def get_supplier_stats(history_df):
    stats = history_df.groupby('Supplier_ID').agg({
        'Status': lambda x: (x == 'Approve').mean(),
        'Invoice_Amount': ['mean', 'count']
    }).reset_index()
    stats.columns = ['Supplier_ID', 'Sup_Approval_Rate', 'Sup_Avg_Inv_Amount', 'Sup_Inv_Count']
    return stats

sup_stats = get_supplier_stats(history)
suppliers = suppliers.merge(sup_stats, on='Supplier_ID', how='left').fillna({
    'Sup_Approval_Rate': 0.5,
    'Sup_Avg_Inv_Amount': history['Invoice_Amount'].mean(),
    'Sup_Inv_Count': 0
})

# --- Shared Logic (Must match App) ---
def match_po(inv_row, pos_df):
    def normalize_po(po):
        if pd.isna(po) or po == "": return ""
        import re
        return re.sub(r'[^A-Z0-9]', '', str(po).upper())

    # --- Stage 1: Strict Matching ---
    inv_po = normalize_po(inv_row.get('PO_Number', ''))
    if inv_po:
        # We need to normalize PO_Number in the table too for matching
        pos_df_temp = pos_df.copy()
        pos_df_temp['Norm_PO'] = pos_df_temp['PO_Number'].apply(normalize_po)
        match = pos_df_temp[pos_df_temp['Norm_PO'] == inv_po]
        if not match.empty: 
            return match.iloc[0], "Strict"

    # --- Stage 2: Conservative Smart Matching ---
    # Must satisfy: Supplier ID match AND date ±14 days AND amount ±2%
    candidates = pos_df[pos_df['Supplier_ID'] == inv_row['Supplier_ID']].copy()
    if not candidates.empty:
        inv_date = pd.to_datetime(inv_row['Invoice_Date'])
        candidates['PO_Date'] = pd.to_datetime(candidates['PO_Date'])
        
        # Absolute differences
        candidates['Date_Diff'] = (candidates['PO_Date'] - inv_date).dt.days.abs()
        candidates['Amt_Diff_Pct'] = abs(candidates['PO_Amount'] - inv_row['Invoice_Amount']) / candidates['PO_Amount']
        
        # Filter by criteria
        potential = candidates[
            (candidates['Date_Diff'] <= 14) & 
            (candidates['Amt_Diff_Pct'] <= 0.02)
        ].copy()
        
        if not potential.empty:
            # Tie-breaking: 1. Closest date, 2. Lowest amount diff
            best = potential.sort_values(['Date_Diff', 'Amt_Diff_Pct']).head(1)
            return best.iloc[0], "Smart"

    # --- Stage 3: Non-PO Assignment ---
    return None, "None"

def enrich_data(invoices, pos_df, suppliers_df):
    enriched = []
    print(f"Enriching {len(invoices)} invoices...")
    for i, row in invoices.iterrows():
        po, match_type = match_po(row, pos_df)
        d = row.to_dict()
        d['Match_Type'] = match_type
        if po is not None:
            d['PO_Amount'] = po['PO_Amount']
            d['PO_Currency'] = po['Currency']
            d['PO_Date'] = po['PO_Date']
        else:
            d['PO_Amount'] = 0; d['PO_Currency'] = "UNK"; d['PO_Date'] = None
            
        sup = suppliers_df[suppliers_df['Supplier_ID'] == row['Supplier_ID']]
        if not sup.empty:
            d['Sup_Risk'] = sup.iloc[0]['Risk_Score']
            d['Sup_Tenure'] = sup.iloc[0]['Tenure_Days']
            d['Sup_Approval_Rate'] = sup.iloc[0]['Sup_Approval_Rate']
            d['Sup_Avg_Inv_Amount'] = sup.iloc[0]['Sup_Avg_Inv_Amount']
            d['Sup_Inv_Count'] = sup.iloc[0]['Sup_Inv_Count']
            d['Sup_Is_New'] = 1 if sup.iloc[0]['Tenure_Days'] < 90 else 0
        else:
            d['Sup_Risk'] = 50; d['Sup_Tenure'] = 0; d['Sup_Approval_Rate'] = 0.5
            d['Sup_Avg_Inv_Amount'] = 0; d['Sup_Inv_Count'] = 0; d['Sup_Is_New'] = 1
        enriched.append(d)
    return pd.DataFrame(enriched)

def engineered_features(df, dept_map=None, req_map=None):
    df['Amt_Variance'] = np.where(df['PO_Amount'] > 0, (df['Invoice_Amount'] - df['PO_Amount'])/df['PO_Amount'], 0)
    df['Amt_Diff_Abs'] = np.where(df['PO_Amount'] > 0, abs(df['Invoice_Amount'] - df['PO_Amount']), 0)
    df['Currency_Mismatch'] = (df['Currency'] != df['PO_Currency']).astype(int)
    df['Is_Non_PO'] = (df['Match_Type'] == "None").astype(int)
    
    df['Invoice_Date'] = pd.to_datetime(df['Invoice_Date'])
    df['PO_Date'] = pd.to_datetime(df['PO_Date'])
    df['Days_Since_PO'] = (df['Invoice_Date'] - df['PO_Date']).dt.days.fillna(-1)
    
    # Metadata Features
    df['Inv_DayOfWeek'] = df['Invoice_Date'].dt.dayofweek
    df['Inv_Month'] = df['Invoice_Date'].dt.month
    
    # Text Pattern Features
    df['Desc_Word_Count'] = df['Description'].fillna("").apply(lambda x: len(str(x).split()))
    urgent_keywords = ['urgent', 'wire', 'transfer', 'immediate', 'asap']
    df['Has_Urgent_Keyword'] = df['Description'].fillna("").apply(
        lambda x: 1 if any(k in str(x).lower() for k in urgent_keywords) else 0
    )
    
    # Mappings
    if dept_map is None:
        dept_map = {k: v for v, k in enumerate(sorted(df['Department'].unique()))}
    df['Dept_Code'] = df['Department'].map(dept_map).fillna(0)
    
    if req_map is None:
        req_map = {k: v for v, k in enumerate(sorted(df['Requestor'].unique()))}
    df['Requestor_Code'] = df['Requestor'].map(req_map).fillna(0)
    
    match_map = {'None':0, 'Smart':1, 'Strict':2}
    df['Match_Type_Code'] = df['Match_Type'].map(match_map).fillna(0)
    
    return df, dept_map, req_map

# 2. Pipeline Execution
print("Enriching History...")
df_raw = enrich_data(history, pos, suppliers)

print("Fitting Text Pipeline...")
tfidf = TfidfVectorizer(max_features=500, stop_words='english')
svd = TruncatedSVD(n_components=5, random_state=42)
txt_matrix = tfidf.fit_transform(df_raw['Description'].fillna(""))
svd_feats = svd.fit_transform(txt_matrix)
txt_df = pd.DataFrame(svd_feats, columns=[f'Text_SVD_{i}' for i in range(5)])
df_raw = pd.concat([df_raw, txt_df], axis=1)

print("Engineering Features...")
df_final, dept_map, req_map = engineered_features(df_raw) # dept_map created here

# 3. Model Training with MLflow
print("Starting MLflow Run...")
mlflow.set_experiment("Invoice_Classifier_Training")

with mlflow.start_run(run_name=f"Training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
    # --- Rolling Window Logic (90-day window) ---
    history['Invoice_Date'] = pd.to_datetime(history['Invoice_Date'])
    max_date = history['Invoice_Date'].max()
    window_start = max_date - timedelta(days=90)
    
    # Filter for rolling window
    df_window = df_final[df_final['Invoice_Date'] >= window_start].copy()
    print(f"Training on rolling window: {window_start.date()} to {max_date.date()} ({len(df_window)} records)")
    
    target = 'Status'
    le_target = LabelEncoder()
    y = le_target.fit_transform(df_window[target])
    labels_map = dict(zip(le_target.transform(le_target.classes_), le_target.classes_))
    
    features = ['Invoice_Amount', 'Sup_Risk', 'Sup_Tenure', 'Sup_Approval_Rate',
            'Sup_Avg_Inv_Amount', 'Sup_Inv_Count', 'Sup_Is_New',
            'Amt_Variance', 'Amt_Diff_Abs', 'Currency_Mismatch', 'Is_Non_PO',
            'Days_Since_PO', 'Inv_DayOfWeek', 'Inv_Month',
            'Desc_Word_Count', 'Has_Urgent_Keyword',
            'Dept_Code', 'Requestor_Code', 'Match_Type_Code'] + [f'Text_SVD_{i}' for i in range(5)]
            
    X = df_window[features]
    
    # --- Hyperparameter Tuning ---
    print("Performing Hyperparameter Tuning...")
    # Simplified Grid for demo
    param_grid = [
        {'num_leaves': 31, 'learning_rate': 0.1, 'n_estimators': 50},
        {'num_leaves': 20, 'learning_rate': 0.05, 'n_estimators': 100},
        {'num_leaves': 40, 'learning_rate': 0.15, 'n_estimators': 30}
    ]
    
    best_model = None
    best_f1 = -1
    best_metrics = {}
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    for i, params in enumerate(param_grid):
        with mlflow.start_run(run_name=f"Trial_{i}", nested=True):
            mlflow.log_params(params)
            m = lgb.LGBMClassifier(**params, random_state=42)
            m.fit(X_train, y_train)
            
            y_pred_trial = m.predict(X_val)
            f1_trial = f1_score(y_val, y_pred_trial, average='weighted')
            
            if f1_trial > best_f1:
                best_f1 = f1_trial
                best_model = m # This is the best uncalibrated model
                
            mlflow.log_metrics({
                "f1_score": f1_trial
            })

    print(f"Best Trial F1 (uncalibrated): {best_f1:.4f}")
    
    # 4. Probability Calibration (Isotonic Regression)
    print("Calibrating Model...")
    calibrated_model = CalibratedClassifierCV(best_model, method='isotonic', cv='prefit')
    calibrated_model.fit(X_train, y_train)
    
    y_pred = calibrated_model.predict(X_val)
    y_prob = calibrated_model.predict_proba(X_val)
    
    # Evaluate Calibrated Model
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    prec = precision_score(y_val, y_pred, average='weighted')
    rec = recall_score(y_val, y_pred, average='weighted')
    
    mlflow.log_metrics({
        "accuracy": acc,
        "f1_score": f1,
        "precision": prec,
        "recall": rec
    })
    
    # --- Per-Class Metrics ---
    report = classification_report(y_val, y_pred, target_names=le_target.classes_, output_dict=True)
    for class_name, metrics in report.items():
        if class_name in le_target.classes_:
            mlflow.log_metrics({
                f"{class_name}_f1": metrics['f1-score'],
                f"{class_name}_precision": metrics['precision'],
                f"{class_name}_recall": metrics['recall']
            })
            
    best_metrics = {"accuracy": acc, "f1_score": f1, "precision": prec, "recall": rec}
    best_y_pred = y_pred # Store for confusion matrix
    
    # Track Best Metrics on main run
    mlflow.log_metrics(best_metrics)
    
    # --- Confusion Matrix ---
    print("Generating Confusion Matrix...")
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(y_val, best_y_pred, display_labels=le_target.classes_, ax=ax, cmap='Blues')
    plt.title("Confusion Matrix")
    conf_matrix_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
    plt.savefig(conf_matrix_path)
    mlflow.log_artifact(conf_matrix_path)
    plt.close()

    # Calibration Curve Validation
    print("Generating Calibration Curve...")
    plt.figure(figsize=(10, 6))
    for i, class_label in enumerate(le_target.classes_):
        # We check calibration for each class vs rest
        y_val_bin = (y_val == le_target.transform([class_label])[0]).astype(int)
        prob_pos = y_prob[:, i]
        fraction_of_positives, mean_predicted_value = calibration_curve(y_val_bin, prob_pos, n_bins=10)
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=f"{class_label}")
    
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.ylabel("Fraction of positives")
    plt.xlabel("Mean predicted value")
    plt.legend(loc="lower right")
    plt.title("Calibration Curves")
    cal_path = os.path.join(MODEL_DIR, "calibration_curve.png")
    plt.savefig(cal_path)
    mlflow.log_artifact(cal_path)
    plt.close()

    # --- SHAP Analysis ---
    print("Generating SHAP Plots...")
    # SHAP explainer should use the base model, not the calibrated one, for feature importance
    explainer = shap.TreeExplainer(best_model) 
    shap_values = explainer.shap_values(X_val)
    
    # SHAP Summary Plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_val, show=False)
    shap_summary_path = os.path.join(MODEL_DIR, "shap_summary.png")
    plt.savefig(shap_summary_path, bbox_inches='tight')
    mlflow.log_artifact(shap_summary_path)
    plt.close()

    # --- Model Signature ---
    signature = infer_signature(X_train, calibrated_model.predict(X_train))

    # Log Model to MLflow
    mlflow.sklearn.log_model(
        sk_model=calibrated_model,
        artifact_path="model",
        registered_model_name="InvoiceClassifier",
        signature=signature
    )
    
    # 4. Save Artifacts (For local Streamlit app use)
    artifacts = {
        'model': calibrated_model,
        'tfidf': tfidf,
        'svd': svd,
        'dept_map': dept_map,
        'req_map': req_map,
        'labels_map': labels_map,
        'features': features,
        'metrics': best_metrics,
        'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'run_id': mlflow.active_run().info.run_id
    }

with open(os.path.join(MODEL_DIR, 'invoice_classifier.pkl'), 'wb') as f:
    pickle.dump(artifacts, f)

print(f"Success! Model saved to {os.path.join(MODEL_DIR, 'invoice_classifier.pkl')}")
