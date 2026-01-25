import streamlit as st
import pandas as pd
import numpy as np
import pickle
import google.generativeai as genai
import os
import time
import mlflow
from dotenv import load_dotenv

# Page Config
st.set_page_config(page_title="Invoice Classifier", layout="wide")

# Load Env
load_dotenv('.env')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# --- Path Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'dataset')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'invoice_classifier.pkl')
ENV_PATH = os.path.join(BASE_DIR, '.env')

# Load Env
load_dotenv(ENV_PATH)
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    st.warning(f"Gemini API Key not found in {ENV_PATH}")

# ... (rest of imports)

@st.cache_data
def load_data():
    suppliers = pd.read_csv(os.path.join(DATA_DIR, "suppliers.csv"))
    pos = pd.read_csv(os.path.join(DATA_DIR, "purchase_orders.csv"))
    new_inv = pd.read_csv(os.path.join(DATA_DIR, "invoices_new.csv"))
    history = pd.read_csv(os.path.join(DATA_DIR, "invoices_history.csv"))
    
    # Pre-calc supplier stats
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
    
    return suppliers, pos, new_inv

@st.cache_resource
def load_model():
    """Load Pre-trained Model Artifacts"""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at {MODEL_PATH}! Run train_model.py first.")
        return None
        
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

# --- Helper Logic ---
def match_po(inv_row, pos_df):
    def normalize_po(po):
        if pd.isna(po) or po == "": return ""
        import re
        return re.sub(r'[^A-Z0-9]', '', str(po).upper())

    # --- Stage 1: Strict Matching ---
    inv_po = normalize_po(inv_row.get('PO_Number', ''))
    if inv_po:
        pos_df_temp = pos_df.copy()
        pos_df_temp['Norm_PO'] = pos_df_temp['PO_Number'].apply(normalize_po)
        match = pos_df_temp[pos_df_temp['Norm_PO'] == inv_po]
        if not match.empty: 
            return match.iloc[0], "Strict"

    # --- Stage 2: Conservative Smart Matching ---
    candidates = pos_df[pos_df['Supplier_ID'] == inv_row['Supplier_ID']].copy()
    if not candidates.empty:
        inv_date = pd.to_datetime(inv_row['Invoice_Date'])
        candidates['PO_Date'] = pd.to_datetime(candidates['PO_Date'])
        candidates['Date_Diff'] = (candidates['PO_Date'] - inv_date).dt.days.abs()
        candidates['Amt_Diff_Pct'] = abs(candidates['PO_Amount'] - inv_row['Invoice_Amount']) / candidates['PO_Amount']
        
        potential = candidates[
            (candidates['Date_Diff'] <= 14) & 
            (candidates['Amt_Diff_Pct'] <= 0.02)
        ].copy()
        
        if not potential.empty:
            best = potential.sort_values(['Date_Diff', 'Amt_Diff_Pct']).head(1)
            return best.iloc[0], "Smart"

    # --- Stage 3: Non-PO Assignment ---
    return None, "None"

def enrich_data(df, pos_df, suppliers_df):
    # Handle both single row (Series) and DataFrame
    if isinstance(df, pd.Series):
        df = pd.DataFrame([df])
        
    enriched = []
    for _, row in df.iterrows():
        po, match_type = match_po(row, pos_df)
        d = row.to_dict()
        d['Match_Type'] = match_type
        
        if po is not None:
            d['PO_Number_Matched'] = po['PO_Number']
            d['PO_Amount'] = po['PO_Amount']
            d['PO_Currency'] = po['Currency']
            d['PO_Date'] = po['PO_Date']
        else:
            d['PO_Number_Matched'] = None
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

def engineered_features(df, dept_map, req_map):
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
    
    df['Dept_Code'] = df['Department'].map(dept_map).fillna(0)
    df['Requestor_Code'] = df['Requestor'].map(req_map).fillna(0)
    
    match_map = {'None':0, 'Smart':1, 'Strict':2}
    df['Match_Type_Code'] = df['Match_Type'].map(match_map).fillna(0)
    return df

@st.cache_data
def batch_predict(df, pos_df, suppliers_df, _model, _tfidf, _svd, _dept_map, _req_map, _labels_map, _features):
    """Run pipeline on full dataframe to show predictions upfront"""
    if df.empty: return df
    
    # 1. Enrich
    df_enr = enrich_data(df, pos_df, suppliers_df)
    
    # 2. Text
    txt_matrix = _tfidf.transform(df_enr['Description'].fillna(""))
    svd_feats = _svd.transform(txt_matrix)
    txt_df = pd.DataFrame(svd_feats, columns=[f'Text_SVD_{i}' for i in range(5)])
    df_enr = pd.concat([df_enr.reset_index(drop=True), txt_df], axis=1)
    
    # 3. Features
    df_final = engineered_features(df_enr, _dept_map, _req_map)
    
    # 4. Predict
    X = df_final[_features]
    preds = _model.predict(X)
    probs = _model.predict_proba(X)
    
    df_final['Predicted_Status'] = [_labels_map[p] for p in preds]
    df_final['Confidence'] = [max(prob) for prob in probs]
    
    return df_final

# --- UI ---
st.title("🤖 ML Invoice Classification")
st.caption("v2.1 - Batch Predictions")

suppliers, pos, new_invoices = load_data()
artifacts = load_model()

if artifacts:
    model = artifacts['model']
    tfidf = artifacts['tfidf']
    svd = artifacts['svd']
    dept_map = artifacts['dept_map']
    req_map = artifacts['req_map']
    labels_map = artifacts['labels_map']
    features = artifacts['features']

    # RUN BATCH PREDICTION
    with st.spinner("Classifying all invoices..."):
        full_data = batch_predict(new_invoices, pos, suppliers, model, tfidf, svd, dept_map, req_map, labels_map, features)

    # --- DASHBOARD METRICS ---
    st.divider()
    

    tot = len(full_data)
    n_rej = len(full_data[full_data['Predicted_Status'] == 'Reject'])
    n_rev = len(full_data[full_data['Predicted_Status'] == 'Review Further'])
    n_app = len(full_data[full_data['Predicted_Status'] == 'Approve'])
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Invoices", tot)
    m2.metric("❌ To Reject", n_rej, delta=f"{n_rej/tot:.0%} Risk")
    m3.metric("⚠️ To Review", n_rev, delta="Manual Check")
    m4.metric("✅ To Approve", n_app, delta="Auto-Process")
    
    st.divider()

    # --- FILTER & SORT ---
    # Sort by Risk Priority: Reject (0) -> Review (1) -> Approve (2)
    priority_map = {'Reject': 0, 'Review Further': 1, 'Approve': 2}
    full_data['Priority'] = full_data['Predicted_Status'].map(priority_map)
    full_data_sorted = full_data.sort_values(['Priority', 'Invoice_ID'])
    
    st.markdown("### 1. Select Invoice to Audit")
    
    # Filter Toggles
    filter_status = st.multiselect("Filter by Status:", ['Reject', 'Review Further', 'Approve'], default=['Reject', 'Review Further', 'Approve'])
    
    filtered_df = full_data_sorted[full_data_sorted['Predicted_Status'].isin(filter_status)]
    
    if filtered_df.empty:
        st.warning("No invoices match the filter.")
        selected_id = None
    else:
        # Create label for dropdown
        def format_match_label(m_type):
            if m_type == "Strict": return "Strict Match"
            if m_type == "Smart": return "Smart Match"
            return "Non-PO"
            
        filtered_df['UI_Label'] = filtered_df['Invoice_ID'] + " (" + filtered_df['Predicted_Status'] + ") - " + filtered_df['Match_Type'].apply(format_match_label)
        
        selected_label = st.selectbox("Choose Invoice (Sorted by Risk)", filtered_df['UI_Label'])
        selected_id = selected_label.split(" ")[0]

    if selected_id:
        # --- PREVIEW SECTION ---
        # Get the row immediately
        row = full_data[full_data['Invoice_ID'] == selected_id]
        
        if not row.empty:
            # Coupa-style Invoice Header
            with st.container():
                st.markdown(f"### 📄 Invoice: {selected_id}")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Amount", f"${row.iloc[0]['Invoice_Amount']}", row.iloc[0]['Currency'])
                c2.write(f"**Supplier:**\n{row.iloc[0]['Supplier_ID']}")
                c3.write(f"**Department:**\n{row.iloc[0]['Department']}")
                c4.write(f"**Requestor:**\n{row.iloc[0]['Requestor']}")
                st.info(f"**Description:** {row.iloc[0]['Description']}")

            st.divider()

            # ACTION BUTTON
            if st.button("🚀 Analyze Selected Invoice", type="primary"):
                df_final = row 
                status = df_final.iloc[0]['Predicted_Status']
                confidence = df_final.iloc[0]['Confidence']

                # Display Analysis
                st.subheader("🛡️ AI Audit Results")
                c1, c2 = st.columns([1, 2])
                with c1:
                    color = "green" if status == "Approve" else "orange" if status == "Review Further" else "red"
                    st.markdown(f"## :{color}[{status}]")
                    st.metric("Confidence", f"{confidence:.1%}")
                    
                st.markdown("### 📊 Risk Indicators")
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Variance", f"{df_final.iloc[0]['Amt_Variance']:.1%}", delta_color="inverse")
                k2.metric("Supplier Risk", df_final.iloc[0]['Sup_Risk'], help="0-100 Scale (Lower is better)")
                
                m_type = df_final.iloc[0]['Match_Type']
                m_label = "Strict Match" if m_type == "Strict" else "Smart Match" if m_type == "Smart" else "Non-PO"
                branch = "PO-Linked" if m_type in ["Strict", "Smart"] else "Non-PO Branch"
                
                k3.metric("Match Type", m_label)
                k4.metric("Branch", branch)
                    
                with c2:
                    st.subheader("💡 AI Reasoning")
                    
                    # Context for LLM
                    variance_val = df_final.iloc[0]['Amt_Variance']
                    sup_risk = df_final.iloc[0]['Sup_Risk']
                    match_type = df_final.iloc[0]['Match_Type']
                    
                    prompt = f"""
                    Act as an AI Invoice Auditor for Coupa.
                    INVOICE STATUS: {status.upper()}
                    
                    DATA:
                    - Invoice: {selected_id}
                    - Description: "{df_final.iloc[0]['Description']}"
                    - Amount: ${df_final.iloc[0]['Invoice_Amount']} ({df_final.iloc[0]['Currency']})
                    - Matched PO: {df_final.iloc[0]['PO_Number_Matched']} (Amount: ${df_final.iloc[0]['PO_Amount']})
                    - Variance: {variance_val:.1%} (Threshold: 2%)
                    - Match Type: {match_type} (Strict > Smart > None)
                    - Supplier Risk Score: {sup_risk} (0-100, High is risky)
                    - Supplier History: {df_final.iloc[0]['Sup_Approval_Rate']:.0%} approval rate
                    - Tenure: {df_final.iloc[0]['Sup_Tenure']} days
                    
                    TASK:
                    Generate a 'Decision Card' response in Markdown.
                    
                    IF STATUS IS 'APPROVE':
                    - Focus on safety, consistency, and matching details.
                    - Header: "✅ Ready for Approval"
                    - Bullet 1: Why it's safe (e.g. "Strict match", "Trusted supplier").
                    - Bullet 2: Key validation (e.g. "Amount within tolerance").
                    
                    IF STATUS IS 'REJECT' or 'REVIEW FURTHER':
                    - Focus on specific risks and anomalies.
                    - Header: "⚠️ Risk Detected" or "❌ Rejection Recommended"
                    - Bullet 1: The primary reason (e.g. "Significant amount variance").
                    - Bullet 2: Secondary risk (e.g. "High risk supplier" or "Description mismatch").
                    - Bullet 3: Recommended Action (e.g. "Verify with requestor").
                    
                    Tone: Professional, direct, and concise (Finance style).
                    Do not repeat the data values unless necessary for context.
                    """
                    
                    if GEMINI_API_KEY:
                         try:
                            # Tracking Latency
                            start_time = time.time()
                            
                            # UI Progress Bar
                            progress_bar = st.progress(0, text="✨ AI Analysis in progress...")
                            for i in range(1, 40): # Faster initial jump
                                time.sleep(0.01)
                                progress_bar.progress(i)
                            
                            g_model = genai.GenerativeModel('gemini-2.5-flash')
                            
                            # Streaming Response for Speed
                            response_placeholder = st.empty()
                            full_text = ""
                            
                            current_p = 60
                            responses = g_model.generate_content(prompt, stream=True)
                            for chunk in responses:
                                try:
                                    if chunk.text:
                                        full_text += chunk.text
                                        response_placeholder.markdown(full_text + "▌")
                                except (ValueError, AttributeError):
                                    # Fallback if text is not available in this chunk
                                    continue
                                
                                # Minor progress crawl
                                current_p = min(99, current_p + 1)
                                progress_bar.progress(current_p)
                            
                            response_placeholder.markdown(full_text)
                            progress_bar.empty()
                            
                            latency = time.time() - start_time
                            
                            # Log Inference to MLflow
                            mlflow.set_experiment("Invoice_Inference_Monitoring")
                            with mlflow.start_run(run_name=f"Inference_{selected_id}", nested=True):
                                mlflow.log_params({
                                    "invoice_id": selected_id,
                                    "predicted_status": status,
                                    "match_type": match_type
                                })
                                mlflow.log_metrics({
                                    "latency_seconds": latency,
                                    "amount": df_final.iloc[0]['Invoice_Amount'],
                                    "variance": variance_val
                                })
                                
                         except Exception as e: 
                            st.error(f"AI Reasoning Error: {str(e)}")
                    else:
                        st.warning("Gemini API Key Missing")
                
                # Show details table at the end
                with st.expander("Detailed Match Data"):
                    feat_cols = ['Invoice_ID', 'Department', 'Predicted_Status', 'Match_Type', 'Amt_Variance', 'Amt_Diff_Abs', 'Sup_Is_New', 'Sup_Approval_Rate', 'Desc_Word_Count', 'Has_Urgent_Keyword']
                    st.dataframe(df_final[feat_cols])
            
    st.markdown("---")
    st.markdown("### 🔍 Live Predictions (All Invoices)")
    
    # --- UI Explanation for Clients ---
    with st.expander("❓ Understanding Live Predictions", expanded=False):
        st.info("""
        **What does this table show?**
        This section provides an automated risk assessment for every incoming invoice using our trained Machine Learning models.
        
        1.  **AI Status**: The system's recommendation (Approve, Review, or Reject).
        2.  **Confidence Score**: A measure of how 'sure' the AI is about its decision. High confidence (>90%) suggests the decision can be automated.
        3.  **Human-in-the-Loop**: Auditors should focus their time on invoices marked as **'Review Further'** or those with **'Reject'** status but low confidence.
        """)

    # Status Legend
    l1, l2, l3 = st.columns(3)
    with l1:
        st.success("**✅ Approve**: Safe to process automatically.")
    with l2:
        st.warning("**⚠️ Review**: Needs manual audit check.")
    with l3:
        st.error("**❌ Reject**: High risk - potential fraud or error.")

    # Formatted Dataframe
    st.dataframe(
        full_data[['Invoice_ID', 'Department', 'Predicted_Status', 'Confidence', 'Description', 'Invoice_Amount']],
        column_config={
            "Invoice_ID": st.column_config.TextColumn("Invoice ID", width="small"),
            "Department": st.column_config.TextColumn("Department", width="small"),
            "Predicted_Status": st.column_config.TextColumn(
                "AI Status",
                help="The model's recommended action",
                width="medium"
            ),
            "Confidence": st.column_config.ProgressColumn(
                "Confidence score",
                help="Model certainty (0.0 to 1.0)",
                format="%.2f",
                min_value=0,
                max_value=1,
            ),
            "Invoice_Amount": st.column_config.NumberColumn(
                "Amount",
                format="$%.2f",
            ),
            "Description": st.column_config.TextColumn("Description", width="large"),
        },
        use_container_width=True,
        hide_index=True
    )
