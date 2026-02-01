import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Setup
np.random.seed(42)
random.seed(42)
base_path = r"f:/Data Science/Neurealm/Intermedia_V3/dataset/"

# --- Metadata Lists ---
# Realistic descriptions for different categories
desc_templates = {
    'IT': [
        "Dell Latitude 5520 Laptop (Count: 5)", "Microsoft 365 Enterprise Licenses (Annual)", 
        "Cisco Meraki MR56 Access Points", "Lenovo ThinkPad X1 Carbon Gen 9", 
        "Cloud Hosting Services - AWS", "Software Development Consultation - Q3",
        "Monitor 27-inch 4K - Dell UltraSharp", "Server Rack Maintenance Fee"
    ],
    'Marketing': [
        "Q3 Digital Ad Campaign Management", "Social Media Content Creation - Oct", 
        "Google Ads Spend - September", "SEO Optimization Services", 
        "Creative Design for Holiday Campaign", "Video Production Services - Promo",
        "Email Marketing Platform Subscription", "Conference Sponsorship - TechSummit"
    ],
    'Finance': [
        "Audit Services - FY2025", "Tax Consultation Fees", 
        "Financial Reporting Software - Seat License", "Payroll Processing Fees - Oct",
        "Corporate Insurance Premium", "Bank Transaction Fees"
    ],
    'Operations': [
        "Office Cleaning Services - Monthly", "Security Guard Services - Nov",
        "HVAC Maintenance and Repair", "Office Supplies - Paper/Pens/Ink",
        "Breakroom Snacks and Coffee", "Facilities Management Retainer",
        "Waste Management Services", "Pest Control Quarterly Service"
    ]
}

currencies = ['USD', 'EUR', 'GBP', 'CAD']
requestors = [f'User_{i}' for i in range(1, 21)]

# 1. Generate Suppliers
num_suppliers = 50
suppliers = pd.DataFrame({
    'Supplier_ID': [f'SUP-{i:03d}' for i in range(1, num_suppliers + 1)],
    'Supplier_Name': [f'Supplier {i}' for i in range(1, num_suppliers + 1)],
    'Risk_Score': np.random.randint(1, 100, num_suppliers), # 1-100, higher is riskier
    'Tenure_Days': np.random.randint(30, 3650, num_suppliers),
    'Category': np.random.choice(['IT', 'Marketing', 'Finance', 'Operations'], num_suppliers)
})
suppliers.to_csv(base_path + "suppliers.csv", index=False)
print("Generated suppliers.csv")

# 2. Generate Purchase Orders (POs)
num_pos = 300
start_date = datetime.now() - timedelta(days=365)
pos = []
for i in range(1, num_pos + 1):
    sup_row = suppliers.sample(1).iloc[0]
    sup_id = sup_row['Supplier_ID']
    category = sup_row['Category']
    
    date_val = start_date + timedelta(days=random.randint(0, 360))
    amount = round(random.uniform(500.0, 50000.0), 2)
    desc = random.choice(desc_templates[category])
    
    pos.append({
        'PO_Number': f'PO-{i:05d}',
        'Supplier_ID': sup_id,
        'PO_Date': date_val.strftime('%Y-%m-%d'),
        'PO_Amount': amount,
        'Currency': random.choice(['USD', 'USD', 'USD', 'EUR']), # Mostly USD
        'Department': category,
        'Requestor': random.choice(requestors),
        'Description': desc, # PO Description
        'Line_Item_Count': random.randint(1, 10)
    })
df_pos = pd.DataFrame(pos)
df_pos.to_csv(base_path + "purchase_orders.csv", index=False)
print("Generated purchase_orders.csv")

# 3. Generate Invoices History (Training Data)
num_invoices = 1500
invoices = []

for i in range(1, num_invoices + 1):
    is_po_backed = random.random() > 0.3 # 70% PO backed
    match_type = 'None'
    
    if is_po_backed:
        linked_po = df_pos.sample(1).iloc[0]
        po_num = linked_po['PO_Number']
        sup_id = linked_po['Supplier_ID']
        category = linked_po['Department']
        inv_desc = linked_po['Description'] # Start with PO desc
        
        # Introduce variances with STRICT buckets for Model Learning
        variance_roll = random.random()
        
        # 50% Clean Approvals
        if variance_roll < 0.5:
            amount = linked_po['PO_Amount']
            currency = linked_po['Currency']
            status = 'Approve'
            match_type = 'Strict'
            
        # 30% Review Further (Small Variance or Currency Mismatch)
        elif variance_roll < 0.8:
            status = 'Review Further'
            if random.random() < 0.8:
                # Variance bucket: 5% to 10% (Distinct from 0%)
                amount = linked_po['PO_Amount'] * random.uniform(1.05, 1.10)
                currency = linked_po['Currency']
                match_type = 'Smart'
                inv_desc += " (Shipping/Tax Diff)"
            else:
                # Currency Flip (Obvious Mismatch)
                amount = linked_po['PO_Amount']
                currency = 'GBP' if linked_po['Currency'] == 'USD' else 'USD'
                match_type = 'Strict'
                
        # 20% Rejections (Large Variance or Mismatch)
        else:
            status = 'Reject'
            match_type = 'Strict' 
            if random.random() < 0.7:
                # Variance bucket: >15% (Clear Reject)
                amount = linked_po['PO_Amount'] * random.uniform(1.15, 1.5)
                currency = linked_po['Currency']
                match_type = 'Smart'
            else:
                # Description Mismatch
                amount = linked_po['PO_Amount']
                currency = linked_po['Currency']
                inv_desc = "Unrelated Service Fee"
                match_type = 'Strict' 
            
        date_inv = datetime.strptime(linked_po['PO_Date'], '%Y-%m-%d') + timedelta(days=random.randint(1, 45))
        dept = linked_po['Department']
        req = linked_po['Requestor']

    else:
        # Non-PO Invoice (Always Review or Reject if no PO)
        po_num = ""
        sup_row = suppliers.sample(1).iloc[0]
        sup_id = sup_row['Supplier_ID']
        category = sup_row['Category']
        
        amount = round(random.uniform(50.0, 5000.0), 2)
        date_inv = start_date + timedelta(days=random.randint(0, 360))
        currency = 'USD'
        dept = category
        req = random.choice(requestors)
        base_desc = random.choice(desc_templates[category])
        
        risk = random.random()
        if risk < 0.3:
            status = 'Review Further' # Non-PO needs review
            inv_desc = base_desc
        else:
            status = 'Reject'
            inv_desc = "Suspicious Non-PO Request"

    invoices.append({
        'Invoice_ID': f'INV-{i:06d}',
        'PO_Number': po_num,
        'Supplier_ID': sup_id,
        'Invoice_Amount': round(amount, 2),
        'Currency': currency,
        'Invoice_Date': date_inv.strftime('%Y-%m-%d'),
        'Description': inv_desc,
        'Department': dept,
        'Requestor': req,
        'Status': status
    })

df_hist = pd.DataFrame(invoices)
df_hist.to_csv(base_path + "invoices_history.csv", index=False)
print("Generated invoices_history.csv")
print("Training Data Counts:\n", df_hist['Status'].value_counts())

# 4. Generate New Invoices (Inference Data - Balanced for Demo)
# We want to force specific scenarios to show off the UI
new_invoices = []

# Scenario A: Clear Approvals (20 items)
# Exact matches, trusted suppliers
for i in range(1, 21):
    linked_po = df_pos.sample(1).iloc[0]
    new_invoices.append({
        'Invoice_ID': f'INV-NEW-{i:04d}',
        'PO_Number': linked_po['PO_Number'],
        'Supplier_ID': linked_po['Supplier_ID'],
        'Invoice_Amount': linked_po['PO_Amount'], # Exact Match
        'Currency': linked_po['Currency'],
        'Invoice_Date': datetime.now().strftime('%Y-%m-%d'),
        'Description': linked_po['Description'],
        'Department': linked_po['Department'],
        'Requestor': linked_po['Requestor']
        # Implicit Status: Approve
    })

# Scenario B: Review Further & Smart Match (20 items)
# Small variances, Shipping charges, Currency flips, AND Missing PO but Smart Match
for i in range(21, 41):
    linked_po = df_pos.sample(1).iloc[0]
    roll = random.random()
    
    if roll < 0.4: # 40% Smart Match (Missing PO Number but matches criteria)
        amount = linked_po['PO_Amount'] * random.uniform(0.99, 1.01) # Within 2%
        curr = linked_po['Currency']
        desc = linked_po['Description']
        po_num = "" # Trigger Smart Match logic
        inv_date = datetime.strptime(linked_po['PO_Date'], '%Y-%m-%d') + timedelta(days=random.randint(-10, 10))
    elif roll < 0.7: # 30% Variance Match (Strict Match but review needed)
        amount = linked_po['PO_Amount'] * random.uniform(1.05, 1.10)
        curr = linked_po['Currency']
        desc = linked_po['Description'] + " (Review)"
        po_num = linked_po['PO_Number']
        inv_date = datetime.now()
    else: # 30% Currency Flip
        amount = linked_po['PO_Amount']
        curr = 'GBP' if linked_po['Currency'] == 'USD' else 'USD'
        desc = linked_po['Description']
        po_num = linked_po['PO_Number']
        inv_date = datetime.now()
        
    new_invoices.append({
        'Invoice_ID': f'INV-NEW-{i:04d}',
        'PO_Number': po_num,
        'Supplier_ID': linked_po['Supplier_ID'],
        'Invoice_Amount': round(amount, 2),
        'Currency': curr,
        'Invoice_Date': inv_date.strftime('%Y-%m-%d'),
        'Description': desc,
        'Department': linked_po['Department'],
        'Requestor': linked_po['Requestor']
    })

# Scenario C: Rejections & Diverse Non-PO (20 items)
# Large variance, Non-PO suspicious, Non-PO Review
for i in range(41, 61):
    case = random.choice(['LargeVar', 'NonPO_Risk', 'NonPO_Review', 'DescMismatch'])
    
    if case == 'LargeVar':
        linked_po = df_pos.sample(1).iloc[0]
        new_invoices.append({
            'Invoice_ID': f'INV-NEW-{i:04d}',
            'PO_Number': linked_po['PO_Number'],
            'Supplier_ID': linked_po['Supplier_ID'],
            'Invoice_Amount': round(linked_po['PO_Amount'] * 2.5, 2),
            'Currency': linked_po['Currency'],
            'Invoice_Date': datetime.now().strftime('%Y-%m-%d'),
            'Description': linked_po['Description'],
            'Department': linked_po['Department'],
            'Requestor': linked_po['Requestor']
        })
    elif case == 'DescMismatch':
        linked_po = df_pos.sample(1).iloc[0]
        new_invoices.append({
            'Invoice_ID': f'INV-NEW-{i:04d}',
            'PO_Number': linked_po['PO_Number'],
            'Supplier_ID': linked_po['Supplier_ID'],
            'Invoice_Amount': linked_po['PO_Amount'],
            'Currency': linked_po['Currency'],
            'Invoice_Date': datetime.now().strftime('%Y-%m-%d'),
            'Description': "Personal Gym Equipment", 
            'Department': linked_po['Department'],
            'Requestor': linked_po['Requestor']
        })
    elif case == 'NonPO_Risk':
        sup_row = suppliers.sample(1).iloc[0]
        new_invoices.append({
            'Invoice_ID': f'INV-NEW-{i:04d}',
            'PO_Number': "",
            'Supplier_ID': sup_row['Supplier_ID'],
            'Invoice_Amount': 9999.99,
            'Currency': 'USD',
            'Invoice_Date': datetime.now().strftime('%Y-%m-%d'),
            'Description': "Consulting Services [URGENT WIRE]",
            'Department': sup_row['Category'],
            'Requestor': "Unknown_User"
        })
    else: # NonPO_Review
        sup_row = suppliers.sample(1).iloc[0]
        new_invoices.append({
            'Invoice_ID': f'INV-NEW-{i:04d}',
            'PO_Number': "",
            'Supplier_ID': sup_row['Supplier_ID'],
            'Invoice_Amount': round(random.uniform(100.0, 500.0), 2),
            'Currency': 'USD',
            'Invoice_Date': datetime.now().strftime('%Y-%m-%d'),
            'Description': "Miscellaneous Office Expense",
            'Department': sup_row['Category'],
            'Requestor': random.choice(requestors)
        })

df_new = pd.DataFrame(new_invoices)
df_new.to_csv(base_path + "invoices_new.csv", index=False)
print(f"Generated invoices_new.csv ({len(df_new)} items)")
