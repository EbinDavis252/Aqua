import streamlit as st
import pandas as pd
import os
import pickle
import sqlite3
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- Auto Create Directories ---
os.makedirs("ml_models", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("database", exist_ok=True)

# --- Create Sample Datasets if not present ---
farmer_data_path = "data/farmer_data.csv"
water_data_path = "data/water_quality_data.csv"

if not os.path.exists(farmer_data_path):
    sample_farmers = pd.DataFrame({
        "farmer_id": ["F001", "F002", "F003", "F004", "F005"],
        "age": [35, 42, 28, 50, 37],
        "income": [50000, 30000, 60000, 25000, 70000],
        "loan_amount": [80000, 100000, 50000, 120000, 90000],
        "region": ["Andhra", "TamilNadu", "Kerala", "Andhra", "Karnataka"],
        "loan_term": [12, 24, 6, 18, 12],
        "previous_default": [0, 1, 0, 1, 0],
        "farm_type": ["Freshwater", "Brackish", "Freshwater", "Brackish", "Freshwater"],
        "loan_default": [0, 1, 0, 1, 0]
    })
    sample_farmers.to_csv(farmer_data_path, index=False)

if not os.path.exists(water_data_path):
    sample_water = pd.DataFrame({
        "farm_id": ["F001", "F002", "F003", "F004", "F005"],
        "temp": [28, 31, 29, 32, 27],
        "pH": [7.5, 6.8, 7.2, 6.5, 7.8],
        "ammonia": [0.2, 1.2, 0.4, 1.0, 0.1],
        "DO": [5.0, 3.5, 6.0, 2.5, 7.0],
        "turbidity": [3, 5, 2, 6, 1],
        "timestamp": pd.date_range(start='2025-07-01 10:00', periods=5, freq='H'),
        "failure": [0, 1, 0, 1, 0]
    })
    sample_water.to_csv(water_data_path, index=False)

# --- Train ML Models if not exist ---
def train_financial_model():
    df = pd.read_csv(farmer_data_path)
    df['region'] = df['region'].astype('category').cat.codes
    df['farm_type'] = df['farm_type'].astype('category').cat.codes
    X = df[['age', 'income', 'loan_amount', 'region', 'loan_term', 'previous_default', 'farm_type']]
    y = df['loan_default']
    model = RandomForestClassifier().fit(*train_test_split(X, y, test_size=0.2)[::2])
    with open("ml_models/financial_model.pkl", "wb") as f:
        pickle.dump(model, f)

def train_technical_model():
    df = pd.read_csv(water_data_path)
    X = df[['temp', 'pH', 'ammonia', 'DO', 'turbidity']]
    y = df['failure']
    model = RandomForestClassifier().fit(*train_test_split(X, y, test_size=0.2)[::2])
    with open("ml_models/technical_model.pkl", "wb") as f:
        pickle.dump(model, f)

if not os.path.exists("ml_models/financial_model.pkl"):
    train_financial_model()
if not os.path.exists("ml_models/technical_model.pkl"):
    train_technical_model()

# --- Load Models ---
with open("ml_models/financial_model.pkl", "rb") as f:
    financial_model = pickle.load(f)
with open("ml_models/technical_model.pkl", "rb") as f:
    technical_model = pickle.load(f)

# --- Create SQLite DB and Tables ---
conn = sqlite3.connect("database/aqua_risk.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS farmer_profiles (
        farmer_id TEXT PRIMARY KEY,
        age INTEGER,
        income REAL,
        loan_amount REAL,
        region TEXT,
        loan_term INTEGER,
        previous_default INTEGER,
        farm_type TEXT
    )
""")
cursor.execute("""
    CREATE TABLE IF NOT EXISTS water_quality (
        farm_id TEXT,
        temp REAL,
        pH REAL,
        ammonia REAL,
        DO REAL,
        turbidity REAL,
        timestamp TEXT
    )
""")
cursor.execute("""
    CREATE TABLE IF NOT EXISTS model_outputs (
        entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
        farmer_id TEXT,
        financial_risk REAL,
        technical_risk REAL,
        result_time TEXT
    )
""")
conn.commit()

# --- Streamlit UI ---
st.set_page_config(page_title="Aqua Risk Dashboard", layout="wide")
st.title("🌊 AI-Driven Risk Assessment System for Aqua Loan Providers")

def encode_inputs(region, farm_type):
    region_map = {"Andhra": 0, "TamilNadu": 1, "Kerala": 2, "Karnataka": 3}
    farm_map = {"Freshwater": 0, "Brackish": 1}
    return region_map.get(region, 0), farm_map.get(farm_type, 0)

tab1, tab2, tab3 = st.tabs(["📝 Risk Input Form", "📊 Risk Prediction", "📁 View Past Records"])

with tab1:
    st.subheader("Farmer and Farm Input")

    with st.form("input_form"):
        col1, col2 = st.columns(2)

        with col1:
            farmer_id = st.text_input("Farmer ID", "F006")
            age = st.slider("Age", 18, 70, 35)
            income = st.number_input("Monthly Income (₹)", 1000, 100000, 50000)
            loan_amount = st.number_input("Loan Amount (₹)", 5000, 200000, 80000)
            loan_term = st.slider("Loan Term (months)", 6, 36, 12)
            previous_default = st.selectbox("Previous Loan Default?", ["No", "Yes"])
            prev_def = 1 if previous_default == "Yes" else 0

        with col2:
            region = st.selectbox("Region", ["Andhra", "TamilNadu", "Kerala", "Karnataka"])
            farm_type = st.selectbox("Farm Type", ["Freshwater", "Brackish"])
            temp = st.slider("Water Temp (°C)", 20.0, 35.0, 28.0)
            ph = st.slider("pH", 5.5, 9.0, 7.5)
            ammonia = st.slider("Ammonia", 0.0, 2.0, 0.2)
            do = st.slider("Dissolved Oxygen", 1.0, 10.0, 5.0)
            turbidity = st.slider("Turbidity", 0.0, 10.0, 3.0)

        submitted = st.form_submit_button("Predict & Save")

        if submitted:
            region_code, farm_code = encode_inputs(region, farm_type)

            X_fin = pd.DataFrame([[age, income, loan_amount, region_code, loan_term, prev_def, farm_code]],
                                 columns=["age", "income", "loan_amount", "region", "loan_term", "previous_default", "farm_type"])
            X_tech = pd.DataFrame([[temp, ph, ammonia, do, turbidity]],
                                  columns=["temp", "pH", "ammonia", "DO", "turbidity"])

            financial_risk = financial_model.predict_proba(X_fin)[0][1]
            technical_risk = technical_model.predict_proba(X_tech)[0][1]

            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.success("Prediction completed.")
            st.metric("📉 Loan Default Risk", f"{financial_risk:.2%}")
            st.metric("🐟 Farm Failure Risk", f"{technical_risk:.2%}")

            cursor.execute("INSERT OR REPLACE INTO farmer_profiles VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                           (farmer_id, age, income, loan_amount, region, loan_term, prev_def, farm_type))
            cursor.execute("INSERT INTO water_quality VALUES (?, ?, ?, ?, ?, ?, ?)",
                           (farmer_id, temp, ph, ammonia, do, turbidity, now))
            cursor.execute("INSERT INTO model_outputs (farmer_id, financial_risk, technical_risk, result_time) VALUES (?, ?, ?, ?)",
                           (farmer_id, financial_risk, technical_risk, now))
            conn.commit()

with tab3:
    st.subheader("📁 Risk Records")
    df = pd.read_sql_query("""
        SELECT m.result_time, m.farmer_id, f.region, f.farm_type,
               m.financial_risk, m.technical_risk
        FROM model_outputs m
        JOIN farmer_profiles f ON m.farmer_id = f.farmer_id
        ORDER BY m.result_time DESC
    """, conn)
    st.dataframe(df, use_container_width=True)
    st.download_button("Download CSV", data=df.to_csv(index=False), file_name="risk_results.csv")

with tab2:
    st.subheader("📊 Predict new risks using the form in Tab 1.")
