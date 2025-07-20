import streamlit as st
import pandas as pd
import pickle
import sqlite3
from datetime import datetime

# Load trained models
with open("ml_models/financial_model.pkl", "rb") as f:
    financial_model = pickle.load(f)

with open("ml_models/technical_model.pkl", "rb") as f:
    technical_model = pickle.load(f)

# Connect to SQLite DB
conn = sqlite3.connect("database/aqua_risk.db", check_same_thread=False)
cursor = conn.cursor()

st.set_page_config(page_title="Aqua Loan Risk Assessment", layout="wide")
st.title("🌊 AI-Driven Risk Assessment System for Aqua Loan Providers")

# Utility function to encode categorical data
def encode_inputs(region, farm_type):
    region_map = {"Andhra": 0, "TamilNadu": 1, "Kerala": 2, "Karnataka": 3}
    farm_map = {"Freshwater": 0, "Brackish": 1}
    return region_map.get(region, 0), farm_map.get(farm_type, 0)

# Tabs
tab1, tab2, tab3 = st.tabs(["📝 Risk Input Form", "📊 Risk Prediction", "📁 View Past Records"])

with tab1:
    st.header("📝 Enter Farmer and Farm Details")

    with st.form("risk_form"):
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
            temp = st.slider("Water Temperature (°C)", 20.0, 35.0, 28.0)
            ph = st.slider("pH Level", 5.5, 9.0, 7.5)
            ammonia = st.slider("Ammonia (mg/L)", 0.0, 2.0, 0.2)
            do = st.slider("Dissolved Oxygen (mg/L)", 1.0, 10.0, 5.0)
            turbidity = st.slider("Turbidity (NTU)", 0.0, 10.0, 3.0)

        submitted = st.form_submit_button("Predict & Save")

        if submitted:
            region_code, farm_type_code = encode_inputs(region, farm_type)

            # Predict financial risk
            X_fin = pd.DataFrame([[age, income, loan_amount, region_code, loan_term, prev_def, farm_type_code]],
                                 columns=["age", "income", "loan_amount", "region", "loan_term", "previous_default", "farm_type"])
            financial_risk = financial_model.predict_proba(X_fin)[0][1]

            # Predict technical risk
            X_tech = pd.DataFrame([[temp, ph, ammonia, do, turbidity]],
                                  columns=["temp", "pH", "ammonia", "DO", "turbidity"])
            technical_risk = technical_model.predict_proba(X_tech)[0][1]

            # Display
            st.success("✅ Predictions Complete")
            st.metric("💰 Loan Default Risk", f"{financial_risk:.2%}")
            st.metric("🧪 Fish Farm Failure Risk", f"{technical_risk:.2%}")

            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Save to DB
            cursor.execute("""
                INSERT INTO farmer_profiles (farmer_id, age, income, loan_amount, region, loan_term, previous_default, farm_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (farmer_id, age, income, loan_amount, region, loan_term, prev_def, farm_type))
            cursor.execute("""
                INSERT INTO water_quality (farm_id, temp, pH, ammonia, DO, turbidity, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (farmer_id, temp, ph, ammonia, do, turbidity, now))
            cursor.execute("""
                INSERT INTO model_outputs (farmer_id, financial_risk, technical_risk, result_time)
                VALUES (?, ?, ?, ?)
            """, (farmer_id, financial_risk, technical_risk, now))
            conn.commit()

with tab2:
    st.header("📊 Risk Prediction Dashboard")
    st.write("Use the input form to get predictions and save results.")

with tab3:
    st.header("📁 Past Risk Records")

    cursor.execute("""
        SELECT m.entry_id, m.result_time, m.farmer_id, f.region, f.farm_type, 
               m.financial_risk, m.technical_risk
        FROM model_outputs m
        JOIN farmer_profiles f ON m.farmer_id = f.farmer_id
        ORDER BY m.result_time DESC
    """)
    records = cursor.fetchall()
    df_records = pd.DataFrame(records, columns=["ID", "Timestamp", "Farmer ID", "Region", "Farm Type",
                                                 "Loan Default Risk", "Farm Failure Risk"])
    st.dataframe(df_records, use_container_width=True)

    st.markdown("⬇️ Download CSV")
    st.download_button("Download Records", data=df_records.to_csv(index=False), file_name="risk_results.csv")

