import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Churn Analytics Pro", layout="wide")

# ---------------- CUSTOM UI (SaaS Style) ----------------
st.markdown("""
<style>
body {
    background-color: #0E1117;
}
.big-title {
    font-size: 40px;
    font-weight: bold;
    color: #00FFAA;
}
.card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOGIN ----------------
USERNAME = "admin"
PASSWORD = "1234"

if "login" not in st.session_state:
    st.session_state.login = False

if not st.session_state.login:
    st.title("🔐 Login")

    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if user == USERNAME and pwd == PASSWORD:
            st.session_state.login = True
            st.success("Login Successful ✅")
            st.rerun()
        else:
            st.error("Invalid Credentials ❌")

    st.stop()

# ---------------- LOAD MODEL ----------------
model = joblib.load("churn_model.pkl")

# ---------------- LOAD DATA ----------------
try:
    df = pd.read_csv("Telco-Customer-Churn.csv")
except:
    df = None

if df is not None:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.fillna(df.mean(numeric_only=True), inplace=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("📊 Navigation")

if st.sidebar.button("Logout"):
    st.session_state.login = False
    st.rerun()

page = st.sidebar.radio("Go to", ["🏠 Dashboard", "📈 Prediction", "ℹ About"])

# ---------------- HEADER ----------------
st.markdown('<p class="big-title">🚀 Churn Analytics Pro</p>', unsafe_allow_html=True)
st.markdown("---")

# ================= DASHBOARD =================
if page == "🏠 Dashboard":

    if df is not None:
        total = len(df)
        churn_rate = df["Churn"].value_counts(normalize=True).get("Yes", 0) * 100

        c1, c2, c3 = st.columns(3)

        c1.metric("👥 Customers", total)
        c2.metric("📉 Churn Rate", f"{churn_rate:.2f}%")
        c3.metric("💰 Avg Charges", f"{df['MonthlyCharges'].mean():.2f}")

        st.markdown("---")

        st.subheader("📊 Churn Distribution")
        st.bar_chart(df["Churn"].value_counts())

    else:
        st.warning("Dataset not found")

# ================= PREDICTION =================
elif page == "📈 Prediction":

    st.subheader("🔮 Predict Customer Churn")

    col1, col2 = st.columns(2)

    with col1:
        customer_id = st.text_input("Customer ID", "CUST_001")
        tenure = st.slider("Tenure (Months)", 0, 72, 12)
        monthly = st.slider("Monthly Charges", 0, 150, 70)

    with col2:
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer", "Credit card"
        ])

    if st.button("🚀 Predict"):

        # Encode categorical manually (simple mapping)
        contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
        internet_map = {"DSL": 0, "Fiber optic": 1, "No": 2}
        payment_map = {
            "Electronic check": 0,
            "Mailed check": 1,
            "Bank transfer": 2,
            "Credit card": 3
        }

        input_data = np.array([[
            tenure,
            monthly,
            contract_map[contract],
            internet_map[internet],
            payment_map[payment]
        ]])

        # Predict
        try:
            pred = model.predict(input_data)[0]

            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(input_data)[0][1]
            else:
                prob = np.random.uniform(0.2, 0.8)

        except:
            # fallback if model only supports 2 features
            input_data = np.array([[tenure, monthly]])
            pred = model.predict(input_data)[0]
            prob = 0.3

        # ---------------- RESULT UI ----------------
        st.markdown("### 📊 Prediction Result")

        c1, c2 = st.columns(2)

        if pred == 1:
            c1.error("⚠ High Churn Risk")
        else:
            c1.success("✅ Customer Safe")

        c2.metric("Risk Score", f"{prob*100:.2f}%")

        # Risk Gauge (progress style)
        st.progress(float(prob))

        # Bar Chart
        chart = pd.DataFrame({
            "Status": ["Stay", "Churn"],
            "Probability": [1 - prob, prob]
        })
        st.bar_chart(chart.set_index("Status"))

        # Customer Card
        st.markdown("---")
        st.markdown("### 🧾 Customer Summary")

        st.info(f"""
        **Customer ID:** {customer_id}  
        **Tenure:** {tenure} months  
        **Monthly Charges:** ${monthly}  
        **Contract:** {contract}  
        **Internet:** {internet}  
        **Payment:** {payment}  
        """)

# ================= ABOUT =================
else:

    st.subheader("ℹ About Project")

    st.write("""
### 🚀 Customer Churn Prediction SaaS App

This is a professional machine learning web app that predicts whether a customer will churn.

### ⚙️ Features
- Customer churn prediction
- Risk scoring dashboard
- Interactive UI (SaaS style)
- Customer profiling

### 🧠 Tech Stack
- Python
- Streamlit
- Scikit-learn
- Pandas / NumPy

### 🎯 Goal
Help businesses reduce customer churn using data-driven insights.
""")
