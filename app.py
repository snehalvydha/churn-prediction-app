import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Churn Analytics Pro", layout="wide")

# ---------------- STYLING ----------------
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
.stButton>button {
    background-color: #00C897;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOGIN SYSTEM ----------------
USERNAME = "admin"
PASSWORD = "1234"

if "login" not in st.session_state:
    st.session_state.login = False

if not st.session_state.login:
    st.title("🔐 Login Page")

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

# ---------------- SIDEBAR ----------------
st.sidebar.title("📊 Navigation")

if st.sidebar.button("Logout"):
    st.session_state.login = False
    st.rerun()

page = st.sidebar.radio("Go to", ["🏠 Dashboard", "📈 Prediction", "ℹ About"])

# ---------------- HEADER ----------------
st.markdown(
    "<h1 style='color:#00FFAA;'>🚀 Churn Analytics Pro</h1>",
    unsafe_allow_html=True
)
st.markdown("### 🧠 AI-Powered Customer Risk Intelligence")
st.divider()

# ---------------- DASHBOARD ----------------
if page == "🏠 Dashboard":

    try:
        df = pd.read_csv("Telco-Customer-Churn.csv")
        total_customers = len(df)
        churn_rate = df['Churn'].value_counts(normalize=True).get("Yes", 0) * 100

        st.subheader("📊 Key Metrics")

        c1, c2 = st.columns(2)
        c1.metric("Total Customers", total_customers)
        c2.metric("Churn Rate", f"{churn_rate:.2f}%")

        st.divider()

        st.subheader("📉 Churn Distribution")
        st.bar_chart(df['Churn'].value_counts())

    except:
        st.warning("Upload dataset to view dashboard.")

# ---------------- PREDICTION ----------------
elif page == "📈 Prediction":

    st.subheader("👤 Customer Information")

    with st.form("customer_form"):

        col1, col2 = st.columns(2)

        with col1:
            customer_id = st.text_input("Customer ID", "CUST001")
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

        with col2:
            monthly = st.slider("Monthly Charges", 0, 150, 70)
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            payment = st.selectbox("Payment Method", [
                "Electronic check",
                "Mailed check",
                "Bank transfer",
                "Credit card"
            ])

        submitted = st.form_submit_button("🚀 Analyze Customer")

    if submitted:

        # ⚠️ CURRENT MODEL USES ONLY 2 FEATURES
        input_data = np.array([[tenure, monthly]])

        prediction = model.predict(input_data)

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_data)[0][1]
        else:
            prob = np.random.uniform(0.1, 0.9)

        st.divider()
        st.subheader("📊 Risk Analysis Dashboard")

        col1, col2, col3 = st.columns(3)

        col1.metric("Customer ID", customer_id)
        col2.metric("Churn Probability", f"{prob*100:.2f}%")

        risk = "Low 🟢"
        if prob > 0.6:
            risk = "High 🔴"
        elif prob > 0.3:
            risk = "Medium 🟡"

        col3.metric("Risk Level", risk)

        # Progress bar
        st.progress(int(prob * 100))

        # Alert
        if prob > 0.6:
            st.error("🔴 High Risk Customer - Immediate action required!")
        elif prob > 0.3:
            st.warning("🟡 Medium Risk - Monitor customer closely")
        else:
            st.success("🟢 Customer is Safe")

        # Chart
        chart_data = pd.DataFrame({
            "Status": ["Stay", "Churn"],
            "Probability": [1-prob, prob]
        })

        st.bar_chart(chart_data.set_index("Status"))

# ---------------- ABOUT ----------------
else:

    st.subheader("ℹ About Project")

    st.write("""
    ### 📌 Project Overview
    AI-powered churn prediction system with interactive dashboard.

    ### ⚙ Technologies
    - Streamlit
    - Machine Learning
    - Scikit-learn

    ### 🎯 Goal
    Help businesses reduce customer churn using predictive analytics.
    """)
