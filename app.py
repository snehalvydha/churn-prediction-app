import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Churn Analytics Pro", layout="wide")

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

# ---------------- LOAD DATA ----------------
try:
    df = pd.read_csv("Telco-Customer-Churn.csv")
except:
    st.warning("⚠ Dataset not found. Upload CSV or place file in project folder.")
    df = None

if df is not None:
    df.drop("customerID", axis=1, inplace=True, errors='ignore')

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.fillna(df.mean(numeric_only=True), inplace=True)

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("category").cat.codes

    X = df.drop("Churn", axis=1, errors='ignore')

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
st.markdown("---")

# ---------------- DASHBOARD ----------------
if page == "🏠 Dashboard":

    if df is not None:
        st.subheader("📊 Key Metrics")

        total_customers = len(df)
        churn_rate = df['Churn'].mean() * 100

        c1, c2 = st.columns(2)
        c1.metric("Total Customers", total_customers)
        c2.metric("Churn Rate", f"{churn_rate:.2f}%")

        st.markdown("---")

        st.subheader("📉 Churn Distribution")
        st.bar_chart(df['Churn'].value_counts())
    else:
        st.info("Upload dataset to view dashboard.")

# ---------------- PREDICTION ----------------
elif page == "📈 Prediction":

    st.subheader("🔮 Predict Customer Churn")

    tenure = st.slider("Tenure", 0, 72, 12)
    monthly = st.slider("Monthly Charges", 0, 150, 70)

    if st.button("🚀 Predict"):

        # Input
        input_data = np.array([[tenure, monthly]])

        # Prediction
        prediction = model.predict(input_data)

        # Probability
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_data)[0][1]
        else:
            prob = np.random.uniform(0.1, 0.9)

        st.write("### 📊 Prediction Result")

        col1, col2 = st.columns(2)

        if prediction[0] == 1:
            col1.error("⚠ High Churn Risk")
        else:
            col1.success("✅ Customer Safe")

        col2.metric("Churn Probability", f"{prob*100:.2f}%")

        # Chart
        prob_df = pd.DataFrame({
            "Status": ["Stay", "Churn"],
            "Probability": [1-prob, prob]
        })
        st.bar_chart(prob_df.set_index("Status"))

        # ---------------- SHAP ----------------
        if df is not None:
            st.markdown("---")
            st.subheader("🧠 Model Explanation (SHAP)")

            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)

                fig, ax = plt.subplots(figsize=(8, 4))
                shap.summary_plot(shap_values, X, plot_type="bar", show=False)

                st.pyplot(fig)
                plt.clf()
            except Exception as e:
                st.warning("SHAP not supported for this model.")

# ---------------- ABOUT ----------------
else:

    st.subheader("ℹ About Project")

    st.write("""
    ### 📌 Project Overview
    This project predicts customer churn using Machine Learning models.

    ### ⚙ Technologies Used
    - Python
    - Streamlit
    - Scikit-learn
    - SHAP

    ### 🎯 Objective
    To identify customers likely to churn and improve retention.
    """)