import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# --- Load Model 
with open("glass_model.pkl", "rb") as file:
    model = pickle.load(file)
    
# If you used a separate scaler during training:
with open("glass_scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# --- Helper information ---
feature_names = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
class_labels = {
    1: "Building Windows (Float processed)",
    2: "Building Windows (Non Float processed)",
    3: "Vehicle Windows (Float processed)",
    4: "Vehicle Windows (Non Float processed)",
    5: "Containers",
    6: "Tableware",
    7: "Headlamps"
}

example_data = {
    'RI': 1.52101, 'Na': 13.64, 'Mg': 4.49, 'Al': 1.10, 'Si': 71.78,
    'K': 0.06, 'Ca': 8.75, 'Ba': 0.00, 'Fe': 0.00
}

# --- Streamlit UI ---
st.set_page_config(page_title="Glass Type Classifier", layout="centered")
st.title("🔬 Glass Type Classifier")
st.write("Enter the chemical composition of a glass sample to predict its type using a machine learning model trained on the UCI Glass dataset.")
with st.expander("ℹ️ About Class Labels"):
    for k, v in class_labels.items():
        st.markdown(f"**{k}:** {v}")

# --- Data Input UI ---
st.markdown("### Input Features")

with st.form("input_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        RI = st.number_input("Refractive Index (RI)", step=0.01)
        Na = st.number_input("Sodium (Na)", step=0.01)
        Mg = st.number_input("Magnesium (Mg)", step=0.01)
    with col2:
        Al = st.number_input("Aluminum (Al)", step=0.01)
        Si = st.number_input("Silicon (Si)", step=0.01)
        K  = st.number_input("Potassium (K)", step=0.01)
    with col3:
        Ca = st.number_input("Calcium (Ca)", step=0.01)
        Ba = st.number_input("Barium (Ba)", step=0.01)
        Fe = st.number_input("Iron (Fe)", step=0.01)
    submitted = st.form_submit_button("Predict")

if submitted:
    # --- Prepare Input and Predict ---
    input_df = pd.DataFrame([[RI, Na, Mg, Al, Si, K, Ca, Ba, Fe]], columns=feature_names)
    input_scaled = scaler.transform(input_df)  # Only if you used a scaler when training
    y_pred = model.predict(input_scaled)
    y_proba = model.predict_proba(input_scaled) if hasattr(model, "predict_proba") else None
    
    st.markdown(f"## 🧪 Glass Type Prediction: **{y_pred[0]} — {class_labels.get(y_pred[0], 'Unknown')}**")
    if y_proba is not None:
        st.write("### Prediction Probability")
        st.bar_chart(pd.Series(y_proba[0], index=model.classes_))
    st.info(f"Class label {y_pred[0]} means: {class_labels.get(y_pred[0], 'Unknown')}.")

# --- Example Data ---
if st.checkbox("Show Example Input Data"):
    st.write(pd.DataFrame([example_data]))

# --- Model Accuracy Summary (saved as .txt/csv, or hard-coded here) ---
with st.expander("📈 Model Performance Summary"):
    st.markdown("""
    - **Accuracy:** 0.85  
    - **Precision, Recall, F1:** See training notebook for full report.
    - **Algorithm:** Random Forest Classifier (example)
    - Evaluate model performance on the test set before deployment.
    """)

# --- Footer / Sidebar ---
st.sidebar.markdown(
    """
    ### Glass Type Classifier App  
    Built using Streamlit  
    [GitHub Repo](https://github.com/MJ-M31/Glass-Prediction)  
    ---
    **Date:** {0}
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M"))
)
