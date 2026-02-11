import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import shap
import joblib
import xgboost as xgb
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Suppress warnings
#st.set_option('deprecation.showPyplotGlobalUse', False)

# 1. Page Configuration
st.set_page_config(
    page_title='GEP-NENs Metastasis Predictor', 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main Title
st.title('A Machine Learning-based Clinical Decision Support Tool for Gastroenteropancreatic Neuroendocrine Neoplasms')
st.markdown('A Machine Learning-based Clinical Decision Support Tool for Gastroenteropancreatic Neuroendocrine Neoplasms')

# 2. Sidebar - Input Features
st.sidebar.header('Patient Clinical Features')

sex_label = st.sidebar.selectbox("Sex", ("Male", "Female"), index=0)

type_label = st.sidebar.selectbox("Type of Tumor", 
                                 ("Type 1", "Type 2", "Type 3", "Type 4", "Type 5"), 
                                 index=0)

chemo_label = st.sidebar.selectbox("Chemotherapy History", 
                                  ("No/Unknown", "Yes"), 
                                  index=0)

grade_label = st.sidebar.selectbox("Tumor Grade", (
    "Grade I (Well differentiated)", 
    "Grade II (Moderately differentiated)",
    "Grade III (Poorly differentiated)", 
    "Grade IV (Undifferentiated)"
), index=0)

site_label = st.sidebar.selectbox("Primary Site", 
                                 ("Site 1", "Site 2", "Site 3", "Site 4", "Site 5", "Site 6", "Site 7"), 
                                 index=6)

total_tumors = st.sidebar.slider("Total Number of Primary Tumors", 1, 12, value=1, step=1)

# Sidebar Footer
st.sidebar.markdown('---')
st.sidebar.info('**Disclaimer:** This tool is for research purposes only and should not replace professional medical judgment.')
st.sidebar.markdown('© 2026 All Rights Reserved')

# 3. Data Mapping Dictionary
map_dict = {
    "Male": 1, "Female": 2,
    "Type 1": 1, "Type 2": 2, "Type 3": 3, "Type 4": 4, "Type 5": 5,
    "No/Unknown": 0, "Yes": 1,
    "Grade I (Well differentiated)": 1, 
    "Grade II (Moderately differentiated)": 2, 
    "Grade III (Poorly differentiated)": 3,
    "Grade IV (Undifferentiated)": 4,
    "Site 1": 1, "Site 2": 2, "Site 3": 3, "Site 4": 4, "Site 5": 5, "Site 6": 6, "Site 7": 7
}

# 4. Load Model and Reference Data
@st.cache_resource
def load_resources():
    model = joblib.load('xgb_model.pkl')
    data = pd.read_csv("train.csv")
    return model, data

try:
    xgb_model, hp_train = load_resources()
except Exception as e:
    st.error(f"Resource Load Error: Please ensure 'xgb_model.pkl' and 'train.csv' are in the same directory.")
    st.stop()

# Prepare Input Data
features = ["Sex", "Type of tumor", "Chemotherapy Recode", "Grade Recode", "Primary Site", "Total Number Of In Situ/Malignant Tumors For Patient"]
input_values = [
    map_dict[sex_label], map_dict[type_label], map_dict[chemo_label],
    map_dict[grade_label], map_dict[site_label], total_tumors
]
input_df = pd.DataFrame([input_values], columns=features)

# 5. Prediction Execution
if st.button('Run Prediction Analysis'):
    # Probability Calculation
    prob_val = xgb_model.predict_proba(input_df)[0][1]
    risk_threshold = 0.5
    is_high_risk = prob_val > risk_threshold
    
    # Results Display
    st.subheader("Prediction Results")
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        risk_text = "High Risk" if is_high_risk else "Low Risk"
        st.metric(label="Risk Stratification", value=risk_text)
        
    with res_col2:
        st.metric(label="Metastasis Probability", value=f"{round(prob_val * 100, 2)}%")

    if not is_high_risk:
        st.success("This patient is classified into the Low Risk group.")
        st.balloons()
    else:
        st.warning("This patient is classified into the High Risk group. Close monitoring recommended.")

    # 6. Model Interpretation (SHAP)
    st.divider()
    st.subheader("Individual Feature Contribution (SHAP Explanation)")
    
    # Use TreeExplainer for XGBoost
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(input_df)
    
    # Force Plot
    st.write("**Force Plot:** Visualizing how each feature pushes the prediction higher or lower.")
    fig_force = shap.force_plot(
        explainer.expected_value, 
        shap_values[0], 
        input_df.iloc[0], 
        matplotlib=True, 
        show=False,
        plot_cmap=['#ff0051', '#4267B2'] # Red for high risk, Blue for low risk
    )
    st.pyplot(fig_force)

    # Waterfall Plot
    st.write("**Waterfall Plot:** Ranking feature importance for this specific prediction.")
    exp = shap.Explanation(
        values=shap_values[0], 
        base_values=explainer.expected_value, 
        data=input_df.iloc[0], 
        feature_names=features
    )
    fig_water, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(exp, show=False)
    st.pyplot(fig_water)

    # 7. Model Validation (Confusion Matrix)
    st.divider()
    st.subheader("Model Performance Validation")
    st.write("Current model performance across the entire study cohort:")
    
    y_pred = xgb_model.predict(hp_train[features])
    cm = confusion_matrix(hp_train["Distant"], y_pred)
    
    fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Metastasis', 'Metastasis'])
    disp.plot(cmap='Blues', ax=ax_cm)
    plt.title("Confusion Matrix (Training/Validation Set)")
    st.pyplot(fig_cm)