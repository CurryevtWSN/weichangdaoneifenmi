import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import shap
import joblib
import xgboost as xgb
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 屏蔽 Matplotlib 的全局使用警告
#st.set_option('deprecation.showPyplotGlobalUse', False)

# 1. 页面基本配置
st.set_page_config(
    page_title='GEP-NENs Metastasis Predictor', 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 获取当前文件的绝对路径，确保云端环境能精准定位文件
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 页面标题
st.title('Development and Validation of a Machine Learning–Based Model for Early Prediction of Distant Metastasis in Gastroenteropancreatic Neuroendocrine Neoplasms: A Cohort Study')
st.markdown('---')

# 2. 侧边栏：输入患者临床特征
st.sidebar.header('Patient Clinical Features')

sex_label = st.sidebar.selectbox("Sex", ("Male", "Female"), index=0)
type_label = st.sidebar.selectbox("Type of Tumor", ("Type 1", "Type 2", "Type 3", "Type 4", "Type 5"), index=0)
chemo_label = st.sidebar.selectbox("Chemotherapy History", ("No/Unknown", "Yes"), index=0)
grade_label = st.sidebar.selectbox("Tumor Grade", (
    "Grade I (Well differentiated)", 
    "Grade II (Moderately differentiated)",
    "Grade III (Poorly differentiated)", 
    "Grade IV (Undifferentiated)"
), index=0)
site_label = st.sidebar.selectbox("Primary Site", ("Site 1", "Site 2", "Site 3", "Site 4", "Site 5", "Site 6", "Site 7"), index=6)
total_tumors = st.sidebar.slider("Total Number of Primary Tumors", 1, 12, value=1, step=1)

# 侧边栏底部信息
st.sidebar.markdown('---')
st.sidebar.info('**Disclaimer:** This tool is for research purposes only and should not replace professional medical judgment.')
st.sidebar.markdown('© 2026 All Rights Reserved')

# 3. 数据映射字典 (必须与模型训练时的编码一致)
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

# 4. 加载资源（模型与参考数据）
# @st.cache_resource
# def load_resources():
#     # 使用 os.path.join 构建绝对路径
#     model_file = os.path.join(CURRENT_DIR, 'xgb_model.pkl')
#     data_file = os.path.join(CURRENT_DIR, 'train.csv')
    
#     # 检查文件物理存在性，若不存在则在前端报错
#     if not os.path.exists(model_file) or not os.path.exists(data_file):
#         return None, None
    
#     model = joblib.load(model_file)
#     data = pd.read_csv(data_file)
#     return model, data
xgb_model = joblib.load('xgb_model.pkl')
hp_train = pd.read_csv("train.csv")
# xgb_model, hp_train = load_resources()

# 如果资源加载失败，停止运行并提示用户
if xgb_model is None or hp_train is None:
    st.error("⚠️ Error: 'xgb_model.pkl' or 'train.csv' not found. Please ensure they are uploaded to the main directory of your GitHub repository.")
    st.stop()
else:
    # 成功加载提示，证明后端逻辑正常
    st.success("✅ Model and training data loaded successfully. Use the sidebar to set features and click the button below.")

# 准备预测输入数据
features = ["Sex", "Type of tumor", "Chemotherapy Recode", "Grade Recode", "Primary Site", "Total Number Of In Situ/Malignant Tumors For Patient"]
input_values = [
    map_dict[sex_label], map_dict[type_label], map_dict[chemo_label],
    map_dict[grade_label], map_dict[site_label], total_tumors
]
input_df = pd.DataFrame([input_values], columns=features)

# 5. 执行预测分析
if st.button('Run Prediction Analysis'):
    with st.spinner('Performing analysis...'):
        # 计算概率
        prob_val = xgb_model.predict_proba(input_df)[0][1]
        risk_threshold = 0.5
        is_high_risk = prob_val > risk_threshold
        
        # 结果展示
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

        # 6. SHAP 解释性分析
        st.divider()
        st.subheader("Individual Feature Contribution (SHAP Explanation)")
        
        # 优化点：在云端计算 SHAP 时，只取 100 条样本作为背景，防止内存溢出和超时
        explainer = shap.TreeExplainer(xgb_model, shap.sample(hp_train[features], 100))
        shap_values = explainer.shap_values(input_df)
        
        # Force Plot 绘图
        st.write("**Force Plot:** Visualizing how each feature pushes the prediction higher or lower.")
        fig_force = shap.force_plot(
            explainer.expected_value, 
            shap_values[0], 
            input_df.iloc[0], 
            matplotlib=True, 
            show=False,
            plot_cmap=['#ff0051', '#4267B2'] 
        )
        st.pyplot(fig_force)

        # Waterfall Plot 绘图
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

        # 7. 混淆矩阵展示 (模型整体性能)
        st.divider()
        st.subheader("Model Performance Validation")
        st.write("Performance metrics based on the clinical cohort:")
        
        y_pred = xgb_model.predict(hp_train[features])
        cm = confusion_matrix(hp_train["Distant"], y_pred)
        
        fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Metastasis', 'Metastasis'])
        disp.plot(cmap='Blues', ax=ax_cm)
        plt.title("Confusion Matrix")

        st.pyplot(fig_cm)
