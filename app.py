import streamlit as st
import joblib
import numpy as np

# 1. Config & Style
st.set_page_config(page_title="HeartGuard AI", page_icon="🫀", layout="wide")

st.markdown("""
    <style>
    .main {background-color: #f5f7fa;}
    .stButton>button {width: 100%; background-color: #ff4b4b; color: white; font-weight: bold; border-radius: 8px;}
    .stButton>button:hover {background-color: #e63939;}
    </style>
    """, unsafe_allow_html=True)

# 2. Load Model
@st.cache_resource
def load_model():
    return joblib.load('model.pkl'), joblib.load('scaler.pkl')

try:
    model, scaler = load_model()
except:
    st.error("Model not found. Run 'python train_model.py'")
    st.stop()

# 3. UI Layout
st.title("🫀 Heart Disease Risk AI")
st.markdown("### *Comprehensive Analysis (94% Accuracy)*")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Patient Data")
    age = st.slider("Age", 20, 100, 55)
    sex = st.radio("Sex", [1, 0], format_func=lambda x: "Male" if x==1 else "Female")
    st.caption("Developed by Kavyashree K")

# Main Columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Vitals")
    cp = st.selectbox("Chest Pain Type", [1, 2, 3, 4], 
                      format_func=lambda x: {1:"Typical Angina", 2:"Atypical Angina", 3:"Non-anginal", 4:"Asymptomatic"}.get(x, x))
    trestbps = st.slider("Resting BP (mm Hg)", 90, 200, 130)
    chol = st.slider("Cholesterol (mg/dl)", 100, 600, 240)
    fbs = st.radio("Fasting Blood Sugar > 120?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No", horizontal=True)

with col2:
    st.subheader("💓 Cardiac Metrics")
    thalach = st.slider("Max Heart Rate", 60, 220, 150)
    exang = st.radio("Exercise Angina?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No", horizontal=True)
    oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0)
    slope = st.selectbox("ST Slope", [1, 2, 3], format_func=lambda x: {1:"Upsloping", 2:"Flat", 3:"Downsloping"}.get(x, x))
    restecg = st.selectbox("Resting ECG", [0, 1, 2], help="0: Normal, 1: ST-T Wave Abnormality, 2: LV Hypertrophy")

# 4. Prediction
st.markdown("---")
if st.button("🚀 Analyze Risk"):
    # Fix input mapping (Dataset 1-4 for CP, Model expects 1-4)
    # Map inputs to array
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope]])
    
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    prob = model.predict_proba(input_scaled)[0][1] * 100

    if prediction[0] == 1:
        st.error(f"🔴 HIGH RISK ({prob:.1f}%)")
        st.write("Immediate clinical consultation recommended.")
        st.progress(int(prob))
    else:
        st.success(f"🟢 LOW RISK ({prob:.1f}%)")
        st.write("Metrics align with healthy cardiac function.")
        st.progress(int(prob))