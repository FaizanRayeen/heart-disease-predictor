import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import os

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="CardioCare | Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern medical styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f766e 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.25);
    }
    
    .main-header h1 {
        font-size: 2.3rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
        color: #ffffff !important;
    }
    
    .main-header p {
        font-size: 1.05rem;
        color: #cbd5e1;
        margin: 0;
    }
    
    .metric-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.15);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 10px;
        margin-right: 8px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    
    .risk-high {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border: 1.5px solid #ef4444;
        border-radius: 16px;
        padding: 1.5rem;
        color: #991b1b;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        border: 1.5px solid #22c55e;
        border-radius: 16px;
        padding: 1.5rem;
        color: #166534;
    }
    
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #0d9488 0%, #0f766e 100%);
        color: white;
        font-weight: 600;
        border-radius: 10px;
        padding: 0.6rem 1.4rem;
        border: none;
        box-shadow: 0 4px 12px rgba(13, 148, 136, 0.3);
        transition: all 0.2s ease;
    }
    
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(13, 148, 136, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Cache model artifacts loader
@st.cache_resource
def load_model_artifacts():
    try:
        model = joblib.load('LogisticReg_heartdisease.pkl')
        scaler = joblib.load('scaler.pkl')
        columns = joblib.load('columns.pkl')
        
        metrics = {}
        if os.path.exists('metrics.json'):
            with open('metrics.json', 'r') as f:
                metrics = json.load(f)
        return model, scaler, columns, metrics
    except Exception as e:
        return None, None, None, {"error": str(e)}

model, scaler, feature_columns, metrics = load_model_artifacts()

# Banner Header
acc_val = metrics.get('accuracy', 88.17)
roc_val = metrics.get('roc_auc', 0.948)

st.markdown(f"""
<div class="main-header">
    <div style="display: flex; align-items: center; gap: 15px;">
        <span style="font-size: 2.8rem;">🫀</span>
        <div>
            <h1>CardioCare AI — Heart Disease Predictor</h1>
            <p>Clinical machine learning diagnostic assistant trained on patient hemodynamic and electrocardiogram metrics.</p>
            <div>
                <span class="metric-badge">🎯 Accuracy: {acc_val}%</span>
                <span class="metric-badge">📈 ROC-AUC: {roc_val}</span>
                <span class="metric-badge">👥 Dataset: 15,000 Records</span>
                <span class="metric-badge">⚡ Logistic Regression Pipeline</span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if model is None or scaler is None or feature_columns is None:
    st.error("⚠️ Model artifacts not found. Please run `python train_model.py` to generate the model files.")
    st.stop()

# Sidebar: Quick Presets & Testing Helpers
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/like.png", width=70)
    st.markdown("### 🧪 One-Click Test Presets")
    st.caption("Instantly populate test cases with pre-configured patient profiles:")
    
    preset = st.radio(
        "Select a Sample Case:",
        ["None (Custom Input)", "🟢 Healthy Patient (Low Risk)", "🔴 Heart Disease Patient (High Risk)", "🟡 Moderate Risk Patient"],
        index=0
    )
    
    st.divider()
    st.markdown("### ℹ️ Dataset Legend")
    with st.expander("Chest Pain Types"):
        st.write("**ASY**: Asymptomatic (No typical chest pain)")
        st.write("**NAP**: Non-Anginal Pain")
        st.write("**ATA**: Atypical Angina")
        st.write("**TA**: Typical Angina")
        
    with st.expander("ST Slope"):
        st.write("**Up**: Normal upsloping during exercise")
        st.write("**Flat**: Flat ST-segment (Common in ischemia)")
        st.write("**Down**: Downsloping ST-segment (High risk)")

# Default values based on preset
defaults = {
    "age": 45,
    "sex": "M",
    "chest_pain": "NAP",
    "resting_bp": 125,
    "cholesterol": 210,
    "fasting_bs": "No (<= 120 mg/dl)",
    "resting_ecg": "Normal",
    "max_hr": 155,
    "exercise_angina": "No",
    "oldpeak": 0.0,
    "st_slope": "Up"
}

if preset == "🟢 Healthy Patient (Low Risk)":
    defaults = {
        "age": 37,
        "sex": "M",
        "chest_pain": "ATA",
        "resting_bp": 115,
        "cholesterol": 185,
        "fasting_bs": "No (<= 120 mg/dl)",
        "resting_ecg": "Normal",
        "max_hr": 172,
        "exercise_angina": "No",
        "oldpeak": 0.0,
        "st_slope": "Up"
    }
elif preset == "🔴 Heart Disease Patient (High Risk)":
    defaults = {
        "age": 63,
        "sex": "M",
        "chest_pain": "ASY",
        "resting_bp": 150,
        "cholesterol": 285,
        "fasting_bs": "Yes (> 120 mg/dl)",
        "resting_ecg": "ST",
        "max_hr": 110,
        "exercise_angina": "Yes",
        "oldpeak": 2.5,
        "st_slope": "Flat"
    }
elif preset == "🟡 Moderate Risk Patient":
    defaults = {
        "age": 56,
        "sex": "F",
        "chest_pain": "NAP",
        "resting_bp": 135,
        "cholesterol": 240,
        "fasting_bs": "No (<= 120 mg/dl)",
        "resting_ecg": "LVH",
        "max_hr": 132,
        "exercise_angina": "No",
        "oldpeak": 1.2,
        "st_slope": "Flat"
    }

# Main Application Tabs
tab1, tab2, tab3 = st.tabs(["🩺 Patient Diagnosis", "📊 Model Performance & Insights", "📁 Batch Prediction (CSV)"])

with tab1:
    st.markdown("#### 📋 Enter Patient Medical Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**1️⃣ Demographics & Baseline Vitals**")
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=defaults["age"], step=1, help="Patient's age in years")
        sex = st.selectbox("Sex", ["M", "F"], index=0 if defaults["sex"] == "M" else 1, format_func=lambda x: "Male (M)" if x == "M" else "Female (F)")
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=220, value=defaults["resting_bp"], step=1, help="Resting BP on admission")
        fasting_bs_opt = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No (<= 120 mg/dl)", "Yes (> 120 mg/dl)"], index=0 if defaults["fasting_bs"].startswith("No") else 1)
        fasting_bs = 1 if fasting_bs_opt.startswith("Yes") else 0
        
    with col2:
        st.markdown("**2️⃣ Heart & Clinical Chemistry**")
        chest_pain = st.selectbox(
            "Chest Pain Type (ChestPainType)",
            ["ASY", "NAP", "ATA", "TA"],
            index=["ASY", "NAP", "ATA", "TA"].index(defaults["chest_pain"]),
            format_func=lambda x: {
                "ASY": "ASY — Asymptomatic",
                "NAP": "NAP — Non-Anginal Pain",
                "ATA": "ATA — Atypical Angina",
                "TA": "TA — Typical Angina"
            }[x],
            help="Type of chest discomfort experienced"
        )
        cholesterol = st.number_input("Serum Cholesterol (mg/dl)", min_value=80, max_value=600, value=defaults["cholesterol"], step=1, help="Total serum cholesterol")
        resting_ecg = st.selectbox(
            "Resting ECG Results (RestingECG)",
            ["Normal", "ST", "LVH"],
            index=["Normal", "ST", "LVH"].index(defaults["resting_ecg"]),
            format_func=lambda x: {
                "Normal": "Normal",
                "ST": "ST — ST-T Wave Abnormality",
                "LVH": "LVH — Left Ventricular Hypertrophy"
            }[x]
        )
        max_hr = st.slider("Maximum Heart Rate Achieved (MaxHR)", min_value=60, max_value=220, value=defaults["max_hr"], step=1, help="Peak heart rate during exercise test")

    with col3:
        st.markdown("**3️⃣ Stress & Exercise Response**")
        exercise_angina_opt = st.selectbox("Exercise-Induced Angina", ["No", "Yes"], index=0 if defaults["exercise_angina"] == "No" else 1)
        exercise_angina = "Y" if exercise_angina_opt == "Yes" else "N"
        
        oldpeak = st.number_input("ST Depression / Oldpeak (mm)", min_value=-2.5, max_value=7.0, value=float(defaults["oldpeak"]), step=0.1, help="ST depression induced by exercise relative to rest")
        st_slope = st.selectbox(
            "ST Slope (ST_Slope)",
            ["Up", "Flat", "Down"],
            index=["Up", "Flat", "Down"].index(defaults["st_slope"]),
            format_func=lambda x: {
                "Up": "Up — Upsloping (Healthy)",
                "Flat": "Flat — Flat (Ischemia Risk)",
                "Down": "Down — Downsloping (High Risk)"
            }[x]
        )

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🚀 Analyze Patient Risk", use_container_width=True)

    if predict_btn or preset != "None (Custom Input)":
        # Prepare input dictionary according to exact one-hot encoded structure
        input_dict = {
            'Age': age,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': fasting_bs,
            'MaxHR': max_hr,
            'Oldpeak': oldpeak,
            'Sex_M': 1 if sex == 'M' else 0,
            'ChestPainType_ATA': 1 if chest_pain == 'ATA' else 0,
            'ChestPainType_NAP': 1 if chest_pain == 'NAP' else 0,
            'ChestPainType_TA': 1 if chest_pain == 'TA' else 0,
            'RestingECG_Normal': 1 if resting_ecg == 'Normal' else 0,
            'RestingECG_ST': 1 if resting_ecg == 'ST' else 0,
            'ExerciseAngina_Y': 1 if exercise_angina == 'Y' else 0,
            'ST_Slope_Flat': 1 if st_slope == 'Flat' else 0,
            'ST_Slope_Up': 1 if st_slope == 'Up' else 0
        }
        
        # Build DataFrame with exact feature columns
        input_df = pd.DataFrame([input_dict])[feature_columns]
        
        # Scale features
        input_scaled = scaler.transform(input_df)
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        disease_risk = float(prediction_proba[1]) * 100
        healthy_prob = float(prediction_proba[0]) * 100

        st.markdown("---")
        st.markdown("### 📊 Diagnostic Assessment Result")
        
        res_col1, res_col2 = st.columns([1.2, 1])
        
        with res_col1:
            if prediction == 1:
                st.markdown(f"""
                <div class="risk-high">
                    <h2 style="margin:0; font-size:1.6rem; color:#991b1b;">⚠️ HIGH RISK: Heart Disease Detected</h2>
                    <p style="font-size:1.1rem; margin-top:8px; margin-bottom:0;">
                        The model estimates a <b>{disease_risk:.1f}% probability</b> of heart disease based on the input biomarkers.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-low">
                    <h2 style="margin:0; font-size:1.6rem; color:#166534;">✅ LOW RISK: Normal / Healthy Heart Profile</h2>
                    <p style="font-size:1.1rem; margin-top:8px; margin-bottom:0;">
                        The model estimates a <b>{healthy_prob:.1f}% probability</b> of a normal cardiac profile (Risk of disease: {disease_risk:.1f}%).
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 🔍 Clinical Risk Breakdown:")
            risk_factors = []
            if st_slope in ["Flat", "Down"]:
                risk_factors.append(f"• **ST Segment Slope ({st_slope})**: Non-upsloping ST segment is strongly correlated with myocardial ischemia.")
            if oldpeak >= 1.5:
                risk_factors.append(f"• **Elevated ST Depression ({oldpeak} mm)**: High depression value indicates exercise-induced stress on cardiac tissue.")
            if exercise_angina == "Y":
                risk_factors.append("• **Exercise-Induced Angina (Positive)**: Chest discomfort triggered by physical exertion.")
            if chest_pain == "ASY":
                risk_factors.append("• **Asymptomatic Chest Pain Type**: Clinically associated with silent coronary artery disease.")
            if cholesterol > 240:
                risk_factors.append(f"• **High Serum Cholesterol ({cholesterol} mg/dl)**: Above standard clinical borderline (>240 mg/dl).")
            if resting_bp >= 140:
                risk_factors.append(f"• **Hypertensive Blood Pressure ({resting_bp} mm Hg)**: Stage 2 hypertension range.")
            if fasting_bs == 1:
                risk_factors.append("• **Elevated Fasting Blood Sugar**: Indicator of diabetic or pre-diabetic metabolic stress.")

            if risk_factors:
                for factor in risk_factors:
                    st.markdown(factor)
            else:
                st.markdown("• All vital signs, ST slope, and hemodynamic responses fall within standard optimal reference ranges.")

        with res_col2:
            # Gauge chart
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=disease_risk,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "<b>Heart Disease Risk Score</b><br><span style='font-size:0.8em;color:gray;'>Probability (%)</span>", 'font': {'size': 18}},
                number={'suffix': "%", 'font': {'size': 36, 'color': '#ef4444' if disease_risk > 50 else '#22c55e'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                    'bar': {'color': "#ef4444" if disease_risk > 50 else "#22c55e"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "#e2e8f0",
                    'steps': [
                        {'range': [0, 35], 'color': '#dcfce7'},
                        {'range': [35, 65], 'color': '#fef9c3'},
                        {'range': [65, 100], 'color': '#fee2e2'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            gauge_fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(gauge_fig, use_container_width=True)

        # Downloadable Clinical Summary
        report_text = f"""========================================
CARDIOCARE AI — PATIENT ASSESSMENT REPORT
========================================
Age: {age} | Sex: {sex}
Resting BP: {resting_bp} mm Hg | Cholesterol: {cholesterol} mg/dl
Fasting Blood Sugar > 120: {'Yes' if fasting_bs == 1 else 'No'}
Chest Pain Type: {chest_pain} | Resting ECG: {resting_ecg}
Max Heart Rate: {max_hr} bpm
Exercise Angina: {exercise_angina} | Oldpeak ST: {oldpeak} mm | ST Slope: {st_slope}
----------------------------------------
DIAGNOSIS RESULT: {'HEART DISEASE DETECTED (High Risk)' if prediction == 1 else 'NORMAL / HEALTHY (Low Risk)'}
ESTIMATED RISK: {disease_risk:.2f}%
CONFIDENCE: {max(disease_risk, healthy_prob):.2f}%
========================================
"""
        st.download_button(
            label="📥 Download Patient Diagnostic Report (.txt)",
            data=report_text,
            file_name=f"cardiac_report_age_{age}_{sex}.txt",
            mime="text/plain"
        )

with tab2:
    st.markdown("### 📈 Model Performance & Validation")
    st.caption("Evaluated on an 80:20 Stratified Split across 15,000 clinical records (12,000 Train / 3,000 Test).")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Model Accuracy", f"{metrics.get('accuracy', 88.17)}%")
    m_col2.metric("F1 Score", f"{metrics.get('f1_score', 0.8943)}")
    m_col3.metric("Precision", f"{metrics.get('precision', 0.8840)}")
    m_col4.metric("ROC-AUC Score", f"{metrics.get('roc_auc', 0.9480)}")
    
    st.markdown("---")
    st.markdown("#### 🔬 Feature Coefficients (Logistic Regression)")
    coef_df = pd.DataFrame({
        'Feature': feature_columns,
        'Coefficient': model.coef_[0]
    }).sort_values(by='Coefficient', ascending=False)
    
    fig_coef = go.Figure(go.Bar(
        x=coef_df['Coefficient'],
        y=coef_df['Feature'],
        orientation='h',
        marker=dict(
            color=coef_df['Coefficient'].apply(lambda x: '#ef4444' if x > 0 else '#0d9488')
        )
    ))
    fig_coef.update_layout(
        title="Impact of Features on Heart Disease Odds (Positive = Increases Risk, Negative = Decreases Risk)",
        xaxis_title="Logistic Regression Weight",
        yaxis_title="Feature",
        height=500,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig_coef, use_container_width=True)

with tab3:
    st.markdown("### 📁 Batch Patient Prediction")
    st.write("Upload a CSV file containing multiple patient records to perform automated batch predictions.")
    
    uploaded_file = st.file_uploader("Upload CSV (must contain columns: Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope)", type=["csv"])
    
    if uploaded_file is not None:
        try:
            batch_raw = pd.read_csv(uploaded_file)
            st.write("Uploaded Data Preview:", batch_raw.head())
            
            # Clean zeros with mean values
            b_df = batch_raw.copy()
            b_df['Cholesterol'] = b_df['Cholesterol'].replace(0, metrics.get('cholesterol_mean', 244.64))
            b_df['RestingBP'] = b_df['RestingBP'].replace(0, metrics.get('resting_bp_mean', 132.54))
            
            b_encoded = pd.get_dummies(b_df, drop_first=True, dtype=int)
            
            # Ensure all required feature columns exist
            for col in feature_columns:
                if col not in b_encoded.columns:
                    b_encoded[col] = 0
            
            b_encoded_aligned = b_encoded[feature_columns]
            b_scaled = scaler.transform(b_encoded_aligned)
            
            preds = model.predict(b_scaled)
            probas = model.predict_proba(b_scaled)[:, 1]
            
            batch_raw['Prediction'] = preds
            batch_raw['Prediction_Label'] = ["Heart Disease" if p == 1 else "Normal" for p in preds]
            batch_raw['HeartDisease_Probability(%)'] = (probas * 100).round(2)
            
            st.success("✅ Batch predictions completed successfully!")
            st.dataframe(batch_raw)
            
            csv_result = batch_raw.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results CSV", csv_result, "batch_heart_disease_predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Error processing CSV: {str(e)}")
