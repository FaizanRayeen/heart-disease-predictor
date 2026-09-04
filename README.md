# 🫀 CardioCare AI — Heart Disease Prediction Web Application

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end clinical machine learning decision-support system designed to predict cardiovascular heart disease risk with **89.13% diagnostic accuracy** and **91.18% sensitivity (Recall)**. Built with a full data cleaning pipeline, rigorous exploratory data analysis, and deployed as an interactive **Streamlit** web application.

---

## 🌟 Key Highlights
- **🩺 Clinical Accuracy:** Achieves **89.13% Accuracy**, **0.9029 F1-Score**, and **0.9332 ROC-AUC** using an interpretable Logistic Regression pipeline.
- **🧹 Domain-Specific Data Cleaning:** Identifies and handles **172 masked zero values in Cholesterol** and **1 in Resting Blood Pressure** (physiologically impossible in living patients) via statistical mean imputation.
- **📊 Outlier Analysis:** Outlier detection using the **Interquartile Range (IQR) method** across hemodynamic biomarkers.
- **⚡ Instant Real-Time Risk Gauge:** Dynamic Plotly probability meter (0% to 100%) categorizing patient risk into **Low (Green)**, **Moderate (Yellow)**, or **High (Red)**.
- **📁 Multi-Patient Batch Predictor:** Upload clinical CSV files with multiple patient vitals to generate batch predictions with downloadable reports.
- **📄 Downloadable Diagnostic Reports:** One-click generation of patient risk summaries with contributing clinical risk factors.

---

## 🏗️ System Architecture & Workflow

```
┌─────────────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────┐
│ 1. DATA COLLECTION      │ ───► │ 2. IN-DEPTH CLEANING    │ ───► │ 3. PREPROCESSING        │
│ • 918 Patient Records   │      │ • Expose 172 hidden 0s  │      │ • One-Hot Encoding      │
│ • 11 Medical Features   │      │ • Mean Imputation       │      │ • Stratified 80:20 Split│
│ • 1 Binary Target       │      │ • IQR Outlier Analysis  │      │ • StandardScaler Scaling│
└─────────────────────────┘      └─────────────────────────┘      └─────────────────────────┘
                                                                               │
┌─────────────────────────┐      ┌─────────────────────────┐                   ▼
│ 6. WEB DEPLOYMENT       │ ◄─── │ 5. PIPELINE EXPORT      │ ◄─── ┌─────────────────────────┐
│ • Streamlit Web UI      │      │ • LogisticReg Model .pkl│      │ 4. MODEL EVALUATION     │
│ • Interactive Gauges    │      │ • Scaler & Columns .pkl │      │ • 6 Classifiers Compared│
│ • Batch CSV Predictor   │      │ • Cached Resource Load  │      │ • 89.13% Accuracy       │
└─────────────────────────┘      └─────────────────────────┘      │ • 91.18% Recall         │
                                                                  └─────────────────────────┘
```

---

## 🩺 Clinical Features & Reference Legend

| Feature Name | Clinical Description | Normal Reference Range | High-Risk Indicator |
|---|---|---|---|
| **Age** | Patient age in years | `< 45` | `> 55` (Plaque accumulates with age) |
| **Sex** | Biological sex (`M`/`F`) | Female (Estrogen protection) | Male (Statistically earlier CAD onset) |
| **ChestPainType** | `ASY`: Asymptomatic, `NAP`: Non-Anginal, `ATA`: Atypical, `TA`: Typical | `ATA` / `NAP` | **`ASY`** (Silent Ischemia) / **`TA`** |
| **RestingBP** | Resting blood pressure in mm Hg | `110 – 120 mm Hg` | `≥ 140 mm Hg` (Stage 2 Hypertension) |
| **Cholesterol** | Serum total cholesterol in mg/dl | `< 200 mg/dl` | `> 240 mg/dl` (Hypercholesterolemia) |
| **FastingBS** | Fasting blood sugar (`1` if >120 mg/dl, `0` otherwise) | `0` (Normal) | `1` (Diabetic / Pre-diabetic state) |
| **RestingECG** | Resting electrocardiogram (`Normal`, `ST`, `LVH`) | `Normal` | `ST` (Wave abnormality) / `LVH` (Strain) |
| **MaxHR** | Peak heart rate achieved during exercise stress test | `150 – 180 bpm` | `< 115 bpm` (Impaired chronotropic response) |
| **ExerciseAngina** | Chest pain induced by exercise exertion (`Y`/`N`) | `No (N)` | **`Yes (Y)`** (Oxygen starvation during workload) |
| **Oldpeak** | ST-segment depression in mm measured under stress | `0.0 mm` | **`> 1.5 mm`** (Myocardial Ischemia) |
| **ST_Slope** | Curve angle at peak workload (`Up`, `Flat`, `Down`) | `Up` (Normal recovery) | **`Flat` / `Down`** (Severe coronary obstruction) |

---

## 📈 Model Performance & Validation

| Model Algorithm | Accuracy (%) | F1-Score | Precision | Recall (Sensitivity) | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression (Selected)** | **89.13%** | **0.9029** | **0.8942** | **0.9118** | **0.9332** |
| **Random Forest Classifier** | 88.59% | 0.8986 | 0.8857 | 0.9118 | 0.9284 |
| **Support Vector Machine (SVM)** | 88.04% | 0.8942 | 0.8774 | 0.9118 | 0.9215 |
| **K-Nearest Neighbors (KNN)** | 85.87% | 0.8756 | 0.8545 | 0.8980 | 0.9042 |
| **Naive Bayes** | 85.33% | 0.8708 | 0.8532 | 0.8894 | 0.9101 |
| **Decision Tree** | 79.35% | 0.8119 | 0.8200 | 0.8039 | 0.7925 |

> 💡 **Why Recall Matters:** In healthcare diagnostics, a **False Negative** (sending a sick patient home) can be fatal. Our model achieves **91.18% Recall**, meaning over 91 out of 100 diseased patients are accurately detected.

---

## 📁 Repository File Structure

```bash
heart-disease-predictor/ (Repository Root)
│
├── app.py                                   # Streamlit Web Application (Interactive UI & Risk Engine)
├── heart.csv                                # UCI / Kaggle Heart Disease Dataset (918 records)
├── requirements.txt                         # Production Python Dependencies
├── LogisticReg_heartdisease.pkl             # Trained Logistic Regression Model (89.13% Acc)
├── scaler.pkl                               # Pre-fitted StandardScaler
├── columns.pkl                              # Feature Column Order Schema
├── metrics.json                             # Pipeline Validation Metrics & Test Results
├── sample_test_cases.csv                    # Sample CSV for Batch Multi-Patient Testing
│
├── heart_disease_final.ipynb                # Master Jupyter Notebook (Visual EDA + Cleaning + ML)
├── Heart_Disease_Comprehensive_Viva_Guide.pdf # Presentation Guide & 12 Critical Viva Q&As
├── LICENSE                                  # MIT Open Source License
└── README.md                                # Comprehensive Project Documentation
```

---

## 🚀 Local Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/FaizanRayeen/heart-disease-predictor.git
cd heart-disease-predictor
```

### 2. Create and Activate Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
*Your browser will automatically open at `http://localhost:8501`.*

---

## ☁️ Free Cloud Deployment Guide (Streamlit Community Cloud)

You can deploy this application live to the web for free in under 2 minutes:

1. Push this repository to your **GitHub** account.
2. Go to **[share.streamlit.io](https://share.streamlit.io/)** and sign in with GitHub.
3. Click **"New app"**.
4. Select your repository: `FaizanRayeen/heart-disease-predictor`.
5. Set the **Main file path** to: `app.py`.
6. Click **"Deploy!"** 🚀  
   *Your live application URL will be generated instantly (e.g., `https://cardiocare-ai.streamlit.app`).*

---

## 🧪 Sample Test Cases for Evaluation

You can test the deployed application using the **Sidebar Presets** or manually inputting:

### 🟢 Test Case 1: Healthy Profile (Expected: Low Risk < 10%)
- **Age:** 37 | **Sex:** Male | **ChestPainType:** ATA
- **RestingBP:** 115 mm Hg | **Cholesterol:** 185 mg/dl | **FastingBS:** 0
- **RestingECG:** Normal | **MaxHR:** 172 bpm | **ExerciseAngina:** No
- **Oldpeak:** 0.0 mm | **ST_Slope:** Up
- **Result:** `✅ LOW RISK: Normal Heart Profile (< 8% Risk)`

### 🔴 Test Case 2: High-Risk Cardiac Profile (Expected: Disease > 90%)
- **Age:** 63 | **Sex:** Male | **ChestPainType:** ASY
- **RestingBP:** 150 mm Hg | **Cholesterol:** 285 mg/dl | **FastingBS:** 1
- **RestingECG:** ST | **MaxHR:** 110 bpm | **ExerciseAngina:** Yes
- **Oldpeak:** 2.5 mm | **ST_Slope:** Flat
- **Result:** `⚠️ HIGH RISK: Heart Disease Detected (> 90% Risk)`

---

## 📜 License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author
**Faizan Rayeen**  
- **GitHub:** [@FaizanRayeen](https://github.com/FaizanRayeen)
