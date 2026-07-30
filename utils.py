"""
utils.py
========
Utility functions สำหรับ Dehydration Prediction System
- โหลด Dataset จาก Kaggle หรือ fallback synthetic data
- Cache ข้อมูลและโมเดล
- ฟังก์ชัน UI helpers
"""

import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# =========================
# Constants
# =========================
DATA_DIR = Path("data")
MODEL_DIR = Path("model")
DATA_FILE = DATA_DIR / "hydration_data.csv"
MODEL_FILE = MODEL_DIR / "model.joblib"
SCALER_FILE = MODEL_DIR / "scaler.joblib"
ENCODER_FILE = MODEL_DIR / "encoder.joblib"

# Kaggle dataset slug (hydration dataset)
KAGGLE_DATASET = "abdulmalik1518/hydration-dataset"

# Feature columns ที่ใช้ฝึกโมเดล
FEATURE_COLS = [
    "Age", "Gender", "Weight_kg", "Height_cm", "BMI",
    "Daily_Water_Intake_ml", "Exercise_Time_min",
    "Temperature_C", "Humidity_pct", "Heart_Rate_bpm",
    "Sleep_Hours", "Physical_Activity_Level",
    "Urine_Color", "Thirst_Level"
]

TARGET_COL = "Dehydration_Status"


# =========================
# Dataset Loading
# =========================
def _download_from_kaggle() -> bool:
    """ดาวน์โหลด dataset จาก Kaggle API"""
    try:
        # ตั้งค่า kaggle credentials จาก environment / secrets
        kaggle_user = None
        kaggle_key = None

        # ลองอ่านจาก st.secrets (Streamlit Cloud)
        try:
            kaggle_user = st.secrets.get("KAGGLE_USERNAME")
            kaggle_key = st.secrets.get("KAGGLE_KEY")
        except Exception:
            pass

        # ถ้าไม่มี ลองอ่านจาก environment variables
        if not kaggle_user:
            kaggle_user = os.environ.get("KAGGLE_USERNAME")
            kaggle_key = os.environ.get("KAGGLE_KEY")

        if kaggle_user and kaggle_key:
            os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
            cred_path = os.path.expanduser("~/.kaggle/kaggle.json")
            with open(cred_path, "w") as f:
                json.dump({"username": kaggle_user, "key": kaggle_key}, f)
            os.chmod(cred_path, 0o600)

        import kaggle
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        kaggle.api.dataset_download_files(
            KAGGLE_DATASET, path=str(DATA_DIR), unzip=True
        )

        # หาไฟล์ csv ในโฟลเดอร์ data
        csv_files = list(DATA_DIR.glob("*.csv"))
        if csv_files:
            # ถ้าชื่อไฟล์ไม่ใช่ hydration_data.csv ให้ rename
            src = csv_files[0]
            if src.name != "hydration_data.csv":
                dst = DATA_DIR / "hydration_data.csv"
                src.rename(dst)
            return True
        return False
    except Exception as e:
        print(f"[Kaggle Download Error] {e}")
        return False


def _generate_synthetic_data(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    """สร้าง synthetic hydration dataset (fallback เมื่อ Kaggle ไม่พร้อมใช้งาน)"""
    np.random.seed(seed)

    age = np.random.randint(15, 75, n_samples)
    gender = np.random.choice(["Male", "Female"], n_samples)
    weight = np.random.normal(65, 12, n_samples).clip(35, 130)
    height = np.random.normal(168, 10, n_samples).clip(140, 200)
    bmi = weight / ((height / 100) ** 2)
    water = np.random.normal(1800, 600, n_samples).clip(300, 4500)
    exercise = np.random.normal(40, 20, n_samples).clip(0, 180)
    temp = np.random.normal(30, 5, n_samples).clip(10, 45)
    humidity = np.random.normal(60, 15, n_samples).clip(10, 100)
    heart_rate = np.random.normal(75, 12, n_samples).clip(50, 130)
    sleep = np.random.normal(7, 1.5, n_samples).clip(3, 11)
    activity = np.random.choice(["Low", "Moderate", "High"], n_samples)
    urine = np.random.choice(
        ["Clear", "Light Yellow", "Yellow", "Dark Yellow", "Amber"],
        n_samples
    )
    thirst = np.random.choice(["None", "Mild", "Moderate", "Severe"], n_samples)

    # สร้าง label ตาม logic ทางการแพทย์
    score = np.zeros(n_samples)
    score += (water < 1500).astype(float) * 2
    score += (temp > 32).astype(float) * 1.5
    score += (exercise > 60).astype(float) * 1.2
    score += (humidity > 70).astype(float) * 0.8
    score += (heart_rate > 85).astype(float) * 1.0
    score += (sleep < 6).astype(float) * 1.0
    score += np.isin(urine, ["Dark Yellow", "Amber"]).astype(float) * 2.0
    score += np.isin(thirst, ["Moderate", "Severe"]).astype(float) * 1.5
    score += (activity == "High").astype(float) * 0.8
    score += (bmi > 28).astype(float) * 0.5
    score += np.random.normal(0, 0.8, n_samples)

    threshold = np.percentile(score, 55)
    dehydration = (score > threshold).astype(int)

    df = pd.DataFrame({
        "Age": age.astype(int),
        "Gender": gender,
        "Weight_kg": np.round(weight, 1),
        "Height_cm": np.round(height, 1),
        "BMI": np.round(bmi, 1),
        "Daily_Water_Intake_ml": water.astype(int),
        "Exercise_Time_min": exercise.astype(int),
        "Temperature_C": np.round(temp, 1),
        "Humidity_pct": np.round(humidity, 1),
        "Heart_Rate_bpm": heart_rate.astype(int),
        "Sleep_Hours": np.round(sleep, 1),
        "Physical_Activity_Level": activity,
        "Urine_Color": urine,
        "Thirst_Level": thirst,
        "Dehydration_Status": dehydration,
    })
    return df


@st.cache_data(show_spinner="Loading dataset...")
def load_dataset() -> pd.DataFrame:
    """โหลด dataset: ใช้ cache, โหลดจาก Kaggle หรือสร้าง synthetic data"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ถ้าไฟล์มีอยู่แล้ว ให้ใช้เลย
    if DATA_FILE.exists() and DATA_FILE.stat().st_size > 0:
        try:
            df = pd.read_csv(DATA_FILE)
            if not df.empty:
                return df
        except Exception:
            pass

    # ลองดาวน์โหลดจาก Kaggle
    if _download_from_kaggle() and DATA_FILE.exists():
        try:
            df = pd.read_csv(DATA_FILE)
            if not df.empty:
                return df
        except Exception:
            pass

    # Fallback: synthetic data
    df = _generate_synthetic_data()
    df.to_csv(DATA_FILE, index=False)
    return df


# =========================
# Data Cleaning & Encoding
# =========================
@st.cache_data
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """ทำความสะอาดข้อมูล: ลบ missing, normalize column names"""
    df = df.copy()
    # Normalize column names
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    # Mapping column names ให้ตรงกับ FEATURE_COLS
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if "age" in cl: col_map[c] = "Age"
        elif "gender" in cl or "sex" in cl: col_map[c] = "Gender"
        elif "weight" in cl: col_map[c] = "Weight_kg"
        elif "height" in cl: col_map[c] = "Height_cm"
        elif cl == "bmi": col_map[c] = "BMI"
        elif "water" in cl and "intake" in cl: col_map[c] = "Daily_Water_Intake_ml"
        elif "exercise" in cl: col_map[c] = "Exercise_Time_min"
        elif "temp" in cl: col_map[c] = "Temperature_C"
        elif "humid" in cl: col_map[c] = "Humidity_pct"
        elif "heart" in cl: col_map[c] = "Heart_Rate_bpm"
        elif "sleep" in cl: col_map[c] = "Sleep_Hours"
        elif "activity" in cl: col_map[c] = "Physical_Activity_Level"
        elif "urine" in cl: col_map[c] = "Urine_Color"
        elif "thirst" in cl: col_map[c] = "Thirst_Level"
        elif "dehydr" in cl or "label" in cl or "target" in cl or "status" in cl:
            col_map[c] = "Dehydration_Status"
    df = df.rename(columns=col_map)

    # Drop rows with missing target
    if "Dehydration_Status" in df.columns:
        df = df.dropna(subset=["Dehydration_Status"])

    # Fill numeric NaN with median
    for c in df.select_dtypes(include=[np.number]).columns:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    # Fill categorical NaN with mode
    for c in df.select_dtypes(include=["object"]).columns:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].mode()[0])

    return df


def encode_features(df: pd.DataFrame, fit: bool = False, encoder=None):
    """
    Encode categorical features เป็น numeric
    ถ้า fit=True จะสร้าง encoder ใหม่ (สำหรับ training)
    """
    df = df.copy()
    cat_cols = ["Gender", "Physical_Activity_Level", "Urine_Color", "Thirst_Level"]

    if fit:
        encoder = {}
        # Gender: Male=1, Female=0
        if "Gender" in df.columns:
            df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0}).fillna(0)
            encoder["Gender"] = {"Male": 1, "Female": 0}
        else:
            df["Gender"] = 0

        # Physical Activity
        if "Physical_Activity_Level" in df.columns:
            df["Physical_Activity_Level"] = df["Physical_Activity_Level"].map(
                {"Low": 0, "Moderate": 1, "High": 2}
            ).fillna(1)
            encoder["Physical_Activity_Level"] = {"Low": 0, "Moderate": 1, "High": 2}
        else:
            df["Physical_Activity_Level"] = 1

        # Urine Color (ยิ่งเข้มยิ่งขาดน้ำ)
        if "Urine_Color" in df.columns:
            df["Urine_Color"] = df["Urine_Color"].map({
                "Clear": 0, "Light Yellow": 1, "Yellow": 2,
                "Dark Yellow": 3, "Amber": 4
            }).fillna(2)
            encoder["Urine_Color"] = {
                "Clear": 0, "Light Yellow": 1, "Yellow": 2,
                "Dark Yellow": 3, "Amber": 4
            }
        else:
            df["Urine_Color"] = 2

        # Thirst Level
        if "Thirst_Level" in df.columns:
            df["Thirst_Level"] = df["Thirst_Level"].map({
                "None": 0, "Mild": 1, "Moderate": 2, "Severe": 3
            }).fillna(1)
            encoder["Thirst_Level"] = {
                "None": 0, "Mild": 1, "Moderate": 2, "Severe": 3
            }
        else:
            df["Thirst_Level"] = 1

        return df, encoder
    else:
        # ใช้ encoder ที่ train ไว้
        if encoder is None:
            encoder = {}
        if "Gender" in df.columns and "Gender" in encoder:
            df["Gender"] = df["Gender"].map(encoder["Gender"]).fillna(0)
        elif "Gender" in df.columns:
            df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0}).fillna(0)

        if "Physical_Activity_Level" in df.columns and "Physical_Activity_Level" in encoder:
            df["Physical_Activity_Level"] = df["Physical_Activity_Level"].map(
                encoder["Physical_Activity_Level"]
            ).fillna(1)
        elif "Physical_Activity_Level" in df.columns:
            df["Physical_Activity_Level"] = df["Physical_Activity_Level"].map(
                {"Low": 0, "Moderate": 1, "High": 2}
            ).fillna(1)

        if "Urine_Color" in df.columns and "Urine_Color" in encoder:
            df["Urine_Color"] = df["Urine_Color"].map(encoder["Urine_Color"]).fillna(2)
        elif "Urine_Color" in df.columns:
            df["Urine_Color"] = df["Urine_Color"].map({
                "Clear": 0, "Light Yellow": 1, "Yellow": 2,
                "Dark Yellow": 3, "Amber": 4
            }).fillna(2)

        if "Thirst_Level" in df.columns and "Thirst_Level" in encoder:
            df["Thirst_Level"] = df["Thirst_Level"].map(encoder["Thirst_Level"]).fillna(1)
        elif "Thirst_Level" in df.columns:
            df["Thirst_Level"] = df["Thirst_Level"].map({
                "None": 0, "Mild": 1, "Moderate": 2, "Severe": 3
            }).fillna(1)

        return df, encoder


# =========================
# UI Helpers
# =========================
def inject_custom_css():
    """Inject custom CSS สำหรับ UI สวยงาม โทนสีฟ้า"""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    /* Global */
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 100%);
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #2563EB 0%, #3B82F6 100%);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(37, 99, 235, 0.25);
        margin-bottom: 2rem;
        animation: fadeIn 0.8s ease-in;
    }
    .main-header h1 {
        color: white !important;
        margin: 0;
        font-weight: 700;
        font-size: 2.2rem;
    }
    .main-header p {
        color: rgba(255,255,255,0.95) !important;
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
        font-weight: 300;
    }
    .logo-icon {
        font-size: 3rem;
        margin-bottom: 0.5rem;
        display: inline-block;
        animation: float 3s ease-in-out infinite;
    }

    /* Card */
    .card {
        background: white;
        padding: 1.8rem;
        border-radius: 20px;
        box-shadow: 0 4px 20px rgba(37, 99, 235, 0.08);
        border: 1px solid rgba(37, 99, 235, 0.08);
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }
    .card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(37, 99, 235, 0.15);
    }

    /* Result Card */
    .result-normal {
        background: linear-gradient(135deg, #10B981 0%, #34D399 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(16, 185, 129, 0.3);
        animation: popIn 0.5s ease;
    }
    .result-dehydrated {
        background: linear-gradient(135deg, #EF4444 0%, #F87171 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(239, 68, 68, 0.3);
        animation: popIn 0.5s ease;
    }
    .result-normal h2, .result-dehydrated h2 {
        color: white !important;
        margin: 0;
        font-size: 1.8rem;
    }
    .result-normal p, .result-dehydrated p {
        color: rgba(255,255,255,0.95) !important;
        margin: 0.5rem 0;
    }

    /* Metric Box */
    .metric-box {
        background: white;
        padding: 1.2rem;
        border-radius: 15px;
        text-align: center;
        border-left: 4px solid #2563EB;
        box-shadow: 0 2px 10px rgba(37, 99, 235, 0.08);
    }
    .metric-box .value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2563EB;
    }
    .metric-box .label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #2563EB 0%, #3B82F6 100%);
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.4);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%);
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #1E40AF !important;
    }

    /* Recommendation Box */
    .recommendation {
        background: #EFF6FF;
        border-left: 4px solid #2563EB;
        padding: 1rem 1.2rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    .recommendation strong {
        color: #1E40AF;
    }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes popIn {
        0% { opacity: 0; transform: scale(0.9); }
        100% { opacity: 1; transform: scale(1); }
    }
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-8px); }
    }

    /* Hide default streamlit footer */
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)


def render_header(title: str, subtitle: str, icon: str = "💧"):
    """Render header แบบ premium"""
    st.markdown(f"""
    <div class="main-header">
        <div class="logo-icon">{icon}</div>
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar_info(accuracy: float, dataset_info: dict, model_info: dict):
    """Render sidebar information"""
    with st.sidebar:
        st.markdown("### 💧 About")
        st.markdown("""
        ระบบทำนายภาวะขาดน้ำโดยใช้ **Support Vector Machine (SVM)**
        วิเคราะห์ข้อมูลสุขภาพเพื่อประเมินความเสี่ยง
        """)

        st.markdown("---")
        st.markdown("### 📊 Model Information")
        st.markdown(f"""
        - **Algorithm:** Support Vector Machine (SVC)
        - **Kernel:** RBF
        - **Accuracy:** `{accuracy:.2%}`
        - **Status:** ✅ Ready
        """)

        st.markdown("---")
        st.markdown("### 🗂️ Dataset Information")
        st.markdown(f"""
        - **Samples:** `{dataset_info.get('samples', 0):,}`
        - **Features:** `{dataset_info.get('features', 0)}`
        - **Source:** Kaggle Hydration Dataset
        """)

        st.markdown("---")
        st.markdown("### 📈 Features Used")
        for f in FEATURE_COLS:
            st.markdown(f"- {f.replace('_', ' ').title()}")

        st.markdown("---")
        st.markdown("""
        <div style="text-align:center; color:#64748B; font-size:0.8rem; padding:1rem 0;">
            Made with ❤️ using Streamlit + SVM
        </div>
        """, unsafe_allow_html=True)