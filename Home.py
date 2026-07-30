"""
Home.py
=======
หน้าหลักของ Dehydration Prediction System
- Input form สำหรับผู้ใช้
- Prediction result
- Recommendation
"""

import streamlit as st
import pandas as pd
from utils import (
    inject_custom_css, render_header, render_sidebar_info,
    load_dataset, FEATURE_COLS
)
from train_model import load_or_train_model, predict, get_feature_importance

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Dehydration Prediction System",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# Custom CSS
# =========================
inject_custom_css()

# =========================
# Header
# =========================
render_header(
    title="Dehydration Prediction System",
    subtitle="Machine Learning based Health Prediction using Support Vector Machine",
    icon="💧"
)

# =========================
# Load Data & Model
# =========================
with st.spinner("🔄 Loading dataset..."):
    df = load_dataset()

with st.spinner("🧠 Loading model..."):
    model, scaler, encoder, metrics = load_or_train_model(df)

# =========================
# Sidebar
# =========================
dataset_info = {
    "samples": len(df),
    "features": len(FEATURE_COLS),
}
render_sidebar_info(
    accuracy=metrics["accuracy"],
    dataset_info=dataset_info,
    model_info={"algorithm": "SVM", "kernel": "rbf"}
)

# =========================
# Main Content - 2 Columns
# =========================
col_left, col_right = st.columns([1.1, 1], gap="large")

# ---------- LEFT: Input Form ----------
with col_left:
    st.markdown("### 📝 Health Information Input")
    st.markdown('<div class="card">', unsafe_allow_html=True)

    with st.form("prediction_form"):
        # Row 1: Age & Gender
        r1a, r1b = st.columns(2)
        with r1a:
            age = st.slider("🎂 Age (years)", 10, 90, 30, 1)
        with r1b:
            gender = st.selectbox("⚧ Gender", ["Male", "Female"])

        # Row 2: Weight & Height
        r2a, r2b = st.columns(2)
        with r2a:
            weight = st.number_input("⚖️ Weight (kg)", 30.0, 200.0, 65.0, 0.5)
        with r2b:
            height = st.number_input("📏 Height (cm)", 120.0, 220.0, 168.0, 0.5)

        # BMI (auto calculate)
        bmi = weight / ((height / 100) ** 2)
        st.markdown(f"**Calculated BMI:** `{bmi:.1f}`")

        # Row 3: Water & Exercise
        r3a, r3b = st.columns(2)
        with r3a:
            water = st.slider("💧 Daily Water Intake (ml)", 0, 5000, 1800, 50)
        with r3b:
            exercise = st.slider("🏃 Exercise Time (min/day)", 0, 180, 40, 5)

        # Row 4: Temperature & Humidity
        r4a, r4b = st.columns(2)
        with r4a:
            temperature = st.slider("🌡️ Temperature (°C)", 5.0, 45.0, 30.0, 0.5)
        with r4b:
            humidity = st.slider("💨 Humidity (%)", 0.0, 100.0, 60.0, 1.0)

        # Row 5: Heart Rate & Sleep
        r5a, r5b = st.columns(2)
        with r5a:
            heart_rate = st.slider("❤️ Heart Rate (bpm)", 40, 150, 75, 1)
        with r5b:
            sleep = st.slider("😴 Sleep Hours", 2.0, 12.0, 7.0, 0.5)

        # Row 6: Activity & Urine
        r6a, r6b = st.columns(2)
        with r6a:
            activity = st.selectbox(
                "🏋️ Physical Activity Level",
                ["Low", "Moderate", "High"]
            )
        with r6b:
            urine = st.selectbox(
                "🚽 Urine Color",
                ["Clear", "Light Yellow", "Yellow", "Dark Yellow", "Amber"]
            )

        # Row 7: Thirst
        thirst = st.selectbox(
            "👅 Thirst Level",
            ["None", "Mild", "Moderate", "Severe"]
        )

        st.markdown("")
        submitted = st.form_submit_button("🔮 Predict Dehydration Risk", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- RIGHT: Prediction Result ----------
with col_right:
    st.markdown("### 🎯 Prediction Result")

    if submitted:
        input_data = {
            "Age": age,
            "Gender": gender,
            "Weight_kg": weight,
            "Height_cm": height,
            "BMI": bmi,
            "Daily_Water_Intake_ml": water,
            "Exercise_Time_min": exercise,
            "Temperature_C": temperature,
            "Humidity_pct": humidity,
            "Heart_Rate_bpm": heart_rate,
            "Sleep_Hours": sleep,
            "Physical_Activity_Level": activity,
            "Urine_Color": urine,
            "Thirst_Level": thirst,
        }

        with st.spinner("🧠 Analyzing..."):
            result = predict(model, scaler, encoder, input_data)

        # Result Card
        if result["prediction"] == 0:
            st.markdown(f"""
            <div class="result-normal">
                <h2>✅ Normal Hydration</h2>
                <p>ร่างกายของคุณอยู่ในภาวะปกติ</p>
            </div>
            """, unsafe_allow_html=True)
            recommendation = """
            <div class="recommendation">
                <strong>💡 Recommendation:</strong><br>
                • ดื่มน้ำวันละ 8 แก้ว (2 ลิตร) เพื่อรักษาสภาพร่างกาย<br>
                • รักษาพฤติกรรมการดื่มน้ำและพักผ่อนที่ดี<br>
                • ออกกำลังกายสม่ำเสมอแต่อย่าหักโหม
            </div>
            """
        else:
            st.markdown(f"""
            <div class="result-dehydrated">
                <h2>⚠️ Dehydration Risk</h2>
                <p>คุณมีความเสี่ยงต่อภาวะขาดน้ำ</p>
            </div>
            """, unsafe_allow_html=True)
            recommendation = """
            <div class="recommendation">
                <strong>💡 Recommendation:</strong><br>
                • <strong>ดื่มน้ำเพิ่มทันที</strong> อย่างน้อย 500 ml<br>
                • พักผ่อนในที่ร่มหรือที่เย็น<br>
                • หลีกเลี่ยงกิจกรรมหนักจนกว่าอาการจะดีขึ้น<br>
                • หากมีอาการรุนแรง (หน้ามืด, ใจสั่น) ให้พบแพทย์
            </div>
            """

        # Metrics
        st.markdown("")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="value">{result['confidence']:.1%}</div>
                <div class="label">Confidence</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="value">{result['probability_normal']:.1%}</div>
                <div class="label">Normal Prob.</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-box">
                <div class="value">{result['probability_dehydrated']:.1%}</div>
                <div class="label">Dehydration Prob.</div>
            </div>
            """, unsafe_allow_html=True)

        # Probability Bar
        st.markdown("#### 📊 Probability Distribution")
        prob_df = pd.DataFrame({
            "Status": ["Normal", "Dehydrated"],
            "Probability": [result["probability_normal"], result["probability_dehydrated"]]
        })
        st.bar_chart(prob_df.set_index("Status"), color="#2563EB", height=200)

        # Recommendation
        st.markdown(recommendation, unsafe_allow_html=True)

    else:
        # Placeholder when no prediction yet
        st.markdown("""
        <div class="card" style="text-align:center; padding:3rem 2rem;">
            <div style="font-size:4rem; margin-bottom:1rem;">💧</div>
            <h3 style="color:#2563EB;">Fill the form to get prediction</h3>
            <p style="color:#64748B;">
                กรอกข้อมูลสุขภาพด้านซ้าย แล้วกดปุ่ม Predict<br>
                ระบบจะวิเคราะห์และแสดงผลทันที
            </p>
        </div>
        """, unsafe_allow_html=True)

# =========================
# Footer
# =========================
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#64748B; padding:1rem 0;">
    💧 Dehydration Prediction System • Powered by SVM Machine Learning<br>
    <small>© 2026 — For educational and health awareness purposes only</small>
</div>
""", unsafe_allow_html=True)