"""
Dashboard.py
============
หน้า Dashboard แสดงข้อมูลเชิงลึกของโมเดลและ dataset
- Model metrics
- Confusion matrix
- Prediction distribution
- Feature importance
- Correlation heatmap
- Data preview
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils import (
    inject_custom_css, render_header, render_sidebar_info,
    load_dataset, FEATURE_COLS
)
from train_model import load_or_train_model, get_feature_importance

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Dashboard • Dehydration Prediction",
    page_icon="📊",
    layout="wide",
)

inject_custom_css()

render_header(
    title="📊 Analytics Dashboard",
    subtitle="Model Performance & Data Insights",
    icon="📊"
)

# =========================
# Load Data & Model
# =========================
with st.spinner("🔄 Loading dataset..."):
    df = load_dataset()

with st.spinner("🧠 Loading model..."):
    model, scaler, encoder, metrics = load_or_train_model(df)

# Sidebar info
dataset_info = {"samples": len(df), "features": len(FEATURE_COLS)}
render_sidebar_info(
    accuracy=metrics["accuracy"],
    dataset_info=dataset_info,
    model_info={"algorithm": "SVM", "kernel": "rbf"}
)

# =========================
# Top Metrics
# =========================
st.markdown("### 🎯 Model Performance Metrics")
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f"""
    <div class="metric-box">
        <div class="value">{metrics['accuracy']:.2%}</div>
        <div class="label">Accuracy</div>
    </div>
    """, unsafe_allow_html=True)
with m2:
    st.markdown(f"""
    <div class="metric-box">
        <div class="value">{metrics['precision']:.2%}</div>
        <div class="label">Precision</div>
    </div>
    """, unsafe_allow_html=True)
with m3:
    st.markdown(f"""
    <div class="metric-box">
        <div class="value">{metrics['recall']:.2%}</div>
        <div class="label">Recall</div>
    </div>
    """, unsafe_allow_html=True)
with m4:
    st.markdown(f"""
    <div class="metric-box">
        <div class="value">{metrics['f1']:.2%}</div>
        <div class="label">F1-Score</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("")

# =========================
# Row 1: Confusion Matrix & Prediction Distribution
# =========================
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("#### 🔢 Confusion Matrix")
    cm = metrics["confusion_matrix"]
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Normal", "Dehydrated"],
        y=["Normal", "Dehydrated"],
        colorscale=[[0, "#10B981"], [1, "#EF4444"]],
        showscale=False,
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 20, "color": "white"},
        hovertemplate="Predicted: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>"
    ))
    fig_cm.update_layout(
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=380,
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig_cm.update_xaxes(showgrid=False)
    fig_cm.update_yaxes(showgrid=False)
    st.plotly_chart(fig_cm, use_container_width=True)

with col2:
    st.markdown("#### 🥧 Target Distribution")
    target_counts = df["Dehydration_Status"].value_counts().reset_index()
    target_counts.columns = ["Status", "Count"]
    target_counts["Status"] = target_counts["Status"].map(
        {0: "Normal", 1: "Dehydrated"}
    )
    fig_pie = px.pie(
        target_counts,
        names="Status",
        values="Count",
        color="Status",
        color_discrete_map={"Normal": "#10B981", "Dehydrated": "#EF4444"},
        hole=0.5
    )
    fig_pie.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
    )
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("")

# =========================
# Row 2: Feature Importance
# =========================
st.markdown("#### 📈 Feature Importance (SVM Approximation)")
feat_imp = get_feature_importance(model, FEATURE_COLS)
imp_df = pd.DataFrame({
    "Feature": list(feat_imp.keys()),
    "Importance": list(feat_imp.values())
}).sort_values("Importance", ascending=True)

fig_imp = px.bar(
    imp_df,
    x="Importance",
    y="Feature",
    orientation="h",
    color="Importance",
    color_continuous_scale=["#93C5FD", "#2563EB"],
)
fig_imp.update_layout(
    height=450,
    margin=dict(l=20, r=20, t=20, b=20),
    showlegend=False,
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    coloraxis_showscale=False,
)
fig_imp.update_yaxes(title=None)
fig_imp.update_xaxes(title="Relative Importance")
st.plotly_chart(fig_imp, use_container_width=True)

st.markdown("")

# =========================
# Row 3: Correlation Heatmap & Data Preview
# =========================
col3, col4 = st.columns([1.3, 1], gap="large")

with col3:
    st.markdown("#### 🔥 Correlation Heatmap")
    # Encode categorical for correlation
    df_plot = df.copy()
    df_plot["Gender_num"] = df_plot["Gender"].map({"Male": 1, "Female": 0}).fillna(0)
    df_plot["Activity_num"] = df_plot["Physical_Activity_Level"].map(
        {"Low": 0, "Moderate": 1, "High": 2}
    ).fillna(1)
    df_plot["Urine_num"] = df_plot["Urine_Color"].map({
        "Clear": 0, "Light Yellow": 1, "Yellow": 2,
        "Dark Yellow": 3, "Amber": 4
    }).fillna(2)
    df_plot["Thirst_num"] = df_plot["Thirst_Level"].map({
        "None": 0, "Mild": 1, "Moderate": 2, "Severe": 3
    }).fillna(1)

    num_cols = [
        "Age", "Gender_num", "Weight_kg", "Height_cm", "BMI",
        "Daily_Water_Intake_ml", "Exercise_Time_min",
        "Temperature_C", "Humidity_pct", "Heart_Rate_bpm",
        "Sleep_Hours", "Activity_num", "Urine_num", "Thirst_num",
        "Dehydration_Status"
    ]
    num_cols = [c for c in num_cols if c in df_plot.columns]
    corr = df_plot[num_cols].corr()

    # Rename for display
    rename_map = {
        "Gender_num": "Gender",
        "Activity_num": "Activity",
        "Urine_num": "Urine Color",
        "Thirst_num": "Thirst",
        "Daily_Water_Intake_ml": "Water Intake",
        "Exercise_Time_min": "Exercise",
        "Temperature_C": "Temperature",
        "Humidity_pct": "Humidity",
        "Heart_Rate_bpm": "Heart Rate",
        "Sleep_Hours": "Sleep",
        "Weight_kg": "Weight",
        "Height_cm": "Height",
        "Dehydration_Status": "Dehydration"
    }
    corr_display = corr.rename(index=rename_map, columns=rename_map)

    fig_corr = px.imshow(
        corr_display,
        text_auto=".2f",
        color_continuous_scale=["#2563EB", "#FFFFFF", "#EF4444"],
        zmin=-1, zmax=1,
        aspect="auto",
    )
    fig_corr.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

with col4:
    st.markdown("#### 📋 Dataset Preview")
    st.dataframe(
        df.head(10),
        use_container_width=True,
        height=500,
        hide_index=True
    )

st.markdown("")

# =========================
# Row 4: Water Intake vs Dehydration
# =========================
st.markdown("#### 💧 Water Intake vs Dehydration Risk")
fig_scatter = px.scatter(
    df.sample(min(500, len(df)), random_state=42),
    x="Daily_Water_Intake_ml",
    y="Heart_Rate_bpm",
    color="Dehydration_Status",
    color_discrete_map={0: "#10B981", 1: "#EF4444"},
    opacity=0.7,
    labels={
        "Daily_Water_Intake_ml": "Daily Water Intake (ml)",
        "Heart_Rate_bpm": "Heart Rate (bpm)",
        "Dehydration_Status": "Status"
    },
)
fig_scatter.update_layout(
    height=420,
    margin=dict(l=20, r=20, t=20, b=20),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
)
st.plotly_chart(fig_scatter, use_container_width=True)

# =========================
# Footer
# =========================
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#64748B; padding:1rem 0;">
    💧 Dehydration Prediction System • Analytics Dashboard<br>
    <small>© 2026 — Powered by Streamlit + SVM</small>
</div>
""", unsafe_allow_html=True)