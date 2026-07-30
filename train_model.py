"""
train_model.py
==============
Training pipeline สำหรับ SVM Model
- เตรียมข้อมูล
- Train/Test split
- StandardScaler
- Train SVM
- Evaluate
- Save model
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
from pathlib import Path

from utils import (
    FEATURE_COLS, TARGET_COL,
    clean_data, encode_features,
    MODEL_DIR, MODEL_FILE, SCALER_FILE, ENCODER_FILE
)


@st.cache_resource(show_spinner="Training model...")
def train_model(df: pd.DataFrame):
    """
    ฝึก SVM model และ return (model, scaler, encoder, metrics)
    ใช้ @st.cache_resource เพื่อ cache ทั้ง pipeline
    """
    # 1. Clean data
    df = clean_data(df)

    # ตรวจสอบว่ามี columns ที่จำเป็นหรือไม่
    required = FEATURE_COLS + [TARGET_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    # 2. Encode categorical features
    df_encoded, encoder = encode_features(df, fit=True)

    # 3. Prepare X, y
    X = df_encoded[FEATURE_COLS].values
    y = df_encoded[TARGET_COL].astype(int).values

    # 4. Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. Train SVM
    model = SVC(kernel="rbf", C=1.0, probability=True, random_state=42, class_weight="balanced")
    model.fit(X_train_scaled, y_train)

    # 7. Evaluate
    y_pred = model.predict(X_test_scaled)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "test_size": len(y_test),
        "train_size": len(y_train),
    }

    # 8. Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(encoder, ENCODER_FILE)

    return model, scaler, encoder, metrics


def load_or_train_model(df: pd.DataFrame):
    """
    โหลด model จากไฟล์ ถ้าไม่มีให้ train ใหม่
    """
    if MODEL_FILE.exists() and SCALER_FILE.exists() and ENCODER_FILE.exists():
        try:
            model = joblib.load(MODEL_FILE)
            scaler = joblib.load(SCALER_FILE)
            encoder = joblib.load(ENCODER_FILE)

            # คำนวณ metrics จาก test set
            df_clean = clean_data(df)
            df_encoded, _ = encode_features(df_clean, fit=False, encoder=encoder)
            X = df_encoded[FEATURE_COLS].values
            y = df_encoded[TARGET_COL].astype(int).values
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            X_test_scaled = scaler.transform(X_test)
            y_pred = model.predict(X_test_scaled)

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "confusion_matrix": confusion_matrix(y_test, y_pred),
                "test_size": len(y_test),
                "train_size": len(y_train),
            }
            return model, scaler, encoder, metrics
        except Exception as e:
            print(f"[Load Model Error] {e}")

    # Train ใหม่
    return train_model(df)


def predict(model, scaler, encoder, input_data: dict) -> dict:
    """ทำนายจาก input data"""
    df_input = pd.DataFrame([input_data])
    df_encoded, _ = encode_features(df_input, fit=False, encoder=encoder)

    # ตรวจสอบ columns
    for c in FEATURE_COLS:
        if c not in df_encoded.columns:
            df_encoded[c] = 0

    X = df_encoded[FEATURE_COLS].values
    X_scaled = scaler.transform(X)

    prediction = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]

    return {
        "prediction": int(prediction),
        "label": "Dehydrated" if prediction == 1 else "Normal",
        "probability_normal": float(proba[0]),
        "probability_dehydrated": float(proba[1]),
        "confidence": float(max(proba)),
    }


def get_feature_importance(model, feature_names):
    """
    SVM ไม่มี feature_importances_ โดยตรง
    ใช้ approximation จาก |coef_| ของ linear kernel
    สำหรับ RBF kernel ใช้ permutation-style approximation
    """
    # ใช้ absolute value ของ support vectors mean เป็น approximation
    try:
        # สำหรับ RBF kernel ใช้ magnitude ของ support vectors
        sv = model.support_vectors_
        importance = np.abs(sv).mean(axis=0)
        importance = importance / importance.sum()
        return dict(zip(feature_names, importance))
    except Exception:
        # Fallback: equal importance
        n = len(feature_names)
        return {f: 1.0 / n for f in feature_names}