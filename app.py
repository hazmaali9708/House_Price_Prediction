"""
==============================================================
House Price Prediction System - Streamlit App
==============================================================
Week 1 Project | AlgoHub Software House - ML Internship Program

Run with:
    streamlit run app.py
==============================================================
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
IMAGES_DIR = os.path.join(BASE_DIR, "images")


# --------------------------------------------------------------------------
# Load artifacts
# --------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    required = [
        "linear_regression.pkl", "random_forest.pkl", "scaler.pkl",
        "feature_names.pkl", "zip_codes.pkl", "best_model_name.pkl",
    ]
    missing = [f for f in required if not os.path.exists(os.path.join(MODELS_DIR, f))]

    # If models haven't been trained yet (e.g. fresh deployment where the
    # models/ folder wasn't uploaded), train them once automatically instead
    # of requiring the user to upload large .pkl files to GitHub.
    if missing:
        with st.spinner("First-time setup: training models (this happens once)..."):
            import train_model
            train_model.main()
        missing = [f for f in required if not os.path.exists(os.path.join(MODELS_DIR, f))]

    if missing:
        return None

    lr = joblib.load(os.path.join(MODELS_DIR, "linear_regression.pkl"))
    rf = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
    zip_codes = joblib.load(os.path.join(MODELS_DIR, "zip_codes.pkl"))
    best_name = joblib.load(os.path.join(MODELS_DIR, "best_model_name.pkl"))
    results = None
    results_path = os.path.join(MODELS_DIR, "model_results.csv")
    if os.path.exists(results_path):
        results = pd.read_csv(results_path, index_col=0)
    return {
        "Linear Regression": lr,
        "Random Forest": rf,
        "scaler": scaler,
        "feature_names": feature_names,
        "zip_codes": zip_codes,
        "best_name": best_name,
        "results": results,
    }


def build_feature_row(beds, baths, size, zip_code, feature_names, zip_codes):
    """Recreate the exact feature engineering used in train_model.py."""
    total_rooms = beds + baths
    zip_grouped = zip_code if zip_code in zip_codes else -1

    row = {col: 0 for col in feature_names}
    row["beds"] = beds
    row["baths"] = baths
    row["size"] = size
    row["total_rooms"] = total_rooms

    zip_col = f"zip_{zip_grouped}"
    if zip_col in row:
        row[zip_col] = 1
    # if the zip one-hot column doesn't exist (unseen zip), all zip_* stay 0,
    # which the model treats as an implicit "reference" zip group.

    return pd.DataFrame([row])[feature_names]


# --------------------------------------------------------------------------
# Sidebar - inputs
# --------------------------------------------------------------------------
artifacts = load_artifacts()

st.title("🏠 House Price Prediction System")
st.markdown(
    "Predict house prices using **Linear Regression** and **Random Forest** "
    "models trained on real housing data. *(Week 1 — AlgoHub ML Internship)*"
)

if artifacts is None:
    st.error(
        "⚠️ Model artifacts not found. Please run `python train_model.py` "
        "first to train and save the models."
    )
    st.stop()

st.sidebar.header("🔧 Enter House Details")

beds = st.sidebar.slider("Bedrooms", min_value=1, max_value=10, value=3, step=1)
baths = st.sidebar.slider("Bathrooms", min_value=1.0, max_value=8.0, value=2.0, step=0.5)
size = st.sidebar.number_input(
    "Size (sqft)", min_value=200, max_value=10000, value=1800, step=50
)
zip_code = st.sidebar.selectbox(
    "Zip Code", options=artifacts["zip_codes"], index=0
)
model_choice = st.sidebar.radio(
    "Choose Model", options=["Random Forest", "Linear Regression", "Compare Both"], index=0
)
predict_btn = st.sidebar.button("🔮 Predict Price", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption(f"Best performing model: **{artifacts['best_name']}**")


# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["💰 Prediction", "📊 Data Insights (EDA)", "📈 Model Performance"])

# ---- Tab 1: Prediction ----
with tab1:
    if predict_btn:
        X_row = build_feature_row(
            beds, baths, size, zip_code,
            artifacts["feature_names"], artifacts["zip_codes"],
        )
        X_scaled = artifacts["scaler"].transform(X_row)

        def predict_with(name):
            model = artifacts[name]
            pred = model.predict(X_scaled)[0]
            return max(pred, 0)

        if model_choice == "Compare Both":
            col1, col2 = st.columns(2)
            lr_pred = predict_with("Linear Regression")
            rf_pred = predict_with("Random Forest")
            with col1:
                st.metric("Linear Regression Estimate", f"${lr_pred:,.0f}")
            with col2:
                st.metric("Random Forest Estimate", f"${rf_pred:,.0f}")
            st.info(
                f"Estimates differ by **${abs(lr_pred - rf_pred):,.0f}**. "
                f"The Random Forest model generally captures non-linear "
                f"patterns (like location effects) better."
            )
        else:
            pred = predict_with(model_choice)
            st.success(f"### Estimated Price ({model_choice}): ${pred:,.0f}")

        st.markdown("#### Input Summary")
        st.table(pd.DataFrame({
            "Feature": ["Bedrooms", "Bathrooms", "Size (sqft)", "Zip Code"],
            "Value": [beds, baths, size, zip_code],
        }))
    else:
        st.info("👈 Enter house details in the sidebar and click **Predict Price**.")

    img_path = os.path.join(IMAGES_DIR, "actual_vs_predicted.png")
    if os.path.exists(img_path):
        st.markdown("#### Model Fit: Actual vs Predicted Prices (Test Set)")
        st.image(img_path, use_container_width=True)

# ---- Tab 2: EDA ----
with tab2:
    st.markdown("### Exploratory Data Analysis")
    col1, col2 = st.columns(2)
    with col1:
        p = os.path.join(IMAGES_DIR, "price_distribution.png")
        if os.path.exists(p):
            st.image(p, caption="Price Distribution", use_container_width=True)
        p = os.path.join(IMAGES_DIR, "size_vs_price.png")
        if os.path.exists(p):
            st.image(p, caption="Size vs Price", use_container_width=True)
    with col2:
        p = os.path.join(IMAGES_DIR, "correlation_heatmap.png")
        if os.path.exists(p):
            st.image(p, caption="Correlation Heatmap", use_container_width=True)
        p = os.path.join(IMAGES_DIR, "beds_vs_price.png")
        if os.path.exists(p):
            st.image(p, caption="Bedrooms vs Price", use_container_width=True)

# ---- Tab 3: Model performance ----
with tab3:
    st.markdown("### Model Performance Comparison")
    if artifacts["results"] is not None:
        st.dataframe(artifacts["results"].style.format("{:,.2f}"), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        p = os.path.join(IMAGES_DIR, "model_comparison.png")
        if os.path.exists(p):
            st.image(p, caption="R² Score Comparison", use_container_width=True)
    with col2:
        p = os.path.join(IMAGES_DIR, "feature_importance.png")
        if os.path.exists(p):
            st.image(p, caption="Random Forest Feature Importance", use_container_width=True)

st.markdown("---")
