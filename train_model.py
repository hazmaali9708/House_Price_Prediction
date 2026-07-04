"""
==============================================================
House Price Prediction System - Full ML Pipeline
==============================================================
Week 1 Project | AlgoHub Software House - ML Internship Program

This script performs the complete machine learning pipeline:
    1. Data Loading
    2. Data Cleaning
    3. Exploratory Data Analysis (EDA) + saved plots
    4. Feature Engineering
    5. Train/Test Split + Scaling
    6. Model Training (Linear Regression & Random Forest Regressor)
    7. Model Evaluation & Comparison
    8. Saving the best model + scaler for the Streamlit app

Run:
    python train_model.py
==============================================================
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "house_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
IMAGES_DIR = os.path.join(BASE_DIR, "images")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)


# --------------------------------------------------------------------------
# 1. Data Loading
# --------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    print("\n[1/8] Loading data...")
    df = pd.read_csv(r"C:\Users\Hamaz Ali\OneDrive\Desktop\house price prididtion\datsets\house_data.csv")
    # drop stray index column saved from a previous pandas export, if present
    unnamed_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
    print(f"      Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
    return df


# --------------------------------------------------------------------------
# 2. Data Cleaning
# --------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[2/8] Cleaning data...")
    before = len(df)

    # Drop exact duplicate rows
    df = df.drop_duplicates()

    # Drop rows with missing values (dataset has none, but keep pipeline robust)
    df = df.dropna()

    # Keep only sane / physically plausible values
    df = df[(df["beds"] > 0) & (df["beds"] <= 10)]
    df = df[(df["baths"] > 0) & (df["baths"] <= 10)]
    df = df[(df["size"] > 100) & (df["size"] < 10000)]
    df = df[(df["price"] > 0)]

    # Remove extreme price outliers using the IQR method
    q1, q3 = df["price"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    df = df[(df["price"] >= lower) & (df["price"] <= upper)]

    df = df.reset_index(drop=True)
    print(f"      Removed {before - len(df)} rows (duplicates/outliers/invalid). "
          f"{len(df)} rows remain.")
    return df


# --------------------------------------------------------------------------
# 3. Exploratory Data Analysis
# --------------------------------------------------------------------------
def run_eda(df: pd.DataFrame):
    print("\n[3/8] Running EDA and saving plots to 'images/'...")

    # Price distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df["price"], kde=True, color="#1f77b4")
    plt.title("Distribution of House Prices")
    plt.xlabel("Price ($)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "price_distribution.png"), dpi=150)
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(6, 5))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close()

    # Size vs Price scatter
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="size", y="price", alpha=0.5, color="#2ca02c")
    plt.title("House Size vs Price")
    plt.xlabel("Size (sqft)")
    plt.ylabel("Price ($)")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "size_vs_price.png"), dpi=150)
    plt.close()

    # Beds vs Price boxplot
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="beds", y="price", color="#ff7f0e")
    plt.title("Bedrooms vs Price")
    plt.xlabel("Bedrooms")
    plt.ylabel("Price ($)")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "beds_vs_price.png"), dpi=150)
    plt.close()

    print("      Saved: price_distribution.png, correlation_heatmap.png, "
          "size_vs_price.png, beds_vs_price.png")


# --------------------------------------------------------------------------
# 4. Feature Engineering
# --------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[4/8] Engineering features...")
    df = df.copy()

    # Total rooms (a simple, interpretable derived feature)
    df["total_rooms"] = df["beds"] + df["baths"]

    # Price per sqft is useful for EDA but must NOT be used as a model
    # input (it is derived directly from the target), so it's dropped
    # before modelling further down.
    df["price_per_sqft"] = df["price"] / df["size"]

    # Group rare zip codes ("long tail") into an "Other" bucket so the
    # model doesn't overfit to zip codes with very few samples
    zip_counts = df["zip_code"].value_counts()
    common_zips = zip_counts[zip_counts >= 10].index
    df["zip_code_grouped"] = df["zip_code"].where(
        df["zip_code"].isin(common_zips), other=-1
    )

    print(f"      Added total_rooms, price_per_sqft (EDA-only) "
          f"and zip_code_grouped ({df['zip_code_grouped'].nunique()} groups).")
    return df


# --------------------------------------------------------------------------
# 5. Train / Test Split + Scaling
# --------------------------------------------------------------------------
def prepare_model_data(df: pd.DataFrame):
    print("\n[5/8] Preparing train/test split...")

    feature_cols = ["beds", "baths", "size", "zip_code_grouped", "total_rooms"]
    X = df[feature_cols]
    y = df["price"]

    # One-hot encode the grouped zip code (categorical, not ordinal)
    X = pd.get_dummies(X, columns=["zip_code_grouped"], prefix="zip")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"      Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]} "
          f"| Features: {X_train.shape[1]}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns.tolist()


# --------------------------------------------------------------------------
# 6. Model Training
# --------------------------------------------------------------------------
def train_models(X_train, y_train):
    print("\n[6/8] Training models...")

    # ---- Linear Regression ----
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    print("      Linear Regression trained.")

    # ---- Random Forest Regressor (with light hyperparameter search) ----
    rf_param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    }
    rf_search = GridSearchCV(
        RandomForestRegressor(random_state=42),
        rf_param_grid,
        cv=3,
        scoring="r2",
        n_jobs=-1,
    )
    rf_search.fit(X_train, y_train)
    rf = rf_search.best_estimator_
    print(f"      Random Forest trained. Best params: {rf_search.best_params_}")

    return {"Linear Regression": lr, "Random Forest": rf}


# --------------------------------------------------------------------------
# 7. Evaluation
# --------------------------------------------------------------------------
def evaluate_models(models, X_test, y_test):
    print("\n[7/8] Evaluating models...")
    results = {}

    plt.figure(figsize=(10, 5))
    for i, (name, model) in enumerate(models.items(), start=1):
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}

        print(f"      {name:18s} -> MAE: {mae:,.0f} | RMSE: {rmse:,.0f} | R2: {r2:.4f}")

        plt.subplot(1, 2, i)
        plt.scatter(y_test, preds, alpha=0.4, color="#1f77b4")
        lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
        plt.plot(lims, lims, "r--", lw=2)
        plt.xlabel("Actual Price")
        plt.ylabel("Predicted Price")
        plt.title(name)

    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "actual_vs_predicted.png"), dpi=150)
    plt.close()

    # Model comparison bar chart
    results_df = pd.DataFrame(results).T
    plt.figure(figsize=(7, 5))
    results_df["R2"].plot(kind="bar", color=["#1f77b4", "#2ca02c"])
    plt.title("Model Comparison (R² Score)")
    plt.ylabel("R² Score")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "model_comparison.png"), dpi=150)
    plt.close()

    print("      Saved: actual_vs_predicted.png, model_comparison.png")
    return results_df


# --------------------------------------------------------------------------
# 8. Feature Importance (Random Forest) + Save Artifacts
# --------------------------------------------------------------------------
def save_feature_importance(rf_model, feature_names):
    importances = pd.Series(rf_model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=True)

    plt.figure(figsize=(8, 6))
    importances.plot(kind="barh", color="#d62728")
    plt.title("Random Forest - Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "feature_importance.png"), dpi=150)
    plt.close()
    print("      Saved: feature_importance.png")


def main():
    df_raw = load_data(DATA_PATH)
    df_clean = clean_data(df_raw)
    run_eda(df_clean)
    df_feat = engineer_features(df_clean)

    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_model_data(df_feat)

    models = train_models(X_train, y_train)
    results_df = evaluate_models(models, X_test, y_test)
    save_feature_importance(models["Random Forest"], feature_names)

    # Pick the best model by R2
    best_name = results_df["R2"].idxmax()
    best_model = models[best_name]
    print(f"\n>>> Best model: {best_name} (R2 = {results_df.loc[best_name, 'R2']:.4f})")

    # Save all artifacts needed by the Streamlit app
    joblib.dump(models["Linear Regression"], os.path.join(MODELS_DIR, "linear_regression.pkl"))
    joblib.dump(models["Random Forest"], os.path.join(MODELS_DIR, "random_forest.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(feature_names, os.path.join(MODELS_DIR, "feature_names.pkl"))
    joblib.dump(sorted(df_feat["zip_code"].unique().tolist()), os.path.join(MODELS_DIR, "zip_codes.pkl"))
    joblib.dump(best_name, os.path.join(MODELS_DIR, "best_model_name.pkl"))
    results_df.to_csv(os.path.join(MODELS_DIR, "model_results.csv"))

    print("\n[8/8] All models & artifacts saved to 'models/' directory.")
    print("\n✅ Pipeline complete. You can now run: streamlit run app.py\n")


if __name__ == "__main__":
 main()
