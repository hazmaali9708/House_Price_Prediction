# 🏠 House Price Prediction System

**Week 1 Project — AlgoHub Software House | Machine Learning Internship Program**
*Level: Beginner*

A complete, end-to-end machine learning system that predicts house prices from
property features (bedrooms, bathrooms, size, and zip code) using **Linear
Regression** and **Random Forest Regressor**, with an interactive **Streamlit**
web app for live predictions.

---

## 📌 Objective

Predict house prices using regression techniques, following the full ML
lifecycle: data cleaning → EDA → feature engineering → model training →
evaluation → deployment.

## 🧠 Concepts Covered

- Data Cleaning (duplicates, invalid values, outlier removal via IQR)
- Regression (supervised learning on a continuous target)
- Feature Engineering (derived features, categorical encoding)

## 🤖 Models Used

- **Linear Regression** — simple, interpretable baseline model
- **Random Forest Regressor** — ensemble model tuned with `GridSearchCV`,
  used to capture non-linear relationships (e.g. location effects)

## 🛠 Skills Practiced

- Data Analysis
- Regression Modeling
- Model Evaluation (MAE, RMSE, R²)

## 🚀 Deployment

- **Streamlit** interactive web app for real-time price predictions

---

## 📂 Project Structure

```
house_price_prediction/
├── data/
│   └── house_data.csv          # Raw dataset (beds, baths, size, zip_code, price)
├── images/                     # Auto-generated EDA & evaluation plots
│   ├── price_distribution.png
│   ├── correlation_heatmap.png
│   ├── size_vs_price.png
│   ├── beds_vs_price.png
│   ├── actual_vs_predicted.png
│   ├── model_comparison.png
│   └── feature_importance.png
├── models/                     # Auto-generated trained model artifacts
│   ├── linear_regression.pkl
│   ├── random_forest.pkl
│   ├── scaler.pkl
│   ├── feature_names.pkl
│   ├── zip_codes.pkl
│   ├── best_model_name.pkl
│   └── model_results.csv
├── train_model.py              # Full ML pipeline (run this first)
├── app.py                      # Streamlit web app
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

1. **Clone / download this project**, then move into the folder:
   ```bash
   cd house_price_prediction
   ```

2. **(Recommended) create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ How to Run

### Step 1 — Run the ML pipeline (trains & saves the models)

```bash
python train_model.py
```

This will:
- Load and clean `data/house_data.csv`
- Generate EDA plots into `images/`
- Engineer features (total rooms, grouped zip codes)
- Train **Linear Regression** and **Random Forest** models
- Evaluate both models (MAE, RMSE, R²) and save comparison plots
- Save the trained models + scaler into `models/` for the app to use

### Step 2 — Launch the Streamlit app

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (typically `http://localhost:8501`)
in your browser.

In the app you can:
- Enter bedrooms, bathrooms, size, and zip code in the sidebar
- Choose Linear Regression, Random Forest, or compare both
- View EDA visualizations and model performance metrics in dedicated tabs

---

## 📊 Dataset

| Column     | Description                              |
|------------|-------------------------------------------|
| `beds`     | Number of bedrooms                        |
| `baths`    | Number of bathrooms                       |
| `size`     | House size in square feet                 |
| `zip_code` | Zip code of the property                  |
| `price`    | Sale price in USD (**target variable**)   |

## 📈 Model Evaluation Metrics

Both models are evaluated using:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **R² Score** (coefficient of determination)

Results are saved to `models/model_results.csv` after training, and the
better-performing model (by R²) is highlighted automatically in the app.

---

## 🔮 Future Improvements

- Add more features (lot size, year built, property condition)
- Try gradient boosting models (XGBoost / LightGBM)
- Deploy publicly via Streamlit Community Cloud or Docker

---

*Built as part of the AlgoHub Software House 8-Week Machine Learning
Internship Roadmap — Week 1: House Price Prediction System.*
