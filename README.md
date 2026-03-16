# Nairobi Housing Price Prediction
## Setup & Deployment Guide

---

### STEP 1 — Download the Dataset
1. Go to: https://www.kaggle.com/datasets/destro7/nairobi-house-prices-dataset
2. Download the CSV file
3. Rename it to: `nairobi_house_prices.csv`
4. Place it in the same folder as the notebook and `app.py`

---

### STEP 2 — Install Dependencies
Open your terminal and run:
```bash
pip install -r requirements.txt
```

---

### STEP 3 — Run the Jupyter Notebook
Open `nairobi_housing.ipynb` and run all cells from top to bottom.

This will:
- Clean and preprocess the data
- Train both models (Linear Regression + Random Forest)
- Save all model assets to a folder called `model_assets/`

Make sure you see this at the end:
```
All assets saved to model_assets/
You are ready for Streamlit deployment!
```

---

### STEP 4 — Run the Streamlit App Locally
In your terminal, from the project folder:
```bash
streamlit run app.py
```
The dashboard will open in your browser at http://localhost:8501

---

### STEP 5 — Deploy to Streamlit Community Cloud (Free)

1. Push your project to a GitHub repository
   - Include: `app.py`, `requirements.txt`, `nairobi_house_prices.csv`, and the `model_assets/` folder

2. Go to: https://streamlit.io/cloud
3. Sign in with GitHub
4. Click **"New app"**
5. Select your repository, branch, and set the main file as `app.py`
6. Click **"Deploy"**

Your app will be live at:
`https://your-app-name.streamlit.app`

---

### Project Folder Structure
```
project/
├── nairobi_housing.ipynb       # Jupyter notebook (run this first)
├── app.py                      # Streamlit dashboard
├── requirements.txt            # Python dependencies
├── nairobi_house_prices.csv    # Dataset (download from Kaggle)
└── model_assets/               # Auto-generated after running notebook
    ├── random_forest_model.pkl
    ├── linear_regression_model.pkl
    ├── scaler.pkl
    ├── label_encoders.pkl
    ├── feature_columns.pkl
    └── metrics.json
```

---

### Dashboard Sections
| Tab | What it shows |
|-----|--------------|
| EDA | Price distribution, top locations, correlations, property types |
| Model Comparison | RMSE, MAE, R2 for both models side by side |
| Feature Importance | Top 10 features driving price (Random Forest) |
| Price Predictor | Input property details and get a live price estimate |