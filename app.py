import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nairobi Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f0f1a; }
    .block-container { padding: 1.5rem 2rem; }
    h1 { color: #FF6B6B; font-size: 1.8rem !important; }
    h2 { color: #4ECDC4; font-size: 1.2rem !important; }
    h3 { color: #FFEAA7; font-size: 1rem !important; }
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e, #2a2a3e);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #FF6B6B;
        margin-bottom: 0.5rem;
    }
    .metric-label { color: #aaa; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { color: #fff; font-size: 1.4rem; font-weight: bold; }
    .predict-box {
        background: linear-gradient(135deg, #1a2a1a, #1e2e1e);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid #4ECDC4;
        text-align: center;
    }
    .price-result { color: #4ECDC4; font-size: 2rem; font-weight: bold; }
    .stTabs [data-baseweb="tab"] { color: #ccc; }
    .stTabs [aria-selected="true"] { color: #FF6B6B !important; border-bottom: 2px solid #FF6B6B; }
</style>
""", unsafe_allow_html=True)

COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
          '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']

plt.rcParams.update({
    'figure.facecolor': '#1e1e2e',
    'axes.facecolor': '#1e1e2e',
    'axes.edgecolor': '#444',
    'axes.labelcolor': '#ccc',
    'xtick.color': '#ccc',
    'ytick.color': '#ccc',
    'text.color': '#eee',
    'grid.color': '#333',
    'grid.alpha': 0.4
})


# ── Data & Model Loading ───────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('nairobi_house_prices.csv')
        df.columns = (
            df.columns.str.strip().str.lower()
            .str.replace(' ', '_').str.replace('[^a-z0-9_]', '', regex=True)
        )
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in ['float64', 'int64']:
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
        price_col = [c for c in df.columns if 'price' in c][0]
        lower = df[price_col].quantile(0.01)
        upper = df[price_col].quantile(0.99)
        df = df[(df[price_col] >= lower) & (df[price_col] <= upper)]
        return df, price_col
    except FileNotFoundError:
        return None, None


@st.cache_resource
def load_models():
    try:
        rf = joblib.load('model_assets/random_forest_model.pkl')
        lr = joblib.load('model_assets/linear_regression_model.pkl')
        scaler = joblib.load('model_assets/scaler.pkl')
        encoders = joblib.load('model_assets/label_encoders.pkl')
        features = joblib.load('model_assets/feature_columns.pkl')
        with open('model_assets/metrics.json') as f:
            metrics = json.load(f)
        return rf, lr, scaler, encoders, features, metrics
    except Exception:
        return None, None, None, None, None, None


df, price_col = load_data()
rf_model, lr_model, scaler, encoders, feature_cols, metrics = load_models()

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏠 Nairobi Housing")
    st.markdown("**Price Prediction Dashboard**")
    st.markdown("---")
    st.markdown("#### About")
    st.markdown(
        "This dashboard uses Linear Regression and Random Forest "
        "to predict residential property prices in Nairobi, Kenya."
    )
    st.markdown("---")
    if metrics:
        st.markdown("#### Model Performance")
        for model_name, m in metrics.items():
            label = "Random Forest" if "random" in model_name else "Linear Regression"
            color = "#4ECDC4" if "random" in model_name else "#FF6B6B"
            st.markdown(f"""
            <div style='background:#1e1e2e;border-radius:10px;padding:0.7rem;
                        border-left:3px solid {color};margin-bottom:0.5rem;'>
                <div style='color:#aaa;font-size:0.7rem;'>{label}</div>
                <div style='color:#fff;font-size:0.85rem;'>
                    R² <b style='color:{color}'>{m['R2']}</b> &nbsp;
                    RMSE <b>{m['RMSE']}</b>
                </div>
            </div>""", unsafe_allow_html=True)

# ── Main Title ─────────────────────────────────────────────────────────────
st.markdown("# 🏠 Nairobi Housing Price Prediction")
st.markdown("Exploring and predicting residential property prices using machine learning.")
st.markdown("---")

if df is None:
    st.error("Dataset not found. Place `nairobi_house_prices.csv` in the same folder as `app.py` and run the Jupyter notebook first to generate model assets.")
    st.stop()

# ── Tabs ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA", "📈 Model Comparison", "🔍 Feature Importance", "🏷️ Price Predictor"])

# ────────────────────────────────────────────────────────────────────────
# TAB 1: EDA
# ────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("## Exploratory Data Analysis")

    # Top KPIs
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Total Properties</div>
            <div class='metric-value'>{len(df):,}</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-card' style='border-color:#4ECDC4'>
            <div class='metric-label'>Median Price (KES)</div>
            <div class='metric-value'>{df[price_col].median():,.0f}</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='metric-card' style='border-color:#FFEAA7'>
            <div class='metric-label'>Min Price (KES)</div>
            <div class='metric-value'>{df[price_col].min():,.0f}</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class='metric-card' style='border-color:#96CEB4'>
            <div class='metric-label'>Max Price (KES)</div>
            <div class='metric-value'>{df[price_col].max():,.0f}</div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Price Distribution")
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(np.log1p(df[price_col]), bins=35, color='#FF6B6B', edgecolor='#FF6B6B', alpha=0.8)
        ax.set_xlabel('Log(Price)')
        ax.set_ylabel('Frequency')
        ax.set_title('Log-Transformed Price Distribution', fontsize=10, color='#eee')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        location_col = next(
            (c for c in df.columns if c in ['location', 'area', 'neighbourhood', 'estate', 'suburb']),
            df.select_dtypes(include='object').columns[0] if len(df.select_dtypes(include='object').columns) > 0 else None
        )
        if location_col:
            st.markdown("### Top 10 Locations by Median Price")
            top_loc = df.groupby(location_col)[price_col].median().sort_values(ascending=True).tail(10)
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.barh(top_loc.index, top_loc.values, color=COLORS[:len(top_loc)])
            ax.set_xlabel('Median Price (KES)')
            ax.set_title('Top Locations', fontsize=10, color='#eee')
            ax.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

    st.markdown("### Correlation Heatmap")
    numeric_df = df.select_dtypes(include=np.number)
    if len(numeric_df.columns) > 1:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        sns.heatmap(
            numeric_df.corr(), annot=True, fmt='.2f',
            cmap='RdYlGn', linewidths=0.5, ax=ax,
            annot_kws={'size': 8}, cbar_kws={'shrink': 0.8}
        )
        ax.set_title('Feature Correlations', fontsize=10, color='#eee')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Property type distribution
    type_col = next((c for c in df.columns if 'type' in c or 'property' in c), None)
    if type_col and df[type_col].dtype == 'object':
        st.markdown("### Property Type Distribution")
        type_counts = df[type_col].value_counts().head(8)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(type_counts.index, type_counts.values, color=COLORS[:len(type_counts)])
        ax.set_ylabel('Count')
        ax.set_title('Property Types', fontsize=10, color='#eee')
        plt.xticks(rotation=30, ha='right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()


# ────────────────────────────────────────────────────────────────────────
# TAB 2: Model Comparison
# ────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("## Model Performance Comparison")

    if metrics:
        lr_m = metrics.get('linear_regression', {})
        rf_m = metrics.get('random_forest', {})

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Linear Regression")
            for k, color in [('RMSE', '#FF6B6B'), ('MAE', '#FFEAA7'), ('R2', '#96CEB4')]:
                val = lr_m.get(k, 'N/A')
                st.markdown(f"""<div class='metric-card' style='border-color:{color}'>
                    <div class='metric-label'>{k}</div>
                    <div class='metric-value' style='color:{color}'>{val}</div></div>""",
                    unsafe_allow_html=True)

        with c2:
            st.markdown("#### Random Forest")
            for k, color in [('RMSE', '#4ECDC4'), ('MAE', '#45B7D1'), ('R2', '#DDA0DD')]:
                val = rf_m.get(k, 'N/A')
                st.markdown(f"""<div class='metric-card' style='border-color:{color}'>
                    <div class='metric-label'>{k}</div>
                    <div class='metric-value' style='color:{color}'>{val}</div></div>""",
                    unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Side-by-Side Metrics")
        metrics_names = ['RMSE', 'MAE', 'R2']
        lr_vals = [lr_m.get(m, 0) for m in metrics_names]
        rf_vals = [rf_m.get(m, 0) for m in metrics_names]

        x = np.arange(len(metrics_names))
        width = 0.35
        fig, ax = plt.subplots(figsize=(6, 3.5))
        bars1 = ax.bar(x - width/2, lr_vals, width, label='Linear Regression', color='#FF6B6B', alpha=0.85)
        bars2 = ax.bar(x + width/2, rf_vals, width, label='Random Forest', color='#4ECDC4', alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names)
        ax.set_title('Model Metrics Comparison', fontsize=11, color='#eee')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7, color='#FF6B6B')
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7, color='#4ECDC4')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        winner = "Random Forest" if rf_m.get('R2', 0) > lr_m.get('R2', 0) else "Linear Regression"
        color = "#4ECDC4" if winner == "Random Forest" else "#FF6B6B"
        st.markdown(f"""
        <div style='background:#1e1e2e;border-radius:12px;padding:1rem;
                    border:1px solid {color};margin-top:1rem;text-align:center;'>
            <span style='color:#aaa;font-size:0.85rem;'>Best Performing Model</span><br>
            <span style='color:{color};font-size:1.4rem;font-weight:bold;'>{winner}</span>
        </div>""", unsafe_allow_html=True)
    else:
        st.info("Run the Jupyter notebook first to generate model metrics.")


# ────────────────────────────────────────────────────────────────────────
# TAB 3: Feature Importance
# ────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("## Feature Importance (Random Forest)")

    if rf_model and feature_cols:
        importances = pd.Series(rf_model.feature_importances_, index=feature_cols)
        importances_sorted = importances.sort_values(ascending=True).tail(10)

        col_colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(importances_sorted)))

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.barh(importances_sorted.index, importances_sorted.values, color=col_colors)
        ax.set_xlabel('Importance Score')
        ax.set_title('Top 10 Most Important Features', fontsize=11, color='#eee')
        ax.grid(True, alpha=0.3, axis='x')
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', va='center', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown("---")
        st.markdown("### Importance Scores Table")
        imp_df = importances.sort_values(ascending=False).reset_index()
        imp_df.columns = ['Feature', 'Importance Score']
        imp_df['Importance Score'] = imp_df['Importance Score'].round(4)
        imp_df['Rank'] = range(1, len(imp_df) + 1)
        imp_df = imp_df[['Rank', 'Feature', 'Importance Score']]
        st.dataframe(imp_df, use_container_width=True, hide_index=True)
    else:
        st.info("Run the Jupyter notebook first to generate model assets.")


# ────────────────────────────────────────────────────────────────────────
# TAB 4: Price Predictor
# ────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("## Price Predictor")
    st.markdown("Enter property details below to get an estimated price.")

    if rf_model and lr_model and scaler and encoders and feature_cols:

        col1, col2 = st.columns([1, 1])
        input_data = {}

        with col1:
            st.markdown("#### Property Details")
            for feat in feature_cols:
                if feat in df.columns:
                    col_data = df[feat]
                    if feat in encoders:
                        options = list(encoders[feat].classes_)
                        selected = st.selectbox(feat.replace('_', ' ').title(), options, key=feat)
                        input_data[feat] = encoders[feat].transform([selected])[0]
                    elif col_data.dtype in ['float64', 'int64']:
                        min_v = float(col_data.min())
                        max_v = float(col_data.max())
                        med_v = float(col_data.median())
                        val = st.slider(
                            feat.replace('_', ' ').title(),
                            min_value=min_v, max_value=max_v,
                            value=med_v, key=feat
                        )
                        input_data[feat] = val

        with col2:
            st.markdown("#### Prediction")
            model_choice = st.radio(
                "Select Model",
                ["Random Forest", "Linear Regression"],
                horizontal=True
            )

            if st.button("Predict Price", use_container_width=True, type="primary"):
                try:
                    input_array = [input_data.get(f, 0) for f in feature_cols]
                    input_scaled = scaler.transform([input_array])[0]

                    if model_choice == "Random Forest":
                        pred_log = rf_model.predict([input_scaled])[0]
                        model_r2 = metrics.get('random_forest', {}).get('R2', 'N/A')
                    else:
                        pred_log = lr_model.predict([input_scaled])[0]
                        model_r2 = metrics.get('linear_regression', {}).get('R2', 'N/A')

                    pred_price = np.expm1(pred_log)

                    st.markdown(f"""
                    <div class='predict-box'>
                        <div style='color:#aaa;font-size:0.8rem;margin-bottom:0.5rem;'>
                            Estimated Property Price
                        </div>
                        <div class='price-result'>KES {pred_price:,.0f}</div>
                        <div style='color:#aaa;font-size:0.75rem;margin-top:0.5rem;'>
                            Model: {model_choice} &nbsp;|&nbsp; R² = {model_r2}
                        </div>
                    </div>""", unsafe_allow_html=True)

                    # Context bar
                    median_p = df[price_col].median()
                    if pred_price < median_p * 0.7:
                        segment, seg_color = "Below Market Average", "#FF6B6B"
                    elif pred_price > median_p * 1.3:
                        segment, seg_color = "Above Market Average", "#4ECDC4"
                    else:
                        segment, seg_color = "Near Market Average", "#FFEAA7"

                    st.markdown(f"""
                    <div style='text-align:center;margin-top:0.8rem;'>
                        <span style='background:{seg_color}22;color:{seg_color};
                               border:1px solid {seg_color};border-radius:20px;
                               padding:0.3rem 1rem;font-size:0.85rem;'>
                            {segment}
                        </span>
                    </div>""", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Prediction error: {e}")

            st.markdown("---")
            st.markdown("#### Dataset Price Range")
            fig, ax = plt.subplots(figsize=(5, 2.5))
            ax.hist(df[price_col], bins=30, color='#45B7D1', alpha=0.7, edgecolor='#45B7D1')
            ax.set_xlabel('Price (KES)')
            ax.set_ylabel('Count')
            ax.set_title('Price Distribution', fontsize=9, color='#eee')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

    else:
        st.info("Run the Jupyter notebook first to generate model assets.")

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#555;font-size:0.75rem;'>"
    "Nairobi Housing Price Prediction | Machine Learning Case Study</div>",
    unsafe_allow_html=True)