"""
Loyalty Membership Predictions
================================
A comprehensive Streamlit dashboard for predicting customer loyalty membership
using trained ML models (Logistic Regression & Random Forest).

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sqlite3
import base64
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve,
    precision_recall_curve, average_precision_score,
)
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "customer_loyalty_model_bundle_v2.pkl")
DATA_PATH = os.path.join(BASE_DIR, "ecommerce_customer_features.csv")
DB_PATH = os.path.join(BASE_DIR, "prediction_history.db")
LOGO_PATH = os.path.join(BASE_DIR, "loyalty_logo.png")

FEATURE_NAMES = [
    "account_age_months", "avg_order_value", "total_orders",
    "days_since_last_purchase", "discount_usage_rate", "return_rate",
    "customer_support_tickets", "browsing_frequency_per_week",
    "cart_abandonment_rate", "product_review_score_avg",
    "engagement_score", "satisfaction_score", "price_sensitivity_index",
]

FEATURE_LABELS = {
    "account_age_months": "Account Age (months)",
    "avg_order_value": "Average Order Value ($)",
    "total_orders": "Total Orders",
    "days_since_last_purchase": "Days Since Last Purchase",
    "discount_usage_rate": "Discount Usage Rate (0–1)",
    "return_rate": "Return Rate (0–1)",
    "customer_support_tickets": "Customer Support Tickets",
    "browsing_frequency_per_week": "Browsing Frequency / Week",
    "cart_abandonment_rate": "Cart Abandonment Rate (0–1)",
    "product_review_score_avg": "Avg Product Review Score (1–5)",
    "engagement_score": "Engagement Score",
    "satisfaction_score": "Satisfaction Score",
    "price_sensitivity_index": "Price Sensitivity Index",
}

BRAND_PRIMARY = "#1B2A4A"
BRAND_ACCENT = "#00B4D8"
BRAND_SUCCESS = "#06D6A0"
BRAND_DANGER = "#EF476F"
BRAND_WARNING = "#FFD166"
BRAND_BG = "#0E1117"

# ---------------------------------------------------------------------------
# Custom CSS — formal website styling
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.block-container {
    padding-top: 0.8rem;
    padding-bottom: 2rem;
}

/* ── Header bar ── */
.site-header {
    background: linear-gradient(135deg, #1B2A4A 0%, #0D1B2A 60%, #1B3A5C 100%);
    padding: 0.5rem 1rem;
    border-radius: 8px;
    margin: 0 0 0.8rem 0;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    border: 1px solid rgba(255,255,255,0.07);
    box-sizing: border-box;
    width: 100%;
}

.site-header img {
    height: 36px;
    width: 36px;
    border-radius: 6px;
    flex-shrink: 0;
    object-fit: contain;
}

.site-header h1 {
    color: #FFFFFF;
    font-size: 1.05rem;
    font-weight: 700;
    margin: 0;
    line-height: 1.4;   /* prevents top clipping */
    letter-spacing: -0.2px;
}

.site-header .subtitle {
    color: #90CAF9;
    font-size: 0.72rem;
    font-weight: 400;
    margin: 0;
    opacity: 0.85;
}
/* ── Metric cards ── */
.metric-card {
    background: linear-gradient(135deg, #1a1f2e 0%, #161b22 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0,180,216,0.15);
}
.metric-card .metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #00B4D8;
    margin: 0.3rem 0;
}
.metric-card .metric-label {
    font-size: 0.8rem;
    font-weight: 500;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

/* ── Section headers ── */
.section-header {
    font-size: 1.25rem;
    font-weight: 600;
    color: #e6edf3;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #00B4D8;
    margin: 1.5rem 0 1rem 0;
}

/* ── Pipeline step cards ── */
.pipeline-step {
    background: #161b22;
    border-left: 4px solid #00B4D8;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
}
.pipeline-step .step-num {
    color: #00B4D8;
    font-weight: 700;
    font-size: 0.85rem;
}
.pipeline-step .step-title {
    color: #e6edf3;
    font-weight: 600;
    font-size: 0.95rem;
    margin-bottom: 0.2rem;
}
.pipeline-step .step-desc {
    color: #8b949e;
    font-size: 0.82rem;
}

/* ── Score badge ── */
.score-badge {
    display: inline-block;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.85rem;
}
.score-good { background: rgba(6,214,160,0.15); color: #06D6A0; }
.score-ok { background: rgba(255,209,102,0.15); color: #FFD166; }
.score-bad { background: rgba(239,71,111,0.15); color: #EF476F; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1B2A 0%, #1B2A4A 100%);
    width: 175px !important;
    min-width: 175px !important;
    max-width: 175px !important;
}
section[data-testid="stSidebar"] > div:first-child {
    padding: 0.2rem 0.35rem;
    overflow-y: hidden !important;
}

/* Sidebar nav-group labels */
.nav-group-label {
    font-size: 0.58rem;
    font-weight: 700;
    color: #4a5568;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 0.35rem 0 0.1rem 0.5rem;
    margin: 0;
}

/* Sidebar nav buttons */
section[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    text-align: left;
    background: transparent;
    color: #9ca3af;
    border: none;
    border-radius: 6px;
    padding: 0.2rem 0.35rem;
    font-size: 0.7rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    font-weight: 500;
    cursor: pointer;
    transition: background 0.12s, color 0.12s;
    margin: 0;
    line-height: 1.3;
    min-height: 0;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(0,180,216,0.08);
    color: #e6edf3;
}
section[data-testid="stSidebar"] .stButton > button:focus {
    box-shadow: none;
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: rgba(0,180,216,0.12);
    color: #00B4D8;
    border-left: 3px solid #00B4D8;
    font-weight: 600;
}

/* ── Prediction result cards ── */
.prediction-card-loyal {
    background: linear-gradient(135deg, rgba(6,214,160,0.1) 0%, rgba(6,214,160,0.03) 100%);
    border: 2px solid #06D6A0;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.prediction-card-not-loyal {
    background: linear-gradient(135deg, rgba(239,71,111,0.1) 0%, rgba(239,71,111,0.03) 100%);
    border: 2px solid #EF476F;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}

/* ── Footer ── */
.site-footer {
    text-align: center;
    color: #484f58;
    font-size: 0.75rem;
    padding: 2rem 0 0.5rem 0;
    border-top: 1px solid #21262d;
    margin-top: 3rem;
}
</style>
"""

# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

def init_db():
    """Create the prediction_history table if it does not exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            model_name TEXT NOT NULL,
            prediction_type TEXT NOT NULL,
            prediction TEXT NOT NULL,
            loyalty_probability REAL NOT NULL,
            input_summary TEXT,
            batch_total INTEGER,
            batch_loyal INTEGER,
            batch_not_loyal INTEGER
        )
        """
    )
    conn.commit()
    conn.close()


def save_prediction(model_name: str, prediction_type: str, prediction: str,
                    loyalty_prob: float, input_summary: str = None,
                    batch_total: int = None, batch_loyal: int = None,
                    batch_not_loyal: int = None):
    """Insert a prediction record into SQLite."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        INSERT INTO prediction_history
            (timestamp, model_name, prediction_type, prediction,
             loyalty_probability, input_summary, batch_total,
             batch_loyal, batch_not_loyal)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model_name, prediction_type, prediction,
            round(loyalty_prob, 2), input_summary,
            batch_total, batch_loyal, batch_not_loyal,
        ),
    )
    conn.commit()
    conn.close()


def get_history(limit: int = 200) -> pd.DataFrame:
    """Retrieve prediction history from SQLite."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM prediction_history ORDER BY id DESC LIMIT ?",
        conn,
        params=(limit,),
    )
    conn.close()
    return df


def clear_history():
    """Delete all rows from prediction_history."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM prediction_history")
    conn.commit()
    conn.close()


def get_logo_base64():
    """Return the logo as a base64-encoded data URI for embedding in HTML."""
    if not os.path.exists(LOGO_PATH):
        return None
    with open(LOGO_PATH, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{data}"


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model(path: str = MODEL_PATH):
    """Load the model bundle (.pkl) and return its components."""
    if not os.path.exists(path):
        st.error(f"Model file not found at: {path}")
        st.stop()
    bundle = joblib.load(path)
    from sklearn.utils.validation import check_is_fitted

    normalised = {
        "rf_model": bundle.get("random_forest_model", bundle.get("model")),
        "lr_model": None,
        "preprocessor": bundle["preprocessor"],
        "feature_names": bundle.get("feature_columns", bundle.get("feature_names")),
    }

    lr_candidate = bundle.get("logistic_regression_model")
    if lr_candidate is not None:
        try:
            check_is_fitted(lr_candidate)
            normalised["lr_model"] = lr_candidate
        except Exception:
            pass

    return normalised


@st.cache_data
def load_dataset(path: str = DATA_PATH):
    """Load the raw CSV dataset for EDA and metric computation."""
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@st.cache_data
def compute_model_metrics(_bundle, _df):
    """
    Re-run the same preprocessing + train/test split from the notebook
    so we can compute live metrics, confusion matrices, and ROC curves.
    Returns a dict with detailed results for each model.
    """
    df = _df.copy()
    if "Customer_ID" in df.columns:
        df.drop(columns=["Customer_ID"], inplace=True)
    df["loyalty_member"] = df["loyalty_member"].map({"No": 0, "Yes": 1})
    if df["loyalty_member"].isna().any():
        df["loyalty_member"] = df["loyalty_member"].astype(int)

    X = df.drop(columns=["loyalty_member"])
    y = df["loyalty_member"]

    feature_names = _bundle["feature_names"]
    X = X[feature_names]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = _bundle["preprocessor"]
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_processed, y_train)

    results = {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "X_train_processed": X_train_processed, "X_test_processed": X_test_processed,
        "y_train_balanced": y_train_bal,
        "train_size": len(X_train), "test_size": len(X_test),
        "smote_before": int(y_train.sum()), "smote_after": int(y_train_bal.sum()),
        "smote_total_before": len(y_train), "smote_total_after": len(y_train_bal),
        "models": {},
    }

    models_to_eval = {}
    if _bundle.get("lr_model") is not None:
        models_to_eval["Logistic Regression"] = _bundle["lr_model"]
    if _bundle.get("rf_model") is not None:
        models_to_eval["Random Forest"] = _bundle["rf_model"]

    for name, mdl in models_to_eval.items():
        y_pred = mdl.predict(X_test_processed)
        y_prob = mdl.predict_proba(X_test_processed)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        results["models"][name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob),
            "avg_precision": average_precision_score(y_test, y_prob),
            "fpr": fpr, "tpr": tpr,
            "prec_curve": prec_curve, "rec_curve": rec_curve,
            "confusion_matrix": cm,
            "report": report,
            "y_pred": y_pred, "y_prob": y_prob,
        }

    return results


def preprocess_data(df: pd.DataFrame, preprocessor, feature_names: list) -> np.ndarray:
    df = df.copy()
    for col in ["Customer_ID", "loyalty_member"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df[feature_names]
    for col in df.columns:
        if hasattr(df[col].dtype, 'name') and df[col].dtype.name in ('string', 'String'):
            df[col] = df[col].astype(object)
    return preprocessor.transform(df)


def predict_loyalty(model, preprocessor, feature_names: list, df: pd.DataFrame):
    processed = preprocess_data(df, preprocessor, feature_names)
    predictions = model.predict(processed)
    probabilities = model.predict_proba(processed)
    results = df.copy()
    for col in ["Customer_ID", "loyalty_member"]:
        if col in results.columns:
            results.drop(columns=[col], inplace=True)
    results["Prediction"] = np.where(predictions == 1, "LOYAL CUSTOMER", "NOT LOYAL CUSTOMER")
    results["Loyalty Probability (%)"] = np.round(probabilities[:, 1] * 100, 2)
    results["Not-Loyal Probability (%)"] = np.round(probabilities[:, 0] * 100, 2)
    return results


def score_badge(val, thresholds=(0.7, 0.85)):
    """Return an HTML badge coloured by threshold."""
    pct = f"{val*100:.1f}%"
    if val >= thresholds[1]:
        return f'<span class="score-badge score-good">{pct}</span>'
    elif val >= thresholds[0]:
        return f'<span class="score-badge score-ok">{pct}</span>'
    return f'<span class="score-badge score-bad">{pct}</span>'


def render_header():
    logo_b64 = get_logo_base64()
    logo_html = f'<img src="{logo_b64}">' if logo_b64 else ''
    st.markdown(
        f'<div class="site-header">'
        f'{logo_html}'
        f'<div>'
        f'<h1>Loyalty Membership Predictions</h1>'
        f'<p class="subtitle">E-Commerce Customer Loyalty Intelligence Dashboard</p>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_footer():
    st.markdown(
        '<div class="site-footer">Loyalty Membership Predictions &copy; 2026 &mdash; '
        'Built with Streamlit &bull; Powered by Scikit-Learn</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page: Dashboard Overview
# ---------------------------------------------------------------------------

def page_dashboard(bundle, raw_df, metrics):
    render_header()
    st.markdown('<div class="section-header">Dashboard Overview</div>', unsafe_allow_html=True)

    if raw_df is None:
        st.warning("Dataset file not found — some dashboard widgets are unavailable.")
        return

    # --- Top KPI row ---
    loyal_count = int((raw_df["loyalty_member"] == "Yes").sum()) if "loyalty_member" in raw_df.columns else 0
    total_count = len(raw_df)
    not_loyal_count = total_count - loyal_count
    loyalty_rate = loyal_count / total_count * 100 if total_count else 0

    k1, k2, k3, k4 = st.columns(4)
    for col, label, value, color in [
        (k1, "Total Customers", f"{total_count:,}", BRAND_ACCENT),
        (k2, "Loyal Members", f"{loyal_count:,}", BRAND_SUCCESS),
        (k3, "Non-Loyal", f"{not_loyal_count:,}", BRAND_DANGER),
        (k4, "Loyalty Rate", f"{loyalty_rate:.1f}%", BRAND_WARNING),
    ]:
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value" style="color:{color}">{value}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    # --- Class distribution + feature averages ---
    col_a, col_b = st.columns([1, 2])

    with col_a:
        st.markdown('<div class="section-header">Class Distribution</div>', unsafe_allow_html=True)
        counts = raw_df["loyalty_member"].value_counts()
        fig_dist = px.pie(
            names=counts.index.map({"Yes": "Loyal", "No": "Not Loyal"}),
            values=counts.values,
            color_discrete_sequence=[BRAND_SUCCESS, BRAND_DANGER],
            hole=0.55,
        )
        fig_dist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#c9d1d9", showlegend=True, margin=dict(t=10, b=10, l=10, r=10),
            height=300,
        )
        fig_dist.update_traces(textinfo="percent+label")
        st.plotly_chart(fig_dist, use_container_width=True, key="dash_dist")

    with col_b:
        st.markdown('<div class="section-header">Feature Averages by Loyalty Status</div>', unsafe_allow_html=True)
        numeric_df = raw_df.drop(columns=["Customer_ID"], errors="ignore")
        if "loyalty_member" in numeric_df.columns:
            avg_by_loyalty = numeric_df.groupby("loyalty_member")[FEATURE_NAMES].mean().T
            avg_by_loyalty.columns = ["Not Loyal", "Loyal"]
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(name="Loyal", x=avg_by_loyalty.index, y=avg_by_loyalty["Loyal"], marker_color=BRAND_SUCCESS))
            fig_bar.add_trace(go.Bar(name="Not Loyal", x=avg_by_loyalty.index, y=avg_by_loyalty["Not Loyal"], marker_color=BRAND_DANGER))
            fig_bar.update_layout(
                barmode="group", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#c9d1d9", height=300, margin=dict(t=10, b=10),
                xaxis_tickangle=-45, legend=dict(orientation="h", y=1.1),
            )
            st.plotly_chart(fig_bar, use_container_width=True, key="dash_avg")

    # --- Quick model score summary ---
    if metrics and metrics["models"]:
        st.markdown('<div class="section-header">Model Performance Summary</div>', unsafe_allow_html=True)
        cols = st.columns(len(metrics["models"]))
        for i, (name, m) in enumerate(metrics["models"].items()):
            with cols[i]:
                st.markdown(f"**{name}**")
                st.markdown(
                    f"Accuracy: {score_badge(m['accuracy'])} &nbsp; "
                    f"F1: {score_badge(m['f1'])} &nbsp; "
                    f"ROC AUC: {score_badge(m['roc_auc'])}",
                    unsafe_allow_html=True,
                )

    render_footer()


# ---------------------------------------------------------------------------
# Page: Data Exploration
# ---------------------------------------------------------------------------

def page_data_exploration(raw_df):
    render_header()
    st.markdown('<div class="section-header">Data Exploration</div>', unsafe_allow_html=True)

    if raw_df is None:
        st.warning("Dataset file not found.")
        return

    tab_preview, tab_stats, tab_dist, tab_corr, tab_box = st.tabs([
        "Preview", "Statistics", "Distributions", "Correlation Matrix", "Outlier Detection"
    ])

    with tab_preview:
        st.dataframe(raw_df.head(50), use_container_width=True, height=400)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{len(raw_df):,}")
        c2.metric("Columns", f"{len(raw_df.columns)}")
        c3.metric("Missing Values", f"{int(raw_df.isnull().sum().sum()):,}")
        c4.metric("Duplicates", f"{int(raw_df.duplicated().sum()):,}")

    with tab_stats:
        st.markdown("**Descriptive Statistics**")
        st.dataframe(raw_df.describe().T.style.format("{:.3f}"), use_container_width=True)
        st.markdown("**Data Types**")
        dtype_df = pd.DataFrame({"Column": raw_df.columns, "Type": raw_df.dtypes.astype(str).values, "Non-Null": raw_df.notnull().sum().values})
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)

    with tab_dist:
        feat = st.selectbox("Select feature", FEATURE_NAMES, key="dist_feat")
        fig_hist = px.histogram(
            raw_df, x=feat, color="loyalty_member", nbins=40, barmode="overlay",
            color_discrete_map={"Yes": BRAND_SUCCESS, "No": BRAND_DANGER},
            labels={"loyalty_member": "Loyalty Member"},
        )
        fig_hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#c9d1d9", height=400,
        )
        st.plotly_chart(fig_hist, use_container_width=True, key="explore_hist")

    with tab_corr:
        numeric_cols = raw_df[FEATURE_NAMES]
        corr = numeric_cols.corr()
        fig_corr = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1, aspect="auto",
        )
        fig_corr.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#c9d1d9", height=600, margin=dict(t=30),
        )
        st.plotly_chart(fig_corr, use_container_width=True, key="explore_corr")

    with tab_box:
        feat_box = st.selectbox("Select feature", FEATURE_NAMES, key="box_feat")
        fig_box = px.box(
            raw_df, x="loyalty_member", y=feat_box, color="loyalty_member",
            color_discrete_map={"Yes": BRAND_SUCCESS, "No": BRAND_DANGER},
        )
        fig_box.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#c9d1d9", height=400, showlegend=False,
        )
        st.plotly_chart(fig_box, use_container_width=True, key="explore_box")

    render_footer()


# ---------------------------------------------------------------------------
# Page: Preprocessing Pipeline
# ---------------------------------------------------------------------------

def page_preprocessing(bundle, metrics):
    render_header()
    st.markdown('<div class="section-header">Preprocessing Pipeline</div>', unsafe_allow_html=True)

    steps = [
        ("01", "Load Data", "Read ecommerce_customer_features.csv with 15 columns including Customer_ID and loyalty_member."),
        ("02", "Drop Non-Feature Columns", "Remove Customer_ID (identifier) — not predictive."),
        ("03", "Encode Target", "Map loyalty_member: Yes → 1, No → 0."),
        ("04", "Train / Test Split", f"80/20 stratified split — Train: {metrics['train_size']:,} | Test: {metrics['test_size']:,} samples."),
        ("05", "Imputation", "Median imputation for missing numerical values using SimpleImputer(strategy='median')."),
        ("06", "Feature Scaling", "StandardScaler — zero mean, unit variance on all 13 numerical features."),
        ("07", "SMOTE Oversampling", "Synthetic Minority Oversampling on training data to balance class distribution."),
    ]

    for num, title, desc in steps:
        st.markdown(
            f'<div class="pipeline-step">'
            f'<div class="step-num">STEP {num}</div>'
            f'<div class="step-title">{title}</div>'
            f'<div class="step-desc">{desc}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # SMOTE visualisation
    st.markdown('<div class="section-header">SMOTE Class Balancing</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    before_loyal = metrics["smote_before"]
    before_not = metrics["smote_total_before"] - before_loyal
    after_loyal = metrics["smote_after"]
    after_not = metrics["smote_total_after"] - after_loyal

    with col1:
        fig_before = px.pie(
            names=["Loyal (1)", "Not Loyal (0)"], values=[before_loyal, before_not],
            color_discrete_sequence=[BRAND_SUCCESS, BRAND_DANGER], hole=0.5,
            title="Before SMOTE",
        )
        fig_before.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#c9d1d9", height=300, margin=dict(t=40, b=10))
        st.plotly_chart(fig_before, use_container_width=True, key="smote_before")
        st.caption(f"Total: {metrics['smote_total_before']:,} — Loyal: {before_loyal:,} — Not Loyal: {before_not:,}")

    with col2:
        fig_after = px.pie(
            names=["Loyal (1)", "Not Loyal (0)"], values=[after_loyal, after_not],
            color_discrete_sequence=[BRAND_SUCCESS, BRAND_DANGER], hole=0.5,
            title="After SMOTE",
        )
        fig_after.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#c9d1d9", height=300, margin=dict(t=40, b=10))
        st.plotly_chart(fig_after, use_container_width=True, key="smote_after")
        st.caption(f"Total: {metrics['smote_total_after']:,} — Loyal: {after_loyal:,} — Not Loyal: {after_not:,}")

    # Preprocessor internals
    st.markdown('<div class="section-header">Preprocessor Details</div>', unsafe_allow_html=True)
    preprocessor = bundle["preprocessor"]
    with st.expander("ColumnTransformer configuration", expanded=False):
        for name, transformer, cols in preprocessor.transformers_:
            st.markdown(f"**`{name}`** → `{transformer}`")
            st.code(f"Columns: {list(cols)}", language="text")

    render_footer()


# ---------------------------------------------------------------------------
# Page: Model Performance
# ---------------------------------------------------------------------------

def page_model_performance(metrics):
    render_header()
    st.markdown('<div class="section-header">Model Evaluation &amp; Performance</div>', unsafe_allow_html=True)

    if not metrics or not metrics["models"]:
        st.info("No model metrics available.")
        return

    model_names = list(metrics["models"].keys())
    selected = st.selectbox("Select Model", model_names, key="perf_model")
    m = metrics["models"][selected]

    # --- Score cards ---
    s1, s2, s3, s4, s5 = st.columns(5)
    for col, label, val in [
        (s1, "Accuracy", m["accuracy"]),
        (s2, "Precision", m["precision"]),
        (s3, "Recall", m["recall"]),
        (s4, "F1 Score", m["f1"]),
        (s5, "ROC AUC", m["roc_auc"]),
    ]:
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value">{val*100:.1f}%</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    tab_cm, tab_roc, tab_pr, tab_report = st.tabs([
        "Confusion Matrix", "ROC Curve", "Precision-Recall Curve", "Classification Report"
    ])

    # --- Confusion Matrix ---
    with tab_cm:
        cm = m["confusion_matrix"]
        labels = ["Not Loyal (0)", "Loyal (1)"]
        fig_cm = px.imshow(
            cm, x=labels, y=labels, text_auto=True,
            color_continuous_scale=[[0, "#161b22"], [1, BRAND_ACCENT]],
            labels=dict(x="Predicted", y="Actual", color="Count"),
        )
        fig_cm.update_layout(
            title=f"Confusion Matrix — {selected}",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#c9d1d9", height=450,
        )
        st.plotly_chart(fig_cm, use_container_width=True, key="cm_chart")

        tn, fp, fn, tp = cm.ravel()
        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.metric("True Negatives", f"{tn:,}")
        cc2.metric("False Positives", f"{fp:,}")
        cc3.metric("False Negatives", f"{fn:,}")
        cc4.metric("True Positives", f"{tp:,}")

    # --- ROC Curve ---
    with tab_roc:
        fig_roc = go.Figure()
        for name2, m2 in metrics["models"].items():
            fig_roc.add_trace(go.Scatter(
                x=m2["fpr"], y=m2["tpr"], mode="lines",
                name=f"{name2} (AUC={m2['roc_auc']:.3f})",
            ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(dash="dash", color="#484f58"), name="Random Baseline",
        ))
        fig_roc.update_layout(
            title="ROC Curve Comparison",
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#c9d1d9", height=480, legend=dict(x=0.55, y=0.05),
        )
        st.plotly_chart(fig_roc, use_container_width=True, key="roc_chart")

    # --- Precision-Recall Curve ---
    with tab_pr:
        fig_pr = go.Figure()
        for name2, m2 in metrics["models"].items():
            fig_pr.add_trace(go.Scatter(
                x=m2["rec_curve"], y=m2["prec_curve"], mode="lines",
                name=f"{name2} (AP={m2['avg_precision']:.3f})",
            ))
        fig_pr.update_layout(
            title="Precision-Recall Curve Comparison",
            xaxis_title="Recall", yaxis_title="Precision",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#c9d1d9", height=480,
        )
        st.plotly_chart(fig_pr, use_container_width=True, key="pr_chart")

    # --- Classification Report ---
    with tab_report:
        report = m["report"]
        report_df = pd.DataFrame(report).T
        st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)

    render_footer()


# ---------------------------------------------------------------------------
# Page: Model Comparison
# ---------------------------------------------------------------------------

def page_model_comparison(metrics):
    render_header()
    st.markdown('<div class="section-header">Model Comparison</div>', unsafe_allow_html=True)

    if not metrics or len(metrics["models"]) < 2:
        st.info("At least two models are needed for comparison. Only one model is available.")
        return

    # Side-by-side metrics table
    rows = []
    for name, m in metrics["models"].items():
        rows.append({
            "Model": name,
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1 Score": m["f1"],
            "ROC AUC": m["roc_auc"],
            "Avg Precision": m["avg_precision"],
        })
    comp_df = pd.DataFrame(rows)
    st.dataframe(
        comp_df.style.format({
            "Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}",
            "F1 Score": "{:.4f}", "ROC AUC": "{:.4f}", "Avg Precision": "{:.4f}",
        }).highlight_max(axis=0, subset=["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC", "Avg Precision"], color="#0e4429"),
        use_container_width=True, hide_index=True,
    )

    # Radar chart
    metric_keys = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
    fig_radar = go.Figure()
    colors = [BRAND_ACCENT, BRAND_SUCCESS, BRAND_WARNING]
    for i, (name, m) in enumerate(metrics["models"].items()):
        vals = [m["accuracy"], m["precision"], m["recall"], m["f1"], m["roc_auc"]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=metric_keys + [metric_keys[0]],
            fill="toself",
            name=name,
            line_color=colors[i % len(colors)],
            opacity=0.7,
        ))
    fig_radar.update_layout(
        polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(visible=True, range=[0, 1], color="#484f58")),
        paper_bgcolor="rgba(0,0,0,0)", font_color="#c9d1d9", height=480,
        title="Performance Radar",
    )
    st.plotly_chart(fig_radar, use_container_width=True, key="radar_chart")

    # Confusion matrices side-by-side
    st.markdown('<div class="section-header">Confusion Matrices</div>', unsafe_allow_html=True)
    cm_cols = st.columns(len(metrics["models"]))
    labels = ["Not Loyal", "Loyal"]
    for i, (name, m) in enumerate(metrics["models"].items()):
        with cm_cols[i]:
            cm = m["confusion_matrix"]
            fig_cm = px.imshow(
                cm, x=labels, y=labels, text_auto=True,
                color_continuous_scale=[[0, "#161b22"], [1, colors[i % len(colors)]]],
                labels=dict(x="Predicted", y="Actual"),
            )
            fig_cm.update_layout(
                title=name, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#c9d1d9", height=350, margin=dict(t=40),
            )
            st.plotly_chart(fig_cm, use_container_width=True, key=f"cmp_cm_{i}")

    render_footer()


# ---------------------------------------------------------------------------
# Page: Feature Importance
# ---------------------------------------------------------------------------

def page_feature_importance(bundle, metrics):
    render_header()
    st.markdown('<div class="section-header">Feature Importance Analysis</div>', unsafe_allow_html=True)

    rf_model = bundle.get("rf_model")
    lr_model = bundle.get("lr_model")
    feature_names = bundle["feature_names"]

    if rf_model is not None and hasattr(rf_model, "feature_importances_"):
        st.markdown("#### Random Forest — Gini Importance")
        importances = rf_model.feature_importances_
        imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values("Importance", ascending=True)
        fig_imp = px.bar(
            imp_df, x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale=[[0, BRAND_DANGER], [0.5, BRAND_WARNING], [1, BRAND_SUCCESS]],
        )
        fig_imp.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#c9d1d9", height=500, margin=dict(t=10),
        )
        st.plotly_chart(fig_imp, use_container_width=True, key="rf_importance")

    if lr_model is not None and hasattr(lr_model, "coef_"):
        st.markdown("#### Logistic Regression — Coefficients")
        coefs = lr_model.coef_[0]
        coef_df = pd.DataFrame({"Feature": feature_names, "Coefficient": coefs}).sort_values("Coefficient", ascending=True)
        fig_coef = px.bar(
            coef_df, x="Coefficient", y="Feature", orientation="h",
            color="Coefficient", color_continuous_scale="RdBu_r", color_continuous_midpoint=0,
        )
        fig_coef.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#c9d1d9", height=500, margin=dict(t=10),
        )
        st.plotly_chart(fig_coef, use_container_width=True, key="lr_coefs")

    render_footer()


# ---------------------------------------------------------------------------
# Page: Single Prediction
# ---------------------------------------------------------------------------

def page_single_prediction(bundle):
    render_header()
    st.markdown('<div class="section-header">Single Customer Prediction</div>', unsafe_allow_html=True)
    st.caption("Enter the customer's details below and click **Predict** to determine loyalty membership status.")

    model_options = {}
    if bundle.get("lr_model") is not None:
        model_options["Logistic Regression"] = bundle["lr_model"]
    if bundle.get("rf_model") is not None:
        model_options["Random Forest"] = bundle["rf_model"]
    chosen_model_name = st.selectbox("Model", list(model_options.keys()), key="single_model")
    model = model_options[chosen_model_name]
    preprocessor = bundle["preprocessor"]
    feature_names = bundle["feature_names"]

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Customer Profile**")
        account_age_months = st.number_input(FEATURE_LABELS["account_age_months"], min_value=0, max_value=120, value=24, step=1)
        total_orders = st.number_input(FEATURE_LABELS["total_orders"], min_value=0, max_value=500, value=10, step=1)
        avg_order_value = st.number_input(FEATURE_LABELS["avg_order_value"], min_value=0.0, max_value=1000.0, value=50.0, step=0.5)
        days_since_last_purchase = st.number_input(FEATURE_LABELS["days_since_last_purchase"], min_value=0, max_value=365, value=15, step=1)
        customer_support_tickets = st.number_input(FEATURE_LABELS["customer_support_tickets"], min_value=0, max_value=50, value=1, step=1)

    with col2:
        st.markdown("**Engagement Metrics**")
        engagement_score = st.number_input(FEATURE_LABELS["engagement_score"], min_value=0.0, max_value=10.0, value=5.0, step=0.1)
        satisfaction_score = st.number_input(FEATURE_LABELS["satisfaction_score"], min_value=0.0, max_value=10.0, value=7.0, step=0.1)
        browsing_frequency_per_week = st.number_input(FEATURE_LABELS["browsing_frequency_per_week"], min_value=0.0, max_value=30.0, value=3.0, step=0.1)
        product_review_score_avg = st.number_input(FEATURE_LABELS["product_review_score_avg"], min_value=1.0, max_value=5.0, value=4.0, step=0.1)

    with col3:
        st.markdown("**Behavioral Rates**")
        discount_usage_rate = st.slider(FEATURE_LABELS["discount_usage_rate"], 0.0, 1.0, 0.3, 0.01)
        return_rate = st.slider(FEATURE_LABELS["return_rate"], 0.0, 1.0, 0.05, 0.01)
        cart_abandonment_rate = st.slider(FEATURE_LABELS["cart_abandonment_rate"], 0.0, 1.0, 0.4, 0.01)
        price_sensitivity_index = st.number_input(FEATURE_LABELS["price_sensitivity_index"], min_value=0.0, max_value=10.0, value=5.0, step=0.1)

    st.markdown("")

    if st.button("Predict Loyalty Status", use_container_width=True, type="primary"):
        input_data = pd.DataFrame([{
            "account_age_months": account_age_months,
            "avg_order_value": avg_order_value,
            "total_orders": total_orders,
            "days_since_last_purchase": days_since_last_purchase,
            "discount_usage_rate": discount_usage_rate,
            "return_rate": return_rate,
            "customer_support_tickets": customer_support_tickets,
            "browsing_frequency_per_week": browsing_frequency_per_week,
            "cart_abandonment_rate": cart_abandonment_rate,
            "product_review_score_avg": product_review_score_avg,
            "engagement_score": engagement_score,
            "satisfaction_score": satisfaction_score,
            "price_sensitivity_index": price_sensitivity_index,
        }])

        try:
            result = predict_loyalty(model, preprocessor, feature_names, input_data)
            prediction = result["Prediction"].iloc[0]
            prob_loyal = result["Loyalty Probability (%)"].iloc[0]
            prob_not_loyal = result["Not-Loyal Probability (%)"].iloc[0]

            st.markdown("---")

            # Result card
            card_class = "prediction-card-loyal" if prediction == "LOYAL CUSTOMER" else "prediction-card-not-loyal"
            icon = "&#10003;" if prediction == "LOYAL CUSTOMER" else "&#10007;"
            color = BRAND_SUCCESS if prediction == "LOYAL CUSTOMER" else BRAND_DANGER
            st.markdown(
                f'<div class="{card_class}">'
                f'<h2 style="color:{color}; margin:0;">{icon} {prediction}</h2>'
                f'<p style="color:#8b949e; margin-top:0.5rem;">Predicted using {chosen_model_name}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Save to history
            summary = "; ".join(f"{k}={v}" for k, v in input_data.iloc[0].items())
            save_prediction(
                model_name=chosen_model_name,
                prediction_type="Single",
                prediction=prediction,
                loyalty_prob=prob_loyal,
                input_summary=summary,
            )

            st.markdown("")
            pc1, pc2 = st.columns(2)
            pc1.metric("Loyalty Probability", f"{prob_loyal}%", delta=f"{prob_loyal - 50:.1f}% vs 50%")
            pc2.metric("Not-Loyal Probability", f"{prob_not_loyal}%")

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_loyal,
                title={"text": "Loyalty Confidence", "font": {"color": "#c9d1d9"}},
                number={"suffix": "%", "font": {"color": "#e6edf3"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#484f58"},
                    "bar": {"color": BRAND_SUCCESS if prob_loyal >= 50 else BRAND_DANGER},
                    "bgcolor": "#161b22",
                    "steps": [
                        {"range": [0, 30], "color": "rgba(239,71,111,0.2)"},
                        {"range": [30, 70], "color": "rgba(255,209,102,0.2)"},
                        {"range": [70, 100], "color": "rgba(6,214,160,0.2)"},
                    ],
                    "threshold": {"line": {"color": "#ffffff", "width": 2}, "thickness": 0.75, "value": 50},
                },
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", font_color="#c9d1d9", height=300,
                margin=dict(t=60, b=10),
            )
            st.plotly_chart(fig_gauge, use_container_width=True, key="single_gauge")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    render_footer()


# ---------------------------------------------------------------------------
# Page: Batch Prediction
# ---------------------------------------------------------------------------

def page_batch_prediction(bundle):
    render_header()
    st.markdown('<div class="section-header">Batch CSV Prediction</div>', unsafe_allow_html=True)
    st.caption("Upload a CSV file containing customer data to generate loyalty predictions in bulk.")

    model_options = {}
    if bundle.get("lr_model") is not None:
        model_options["Logistic Regression"] = bundle["lr_model"]
    if bundle.get("rf_model") is not None:
        model_options["Random Forest"] = bundle["rf_model"]
    chosen_model_name = st.selectbox("Model", list(model_options.keys()), key="batch_model")
    model = model_options[chosen_model_name]
    preprocessor = bundle["preprocessor"]
    feature_names = bundle["feature_names"]

    with st.expander("Required CSV columns"):
        st.code(", ".join(feature_names), language="text")
        st.info("Columns `Customer_ID` and `loyalty_member` are optional and will be ignored.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="batch_upload")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            return

        st.markdown("**Data Preview**")
        st.dataframe(df.head(10), use_container_width=True)
        st.caption(f"{len(df):,} rows &times; {len(df.columns)} columns")

        missing_cols = [c for c in feature_names if c not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            return

        if st.button("Run Batch Prediction", use_container_width=True, type="primary"):
            try:
                with st.spinner("Running predictions..."):
                    results = predict_loyalty(model, preprocessor, feature_names, df)

                st.markdown('<div class="section-header">Prediction Results</div>', unsafe_allow_html=True)
                st.dataframe(results, use_container_width=True, height=400)

                counts = results["Prediction"].value_counts()
                loyal_count = counts.get("LOYAL CUSTOMER", 0)
                not_loyal_count = counts.get("NOT LOYAL CUSTOMER", 0)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total", f"{len(results):,}")
                m2.metric("Loyal", f"{loyal_count:,}")
                m3.metric("Not Loyal", f"{not_loyal_count:,}")
                m4.metric("Loyalty %", f"{loyal_count/len(results)*100:.1f}%")

                col_pie, col_hist = st.columns(2)
                with col_pie:
                    fig_pie = px.pie(
                        names=counts.index, values=counts.values,
                        color_discrete_sequence=[BRAND_SUCCESS, BRAND_DANGER], hole=0.5,
                        title="Prediction Distribution",
                    )
                    fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#c9d1d9", height=350)
                    fig_pie.update_traces(textinfo="percent+label")
                    st.plotly_chart(fig_pie, use_container_width=True, key="batch_pie")

                with col_hist:
                    fig_hist = px.histogram(
                        results, x="Loyalty Probability (%)", color="Prediction", nbins=30,
                        color_discrete_map={"LOYAL CUSTOMER": BRAND_SUCCESS, "NOT LOYAL CUSTOMER": BRAND_DANGER},
                        title="Probability Distribution",
                    )
                    fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#c9d1d9", height=350)
                    st.plotly_chart(fig_hist, use_container_width=True, key="batch_hist")

                # Save batch to history
                save_prediction(
                    model_name=chosen_model_name,
                    prediction_type="Batch",
                    prediction=f"{loyal_count} loyal / {not_loyal_count} not loyal",
                    loyalty_prob=round(loyal_count / len(results) * 100, 2),
                    input_summary=f"CSV with {len(df)} rows",
                    batch_total=len(results),
                    batch_loyal=int(loyal_count),
                    batch_not_loyal=int(not_loyal_count),
                )

                csv_output = results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Results as CSV",
                    data=csv_output,
                    file_name="loyalty_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    render_footer()


# ---------------------------------------------------------------------------
# Page: About
# ---------------------------------------------------------------------------

def page_about():
    render_header()
    st.markdown('<div class="section-header">About This System</div>', unsafe_allow_html=True)

    st.markdown(
        """
        **Loyalty Membership Predictions** is an end-to-end machine learning dashboard
        that predicts whether an e-commerce customer will become a loyalty member
        based on their transactional and behavioral data.

        ---

        #### Methodology
        | Stage | Details |
        |-------|---------|
        | **Data** | 15 features from e-commerce transaction logs |
        | **Preprocessing** | Median imputation, StandardScaler normalization |
        | **Class Balancing** | SMOTE oversampling on training split |
        | **Models** | Logistic Regression, Random Forest Classifier |
        | **Tuning** | GridSearchCV with cross-validation |
        | **Evaluation** | Accuracy, Precision, Recall, F1, ROC AUC |

        ---

        #### Features Used
        """
    )

    feature_df = pd.DataFrame({
        "Feature": FEATURE_NAMES,
        "Description": [FEATURE_LABELS[f] for f in FEATURE_NAMES],
    })
    st.dataframe(feature_df, use_container_width=True, hide_index=True)

    st.markdown(
        """
        ---

        #### Technology Stack
        - **Frontend:** Streamlit
        - **Visualization:** Plotly
        - **ML Framework:** Scikit-Learn, Imbalanced-Learn
        - **Data Processing:** Pandas, NumPy
        """
    )

    render_footer()


# ---------------------------------------------------------------------------
# Page: Prediction History
# ---------------------------------------------------------------------------

def page_prediction_history():
    render_header()
    st.markdown('<div class="section-header">Prediction History</div>', unsafe_allow_html=True)
    st.caption("All predictions are stored in a local SQLite database for audit and review.")

    history = get_history(500)

    if history.empty:
        st.info("No predictions have been recorded yet. Make a prediction to see it here.")
        render_footer()
        return

    # --- KPI row ---
    total_preds = len(history)
    single_count = int((history["prediction_type"] == "Single").sum())
    batch_count = int((history["prediction_type"] == "Batch").sum())
    avg_prob = history["loyalty_probability"].mean()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Predictions", f"{total_preds:,}")
    k2.metric("Single Predictions", f"{single_count:,}")
    k3.metric("Batch Runs", f"{batch_count:,}")
    k4.metric("Avg Loyalty Prob", f"{avg_prob:.1f}%")

    st.markdown("")

    tab_table, tab_charts = st.tabs(["History Table", "Analytics"])

    with tab_table:
        # Filters
        fc1, fc2 = st.columns(2)
        with fc1:
            type_filter = st.multiselect(
                "Prediction Type", ["Single", "Batch"],
                default=["Single", "Batch"], key="hist_type",
            )
        with fc2:
            model_filter = st.multiselect(
                "Model", history["model_name"].unique().tolist(),
                default=history["model_name"].unique().tolist(), key="hist_model",
            )

        filtered = history[
            history["prediction_type"].isin(type_filter)
            & history["model_name"].isin(model_filter)
        ]
        st.dataframe(filtered, use_container_width=True, height=420, hide_index=True)
        st.caption(f"Showing {len(filtered):,} of {total_preds:,} records")

    with tab_charts:
        col_a, col_b = st.columns(2)

        with col_a:
            # Predictions over time
            if "timestamp" in history.columns:
                hist_time = history.copy()
                hist_time["date"] = pd.to_datetime(hist_time["timestamp"]).dt.date
                daily = hist_time.groupby("date").size().reset_index(name="count")
                fig_time = px.bar(
                    daily, x="date", y="count",
                    title="Predictions Over Time",
                    color_discrete_sequence=[BRAND_ACCENT],
                )
                fig_time.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#c9d1d9", height=350,
                )
                st.plotly_chart(fig_time, use_container_width=True, key="hist_time")

        with col_b:
            # Model usage distribution
            model_counts = history["model_name"].value_counts()
            fig_model = px.pie(
                names=model_counts.index, values=model_counts.values,
                color_discrete_sequence=[BRAND_ACCENT, BRAND_SUCCESS, BRAND_WARNING],
                title="Model Usage", hole=0.5,
            )
            fig_model.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", font_color="#c9d1d9", height=350,
            )
            st.plotly_chart(fig_model, use_container_width=True, key="hist_model_pie")

        # Loyalty probability distribution across history
        fig_prob = px.histogram(
            history, x="loyalty_probability", color="prediction_type", nbins=25,
            color_discrete_map={"Single": BRAND_ACCENT, "Batch": BRAND_SUCCESS},
            title="Loyalty Probability Distribution (All History)",
        )
        fig_prob.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#c9d1d9", height=350,
        )
        st.plotly_chart(fig_prob, use_container_width=True, key="hist_prob")

    # --- Clear history ---
    st.markdown("---")
    with st.expander("Danger Zone"):
        st.warning("This will permanently delete all prediction history records.")
        if st.button("Clear All History", type="primary", key="clear_hist"):
            clear_history()
            st.success("History cleared.")
            st.rerun()

    render_footer()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Loyalty Membership Predictions",
        page_icon="https://img.icons8.com/fluency/48/loyalty-card.png",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Initialise SQLite database
    init_db()

    # --- Sidebar ---
    if "page" not in st.session_state:
        st.session_state["page"] = "Dashboard"

    with st.sidebar:
        logo_b64 = get_logo_base64()
        if logo_b64:
            st.markdown(
                f'<div style="text-align:center; padding:0.3rem 0 0.2rem 0;">'
                f'<img src="{logo_b64}" style="height:36px; border-radius:6px;">'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="text-align:center; padding:0.3rem 0 0.2rem 0;">'
                '<span style="color:#00B4D8; font-weight:700; font-size:0.9rem;">LMP</span>'
                '</div>',
                unsafe_allow_html=True,
            )

        # --- Compact grouped navigation ---
        nav_groups = {
            "OVERVIEW": ["Dashboard"],
            "DATA": ["Data Exploration", "Preprocessing"],
            "MODELS": ["Model Performance", "Model Comparison", "Feature Importance"],
            "PREDICT": ["Single Prediction", "Batch Prediction", "Prediction History"],
            "": ["About"],
        }

        for group_label, items in nav_groups.items():
            if group_label:
                st.markdown(f'<p class="nav-group-label">{group_label}</p>', unsafe_allow_html=True)
            for page_name in items:
                btn_type = "primary" if st.session_state["page"] == page_name else "secondary"
                if st.button(
                    page_name,
                    key=f"nav_{page_name}",
                    use_container_width=True,
                    type=btn_type,
                ):
                    st.session_state["page"] = page_name
                    st.rerun()

        # Sidebar footer
        st.markdown(
            '<div style="font-size:0.6rem; color:#3d4450; text-align:center; padding:0.4rem 0 0;">'
            '&copy; 2026 LMP v2.0'
            '</div>',
            unsafe_allow_html=True,
        )

    page = st.session_state["page"]

    # --- Load resources ---
    bundle = load_model()
    raw_df = load_dataset()

    metrics = None
    if raw_df is not None:
        metrics = compute_model_metrics(bundle, raw_df)

    # --- Route pages ---
    if page == "Dashboard":
        page_dashboard(bundle, raw_df, metrics)
    elif page == "Data Exploration":
        page_data_exploration(raw_df)
    elif page == "Preprocessing":
        if metrics:
            page_preprocessing(bundle, metrics)
        else:
            render_header()
            st.warning("Dataset not found — preprocessing details unavailable.")
    elif page == "Model Performance":
        page_model_performance(metrics)
    elif page == "Model Comparison":
        page_model_comparison(metrics)
    elif page == "Feature Importance":
        page_feature_importance(bundle, metrics)
    elif page == "Single Prediction":
        page_single_prediction(bundle)
    elif page == "Batch Prediction":
        page_batch_prediction(bundle)
    elif page == "Prediction History":
        page_prediction_history()
    elif page == "About":
        page_about()


if __name__ == "__main__":
    main()
