"""
app/streamlit_app.py
-----------------------
Cardio AI — Clinical Decision Support & Explainable AI Dashboard
AI-Powered Cardiovascular Risk Prediction, SHAP Explainability & Clinical Analytics.

Architecture:
  - Navigation: Multi-view executive dashboard with sidebar navigation
  - 1. Executive Dashboard (KPIs, active model info, risk summary)
  - 2. Patient Assessment & Risk Prediction (Grouped clinical cards, sample presets, prediction engine)
  - 3. Explainable AI (SHAP local attribution, waterfall/bar, plain-language breakdown, GenAI communicator)
  - 4. Clinical Risk Factor Analysis (AHA/ACC benchmark comparisons for BP, cholesterol, heart rate, ST depression)
  - 5. Model Performance & Validation (Real test metrics, ROC, PR, calibration, confusion matrix)
  - 6. Dataset Insights & EDA (Feature distributions, correlation heatmap, statistical summaries)
  - 7. Model Architecture & Selection (Methodology, 5-candidate CV comparison, calibration details)
  - 8. Prediction History (Session audit log with export)

Integrates seamlessly with the FastAPI backend over HTTP when online, with
instant fallback to local in-process src/ modules if backend is offline.
"""

from __future__ import annotations

import datetime
import io
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

from src.data.preprocessing import CATEGORY_LABELS, display_name
from src.data.validation import NUMERIC_RANGES, validate_patient_record
from src.explainability.shap_explainer import build_explainer, build_full_evidence
from src.genai.explainer import generate_explanation
from src.models.inference import load_artifacts, predict_one
from src.utils.config import CONFIG

# -----------------------------------------------------------------------------
# Streamlit Page Setup & Custom Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Cardio AI | Explainable Cardiovascular Risk Dashboard",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    /* Global Card Styles */
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-subtitle {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 2px;
    }

    /* Risk Status Cards */
    .risk-card-high {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border: 1px solid #f87171;
        border-left: 6px solid #dc2626;
        border-radius: 10px;
        padding: 20px 24px;
        margin: 16px 0;
    }
    .risk-card-low {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 1px solid #86efac;
        border-left: 6px solid #16a34a;
        border-radius: 10px;
        padding: 20px 24px;
        margin: 16px 0;
    }

    /* Badges */
    .badge-normal {
        background-color: #dcfce7;
        color: #15803d;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .badge-elevated {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .badge-warning {
        background-color: #fef3c7;
        color: #b45309;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .badge-info {
        background-color: #e0f2fe;
        color: #0369a1;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
    }

    /* Disclaimer box */
    .disclaimer-box {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #92400e;
        margin: 16px 0;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")


def check_backend_api() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=1.0)
        return r.status_code == 200 and r.json().get("model_loaded") is True
    except Exception:
        return False


try:
    if "OPENAI_API_KEY" in st.secrets and not CONFIG.openai_api_key:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass


# -----------------------------------------------------------------------------
# Cached Model & Explainer Loading
# -----------------------------------------------------------------------------
@st.cache_resource
def get_loaded_model():
    loaded = load_artifacts(CONFIG.models_dir)
    background = pd.read_csv(CONFIG.models_dir / "background_sample.csv")
    explainer = build_explainer(loaded.model, loaded.model_type, background)
    from src.explainability.shap_explainer import EvidenceBundle, SHAP_OUTPUT_SPACE, add_global_importance
    dummy = EvidenceBundle(base_value=0.5, output_space=SHAP_OUTPUT_SPACE)
    add_global_importance(dummy, explainer, background)
    return loaded, explainer, background


@st.cache_data
def get_reference_dataset():
    path = CONFIG.data_dir / "sample_synthetic_heart.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


# -----------------------------------------------------------------------------
# Data Wrappers for Consistent API / In-Process Output
# -----------------------------------------------------------------------------
class FeatureItem(SimpleNamespace):
    def as_dict(self):
        return {
            "feature": self.feature,
            "feature_label": getattr(self, "feature_label", display_name(self.feature)),
            "value": getattr(self, "value", 0.0),
            "contribution": getattr(self, "contribution", 0.0),
            "direction": getattr(self, "direction", ""),
            "explanation": getattr(self, "explanation", ""),
        }


class GlobalItem(SimpleNamespace):
    def as_dict(self):
        return {
            "feature": self.feature,
            "feature_label": getattr(self, "feature_label", display_name(self.feature)),
            "mean_abs_shap": getattr(self, "mean_abs_shap", 0.0),
        }


class ApiEvidence:
    def __init__(self, data: dict):
        self.base_value = data.get("base_value", 0.5)
        self.output_space = data.get("output_space", "probability_space")
        self.local_contributions = [FeatureItem(**c) for c in data.get("top_contributions", [])]
        self.global_importance = [GlobalItem(**g) for g in data.get("global_importance", [])]

    def top(self, n=5):
        return sorted(self.local_contributions, key=lambda c: abs(c.contribution), reverse=True)[:n]


# -----------------------------------------------------------------------------
# Chart Helpers
# -----------------------------------------------------------------------------
def plot_shap_bar(contributions, n=8):
    top = sorted(contributions, key=lambda c: abs(c.contribution), reverse=True)[:n]
    labels = [c.feature_label for c in reversed(top)]
    values = [c.contribution for c in reversed(top)]
    colors = ["#dc2626" if v > 0 else "#2563eb" for v in values]

    fig, ax = plt.subplots(figsize=(6, 3.8))
    bars = ax.barh(labels, values, color=colors, height=0.6)
    ax.axvline(0, color="#64748b", linewidth=0.9, linestyle="--")
    ax.set_xlabel("SHAP Impact (Probability Space: + Risk, - Protective)", fontsize=9, color="#475569")
    ax.set_title("Local Feature Attribution (This Patient)", fontsize=11, fontweight="bold", pad=10)
    ax.tick_params(axis="both", labelsize=8.5)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    fig.tight_layout()
    return fig


def plot_global_importance(global_importance, n=8):
    top = global_importance[:n]
    labels = [g.feature_label for g in reversed(top)]
    values = [g.mean_abs_shap for g in reversed(top)]

    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.barh(labels, values, color="#475569", height=0.6)
    ax.set_xlabel("Mean |SHAP| (Probability Space)", fontsize=9, color="#475569")
    ax.set_title("Global Feature Importance (Cohort Snapshot)", fontsize=11, fontweight="bold", pad=10)
    ax.tick_params(axis="both", labelsize=8.5)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Clinical Benchmark Rules (AHA / ACC Educational Guidelines)
# -----------------------------------------------------------------------------
def analyze_clinical_factors(patient: dict) -> list[dict]:
    factors = []
    # Resting Blood Pressure
    bp = float(patient["trestbps"])
    if bp < 120:
        bp_status = ("Normal", "badge-normal", "Optimal (<120 mm Hg) per AHA guidelines.")
    elif bp < 130:
        bp_status = ("Elevated", "badge-warning", "Elevated (120-129 mm Hg). Lifestyle interventions suggested.")
    elif bp < 140:
        bp_status = ("Stage 1 Hypertension", "badge-elevated", "Stage 1 HTN (130-139 mm Hg).")
    else:
        bp_status = ("Stage 2 Hypertension", "badge-elevated", "Stage 2 HTN (>=140 mm Hg). Elevated cardiovascular stress.")
    factors.append({"factor": "Resting Blood Pressure", "value": f"{bp:.0f} mm Hg", "status": bp_status[0], "badge": bp_status[1], "benchmark": "< 120 mm Hg (Normal)", "notes": bp_status[2]})

    # Serum Cholesterol
    chol = float(patient["chol"])
    if chol < 200:
        chol_status = ("Desirable", "badge-normal", "Desirable blood cholesterol level.")
    elif chol < 240:
        chol_status = ("Borderline High", "badge-warning", "Borderline high (200-239 mg/dl).")
    else:
        chol_status = ("High", "badge-elevated", "High (>=240 mg/dl). Major vascular risk contributor.")
    factors.append({"factor": "Serum Cholesterol", "value": f"{chol:.0f} mg/dl", "status": chol_status[0], "badge": chol_status[1], "benchmark": "< 200 mg/dl (Desirable)", "notes": chol_status[2]})

    # Max Heart Rate Achieved vs Age-Predicted Max (220 - age)
    age = float(patient["age"])
    hr = float(patient["thalach"])
    pred_max = 220 - age
    target_85 = 0.85 * pred_max
    if hr >= target_85:
        hr_status = ("Adequate Target Achieved", "badge-normal", f">=85% of age-predicted maximum ({pred_max:.0f} bpm).")
    else:
        hr_status = ("Sub-Target (<85% Max)", "badge-warning", f"Below 85% target ({target_85:.0f} bpm). May indicate chronotropic incompetence or fatigue.")
    factors.append({"factor": "Max Heart Rate (Stress Test)", "value": f"{hr:.0f} bpm", "status": hr_status[0], "badge": hr_status[1], "benchmark": f">= {target_85:.0f} bpm (85% Age Max)", "notes": hr_status[2]})

    # ST Depression (Oldpeak)
    oldpeak = float(patient["oldpeak"])
    if oldpeak < 1.0:
        op_status = ("Physiological", "badge-normal", "Oldpeak < 1.0 mm ST depression.")
    elif oldpeak < 2.0:
        op_status = ("Mild-to-Moderate Depression", "badge-warning", "ST depression 1.0-1.9 mm. Suggestive of exercise-induced ischemia.")
    else:
        op_status = ("Significant Depression", "badge-elevated", "ST depression >= 2.0 mm. Pronounced indicator of myocardial strain.")
    factors.append({"factor": "ST Depression (Oldpeak)", "value": f"{oldpeak:.1f} mm", "status": op_status[0], "badge": op_status[1], "benchmark": "< 1.0 mm (Normal)", "notes": op_status[2]})

    # Fasting Blood Sugar
    fbs = int(patient["fbs"])
    fbs_text = "Fasting glucose > 120 mg/dl" if fbs == 1 else "<= 120 mg/dl"
    fbs_status = ("Elevated (>120 mg/dl)", "badge-elevated", "Suggests impaired fasting glucose / metabolic syndrome.") if fbs == 1 else ("Normal (<=120 mg/dl)", "badge-normal", "Within normal glycemic range.")
    factors.append({"factor": "Fasting Blood Sugar", "value": fbs_text, "status": fbs_status[0], "badge": fbs_status[1], "benchmark": "<= 120 mg/dl", "notes": fbs_status[2]})

    # Exercise Angina
    exang = int(patient["exang"])
    ex_text = "Yes" if exang == 1 else "No"
    ex_status = ("Present", "badge-elevated", "Exercise-induced chest tightness/ischemia.") if exang == 1 else ("Absent", "badge-normal", "No exercise-induced angina recorded.")
    factors.append({"factor": "Exercise-Induced Angina", "value": ex_text, "status": ex_status[0], "badge": ex_status[1], "benchmark": "Absent (No)", "notes": ex_status[2]})

    return factors


# -----------------------------------------------------------------------------
# Pre-set Patient Sample Profiles
# -----------------------------------------------------------------------------
SAMPLE_PROFILES = {
    "Profile 1: Lower Risk Profile (Female, 42y)": {
        "age": 42, "sex": 0, "cp": 1, "trestbps": 118, "chol": 185, "fbs": 0,
        "restecg": 0, "thalach": 168, "exang": 0, "oldpeak": 0.2, "slope": 2, "ca": 0, "thal": 1,
    },
    "Profile 2: Elevated Risk Profile (Male, 63y)": {
        "age": 63, "sex": 1, "cp": 0, "trestbps": 152, "chol": 288, "fbs": 1,
        "restecg": 1, "thalach": 125, "exang": 1, "oldpeak": 2.8, "slope": 1, "ca": 2, "thal": 3,
    },
    "Profile 3: Intermediate Profile (Male, 54y)": {
        "age": 54, "sex": 1, "cp": 2, "trestbps": 134, "chol": 236, "fbs": 0,
        "restecg": 0, "thalach": 148, "exang": 0, "oldpeak": 1.1, "slope": 1, "ca": 1, "thal": 2,
    },
}


# -----------------------------------------------------------------------------
# Main Application Flow & View Router
# -----------------------------------------------------------------------------
def main():
    # Session State Initialization
    if "patient_data" not in st.session_state:
        st.session_state.patient_data = dict(SAMPLE_PROFILES["Profile 3: Intermediate Profile (Male, 54y)"])
    if "analysis_cache" not in st.session_state:
        st.session_state.analysis_cache = None
    if "genai_result" not in st.session_state:
        st.session_state.genai_result = None
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []

    # Model and artifact check
    models_dir = CONFIG.models_dir
    if not (models_dir / "metadata.json").exists():
        st.error(
            "⚠️ No trained model found in `models/`. Please run: "
            "`python -m src.models.train --data data/sample_synthetic_heart.csv`"
        )
        st.stop()

    loaded, explainer, background = get_loaded_model()
    m = loaded.metadata

    # Check live backend API
    backend_online = check_backend_api()

    # Sidebar Header & Navigation
    st.sidebar.markdown("## ❤️ **Cardio AI**")
    st.sidebar.caption("Clinical Decision Support & Explainable AI Prototype")

    if backend_online:
        st.sidebar.markdown(f'<span class="badge-normal">● FastAPI Backend Online (Port 8000)</span>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown(f'<span class="badge-warning">○ In-Process Engine (Backend Offline)</span>', unsafe_allow_html=True)

    st.sidebar.markdown("---")
    PAGES = [
        "📊 Executive Dashboard",
        "🩺 Patient Risk Assessment",
        "🔍 Explainable AI (SHAP & GenAI)",
        "⚠️ Clinical Risk Factor Analysis",
        "📈 Model Performance & Validation",
        "🔬 Dataset Insights & EDA",
        "🧬 Model Architecture & Registry",
        "📜 Prediction History",
    ]

    menu_selection = st.sidebar.radio(
        "Navigation",
        PAGES,
        index=1,          # Default to Patient Risk Assessment
        key="nav_radio",  # Stable key lets Streamlit persist selection across reruns
    )
    st.sidebar.markdown("---")

    # Global Medical Disclaimer Banner
    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>⚕️ Clinical Research Disclaimer:</strong> Cardio AI is an educational and research prototype.
            It is <strong>not a certified medical device</strong>, does not produce clinical diagnoses, and must not be used for
            treatment planning. All probabilities reflect tabular training distributions and require licensed physician interpretation.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # VIEW 1: EXECUTIVE DASHBOARD
    # -------------------------------------------------------------------------
    if menu_selection == "📊 Executive Dashboard":
        st.subheader("📊 Executive Overview & Active Model Status")
        st.markdown(
            "Cardio AI combines **machine learning (SVM with Platt sigmoid calibration)**, **SHAP game-theoretic explainability**, "
            "and a **guarded GenAI communication layer** to assist clinical teams in evaluating cardiovascular risk."
        )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Active Model</div>
                <div class="metric-value">{loaded.model_type.upper()}</div>
                <div class="metric-subtitle">Sigmoid Calibrated ({m.get('model_version', 'v1')[:16]})</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            roc_auc = m.get("test_metrics", {}).get("roc_auc", 0.79)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Test ROC-AUC</div>
                <div class="metric-value">{roc_auc:.3f}</div>
                <div class="metric-subtitle">Single-pass evaluation on held-out test</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            brier = m.get("brier_after_calibration", 0.1896)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Brier Calibration Score</div>
                <div class="metric-value">{brier:.4f}</div>
                <div class="metric-subtitle">Lower is better (Platt scaled)</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            n_features = m.get("n_features", 21)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Engineered Features</div>
                <div class="metric-value">{n_features}</div>
                <div class="metric-subtitle">From 13 raw clinical inputs</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 🎯 Workflow: Predict → Explain → Communicate")
        st.markdown(
            """
            ```mermaid
            flowchart LR
                A[Patient Vitals & Labs] -->|StandardScaler + OneHot| B[Calibrated SVM Model]
                B -->|Outputs Probability| C[Risk Probability Score]
                B -->|Permutation Explainer| D[SHAP Attribution Evidence]
                D -->|Validated Guardrails| E[GenAI Plain-Language Summary]
            ```
            """
        )

        if st.session_state.analysis_cache:
            p, ev, src = st.session_state.analysis_cache
            st.markdown("### 👤 Last Evaluated Patient Summary")
            c_a, c_b = st.columns([1, 2])
            with c_a:
                if p >= 0.5:
                    st.markdown(f"""
                    <div class="risk-card-high">
                        <span class="badge-elevated">ELEVATED PREDICTED RISK</span>
                        <h2 style="margin: 8px 0; color: #991b1b;">{p * 100:.1f}% Probability</h2>
                        <p style="margin:0; font-size: 0.85rem; color: #7f1d1d;">Model indicates positive cardiovascular risk alignment based on supplied clinical factors.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="risk-card-low">
                        <span class="badge-normal">LOW PREDICTED RISK</span>
                        <h2 style="margin: 8px 0; color: #166534;">{p * 100:.1f}% Probability</h2>
                        <p style="margin:0; font-size: 0.85rem; color: #14532d;">Model indicates low probability for cardiovascular risk in this cohort.</p>
                    </div>
                    """, unsafe_allow_html=True)
            with c_b:
                st.markdown("**Top Local Drivers for this Prediction:**")
                top_items = ev.top(4)
                for item in top_items:
                    d_badge = "badge-elevated" if item.contribution > 0 else "badge-info"
                    st.markdown(
                        f"- **{item.feature_label}**: SHAP contribution `{item.contribution:+.3f}` "
                        f'<span class="{d_badge}">{item.direction.replace("_", " ")}</span>',
                        unsafe_allow_html=True
                    )
        else:
            st.info("💡 No patient has been evaluated in this session yet. Navigate to **'Patient Risk Assessment'** to run an analysis.")

    # -------------------------------------------------------------------------
    # VIEW 2: PATIENT RISK ASSESSMENT
    # -------------------------------------------------------------------------
    elif menu_selection == "🩺 Patient Risk Assessment":
        st.subheader("🩺 Patient Risk Assessment")
        st.caption("Enter patient demographics, clinical measurements, and resting/exercise ECG test results.")

        # Preset Loader & Reset Bar
        p_col1, p_col2 = st.columns([3, 1])
        with p_col1:
            selected_preset = st.selectbox(
                "📥 Quick-Load Standard Sample Patient Profile",
                list(SAMPLE_PROFILES.keys()),
                index=0,
                help="Select a clinically plausible reference profile to demonstrate the model instantly.",
            )
        with p_col2:
            st.write("")
            st.write("")
            if st.button("Apply Selected Profile", width='stretch'):
                st.session_state.patient_data = dict(SAMPLE_PROFILES[selected_preset])
                st.session_state.analysis_cache = None
                st.session_state.genai_result = None
                st.toast(f"✅ Profile loaded: {selected_preset}", icon="📥")

        st.markdown("---")

        # Grouped Input Cards
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("#### 👤 1. Demographics")
            cur = st.session_state.patient_data
            age_val = st.number_input(
                "Age (years)",
                min_value=18, max_value=100, value=int(cur["age"]),
                help="Patient chronological age (18 - 100 years).",
            )
            sex_val = st.selectbox(
                "Biological Sex",
                options=[0, 1],
                index=0 if cur["sex"] == 0 else 1,
                format_func=lambda k: "Female (0)" if k == 0 else "Male (1)",
                help="Biological sex at birth as recorded in Cleveland heart study schema.",
            )

            st.markdown("#### 💓 2. Symptoms")
            cp_val = st.selectbox(
                "Chest Pain Type (cp)",
                options=[0, 1, 2, 3],
                index=int(cur["cp"]),
                format_func=lambda k: f"{k} — {CATEGORY_LABELS['cp'][k]}",
                help="0: Typical Angina, 1: Atypical Angina, 2: Non-Anginal, 3: Asymptomatic.",
            )
            exang_val = st.selectbox(
                "Exercise-Induced Angina (exang)",
                options=[0, 1],
                index=int(cur["exang"]),
                format_func=lambda k: "No (0)" if k == 0 else "Yes (1)",
                help="Did the patient experience angina during treadmill exercise stress testing?",
            )

        with c2:
            st.markdown("#### 🩺 3. Vitals & Blood Chemistry")
            trestbps_val = st.number_input(
                "Resting Blood Pressure (mm Hg)",
                min_value=80, max_value=220, value=int(cur["trestbps"]),
                help="Resting blood pressure measured upon hospital admission (mm Hg). Normal is < 120 mm Hg.",
            )
            chol_val = st.number_input(
                "Serum Cholesterol (mg/dl)",
                min_value=100, max_value=600, value=int(cur["chol"]),
                help="Serum total cholesterol in mg/dl. Normal desirable level is < 200 mg/dl.",
            )
            fbs_val = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dl (fbs)",
                options=[0, 1],
                index=int(cur["fbs"]),
                format_func=lambda k: "No (<= 120 mg/dl)" if k == 0 else "Yes (> 120 mg/dl)",
                help="Indicates elevated fasting glucose (> 120 mg/dl), a metabolic risk factor.",
            )
            thalach_val = st.number_input(
                "Max Heart Rate Achieved (bpm)",
                min_value=60, max_value=230, value=int(cur["thalach"]),
                help="Maximum heart rate attained during standardized exercise stress protocol.",
            )

        with c3:
            st.markdown("#### 🔬 4. ECG & Fluoroscopy")
            restecg_val = st.selectbox(
                "Resting ECG Results (restecg)",
                options=[0, 1, 2],
                index=int(cur["restecg"]),
                format_func=lambda k: f"{k} — {CATEGORY_LABELS['restecg'][k]}",
                help="0: Normal, 1: ST-T wave changes, 2: Left ventricular hypertrophy.",
            )
            oldpeak_val = st.number_input(
                "ST Depression (Oldpeak mm)",
                min_value=0.0, max_value=7.0, value=float(cur["oldpeak"]), step=0.1,
                help="ST depression induced by exercise relative to rest (mm). Values >= 1.0 indicate potential ischemia.",
            )
            slope_val = st.selectbox(
                "ST Segment Slope (slope)",
                options=[0, 1, 2],
                index=int(cur["slope"]),
                format_func=lambda k: f"{k} — {CATEGORY_LABELS['slope'][k]}",
                help="Slope of peak exercise ST segment: 0: Upsloping, 1: Flat, 2: Downsloping.",
            )
            ca_val = st.selectbox(
                "Major Vessels by Fluoroscopy (ca)",
                options=[0, 1, 2, 3, 4],
                index=int(cur["ca"]),
                format_func=lambda k: f"{k} vessels colored",
                help="Number of major vessels (0-3, rarely 4) colored by fluoroscopy.",
            )
            thal_val = st.selectbox(
                "Thallium Scintigraphy (thal)",
                options=[1, 2, 3],
                index=max(0, min(2, int(cur["thal"]) - 1)),
                format_func=lambda k: f"{k} — {CATEGORY_LABELS['thal'][k]}",
                help="1: Normal, 2: Fixed defect, 3: Reversible defect.",
            )

        # Assemble current record
        active_patient = {
            "age": age_val, "sex": sex_val, "cp": cp_val, "trestbps": trestbps_val,
            "chol": chol_val, "fbs": fbs_val, "restecg": restecg_val, "thalach": thalach_val,
            "exang": exang_val, "oldpeak": oldpeak_val, "slope": slope_val, "ca": ca_val, "thal": thal_val,
        }
        st.session_state.patient_data = active_patient

        st.markdown("---")
        btn_c1, btn_c2, _ = st.columns([2, 1, 3])
        with btn_c1:
            run_clicked = st.button("🚀 Run Comprehensive Assessment", type="primary", width='stretch')
        with btn_c2:
            if st.button("🔄 Reset Analysis", width='stretch'):
                st.session_state.analysis_cache = None
                st.session_state.genai_result = None
                st.toast("🔄 Analysis reset. Adjust inputs and run again.", icon="🔄")

        # Handle Execution
        if run_clicked:
            validation = validate_patient_record(active_patient)
            if not validation.is_valid:
                for err in validation.errors:
                    st.error(f"❌ Input Validation Error: {err}")
                st.stop()

            with st.spinner("Executing calibrated SVM inference and calculating SHAP attributions..."):
                if backend_online:
                    try:
                        p_res = requests.post(f"{API_URL}/predict", json=active_patient, timeout=10).json()
                        proba = float(p_res["probability"])
                        e_res = requests.post(f"{API_URL}/explain", json=active_patient, timeout=60).json()
                        evidence = ApiEvidence(e_res)
                        source_engine = "FastAPI Backend (REST API)"
                    except Exception as ex:
                        st.warning(f"Backend API unavailable ({ex}); using local in-process engine.")
                        proba, instance = predict_one(loaded, active_patient)
                        evidence = build_full_evidence(explainer, instance, loaded.model_type, background)
                        source_engine = "In-Process Engine"
                else:
                    proba, instance = predict_one(loaded, active_patient)
                    evidence = build_full_evidence(explainer, instance, loaded.model_type, background)
                    source_engine = "In-Process Engine"

                st.session_state.analysis_cache = (proba, evidence, source_engine)

                # Append to Prediction History
                history_record = {
                    "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Age": age_val,
                    "Sex": "Male" if sex_val == 1 else "Female",
                    "Resting BP": f"{trestbps_val} mmHg",
                    "Cholesterol": f"{chol_val} mg/dl",
                    "Chest Pain": CATEGORY_LABELS["cp"][cp_val],
                    "Probability": f"{proba * 100:.1f}%",
                    "Risk Status": "ELEVATED" if proba >= 0.5 else "LOW",
                    "Engine": source_engine,
                }
                st.session_state.prediction_history.insert(0, history_record)

        # Display Results If Available
        if st.session_state.analysis_cache:
            proba, evidence, source_engine = st.session_state.analysis_cache

            st.markdown("### 📊 Assessment Findings")
            # Native metric — enables programmatic access (tests, screen readers, dashboards)
            risk_label = "ELEVATED RISK" if proba >= 0.5 else "LOW RISK"
            st.metric(
                label="Model-Predicted Probability (Positive Class)",
                value=f"{proba * 100:.1f}%",
                delta=risk_label,
                delta_color="inverse" if proba >= 0.5 else "normal",
            )
            res_col1, res_col2 = st.columns([1, 1])

            with res_col1:
                if proba >= 0.5:
                    st.markdown(f"""
                    <div class="risk-card-high">
                        <span class="badge-elevated">ELEVATED CARDIOVASCULAR RISK</span>
                        <h1 style="color: #991b1b; margin: 8px 0;">{proba * 100:.1f}%</h1>
                        <p style="margin: 0; font-size: 0.9rem; color: #7f1d1d;">
                            The calibrated machine learning model predicted an elevated likelihood for the positive class (target=1).
                            Primary drivers include exercise-induced symptoms, ST depression, or vessel occlusion.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="risk-card-low">
                        <span class="badge-normal">LOW CARDIOVASCULAR RISK</span>
                        <h1 style="color: #166534; margin: 8px 0;">{proba * 100:.1f}%</h1>
                        <p style="margin: 0; font-size: 0.9rem; color: #14532d;">
                            The calibrated model estimates a low probability for the positive class (target=1) in this tabular cohort.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

            with res_col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Inference Engine & Calibration</div>
                    <p style="font-size: 0.95rem; margin-bottom: 6px;"><strong>Active Model:</strong> Support Vector Classifier (SVM)</p>
                    <p style="font-size: 0.95rem; margin-bottom: 6px;"><strong>Probability Calibration:</strong> Sigmoid (Platt Scaling via 5-Fold Inner CV)</p>
                    <p style="font-size: 0.95rem; margin-bottom: 6px;"><strong>Execution Source:</strong> <code>{source_engine}</code></p>
                    <p style="font-size: 0.85rem; color: #64748b; margin: 0;">Probabilities are calibrated to match empirical true likelihoods across fold validation rather than raw distance-to-margin scores.</p>
                </div>
                """, unsafe_allow_html=True)

            # Quick XAI Preview
            st.markdown("#### 🔍 Primary Attributions (SHAP Preview)")
            top_factors = evidence.top(4)
            t_cols = st.columns(4)
            for idx, item in enumerate(top_factors):
                with t_cols[idx]:
                    direction_color = "#dc2626" if item.contribution > 0 else "#2563eb"
                    badge_cls = "badge-elevated" if item.contribution > 0 else "badge-info"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{item.feature_label}</div>
                        <div style="font-size: 1.25rem; font-weight: 700; color: {direction_color};">
                            {item.contribution:+.3f}
                        </div>
                        <span class="{badge_cls}">{item.direction.replace('_', ' ')}</span>
                    </div>
                    """, unsafe_allow_html=True)

            st.info("👉 For in-depth SHAP plots and the patient-friendly GenAI explanation, open the **'Explainable AI (SHAP & GenAI)'** tab in the sidebar.")

    # -------------------------------------------------------------------------
    # VIEW 3: EXPLAINABLE AI (XAI)
    # -------------------------------------------------------------------------
    elif menu_selection == "🔍 Explainable AI (SHAP & GenAI)":
        st.subheader("🔍 Explainable AI (XAI): SHAP Evidence & Patient Communication")
        st.caption("Inspect the exact mathematical feature contributions behind this prediction and generate plain-language explanations.")

        if not st.session_state.analysis_cache:
            st.warning("⚠️ No patient prediction has been computed yet. Please go to **'Patient Risk Assessment'** and click 'Run Comprehensive Assessment'.")
            st.stop()

        proba, evidence, source_engine = st.session_state.analysis_cache
        active_patient = st.session_state.patient_data

        # SHAP Waterfall/Bar Plots
        st.markdown("### 1. Local & Global SHAP Visualizations")
        col_plot1, col_plot2 = st.columns(2)
        with col_plot1:
            st.pyplot(plot_shap_bar(evidence.local_contributions, n=8))
            st.caption("🔴 Red bars pushed probability UP (increased risk score). 🔵 Blue bars pushed probability DOWN (protective factor).")
        with col_plot2:
            st.pyplot(plot_global_importance(evidence.global_importance, n=8))
            st.caption("Global mean |SHAP| computed across background baseline sample. Shows population-level model feature sensitivity.")

        st.markdown("---")

        # Detailed Factor Contribution Table
        st.markdown("### 2. Tabular Feature Attribution Breakdown")
        with st.expander("View Complete Local Feature Contributions (All Features)", expanded=True):
            contrib_df = pd.DataFrame([c.as_dict() for c in evidence.local_contributions])
            st.dataframe(
                contrib_df[["feature_label", "value", "contribution", "direction", "explanation"]],
                column_config={
                    "feature_label": "Clinical Feature",
                    "value": "Patient Value",
                    "contribution": st.column_config.NumberColumn("SHAP Impact", format="%+.4f"),
                    "direction": "Impact Direction",
                    "explanation": "Human-Readable Influence",
                },
                width='stretch',
                hide_index=True,
            )

        st.markdown("---")

        # Stage 3: GenAI Patient Communication Layer
        st.markdown("### 3. Patient-Friendly Clinical Communication (GenAI)")
        st.caption("Translates SHAP evidence into clear, empathetic language with multi-tier medical disclaimers and safety guardrails.")

        if st.button("✨ Generate Plain-Language Clinical Explanation", type="primary"):
            with st.spinner("Generating guarded plain-language communication..."):
                top_dict_factors = [c.as_dict() for c in evidence.top(5)]
                if backend_online:
                    try:
                        g_res = requests.post(f"{API_URL}/generate-explanation", json=active_patient, timeout=60).json()
                        st.session_state.genai_result = SimpleNamespace(
                            summary=g_res["summary"],
                            key_factors=[SimpleNamespace(**kf) for kf in g_res["key_factors"]],
                            limitations=g_res["limitations"],
                            safety_note=g_res["safety_note"],
                            source=g_res["source"],
                        )
                    except Exception:
                        st.session_state.genai_result = generate_explanation(proba, loaded.model_type, top_dict_factors)
                else:
                    st.session_state.genai_result = generate_explanation(proba, loaded.model_type, top_dict_factors)

        if st.session_state.genai_result:
            r = st.session_state.genai_result
            st.markdown(f"""
            <div class="metric-card" style="background-color: #ffffff; border-left: 5px solid #2563eb;">
                <h4 style="margin-top:0; color: #1e3a8a;">📋 Plain-Language Summary</h4>
                <p style="font-size: 1rem; color: #1e293b;">{r.summary}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### Key Influencing Factors Explained:")
            for kf in r.key_factors:
                badge = "badge-elevated" if "push" in kf.direction or "increase" in kf.direction else "badge-info"
                st.markdown(f"- **{kf.feature}**: {kf.explanation}")

            with st.expander("Clinical Limitations & Context", expanded=False):
                for lim in r.limitations:
                    st.markdown(f"- ⚠️ {lim}")

            st.caption(f"🔒 **Safety Note:** {r.safety_note}")
            st.caption(f"ℹ️ **Generation Source:** `{r.source}` (Template fallback is deterministic; live GenAI activates when `OPENAI_API_KEY` is present in `.env`).")

    # -------------------------------------------------------------------------
    # VIEW 4: CLINICAL RISK FACTOR ANALYSIS
    # -------------------------------------------------------------------------
    elif menu_selection == "⚠️ Clinical Risk Factor Analysis":
        st.subheader("⚠️ Clinical Risk Factor Benchmark Analysis")
        st.caption("Compares patient vitals against published American Heart Association (AHA) and ACC clinical reference targets.")

        if not st.session_state.patient_data:
            st.warning("⚠️ No patient data available. Please open 'Patient Risk Assessment' first.")
            st.stop()

        factors = analyze_clinical_factors(st.session_state.patient_data)

        # Factor table
        table_rows = []
        for f in factors:
            table_rows.append({
                "Measurement": f["factor"],
                "Patient Value": f["value"],
                "Clinical Benchmark": f["benchmark"],
                "Assessment Status": f["status"],
                "Clinical Context": f["notes"],
            })
        st.table(pd.DataFrame(table_rows))

        st.markdown("### 💡 Clinical Discussion Points for Providers")
        st.markdown(
            """
            - **Blood Pressure Management:** Persistent readings above 130/80 mm Hg warrant ambulatory blood pressure monitoring and sodium reduction guidance.
            - **Lipid Optimization:** Fasting lipid panel evaluation helps assess whether statin therapy aligns with primary ASCVD risk calculation.
            - **Exercise Capacity:** Inability to achieve >=85% of age-predicted maximum heart rate during stress testing correlates with adverse outcomes independently of ECG findings.
            - **ST Changes:** Exercise-induced ST depression >= 1.0 mm signifies subendocardial ischemia until proven otherwise by anatomical imaging (e.g. coronary angiography or CCTA).
            """
        )

    # -------------------------------------------------------------------------
    # VIEW 5: MODEL PERFORMANCE & VALIDATION
    # -------------------------------------------------------------------------
    elif menu_selection == "📈 Model Performance & Validation":
        st.subheader("📈 Model Performance & Validation Suite")
        st.caption("Evaluation metrics calculated on the held-out test split (test set was never used during model selection or cross-validation).")

        test_m = m.get("test_metrics", {})
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Test Accuracy", f"{test_m.get('accuracy', 0.696):.1%}")
        with col2:
            st.metric("Precision", f"{test_m.get('precision', 0.704):.1%}")
        with col3:
            st.metric("Recall (Sensitivity)", f"{test_m.get('recall', 0.679):.1%}")
        with col4:
            st.metric("F1 Score", f"{test_m.get('f1_score', 0.691):.3f}")
        with col5:
            st.metric("ROC-AUC", f"{test_m.get('roc_auc', 0.790):.3f}")

        st.markdown("---")
        st.markdown("### 📉 Diagnostic Validation Curves (Held-out Test Cohort)")

        reports_dir = CONFIG.reports_dir
        c_p1, c_p2 = st.columns(2)

        with c_p1:
            roc_path = reports_dir / "svm_roc_curve.png"
            if roc_path.exists():
                st.image(str(roc_path), caption="Receiver Operating Characteristic (ROC Curve) on Test Split", width='stretch')
            else:
                st.info("ROC plot not found in reports/.")

            cm_path = reports_dir / "svm_confusion_matrix.png"
            if cm_path.exists():
                st.image(str(cm_path), caption="Confusion Matrix on Test Split", width='stretch')
            else:
                st.info("Confusion matrix plot not found in reports/.")

        with c_p2:
            pr_path = reports_dir / "svm_pr_curve.png"
            if pr_path.exists():
                st.image(str(pr_path), caption="Precision-Recall Curve (PR-AUC on Test Split)", width='stretch')
            else:
                st.info("PR curve plot not found in reports/.")

            cal_path = reports_dir / "svm_calibration_curve.png"
            if cal_path.exists():
                st.image(str(cal_path), caption="Reliability Curve (Calibration Before/After Sigmoid Scaling)", width='stretch')
            else:
                st.info("Calibration curve plot not found in reports/.")

    # -------------------------------------------------------------------------
    # VIEW 6: DATASET INSIGHTS & EDA
    # -------------------------------------------------------------------------
    elif menu_selection == "🔬 Dataset Insights & EDA":
        st.subheader("🔬 Dataset Insights & Exploratory Data Analysis")
        st.caption("Distribution analysis and correlation structure of the UCI-format cardiovascular dataset.")

        df = get_reference_dataset()
        if df is None:
            st.warning("⚠️ Reference dataset `data/sample_synthetic_heart.csv` not found.")
            st.stop()

        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            st.metric("Total Cohort Size", f"{df.shape[0]} patients")
        with col_d2:
            pos_pct = (df['target'].sum() / len(df)) * 100
            st.metric("Positive Class Prevalence", f"{pos_pct:.1f}%", f"{df['target'].sum()} positive cases")
        with col_d3:
            st.metric("Raw Clinical Features", f"{df.shape[1] - 1} predictors")

        st.markdown("### 📊 Target Balance & Age Distribution")
        e_col1, e_col2 = st.columns(2)
        with e_col1:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            target_counts = df["target"].value_counts().rename({0: "Low Risk (0)", 1: "Elevated Risk (1)"})
            ax.bar(target_counts.index, target_counts.values, color=["#3b82f6", "#ef4444"], width=0.5)
            ax.set_ylabel("Patient Count", fontsize=9)
            ax.set_title("Cohort Target Distribution", fontsize=11, fontweight="bold")
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)

        with e_col2:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.hist(df[df["target"] == 0]["age"], bins=15, alpha=0.6, label="Low Risk (0)", color="#3b82f6")
            ax.hist(df[df["target"] == 1]["age"], bins=15, alpha=0.6, label="Elevated Risk (1)", color="#ef4444")
            ax.set_xlabel("Age (years)", fontsize=9)
            ax.set_ylabel("Frequency", fontsize=9)
            ax.set_title("Age Distribution by Target Class", fontsize=11, fontweight="bold")
            ax.legend(fontsize=8)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)

        st.markdown("### 🧮 Numerical Features Correlation Heatmap")
        num_cols = ["age", "trestbps", "chol", "thalach", "oldpeak", "target"]
        corr = df[num_cols].corr()

        fig, ax = plt.subplots(figsize=(7, 4.5))
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(num_cols)))
        ax.set_yticks(range(len(num_cols)))
        ax.set_xticklabels(num_cols, rotation=35, ha="right", fontsize=9)
        ax.set_yticklabels(num_cols, fontsize=9)
        for i in range(len(num_cols)):
            for j in range(len(num_cols)):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="black" if abs(corr.iloc[i, j]) < 0.6 else "white", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("Correlation Matrix of Key Clinical Measurements", fontsize=11, fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig)

        with st.expander("View Raw Dataset Statistics (Descriptive Summary)"):
            st.dataframe(df.describe().round(2), width='stretch')

    # -------------------------------------------------------------------------
    # VIEW 7: MODEL ARCHITECTURE & REGISTRY
    # -------------------------------------------------------------------------
    elif menu_selection == "🧬 Model Architecture & Registry":
        st.subheader("🧬 Model Selection Methodology & Registry")
        st.markdown(
            "To prevent **model-selection leakage**, the winning model is selected purely based on **mean CV ROC-AUC** "
            "across 5 folds on the training split only. The test set is evaluated exactly once on the selected winner."
        )

        st.markdown("### 🏆 5-Candidate K-Fold Cross-Validation Comparison")
        cv_results = m.get("cv_results", {})
        if cv_results:
            rows = []
            for k, v in cv_results.items():
                is_selected = "✅ Selected Winner" if k == m.get("best_model_type") else "Candidate"
                rows.append({
                    "Candidate Algorithm": k.replace("_", " ").title(),
                    "Mean CV ROC-AUC": f"{v.get('mean_roc_auc', 0):.3f}",
                    "CV Std Deviation": f"± {v.get('std_roc_auc', 0):.3f}",
                    "Status": is_selected,
                })
            st.table(pd.DataFrame(rows))

        st.markdown("### ⚙️ Complete Processing Architecture")
        st.markdown(
            """
            ```mermaid
            flowchart TD
                A[Patient Input JSON] --> B[Data Validation: Range & Type Checks]
                B --> C[Preprocessing: StandardScaler & OneHotEncoder]
                C --> D[Calibrated SVM Model: Platt Sigmoid Scaling]
                D --> E[Predicted Risk Probability Score]
                D --> F[SHAP Permutation Explainer]
                F --> G[Structured Feature Evidence Bundle]
                G --> H[Guarded GenAI Clinical Communicator]
            ```
            """
        )

        st.markdown("### 📋 Model Metadata & Provenance")
        st.json({
            "best_model_type": m.get("best_model_type"),
            "model_version": m.get("model_version"),
            "dataset_hash": m.get("dataset_hash"),
            "trained_at_utc": m.get("trained_at_utc"),
            "random_seed": m.get("random_seed"),
            "calibration_applied": m.get("calibration_applied"),
            "brier_before_calibration": m.get("brier_before_calibration"),
            "brier_after_calibration": m.get("brier_after_calibration"),
        })

    # -------------------------------------------------------------------------
    # VIEW 8: PREDICTION HISTORY
    # -------------------------------------------------------------------------
    elif menu_selection == "📜 Prediction History":
        st.subheader("📜 Session Prediction History & Audit Trail")
        st.caption("Logs all patient assessments conducted during the current browser session.")

        if not st.session_state.prediction_history:
            st.info("ℹ️ No predictions have been made in this session yet.")
        else:
            hist_df = pd.DataFrame(st.session_state.prediction_history)
            st.dataframe(hist_df, width='stretch', hide_index=True)

            # CSV Download
            csv_buffer = io.StringIO()
            hist_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Export Session History as CSV",
                data=csv_buffer.getvalue(),
                file_name=f"cardio_ai_session_{datetime.date.today()}.csv",
                mime="text/csv",
            )

            if st.button("Clear Prediction History"):
                st.session_state.prediction_history = []
                st.toast("History cleared.", icon="🗑️")


if __name__ == "__main__":
    main()
