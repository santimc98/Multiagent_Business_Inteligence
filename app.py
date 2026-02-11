import streamlit as st
import pandas as pd
import os
import json
import sys
import time
import glob
import signal
import threading
import io
import zipfile
from datetime import datetime

# Ensure src is in path
APP_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(APP_ROOT)

from src.graph.graph import app_graph, request_abort, clear_abort
from src.utils.run_workspace import recover_orphaned_workspace_cwd

# Auto-heal cwd when prior run crashed inside runs/<run_id>/work.
recover_orphaned_workspace_cwd(project_root=APP_ROOT)
try:
    os.chdir(APP_ROOT)
except Exception as cwd_err:
    print(f"APP_CWD_WARNING: {cwd_err}")

_SIGNAL_HANDLER_INSTALLED = False

def _load_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _handle_shutdown(signum, frame):
    request_abort(f"signal={signum}")
    raise KeyboardInterrupt

def _install_signal_handlers():
    global _SIGNAL_HANDLER_INSTALLED
    if _SIGNAL_HANDLER_INSTALLED:
        return
    if threading.current_thread() is not threading.main_thread():
        return
    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)
    _SIGNAL_HANDLER_INSTALLED = True

_install_signal_handlers()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="StrategyEngine AI",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Professional CSS Design System
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ---------- Base & Variables ---------- */
:root {
    --bg-dark: #0e1117;
    --bg-content: #fafbfc;
    --accent: #4F8BF9;
    --accent-dark: #3a6fd8;
    --success: #28a745;
    --warning: #f0ad4e;
    --danger: #dc3545;
    --text-primary: #1a1a2e;
    --text-secondary: #6c757d;
    --card-bg: #ffffff;
    --card-border: #e9ecef;
    --radius: 12px;
    --shadow: 0 2px 12px rgba(0,0,0,.08);
    --shadow-hover: 0 4px 20px rgba(0,0,0,.12);
}

/* ---------- Hide Streamlit defaults ---------- */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent;}

/* ---------- Typography ---------- */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0e1117 0%, #161b22 100%);
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li,
section[data-testid="stSidebar"] .stMarkdown label,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4,
section[data-testid="stSidebar"] h5,
section[data-testid="stSidebar"] h6 {
    color: #e6edf3 !important;
}
section[data-testid="stSidebar"] .stTextArea label,
section[data-testid="stSidebar"] .stFileUploader label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stNumberInput label,
section[data-testid="stSidebar"] .stTextInput label,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: #ffffff !important;
}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label p,
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label span {
    color: #8b949e !important;
}

/* ---------- Cards ---------- */
.card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: var(--radius);
    padding: 1.5rem;
    box-shadow: var(--shadow);
    margin-bottom: 1rem;
    transition: box-shadow 0.2s ease;
}
.card:hover {
    box-shadow: var(--shadow-hover);
}
.card-header {
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-secondary);
    margin-bottom: 0.5rem;
}
.card-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--text-primary);
}

/* ---------- Status Badges ---------- */
.badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 50px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.03em;
}
.badge-success { background: #d4edda; color: #155724; }
.badge-progress { background: #cce5ff; color: #004085; }
.badge-warning { background: #fff3cd; color: #856404; }
.badge-error { background: #f8d7da; color: #721c24; }

/* ---------- Pipeline Steps ---------- */
.pipeline-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 0.25rem;
    padding: 1rem 0;
    flex-wrap: wrap;
}
.pipeline-step {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.35rem;
    flex: 1;
    min-width: 80px;
}
.step-icon {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    border: 2px solid #dee2e6;
    background: #f8f9fa;
    color: #adb5bd;
    transition: all 0.3s ease;
}
.step-icon.active {
    border-color: var(--accent);
    background: var(--accent);
    color: white;
    animation: pulse 1.5s infinite;
}
.step-icon.completed {
    border-color: var(--success);
    background: var(--success);
    color: white;
}
.step-label {
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--text-secondary);
    text-align: center;
}
.step-label.active { color: var(--accent); }
.step-label.completed { color: var(--success); }

@keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(79,139,249,.5); }
    70%  { box-shadow: 0 0 0 10px rgba(79,139,249,0); }
    100% { box-shadow: 0 0 0 0 rgba(79,139,249,0); }
}

/* ---------- Activity Log ---------- */
.activity-log {
    background: #1e1e2e;
    border-radius: var(--radius);
    padding: 1rem 1.25rem;
    max-height: 300px;
    overflow-y: auto;
    font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
    font-size: 0.82rem;
    line-height: 1.7;
    color: #cdd6f4;
}
.log-entry { margin-bottom: 0.2rem; }
.log-time { color: #6c7086; }
.log-agent { color: #89b4fa; font-weight: 600; }
.log-ok { color: #a6e3a1; }
.log-warn { color: #f9e2af; }

/* ---------- Metric Pills ---------- */
.metric-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: #f0f4ff;
    border: 1px solid #d0daf5;
    border-radius: 50px;
    padding: 0.3rem 0.85rem;
    font-size: 0.8rem;
    font-weight: 500;
    color: #3a5ba0;
    margin-right: 0.5rem;
    margin-bottom: 0.5rem;
}

/* ---------- Hero Section ---------- */
.hero {
    text-align: center;
    padding: 3rem 1rem 2rem;
}
.hero h1 {
    font-size: 2.5rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
}
.hero-gradient {
    background: linear-gradient(135deg, var(--accent) 0%, #7c3aed 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-subtitle {
    font-size: 1.1rem;
    color: var(--text-secondary);
    line-height: 1.6;
    text-align: center;
}

/* ---------- Feature Cards ---------- */
.feature-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: var(--radius);
    padding: 1.75rem;
    text-align: center;
    box-shadow: var(--shadow);
    height: 100%;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.feature-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-hover);
}
.feature-icon {
    font-size: 2.2rem;
    margin-bottom: 0.75rem;
}
.feature-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
}
.feature-desc {
    font-size: 0.88rem;
    color: var(--text-secondary);
    line-height: 1.5;
}

/* ---------- Steps ---------- */
.steps-container {
    display: flex;
    justify-content: center;
    gap: 2rem;
    margin: 2rem 0;
    flex-wrap: wrap;
}
.step-item {
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    max-width: 220px;
}
.step-number {
    background: var(--accent);
    color: white;
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.9rem;
    flex-shrink: 0;
}
.step-text {
    font-size: 0.9rem;
    color: var(--text-secondary);
    line-height: 1.5;
}
.step-text strong {
    color: var(--text-primary);
    display: block;
    margin-bottom: 0.15rem;
}

/* ---------- Results Banner ---------- */
.result-banner {
    border-radius: var(--radius);
    padding: 1.25rem 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.result-banner.success {
    background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    border: 1px solid #b1dfbb;
}
.result-banner.error {
    background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
    border: 1px solid #f1b0b7;
}
.result-banner-icon { font-size: 1.5rem; }
.result-banner-text {
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--text-primary);
}

/* ---------- Winner Card ---------- */
.winner-card {
    background: linear-gradient(135deg, #f0fff4 0%, #e6ffed 100%);
    border: 2px solid var(--success);
    border-radius: var(--radius);
    padding: 1.25rem 1.5rem;
    margin: 1rem 0;
}

/* ---------- Console Output ---------- */
.console-output {
    background: #1e1e2e;
    border-radius: var(--radius);
    padding: 1rem 1.25rem;
    font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
    font-size: 0.8rem;
    line-height: 1.6;
    color: #cdd6f4;
    max-height: 500px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-wrap: break-word;
}

/* ---------- Download Buttons ---------- */
.stDownloadButton > button {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-dark) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.5rem !important;
    font-weight: 600 !important;
    transition: opacity 0.2s ease !important;
}
.stDownloadButton > button:hover {
    opacity: 0.9 !important;
}

/* ---------- Start Button Override ---------- */
section[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-dark) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    padding: 0.65rem 1rem !important;
    font-size: 1rem !important;
    width: 100% !important;
    transition: opacity 0.2s ease !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    opacity: 0.9 !important;
}

/* ---------- Fade-in animation ---------- */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.fade-in { animation: fadeIn 0.4s ease-out; }

/* ---------- Footer ---------- */
.footer {
    text-align: center;
    padding: 2rem 0 1rem;
    color: var(--text-secondary);
    font-size: 0.78rem;
    border-top: 1px solid var(--card-border);
    margin-top: 3rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div style="padding: 0.5rem 0 0.25rem;">
    <h1 style="margin:0; font-size:2rem; font-weight:800;">
        <span class="hero-gradient">StrategyEngine AI</span>
    </h1>
    <p style="margin:0.25rem 0 0; color:#6c757d; font-size:0.95rem;">
        Plataforma de Inteligencia de Negocio Autonoma &mdash; Multi-Agent AI
    </p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 0.5rem 0 1rem;">
        <span style="font-size:1.6rem; font-weight:800;" class="hero-gradient">StrategyEngine AI</span>
        <br>
        <span style="font-size:0.72rem; color:#8b949e; letter-spacing:0.05em;">v2.0 &bull; Enterprise AI Platform</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### Fuente de Datos")

    data_source = st.radio(
        "Selecciona la fuente de datos",
        ["Archivo Local", "Salesforce", "HubSpot"],
        label_visibility="collapsed",
    )

    uploaded_file = None

    # ---- Archivo Local ----
    if data_source == "Archivo Local":
        uploaded_file = st.file_uploader("Cargar archivo CSV o Excel", type=["csv", "xlsx", "xls"])
        if uploaded_file is not None:
            file_size = uploaded_file.size
            if file_size < 1024:
                size_str = f"{file_size} B"
            elif file_size < 1024 * 1024:
                size_str = f"{file_size / 1024:.1f} KB"
            else:
                size_str = f"{file_size / (1024*1024):.1f} MB"
            st.markdown(f'<span class="metric-pill">{uploaded_file.name} &mdash; {size_str}</span>', unsafe_allow_html=True)

    # ---- Salesforce ----
    elif data_source == "Salesforce":
        sf_auth_mode = st.selectbox("Modo de autenticacion", ["Token API", "OAuth (Access Token)"], key="sf_auth_mode")

        if sf_auth_mode == "Token API":
            sf_username = st.text_input("Username", key="sf_username")
            sf_password = st.text_input("Password", type="password", key="sf_password")
            sf_token = st.text_input("Security Token", type="password", key="sf_security_token")
            sf_connect = st.button("Conectar a Salesforce", key="sf_connect")

            if sf_connect and sf_username and sf_password and sf_token:
                try:
                    from src.connectors.salesforce_connector import SalesforceConnector
                    connector = SalesforceConnector()
                    connector.authenticate({
                        "mode": "token",
                        "username": sf_username,
                        "password": sf_password,
                        "security_token": sf_token,
                    })
                    st.session_state["crm_connector"] = connector
                    st.session_state["crm_authenticated"] = True
                    st.session_state["crm_objects"] = connector.list_objects()
                except Exception as exc:
                    st.error(f"Error: {exc}")
                    st.session_state["crm_authenticated"] = False
        else:
            sf_access_token = st.text_input("Access Token", type="password", key="sf_oauth_token")
            sf_instance_url = st.text_input("Instance URL", placeholder="https://your-instance.salesforce.com", key="sf_instance_url")
            sf_connect_oauth = st.button("Conectar a Salesforce", key="sf_connect_oauth")

            if sf_connect_oauth and sf_access_token and sf_instance_url:
                try:
                    from src.connectors.salesforce_connector import SalesforceConnector
                    connector = SalesforceConnector()
                    connector.authenticate({
                        "mode": "oauth",
                        "access_token": sf_access_token,
                        "instance_url": sf_instance_url,
                    })
                    st.session_state["crm_connector"] = connector
                    st.session_state["crm_authenticated"] = True
                    st.session_state["crm_objects"] = connector.list_objects()
                except Exception as exc:
                    st.error(f"Error: {exc}")
                    st.session_state["crm_authenticated"] = False

        # Object selection & data fetch (shared for both SF auth modes)
        if st.session_state.get("crm_authenticated") and type(st.session_state.get("crm_connector")).__name__ == "SalesforceConnector":
            st.markdown('<span class="badge badge-success">Conectado a Salesforce</span>', unsafe_allow_html=True)
            crm_objects = st.session_state.get("crm_objects", [])
            if crm_objects:
                obj_labels = [f"{o['label']} ({o['name']})" for o in crm_objects]
                selected_idx = st.selectbox("Objeto CRM", range(len(obj_labels)), format_func=lambda i: obj_labels[i], key="sf_obj_select")
                max_recs = st.number_input("Max registros", min_value=100, max_value=50000, value=10000, step=500, key="sf_max_recs")
                fetch_btn = st.button("Extraer Datos", key="sf_fetch")

                if fetch_btn:
                    selected_obj = crm_objects[selected_idx]["name"]
                    try:
                        connector = st.session_state["crm_connector"]
                        df_crm = connector.fetch_object_data(selected_obj, max_records=int(max_recs))
                        if df_crm.empty:
                            st.warning(f"El objeto '{selected_obj}' no contiene registros.")
                        else:
                            os.makedirs("data", exist_ok=True)
                            crm_csv = os.path.join("data", f"crm_{selected_obj.lower()}.csv")
                            df_crm.to_csv(crm_csv, index=False, encoding="utf-8")
                            st.session_state["crm_data_path"] = crm_csv
                            st.session_state["crm_preview_df"] = df_crm
                            st.markdown(f'<span class="metric-pill">{len(df_crm):,} registros extraidos</span>', unsafe_allow_html=True)
                    except Exception as exc:
                        st.error(f"Error al extraer datos: {exc}")

            if st.session_state.get("crm_data_path"):
                preview_df = st.session_state.get("crm_preview_df")
                if preview_df is not None:
                    st.markdown(f'<span class="metric-pill">{len(preview_df):,} registros listos</span>', unsafe_allow_html=True)

    # ---- HubSpot ----
    elif data_source == "HubSpot":
        hs_auth_mode = st.selectbox("Modo de autenticacion", ["Private App Token", "OAuth (Access Token)"], key="hs_auth_mode")
        hs_token = st.text_input("Token", type="password", key="hs_token")
        hs_connect = st.button("Conectar a HubSpot", key="hs_connect")

        if hs_connect and hs_token:
            try:
                from src.connectors.hubspot_connector import HubSpotConnector
                connector = HubSpotConnector()
                connector.authenticate({"access_token": hs_token})
                st.session_state["crm_connector"] = connector
                st.session_state["crm_authenticated"] = True
                st.session_state["crm_objects"] = connector.list_objects()
            except Exception as exc:
                st.error(f"Error: {exc}")
                st.session_state["crm_authenticated"] = False

        if st.session_state.get("crm_authenticated") and type(st.session_state.get("crm_connector")).__name__ == "HubSpotConnector":
            st.markdown('<span class="badge badge-success">Conectado a HubSpot</span>', unsafe_allow_html=True)
            crm_objects = st.session_state.get("crm_objects", [])
            if crm_objects:
                obj_labels = [f"{o['label']} ({o['name']})" for o in crm_objects]
                selected_idx = st.selectbox("Objeto CRM", range(len(obj_labels)), format_func=lambda i: obj_labels[i], key="hs_obj_select")
                max_recs = st.number_input("Max registros", min_value=100, max_value=50000, value=10000, step=500, key="hs_max_recs")
                fetch_btn = st.button("Extraer Datos", key="hs_fetch")

                if fetch_btn:
                    selected_obj = crm_objects[selected_idx]["name"]
                    try:
                        connector = st.session_state["crm_connector"]
                        df_crm = connector.fetch_object_data(selected_obj, max_records=int(max_recs))
                        if df_crm.empty:
                            st.warning(f"El objeto '{selected_obj}' no contiene registros.")
                        else:
                            os.makedirs("data", exist_ok=True)
                            crm_csv = os.path.join("data", f"crm_{selected_obj.lower()}.csv")
                            df_crm.to_csv(crm_csv, index=False, encoding="utf-8")
                            st.session_state["crm_data_path"] = crm_csv
                            st.session_state["crm_preview_df"] = df_crm
                            st.markdown(f'<span class="metric-pill">{len(df_crm):,} registros extraidos</span>', unsafe_allow_html=True)
                    except Exception as exc:
                        st.error(f"Error al extraer datos: {exc}")

            if st.session_state.get("crm_data_path"):
                preview_df = st.session_state.get("crm_preview_df")
                if preview_df is not None:
                    st.markdown(f'<span class="metric-pill">{len(preview_df):,} registros listos</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### Objetivo de Negocio")

    business_objective = st.text_area(
        "Describe el objetivo que deseas lograr con tus datos",
        placeholder="Ej: Reducir el churn de clientes en un 10% identificando los factores clave de abandono...",
        height=130,
        label_visibility="collapsed"
    )

    st.markdown("")  # spacer
    start_btn = st.button("Iniciar Analisis", use_container_width=True)

# ---------------------------------------------------------------------------
# Session State init
# ---------------------------------------------------------------------------
if "analysis_complete" not in st.session_state:
    st.session_state["analysis_complete"] = False
if "analysis_result" not in st.session_state:
    st.session_state["analysis_result"] = None

# ---------------------------------------------------------------------------
# Resolve data_path from any source
# ---------------------------------------------------------------------------
data_path = None

if data_source == "Archivo Local" and uploaded_file is not None:
    os.makedirs("data", exist_ok=True)
    temp_path = os.path.join("data", uploaded_file.name)
    with open(temp_path, "wb") as f:
        uploaded_file.seek(0)
        while chunk := uploaded_file.read(8 * 1024 * 1024):
            f.write(chunk)

    ext = os.path.splitext(uploaded_file.name)[1].lower()
    if ext in ('.xlsx', '.xls'):
        from src.connectors.excel_converter import convert_to_csv
        data_path = convert_to_csv(temp_path)
        st.info(f"Archivo Excel convertido a CSV: {os.path.basename(data_path)}")
    else:
        data_path = temp_path

elif data_source in ("Salesforce", "HubSpot"):
    data_path = st.session_state.get("crm_data_path")

# ---------------------------------------------------------------------------
# Welcome Screen (no data loaded and no results)
# ---------------------------------------------------------------------------
if data_path is None and not st.session_state.get("analysis_complete"):
    st.markdown("""
    <div class="hero fade-in">
        <h1><span class="hero-gradient">Inteligencia de Negocio Autonoma</span></h1>
        <p class="hero-subtitle">
            Sube tus datos, define un objetivo y deja que nuestro equipo de agentes IA
            audite, diseñe estrategias, construya modelos y genere un reporte ejecutivo&nbsp;&mdash; todo en minutos.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        st.markdown("""
        <div class="feature-card fade-in">
            <div class="feature-icon">&#128269;</div>
            <div class="feature-title">Auditoria de Datos</div>
            <div class="feature-desc">Analisis automatico de calidad, integridad y distribucion de tus datos con recomendaciones accionables.</div>
        </div>
        """, unsafe_allow_html=True)
    with col_f2:
        st.markdown("""
        <div class="feature-card fade-in">
            <div class="feature-icon">&#127919;</div>
            <div class="feature-title">Estrategia IA</div>
            <div class="feature-desc">Generacion y evaluacion de multiples estrategias analiticas con deliberacion experta para seleccionar la optima.</div>
        </div>
        """, unsafe_allow_html=True)
    with col_f3:
        st.markdown("""
        <div class="feature-card fade-in">
            <div class="feature-icon">&#9881;&#65039;</div>
            <div class="feature-title">ML Automatizado</div>
            <div class="feature-desc">Ingenieria de datos, entrenamiento de modelos y evaluacion iterativa con generacion de reportes ejecutivos.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="steps-container fade-in">
        <div class="step-item">
            <div class="step-number">1</div>
            <div class="step-text"><strong>Sube tus datos</strong>Carga un archivo CSV/Excel o conecta tu CRM desde el panel lateral.</div>
        </div>
        <div class="step-item">
            <div class="step-number">2</div>
            <div class="step-text"><strong>Define tu objetivo</strong>Describe que quieres lograr con tus datos.</div>
        </div>
        <div class="step-item">
            <div class="step-number">3</div>
            <div class="step-text"><strong>Obtiene resultados</strong>Recibe un reporte ejecutivo con insights accionables.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Unified preview for any data source
# ---------------------------------------------------------------------------
if data_path is not None:
    if not st.session_state["analysis_complete"] and not start_btn:
        try:
            df_preview = pd.read_csv(data_path, nrows=50)
        except Exception:
            df_preview = None

        # Fallback: try CSV with semicolon separator
        if df_preview is None or (df_preview is not None and len(df_preview.columns) <= 1):
            try:
                df_preview = pd.read_csv(data_path, sep=';', nrows=50)
            except Exception:
                df_preview = None

        if df_preview is not None and len(df_preview.columns) > 1:
            n_rows, n_cols = df_preview.shape
            dtypes_summary = df_preview.dtypes.value_counts()
            dtype_parts = [f"{count} {str(dtype)}" for dtype, count in dtypes_summary.items()]

            st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
            st.markdown("#### Vista Previa del Dataset")

            pills_html = (
                f'<span class="metric-pill">{n_rows:,} filas</span>'
                f'<span class="metric-pill">{n_cols} columnas</span>'
            )
            for part in dtype_parts:
                pills_html += f'<span class="metric-pill">{part}</span>'
            st.markdown(pills_html, unsafe_allow_html=True)

            st.dataframe(df_preview.head(10), use_container_width=True, height=300)
            st.markdown('</div>', unsafe_allow_html=True)
        elif df_preview is not None:
            st.warning("El dataset parece tener solo una columna. Verifica el formato del archivo.")

# ---------------------------------------------------------------------------
# Pipeline steps definition (for visual tracker)
# ---------------------------------------------------------------------------
PIPELINE_STEPS = [
    ("steward",        "Steward",    "&#128270;"),
    ("strategist",     "Strategist", "&#129504;"),
    ("domain_expert",  "Expert",     "&#127942;"),
    ("data_engineer",  "Data Eng",   "&#128295;"),
    ("engineer",       "ML Eng",     "&#9881;"),
    ("evaluate_results","Reviewer",  "&#128269;"),
    ("translator",     "Report",     "&#128202;"),
]

def _render_pipeline(completed_steps: set, active_step: str | None = None):
    """Render the visual pipeline tracker."""
    parts = []
    for key, label, icon in PIPELINE_STEPS:
        if key in completed_steps:
            cls_icon = "completed"
            cls_label = "completed"
        elif key == active_step:
            cls_icon = "active"
            cls_label = "active"
        else:
            cls_icon = ""
            cls_label = ""
        parts.append(
            f'<div class="pipeline-step">'
            f'  <div class="step-icon {cls_icon}">{icon}</div>'
            f'  <div class="step-label {cls_label}">{label}</div>'
            f'</div>'
        )
    return '<div class="pipeline-container">' + "".join(parts) + '</div>'

# ---------------------------------------------------------------------------
# Start Analysis
# ---------------------------------------------------------------------------
if start_btn:
    if data_path is None:
        st.sidebar.error("Por favor carga datos: sube un archivo o conecta un CRM.")
    elif not business_objective:
        st.sidebar.error("Por favor define un objetivo de negocio.")
    else:
        st.session_state["analysis_complete"] = False
        st.session_state["analysis_result"] = None
        clear_abort()

        if os.path.exists("static/plots"):
            files = glob.glob("static/plots/*")
            for f in files:
                os.remove(f)

        try:
            # Pipeline visual tracker
            pipeline_placeholder = st.empty()
            log_placeholder = st.empty()

            completed_steps: set = set()
            active_step: str | None = "steward"
            log_entries: list[str] = []

            def add_log(agent: str, message: str, level: str = "info"):
                ts = datetime.now().strftime("%H:%M:%S")
                cls = {"ok": "log-ok", "warn": "log-warn", "info": ""}.get(level, "")
                log_entries.append(
                    f'<div class="log-entry">'
                    f'<span class="log-time">[{ts}]</span> '
                    f'<span class="log-agent">{agent}</span> '
                    f'<span class="{cls}">{message}</span>'
                    f'</div>'
                )

            def refresh_ui():
                pipeline_placeholder.markdown(
                    '<div class="card">'
                    + _render_pipeline(completed_steps, active_step)
                    + '</div>',
                    unsafe_allow_html=True
                )
                log_placeholder.markdown(
                    '<div class="activity-log">' + "\n".join(log_entries) + '</div>',
                    unsafe_allow_html=True
                )

            add_log("Sistema", "Iniciando pipeline de analisis...", "info")
            add_log("Data Steward", "Analizando calidad e integridad de datos...", "info")
            refresh_ui()

            initial_state = {
                "csv_path": data_path,
                "business_objective": business_objective
            }

            final_state = initial_state.copy()

            for event in app_graph.stream(initial_state, config={"recursion_limit": 100}):
                if event is None:
                    continue

                for key, value in event.items():
                    if value is not None:
                        final_state.update(value)

                if 'steward' in event:
                    completed_steps.add("steward")
                    active_step = "strategist"
                    add_log("Data Steward", "Auditoria completada.", "ok")
                    add_log("Strategist", "Diseñando estrategias de alto impacto...", "info")

                elif 'strategist' in event:
                    completed_steps.add("strategist")
                    active_step = "domain_expert"
                    add_log("Strategist", "3 estrategias generadas. Iniciando deliberacion...", "ok")

                elif 'domain_expert' in event:
                    completed_steps.add("domain_expert")
                    active_step = "data_engineer"
                    selected = final_state.get('selected_strategy', {})
                    add_log("Domain Expert", f"Estrategia seleccionada: {selected.get('title', 'N/A')}", "ok")
                    add_log("Data Engineer", "Limpiando y estandarizando dataset...", "info")

                elif 'data_engineer' in event:
                    completed_steps.add("data_engineer")
                    active_step = "engineer"
                    add_log("Data Engineer", "Datos limpiados y estandarizados.", "ok")
                    add_log("ML Engineer", "Optimizando modelo (iteracion en curso)...", "info")

                elif 'engineer' in event:
                    pass

                elif 'execute_code' in event:
                    add_log("ML Engineer", "Ejecucion de codigo finalizada.", "ok")
                    add_log("Reviewer", "Evaluando resultados de negocio...", "info")
                    active_step = "evaluate_results"

                elif 'evaluate_results' in event:
                    verdict = final_state.get('review_verdict', 'APPROVED')
                    if verdict == "NEEDS_IMPROVEMENT":
                        feedback = final_state.get('execution_feedback', '')
                        add_log("Reviewer", f"Resultados insuficientes: {feedback}", "warn")
                        add_log("ML Engineer", "Refinando modelo (retry)...", "info")
                        active_step = "engineer"
                    else:
                        completed_steps.add("engineer")
                        completed_steps.add("evaluate_results")
                        active_step = "translator"
                        add_log("Reviewer", "Resultados aprobados.", "ok")

                elif 'retry_handler' in event:
                    pass

                elif 'translator' in event:
                    completed_steps.add("translator")
                    active_step = None
                    add_log("Translator", "Reporte ejecutivo generado.", "ok")

                elif 'generate_pdf' in event:
                    add_log("Sistema", "PDF final generado.", "ok")

                refresh_ui()

            st.session_state["analysis_result"] = final_state
            st.session_state["analysis_complete"] = True

            # Final pipeline: all complete
            completed_steps = {s[0] for s in PIPELINE_STEPS}
            active_step = None
            add_log("Sistema", "Pipeline completado exitosamente.", "ok")
            refresh_ui()

            time.sleep(0.5)
            st.rerun()

        except Exception as e:
            st.error(f"Ocurrio un error critico: {e}")
            st.exception(e)

# ---------------------------------------------------------------------------
# Results Dashboard
# ---------------------------------------------------------------------------
if st.session_state.get("analysis_complete") and st.session_state.get("analysis_result"):
    result = st.session_state["analysis_result"]

    # Success Banner
    verdict = result.get('review_verdict', 'APPROVED')
    if verdict == "NEEDS_IMPROVEMENT":
        st.markdown("""
        <div class="result-banner error fade-in">
            <div class="result-banner-icon">&#9888;&#65039;</div>
            <div class="result-banner-text">Analisis completado con observaciones del revisor</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-banner success fade-in">
            <div class="result-banner-icon">&#9989;</div>
            <div class="result-banner-text">Analisis completado exitosamente</div>
        </div>
        """, unsafe_allow_html=True)

    # Summary metric cards
    iteration_count = result.get('iteration_count', result.get('current_iteration', 'N/A'))
    selected_strat = result.get('selected_strategy', {})
    strat_title = selected_strat.get('title', 'N/A') if isinstance(selected_strat, dict) else 'N/A'

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.markdown(f"""
        <div class="card fade-in" style="text-align:center;">
            <div class="card-header">Estrategia</div>
            <div style="font-size:1rem; font-weight:700; color:var(--text-primary);">{strat_title}</div>
        </div>
        """, unsafe_allow_html=True)
    with mc2:
        st.markdown(f"""
        <div class="card fade-in" style="text-align:center;">
            <div class="card-header">Iteraciones ML</div>
            <div class="card-value">{iteration_count}</div>
        </div>
        """, unsafe_allow_html=True)
    with mc3:
        rv = result.get('review_verdict', 'N/A')
        badge_cls = "badge-success" if rv == "APPROVED" else "badge-warning"
        st.markdown(f"""
        <div class="card fade-in" style="text-align:center;">
            <div class="card-header">Veredicto</div>
            <div style="margin-top:0.5rem;"><span class="badge {badge_cls}">{rv}</span></div>
        </div>
        """, unsafe_allow_html=True)
    with mc4:
        gate_status = "Pass" if result.get('gate_status', '') == 'PASSED' else result.get('gate_status', 'N/A')
        st.markdown(f"""
        <div class="card fade-in" style="text-align:center;">
            <div class="card-header">Gate Status</div>
            <div style="font-size:1rem; font-weight:700; color:var(--text-primary);">{gate_status}</div>
        </div>
        """, unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab_de, tab3, tab4 = st.tabs([
        "Auditoria de Datos",
        "Estrategia",
        "Ingenieria de Datos",
        "ML Engineer",
        "Reporte Ejecutivo"
    ])

    # --- Tab 1: Data Audit ---
    with tab1:
        st.markdown("#### Auditoria de Datos")
        data_summary = result.get('data_summary', 'No disponible')
        st.markdown(f'<div class="card fade-in">{data_summary}</div>', unsafe_allow_html=True)

    # --- Tab 2: Strategy ---
    with tab2:
        st.markdown("#### Plan Estrategico")
        strategies = result.get('strategies', {})

        if isinstance(strategies, dict) and 'strategies' in strategies:
            for i, strat in enumerate(strategies['strategies'], 1):
                with st.expander(f"Estrategia {i}: {strat.get('title')}", expanded=(i == 1)):
                    st.write(f"**Hipotesis:** {strat.get('hypothesis')}")
                    st.write(f"**Dificultad:** {strat.get('estimated_difficulty')}")
                    st.write(f"**Razonamiento:** {strat.get('reasoning')}")
        else:
            st.json(strategies)

        selected = result.get('selected_strategy', {})
        reviews = result.get('domain_expert_reviews', [])

        if selected:
            st.markdown(f"""
            <div class="winner-card fade-in">
                <strong>Estrategia Ganadora:</strong> {selected.get('title', 'N/A')}<br>
                <span style="color:var(--text-secondary);">{result.get('selection_reason', 'N/A')}</span>
            </div>
            """, unsafe_allow_html=True)

        if reviews:
            st.markdown("#### Deliberacion del Experto")
            for rev in reviews:
                score = rev.get('score', 'N/A')
                badge_cls = "badge-success" if isinstance(score, (int, float)) and score >= 7 else "badge-warning"
                with st.expander(f"{rev.get('title')} — Score: {score}/10"):
                    st.write(f"**Razonamiento:** {rev.get('reasoning')}")
                    st.write(f"**Riesgos:** {rev.get('risks')}")
                    st.write(f"**Recomendacion:** {rev.get('recommendation')}")

    # --- Tab 3: Data Engineering ---
    with tab_de:
        st.markdown("#### Ingenieria de Datos")

        code = result.get('cleaning_code', '# No code available')
        preview = result.get('cleaned_data_preview', 'No preview available')

        col_de_code, col_de_preview = st.columns(2)

        with col_de_code:
            st.markdown("**Script de Limpieza Generado**")
            st.code(code, language='python')

        with col_de_preview:
            st.markdown("**Vista Previa (Cleaned Data)**")
            if isinstance(preview, str) and preview.strip().startswith('{'):
                try:
                    from io import StringIO
                    st.dataframe(pd.read_json(StringIO(preview), orient='split'), use_container_width=True)
                except Exception as e:
                    st.write(f"Cannot render dataframe: {e}")
                    st.write(preview)
            else:
                st.write(preview)

    # --- Tab 4: ML Engineer ---
    with tab3:
        st.markdown("#### ML Engineer")

        col_code, col_out = st.columns(2)

        with col_code:
            st.markdown("**Codigo Generado**")
            ml_code = result.get('generated_code', '# No code')
            if ml_code.strip() == "# Generation Failed":
                ml_code = result.get('last_generated_code', ml_code)
            st.code(ml_code, language='python')

        with col_out:
            st.markdown("**Salida de Consola**")
            ml_output = result.get('execution_output', '')
            last_ok = result.get('last_successful_execution_output')
            if "BUDGET_EXCEEDED" in str(ml_output) and last_ok:
                ml_output = f"{ml_output}\n\n--- Last successful execution output ---\n{last_ok}"
            # Render in dark console style
            import html as html_mod
            escaped_output = html_mod.escape(str(ml_output))
            st.markdown(f'<div class="console-output">{escaped_output}</div>', unsafe_allow_html=True)

    # --- Tab 5: Executive Report ---
    with tab4:
        st.markdown("#### Informe Ejecutivo")
        st.markdown(result.get('final_report', 'No disponible'))

        # Plot gallery
        plots = glob.glob("static/plots/*.png")
        if plots:
            st.markdown("#### Visualizaciones")
            cols = st.columns(min(len(plots), 3))
            for i, plot_path in enumerate(plots):
                with cols[i % len(cols)]:
                    st.image(plot_path, caption=os.path.basename(plot_path), use_container_width=True)

        # Downloads section
        st.markdown("---")
        st.markdown("#### Descargas")
        dl_col1, dl_col2 = st.columns(2)

        # PDF Download
        if 'pdf_binary' not in st.session_state:
            pdf_path = result.get('pdf_path')
            if pdf_path and os.path.exists(pdf_path):
                try:
                    with open(pdf_path, "rb") as pdf_file:
                        st.session_state['pdf_binary'] = pdf_file.read()
                except Exception as e:
                    st.warning(f"Could not reload PDF: {e}")

        with dl_col1:
            if 'pdf_binary' in st.session_state:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M')
                st.download_button(
                    label="Descargar Reporte PDF",
                    data=st.session_state['pdf_binary'],
                    file_name=f"Reporte_Ejecutivo_{timestamp}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

        # ML Artifacts ZIP
        output_report = result.get("output_contract_report")
        if not isinstance(output_report, dict):
            output_report = _load_json("data/output_contract_report.json") or {}
        present_outputs = output_report.get("present", []) if isinstance(output_report, dict) else []
        present_files = [p for p in present_outputs if isinstance(p, str) and os.path.exists(p)]

        with dl_col2:
            if present_files:
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    for file_path in present_files:
                        arcname = os.path.relpath(file_path, start=os.getcwd())
                        zf.write(file_path, arcname=arcname)
                zip_buffer.seek(0)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M')
                st.download_button(
                    label="Descargar Entregables ML (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name=f"Entregables_ML_{timestamp}.zip",
                    mime="application/zip",
                    use_container_width=True
                )
            else:
                st.info("No se encontraron entregables ML en esta ejecucion.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("""
<div class="footer">
    <span>&copy; 2025 StrategyEngine AI &mdash; Powered by Multi-Agent AI</span>
</div>
""", unsafe_allow_html=True)
