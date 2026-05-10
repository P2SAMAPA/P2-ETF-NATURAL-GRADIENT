import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px
from huggingface_hub import HfFileSystem
import config
from us_calendar import next_trading_day

# ========== Page configuration ==========
st.set_page_config(
    page_title="Natural Gradient Portfolio Allocation",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== Custom CSS for larger fonts and professional look ==========
st.markdown("""
<style>
    /* Main headers */
    .main-header {
        font-size: 2.8rem !important;
        font-weight: 700 !important;
        color: #1E3A8A !important;
        margin-bottom: 0.5rem !important;
    }
    .sub-header {
        font-size: 1.2rem !important;
        color: #4B5563 !important;
        margin-bottom: 1.5rem !important;
    }
    /* Metric cards */
    .stMetric {
        background-color: #F3F4F6;
        border-radius: 0.75rem;
        padding: 0.75rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .stMetric label {
        font-size: 1rem !important;
        font-weight: 500 !important;
    }
    .stMetric .metric-value {
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    /* Dataframe font */
    .stDataFrame {
        font-size: 1rem !important;
    }
    /* Sidebar */
    .css-1d391kg {
        font-size: 1rem !important;
    }
    /* Buttons */
    .stButton button {
        font-size: 1rem !important;
        font-weight: 500 !important;
        border-radius: 0.5rem !important;
    }
    /* Expander */
    .streamlit-expanderHeader {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# ========== Header ==========
st.markdown('<div class="main-header">🧭 Natural Gradient Portfolio Allocation</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Fisher‑informed optimisation | Sortino objective | Riemannian geometry</div>', unsafe_allow_html=True)

OUTPUT_REPO = config.OUTPUT_REPO
HF_TOKEN = config.HF_TOKEN

# ========== Helper functions ==========
@st.cache_data(ttl=3600)
def list_repo_files():
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        all_files = [f['name'] for f in fs.ls(f"datasets/{OUTPUT_REPO}", detail=True, recursive=True) if f['type'] == 'file']
        return all_files
    except Exception as e:
        return [f"Error: {e}"]

def find_latest_json(file_list):
    json_files = [f for f in file_list if f.endswith('.json') and 'natural_gradient' in f]
    if not json_files:
        return None
    json_files.sort(reverse=True)
    return json_files[0]

@st.cache_data(ttl=3600)
def load_json_from_path(full_path):
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        with fs.open(full_path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}

# ========== Load data ==========
file_list = list_repo_files()
latest_json_path = find_latest_json(file_list)

if latest_json_path is None:
    st.error("❌ No JSON result file found in the Hugging Face dataset. Please run the trainer first.")
    st.stop()

data = load_json_from_path(latest_json_path)
if "error" in data:
    st.error(f"❌ Failed to load JSON: {data['error']}")
    st.stop()

if "run_date" not in data or "universes" not in data:
    st.error("❌ JSON does not contain expected keys ('run_date', 'universes').")
    st.json(data)
    st.stop()

# ========== Sidebar ==========
st.sidebar.markdown("## ℹ️ Information")
st.sidebar.markdown(f"**Run date:**  \n`{data['run_date']}`")
st.sidebar.markdown(f"**Next trading day:**  \n`{next_trading_day()}`")
st.sidebar.markdown(f"**Method:**  \nNatural gradient (Fisher matrix) on probability simplex")
st.sidebar.markdown(f"**Data source:**  \n`{config.DATA_REPO}`")
st.sidebar.markdown(f"**Results repo:**  \n`{OUTPUT_REPO}`")

universes = data["universes"]
if not universes:
    st.warning("No universe data found in the JSON.")
    st.stop()

universe_names = list(universes.keys())
selected = st.selectbox("🌍 Select Universe", universe_names, help="Choose the ETF universe to analyse")

# ========== Main content ==========
if selected:
    info = universes[selected]
    if "weights" not in info or "top_picks" not in info:
        st.error(f"Universe '{selected}' missing required keys. Structure: {list(info.keys())}")
        st.stop()

    weights = info["weights"]
    top_picks = info["top_picks"]

    # ---- Hero cards ----
    st.markdown(f"## 📊 {selected} – Optimal Portfolio")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📈 Number of Assets", len(weights))
    with col2:
        st.metric("🔍 Lookback Days", info.get("lookback_days", "—"))
    with col3:
        st.metric("📅 Training End Date", info.get("training_end_date", "—"))
    with col4:
        # Display top ETF weight
        top_ticker, top_weight = top_picks[0].values()
        st.metric("🥇 Top ETF", f"{top_ticker} ({top_weight:.1%})")

    # ---- Top picks table ----
    st.markdown("### ⭐ Top 3 Recommended ETFs")
    df_top = pd.DataFrame(top_picks)
    # Style the dataframe
    st.dataframe(
        df_top.style.format({"weight": "{:.1%}"}).set_properties(**{"font-size": "16px"}),
        use_container_width=True,
        hide_index=True
    )

    # ---- Allocation pie chart ----
    st.markdown("### 🥧 Portfolio Allocation (Top 10)")
    sorted_w = sorted(weights.items(), key=lambda x: -x[1])
    top10 = dict(sorted_w[:10])
    other = sum(v for _, v in sorted_w[10:])
    if other > 0:
        top10["Others"] = other
    fig_pie = px.pie(
        names=list(top10.keys()),
        values=list(top10.values()),
        title=None,
        hole=0.3,
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_pie.update_layout(height=500, font=dict(size=14))
    st.plotly_chart(fig_pie, use_container_width=True)

    # ---- Full allocation bar chart ----
    st.markdown("### 📊 Full Asset Allocation")
    df_w = pd.DataFrame([{"Asset": k, "Weight": v} for k, v in weights.items()])
    df_w = df_w.sort_values("Weight", ascending=False)
    fig_bar = px.bar(
        df_w,
        x="Asset",
        y="Weight",
        title=None,
        labels={"Weight": "Portfolio Weight", "Asset": "ETF Ticker"},
        color="Weight",
        color_continuous_scale="Blues"
    )
    fig_bar.update_layout(height=500, font=dict(size=14), xaxis_tickangle=-45)
    st.plotly_chart(fig_bar, use_container_width=True)

# ========== Footer ==========
st.markdown("---")
st.markdown(
    f"<div style='text-align: center; color: #6B7280; font-size: 0.8rem;'>"
    f"Optimises Sortino ratio using natural gradient (Fisher matrix). "
    f"Data from {config.DATA_REPO} | Results stored in {OUTPUT_REPO}"
    f"</div>",
    unsafe_allow_html=True
)
