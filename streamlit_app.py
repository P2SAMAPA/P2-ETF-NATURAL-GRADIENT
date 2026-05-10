import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px
from huggingface_hub import HfFileSystem
import config
from us_calendar import next_trading_day

st.set_page_config(page_title="Natural Gradient Allocation", layout="wide")
st.title("🧭 Natural Gradient Portfolio Allocation")
st.caption("Fisher‑informed optimisation | Sortino objective | Riemannian geometry")

# ... (keep custom CSS as before) ...

OUTPUT_REPO = config.OUTPUT_REPO
HF_TOKEN = config.HF_TOKEN

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

file_list = list_repo_files()
latest_json_path = find_latest_json(file_list)
if latest_json_path is None:
    st.error("❌ No JSON result file found.")
    st.stop()

data = load_json_from_path(latest_json_path)
if "error" in data:
    st.error(f"Failed to load JSON: {data['error']}")
    st.stop()

if "run_date" not in data or "universes" not in data:
    st.error("JSON missing required keys.")
    st.stop()

st.sidebar.header("ℹ️ Info")
st.sidebar.write(f"**Run date:** {data['run_date']}")
st.sidebar.write(f"**Next trading day:** {next_trading_day()}")

# Mode selector
mode = st.sidebar.radio("Select allocation mode", ["Global", "Last 252 Days"], index=1)

universes = data["universes"]
if not universes:
    st.warning("No universe data found.")
    st.stop()

universe_names = list(universes.keys())
selected = st.selectbox("🌍 Select Universe", universe_names)

if selected:
    uni_data = universes[selected]
    # mode must exist in the dict
    if mode not in uni_data:
        st.error(f"Mode '{mode}' not available for universe '{selected}'. Available: {list(uni_data.keys())}")
        st.stop()
    info = uni_data[mode]

    if "weights" not in info or "top_picks" not in info:
        st.error(f"Missing keys in {mode} data.")
        st.stop()

    weights = info["weights"]
    top_picks = info["top_picks"]

    st.subheader(f"📊 {selected} – {mode} Portfolio")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📈 Number of Assets", len(weights))
    with col2:
        st.metric("🔍 Lookback Days", info.get("lookback_days", "—"))
    with col3:
        st.metric("📅 Training End Date", info.get("training_end_date", data['run_date']))
    with col4:
        top_ticker, top_weight = top_picks[0].values()
        st.metric("🥇 Top ETF", f"{top_ticker} ({top_weight:.1%})")

    st.markdown("### ⭐ Top 3 Recommended ETFs")
    df_top = pd.DataFrame(top_picks)
    st.dataframe(df_top.style.format({"weight": "{:.1%}"}), use_container_width=True, hide_index=True)

    # Pie chart of weights (only non‑zero)
    non_zero = {k: v for k, v in weights.items() if v > 0}
    if non_zero:
        fig = px.pie(names=list(non_zero.keys()), values=list(non_zero.values()), title="Portfolio Weights", hole=0.3)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("All weights are zero (should not happen).")

st.markdown("---")
st.caption(f"Results from {OUTPUT_REPO} | Global uses full history (2008–present); Last 252 Days uses recent year.")
