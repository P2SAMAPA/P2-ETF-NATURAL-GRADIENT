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

OUTPUT_REPO = config.OUTPUT_REPO
HF_TOKEN = config.HF_TOKEN

# Debug: list all files in the repo
@st.cache_data(ttl=3600)
def list_repo_files():
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        all_files = [f['name'] for f in fs.ls(f"datasets/{OUTPUT_REPO}", detail=True, recursive=True) if f['type'] == 'file']
        return all_files
    except Exception as e:
        return [f"Error: {e}"]

file_list = list_repo_files()
st.sidebar.subheader("Debug: Files in HF dataset")
for f in file_list:
    st.sidebar.code(f, language="text")

# Find the latest JSON file (by filename)
def find_latest_json():
    json_files = [f for f in file_list if f.endswith('.json') and 'natural_gradient' in f]
    if not json_files:
        return None
    # Sort by filename (date part) descending
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

latest_json_path = find_latest_json()
if latest_json_path is None:
    st.error("No JSON result file found in the HF dataset. Please run trainer.py first.")
    st.stop()

st.sidebar.success(f"Latest result: {latest_json_path}")
data = load_json_from_path(latest_json_path)
if "error" in data:
    st.error(f"Failed to load JSON: {data['error']}")
    st.stop()

if "run_date" not in data or "universes" not in data:
    st.error("JSON does not contain expected keys ('run_date', 'universes'). Raw content:")
    st.json(data)
    st.stop()

st.sidebar.header("ℹ️ Info")
st.sidebar.write(f"**Run date:** {data['run_date']}")
st.sidebar.write(f"**Next trading day:** {next_trading_day()}")
st.sidebar.write("**Method:** Natural gradient (Fisher matrix) on Simplex")

universes = data["universes"]
if not universes:
    st.warning("No universe data found in the JSON.")
    st.stop()

universe_names = list(universes.keys())
selected = st.selectbox("Select Universe", universe_names)

if selected:
    info = universes[selected]
    # Check expected keys
    if "weights" not in info or "top_picks" not in info:
        st.error(f"Universe '{selected}' missing 'weights' or 'top_picks'. Structure: {list(info.keys())}")
        st.stop()
    weights = info["weights"]
    top = info["top_picks"]

    st.subheader(f"Recommended Portfolio for {selected}")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.metric("Number of Assets", len(weights))
        st.metric("Lookback days", info.get("lookback_days", "?"))
        st.metric("Training end", info.get("training_end_date", "?"))
    with col2:
        df_top = pd.DataFrame(top)
        st.dataframe(df_top, hide_index=True, use_container_width=True)

    # Pie chart of weights (top 10)
    sorted_w = sorted(weights.items(), key=lambda x: -x[1])
    top10 = dict(sorted_w[:10])
    other = sum(v for _, v in sorted_w[10:])
    if other > 0:
        top10["Others"] = other
    fig = px.pie(names=list(top10.keys()), values=list(top10.values()), title="Portfolio Weights")
    st.plotly_chart(fig, use_container_width=True)

    # Bar chart of all weights
    df_w = pd.DataFrame([{"Asset": k, "Weight": v} for k, v in weights.items()])
    fig_bar = px.bar(df_w, x="Asset", y="Weight", title="Full Allocation")
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")
st.caption(f"Data: {OUTPUT_REPO} | Optimises Sortino ratio using natural gradient")
