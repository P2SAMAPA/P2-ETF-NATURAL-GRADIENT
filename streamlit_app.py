import streamlit as st
import pandas as pd
import json
import plotly.express as px
from huggingface_hub import HfFileSystem
import config
from us_calendar import next_trading_day

st.set_page_config(page_title="Natural Gradient Allocation", layout="wide")
st.title("🧭 Natural Gradient Portfolio Allocation")
st.caption("Fisher‑informed optimisation | Sortino objective | Riemannian geometry")

# Show token status
token = config.HF_TOKEN
if token:
    st.sidebar.success("HF_TOKEN is set")
else:
    st.sidebar.error("HF_TOKEN is NOT set – add it in Streamlit secrets")

OUTPUT_REPO = config.OUTPUT_REPO

@st.cache_data(ttl=3600)
def list_repo_files():
    fs = HfFileSystem(token=token)
    try:
        # If token is None, it will try public access only
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
    fs = HfFileSystem(token=token)
    try:
        with fs.open(full_path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}

# Sidebar debug
st.sidebar.subheader("Debug Info")
file_list = list_repo_files()
st.sidebar.write(f"Files in repo: {len(file_list)}")
for f in file_list[:5]:
    st.sidebar.code(f)

latest_json = find_latest_json(file_list)
if latest_json is None:
    st.sidebar.error("No JSON file found")
    st.error("No result JSON found. Run trainer first.")
    st.stop()
else:
    st.sidebar.success(f"Latest: {latest_json}")

data = load_json_from_path(latest_json)
if "error" in data:
    st.error(f"Failed to load JSON: {data['error']}")
    st.stop()

st.sidebar.write(f"Run date: {data.get('run_date', '?')}")
st.sidebar.write(f"Next trading day: {next_trading_day()}")

universes = data.get("universes", {})
if not universes:
    st.error("No 'universes' key in JSON")
    st.json(data)
    st.stop()

# Mode selector: based on available keys in first universe
first_universe = next(iter(universes.values()))
available_modes = [k for k in first_universe.keys() if k in ['global', 'last_252']]
if not available_modes:
    st.error(f"Unexpected keys: {list(first_universe.keys())}")
    st.stop()
mode = st.sidebar.radio("Allocation mode", available_modes, index=0 if 'global' in available_modes else 0)

universe_names = list(universes.keys())
selected = st.selectbox("Select Universe", universe_names)

if selected:
    uni_data = universes[selected]
    if mode not in uni_data:
        st.error(f"Mode '{mode}' not available for {selected}")
    else:
        info = uni_data[mode]
        weights = info.get("weights", {})
        top_picks = info.get("top_picks", [])
        if not weights:
            st.warning("No weights found")
        else:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Assets", len(weights))
            col2.metric("Lookback", info.get("lookback_days", "?"))
            col3.metric("Top ETF", top_picks[0]['ticker'] if top_picks else "?")
            col4.metric("Top Weight", f"{top_picks[0]['weight']:.1%}" if top_picks else "?")

            st.subheader("Top 3 ETFs")
            df_top = pd.DataFrame(top_picks)
            st.dataframe(df_top.style.format({"weight": "{:.1%}"}), hide_index=True, use_container_width=True)

            non_zero = {k: v for k, v in weights.items() if v > 0}
            if non_zero:
                fig = px.pie(names=list(non_zero.keys()), values=list(non_zero.values()), title="Portfolio Allocation", hole=0.3)
                st.plotly_chart(fig, use_container_width=True)

st.caption(f"Data from {OUTPUT_REPO}")
