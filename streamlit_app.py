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

@st.cache_data(ttl=3600)
def load_latest():
    fs = HfFileSystem(token=config.HF_TOKEN)
    try:
        files = fs.ls(f"datasets/{config.OUTPUT_REPO}")
        json_files = [f for f in files if f.endswith('.json')]
        if not json_files:
            return None
        latest = max(json_files)
        with fs.open(latest, "r") as fp:
            return json.load(fp)
    except:
        return None

data = load_latest()
if not data:
    st.warning("No results found. Run trainer.py first.")
    st.stop()

st.sidebar.header("ℹ️ Info")
st.sidebar.write(f"**Run date:** {data['run_date']}")
st.sidebar.write(f"**Next trading day:** {next_trading_day()}")
st.sidebar.write("**Method:** Natural gradient (Fisher matrix) on Simplex")

universes = data["universes"]
universe_names = list(universes.keys())
selected = st.selectbox("Select Universe", universe_names)

if selected:
    info = universes[selected]
    weights = info["weights"]
    top = info["top_picks"]

    # Hero card
    st.subheader(f"Recommended Portfolio for {selected}")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.metric("Number of Assets", len(weights))
        st.metric("Lookback days", info["lookback_days"])
        st.metric("Training end", info["training_end_date"])
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
st.caption(f"Data: {config.OUTPUT_REPO} | Optimises Sortino ratio using natural gradient")
