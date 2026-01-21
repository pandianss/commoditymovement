import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import sys
import datetime
import re
import yfinance as yf

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, COMMODITIES
from core.state_store import StateStore
from core.registry import ModelRegistry
from core.risk import RiskProfile, CapitalConstitution, PortfolioState
from ops.health import HealthMonitor
from news_engine.nlp_processor import NLPProcessor

st.set_page_config(
    layout="wide", 
    page_title="Commodity Sovereignty Command", 
    page_icon="🕋"
)

# --- Premium Obsidian Theme CSS ---
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: radial-gradient(circle at top left, #0d1117, #010409);
        color: #c9d1d9;
    }
    
    /* Custom Card Styling */
    .metric-card {
        background: rgba(22, 27, 34, 0.6);
        border: 1px solid rgba(48, 54, 61, 0.8);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        backdrop-filter: blur(10px);
    }
    
    /* Pulsing Indicators */
    @keyframes pulse-green {
        0% { box-shadow: 0 0 0 0 rgba(46, 160, 67, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(46, 160, 67, 0); }
        100% { box-shadow: 0 0 0 0 rgba(46, 160, 67, 0); }
    }
    @keyframes pulse-red {
        0% { box-shadow: 0 0 0 0 rgba(248, 81, 73, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(248, 81, 73, 0); }
        100% { box-shadow: 0 0 0 0 rgba(248, 81, 73, 0); }
    }
    
    .status-indicator {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
    }
    .status-green { background-color: #2ea043; animation: pulse-green 2s infinite; }
    .status-red { background-color: #f85149; animation: pulse-red 2s infinite; }
    .status-gray { background-color: #484f58; }

    /* Header styling */
    h1, h2, h3 {
        color: #58a6ff !important;
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

st.title("🕋 Commodity Sovereignty")
st.caption("Active Governance Layer: v2.0-Alpha | System Status: DETERMINISTIC")

# --- Refresh Rate ---
if st.button('Refresh Data'):
    st.cache_data.clear()
    st.rerun()

# --- Helper: Unit Conversion ---
def get_conversion_info(ticker_key):
    """
    Returns (unit_label, conversion_factor_func)
    """
    # Factors
    # 1 oz = 31.1035 g
    # 10g Gold = (Price / 31.1035) * 10 = Price * 0.321507
    # 1kg Silver = (Price / 31.1035) * 1000 = Price * 32.1507
    # 1 lb Copper = 0.4536 kg -> 1kg = 2.20462 lbs. Price (USD/lb) * 2.20462 = USD/kg.
    
    if ticker_key == "GOLD":
        return "INR / 1g", lambda p, r: p * r * (1 / 31.1034768)
    elif ticker_key == "SILVER":
        return "INR / 1kg", lambda p, r: p * r * (1000 / 31.1034768)
    elif ticker_key == "COPPER":
        return "INR / 1kg", lambda p, r: p * r * 2.20462
    elif ticker_key == "CRUDE_OIL":
        return "INR / bbl", lambda p, r: p * r
    elif ticker_key == "NATURAL_GAS":
        return "INR / MMBtu", lambda p, r: p * r
    else:
        return "INR (derived)", lambda p, r: p * r

# --- Load Data ---
@st.cache_data(ttl=300)
def fetch_usdinr():
    try:
        data = yf.download("USDINR=X", period="1d", interval="1h", progress=False)
        if not data.empty:
            # Get latest close
            return float(data['Close'].iloc[-1].item())
    except Exception as e:
        print(f"Error fetching USDINR: {e}")
    return 84.0 # Fallback

@st.cache_data(ttl=60)
def load_data():
    # 1h Intraday Data
    prices_path = os.path.join(RAW_DATA_DIR, "commodities_1h_raw.csv")
    if os.path.exists(prices_path):
        try:
            # Handle multi-index header if present (yfinance often saves with Ticker as header)
            # We'll read without header first to sniff, or just assume format
            df = pd.read_csv(prices_path, header=[0, 1], index_col=0, parse_dates=True)
            return df
        except:
            return None
    return None

def parse_live_logs():
    log_path = os.path.join("logs", "live_intel.log")
    if not os.path.exists(log_path):
        return [], []
    
    predictions = []
    shocks = []
    
    with open(log_path, "r") as f:
        lines = f.readlines()
        
    for line in reversed(lines):
        if "PREDICTION" in line:
            match = re.search(r"Med=([-\d\.]+)% \| Range=\[([-\d\.]+)%, ([-\d\.]+)%\]", line)
            if match:
                predictions.append({
                    "timestamp": line.split(" [INFO]")[0],
                    "median": float(match.group(1)),
                    "lower": float(match.group(2)),
                    "upper": float(match.group(3))
                })
        
        if "SHOCK DETECTED" in line:
            parts = line.split("SHOCK DETECTED: ")
            if len(parts) > 1:
                shocks.append({
                    "timestamp": line.split(" [INFO]")[0],
                    "msg": parts[1].strip()
                })
                
    return predictions, shocks

# --- Initialized Components ---
@st.cache_resource
def get_governance_core():
    return {
        "state": StateStore("state/run_state.json"),
        "registry": ModelRegistry(),
        "health": HealthMonitor(),
        "nlp": NLPProcessor()
    }

core = get_governance_core()
usdinr = fetch_usdinr()

# --- Sidebar ---
st.sidebar.header("Asset Selection")
selected_ticker_key = st.sidebar.selectbox("Select Mandate", list(COMMODITIES.keys()))
ticker_symbol = COMMODITIES[selected_ticker_key]
st.sidebar.metric("USD/INR Spot", f"₹{usdinr:.2f}")

# --- Pillar 1: System Sovereignty (Executive Dashboard) ---
top_col1, top_col2, top_col3, top_col4 = st.columns(4)

with top_col1:
    # Health Monitoring
    status = core["health"].get_system_status()
    is_alive = status.get("status") == "alive"
    pulse_class = "status-green" if is_alive else "status-red"
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 0.8em; color: #8b949e;">SYSTEM HEARTBEAT</div>
        <div style="margin-top: 5px;">
            <span class="status-indicator {pulse_class}"></span>
            <span style="font-weight: bold;">{"OPERATIONAL" if is_alive else "STALLED"}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with top_col2:
    # Model Registry
    champion = core["registry"].get_champion("tcn_gold")
    champ_id = champion.get("model_id", "N/A") if champion else "NONE"
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 0.8em; color: #8b949e;">ACTIVE CHAMPION</div>
        <div style="margin-top: 5px; font-weight: bold; color: #d2a8ff;">{champ_id}</div>
    </div>
    """, unsafe_allow_html=True)

with top_col3:
    # Risk compliance
    last_processed = core["state"].get("last_successful_cycle")
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 0.8em; color: #8b949e;">LAST INTELLIGENCE CYCLE</div>
        <div style="margin-top: 5px; font-weight: bold;">{last_processed or "BOOTING..."}</div>
    </div>
    """, unsafe_allow_html=True)

with top_col4:
    # Risk Metric
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 0.8em; color: #8b949e;">RISK MANDATE</div>
        <div style="margin-top: 5px; font-weight: bold; color: #ff7b72;">MAX DD: 20%</div>
    </div>
    """, unsafe_allow_html=True)

# --- Pillar 2 & 3: Market Analysis & Forecasting ---
main_col, side_col = st.columns([2.5, 1])

with main_col:
    prices_df = load_data()
    unit_label, conv_func = get_conversion_info(selected_ticker_key)
    
    if prices_df is not None:
        try:
            p_cols = prices_df.xs(ticker_symbol, level=1, axis=1)
            series_usd = p_cols["Close"] if "Close" in p_cols.columns else p_cols.iloc[:, 0]
            series_inr = series_usd.apply(lambda x: conv_func(x, usdinr))
            
            # Trend Detection for Pulsing Lights
            returns_5d = series_inr.pct_change(5).iloc[-1]
            trend_class = "status-green" if returns_5d > 0.02 else ("status-red" if returns_5d < -0.02 else "status-gray")
            trend_label = "UPTREND" if returns_5d > 0.02 else ("DOWNTREND" if returns_5d < -0.02 else "STABLE")

            st.markdown(f"""
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <span class="status-indicator {trend_class}"></span>
                    <h3 style="margin: 0;">{selected_ticker_key} {trend_label} (5D: {returns_5d:.2%})</h3>
                </div>
            """, unsafe_allow_html=True)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=series_inr.index, y=series_inr.values, mode='lines', name='Price', line=dict(color='#58a6ff', width=2)))
            fig.update_layout(height=400, template="plotly_dark", margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
            st.metric(f"Current Spot ({unit_label})", f"₹{series_inr.iloc[-1]:,.2f}")
            
        except Exception as e:
            st.error(f"Render Error: {e}")
    else:
        st.warning("Awaiting market data stream...")

with side_col:
    preds, shocks = parse_live_logs()
    
    st.subheader("🔮 Probability Cone")
    if preds:
        latest = preds[0]
        med_ret = latest["median"]
        latest_price = series_inr.iloc[-1]
        proj_price = latest_price * (1 + med_ret)
        
        # Color based on bullish/bearish
        card_border = "#2ea043" if med_ret > 0.005 else ("#f85149" if med_ret < -0.005 else "#30363d")
        
        st.markdown(f"""
        <div class="metric-card" style="border-left: 5px solid {card_border};">
            <div style="font-size: 1.2em; font-weight: bold;">₹{proj_price:,.2f}</div>
            <div style="color: #8b949e;">Target (24H)</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.write(f"Confidence Bound: [₹{latest_price*(1+latest['lower']):,.0f}, ₹{latest_price*(1+latest['upper']):,.0f}]")
    
    st.subheader("📰 Sentiment Alpha")
    # Fetch live news from CSV if available
    news_path = os.path.join(RAW_DATA_DIR, "live_news_feed.csv")
    if os.path.exists(news_path):
        news_df = pd.read_csv(news_path).tail(5)
        for _, row in news_df.iterrows():
            sentiment = core["nlp"].get_sentiment(row['headline'])
            s_color = "#2ea043" if sentiment > 0.3 else ("#f85149" if sentiment < -0.3 else "#8b949e")
            st.markdown(f"""
                <div style="font-size: 0.85em; border-bottom: 1px solid #30363d; padding: 5px 0;">
                    <span style="color: {s_color}; font-weight: bold;">[{sentiment:+.1f}]</span> 
                    {row['headline'][:60]}...
                </div>
            """, unsafe_allow_html=True)
    else:
        st.caption("No live news feed detected.")

    st.subheader("⚡ Signal Matrix")
    if shocks:
        for s in shocks[:2]:
            with st.expander(f"EVENT: {s['timestamp'][:16]}", expanded=True):
                st.write(s['msg'])
    else:
        st.info("No active anomalies.")
