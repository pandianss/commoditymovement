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

from config import RAW_DATA_DIR, COMMODITIES

st.set_page_config(layout="wide", page_title="Commodity Intelligence (INR)")

st.title("Commodity Intelligence (Indian Standards)")

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

# --- Main App ---

# 1. Sidebar Config
st.sidebar.header("Configuration")
selected_ticker_key = st.sidebar.selectbox("Select Asset", list(COMMODITIES.keys()))
ticker_symbol = COMMODITIES[selected_ticker_key]

usdinr = fetch_usdinr()
st.sidebar.metric("USD/INR Rate", f"₹{usdinr:.2f}")

# 2. Data Processing
prices_df = load_data()
unit_label, conv_func = get_conversion_info(selected_ticker_key)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"Price Action: {selected_ticker_key} ({unit_label})")
    
    if prices_df is not None:
        try:
            # Extract specific ticker series
            # Schema: Level 0 = Price Type, Level 1 = Ticker
            p_cols = prices_df.xs(ticker_symbol, level=1, axis=1)
            
            if "Close" in p_cols.columns:
                series_usd = p_cols["Close"]
            elif "Adj Close" in p_cols.columns:
                series_usd = p_cols["Adj Close"]
            else:
                series_usd = p_cols.iloc[:, 0]
            
            # CONVERT TO INR
            series_inr = series_usd.apply(lambda x: conv_func(x, usdinr))
            
            # Plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=series_inr.index, 
                y=series_inr.values, 
                mode='lines', 
                name=f'Price ({unit_label})',
                line=dict(color='#00CC96')
            ))
            
            fig.update_layout(
                height=450, 
                margin=dict(l=0, r=0, t=10, b=0),
                yaxis_title=unit_label,
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Latest Stats
            latest_price = series_inr.iloc[-1]
            st.metric(f"Current Price ({unit_label})", f"₹{latest_price:,.2f}")
            
        except Exception as e:
            st.error(f"Error processing data for {ticker_symbol}: {e}")
            st.dataframe(prices_df.head()) # Debug
    else:
        st.warning("Waiting for data stream...")

with col2:
    preds, shocks = parse_live_logs()
    
    st.subheader("AI Forecast (TCN)")
    if preds:
        latest = preds[0]
        med_ret = latest["median"]
        
        # Calculate Target Price in INR
        if prices_df is not None:
            # Simple approximation: Current INR Price * (1 + predicted_return)
            # Note: This return is typically 1-day log return or simple return depending on training
            # Assuming simple return from the log msg format
            proj_price = latest_price * (1 + med_ret)
            delta = proj_price - latest_price
            
            st.metric("Proj. Target (1D)", f"₹{proj_price:,.2f}", f"{delta:,.2f} ({med_ret:.2%} in USD)")
            
            if med_ret > 0.01:
                st.success("SIGNAL: BULLISH")
            elif med_ret < -0.01:
                st.error("SIGNAL: BEARISH")
            else:
                st.info("SIGNAL: NEUTRAL")
                
            st.progress(min(max(med_ret * 5 + 0.5, 0.0), 1.0)) # Visual gauge centered at 0
            
    else:
        st.write("Initializing model...")

    st.subheader("Market Shocks")
    if shocks:
        for s in shocks[:5]:
            st.caption(f"{s['timestamp']}")
            st.write(f"⚡ {s['msg']}")
    else:
        st.write("No major anomalies.")
