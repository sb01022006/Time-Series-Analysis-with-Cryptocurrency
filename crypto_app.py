import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from textblob import TextBlob
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import warnings
import time
import requests
from streamlit_lottie import st_lottie
import json
import os
from streamlit_autorefresh import st_autorefresh

# Suppress warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION & PAGE SETUP ---
st.set_page_config(
    page_title="Crypto Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="⚡"
)

# --- AUTO-REFRESH (LIVE MODE) ---
# Refresh the page every 60 seconds (60000ms) to pull new data
count = st_autorefresh(interval=60000, key="data_refresh")

# --- ANIMATION ASSET LOADER ---
@st.cache_data(show_spinner=False)
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=3)
        if r.status_code == 200:
            return r.json()
    except:
        return None
    return None

# Load Assets
lottie_crypto = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_jm7555b9.json") 
lottie_login = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_ktwnwv5m.json") 

# --- ADVANCED CUSTOM CSS ---
st.markdown("""
<style>
    /* 1. Main Background & Fonts */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(14, 17, 23) 0%, rgb(20, 25, 35) 90%);
    }
    
    /* 2. TEXTURE & INTERACTIVE HEADINGS (NEW) */
    h1, h2, h3 {
        font-family: 'Segoe UI', sans-serif;
        font-weight: 800;
        /* Metallic Texture Gradient */
        background: -webkit-linear-gradient(45deg, #eee, #94a3b8, #eee);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        transition: transform 0.3s ease, letter-spacing 0.3s ease;
        cursor: default;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Interactive Hover State for Headings */
    h1:hover, h2:hover, h3:hover {
        transform: scale(1.02);
        letter-spacing: 1px;
        /* Shift to Brand Colors on Hover */
        background: -webkit-linear-gradient(45deg, #3B82F6, #10B981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* 3. Gradient Text Helper Class */
    .gradient-text {
        font-weight: bold;
        background: linear-gradient(45deg, #10B981, #3B82F6, #8B5CF6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3em;
        padding-bottom: 10px;
    }
    
    /* 4. Ticker Tape Animation */
    .ticker-wrap {
        width: 100%;
        overflow: hidden;
        background-color: #1F2937;
        padding-top: 10px;
        padding-bottom: 10px;
        white-space: nowrap;
        border-bottom: 1px solid #374151;
        margin-bottom: 20px;
    }
    .ticker {
        display: inline-block;
        animation: marquee 30s linear infinite;
    }
    .ticker-item {
        display: inline-block;
        padding: 0 2rem;
        font-size: 1rem;
        color: #10B981;
    }
    @keyframes marquee {
        0% { transform: translateX(100%); }
        100% { transform: translateX(-100%); }
    }

    /* 5. Metric Cards with Hover Effect */
    div[data-testid="stMetric"] {
        background-color: rgba(31, 41, 55, 0.6);
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: transform 0.2s ease-in-out, box-shadow 0.2s;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.3);
        border-color: #3B82F6;
    }
    
    /* 6. Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #30363D;
    }
    
    /* 7. Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 4px;
        color: #8b949e;
        border: 1px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f6feb;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- PERSISTENT DATABASE SYSTEM (JSON) ---
DB_FILE = "users_db.json"

def load_users_from_db():
    """Load users from the local JSON file."""
    if not os.path.exists(DB_FILE):
        default_data = {"admin": "password123", "syamantak06": "1234"}
        with open(DB_FILE, 'w') as f:
            json.dump(default_data, f)
        return default_data
    try:
        with open(DB_FILE, 'r') as f:
            return json.load(f)
    except:
        return {} 

def save_user_to_db(username, password):
    """Save a new user to the local JSON file."""
    users = load_users_from_db()
    users[username] = password
    with open(DB_FILE, 'w') as f:
        json.dump(users, f)

# --- SESSION STATE SETUP ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if 'username' not in st.session_state:
    st.session_state['username'] = ""

# --- HELPER: PLOTLY THEME ---
def style_chart(fig, title="", height=500):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(text=title, font=dict(size=18, color="#E5E7EB")),
        font=dict(color="#9CA3AF"),
        height=height,
        xaxis=dict(gridcolor="#374151", showgrid=True),
        yaxis=dict(gridcolor="#374151", showgrid=True),
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode="x unified"
    )
    return fig

# --- DATA LOADER ---
@st.cache_data(ttl=60) # Cache clears every 60s for new data
def load_data(ticker, start, end):
    try:
        # Use a spinner only on initial load or manual refresh
        df = yf.download(ticker, start=start, end=end, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.reset_index(inplace=True)
        
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        return pd.DataFrame()

# =========================================================
#  SIDEBAR (INCLUDES AUTH SYSTEM & NAVIGATION)
# =========================================================
with st.sidebar:
    if lottie_crypto:
        st_lottie(lottie_crypto, height=120, key="sidebar_anim")
    
    # --- NEON ANIMATION ---
    st.markdown("""
    <style>
    @keyframes gradient-text { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
    .neon-title { font-family: 'Segoe UI', sans-serif; font-size: 2.5rem; font-weight: 800; text-align: center; background: linear-gradient(270deg, #ff00cc, #333399, #00bfff, #ff00cc); background-size: 300% 300%; -webkit-background-clip: text; -webkit-text-fill-color: transparent; animation: gradient-text 4s ease infinite; text-shadow: 0 0 10px rgba(0, 191, 255, 0.3); margin-bottom: 0px; }
    .neon-subtitle { text-align: center; color: #a1a1aa; font-size: 0.8rem; margin-top: -5px; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 20px; }
    </style>
    <div class="neon-title">CryptoAI</div>
    <div class="neon-subtitle">Neural Engine v2.5</div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")

    # --- AUTH SYSTEM IN SIDEBAR ---
    if st.session_state['logged_in']:
        st.success(f"🟢 Authenticated: **{st.session_state['username']}**")
        if st.button("🔒 Logout", use_container_width=True):
            st.session_state['logged_in'] = False
            st.rerun()
    else:
        with st.expander("👤 **User Access** (Login/Register)", expanded=False):
            tab_login, tab_register = st.tabs(["Login", "Register"])
            
            with tab_login:
                with st.form("sidebar_login"):
                    user_in = st.text_input("Username", key="login_user")
                    pass_in = st.text_input("Password", type="password", key="login_pass")
                    if st.form_submit_button("Login"):
                        db = load_users_from_db()
                        clean_u = user_in.strip()
                        clean_p = pass_in.strip()
                        if clean_u in db and db[clean_u] == clean_p:
                            st.session_state['logged_in'] = True
                            st.session_state['username'] = clean_u
                            st.rerun()
                        else:
                            st.error("Invalid credentials")

            with tab_register:
                with st.form("sidebar_register"):
                    reg_user = st.text_input("New Username", key="reg_user")
                    reg_pass = st.text_input("New Password", type="password", key="reg_pass")
                    if st.form_submit_button("Create Account"):
                        db = load_users_from_db()
                        c_u = reg_user.strip()
                        c_p = reg_pass.strip()
                        if c_u and c_p:
                            if c_u in db:
                                st.error("User exists.")
                            else:
                                save_user_to_db(c_u, c_p)
                                st.success("Created! Login now.")
                        else:
                            st.warning("Fields cannot be empty")

    st.markdown("---")
    
    # --- NAVIGATION ---
    page = st.radio("📂 **Modules**", [
        "1. Overview/KPIs", "2. Price Explorer", "3. Advanced Forecasting", 
        "4. Sentiment Analysis", "5. Risk & Volatility", "6. Technical Indicators", 
        "7. Correlations", "8. Feature Importance", "9. Strategy Backtest", "10. Data Export"
    ])
    
    st.markdown("---")
    st.subheader("⚙️ Parameter Tuning")
    ticker_symbol = st.text_input("Asset Ticker", "BTC-USD")
    start_date = st.date_input("Start Date", datetime.date(2022, 1, 1))
    end_date = st.date_input("End Date", datetime.date.today())

# =========================================================
#  MAIN DASHBOARD (VISIBLE TO EVERYONE)
# =========================================================

# --- MAIN DATA FETCH ---
data = load_data(ticker_symbol, start_date, end_date)

if data.empty:
    st.warning(f"⚠️ Unable to fetch data for **{ticker_symbol}**. Please check the ticker.")
    st.stop()

# --- GLOBAL INDICATORS ---
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

data['RSI'] = calculate_rsi(data)
data['MA50'] = data['Close'].rolling(window=50).mean()
data['MA200'] = data['Close'].rolling(window=200).mean()
data['Returns'] = data['Close'].pct_change()
data['Volatility'] = data['Returns'].rolling(window=30).std()

# --- SCROLLING TICKER TAPE ---
last_price = data['Close'].iloc[-1]
last_vol = data['Volume'].iloc[-1]

# Removed the infinite marquee animation as per previous request
# Replaced with a static, modern status bar
st.markdown(f"""
<div style="background-color: #1F2937; padding: 10px; border-radius: 10px; border: 1px solid #374151; display: flex; justify-content: space-around; margin-bottom: 20px;">
    <span style="color: #10B981; font-weight: bold;">🔥 {ticker_symbol}: Active</span>
    <span style="color: #3B82F6; font-weight: bold;">💎 Status: Online</span>
    <span style="color: #F59E0B; font-weight: bold;">📈 Price: ${last_price:,.2f}</span>
    <span style="color: #8B5CF6; font-weight: bold;">⚡ AI Models: Ready</span>
</div>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, mae

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

# --- PAGE 1: OVERVIEW ---
if page == "1. Overview/KPIs":
    st.markdown(f'<p class="gradient-text">Executive Analysis: {ticker_symbol}</p>', unsafe_allow_html=True)
    st.markdown("### Real-time Market Intelligence")
    
    current_price = float(data['Close'].iloc[-1])
    prev_close = float(data['Close'].iloc[-2])
    daily_change = ((current_price - prev_close) / prev_close) * 100
    volume = int(data['Volume'].iloc[-1])
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💰 Current Price", f"${current_price:,.2f}", f"{daily_change:.2f}%")
    c2.metric("📈 Period High", f"${float(data['High'].max()):,.2f}")
    c3.metric("📉 Period Low", f"${float(data['Low'].min()):,.2f}")
    c4.metric("📊 24h Volume", f"{volume:,}", delta_color="off")
    
    st.markdown("---")
    
    st.subheader("Price Movement & Trend")
    fig = px.area(data, x='Date', y='Close', title=f'{ticker_symbol} Price History')
    fig.update_traces(line_color='#3B82F6', fillcolor="rgba(59, 130, 246, 0.1)")
    st.plotly_chart(style_chart(fig), use_container_width=True)

# --- PAGE 2: EXPLORER ---
elif page == "2. Price Explorer":
    st.markdown('<p class="gradient-text">Market Deep Dive</p>', unsafe_allow_html=True)
    
    fig = go.Figure(data=[go.Candlestick(
        x=data['Date'],
        open=data['Open'], high=data['High'],
        low=data['Low'], close=data['Close'],
        name="OHLC"
    )])
    fig = style_chart(fig, "Interactive Candle Analysis")
    fig.update_layout(xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("📄 Access Raw Ledger Data"):
        st.dataframe(data.tail(10), use_container_width=True)

# --- PAGE 3: FORECASTING ---
elif page == "3. Advanced Forecasting":
    st.markdown('<p class="gradient-text">AI Predictive Engine</p>', unsafe_allow_html=True)
    col_conf, col_graph = st.columns([1, 3])
    
    with col_conf:
        st.info("Config")
        model_choice = st.selectbox("Select Model", ["ARIMA", "SARIMA", "Prophet", "LSTM"])
        run_btn = st.button("⚡ Generate Forecast", type="primary")

    train_size = int(len(data) * 0.85)
    train_data, test_data = data.iloc[:train_size], data.iloc[train_size:]

    with col_graph:
        if not run_btn:
            st.info("👈 Select a model and click 'Generate Forecast' to begin.")
        else:
            if model_choice == "Prophet":
                with st.spinner("Running Prophet..."):
                    try:
                        df_p = data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
                        m = Prophet()
                        m.fit(df_p)
                        future = m.make_future_dataframe(periods=30)
                        forecast = m.predict(future)
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=df_p['ds'], y=df_p['y'], name='Actual', line=dict(color='#3B82F6')))
                        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast', line=dict(color='#10B981')))
                        st.plotly_chart(style_chart(fig, "Prophet Forecast"), use_container_width=True)
                    except Exception as e: st.error(f"Error: {e}")

            elif model_choice == "ARIMA":
                with st.spinner("Calculating ARIMA..."):
                    try:
                        history = [x for x in train_data['Close'].values]
                        model = ARIMA(history, order=(5,1,0)) 
                        model_fit = model.fit()
                        forecast = model_fit.forecast(steps=len(test_data))
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=train_data['Date'], y=train_data['Close'], name='Train', line=dict(color='#3B82F6')))
                        fig.add_trace(go.Scatter(x=test_data['Date'], y=test_data['Close'], name='Test', line=dict(color='#60A5FA')))
                        fig.add_trace(go.Scatter(x=test_data['Date'], y=forecast, name='ARIMA', line=dict(color='#EF4444')))
                        st.plotly_chart(style_chart(fig, "ARIMA Projection"), use_container_width=True)
                    except Exception as e: st.error(f"Error: {e}")

            elif model_choice == "SARIMA":
                with st.spinner("Calculating SARIMA..."):
                    try:
                        model = SARIMAX(train_data['Close'].values, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
                        model_fit = model.fit(disp=False)
                        forecast = model_fit.forecast(steps=len(test_data))
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Actual', line=dict(color='#3B82F6')))
                        fig.add_trace(go.Scatter(x=test_data['Date'], y=forecast, name='SARIMA', line=dict(color='#F59E0B')))
                        st.plotly_chart(style_chart(fig, "SARIMA Projection"), use_container_width=True)
                    except Exception as e: st.error(f"Error: {e}")

            elif model_choice == "LSTM":
                with st.spinner("Training LSTM..."):
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))
                    
                    X, y = [], []
                    for i in range(60, len(scaled_data)):
                        X.append(scaled_data[i-60:i, 0])
                        y.append(scaled_data[i, 0])
                    X, y = np.array(X), np.array(y)
                    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
                    
                    model = Sequential()
                    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
                    model.add(LSTM(units=50))
                    model.add(Dense(units=1))
                    model.compile(optimizer='adam', loss='mean_squared_error')
                    model.fit(X, y, epochs=5, batch_size=32, verbose=0)
                    
                    test_inputs = scaled_data[len(scaled_data) - len(test_data) - 60:]
                    X_test = []
                    for i in range(60, len(test_inputs)):
                        X_test.append(test_inputs[i-60:i, 0])
                    X_test = np.array(X_test)
                    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
                    
                    predictions = model.predict(X_test)
                    predictions = scaler.inverse_transform(predictions)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Actual', line=dict(color='#3B82F6')))
                    fig.add_trace(go.Scatter(x=test_data['Date'], y=predictions.flatten(), name='LSTM', line=dict(color='#EC4899')))
                    st.plotly_chart(style_chart(fig, "LSTM Neural Net"), use_container_width=True)

# --- PAGE 4: SENTIMENT ---
elif page == "4. Sentiment Analysis":
    st.markdown('<p class="gradient-text">Sentiment & News Pulse</p>', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Live News Feed Simulation")
        headlines = [
            f"🔥 {ticker_symbol} price soars amid institutional interest",
            "⚖️ Market uncertainty rises with new SEC regulations",
            f"🚀 Analysts predict bullish run for {ticker_symbol}",
            "⚠️ Technical indicators flash sell signal on 4H chart",
            "🐋 Whale movement detected: 5000 BTC moved to cold wallet"
        ]
        
        for h in headlines:
            score = TextBlob(h).sentiment.polarity
            if score > 0.1:
                color = "#10B981"
                emoji = "🚀"
                bg = "rgba(16, 185, 129, 0.1)"
            elif score < -0.1:
                color = "#EF4444"
                emoji = "🔻"
                bg = "rgba(239, 68, 68, 0.1)"
            else:
                color = "#9CA3AF"
                emoji = "⚖️"
                bg = "rgba(156, 163, 175, 0.1)"
                
            st.markdown(
                f"""
                <div style="background-color: {bg}; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 5px solid {color}; transition: transform 0.2s;">
                    <div style="font-weight: bold; color: white;">{emoji} {h}</div>
                    <div style="color: {color}; font-size: 0.8em; margin-top: 5px;">AI Sentiment Score: {score:.2f}</div>
                </div>
                """, unsafe_allow_html=True
            )
            
    with col2:
        st.subheader("Market Mood")
        labels = ['Bullish', 'Neutral', 'Bearish']
        values = [45, 30, 25]
        fig = px.pie(values=values, names=labels, hole=0.6, 
                     color_discrete_sequence=['#10B981', '#6B7280', '#EF4444'])
        fig.update_layout(showlegend=False, annotations=[dict(text='Bullish', x=0.5, y=0.5, font_size=20, showarrow=False)])
        st.plotly_chart(style_chart(fig), use_container_width=True)

# --- PAGE 5: VOLATILITY ---
elif page == "5. Risk & Volatility":
    st.markdown('<p class="gradient-text">Risk Management</p>', unsafe_allow_html=True)
    
    sma = data['Close'].rolling(window=20).mean()
    std = data['Close'].rolling(window=20).std()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Price', line=dict(color='#3B82F6')))
    fig.add_trace(go.Scatter(x=data['Date'], y=sma+2*std, name='Upper BB', line=dict(color='#6B7280', width=0)))
    fig.add_trace(go.Scatter(x=data['Date'], y=sma-2*std, name='Lower BB', line=dict(color='#6B7280', width=0), fill='tonexty', fillcolor='rgba(107, 114, 128, 0.2)'))
    st.plotly_chart(style_chart(fig, "Bollinger Bands (Volatility Envelope)"), use_container_width=True)
    
    c1, c2 = st.columns(2)
    with c1:
        fig_vol = px.line(data, x='Date', y='Volatility', title="Rolling Volatility Index (30D)")
        fig_vol.update_traces(line_color='#EF4444', fill='tozeroy', fillcolor='rgba(239, 68, 68, 0.1)')
        st.plotly_chart(style_chart(fig_vol), use_container_width=True)
    with c2:
        fig_ret = px.histogram(data, x='Returns', title="Return Distribution Histogram", nbins=50)
        fig_ret.update_traces(marker_color='#8B5CF6')
        st.plotly_chart(style_chart(fig_ret), use_container_width=True)

# --- PAGE 6: INDICATORS ---
elif page == "6. Technical Indicators":
    st.markdown('<p class="gradient-text">Technical Oscillators</p>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["⚡ RSI Momentum", "🌊 MACD Trend"])
    
    with tab1:
        fig_rsi = px.line(data, x='Date', y='RSI', title="Relative Strength Index")
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="#EF4444", annotation_text="Overbought")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="#10B981", annotation_text="Oversold")
        fig_rsi.update_traces(line_color='#8B5CF6')
        st.plotly_chart(style_chart(fig_rsi), use_container_width=True)
        
    with tab2:
        exp12 = data['Close'].ewm(span=12).mean()
        exp26 = data['Close'].ewm(span=26).mean()
        macd = exp12 - exp26
        signal = macd.ewm(span=9).mean()
        
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=data['Date'], y=macd, name='MACD', line=dict(color='#3B82F6')))
        fig_macd.add_trace(go.Scatter(x=data['Date'], y=signal, name='Signal', line=dict(color='#F59E0B')))
        fig_macd.add_bar(x=data['Date'], y=macd-signal, name='Histogram', marker_color='rgba(255,255,255,0.1)')
        st.plotly_chart(style_chart(fig_macd, "MACD Oscillator"), use_container_width=True)

# --- PAGE 7: CORRELATIONS ---
elif page == "7. Correlations":
    st.markdown('<p class="gradient-text">Market Correlations</p>', unsafe_allow_html=True)
    corr = data[['Open','High','Low','Close','Volume']].corr()
    fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', aspect="auto")
    st.plotly_chart(style_chart(fig, "Statistical Correlation Matrix"), use_container_width=True)

# --- PAGE 8: FEATURE IMPORTANCE ---
elif page == "8. Feature Importance":
    st.markdown('<p class="gradient-text">Lag & Autocorrelation</p>', unsafe_allow_html=True)
    
    df_lag = data[['Close']].copy()
    for i in range(1, 8): 
        df_lag[f'Day -{i}'] = df_lag['Close'].shift(i)
    
    corr_matrix = df_lag.corr()
    lag_corr = corr_matrix['Close'].drop('Close')
    
    fig = px.bar(x=lag_corr.index, y=lag_corr.values, color=lag_corr.values, color_continuous_scale='Viridis')
    fig.update_layout(xaxis_title="Historical Lag", yaxis_title="Correlation Strength")
    st.plotly_chart(style_chart(fig, "Predictive Power of Past Days"), use_container_width=True)

# --- PAGE 9: STRATEGY ---
elif page == "9. Strategy Backtest":
    st.markdown('<p class="gradient-text">Algorithmic Backtest</p>', unsafe_allow_html=True)
    
    sim = data.copy()
    sim['Signal'] = np.where(sim['Close'] > sim['MA50'], 1, 0)
    sim['Strat_Ret'] = sim['Returns'] * sim['Signal'].shift(1)
    sim['Cum_Ret'] = (1 + sim['Strat_Ret']).cumprod()
    sim['Buy_Hold'] = (1 + sim['Returns']).cumprod()
    
    final_strat = (sim['Cum_Ret'].iloc[-1] - 1) * 100
    final_bh = (sim['Buy_Hold'].iloc[-1] - 1) * 100
    
    c1, c2 = st.columns(2)
    c1.metric("🤖 Algo Strategy Return", f"{final_strat:.2f}%", delta=f"{final_strat-final_bh:.2f}% vs B&H")
    c2.metric("✊ Buy & Hold Return", f"{final_bh:.2f}%")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sim['Date'], y=sim['Cum_Ret'], name='Algo Strategy', line=dict(color='#10B981', width=3)))
    fig.add_trace(go.Scatter(x=sim['Date'], y=sim['Buy_Hold'], name='Buy & Hold', line=dict(color='#6B7280', dash='dash')))
    fig.add_trace(go.Scatter(x=sim['Date'], y=np.ones(len(sim)), name='Baseline', line=dict(color='white', width=1)))
    st.plotly_chart(style_chart(fig, "Equity Curve Growth"), use_container_width=True)

# --- PAGE 10: EXPORT ---
elif page == "10. Data Export":
    st.header("💾 Data Extraction")
    st.dataframe(data, use_container_width=True)
    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Full CSV",
        data=csv,
        file_name=f"{ticker_symbol}_full_data.csv",
        mime="text/csv",
        use_container_width=True
    )

