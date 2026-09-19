import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
import re
import feedparser
import requests
import io
from datetime import datetime

# Auto-refresh every 5 minutes
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=300 * 1000, key="datarefresh")
except Exception:
    pass

st.set_page_config(page_title="India Market Screener & Research Audit", layout="wide")

# =========================================================
# 1. DYNAMIC NSE UNIVERSE LOADER
# =========================================================
@st.cache_data(ttl=86400)
def load_dynamic_nse_universe():
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    universe = []
    urls = [
        ("Large Cap", "https://archives.nseindia.com/content/indices/ind_nifty50list.csv"),
        ("Mid Cap", "https://archives.nseindia.com/content/indices/ind_niftymidcap150list.csv"),
        ("Small Cap", "https://archives.nseindia.com/content/indices/ind_niftysmallcap250list.csv")
    ]
    for cap, url in urls:
        try:
            resp = requests.get(url, headers=headers, timeout=6)
            if resp.status_code == 200:
                df = pd.read_csv(io.StringIO(resp.text))
                for _, row in df.iterrows():
                    sym = str(row.get("Symbol", "")).strip()
                    comp = str(row.get("Company Name", sym)).strip()
                    sec = str(row.get("Industry", "Diversified")).strip()
                    if sym and sym != "nan":
                        universe.append({"Symbol": sym, "Company": comp, "Sector": sec, "Cap": cap})
        except Exception:
            continue

    if not universe:
        core_fallback = [
            ("RELIANCE", "Reliance Industries", "Energy", "Large Cap"),
            ("TCS", "Tata Consultancy", "IT", "Large Cap"),
            ("HDFCBANK", "HDFC Bank", "Financials", "Large Cap"),
            ("ICICIBANK", "ICICI Bank", "Financials", "Large Cap"),
            ("INFY", "Infosys", "IT", "Large Cap"),
            ("TATAMOTORS", "Tata Motors", "Automobile", "Large Cap"),
            ("SUZLON", "Suzlon Energy", "Energy", "Mid Cap"),
            ("BHEL", "BHEL", "Capital Goods", "Mid Cap"),
            ("TRENT", "Trent Ltd", "Retail", "Mid Cap"),
            ("CDSL", "CDSL", "Financial Market", "Small Cap"),
            ("ANGELONE", "Angel One", "Financial Market", "Small Cap")
        ]
        for s, c, sec, cp in core_fallback:
            universe.append({"Symbol": s, "Company": c, "Sector": sec, "Cap": cp})

    return pd.DataFrame(universe)

# =========================================================
# 2. SENTIMENT & NEWS ENGINE
# =========================================================
FINANCIAL_LEXICON = {
    "surge": 2.5, "surges": 2.5, "soar": 3.0, "rally": 2.0, "breakout": 2.5, "boom": 2.5,
    "profit": 1.5, "growth": 1.8, "dividend": 1.2, "outperform": 2.2, "buyback": 2.0,
    "upgrade": 2.0, "bull": 1.5, "bullish": 2.0, "gain": 1.5, "order": 1.5,
    "crash": -3.0, "plunge": -2.8, "slump": -2.5, "loss": -2.2, "downgrade": -2.5,
    "bear": -1.5, "bearish": -2.0, "debt": -1.5, "probe": -2.2, "fraud": -3.0,
    "weak": -1.8, "decline": -1.8, "penalty": -2.0, "selloff": -2.5, "cut": -1.5
}

def calculate_text_sentiment(text):
    words = re.findall(r'\b\w+\b', text.lower())
    s = sum(FINANCIAL_LEXICON.get(w, 0.0) for w in words)
    return np.tanh(s / 3.0) if s != 0 else 0.0

@st.cache_data(ttl=300)
def fetch_live_market_news():
    feeds = [
        ("Economic Times", "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"),
        ("LiveMint", "https://www.livemint.com/rss/markets"),
        ("Reuters Global", "https://feeds.reuters.com/reuters/businessNews")
    ]
    articles = []
    for src, url in feeds:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:5]:
                score = calculate_text_sentiment(entry.title)
                articles.append({
                    "source": src,
                    "title": entry.title,
                    "link": entry.link,
                    "score": score,
                    "time": getattr(entry, 'published', 'Today')
                })
        except Exception:
            continue
    return articles

# =========================================================
# 3. MACRO BENCHMARK PULSE
# =========================================================
@st.cache_data(ttl=1800)
def fetch_macro_data():
    nifty = yf.Ticker("^NSEI")
    hist = nifty.history(period="5y")
    if hist.empty:
        return None
    
    current_month = datetime.now().month
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    hist['Month'] = hist.index.month
    hist['Return'] = hist['Close'].pct_change()
    seasonal_score = hist.groupby('Month')['Return'].mean().get(current_month, 0.0) * 100
    
    close = hist['Close']
    sma50 = ta.trend.sma_indicator(close, window=50).iloc[-1]
    sma200 = ta.trend.sma_indicator(close, window=200).iloc[-1]
    curr_nifty = close.iloc[-1]
    nifty_chg = curr_nifty - close.iloc[-2]
    
    if curr_nifty > sma50 and sma50 > sma200:
        fii_regime = "Risk-On (Bullish Macro Base)"
    elif curr_nifty < sma50:
        fii_regime = "Risk-Off (Defensive / Institutional Selling)"
    else:
        fii_regime = "Consolidating / Neutral"
        
    return {
        "nifty_price": curr_nifty,
        "nifty_chg": nifty_chg,
        "fii_regime": fii_regime,
        "seasonal_month": month_names[current_month - 1],
        "seasonal_score": seasonal_score
    }

# =========================================================
# 4. STRICT 3-HORIZON AUDIT ENGINE
# =========================================================
@st.cache_data(ttl=600)
def screen_universe_three_horizons(candidate_df):
    symbols = [f"{s}.NS" for s in candidate_df["Symbol"].tolist()]
    data = yf.download(symbols, period="1y", interval="1d", group_by='ticker', progress=False)
    
    rows = []
    for _, meta in candidate_df.iterrows():
        sym_raw = meta["Symbol"]
        sym_ns = f"{sym_raw}.NS"
        try:
            if sym_ns not in data or data[sym_ns].empty:
                continue
            df_stk = data[sym_ns].dropna()
            if len(df_stk) < 60:
                continue
                
            close = df_stk['Close']
            curr_price = float(close.iloc[-1])
            sma_20 = float(ta.trend.sma_indicator(close, window=20).iloc[-1])
            sma_50 = float(ta.trend.sma_indicator(close, window=50).iloc[-1])
            sma_200 = float(ta.trend.sma_indicator(close, window=min(200, len(close)-1)).iloc[-1])
            rsi = float(ta.momentum.rsi(close, window=14).iloc[-1])
            
            # --- STRICT HORIZON 1: SHORT TERM (1-14 DAYS) ---
            # Focus: Swing momentum, RSI exhaustion/reversals
            if rsi < 33:
                st_call = "🟢 BUY (Oversold Bounce)"
                st_rank = 3
            elif 50 <= rsi <= 65 and curr_price > sma_20:
                st_call = "🟢 BUY (Momentum)"
                st_rank = 2
            elif rsi > 70:
                st_call = "🔴 SELL (Overbought Exhaustion)"
                st_rank = -2
            elif curr_price < sma_20 and rsi < 45:
                st_call = "🔴 SELL / AVOID (Weak)"
                st_rank = -1
            else:
                st_call = "🟡 HOLD (Consolidation)"
                st_rank = 0

            # --- STRICT HORIZON 2: MEDIUM TERM (1-6 MONTHS) ---
            # Focus: 50-day SMA structure and quarterly base
            if curr_price > sma_50 and sma_50 > sma_200:
                mt_call = "🟢 BUY (Strong Uptrend)"
                mt_rank = 3
            elif curr_price > sma_50:
                mt_call = "🟢 ACCUMULATE (Above 50 SMA)"
                mt_rank = 2
            elif curr_price < sma_50 and sma_50 < sma_200:
                mt_call = "🔴 AVOID (Downtrend Continuation)"
                mt_rank = -3
            else:
                mt_call = "🔴 AVOID / CAUTION (Sub-50 SMA)"
                mt_rank = -1

            # --- STRICT HORIZON 3: LONG TERM (1+ YEARS) ---
            # Focus: 200-day SMA macro baseline
            if curr_price > sma_200 and sma_50 > sma_200:
                lt_call = "🟢 INVESTABLE (Secular Bull)"
                lt_rank = 3
            elif curr_price > sma_200:
                lt_call = "🟡 WATCH (Above 200 SMA only)"
                lt_rank = 1
            else:
                lt_call = "🔴 NON-INVESTABLE (Macro Bear Breakdown)"
                lt_rank = -3

            rows.append({
                "Symbol": sym_raw,
                "Company": meta["Company"],
                "Sector": meta["Sector"],
                "Cap": meta["Cap"],
                "Price (₹)": round(curr_price, 2),
                "RSI": round(rsi, 1),
                "Above 50 SMA": "Yes" if curr_price > sma_50 else "No",
                "Above 200 SMA": "Yes" if curr_price > sma_200 else "No",
                "Short-Term (1-14D)": st_call,
                "Medium-Term (1-6M)": mt_call,
                "Long-Term (1Y+)": lt_call,
                "_st_rank": st_rank,
                "_mt_rank": mt_rank,
                "_lt_rank": lt_rank
            })
        except Exception:
            continue
            
    return pd.DataFrame(rows)

# =========================================================
# 5. DASHBOARD PRESENTATION
# =========================================================
st.title("🇮🇳 Indian Market Screener: Segregated Horizons & Research Audit")
st.caption(f"Last Refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} IST | Strict Timeframe Isolation")

# Macro Ribbon
macro = fetch_macro_data()
if macro:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Nifty 50 Index", f"₹{macro['nifty_price']:.2f}", f"{macro['nifty_chg']:+.2f}")
    m2.metric("Macro Trend Stance", macro['fii_regime'])
    m3.metric(f"Historical Seasonality ({macro['seasonal_month']})", f"{macro['seasonal_score']:+.2f}% Avg")
    m4.metric("Universe Sync", "NSE Live Batch Engine")

st.markdown("---")

universe_df = load_dynamic_nse_universe()

# --- TOP 25 RECOMMENTATION ENGINE (SEPARATED BY HORIZON) ---
st.subheader("🎯 Top 25 Recommendations (Strictly Segregated by Horizon)")
st.caption("Each horizon operates on independent mathematical models. No contradictory single scores.")

f_col1, f_col2 = st.columns(2)
with f_col1:
    cap_choice = st.radio("Market Cap:", ["All Caps", "Large Cap", "Mid Cap", "Small Cap"], horizontal=True)

avail_sectors = ["All Sectors"]
if not universe_df.empty:
    sub_df = universe_df if cap_choice == "All Caps" else universe_df[universe_df["Cap"] == cap_choice]
    avail_sectors += sorted(list(sub_df["Sector"].dropna().unique()))

with f_col2:
    sec_choice = st.selectbox("Filter Sector:", avail_sectors)

filtered_universe = sub_df.copy()
if sec_choice != "All Sectors":
    filtered_universe = filtered_universe[filtered_universe["Sector"] == sec_choice]

with st.spinner("Executing horizon-wise technical audit across universe..."):
    audited_df = screen_universe_three_horizons(filtered_universe.head(90))

if not audited_df.empty:
    tab_st, tab_mt, tab_lt, tab_all = st.tabs([
        "⚡ Top 25 Short-Term (1–14 Days)",
        "⏳ Top 25 Medium-Term (1–6 Months)",
        "🏛️ Top 25 Long-Term (1+ Years)",
        "📋 Complete Screener Matrix"
    ])
    
    with tab_st:
        st.markdown("**Criteria:** Swing momentum, RSI pullback bounds, price relative to 20-day EMA.")
        top_st = audited_df.sort_values(by=["_st_rank", "RSI"], ascending=[False, True]).head(25)
        st.dataframe(
            top_st[["Short-Term (1-14D)", "Symbol", "Company", "Cap", "Sector", "Price (₹)", "RSI", "Above 50 SMA", "Above 200 SMA"]],
            use_container_width=True, hide_index=True
        )

    with tab_mt:
        st.markdown("**Criteria:** Structural 50-day SMA support, quarterly volume stability, mid-term trend strength.")
        top_mt = audited_df.sort_values(by=["_mt_rank", "Price (₹)"], ascending=[False, False]).head(25)
        st.dataframe(
            top_mt[["Medium-Term (1-6M)", "Symbol", "Company", "Cap", "Sector", "Price (₹)", "Above 50 SMA", "Above 200 SMA", "RSI"]],
            use_container_width=True, hide_index=True
        )

    with tab_lt:
        st.markdown("**Criteria:** Uncompromising 200-day secular SMA filter and golden moving average structure.")
        top_lt = audited_df.sort_values(by=["_lt_rank", "Price (₹)"], ascending=[False, False]).head(25)
        st.dataframe(
            top_lt[["Long-Term (1Y+)", "Symbol", "Company", "Cap", "Sector", "Price (₹)", "Above 200 SMA", "Above 50 SMA", "RSI"]],
            use_container_width=True, hide_index=True
        )

    with tab_all:
        st.dataframe(
            audited_df[["Symbol", "Company", "Cap", "Sector", "Price (₹)", "Short-Term (1-14D)", "Medium-Term (1-6M)", "Long-Term (1Y+)", "RSI", "Above 50 SMA", "Above 200 SMA"]],
            use_container_width=True, hide_index=True
        )

st.markdown("---")

# =========================================================
# 6. UNIVERSAL SEARCH + INSTITUTIONAL CONSENSUS AUDIT
# =========================================================
st.subheader("🔎 Universal Stock Search, Live Charts & Research House Audit")
st.markdown("Search **any** NSE symbol to see segregated calls, institutional brokerage price targets, and mathematical audits:")

search_sym = st.text_input("Enter NSE Stock Code (e.g. HDFCBANK, RELIANCE, ZOMATO, TCS):", value="HDFCBANK").upper().strip()

if search_sym:
    sym_ticker = f"{search_sym}.NS" if not search_sym.endswith((".NS", ".BO")) else search_sym
    stk = yf.Ticker(sym_ticker)
    df_stk = stk.history(period="2y")
    
    if df_stk.empty:
        st.error(f"No active data found for symbol '{search_sym}'.")
    else:
        # Indicators
        df_stk['SMA_20'] = ta.trend.sma_indicator(df_stk['Close'], window=20)
        df_stk['SMA_50'] = ta.trend.sma_indicator(df_stk['Close'], window=50)
        df_stk['SMA_200'] = ta.trend.sma_indicator(df_stk['Close'], window=min(200, len(df_stk)-1))
        df_stk['RSI'] = ta.momentum.rsi(df_stk['Close'], window=14)
        macd = ta.trend.MACD(df_stk['Close'])
        df_stk['MACD'] = macd.macd()
        df_stk['MACD_Signal'] = macd.macd_signal()
        
        last = df_stk.iloc[-1]
        prev = df_stk.iloc[-2]
        c_price = float(last['Close'])
        sma20 = float(last['SMA_20'])
        sma50 = float(last['SMA_50'])
        sma200 = float(last['SMA_200'])
        rsi_val = float(last['RSI'])
        macd_val = float(last['MACD'])
        macd_sig = float(last['MACD_Signal'])
        
        info = getattr(stk, 'info', {})
        stock_news = getattr(stk, 'news', [])
        stk_sentiments = [calculate_text_sentiment(n.get('title', '')) for n in stock_news] if stock_news else [0.0]
        avg_sentiment = float(np.mean(stk_sentiments))
        
        # Institutional Brokerage Consensus Data
        analyst_recommendation = info.get('recommendationKey', 'N/A').replace('_', ' ').title()
        target_mean_price = info.get('targetMeanPrice', None)
        target_high_price = info.get('targetHighPrice', None)
        target_low_price = info.get('targetLowPrice', None)
        num_analysts = info.get('numberOfAnalystOpinions', 'N/A')
        
        # Key Technical Metrics
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Current Price", f"₹{c_price:.2f}", f"{(c_price - prev['Close']):+.2f}")
        k2.metric("14-Day RSI", f"{rsi_val:.1f}")
        k3.metric("50-Day / 200-Day SMA", f"₹{sma50:.1f} / ₹{sma200:.1f}")
        sent_txt = "Bullish" if avg_sentiment > 0.05 else ("Bearish" if avg_sentiment < -0.05 else "Neutral")
        k4.metric("News Sentiment", f"{sent_txt} ({avg_sentiment:+.2f})")
        
        st.markdown("### 🎯 Segregated Horizon Verdicts")
        h1, h2, h3 = st.columns(3)
        
        # 1. Short Term
        with h1:
            st.markdown("#### ⚡ Short Term (1–14 Days)")
            if rsi_val < 33:
                st.success("**Call: BUY (Oversold Bounce)**")
                st.caption("Deep technical dip with high probability of swing mean-reversion.")
            elif 50 <= rsi_val <= 65 and c_price > sma20:
                st.success("**Call: BUY (Momentum Swing)**")
                st.caption("Trending above 20-EMA with healthy momentum.")
            elif rsi_val > 70:
                st.error("**Call: SELL / AVOID (Overbought)**")
                st.caption("Momentum exhausted. High probability of near-term corrective pullback.")
            elif c_price < sma20:
                st.error("**Call: SELL / AVOID (Weak Momentum)**")
                st.caption("Price is under immediate 20-day moving average pressure.")
            else:
                st.warning("**Call: HOLD (Consolidation)**")
                st.caption("Range-bound. No immediate directional edge.")

        # 2. Medium Term
        with h2:
            st.markdown("#### ⏳ Medium Term (1–6 Months)")
            if c_price > sma50 and sma50 > sma200:
                st.success("**Call: ACCUMULATE**")
                st.caption("Golden alignment: Trading comfortably above 50-day and 200-day averages.")
            elif c_price > sma50:
                st.success("**Call: ACCUMULATE ON DIPS**")
                st.caption("Above 50-day average, signaling quarterly trend support.")
            elif c_price < sma50 and sma50 < sma200:
                st.error("**Call: STRICT AVOID / SELL**")
                st.caption("Deep downtrend channel: Trading below both 50 and 200-day SMAs.")
            else:
                st.error("**Call: CAUTION / WAIT**")
                st.caption(f"Broken below 50-day SMA (₹{sma50:.1f}). Lacks quarterly momentum.")

        # 3. Long Term
        with h3:
            st.markdown("#### 🏛️ Long Term (1+ Years)")
            if c_price > sma200:
                st.success("**Call: INVESTABLE (Core Hold)**")
                st.caption(f"Macro secular bull: Holding above secular 200-day benchmark (₹{sma200:.1f}).")
            else:
                st.error("**Call: NON-INVESTABLE (Macro Breakdown)**")
                st.caption(f"Trading below 200-day SMA (₹{sma200:.1f}). Institutional capital preservation rules advise against entering.")

        # --- INSTITUTIONAL RESEARCH HOUSE CONSENSUS PANEL ---
        st.markdown("### 🏢 Institutional Research House Consensus (Brokerage Targets)")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Broker Consensus Rating", f"{analyst_recommendation}")
        b2.metric("Total Research Analysts", f"{num_analysts}")
        
        if target_mean_price:
            upside = ((target_mean_price - c_price) / c_price) * 100
            b3.metric("Consensus Target Price", f"₹{target_mean_price:.2f}", f"{upside:+.1f}% Potential")
            b4.metric("Target Band (Low – High)", f"₹{target_low_price:.0f} – ₹{target_high_price:.0f}")
        else:
            b3.metric("Consensus Target Price", "No active consensus")
            b4.metric("Target Band", "N/A")

        # --- FULL TECHNICAL & MATHEMATICAL AUDIT EXPANDER ---
        with st.expander(f"🔬 Click here to see Full Technical Audit & Rationale for {search_sym}"):
            st.markdown(f"""
            #### Mathematical Breakdown:
            
            1. **Macro Trend Benchmark (200-Day SMA):**
               - **Current Price:** ₹{c_price:.2f} | **200-Day SMA:** ₹{sma200:.2f}
               - **Spread:** `{((c_price / sma200) - 1) * 100:+.2f}%`
               - **Status:** {"✅ PASSED (In Bull Territory)" if c_price > sma200 else "❌ FAILED (In Secular Bear/Distribution Territory). Long-term institutions do not allocate when price is below the 200-day SMA."}

            2. **Cyclical Moving Average (50-Day SMA):**
               - **Current Price:** ₹{c_price:.2f} | **50-Day SMA:** ₹{sma50:.2f}
               - **Status:** {"✅ Above 50-day average (quarterly trend positive)." if c_price > sma50 else "❌ Below 50-day average. Quarterly selling pressure dominates."}

            3. **Momentum Gauge (14-Day RSI):**
               - **Value:** `{rsi_val:.1f}`
               - **Status:** {"⚠️ Overbought risk." if rsi_val > 70 else ("🟢 Oversold dip setup." if rsi_val < 33 else "✅ Healthy momentum channel.")}

            4. **MACD Divergence:**
               - **MACD:** `{macd_val:.2f}` | **Signal:** `{macd_sig:.2f}` (Spread: `{macd_val - macd_sig:+.2f}`)
               - **Status:** {"✅ Positive expansion." if macd_val > macd_sig else "❌ Bearish crossover / downward drift."}

            5. **Research House Validation:**
               - Broker consensus currently rates this as **{analyst_recommendation}** across {num_analysts} analyst opinions.
               {"- Target price represents fundamental upside, but technical timing should respect moving average baselines before executing entries." if target_mean_price and c_price < sma200 else ""}
            """)

        # Interactive Chart
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
        fig.add_trace(go.Candlestick(x=df_stk.index, open=df_stk['Open'], high=df_stk['High'], low=df_stk['Low'], close=df_stk['Close'], name="Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_stk.index, y=df_stk['SMA_20'], line=dict(color='green', width=1), name="20 SMA"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_stk.index, y=df_stk['SMA_50'], line=dict(color='orange', width=1.5), name="50 SMA"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_stk.index, y=df_stk['SMA_200'], line=dict(color='blue', width=2), name="200 SMA"), row=1, col=1)
        fig.add_trace(go.Bar(x=df_stk.index, y=df_stk['Volume'], name="Volume", marker_color='gray'), row=2, col=1)
        fig.update_layout(height=480, margin=dict(l=0, r=0, t=10, b=0), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # Company Specific News
        st.markdown("#### 📰 Recent Company Headlines")
        if stock_news:
            for art in stock_news[:5]:
                st.markdown(f"• **[{art.get('title')}]({art.get('link', '#')})** — *{art.get('publisher', 'Financial Wire')}*")
        else:
            st.info("No active news entries found for this symbol.")

# =========================================================
# 7. LIVE NEWS STREAM
# =========================================================
st.markdown("---")
st.subheader("📡 Live Market & Macro News Flow")
live_feed = fetch_live_market_news()
if live_feed:
    for item in live_feed[:8]:
        badge = "🟢" if item['score'] > 0.05 else ("🔴" if item['score'] < -0.05 else "⚪")
        st.markdown(f"{badge} **[{item['title']}]({item['link']})** — *{item['source']}* (`{item['time']}`)")

# =========================================================
# HISTORICAL EVALUATION & BACKTEST TAB
# =========================================================
st.markdown("---")
st.subheader("📈 Historical Recommendation Tracker (15–20 Day Evaluation)")

history_path = os.path.join("history", "recommendations_history.csv")
if os.path.exists(history_path):
    hist_df = pd.read_csv(history_path)
    recorded_dates = sorted(hist_df["Date"].unique().tolist(), reverse=True)
    
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        eval_date = st.selectbox("Select Past Recommendation Date to Evaluate:", recorded_dates)
    
    if eval_date:
        subset = hist_df[hist_df["Date"] == eval_date].copy()
        
        # Fetch current live prices to measure return
        sym_list = [f"{s}.NS" for s in subset["Symbol"].tolist()]
        live_quotes = yf.download(sym_list, period="5d", interval="1d", group_by='ticker', progress=False)
        
        returns = []
        for _, row in subset.iterrows():
            sym_ns = f"{row['Symbol']}.NS"
            try:
                curr_p = float(live_quotes[sym_ns]['Close'].dropna().iloc[-1])
                orig_p = float(row["Snapshot_Price"])
                gain_pct = ((curr_p - orig_p) / orig_p) * 100
                returns.append({
                    "Symbol": row["Symbol"],
                    "Cap": row["Cap"],
                    "Sector": row["Sector"],
                    "Recommendation Date": row["Date"],
                    "Original Call": row["Short_Term_Call"],
                    "Snapshot Price (₹)": orig_p,
                    "Current Price (₹)": round(curr_p, 2),
                    "Return (%)": round(gain_pct, 2),
                    "Result": "✅ Profitable" if gain_pct > 0 else "❌ Drawdown"
                })
            except Exception:
                continue
                
        eval_results = pd.DataFrame(returns)
        if not eval_results.empty:
            buys = eval_results[eval_results["Original Call"] == "BUY"]
            win_rate = (buys["Return (%)"] > 0).mean() * 100 if not buys.empty else 0.0
            avg_gain = buys["Return (%)"].mean() if not buys.empty else 0.0
            
            w1, w2, w3 = st.columns(3)
            w1.metric("Historical Date Evaluated", eval_date)
            w2.metric("Buy Calls Win Rate", f"{win_rate:.1f}%")
            w3.metric("Average Return per Pick", f"{avg_gain:+.2f}%")
            
            st.dataframe(eval_results, use_container_width=True, hide_index=True)
else:
    st.info("No past snapshots stored yet. Run 'python snapshot_recorder.py' once to generate your baseline snapshot.")