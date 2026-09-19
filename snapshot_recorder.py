import yfinance as yf
import pandas as pd
import ta
from datetime import datetime
import os
import requests
import io

print("Starting daily snapshot evaluation...")

# 1. Fetch dynamic universe
def fetch_symbols():
    headers = {"User-Agent": "Mozilla/5.0"}
    universe = []
    urls = [
        ("Large Cap", "https://archives.nseindia.com/content/indices/ind_nifty50list.csv"),
        ("Mid Cap", "https://archives.nseindia.com/content/indices/ind_niftymidcap150list.csv"),
        ("Small Cap", "https://archives.nseindia.com/content/indices/ind_niftysmallcap250list.csv")
    ]
    for cap, url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=6)
            if r.status_code == 200:
                df = pd.read_csv(io.StringIO(r.text))
                for _, row in df.iterrows():
                    sym = str(row.get("Symbol", "")).strip()
                    if sym and sym != "nan":
                        universe.append({"Symbol": sym, "Cap": cap, "Sector": str(row.get("Industry", "General"))})
        except Exception:
            continue
    return pd.DataFrame(universe)

uni = fetch_symbols()
if uni.empty:
    print("Could not load dynamic universe, exiting.")
    exit(0)

# Sample candidates per cap
candidates = uni.groupby("Cap").head(40)
sym_list = [f"{s}.NS" for s in candidates["Symbol"].tolist()]

data = yf.download(sym_list, period="1y", interval="1d", group_by='ticker', progress=False)
today_str = datetime.now().strftime("%Y-%m-%d")
records = []

for _, meta in candidates.iterrows():
    sym = meta["Symbol"]
    ns_sym = f"{sym}.NS"
    try:
        if ns_sym not in data or data[ns_sym].empty:
            continue
        df = data[ns_sym].dropna()
        if len(df) < 60:
            continue
            
        close = df['Close']
        c_price = float(close.iloc[-1])
        sma_20 = float(ta.trend.sma_indicator(close, window=20).iloc[-1])
        sma_50 = float(ta.trend.sma_indicator(close, window=50).iloc[-1])
        sma_200 = float(ta.trend.sma_indicator(close, window=min(200, len(close)-1)).iloc[-1])
        rsi = float(ta.momentum.rsi(close, window=14).iloc[-1])
        
        # Determine Verdicts
        # Short Term
        if rsi < 33: st_call = "BUY"
        elif 50 <= rsi <= 65 and c_price > sma_20: st_call = "BUY"
        elif rsi > 70 or c_price < sma_20: st_call = "SELL"
        else: st_call = "HOLD"

        # Medium Term
        if c_price > sma_50: mt_call = "BUY"
        else: mt_call = "AVOID"

        # Long Term
        if c_price > sma_200 and sma_50 > sma_200: lt_call = "INVESTABLE"
        else: lt_call = "AVOID"

        records.append({
            "Date": today_str,
            "Symbol": sym,
            "Cap": meta["Cap"],
            "Sector": meta["Sector"],
            "Snapshot_Price": round(c_price, 2),
            "RSI": round(rsi, 1),
            "Short_Term_Call": st_call,
            "Medium_Term_Call": mt_call,
            "Long_Term_Call": lt_call
        })
    except Exception:
        continue

# 2. Append to permanent CSV database
os.makedirs("history", exist_ok=True)
history_file = os.path.join("history", "recommendations_history.csv")

new_df = pd.DataFrame(records)
if os.path.exists(history_file):
    old_df = pd.read_csv(history_file)
    combined = pd.concat([old_df, new_df]).drop_duplicates(subset=["Date", "Symbol"], keep="last")
else:
    combined = new_df

combined.to_csv(history_file, index=False)
print(f"Recorded {len(records)} stocks for {today_str} into {history_file}")