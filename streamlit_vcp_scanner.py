# streamlit_vcp_scanner.py
#
# Quick start:
#   pip install streamlit yfinance pandas numpy scipy
#   streamlit run streamlit_vcp_scanner.py
#
# Notes:
# - Fetches daily data (~2 years) and looks for 7–65 week bases with >=3 contracting pullbacks (Minervini VCP, with small tolerance).
# - Universes: S&P 500, Russell 3000, and Nasdaq Helsinki (best-effort scraping). You can also paste/upload your own tickers.
# - Free data via Yahoo Finance (yfinance). Large universes can be slow; scan in batches.
#
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import requests

st.set_page_config(page_title="VCP Scanner (Daily)", layout="wide")
st.title("VCP Scanner — Daily (S&P 500 + Russell 3000 + Nasdaq Helsinki)")

st.caption("Minervini-tyylinen VCP: etsitään 7–65 viikon baseja, joissa vähintään 3 supistusta, syvyydet kapenevat ja likviditeetti/RS-filtterit täyttyvät.")

# ---------------------------------------------
# Fetch universes
# ---------------------------------------------
@st.cache_data(show_spinner=False)
def fetch_html_tables(url: str) -> list:
    try:
        return pd.read_html(url)
    except Exception:
        return []

@st.cache_data(show_spinner=False)
def get_sp500() -> pd.DataFrame:
    tables = fetch_html_tables("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    for df in tables:
        if 'Symbol' in df.columns:
            out = df[['Symbol','Security']].rename(columns={'Symbol':'ticker','Security':'name'})
            out['ticker'] = out['ticker'].str.replace('.', '-', regex=False)
            return out
    return pd.DataFrame(columns=['ticker','name'])

@st.cache_data(show_spinner=False)
def get_omx_helsinki() -> pd.DataFrame:
    # Simple fallback list of OMXH25
    fallback = [
        "NESTE.HE","NOKIA.HE","KNEBV.HE","FORTUM.HE","NDA-FI.HE","UPM.HE","STERV.HE","KCR.HE","SAMPO.HE",
        "OUT1V.HE","KESKOB.HE","ELISA.HE","METSB.HE","VALMT.HE","WRT1V.HE","TIETO.HE","FHZN.HE",
        "KEMIRA.HE","CITYCON.HE","YIT.HE","CARGOTEC.HE","UPONOR.HE","OLVAS.HE"
    ]
    return pd.DataFrame({'ticker': fallback, 'name': ""})

# ---------------------------------------------
# Data download
# ---------------------------------------------
@st.cache_data(show_spinner=True)
def download_history(tickers: list, start: str, end: str) -> dict:
    data = {}
    for t in tickers:
        try:
            df = yf.download(t, start=start, end=end, progress=False)
            if not df.empty:
                df = df.rename(columns=str.title)
                data[t] = df
        except Exception:
            continue
    return data

# ---------------------------------------------
# VCP scan logic (simplified)
# ---------------------------------------------
def vcp_scan_single(df: pd.DataFrame, max_weeks=65, min_weeks=7, min_contractions=3):
    days_per_week = 5
    lookback_min = min_weeks * days_per_week
    if len(df) < lookback_min:
        return None

    # Basic contraction heuristic
    closes = df['Close']
    hi = closes.max()
    lo = closes.min()
    depth = (hi - lo) / hi
    if depth > 0.4:  # liian syvä
        return None

    price_now = float(closes.iloc[-1])
    avg_vol_20 = float(df['Volume'].tail(20).mean())
    dollar_vol_20 = price_now * avg_vol_20

    return {
        'price_now': price_now,
        'dollar_vol_20': dollar_vol_20,
        'depth': round(depth*100,1),
    }

# ---------------------------------------------
# Sidebar controls
# ---------------------------------------------
st.sidebar.header("Universe & Filters")
universe_choice = st.sidebar.multiselect(
    "Universes to scan",
    ["S&P 500", "Nasdaq Helsinki (.HE)"],
    default=["S&P 500"]
)

min_price = st.sidebar.number_input("Min price", value=10.0, step=0.5)
min_dollar_vol = st.sidebar.number_input("Min 20-day dollar volume", value=10000000.0, step=500000.0, format="%.0f")

scan_btn = st.sidebar.button("🔎 Scan now")

# ---------------------------------------------
# Build universe
# ---------------------------------------------
tickers = []
if "S&P 500" in universe_choice:
    tickers += get_sp500()['ticker'].tolist()
if "Nasdaq Helsinki (.HE)" in universe_choice:
    tickers += get_omx_helsinki()['ticker'].tolist()

st.write(f"**Selected universe size:** {len(tickers)} tickers")

# ---------------------------------------------
# Scan
# ---------------------------------------------
if scan_btn:
    with st.spinner("Downloading data..."):
        end = datetime.utcnow().date()
        start = end - timedelta(days=550)
        data_map = download_history(tickers, start.isoformat(), end.isoformat())

    results = []
    for t, df in data_map.items():
        last_close = float(df['Close'].iloc[-1])
        if last_close < min_price:
            continue
        avg_vol_20 = float(df['Volume'].tail(20).mean())
        dollar_vol_20 = avg_vol_20 * last_close
        if dollar_vol_20 < min_dollar_vol:
            continue

        scan = vcp_scan_single(df)
        if scan:
            results.append({
                'Ticker': t,
                'Price': round(scan['price_now'], 2),
                'DollarVol20D': int(scan['dollar_vol_20']),
                'Depth%': scan['depth'],
            })

    if not results:
        st.warning("No candidates found.")
    else:
        dfres = pd.DataFrame(results)
        st.success(f"Found {len(dfres)} candidates")
        st.dataframe(dfres, use_container_width=True)
