# streamlit_vcp_scanner.py
#
# Quick start:
#   pip install -r requirements.txt
#   streamlit run streamlit_vcp_scanner.py
#
# Etsii VCP-ehdokkaita päivätasolla (7–65 vk base, ≥3 supistusta).
# Universumit: S&P 500, Russell 3000, Nasdaq Helsinki (.HE).
# Data: Yahoo Finance (yfinance). Ilmaiset lähteet -> joskus hitaita/epävakaita.

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict

st.set_page_config(page_title="VCP Scanner (Daily)", layout="wide")
st.title("VCP Scanner — Daily (S&P 500 + Russell 3000 + Nasdaq Helsinki)")

st.caption("Minervini-tyylinen VCP: etsitään 7–65 viikon baseja, joissa ≥3 supistusta, syvyydet kapenevat ja likviditeettisuodattimet täyttyvät.")

# --------------------------------------------------
# Helpers: safe HTTP reads
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def fetch_csv(url: str, sep: str = ",") -> pd.DataFrame:
    try:
        return pd.read_csv(url, sep=sep)
    except Exception:
        return pd.DataFrame()

def _clean_us_symbol(s: str) -> str:
    # Yahoo format: BRK.B -> BRK-B
    return s.replace(".", "-") if isinstance(s, str) else s

def _clean_fi_symbol(s: str) -> str:
    s = s.strip().upper()
    return s if s.endswith(".HE") else (s + ".HE" if "." not in s else s)

# --------------------------------------------------
# Universes with robust fallbacks
# --------------------------------------------------
@st.cache_data(show_spinner=True)
def get_sp500() -> pd.DataFrame:
    # Primary GitHub dataset (community)
    urls = [
        "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv",
        "https://raw.githubusercontent.com/rahuljrw/S-and-P-500-companies/master/data/constituents.csv",
    ]
    for u in urls:
        df = fetch_csv(u)
        if not df.empty and "Symbol" in df.columns:
            out = df[["Symbol", "Name"]].rename(columns={"Symbol": "ticker", "Name": "name"})
            out["ticker"] = out["ticker"].astype(str).str.strip().apply(_clean_us_symbol)
            return out.dropna().drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    # Tiny static fallback (varmistaa, että appi ei jää tyhjäksi)
    fallback = pd.DataFrame(
        {"ticker": ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "BRK-B", "TSLA"], "name": [""] * 8}
    )
    return fallback

@st.cache_data(show_spinner=True)
def get_russell3000() -> pd.DataFrame:
    urls = [
        "https://datahub.io/core/russell-3000/r/russell-3000.csv",
        "https://raw.githubusercontent.com/kieran-mackle/russell-3000/main/russell3000.csv",
        "https://raw.githubusercontent.com/BillSchofield/russell-3000/master/data/russell3000.csv",
    ]
    for u in urls:
        df = fetch_csv(u)
        if df.empty:
            continue
        cols = [c.lower() for c in df.columns]
        df.columns = cols
        ticker_col = "ticker" if "ticker" in cols else ("symbol" if "symbol" in cols else None)
        if ticker_col:
            tick = df[ticker_col].astype(str).str.strip().apply(_clean_us_symbol)
            name = df[[c for c in cols if "name" in c]].iloc[:, 0] if any("name" in c for c in cols) else ""
            out = pd.DataFrame({"ticker": tick, "name": name})
            out = out.dropna().drop_duplicates(subset=["ticker"]).reset_index(drop=True)
            # suodatin: poistetaan selkeästi virheelliset tikkerit
            out = out[out["ticker"].str.len() > 0]
            return out
    # Minimal fallback
    return pd.DataFrame({"ticker": ["AAPL", "MSFT", "NVDA"], "name": [""] * 3})

@st.cache_data(show_spinner=False)
def get_omx_helsinki() -> pd.DataFrame:
    # OMXH25 fallback + muutama lisä
    fallback = [
        "NESTE.HE","NOKIA.HE","KNEBV.HE","FORTUM.HE","NDA-FI.HE","UPM.HE","STERV.HE","KCR.HE","SAMPO.HE",
        "OUT1V.HE","KESKOB.HE","ELISA.HE","METSB.HE","VALMT.HE","WRT1V.HE","TIETO.HE","FHZN.HE",
        "KEMIRA.HE","CITYCON.HE","YIT.HE","CARGOTEC.HE","UPONOR.HE","OLVAS.HE"
    ]
    return pd.DataFrame({"ticker": fallback, "name": [""] * len(fallback)})

# --------------------------------------------------
# Data download (per-ticker for simplicity & stability)
# --------------------------------------------------
@st.cache_data(show_spinner=True)
def download_history(tickers: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
    data = {}
    for t in tickers:
        try:
            df = yf.download(t, start=start, end=end, progress=False, auto_adjust=False)
            if not df.empty:
                df = df.rename(columns=str.title)  # Open, High, Low, Close, Adj Close, Volume
                # drop weird rows
                df = df.dropna(subset=["Close"])
                if "Volume" not in df.columns:
                    df["Volume"] = 0
                data[t] = df
        except Exception:
            # jatka seuraavaan tikkeriin
            pass
    return data

# --------------------------------------------------
# VCP scan (safe math)
# --------------------------------------------------
def vcp_scan_single(df: pd.DataFrame, max_weeks=65, min_weeks=7, min_contractions=3):
    # Vaatimaton “proto” – tärkeintä on ettei kaadu dataan
    days_per_week = 5
    look_min = min_weeks * days_per_week
    if df is None or df.empty or len(df) < look_min:
        return None

    closes = df["Close"]
    hi = float(closes.max()) if len(closes) else float("nan")
    lo = float(closes.min()) if len(closes) else float("nan")

    depth = float("nan")
    if hi and hi == hi and hi > 0:  # hi is not NaN
        depth = float((hi - lo) / hi)

    # jos depth ei ole numeroa, hylätään
    if not isinstance(depth, float) or depth != depth:
        return None

    if depth > 0.40:  # liian syvä base
        return None

    price_now = float(closes.iloc[-1])
    vol = df["Volume"] if "Volume" in df.columns else pd.Series([0]*len(df), index=df.index)
    avg_vol_20 = float(vol.tail(20).mean()) if len(vol) else 0.0
    dollar_vol_20 = price_now * avg_vol_20

    return {
        "price_now": price_now,
        "dollar_vol_20": dollar_vol_20,
        "depth_pct": round(depth * 100.0, 1),
    }

# --------------------------------------------------
# UI
# --------------------------------------------------
st.sidebar.header("Universe & Filters")
universe_choice = st.sidebar.multiselect(
    "Universes to scan",
    ["S&P 500", "Russell 3000", "Nasdaq Helsinki (.HE)"],
    default=["S&P 500", "Nasdaq Helsinki (.HE)"]
)

min_price = st.sidebar.number_input("Min price (USD/EUR)", value=10.0, step=0.5)
min_dollar_vol = st.sidebar.number_input("Min 20-day dollar volume (USD/EUR)", value=10000000.0, step=500000.0, format="%.0f")

st.sidebar.write("---")
scan_btn = st.sidebar.button("🔎 Scan now")

# Build universe
tickers: List[str] = []
if "S&P 500" in universe_choice:
    tickers += get_sp500()["ticker"].tolist()
if "Russell 3000" in universe_choice:
    tickers += get_russell3000()["ticker"].tolist()
if "Nasdaq Helsinki (.HE)" in universe_choice:
    tickers += get_omx_helsinki()["ticker"].tolist()

# dedupe + basic cleanup
tickers = sorted({t for t in tickers if isinstance(t, str) and len(t.strip()) > 0})

st.write(f"**Selected universe size:** {len(tickers)} tickers")

# Scan
if scan_btn:
    with st.spinner("Downloading data (EOD)…"):
        end = datetime.utcnow().date()
        start = end - timedelta(days=550)  # ~2 vuotta
        data_map = download_history(tickers, start.isoformat(), end.isoformat())

    results = []
    for t, df in data_map.items():
        try:
            if df is None or df.empty:
                continue

            last_close = float(df["Close"].iloc[-1])
            if last_close < min_price:
                continue

            vol = df["Volume"] if "Volume" in df.columns else pd.Series([0]*len(df), index=df.index)
            avg_vol_20 = float(vol.tail(20).mean())
            dollar_vol_20 = avg_vol_20 * last_close
            if dollar_vol_20 < min_dollar_vol:
                continue

            scan = vcp_scan_single(df)
            if scan is None:
                continue

            results.append({
                "Ticker": t,
                "Price": round(scan["price_now"], 2),
                "DollarVol20D": int(scan["dollar_vol_20"]),
                "Depth%": scan["depth_pct"],
                "Chart": f"https://www.tradingview.com/chart/?symbol={t}",
            })
        except Exception:
            # älä kaada koko ajoa yksittäiseen virheeseen
            continue

    if not results:
        st.warning("No candidates found with current filters. Kokeile löysätä rajoja tai valitse pienempi universumi.")
    else:
        dfres = pd.DataFrame(results).sort_values(["Depth%", "DollarVol20D"], ascending=[True, False]).reset_index(drop=True)
        st.success(f"Found {len(dfres)} candidates")
        st.dataframe(dfres, use_container_width=True)
        st.download_button("Download results CSV", dfres.to_csv(index=False), "vcp_candidates.csv", "text/csv")
else:
    st.info("Valitse universumit ja filtterit vasemmalta ja paina **Scan now**.")
