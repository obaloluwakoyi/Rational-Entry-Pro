"""
engine.py — Data pipeline & indicator engine
Handles: yfinance fetching, pandas_ta indicators, scoring, stop-loss
"""

import streamlit as st
import yfinance as yf
import pandas as pd
from dataclasses import dataclass
from typing import Optional


def calc_sma(series: pd.Series, length: int = 50) -> pd.Series:
    return series.rolling(length, min_periods=length).mean()


def calc_rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


@dataclass
class StockResult:
    ticker:       str
    company_name: str
    sector:       str
    price:        float
    sma50:        float
    rsi:          float
    atr:          float
    ma_dev:       float
    stop_loss:    float
    risk_score:   float
    zone:         str
    df:           pd.DataFrame
    sma50_series: pd.Series
    last_date:    str
    error:        Optional[str] = None


CLOUD_PROVIDERS = {
    "Anthropic (Claude)": {"default_model": "claude-sonnet-4-20250514", "key_hint": "sk-ant-…"},
    "OpenAI":             {"default_model": "gpt-4o-mini",               "key_hint": "sk-…"},
    "Google Gemini":      {"default_model": "gemini-1.5-flash",          "key_hint": "AIza…"},
    "Groq (free tier)":   {"default_model": "llama3-70b-8192",           "key_hint": "gsk_…"},
    "Mistral AI":         {"default_model": "mistral-small-latest",      "key_hint": "mistral key…"},
    "Together AI":        {"default_model": "meta-llama/Llama-3-70b-chat-hf", "key_hint": "together key…"},
    "Cohere":             {"default_model": "command-r-plus",            "key_hint": "cohere key…"},
}

LOCAL_PROVIDERS = {
    "Ollama":    {"url": "http://localhost:11434", "default_model": "llama3"},
    "LM Studio": {"url": "http://localhost:1234",  "default_model": "local-model"},
    "GPT4All":   {"url": "http://localhost:4891",  "default_model": "mistral-7b"},
    "Jan":       {"url": "http://localhost:1337",  "default_model": "llama3-8b"},
    "KoboldCpp": {"url": "http://localhost:5001",  "default_model": "kobold-model"},
    "LocalAI":   {"url": "http://localhost:8080",  "default_model": "gpt-3.5-turbo"},
}


@st.cache_data(ttl=900, show_spinner=False)
def fetch_ohlcv(ticker: str, period: str = "200d") -> Optional[pd.DataFrame]:
    try:
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False, threads=False)
        if df is None or df.empty or len(df) < 60:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_meta(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info
        return {
            "name":   info.get("longName") or info.get("shortName") or ticker,
            "sector": info.get("sector", "Unknown"),
        }
    except Exception:
        return {"name": ticker, "sector": "Unknown"}


def compute(ticker: str, period: str = "200d") -> StockResult:
    df = fetch_ohlcv(ticker, period)
    if df is None:
        return StockResult(
            ticker=ticker, company_name=ticker, sector="—",
            price=0, sma50=0, rsi=0, atr=0, ma_dev=0,
            stop_loss=0, risk_score=0, zone="amber",
            df=pd.DataFrame(), sma50_series=pd.Series(dtype=float),
            last_date="—", error=f"No data found for '{ticker}'. Check the symbol."
        )

    meta  = fetch_meta(ticker)
    close = df["Close"].squeeze()
    high  = df["High"].squeeze()
    low   = df["Low"].squeeze()

    rsi_s   = calc_rsi(close, length=14)
    rsi     = float(rsi_s.dropna().iloc[-1]) if rsi_s is not None and not rsi_s.dropna().empty else 50.0
    sma50_s = calc_sma(close, length=50)
    sma50   = float(sma50_s.dropna().iloc[-1]) if sma50_s is not None and not sma50_s.dropna().empty else float(close.iloc[-1])
    atr_s   = calc_atr(high, low, close, length=14)
    atr     = float(atr_s.dropna().iloc[-1]) if atr_s is not None and not atr_s.dropna().empty else float(close.std())

    price  = float(close.iloc[-1])
    ma_dev = ((price - sma50) / sma50) * 100
    stop   = round(price - 2 * atr, 2)

    if rsi > 70 and ma_dev > 15:
        score = min(100, 67 + (rsi - 70) * 0.9 + (ma_dev - 15) * 0.55)
        zone  = "red"
    elif rsi < 35 or ma_dev <= 0:
        raw   = 20 - (35 - rsi) * 0.5 if rsi < 35 else 33 + ma_dev * 0.4
        score = max(0, min(33, raw))
        zone  = "green"
    else:
        score = 34 + ((rsi - 35) / 35) * 24 + max(0, ma_dev) * 0.45
        score = min(66, max(34, score))
        zone  = "amber"

    return StockResult(
        ticker=ticker, company_name=meta["name"], sector=meta["sector"],
        price=price, sma50=sma50, rsi=rsi, atr=atr, ma_dev=ma_dev,
        stop_loss=stop, risk_score=round(score, 1), zone=zone,
        df=df, sma50_series=sma50_s if sma50_s is not None else pd.Series(dtype=float),
        last_date=df.index[-1].strftime("%b %d, %Y"),
    )
