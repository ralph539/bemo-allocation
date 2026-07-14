import io
import pandas as pd
import requests
import yfinance as yf

ECB_URL = "https://data-api.ecb.europa.eu/service/data/EXR/D.{ccy}.EUR.SP00.A"


def fetch_fx(start: str) -> pd.DataFrame:
    # ECB daily reference rates, foreign units per 1 EUR. EURUSD, EURGBP.
    out = {}
    for ccy, col in [("USD", "EURUSD"), ("GBP", "EURGBP")]:
        r = requests.get(ECB_URL.format(ccy=ccy),
                         params={"startPeriod": start, "format": "csvdata"}, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), usecols=["TIME_PERIOD", "OBS_VALUE"])
        s = pd.Series(df["OBS_VALUE"].values,
                      index=pd.to_datetime(df["TIME_PERIOD"]), name=col).sort_index()
        out[col] = s
    fx = pd.concat(out.values(), axis=1).dropna(how="all")  # drop blank TARGET-holiday rows
    fx.index.name = "date"
    return fx


def fetch_prices(tickers: list[str], start: str) -> pd.DataFrame:
    # Dividend-adjusted close (total return) per ticker. Fail hard on empty.
    cols = {}
    for t in tickers:
        h = yf.Ticker(t).history(start=start, auto_adjust=True)
        if h.empty or h["Close"].isna().all():
            raise RuntimeError(f"no price data for {t}")
        s = h["Close"].copy()
        s.index = s.index.tz_localize(None).normalize()
        cols[t] = s[~s.index.duplicated(keep="last")]
    px = pd.DataFrame(cols).sort_index()
    px.index.name = "date"
    return px
