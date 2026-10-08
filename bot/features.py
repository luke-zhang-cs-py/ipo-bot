"""Features, computed only from what was knowable at the decision time.

Stocks: a prediction for session t+1 is made at 22:00 UTC on session t (after the close and after the VIX
settles). Its features use closes up to t and macro values whose available_at is at or before that moment
(the Treasury curve, posted at 18:00 New York, is often the previous day's in winter).

IPOs: a prediction is made when the registration becomes effective (or when the final prospectus appears, if
no EFFECT notice was read), before the first trade. Features use ranges filed by then, and the first-day
returns of IPOs that had traded by then.
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from bot import ipos, markets

STOCK_FEATURES = ("r1", "r5", "r21", "vol21", "off_high", "mkt_r1", "mkt_r5", "vix", "dvix5", "slope")
IPO_FEATURES = ("range_rev", "range_width", "log_size", "vix", "heat", "top_bank", "foreign")
DECISION_HOUR_UTC = 22
INDEX = "SPX"


def decision_time(day: dt.date) -> dt.datetime:
    """When the prediction for the session after `day` is made: 22:00 UTC on `day`."""
    return dt.datetime.combine(day, dt.time(DECISION_HOUR_UTC), tzinfo=dt.UTC)


def macro_asof(macro_rows: pd.DataFrame, times: Sequence[dt.datetime]) -> pd.DataFrame:
    """For each decision time, each series' newest value available by then. macro_rows: series, date, value,
    available_at (the store's rows). Returns a frame indexed like `times`."""
    out = pd.DataFrame(index=pd.DatetimeIndex(pd.to_datetime(list(times), utc=True), name="t"))
    if macro_rows.empty:
        return out
    m = macro_rows.copy()
    m["avail"] = pd.to_datetime(m["available_at"], utc=True)
    left = pd.DataFrame({"t": out.index}).sort_values("t")
    for series, g in m.groupby("series"):
        g = g.sort_values(["date", "avail"]).drop_duplicates("date", keep="first")
        # the newest *date* known by t: a later date can be available earlier than a revision of an older one
        g = g.sort_values("avail")
        g["best"] = g["date"].cummax()
        val = g.set_index("date")["value"]
        merged = pd.merge_asof(left, g[["avail", "best"]].rename(columns={"avail": "t"}), on="t", direction="backward")
        out[series] = merged["best"].map(val).to_numpy()
    return out


def stock_panel(
    closes: pd.DataFrame, macro_rows: pd.DataFrame, symbols: Optional[Sequence[str]] = None, with_target: bool = True
) -> pd.DataFrame:
    """One row per (date, symbol) with STOCK_FEATURES and, with_target, y (1 if the next session's close is
    higher) and ret (the next session's log return). closes: date x symbol, including INDEX."""
    if closes.empty:
        return pd.DataFrame(columns=["date", "symbol", *STOCK_FEATURES, "y", "ret"])
    syms = [s for s in (symbols if symbols is not None else closes.columns) if s in closes.columns and s != INDEX]
    lc = np.log(closes[syms].astype(float))
    r1 = lc.diff()
    feats: Dict[str, pd.DataFrame] = {
        "r1": r1,
        "r5": lc.diff(5),
        "r21": lc.diff(21),
        "vol21": r1.rolling(21, min_periods=15).std(),
        "off_high": lc - lc.rolling(252, min_periods=60).max(),
    }
    dates = [dt.date.fromisoformat(d) for d in closes.index]
    mac = macro_asof(macro_rows, [decision_time(d) for d in dates])
    mac.index = closes.index
    mkt = np.log(closes[INDEX].astype(float)) if INDEX in closes.columns else pd.Series(np.nan, index=closes.index)
    common = {
        "mkt_r1": mkt.diff(),
        "mkt_r5": mkt.diff(5),
        "vix": mac["VIX"] if "VIX" in mac else pd.Series(np.nan, index=closes.index),
        "slope": (
            (mac["UST10Y"] - mac["UST3M"])
            if {"UST10Y", "UST3M"} <= set(mac.columns)
            else pd.Series(np.nan, index=closes.index)
        ),
    }
    common["dvix5"] = np.log(common["vix"]).diff(5)
    long = {k: v.stack(future_stack=True) for k, v in feats.items()}
    panel = pd.DataFrame(long)
    panel.index.names = ["date", "symbol"]
    panel = panel.reset_index()
    for k, s in common.items():
        panel[k] = panel["date"].map(s)
    if with_target:
        nxt = lc.shift(-1) - lc
        # the next *session*: a missing bar leaves the target missing rather than jumping two days
        panel["ret"] = nxt.stack(future_stack=True).to_numpy()
        panel["y"] = (panel["ret"] > 0).astype(float).where(panel["ret"].notna())
    return panel[["date", "symbol", *STOCK_FEATURES, *(["y", "ret"] if with_target else [])]]


# ---------------------------------------------------------------------------- IPOs


def ipo_moment(d: ipos.Deal) -> Optional[str]:
    """The filing date the IPO prediction is made after: effectiveness, else the final prospectus."""
    return d.effective or d.priced


def ipo_rows(deals: Sequence[ipos.Deal], macro_rows: pd.DataFrame, pop: float) -> pd.DataFrame:
    """One row per first-time IPO deal with a range and a prediction moment: IPO_FEATURES, the moment, and
    (when known) the outcome: ret (first close / offer - 1) and y (ret >= pop)."""
    cands = [d for d in deals if d.ipo and d.ranges and ipo_moment(d)]
    if not cands:
        return pd.DataFrame(columns=["cik", "company", "moment", *IPO_FEATURES, "y", "ret", "trade_date"])
    moments = [str(ipo_moment(d)) for d in cands]
    times = [markets.close_utc(dt.date.fromisoformat(m)) + markets.FILING_DELAY for m in moments]
    vix = macro_asof(macro_rows, times)
    traded = sorted(
        (
            (d.first_trade["date"], d.first_day_return)
            for d in deals
            if d.ipo and d.first_trade and d.first_day_return is not None
        ),
        key=lambda x: x[0],
    )
    rows: List[Dict[str, object]] = []
    for i, (d, m) in enumerate(zip(cands, moments)):
        rngs = [r for r in d.ranges if r[0] <= m] or d.ranges[:1]
        lo0, hi0 = rngs[0][1], rngs[0][2]
        lo, hi = rngs[-1][1], rngs[-1][2]
        mid0, mid = (lo0 + hi0) / 2, (lo + hi) / 2
        past = [r for t, r in traded if t < m][-20:]
        rows.append(
            {
                "cik": d.cik,
                "company": d.company,
                "moment": m,
                "range_rev": mid / mid0 - 1,
                "range_width": (hi - lo) / mid,
                "log_size": math.log(d.shares * mid) if d.shares else np.nan,
                "vix": vix["VIX"].iloc[i] if "VIX" in vix else np.nan,
                "heat": float(np.mean(past)) if len(past) >= 5 else np.nan,
                "top_bank": float(d.lead in ipos.TOP_BANKS),
                "foreign": float(d.foreign),
                "ret": d.first_day_return,
                "trade_date": d.first_trade["date"] if d.first_trade else None,
            }
        )
    df = pd.DataFrame(rows)
    df["ret"] = df["ret"].astype(float)
    df["y"] = (df["ret"] >= pop).astype(float).where(df["ret"].notna())
    return df.sort_values(["moment", "cik"]).reset_index(drop=True)
