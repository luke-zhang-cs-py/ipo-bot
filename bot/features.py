"""Features, computed only from what was knowable at the decision time.

Stocks: a prediction for session t+1 is made at 22:00 UTC on session t (after the close and after the VIX
settles). Its features use closes up to t and macro values whose available_at is at or before that moment
(the Treasury curve, posted at 18:00 New York, is often the previous day's in winter).

IPOs: a prediction is made the evening the registration becomes effective (or the final prospectus appears, if
no EFFECT notice was read), and only if that is before the first trade. Features use the filings dated by then
(never the final prospectus's shares or lead filed later), and the first-day returns of IPOs that had traded
before that day.
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
POP_TOL = 1e-9  # a first-day return this close below the pop threshold is a pop (floating-point division)
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


def popped(ret: float, pop: float) -> bool:
    """Whether a first-day return is a pop: at or above the threshold, with a tolerance of POP_TOL so that an
    offer of 10 closing at 12 (12 / 10 - 1 = 0.19999999999999996 in floating point) is a 20% pop. NaN is not."""
    return bool(ret >= pop - POP_TOL)


def ipo_rows(deals: Sequence[ipos.Deal], macro_rows: pd.DataFrame, pop: float) -> pd.DataFrame:
    """One row per first-time IPO deal with a range and a prediction moment before its first trade: IPO_FEATURES,
    the moment, and (when known) the outcome: ret (first close / offer - 1) and y (popped).

    Every feature uses only what was public by the moment's evening: ranges, shares, lead bank and the foreign
    flag from filings dated on or before it (Deal.asof, never the final prospectus filed later), the VIX by its
    available_at, and the first-day returns of IPOs that traded before the moment's day and whose final
    prospectus (the offer) was on file by then. A deal whose moment is on or after its first trade (no EFFECT
    notice was read and the final prospectus came out after listing) has no honest prediction and is dropped."""
    cands = []
    for d in deals:
        m = ipo_moment(d)
        if not d.ipo or m is None or not any(r[0] <= m for r in d.ranges):
            continue
        if d.first_trade and d.first_trade["date"] <= m:
            continue
        cands.append((d, m))
    if not cands:
        return pd.DataFrame(columns=["cik", "company", "moment", *IPO_FEATURES, "y", "ret", "trade_date"])
    times = [markets.close_utc(dt.date.fromisoformat(m)) + markets.FILING_DELAY for _, m in cands]
    vix = macro_asof(macro_rows, times)
    traded = sorted(
        (
            (d.first_trade["date"], d.priced or d.first_trade["date"], d.first_day_return)
            for d in deals
            if d.ipo and d.first_trade and d.first_day_return is not None
        ),
        key=lambda x: x[0],
    )
    rows: List[Dict[str, object]] = []
    for i, (d, m) in enumerate(cands):
        rngs = [r for r in d.ranges if r[0] <= m]
        lo0, hi0 = rngs[0][1], rngs[0][2]
        lo, hi = rngs[-1][1], rngs[-1][2]
        mid0, mid = (lo0 + hi0) / 2, (lo + hi) / 2
        known = d.asof(m)
        past = [r for t, priced, r in traded if t < m and priced <= m][-20:]
        rows.append(
            {
                "cik": d.cik,
                "company": d.company,
                "moment": m,
                "range_rev": mid / mid0 - 1,
                "range_width": (hi - lo) / mid,
                "log_size": math.log(known["shares"] * mid) if known["shares"] else np.nan,
                "vix": vix["VIX"].iloc[i] if "VIX" in vix else np.nan,
                "heat": float(np.mean(past)) if len(past) >= 5 else np.nan,
                "top_bank": float(known["lead"] in ipos.TOP_BANKS),
                "foreign": float(known["foreign"]),
                "ret": d.first_day_return,
                "trade_date": d.first_trade["date"] if d.first_trade else None,
            }
        )
    df = pd.DataFrame(rows)
    df["ret"] = df["ret"].astype(float)
    df["y"] = df["ret"].map(lambda r: float(popped(r, pop))).where(df["ret"].notna())
    return df.sort_values(["moment", "cik"]).reset_index(drop=True)
