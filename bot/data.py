"""Reading the store back as analysis-ready frames, always as of a moment."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import pandas as pd

from bot.store import Store

SOURCE_ORDER = ("yahoo", "cboe")  # per day, the first source that has a bar is the one used


def bars(store: Store, at: Optional[str] = None, symbols: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Daily bars as knowable at `at`: one row per (symbol, date), Yahoo's when it has one, else Cboe's.
    Columns: symbol, date, open, high, low, close, volume, source, available_at."""
    where = {"symbol": sorted(set(symbols))} if symbols is not None else None
    rows = store.asof("prices", at, where=where)
    if not rows:
        return pd.DataFrame(
            columns=["symbol", "date", "open", "high", "low", "close", "volume", "source", "available_at"]
        )
    df = pd.DataFrame(rows).drop(columns=["run_id"])
    df["rank"] = df["source"].map({s: i for i, s in enumerate(SOURCE_ORDER)}).fillna(len(SOURCE_ORDER))
    df = df.sort_values(["symbol", "date", "rank"]).drop_duplicates(["symbol", "date"]).drop(columns=["rank"])
    return df.reset_index(drop=True)


def closes(store: Store, at: Optional[str] = None, symbols: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Closes as a date x symbol frame (dates as ISO text, sorted), as knowable at `at`."""
    b = bars(store, at, symbols)
    if b.empty:
        return pd.DataFrame()
    return b.pivot(index="date", columns="symbol", values="close").sort_index()


def macro(store: Store, at: Optional[str] = None) -> pd.DataFrame:
    """Macro series as a date x series frame, as knowable at `at` (the first source per day wins)."""
    rows = store.asof("macro", at)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values(["series", "date", "available_at"]).drop_duplicates(["series", "date"])
    return df.pivot(index="date", columns="series", values="value").sort_index()


def members(store: Store, at: Optional[str] = None) -> List[str]:
    """Symbols in the universe at `at`: index members (or the configured list), less those known delisted."""
    rows = store.asof("universe", at)
    member = {r["symbol"] for r in rows if r["source"] in ("wikipedia", "config") and r["member"] == 1}
    delisted = {r["symbol"] for r in rows if r["source"] == "nasdaqtrader" and r["listed"] == 0}
    return sorted(member - delisted)


def splits(store: Store, at: Optional[str] = None) -> Dict[str, Dict[str, float]]:
    """symbol -> {date: ratio} of the splits on record at `at`."""
    out: Dict[str, Dict[str, float]] = {}
    for r in store.asof("actions", at, where={"kind": "split"}):
        out.setdefault(r["symbol"], {})[r["date"]] = float(r["value"])
    return out


def dividends(store: Store, at: Optional[str] = None) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for r in store.asof("actions", at, where={"kind": "dividend"}):
        out.setdefault(r["symbol"], {})[r["date"]] = float(r["value"])
    return out
