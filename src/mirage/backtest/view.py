"""MarketView: the ONLY window through which strategies see data.

Leakage is prevented by construction — the view contains data sliced to
<= decision day t (prices through t's close; news through t's 16:00 ET close).
The engine additionally logs (decision_day, data_max_ts) for the audit in
`mirage.data.leakage`.
"""

from __future__ import annotations

import pandas as pd

from ..data.news import close_utc


class MarketData:
    """Immutable container for a full experiment's data."""

    def __init__(self, adj_close: pd.DataFrame, membership=None, news: pd.DataFrame | None = None):
        self.adj_close = adj_close.sort_index()
        self.returns = self.adj_close.pct_change()
        self.membership = membership
        if news is not None:
            news = news.sort_values("timestamp_utc").reset_index(drop=True)
        self.news = news
        self.calendar = self.adj_close.index


class MarketView:
    def __init__(self, market: MarketData, t: pd.Timestamp, prev_day: pd.Timestamp | None):
        self.t = t
        self.prev_day = prev_day
        self._market = market
        self.prices = market.adj_close.loc[:t]
        self.members = market.membership.members(t) if market.membership else None
        self._news_max_ts: pd.Timestamp | None = None

    # --- news access ------------------------------------------------------ #

    def news_window(self, tickers=None) -> pd.DataFrame:
        """Headlines in (previous trading day's close, today's close]."""
        news = self._market.news
        if news is None or news.empty:
            return pd.DataFrame(columns=["timestamp_utc", "ticker", "headline"])
        hi = close_utc(self.t)
        lo = close_utc(self.prev_day) if self.prev_day is not None else hi - pd.Timedelta(days=1)
        mask = (news["timestamp_utc"] > lo) & (news["timestamp_utc"] <= hi)
        if tickers is not None:
            mask &= news["ticker"].isin(set(tickers))
        out = news[mask]
        if len(out):
            m = out["timestamp_utc"].max()
            if self._news_max_ts is None or m > self._news_max_ts:
                self._news_max_ts = m
        return out

    # --- audit ------------------------------------------------------------ #

    @property
    def data_max_ts(self) -> pd.Timestamp:
        """Latest timestamp of any datum exposed to the strategy this day."""
        price_ts = close_utc(self.prices.index.max())
        if self._news_max_ts is not None and self._news_max_ts > price_ts:
            return self._news_max_ts
        return price_ts
