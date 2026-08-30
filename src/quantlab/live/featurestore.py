"""Append-only bar-level store, partitioned by date and symbol.

Why every bar is logged, not every trade
----------------------------------------
:meth:`FeatureStore.log_bar` is called for **every symbol on every bar**,
whether or not the strategy traded. This is not thoroughness for its own sake.

A trade log answers ``P(outcome | I entered)``. The question that carries the
information is ``P(outcome | signal)``, and answering it requires the bars
where the signal fired and the strategy did *not* act -- because the budget was
spent, because a limit blocked it, because the signal was marginal. Those bars
are the counterfactual. Without them the sample is conditioned on the decision
being taken, which is exactly the selection that makes live results diverge
from backtests for reasons nobody can reconstruct afterwards.

Storing the counterfactual is cheap. Reconstructing it later is impossible.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

__all__ = ["FeatureStore", "BarRecord", "forward_returns", "session_id"]

log = logging.getLogger("quantlab.live.featurestore")

#: Columns every stored bar carries. ``feed`` records which tape produced it;
#: ``signals`` and ``position`` may be null on a backfill and populated live.
CORE_COLUMNS = [
    "timestamp", "symbol", "open", "high", "low", "close",
    "volume", "trade_count", "vwap", "feed",
]


@dataclass
class BarRecord:
    """One bar, plus whatever the runtime knew at the moment it closed."""

    timestamp: pd.Timestamp
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    feed: str
    trade_count: float | None = None
    vwap: float | None = None
    #: Signal values computed on this bar, whether or not they were acted on.
    signals: dict[str, float] = field(default_factory=dict)
    #: Position held after this bar. Zero is a real observation, not a missing one.
    position: float = 0.0
    #: True when the runtime submitted an order on this bar. Never a filter for
    #: analysis -- only a column, so the counterfactual stays queryable.
    traded: bool = False

    def to_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "timestamp": pd.Timestamp(self.timestamp, tz="UTC")
            if pd.Timestamp(self.timestamp).tz is None
            else pd.Timestamp(self.timestamp).tz_convert("UTC"),
            "symbol": self.symbol,
            "open": float(self.open),
            "high": float(self.high),
            "low": float(self.low),
            "close": float(self.close),
            "volume": float(self.volume),
            "trade_count": None if self.trade_count is None else float(self.trade_count),
            "vwap": None if self.vwap is None else float(self.vwap),
            "feed": self.feed,
            "position": float(self.position),
            "traded": bool(self.traded),
        }
        for name, value in self.signals.items():
            row[f"signal_{name}"] = None if value is None else float(value)
        return row


def session_id(timestamps: pd.Series) -> pd.Series:
    """Label each bar with its US/Eastern trading date.

    Sessions matter for forward returns: a 390-bar forward return starting at
    15:50 would otherwise run into the next morning, pricing an overnight gap
    the strategy never held. Grouping by Eastern date rather than UTC date is
    what keeps a session intact -- UTC midnight falls in the middle of the
    US session's calendar day for part of the year.
    """
    ts = pd.to_datetime(timestamps, utc=True)
    return ts.dt.tz_convert("America/New_York").dt.date


class FeatureStore:
    """Append-only Parquet store under ``root``, partitioned ``date/symbol``.

    Writes are append-only by construction: every flush writes files with a
    fresh UUID basename, so no existing file is ever rewritten. There is no
    update or delete method. The store is a record of what was observed, and
    editing it is the same act as editing a lab notebook.
    """

    def __init__(self, root: str | Path = "data/features", buffer_size: int = 5000) -> None:
        self.root = Path(root)
        self.buffer_size = buffer_size
        self._buffer: list[dict[str, Any]] = []

    # ------------------------------------------------------------- writing

    def log_bar(self, bar: BarRecord | dict[str, Any]) -> None:
        """Buffer one bar. Call on every symbol every bar; see the module docstring."""
        row = bar.to_row() if isinstance(bar, BarRecord) else dict(bar)
        self._buffer.append(row)
        if len(self._buffer) >= self.buffer_size:
            self.flush()

    def log_bars(self, frame: pd.DataFrame) -> int:
        """Bulk path for backfills. Same guarantees as :meth:`log_bar`."""
        if frame.empty:
            return 0
        missing = [c for c in ("timestamp", "symbol", "close", "feed") if c not in frame.columns]
        if missing:
            raise KeyError(f"bars are missing required columns: {missing}")
        self._buffer.extend(frame.to_dict("records"))
        self.flush()
        return len(frame)

    def flush(self) -> None:
        """Write the buffer to a new partitioned Parquet file set."""
        if not self._buffer:
            return
        frame = pd.DataFrame(self._buffer)
        self._buffer.clear()

        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame["date"] = session_id(frame["timestamp"]).astype(str)
        frame["symbol"] = frame["symbol"].astype(str)

        self.root.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(frame, preserve_index=False)
        ds.write_dataset(
            table,
            base_dir=self.root,
            format="parquet",
            partitioning=ds.partitioning(
                pa.schema([("date", pa.string()), ("symbol", pa.string())]), flavor="hive"
            ),
            # A fresh UUID per flush means no existing file is ever rewritten,
            # which is what makes the store append-only in practice and not
            # merely by convention.
            basename_template=f"part-{uuid.uuid4().hex}-{{i}}.parquet",
            existing_data_behavior="overwrite_or_ignore",
        )
        log.info("flushed %d bars to %s", len(frame), self.root)

    # ------------------------------------------------------------- reading

    def load(
        self,
        start: date | str | None = None,
        end: date | str | None = None,
        symbols: Sequence[str] | None = None,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Read stored bars, optionally filtered by date range and symbol."""
        if not self.root.exists():
            return pd.DataFrame(columns=CORE_COLUMNS)
        dataset = ds.dataset(self.root, format="parquet", partitioning="hive")

        filters = []
        if start is not None:
            filters.append(ds.field("date") >= str(pd.Timestamp(start).date()))
        if end is not None:
            filters.append(ds.field("date") <= str(pd.Timestamp(end).date()))
        if symbols is not None:
            filters.append(ds.field("symbol").isin(list(symbols)))
        expression = None
        for condition in filters:
            expression = condition if expression is None else (expression & condition)

        table = dataset.to_table(filter=expression, columns=list(columns) if columns else None)
        frame = table.to_pandas()
        if frame.empty:
            return frame
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        return frame.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    def summary(self) -> pd.DataFrame:
        """Bars per symbol and per feed -- the first thing to check after a backfill."""
        frame = self.load(columns=["timestamp", "symbol", "feed", "close"])
        if frame.empty:
            return pd.DataFrame()
        return (
            frame.groupby(["symbol", "feed"])
            .agg(bars=("close", "size"),
                 first=("timestamp", "min"),
                 last=("timestamp", "max"))
            .reset_index()
        )

    def purged_forward_returns(
        self,
        horizon: int,
        embargo: int | None = None,
        field_name: str = "close",
        within_session: bool = True,
        entry_lag: int = 1,
        start: date | str | None = None,
        end: date | str | None = None,
        symbols: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Forward returns over ``horizon`` bars, with an embargo for CV folds.

        Returns a long frame with ``timestamp, symbol, forward_return,
        embargo_until``. ``embargo_until`` is the timestamp at which a training
        sample becomes safe to use relative to this observation: an observation
        at ``t`` consumes returns through ``t + horizon``, so any fold boundary
        must clear ``horizon + embargo`` bars, not one.

        Without the embargo, k-fold on overlapping labels leaks: a training
        sample at ``t-1`` and a test sample at ``t`` share ``horizon - 1``
        periods of the same realised return, so the test set is partly a
        restatement of the training set.
        """
        embargo = horizon if embargo is None else embargo
        frame = self.load(start, end, symbols, columns=["timestamp", "symbol", field_name])
        if frame.empty:
            return pd.DataFrame(columns=["timestamp", "symbol", "forward_return", "embargo_until"])

        panel = frame.pivot_table(index="timestamp", columns="symbol",
                                  values=field_name, aggfunc="last").sort_index()
        forward = forward_returns(panel, horizon, within_session=within_session,
                                  entry_lag=entry_lag)

        out = (
            forward.stack(future_stack=True)
            .rename("forward_return")
            .reset_index()
            .dropna(subset=["forward_return"])
        )
        index = panel.index
        position = pd.Series(np.arange(len(index)), index=index)
        safe = np.minimum(position.to_numpy() + horizon + embargo, len(index) - 1)
        located = position.reindex(out["timestamp"])
        if located.isna().any():
            raise ValueError(
                f"{int(located.isna().sum())} forward-return timestamps are absent from "
                "the price index. reindex would map them to NaN and the embargo "
                "boundary would be silently wrong."
            )
        out["embargo_until"] = index[safe[located.to_numpy().astype(int)]]
        out.attrs["horizon"] = horizon
        out.attrs["embargo"] = embargo
        return out


def forward_returns(
    panel: pd.DataFrame,
    horizon: int,
    within_session: bool = True,
    entry_lag: int = 1,
) -> pd.DataFrame:
    """``P_{t+lag+h} / P_{t+lag} - 1`` per column, NaN where the window leaves the session.

    Why ``entry_lag`` defaults to 1, not 0
    --------------------------------------
    The naive definition ``P_{t+h} / P_t`` uses the same print ``P_t`` that the
    signal was computed from. On one-minute equity bars that manufactures
    reversion out of nothing.

    Closing prints alternate between the bid and the ask as buyer- and
    seller-initiated trades arrive. If bar ``t`` happens to close at the bid,
    the price looks low, so a z-score reversion signal says buy -- and the
    forward return measured *from that same low print* is upward-biased,
    because the next print is more likely to be at the ask. The signal and the
    return share a common microstructure error term, and their correlation is
    the bid-ask bounce rather than any economic effect. It is strongest at
    exactly the short horizons where H001 predicts the most signal.

    Setting ``entry_lag=1`` means the signal is computed from the close of bar
    ``t`` and the return is measured from the close of bar ``t+1`` onward. The
    two prices are different prints, the shared bounce term is gone, and the
    measurement matches what a trader could actually do: decide on a close,
    transact into the next bar. It is the same lag >= 1 discipline
    :func:`quantlab.engine.run_backtest` enforces, applied to measurement.

    ``entry_lag=0`` is retained for diagnosis only. The gap between the two is
    the size of the bounce artifact.

    ``within_session=True`` blanks any window that would cross a session
    boundary. A 390-bar forward return starting at 15:50 otherwise prices an
    overnight gap, which an intraday strategy never holds and cannot earn.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if entry_lag < 0:
        raise ValueError("entry_lag must be >= 0")
    forward = panel.shift(-(horizon + entry_lag)) / panel.shift(-entry_lag) - 1.0

    if within_session and len(panel):
        # .to_numpy() is load-bearing. session_id returns a Series carrying a
        # RangeIndex; passing it to pd.Series(..., index=panel.index) makes
        # pandas *align* on the old index rather than relabel, yielding all-NaN
        # and silently discarding every observation. Same failure class as the
        # Phase 1 regime-label bug: no exception, valid dtypes, empty result.
        labels = session_id(pd.Series(panel.index)).to_numpy()
        sessions = pd.Series(labels, index=panel.index)
        # The whole span from the signal bar t through the exit bar
        # t + entry_lag + horizon must sit inside one session.
        crosses_session = (sessions.shift(-(horizon + entry_lag)) != sessions).to_numpy()
        forward.loc[crosses_session, :] = np.nan

        if bool(crosses_session.all()):
            median_session = int(pd.Series(labels).value_counts().median())
            raise ValueError(
                f"no within-session window exists at horizon={horizon} with "
                f"entry_lag={entry_lag}: the median "
                f"session is {median_session} bars, so every {horizon}-bar forward "
                "return crosses into the next session. This is not a bug -- it is "
                "what a horizon at or beyond the session length means. Either "
                "shorten the horizon, or pass within_session=False and report the "
                "result as an OVERNIGHT-INCLUSIVE return. The two are different "
                "quantities: an overnight return is dominated by gap risk and "
                "news, not by the microstructure effects the intraday hypotheses "
                "name as their mechanism."
            )
    return forward
