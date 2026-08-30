"""Market data access. Historical SIP for research, streaming IEX for live.

Which tape, and why it is a required argument
---------------------------------------------
Alpaca's free tier serves **real-time** data from IEX only -- roughly 2% of US
equity volume, with quotes wider than the NBBO. Free-tier **historical** data
is available from the full SIP tape provided the query's ``end`` is at least
15 minutes old.

Research is not real-time, so that 15-minute constraint costs nothing and
research must use SIP. Live paper trading has no choice but IEX, and therefore
validates plumbing rather than P&L.

``feed`` is a required argument at every call site and is recorded on every
returned bar. A study run on IEX bars and reported as if it described the
market is the same failure as Phase 1's regime-label bug: arithmetically
valid, wrongly labelled, silent.

Time comes from the exchange, not the machine
---------------------------------------------
:func:`market_clock` and :func:`market_calendar` read Alpaca's clock endpoint.
The local system clock can be wrong, can be in the wrong timezone, and knows
nothing about half-days or unscheduled closes.
"""

from __future__ import annotations

import logging
import random
import time
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Iterable, Sequence

import pandas as pd

from alpaca.data.enums import Adjustment as AlpacaAdjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus

from .config import (
    PAPER_BASE_URL,
    SIP_DELAY,
    Adjustment,
    Credentials,
    Feed,
    load_credentials,
)

__all__ = [
    "fetch_history",
    "market_clock",
    "market_calendar",
    "candidate_symbols",
    "build_universe",
    "to_panel",
    "BAR_COLUMNS",
]

log = logging.getLogger("quantlab.live.datafeed")

BAR_COLUMNS = [
    "timestamp", "symbol", "open", "high", "low", "close",
    "volume", "trade_count", "vwap", "feed",
]

_MAX_RETRIES = 5


def _timeframe(minutes: int | None, daily: bool = False) -> TimeFrame:
    if daily:
        return TimeFrame.Day
    if minutes is None or minutes < 1:
        raise ValueError("timeframe_minutes must be a positive integer")
    return TimeFrame(minutes, TimeFrameUnit.Minute)


def _as_utc(value: datetime | date | str) -> datetime:
    ts = pd.Timestamp(value)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC").to_pydatetime()


def _retry(call: Callable[[], object], what: str) -> object:
    """Exponential backoff with jitter. Read-only calls, so retries are safe."""
    delay = 1.0
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return call()
        except Exception as exc:  # noqa: BLE001 - surfaced after the final attempt
            if attempt == _MAX_RETRIES:
                raise
            sleep_for = delay + random.uniform(0, delay)
            log.warning("%s failed (attempt %d/%d): %s; retrying in %.1fs",
                        what, attempt, _MAX_RETRIES, exc, sleep_for)
            time.sleep(sleep_for)
            delay *= 2
    raise RuntimeError("unreachable")


def _check_feed(feed: Feed, end: datetime, allow_iex: bool) -> None:
    """Enforce the two feed rules before any request is made."""
    if not isinstance(feed, Feed):
        raise TypeError(f"feed must be a Feed enum member, got {type(feed).__name__}")

    if feed is Feed.SIP:
        cutoff = datetime.now(timezone.utc) - SIP_DELAY
        if end > cutoff:
            raise ValueError(
                f"free-tier historical SIP requires end <= now - 15min. "
                f"end={end.isoformat()} is later than {cutoff.isoformat()}. "
                "Move the window back, or use feed=Feed.IEX with allow_iex=True "
                "and accept that the result does not describe the market."
            )
    elif feed is Feed.IEX and not allow_iex:
        raise ValueError(
            "refusing an IEX research query. IEX is ~2% of US equity volume with "
            "quotes wider than the NBBO, so an information coefficient measured on "
            "it is not a statement about the market. Pass allow_iex=True to "
            "override, and record why in the research log."
        )


def _client(credentials: Credentials | None) -> StockHistoricalDataClient:
    creds = credentials or load_credentials()
    return StockHistoricalDataClient(creds.key_id, creds.secret_key)


def _chunks(items: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def fetch_history(
    symbols: Sequence[str],
    start: datetime | date | str,
    end: datetime | date | str,
    *,
    feed: Feed,
    adjustment: Adjustment = Adjustment.ALL,
    timeframe_minutes: int | None = 1,
    daily: bool = False,
    credentials: Credentials | None = None,
    allow_iex: bool = False,
    symbol_chunk: int = 100,
    client: StockHistoricalDataClient | None = None,
) -> pd.DataFrame:
    """Fetch historical bars as a long DataFrame with one row per (symbol, bar).

    Parameters
    ----------
    feed
        Required. :class:`~quantlab.live.config.Feed.SIP` for research;
        ``Feed.IEX`` requires ``allow_iex=True``.
    adjustment
        Defaults to ``ALL``. Anything else leaves a phantom gap at every split
        and dividend, which momentum and breakout signals will trade happily
        and which is not a return anyone could have earned.

    Returns
    -------
    DataFrame with :data:`BAR_COLUMNS`, sorted by ``(symbol, timestamp)``. The
    ``feed`` column records the tape each bar came from so a stored dataset can
    never be misattributed later.
    """
    if not symbols:
        raise ValueError("symbols must be non-empty")
    start_dt, end_dt = _as_utc(start), _as_utc(end)
    if start_dt >= end_dt:
        raise ValueError(f"start ({start_dt}) must be before end ({end_dt})")
    _check_feed(feed, end_dt, allow_iex)

    data_client = client or _client(credentials)
    timeframe = _timeframe(timeframe_minutes, daily)
    frames: list[pd.DataFrame] = []
    symbols = list(dict.fromkeys(symbols))  # de-duplicate, preserve order

    for batch in _chunks(symbols, symbol_chunk):
        request = StockBarsRequest(
            symbol_or_symbols=list(batch),
            timeframe=timeframe,
            start=start_dt,
            end=end_dt,
            feed=DataFeed(feed.value),
            adjustment=AlpacaAdjustment(adjustment.value),
        )
        barset = _retry(lambda r=request: data_client.get_stock_bars(r),
                        f"get_stock_bars({len(batch)} symbols)")
        frame = getattr(barset, "df", None)
        if frame is None or frame.empty:
            log.warning("no bars returned for %d symbols in %s..%s",
                        len(batch), start_dt.date(), end_dt.date())
            continue
        frames.append(frame.reset_index())

    if not frames:
        return pd.DataFrame(columns=BAR_COLUMNS)

    bars = pd.concat(frames, ignore_index=True)
    bars = bars.rename(columns={"timestamp": "timestamp", "symbol": "symbol"})
    for column in ("trade_count", "vwap"):
        if column not in bars:
            bars[column] = pd.NA
    bars["feed"] = feed.value
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    return (
        bars[BAR_COLUMNS]
        .sort_values(["symbol", "timestamp"])
        .reset_index(drop=True)
    )


def market_clock(
    credentials: Credentials | None = None, base_url: str = PAPER_BASE_URL
) -> dict[str, object]:
    """Exchange clock, from Alpaca. Never the local system clock."""
    creds = credentials or load_credentials()
    trading = TradingClient(creds.key_id, creds.secret_key, paper=True, url_override=base_url)
    clock = _retry(trading.get_clock, "get_clock")
    return {
        "timestamp": pd.Timestamp(clock.timestamp).tz_convert("UTC"),
        "is_open": bool(clock.is_open),
        "next_open": pd.Timestamp(clock.next_open).tz_convert("UTC"),
        "next_close": pd.Timestamp(clock.next_close).tz_convert("UTC"),
    }


def market_calendar(
    start: date | str,
    end: date | str,
    credentials: Credentials | None = None,
    base_url: str = PAPER_BASE_URL,
) -> pd.DataFrame:
    """Trading sessions between two dates, including half-days.

    Half-days matter for this phase specifically: bars-per-day is the divisor
    that converts an effective sample size into a calendar duration, and
    assuming 390 everywhere overstates how much data a date range contains.
    """
    from alpaca.trading.requests import GetCalendarRequest

    creds = credentials or load_credentials()
    trading = TradingClient(creds.key_id, creds.secret_key, paper=True, url_override=base_url)
    request = GetCalendarRequest(start=pd.Timestamp(start).date(), end=pd.Timestamp(end).date())
    sessions = _retry(lambda: trading.get_calendar(request), "get_calendar")
    rows = [
        {
            "date": pd.Timestamp(s.date),
            "open": pd.Timestamp(f"{s.date} {s.open}"),
            "close": pd.Timestamp(f"{s.date} {s.close}"),
        }
        for s in sessions
    ]
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["minutes"] = (frame["close"] - frame["open"]).dt.total_seconds() / 60.0
    return frame


def candidate_symbols(
    credentials: Credentials | None = None,
    base_url: str = PAPER_BASE_URL,
    exclude_otc: bool = True,
) -> list[str]:
    """Currently-tradable US equity symbols, as Alpaca reports them today.

    **This is the survivorship bias, and it cannot be removed with this data
    source.** Alpaca exposes current asset membership, not point-in-time
    membership, so a company that delisted between the backfill start and today
    is absent from this list and therefore absent from the study. The surviving
    names are, on average, the ones that did not go to zero.

    Reported in the README rather than hidden. The mitigation available here is
    partial: :func:`build_universe` ranks these candidates by liquidity **as of
    the backfill start date** rather than today, which removes the
    look-ahead in the *ranking* even though it cannot restore the missing names.
    """
    creds = credentials or load_credentials()
    trading = TradingClient(creds.key_id, creds.secret_key, paper=True, url_override=base_url)
    request = GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE)
    assets = _retry(lambda: trading.get_all_assets(request), "get_all_assets")

    symbols = []
    for asset in assets:
        if not asset.tradable or not asset.symbol.isalpha():
            continue
        if exclude_otc and str(getattr(asset, "exchange", "")).upper().endswith("OTC"):
            continue
        symbols.append(asset.symbol)
    return sorted(set(symbols))


def build_universe(
    as_of: date | str,
    n_symbols: int = 50,
    candidates: Sequence[str] | None = None,
    lookback_days: int = 30,
    credentials: Credentials | None = None,
    feed: Feed = Feed.SIP,
    min_median_dollar_volume: float = 5e7,
) -> pd.DataFrame:
    """Rank candidates by median dollar volume in the window **ending at** ``as_of``.

    Choosing today's most liquid names and then backtesting them over the past
    two years is a look-ahead: liquidity is correlated with having done well.
    The ranking window therefore ends at the backfill start date and never
    looks past it.

    The residual survivorship bias -- that the candidate list contains only
    names still listed today -- is documented in :func:`candidate_symbols` and
    in the README. It is not fixable with this data source.

    Returns a DataFrame of the selected symbols with the liquidity statistics
    that selected them, so the choice is auditable rather than a bare list.
    """
    as_of_ts = pd.Timestamp(as_of).tz_localize("UTC") if pd.Timestamp(as_of).tz is None else pd.Timestamp(as_of)
    window_start = as_of_ts - pd.Timedelta(days=lookback_days)
    pool = list(candidates) if candidates is not None else candidate_symbols(credentials)
    if not pool:
        raise ValueError("no candidate symbols available")

    daily = fetch_history(
        pool, window_start, as_of_ts, feed=feed, daily=True,
        timeframe_minutes=None, credentials=credentials, symbol_chunk=200,
    )
    if daily.empty:
        raise ValueError(f"no daily bars for any candidate in {window_start.date()}..{as_of_ts.date()}")

    daily["dollar_volume"] = daily["close"] * daily["volume"]
    stats = (
        daily.groupby("symbol")
        .agg(
            median_dollar_volume=("dollar_volume", "median"),
            median_close=("close", "median"),
            n_days=("close", "size"),
        )
        .reset_index()
    )
    # Require a nearly-complete history in the ranking window: a symbol with
    # three bars can post a high median by accident.
    required_days = max(int(stats["n_days"].max() * 0.8), 1)
    eligible = stats[
        (stats["n_days"] >= required_days)
        & (stats["median_dollar_volume"] >= min_median_dollar_volume)
    ]
    selected = eligible.nlargest(n_symbols, "median_dollar_volume").reset_index(drop=True)
    selected.attrs["as_of"] = str(as_of_ts.date())
    selected.attrs["lookback_days"] = lookback_days
    selected.attrs["n_candidates"] = len(pool)
    return selected


def to_panel(bars: pd.DataFrame, field: str = "close") -> pd.DataFrame:
    """Pivot long bars into a wide ``timestamp x symbol`` panel of one field."""
    if bars.empty:
        return pd.DataFrame()
    if field not in bars.columns:
        raise KeyError(f"{field!r} not in bars; have {sorted(bars.columns)}")
    return (
        bars.pivot_table(index="timestamp", columns="symbol", values=field, aggfunc="last")
        .sort_index()
    )
