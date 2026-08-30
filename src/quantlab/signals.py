"""Signal library.

Every signal is a pure function ``(prices, **params) -> pd.DataFrame`` of target
positions in ``[-1, +1]``, one column per asset, computed causally.

Causality contract
------------------
A signal value at index ``t`` may use prices at indices ``<= t`` only.  Rolling
statistics that *end* at ``t`` satisfy this: ``prices.rolling(w).mean()`` at
``t`` reads ``t - w + 1 .. t``, all of which are known at ``t``.

Signals do **not** apply the execution lag themselves.  They report what you
know at ``t``; :func:`quantlab.engine.run_backtest` applies ``.shift(lag)`` with
``lag >= 1`` to turn that into a position that earns the return from ``t`` to
``t + 1``.  Splitting the two means the lag is applied exactly once, in one
place, and can be varied without touching the signal library.

Warm-up is reported as NaN rather than 0.  A NaN says "not yet defined"; a 0
says "defined, and flat".  Conflating them silently prepends a flat position to
every strategy and biases turnover and hit rate.

The ``.lookback`` attribute
---------------------------
Each signal function exposes ``.lookback``, the number of leading bars it
cannot define, evaluated at its *default* parameters.  Because the true
lookback depends on the parameters, use ``fn.bind(**params)`` to get a
:class:`BoundSignal` whose ``.lookback`` matches the parameters actually used.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

__all__ = [
    "BoundSignal",
    "zscore_reversion",
    "zscore_momentum",
    "timeseries_momentum",
    "ma_crossover",
    "vol_breakout",
    "random_signal",
    "always_long",
    "SIGNALS",
    "CONTROLS",
    "PRICE_DEPENDENT",
    "default_signal_set",
]


@dataclass(frozen=True)
class BoundSignal:
    """A signal function with its parameters and the resulting lookback fixed."""

    name: str
    fn: Callable[..., pd.DataFrame]
    params: Mapping[str, Any] = field(default_factory=dict)
    lookback: int = 0

    def __call__(self, prices: pd.DataFrame) -> pd.DataFrame:
        return self.fn(prices, **self.params)

    @property
    def label(self) -> str:
        if not self.params:
            return self.name
        inner = ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.name}({inner})"


def _signal(lookback: Callable[[dict[str, Any]], int]):
    """Attach ``.lookback`` and ``.bind`` to a signal function."""

    def decorate(fn: Callable[..., pd.DataFrame]) -> Callable[..., pd.DataFrame]:
        params = inspect.signature(fn).parameters
        defaults = {
            k: v.default
            for k, v in params.items()
            if v.default is not inspect.Parameter.empty
        }
        fn.default_params = defaults                      # type: ignore[attr-defined]
        fn.lookback = lookback(defaults)                  # type: ignore[attr-defined]
        fn.lookback_for = lookback                        # type: ignore[attr-defined]

        def bind(**overrides: Any) -> BoundSignal:
            merged = {**defaults, **overrides}
            unknown = set(overrides) - set(defaults)
            if unknown:
                raise TypeError(f"{fn.__name__}() has no parameter(s): {sorted(unknown)}")
            return BoundSignal(fn.__name__, fn, merged, lookback(merged))

        fn.bind = bind                                    # type: ignore[attr-defined]
        return fn

    return decorate


def _clip_unit(frame: pd.DataFrame) -> pd.DataFrame:
    """Enforce the ``[-1, +1]`` position contract, preserving NaN warm-up."""
    return frame.clip(lower=-1.0, upper=1.0)


def _rolling_z(log_prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """Trailing z-score of log price over ``window`` bars, ending at ``t``.

    ``z_t = (x_t - mean(x_{t-w+1..t})) / sd(x_{t-w+1..t})`` with ``ddof=1``.
    A zero rolling standard deviation (a perfectly flat window) yields 0
    rather than an infinity.
    """
    mean = log_prices.rolling(window).mean()
    sd = log_prices.rolling(window).std(ddof=1)
    z = (log_prices - mean) / sd.where(sd > 0)
    return z.where(sd.notna(), np.nan).fillna(0.0).where(mean.notna(), np.nan)


@_signal(lambda p: p["window"])
def zscore_reversion(prices: pd.DataFrame, window: int = 20, clip: float = 2.0) -> pd.DataFrame:
    """Fade deviations from the trailing mean: ``position = -z / clip``.

    Bets that a log price ``clip`` standard deviations above its ``window``-bar
    mean will fall back toward it.  Correct on an Ornstein-Uhlenbeck process by
    construction, and a coin flip on a random walk -- where it nonetheless
    trades constantly and therefore loses exactly the cost it pays.

    Note this is the exact negative of :func:`zscore_momentum` at matching
    parameters.  Their *gross* returns sum to zero on any data; their *net*
    returns sum to minus twice the cost.  There is no allocation between the
    two that breaks even.
    """
    z = _rolling_z(np.log(prices), window)
    return _clip_unit(-z / clip)


@_signal(lambda p: p["window"])
def zscore_momentum(prices: pd.DataFrame, window: int = 60, clip: float = 2.0) -> pd.DataFrame:
    """Ride deviations from the trailing mean: ``position = +z / clip``.

    Bets that a log price extended above its ``window``-bar mean keeps rising.
    The exact negative of :func:`zscore_reversion`; see that docstring.
    """
    z = _rolling_z(np.log(prices), window)
    return _clip_unit(z / clip)


@_signal(lambda p: p["window"])
def timeseries_momentum(prices: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Sign of the trailing cumulative log return over ``window`` bars.

    ``position_t = sign(log P_t - log P_{t-window})``.  Binary rather than
    scaled: the classic time-series-momentum result is that the *sign* of past
    returns predicts the sign of future returns, and scaling by magnitude adds
    volatility exposure without adding information.
    """
    log_prices = np.log(prices)
    trailing = log_prices - log_prices.shift(window)
    return _clip_unit(np.sign(trailing))


@_signal(lambda p: p["slow"])
def ma_crossover(prices: pd.DataFrame, fast: int = 10, slow: int = 50) -> pd.DataFrame:
    """Long when the fast moving average is above the slow one, short otherwise.

    ``position_t = sign(MA_fast(t) - MA_slow(t))``.  Both averages end at ``t``.
    A moving-average crossover is a band-pass filter on the log price; it is
    time-series momentum with a smoother entry, and on a random walk it is a
    machine for paying costs.
    """
    if fast >= slow:
        raise ValueError(f"fast ({fast}) must be shorter than slow ({slow})")
    spread = prices.rolling(fast).mean() - prices.rolling(slow).mean()
    return _clip_unit(np.sign(spread))


@_signal(lambda p: p["window"] + 1)
def vol_breakout(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Donchian channel breakout: long on a new ``window``-bar high, short on a low.

    The channel is computed over ``t-window .. t-1`` -- excluding ``t`` itself
    via ``.shift(1)``.  Including the current bar would make ``P_t > max`` a
    tautology that can never fire, which is a silent way to produce a signal
    that is always flat.

    Between breakouts the previous position is carried forward; forward-fill of
    a past value is causal.

    Once the channel exists but before the first breakout has occurred, the
    position is **flat**, not undefined.  That is a real state of the strategy
    -- a Donchian system genuinely holds nothing until price first leaves the
    channel -- and encoding it as 0 rather than NaN is what makes
    ``lookback = window + 1`` an honest claim.  Leaving it NaN would make the
    true warm-up depend on when a breakout happens to occur, which is not a
    number you can declare in advance.
    """
    upper = prices.rolling(window).max().shift(1)
    lower = prices.rolling(window).min().shift(1)
    raw = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    raw = raw.mask(prices > upper, 1.0).mask(prices < lower, -1.0)
    positioned = raw.ffill()
    channel_defined = upper.notna() & lower.notna()
    return _clip_unit(positioned.where(~(channel_defined & positioned.isna()), 0.0))


@_signal(lambda p: 0)
def random_signal(prices: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """CONTROL: iid uniform positions on ``[-1, +1]``, independent of prices.

    This is the noise floor.  It has exactly zero information, trades as much
    as a real strategy, and therefore loses money at the rate the cost model
    charges.  Any strategy that does not beat this is not a strategy.

    Draws are generated in index order from a fixed seed, so extending the
    price history leaves earlier values untouched -- the control obeys the same
    causality contract as everything else, and is included in the causality
    property test rather than exempted from it.
    """
    rng = np.random.default_rng(seed)
    values = rng.uniform(-1.0, 1.0, size=(len(prices), prices.shape[1]))
    return pd.DataFrame(values, index=prices.index, columns=prices.columns)


@_signal(lambda p: 0)
def always_long(prices: pd.DataFrame) -> pd.DataFrame:
    """CONTROL: buy and hold. Constant ``+1``, never trades after entry.

    The benchmark that requires zero work and pays cost exactly once.  In a
    market with positive drift this is a high bar, and a great many published
    strategies do not clear it.
    """
    return pd.DataFrame(1.0, index=prices.index, columns=prices.columns)


SIGNALS: dict[str, Callable[..., pd.DataFrame]] = {
    fn.__name__: fn
    for fn in (
        zscore_reversion,
        zscore_momentum,
        timeseries_momentum,
        ma_crossover,
        vol_breakout,
        random_signal,
        always_long,
    )
}

#: The two mandatory controls. Every evaluation reports these beside the
#: strategies, with the same prominence.
CONTROLS: tuple[str, ...] = ("random_signal", "always_long")

#: Signals whose output depends on the price path. Used by the causality test
#: to verify the perturbation actually had an effect -- a signal that ignores
#: prices would pass a causality check vacuously.
PRICE_DEPENDENT: tuple[str, ...] = tuple(k for k in SIGNALS if k not in CONTROLS)


def default_signal_set() -> list[BoundSignal]:
    """The five research strategies plus the two controls, at default parameters."""
    return [SIGNALS[name].bind() for name in SIGNALS]  # type: ignore[attr-defined]
