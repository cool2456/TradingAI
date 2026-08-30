"""Shared pipeline helper for the recovery tests.

Both ``test_null_recovery`` and ``test_edge_recovery`` need to run the same
end-to-end path -- signal, volatility target, lagged execution, costs -- so it
is defined once here. If the two tests ran different pipelines, a discrepancy
between them would be uninterpretable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.engine import run_backtest, BacktestResult
from quantlab.signals import BoundSignal
from quantlab.sizing import vol_target

VOL_LOOKBACK = 60


def run_strategy(
    prices: pd.DataFrame,
    signal: BoundSignal,
    cost_bps: float = 2.0,
    target_ann_vol: float | None = 0.10,
    max_leverage: float = 3.0,
    lag: int = 1,
) -> BacktestResult:
    """Signal -> volatility target -> lagged execution -> costs."""
    target = signal(prices)
    warmup = signal.lookback
    if target_ann_vol is not None:
        returns = prices.pct_change()
        target = vol_target(
            target, returns, target_ann_vol, lookback=VOL_LOOKBACK, max_leverage=max_leverage
        )
        warmup += VOL_LOOKBACK + 1
    return run_backtest(
        prices, target, cost_bps=cost_bps, lag=lag, warmup=warmup,
        config={"signal": signal.label, "target_ann_vol": target_ann_vol},
    )


def t_stat_of_mean(values: np.ndarray | list[float]) -> float:
    """One-sample t-statistic of the mean against zero."""
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size < 2 or arr.std(ddof=1) == 0:
        return float("nan")
    return float(arr.mean() / (arr.std(ddof=1) / np.sqrt(arr.size)))


# --------------------------------------------------------------------------
# Phase 2: synthetic panels for IC calibration
# --------------------------------------------------------------------------


def synthetic_panel(
    n_bars: int,
    n_symbols: int,
    horizon: int,
    signal_window: int,
    seed: int,
    planted_ic: float = 0.0,
    return_correlation: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A ``(signal, forward_return)`` panel pair with a known planted IC.

    The signal is a trailing rolling mean, which is the shape of every signal in
    :mod:`quantlab.signals` and -- crucially -- is strongly autocorrelated. An
    iid signal would show no variance inflation at all regardless of how much
    the forward returns overlap, because the product's autocovariance is the
    product of the two series' autocovariances.

    ``forward_return[t] = sum(r[t+1 .. t+horizon])``, so consecutive
    observations share ``horizon - 1`` periods exactly as they do on real bars.
    """
    rng = np.random.default_rng(seed)
    common = rng.normal(size=(n_bars, 1))
    idio = rng.normal(size=(n_bars, n_symbols))
    if return_correlation > 0:
        raw = (
            np.sqrt(return_correlation) * common
            + np.sqrt(1.0 - return_correlation) * idio
        )
    else:
        raw = idio

    index = pd.date_range("2024-01-01", periods=n_bars, freq="min", tz="UTC")
    columns = [f"S{i}" for i in range(n_symbols)]
    returns = pd.DataFrame(raw, index=index, columns=columns)
    forward = returns.shift(-1).rolling(horizon).sum().shift(-(horizon - 1))

    base = pd.DataFrame(rng.normal(size=(n_bars, n_symbols)), index=index, columns=columns)
    signal = base.rolling(signal_window).mean()

    if planted_ic:
        stacked = forward.stack()
        z_forward = (forward - stacked.mean()) / stacked.std()
        z_noise = (signal - signal.stack().mean()) / signal.stack().std()
        signal = planted_ic * z_forward + np.sqrt(max(1.0 - planted_ic**2, 0.0)) * z_noise
    return signal, forward
