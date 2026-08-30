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
