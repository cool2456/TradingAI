"""Position sizing: volatility targeting and Kelly scaling.

Sizing is Tier 1 of the regime logic and the only tier that is not a research
bet.  Volatility is genuinely autocorrelated and forecastable at horizons of
days to weeks -- unlike returns, which are not -- so scaling exposure by a
trailing volatility estimate is standard practice rather than a hypothesis.

Note that the ``[-1, +1]`` bound is a contract on *signals*, not on sized
positions.  Volatility targeting deliberately produces leverage above 1 in
quiet markets, which is the entire point and also the entire danger.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["vol_target", "kelly_fraction", "realized_portfolio_vol"]


def realized_portfolio_vol(
    positions: pd.DataFrame,
    returns: pd.DataFrame,
    lookback: int = 60,
    periods_per_year: float = 252.0,
) -> pd.Series:
    """Trailing annualised volatility of the unscaled strategy's own returns.

    Estimated from ``(positions.shift(1) * returns).sum(axis=1)`` -- the return
    the strategy would have earned -- rather than from asset volatility alone,
    because a strategy that trades in and out has far lower realised volatility
    than the assets it trades.

    The estimate is ``.rolling(lookback).std().shift(1)``: the ``shift(1)`` is
    mandatory.  Without it the scale applied at ``t`` would incorporate the
    volatility of the bar it is being applied to, which is a look-ahead that
    flatters exactly the periods that matter -- the volatile ones.
    """
    strategy_returns = (positions.shift(1) * returns).sum(axis=1, min_count=1)
    return (
        strategy_returns.rolling(lookback).std(ddof=1).shift(1) * np.sqrt(periods_per_year)
    ).rename("realized_vol")


def vol_target(
    positions: pd.DataFrame,
    returns: pd.DataFrame,
    target_ann_vol: float = 0.10,
    lookback: int = 60,
    max_leverage: float = 3.0,
    periods_per_year: float = 252.0,
) -> pd.DataFrame:
    """Scale positions so realised portfolio volatility tracks ``target_ann_vol``.

    ``scale_t = clip(target / vol_hat_{t-1}, 0, max_leverage)`` applied to every
    asset's position at ``t``.

    Why the leverage cap is not optional
    ------------------------------------
    The scale is inversely proportional to a trailing volatility estimate, so
    as realised volatility falls the position size grows without bound.
    Volatility is not only autocorrelated, it is *mean-reverting*: the quietest
    periods are the ones most likely to be followed by an expansion.  An
    uncapped vol target therefore builds its largest position immediately
    before the move that will punish it.  The cap converts an unbounded tail
    into a bounded one.

    During warm-up -- and wherever the volatility estimate is zero or undefined
    -- the scale is **0**, not 1.  A missing risk estimate is a reason to hold
    nothing, not a reason to hold a default amount.
    """
    if target_ann_vol <= 0:
        raise ValueError("target_ann_vol must be positive")
    if max_leverage <= 0:
        raise ValueError("max_leverage must be positive")

    vol_hat = realized_portfolio_vol(positions, returns, lookback, periods_per_year)
    scale = (target_ann_vol / vol_hat.where(vol_hat > 0)).clip(upper=max_leverage)
    return positions.mul(scale.fillna(0.0), axis=0)


def kelly_fraction(mu: float, sigma: float, fraction: float = 0.25) -> float:
    """Fractional Kelly position: ``fraction * mu / sigma^2``.

    Two growth rates, often confused
    --------------------------------
    A position of size ``f`` in an asset with arithmetic drift ``mu`` and
    volatility ``sigma`` compounds at the **log**-growth rate
    ``g(f) = f mu - f^2 sigma^2 / 2``.  At ``f = 1`` this is ``mu - sigma^2/2``:
    a fully-invested position grows more slowly than its arithmetic mean
    return, by half its variance.  That gap is not a fee, it is the difference
    between the average of the outcomes and the outcome of the average --
    losing 50% then gaining 50% leaves you down 25%.

    Maximising ``g`` over ``f`` gives the Kelly optimum ``f* = mu / sigma^2``,
    at which the growth rate is ``mu^2 / (2 sigma^2) = SR^2 / 2``.

    Why full Kelly is the wrong target here
    ---------------------------------------
    ``f*`` depends on ``mu``, which is never known and is estimated with
    standard error ``sigma / sqrt(T)``.  At a Sharpe of 0.5 and ten years of
    daily data, the standard error on ``mu`` is roughly 60% of ``mu`` itself,
    so the Kelly estimate is off by a similar proportion.

    The penalty is brutally asymmetric.  Writing ``f = c f*``, the fraction of
    maximum growth retained is ``2c - c^2``.  Underbetting at ``c = 0.5``
    retains 75% of the growth; overbetting at ``c = 2`` retains **zero**, and
    beyond that the growth rate is negative -- you can lose money with a
    positive-edge strategy purely by sizing it wrong.  Since the estimation
    error is symmetric but the payoff is not, the rational response is to bet
    well below the estimate.

    The default ``fraction=0.25`` retains ``2(0.25) - 0.25^2 = 43.75%`` of the
    theoretical maximum growth at a quarter of the volatility, and keeps
    ``c = 2`` four estimation errors away rather than one.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must lie in (0, 1]; full Kelly is fraction=1.0")
    return float(fraction * mu / sigma**2)
