"""Causal regime classification.

Every estimator here uses a trailing window that ends at ``t`` and is therefore
knowable at ``t``.  Regime labels feed the allocator, which applies its own
lag, so nothing in this module may consult the future -- and the causality
property test in ``tests/test_causality.py`` is run over these functions with
exactly the same harness used for the signals.

The three tiers
---------------
Regime conditioning multiplies the hypothesis space.  With ``S`` strategies and
``R`` regimes you are fitting ``S x R`` decisions, and the Sharpe attributable
to luck alone rises accordingly (see :func:`hypothesis_cost`).  The tiers exist
so that the strength of the underlying claim is visible in the API rather than
buried in a config file:

**Tier 1 -- volatility regime, drives position sizing.**  Well supported.
Volatility is strongly autocorrelated and genuinely forecastable at horizons of
days to weeks; the GARCH literature is thirty years of confirmation.  This is
standard practice, not a research bet.

**Tier 2 -- trend/chop regime, drives strategy weights.**  Testable but
unproven.  The claim that trend-following works better in trending markets is
close to tautological in-sample and much weaker out-of-sample, because the
label is estimated from the same prices being traded.  Must be validated out of
sample and charged for in the deflated Sharpe.

**Tier 3 -- fine-grained regimes, drive strategy selection.**  Dangerous.
Twelve regimes across a decade of daily data leaves a few hundred bars per
regime, which is not enough to estimate anything.  Requires an explicit
``allow_tier3=True``.
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
import pandas as pd

from .metrics import expected_max_sharpe

__all__ = [
    "realized_vol",
    "vol_regime",
    "efficiency_ratio",
    "autocorr_regime",
    "classify",
    "regime_sample_counts",
    "hypothesis_cost",
    "VOL_LABELS",
    "TREND_LABELS",
]

VOL_LABELS = ("low_vol", "normal_vol", "high_vol")
TREND_LABELS = ("chop", "trend")

#: Bars required before an expanding quantile is trusted. Too few and the
#: buckets are noise; too many and the strategy has no regime for years. Half a
#: trading year is the compromise, and it is a genuine cost of doing this
#: causally rather than a tuning knob.
DEFAULT_MIN_PERIODS = 126


def _log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices).diff()


def _market_series(frame: pd.DataFrame) -> pd.Series:
    """Collapse a per-asset panel to one market-wide series by cross-sectional mean.

    Regimes in this library are market-wide conditions, not per-asset ones.
    That is a modelling assumption, and it is the assumption the
    :func:`quantlab.simulate.regime_switching` generator is built to match.
    """
    return frame.mean(axis=1)


def realized_vol(
    prices: pd.DataFrame, window: int = 20, periods_per_year: float = 252.0
) -> pd.DataFrame:
    """Annualised trailing volatility of log returns over ``window`` bars.

    ``sd(r_{t-window+1..t}) * sqrt(periods_per_year)``, with ``ddof=1``.  The
    window ends at ``t``, so the value is available at ``t``; the execution lag
    is applied downstream, once, by the engine.
    """
    return _log_returns(prices).rolling(window).std(ddof=1) * np.sqrt(periods_per_year)


def vol_regime(
    prices: pd.DataFrame,
    window: int = 20,
    quantiles: tuple[float, float] = (0.33, 0.67),
    min_periods: int = DEFAULT_MIN_PERIODS,
    periods_per_year: float = 252.0,
) -> pd.Series:
    """Bucket trailing volatility into low / normal / high using **expanding** quantiles.

    The cut points at ``t`` are the ``quantiles`` of the volatility history up
    to and including ``t``.  Using full-sample quantiles instead -- computing
    ``vol.quantile(0.67)`` once over the whole series -- is look-ahead bias,
    and a particularly seductive kind: it feels like a description of the data
    rather than a prediction, so it survives review. It means that in 2015 the
    classifier already knows what counts as a high-volatility day in 2020.

    ``tests/test_causality.py`` checks for exactly this by verifying that early
    labels do not change when later data is appended.

    Before ``min_periods`` observations the label is NaN.  A regime that cannot
    yet be estimated is not "normal"; it is unknown, and the allocator treats
    it as a reason to fall back to the unconditional weights.
    """
    lo, hi = quantiles
    if not 0 < lo < hi < 1:
        raise ValueError(f"quantiles must satisfy 0 < lo < hi < 1, got {quantiles}")

    vol = _market_series(realized_vol(prices, window, periods_per_year))
    cut_lo = vol.expanding(min_periods=min_periods).quantile(lo)
    cut_hi = vol.expanding(min_periods=min_periods).quantile(hi)

    labels = pd.Series(pd.NA, index=prices.index, dtype="object")
    known = cut_lo.notna() & vol.notna()
    labels[known & (vol <= cut_lo)] = VOL_LABELS[0]
    labels[known & (vol > cut_lo) & (vol < cut_hi)] = VOL_LABELS[1]
    labels[known & (vol >= cut_hi)] = VOL_LABELS[2]
    return pd.Series(
        pd.Categorical(labels, categories=VOL_LABELS, ordered=True),
        index=prices.index,
        name="vol_regime",
    )


def efficiency_ratio(prices: pd.DataFrame, window: int = 20) -> pd.Series:
    """Kaufman's efficiency ratio: ``|sum(r)| / sum(|r|)`` over ``window`` bars.

    Bounded in ``[0, 1]``.  It is net displacement divided by total distance
    travelled: 1.0 means the price moved in a straight line, 0.0 means it
    returned exactly to where it started.  This is the trend/chop
    discriminator, and unlike a moving-average slope it is scale-free -- a
    quiet trend and a violent trend score the same.
    """
    returns = _log_returns(prices)
    displacement = returns.rolling(window).sum().abs()
    distance = returns.abs().rolling(window).sum()
    ratio = displacement / distance.where(distance > 0)
    return _market_series(ratio).rename("efficiency_ratio")


def autocorr_regime(prices: pd.DataFrame, window: int = 60) -> pd.Series:
    """Trailing lag-1 autocorrelation of log returns over ``window`` bars.

    ``corr(r_t, r_{t-1})`` estimated on the trailing window.  Positive suggests
    momentum, negative suggests reversion.

    Read with caution.  The standard error of a correlation on ``n`` points is
    about ``1/sqrt(n)``, so a 60-bar window gives +/-0.13 of pure noise, which
    is larger than almost any autocorrelation real markets exhibit.  Bucketing
    this into a regime label mostly buckets noise, which is why it appears only
    in Tier 3.
    """
    returns = _log_returns(prices)
    per_asset = returns.rolling(window).corr(returns.shift(1))
    return _market_series(per_asset).rename("autocorr")


def classify(
    prices: pd.DataFrame,
    tier: Literal[1, 2, 3] = 2,
    vol_window: int = 20,
    trend_window: int = 20,
    autocorr_window: int = 60,
    quantiles: tuple[float, float] = (0.33, 0.67),
    trend_quantile: float = 0.5,
    min_periods: int = DEFAULT_MIN_PERIODS,
    allow_tier3: bool = False,
) -> pd.Series:
    """Categorical regime labels at the requested tier.

    - ``tier=1``: volatility bucket -- ``low_vol`` / ``normal_vol`` / ``high_vol``.
    - ``tier=2``: trend/chop -- efficiency ratio against its **expanding**
      ``trend_quantile``. An expanding cut point rather than a fixed constant
      like 0.3, for the same causality reason as the volatility buckets, and
      because the level of the efficiency ratio depends on the sampling
      frequency and the asset.
    - ``tier=3``: the cross product of volatility (3), trend (2) and the sign
      of trailing autocorrelation (2) -- **12 regimes**.

    Tier 3 requires ``allow_tier3=True``.  Twelve regimes over 2500 daily bars
    is roughly 200 bars each, and the rarest cells will hold far fewer.  A
    Sharpe estimated on 200 observations has a standard error near
    ``sqrt(252/200) = 1.1`` annualised -- the estimate is pure noise.  Check
    :func:`regime_sample_counts` before believing anything conditioned on it.
    """
    if tier == 1:
        return vol_regime(prices, vol_window, quantiles, min_periods).rename("regime")

    if tier == 2:
        ratio = efficiency_ratio(prices, trend_window)
        cut = ratio.expanding(min_periods=min_periods).quantile(trend_quantile)
        labels = pd.Series(pd.NA, index=prices.index, dtype="object")
        known = cut.notna() & ratio.notna()
        labels[known & (ratio >= cut)] = TREND_LABELS[1]
        labels[known & (ratio < cut)] = TREND_LABELS[0]
        return pd.Series(
            pd.Categorical(labels, categories=TREND_LABELS, ordered=False),
            index=prices.index,
            name="regime",
        )

    if tier == 3:
        if not allow_tier3:
            raise ValueError(
                "Tier 3 uses 12 regimes and carries high overfitting risk: each "
                "regime is fitted on a small and unevenly sized subsample, and the "
                "luck threshold rises with the total hypothesis count (see "
                "quantlab.regime.hypothesis_cost). Pass allow_tier3=True to "
                "acknowledge this."
            )
        vol = vol_regime(prices, vol_window, quantiles, min_periods)
        trend = classify(prices, tier=2, trend_window=trend_window,
                         trend_quantile=trend_quantile, min_periods=min_periods)
        autocorr = autocorr_regime(prices, autocorr_window)
        sign = pd.Series(pd.NA, index=prices.index, dtype="object")
        sign[autocorr >= 0] = "ac+"
        sign[autocorr < 0] = "ac-"

        combined = (
            vol.astype("object") + "|" + trend.astype("object") + "|" + sign
        ).where(vol.notna() & trend.notna() & sign.notna())
        categories = [
            f"{v}|{t}|{a}" for v in VOL_LABELS for t in TREND_LABELS for a in ("ac+", "ac-")
        ]
        return pd.Series(
            pd.Categorical(combined, categories=categories),
            index=prices.index,
            name="regime",
        )

    raise ValueError(f"tier must be 1, 2 or 3, got {tier}")


def regime_sample_counts(labels: pd.Series) -> pd.DataFrame:
    """Bars per regime, with the annualised Sharpe standard error each implies.

    The number that makes regime conditioning honest.  A Sharpe estimated on
    ``n`` bars has standard error roughly ``sqrt(periods_per_year / n)`` in
    annualised units, so a regime holding 150 bars supports a standard error of
    1.3 -- wider than any Sharpe you would report.  Conditioning on such a
    regime is not measurement, it is decoration.
    """
    counts = labels.value_counts(dropna=True).sort_index()
    unknown = int(labels.isna().sum())
    frame = pd.DataFrame({"n_bars": counts})
    frame["share"] = frame["n_bars"] / max(int(labels.notna().sum()), 1)
    frame["sharpe_se_ann"] = np.sqrt(252.0 / frame["n_bars"].clip(lower=1))
    frame.attrs["unlabelled_bars"] = unknown
    return frame


def hypothesis_cost(
    n_strategies: int, n_regimes: int, n_periods: int
) -> dict[str, float]:
    """The multiple-testing cost of regime conditioning, in Sharpe units.

    Compares the luck threshold for ``S`` unconditional strategies against
    ``S x R`` regime-conditional ones, using the Bailey-Lopez de Prado expected
    maximum (see :func:`quantlab.metrics.expected_max_sharpe`).

    Returns annualised thresholds and the multiplier between them.  For S=5 and
    R=4 the multiplier is about 1.6 -- moving to regime-conditional allocation
    raises the bar a strategy must clear by roughly 60%, before it has earned a
    single basis point.

    Note this counts only the *hypothesis* inflation.  It does not price the
    second cost, which is that each regime-conditional decision is estimated on
    the subsample where that regime holds -- roughly ``n_periods / R`` bars --
    so the individual estimates are noisier too. :func:`regime_sample_counts`
    reports that half.
    """
    if n_strategies < 1 or n_regimes < 1 or n_periods < 2:
        raise ValueError("n_strategies, n_regimes >= 1 and n_periods >= 2 required")

    variance = 1.0 / (n_periods - 1)          # per-period, under the null
    plain = expected_max_sharpe(n_strategies, variance)
    conditioned = expected_max_sharpe(n_strategies * n_regimes, variance)
    scale = np.sqrt(252.0)
    return {
        "n_strategies": float(n_strategies),
        "n_regimes": float(n_regimes),
        "n_hypotheses": float(n_strategies * n_regimes),
        "luck_threshold_ann": float(plain * scale),
        "luck_threshold_conditioned_ann": float(conditioned * scale),
        "multiplier": float(conditioned / plain) if plain > 0 else float("nan"),
        "bars_per_regime": float(n_periods / n_regimes),
        "sharpe_se_per_regime_ann": float(np.sqrt(252.0 * n_regimes / n_periods)),
    }
