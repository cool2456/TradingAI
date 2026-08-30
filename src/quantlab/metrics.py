"""Performance and falsification metrics.

Unit convention -- read this before using anything here
-------------------------------------------------------
Two Sharpe conventions coexist in this module and confusing them is a serious
bug, not a cosmetic one.

- **Per-period Sharpe** ``mean(r) / std(r)`` over the sampling frequency of the
  data.  This is the only unit in which the Probabilistic and Deflated Sharpe
  formulas are valid, because their test statistics are derived from the
  sampling distribution of the estimator over ``n`` observations.
- **Annualised Sharpe** ``per_period * sqrt(periods_per_year)``.  This is a
  reporting convention only.

Passing an annualised Sharpe into :func:`probabilistic_sharpe` or
:func:`deflated_sharpe` inflates the test statistic by ``sqrt(252)`` and will
declare noise significant.  Every function here states its unit explicitly and
the DSR family accepts *per-period* inputs only.

The fastest falsification tool in the library is
:func:`information_coefficient`, not Sharpe.  Sharpe answers "did this make
money"; IC answers "does the signal contain information", which is the
question you can actually get a standard error on.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, asdict
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "sharpe",
    "sortino",
    "max_drawdown",
    "calmar",
    "turnover",
    "hit_rate",
    "information_coefficient",
    "ICResult",
    "newey_west_mean_se",
    "deflated_sharpe",
    "expected_max_sharpe",
    "probabilistic_sharpe",
    "annualize_sharpe",
    "deannualize_sharpe",
    "sharpe_variance_across_trials",
    "summary",
]

EULER_GAMMA = 0.5772156649015329


def _as_array(x: pd.Series | np.ndarray | pd.DataFrame) -> np.ndarray:
    arr = np.asarray(x, dtype=float).ravel()
    return arr[np.isfinite(arr)]


def annualize_sharpe(sr_per_period: float, periods_per_year: float = 252.0) -> float:
    """Convert a per-period Sharpe to annualised units: ``sr * sqrt(ppy)``.

    The square-root scaling assumes returns are serially uncorrelated.  Under
    positive autocorrelation it overstates the annualised Sharpe.
    """
    return float(sr_per_period * np.sqrt(periods_per_year))


def deannualize_sharpe(sr_annual: float, periods_per_year: float = 252.0) -> float:
    """Convert an annualised Sharpe back to per-period units: ``sr / sqrt(ppy)``."""
    return float(sr_annual / np.sqrt(periods_per_year))


def sharpe(
    returns: pd.Series | np.ndarray,
    periods_per_year: float = 252.0,
    risk_free_ann: float = 0.0,
    annualize: bool = True,
) -> float:
    """Sharpe ratio ``E[r - rf] / sd(r - rf)``.

    Uses the sample standard deviation with ``ddof=1``.  ``risk_free_ann`` is
    an annualised rate, converted to per-period by simple division (an
    approximation that is immaterial at realistic rates and daily frequency).

    Returns annualised units when ``annualize=True`` (the reporting default)
    and per-period units otherwise (the unit required by the DSR family).
    """
    r = _as_array(returns)
    if r.size < 2:
        return float("nan")
    excess = r - risk_free_ann / periods_per_year
    sd = excess.std(ddof=1)
    if sd == 0:
        return float("nan")
    sr = float(excess.mean() / sd)
    return annualize_sharpe(sr, periods_per_year) if annualize else sr


def sortino(
    returns: pd.Series | np.ndarray,
    periods_per_year: float = 252.0,
    target: float = 0.0,
    annualize: bool = True,
) -> float:
    """Sortino ratio ``E[r - target] / downside_deviation``.

    Downside deviation is ``sqrt(mean(min(r - target, 0)^2))`` where the mean
    is taken over **all** ``n`` observations, not only the losing ones.
    Dividing by the count of losses instead is a common variant that makes a
    strategy with few, large losses look better than it is.
    """
    r = _as_array(returns)
    if r.size < 2:
        return float("nan")
    excess = r - target
    downside = np.minimum(excess, 0.0)
    dd = np.sqrt(np.mean(downside**2))
    if dd == 0:
        return float("nan")
    sr = float(excess.mean() / dd)
    return annualize_sharpe(sr, periods_per_year) if annualize else sr


def max_drawdown(returns: pd.Series | np.ndarray, compound: bool = True) -> float:
    """Largest peak-to-trough decline of the equity curve, as a negative fraction.

    ``compound=True`` builds equity as ``prod(1 + r)``; ``compound=False`` as
    ``1 + cumsum(r)``, which is the right choice for a constant-notional book
    where profits are not reinvested.  Returns ``0.0`` for a curve that never
    declines.
    """
    r = _as_array(returns)
    if r.size == 0:
        return float("nan")
    equity = np.cumprod(1.0 + r) if compound else 1.0 + np.cumsum(r)
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1.0 if compound else (equity - peak)
    return float(drawdown.min())


def calmar(returns: pd.Series | np.ndarray, periods_per_year: float = 252.0) -> float:
    """Annualised compound return divided by the absolute maximum drawdown.

    Both numerator and denominator are sample statistics of a single path, so
    Calmar has a far wider sampling distribution than Sharpe and should not be
    used for ranking on short samples.
    """
    r = _as_array(returns)
    if r.size < 2:
        return float("nan")
    total = float(np.prod(1.0 + r))
    if total <= 0:
        return float("-inf")
    cagr = total ** (periods_per_year / r.size) - 1.0
    mdd = abs(max_drawdown(r))
    if mdd == 0:
        return float("nan")
    return float(cagr / mdd)


def turnover(positions: pd.DataFrame | pd.Series, periods_per_year: float = 252.0) -> float:
    """Annualised turnover: mean per-period sum of ``|position_t - position_{t-1}|``.

    A value of 1.0 means the book's entire notional is replaced once per year.
    Positions are notional weights, so this is directly proportional to the
    transaction cost paid at a fixed cost per unit traded.

    ``min_count=1`` keeps the first row -- where ``diff()`` is entirely NaN --
    out of the average. Without it pandas sums an all-NaN row to 0.0 and the
    spurious zero biases turnover downward by a factor of ``(n-1)/n``.
    """
    pos = positions.to_frame() if isinstance(positions, pd.Series) else positions
    per_period = pos.diff().abs().sum(axis=1, min_count=1)
    return float(per_period.mean() * periods_per_year)


def hit_rate(returns: pd.Series | np.ndarray) -> float:
    """Fraction of **non-zero** periods with a positive return.

    Zero-return periods (flat position) are excluded from both numerator and
    denominator; including them would let a strategy improve its hit rate by
    trading less.  Hit rate and P&L diverge routinely -- a strategy can win 70%
    of periods and still lose money -- which is why they must be read together.
    """
    r = _as_array(returns)
    nonzero = r[r != 0]
    if nonzero.size == 0:
        return float("nan")
    return float((nonzero > 0).mean())


# --------------------------------------------------------------------------
# Information coefficient
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ICResult:
    """Result of an information-coefficient test.

    Attributes
    ----------
    ic
        The correlation between signal and forward return.
    se_naive
        ``1 / sqrt(n - 2)``, the standard error under H0 assuming iid
        observations.  Reported for comparison only -- it is almost always an
        underestimate on financial data.
    se_hac
        Newey-West heteroskedasticity- and autocorrelation-consistent standard
        error.  This is the headline standard error.
    t_stat, p_value
        Computed from ``se_hac`` against a two-sided normal.
    """

    ic: float
    n: int
    se_naive: float
    se_hac: float
    hac_lags: int
    t_stat: float
    p_value: float
    method: str

    def to_dict(self) -> dict:
        return asdict(self)


def newey_west_mean_se(x: np.ndarray, lags: int) -> float:
    """Newey-West HAC standard error of the sample mean of ``x``.

    ``S = gamma_0 + 2 sum_{l=1}^{L} (1 - l/(L+1)) gamma_l`` with Bartlett
    weights; ``se = sqrt(S / n)``.  The Bartlett kernel guarantees ``S >= 0``
    in population but not in every finite sample, so a non-positive estimate
    falls back to the iid standard error with a warning.
    """
    n = x.size
    centred = x - x.mean()
    s = float(np.dot(centred, centred) / n)
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        gamma = float(np.dot(centred[lag:], centred[:-lag]) / n)
        s += 2.0 * weight * gamma
    if s <= 0:
        warnings.warn(
            "Newey-West long-run variance was non-positive; falling back to the "
            "iid estimate. The reported t-statistic is likely optimistic.",
            RuntimeWarning,
            stacklevel=2,
        )
        s = float(np.dot(centred, centred) / n)
    return float(np.sqrt(s / n))


def information_coefficient(
    signal: pd.Series | np.ndarray,
    forward_return: pd.Series | np.ndarray,
    method: Literal["pearson", "spearman"] = "pearson",
    overlap: int = 1,
    hac_lags: int | None = None,
) -> ICResult:
    """Correlation between a signal and the return that follows it.

    Definition: ``IC = corr(s_t, r_{t+1..t+h})``.  Written as the mean of a
    per-period product of standardised series,
    ``IC = (1/n) sum_t z_s(t) z_r(t)``, which turns the correlation into a
    sample mean and lets us put a heteroskedasticity- and
    autocorrelation-consistent standard error on it.

    Why the naive standard error is wrong
    -------------------------------------
    ``1 / sqrt(n - 2)`` assumes the ``n`` products are independent.  They are
    not.  When ``forward_return`` is a ``h``-period forward return sampled
    every period, consecutive observations share ``h - 1`` periods of the same
    realised return, and the signal itself is typically a slow-moving rolling
    statistic.  Both induce positive autocorrelation in the product series,
    which inflates the naive t-statistic by roughly ``sqrt(h)``.  With
    ``h = 20`` that is a factor of 4.5 -- more than enough to turn noise into
    a publishable result.

    ``hac_lags`` defaults to ``max(overlap - 1, floor(4 (n/100)^(2/9)))``,
    the standard Newey-West rule of thumb floored at the known overlap.

    Parameters
    ----------
    overlap
        Horizon ``h`` of the forward return, in periods.  Pass it whenever the
        forward return is not a single-period return.
    """
    s = pd.Series(np.asarray(signal, dtype=float).ravel())
    r = pd.Series(np.asarray(forward_return, dtype=float).ravel())
    if s.size != r.size:
        raise ValueError(f"signal and forward_return differ in length: {s.size} vs {r.size}")

    frame = pd.concat([s.rename("s"), r.rename("r")], axis=1).dropna()
    if method == "spearman":
        frame = frame.rank()
    n = len(frame)
    if n < 4:
        return ICResult(float("nan"), n, float("nan"), float("nan"), 0,
                        float("nan"), float("nan"), method)

    sv = frame["s"].to_numpy()
    rv = frame["r"].to_numpy()
    if sv.std(ddof=0) == 0 or rv.std(ddof=0) == 0:
        return ICResult(float("nan"), n, float("nan"), float("nan"), 0,
                        float("nan"), float("nan"), method)

    zs = (sv - sv.mean()) / sv.std(ddof=0)
    zr = (rv - rv.mean()) / rv.std(ddof=0)
    products = zs * zr
    ic = float(products.mean())

    if hac_lags is None:
        rule_of_thumb = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
        hac_lags = max(overlap - 1, rule_of_thumb, 0)
    hac_lags = min(hac_lags, n - 2)

    se_naive = float(1.0 / np.sqrt(n - 2))
    se_hac = newey_west_mean_se(products, hac_lags)
    t_stat = float(ic / se_hac) if se_hac > 0 else float("nan")
    p_value = float(2.0 * stats.norm.sf(abs(t_stat))) if np.isfinite(t_stat) else float("nan")
    return ICResult(ic, n, se_naive, se_hac, hac_lags, t_stat, p_value, method)


# --------------------------------------------------------------------------
# Multiple-testing adjusted Sharpe
# --------------------------------------------------------------------------


def sharpe_variance_across_trials(sharpes: np.ndarray | list[float]) -> float:
    """Sample variance of **per-period** Sharpe ratios across trials.

    This is the correct input to :func:`expected_max_sharpe` and
    :func:`deflated_sharpe`: the deflation asks "how high would the best of N
    trials get by luck", and that depends on how much the trials actually
    scatter.
    """
    arr = _as_array(np.asarray(sharpes, dtype=float))
    if arr.size < 2:
        return float("nan")
    return float(arr.var(ddof=1))


def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """Expected maximum of ``n_trials`` Sharpe ratios under the null. Per-period units.

    Bailey and Lopez de Prado's approximation to the expectation of the maximum
    of ``N`` iid Gaussian draws with variance ``V``::

        E[max] ~ sqrt(V) [ (1 - g) Phi^-1(1 - 1/N) + g Phi^-1(1 - 1/(N e)) ]

    where ``g`` is the Euler-Mascheroni constant.  This is more accurate in the
    relevant range than the crude asymptotic ``sqrt(2 ln N)``, which converges
    only slowly and understates the threshold for small ``N``.

    This is the **luck threshold**: the Sharpe you should expect to see from
    the best of ``N`` worthless strategies.  A strategy must clear it before
    its Sharpe is evidence of anything.  Returns ``0.0`` for ``N <= 1``, where
    the expected maximum is just the expected Sharpe, namely zero.
    """
    if n_trials <= 1:
        return 0.0
    if not np.isfinite(sharpe_variance) or sharpe_variance < 0:
        raise ValueError(f"sharpe_variance must be finite and non-negative, got {sharpe_variance}")
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(sharpe_variance) * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def probabilistic_sharpe(
    returns: pd.Series | np.ndarray,
    benchmark_sr: float = 0.0,
) -> float:
    """Probability that the true Sharpe exceeds ``benchmark_sr``.

    ``PSR(SR*) = Phi[ (SR_hat - SR*) sqrt(n - 1) /
                      sqrt(1 - g3 SR_hat + (g4 - 1)/4 SR_hat^2) ]``

    where ``g3`` is skewness and ``g4`` is **non-excess** kurtosis (3 for a
    Gaussian).  The denominator is the standard error of the Sharpe estimator
    corrected for the third and fourth moments: negative skew and fat tails
    both *widen* it, so the same Sharpe is weaker evidence when the return
    distribution is ugly.  This is exactly the correction that a plain
    ``sqrt(n)`` t-test on Sharpe omits.

    ``benchmark_sr`` and the internal Sharpe are both **per-period**.  Passing
    an annualised benchmark here compares against a threshold ``sqrt(252)``
    times too large.  Use :func:`deannualize_sharpe` first.
    """
    r = _as_array(returns)
    n = r.size
    if n < 4:
        return float("nan")
    sd = r.std(ddof=1)
    if sd == 0:
        return float("nan")
    sr = float(r.mean() / sd)
    skew = float(stats.skew(r, bias=False))
    kurt = float(stats.kurtosis(r, fisher=False, bias=False))

    variance_term = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if variance_term <= 0:
        warnings.warn(
            "The moment-corrected Sharpe variance is non-positive; the PSR "
            "approximation is invalid for this sample. Returning NaN.",
            RuntimeWarning,
            stacklevel=2,
        )
        return float("nan")
    z = (sr - benchmark_sr) * np.sqrt(n - 1) / np.sqrt(variance_term)
    return float(stats.norm.cdf(z))


def deflated_sharpe(
    returns: pd.Series | np.ndarray,
    n_trials: int,
    sharpe_variance: float | None = None,
) -> float:
    """Deflated Sharpe ratio: PSR evaluated against the luck threshold.

    ``DSR = PSR(E[max SR over n_trials])``.  It answers the only question that
    matters after a parameter sweep: given that you looked ``n_trials`` times,
    what is the probability this strategy's true Sharpe is positive?

    Parameters
    ----------
    n_trials
        Number of configurations tried.  In this library it is read from
        ``research_log.jsonl`` by :func:`quantlab.validate.trial_count` so that
        a sweep cannot be run without paying for it.
    sharpe_variance
        Variance of **per-period** Sharpes across those trials.  When omitted
        it defaults to ``1 / (n - 1)``, the sampling variance of a single
        Sharpe estimate under the null.

        That default is **anti-conservative**.  It assumes the trials differ
        only by sampling noise; if they explore genuinely different strategies
        their Sharpes scatter more, the luck threshold rises, and the true DSR
        is lower than the one returned here.  Pass
        :func:`sharpe_variance_across_trials` over the observed trial Sharpes
        whenever you have them.
    """
    r = _as_array(returns)
    if r.size < 4:
        return float("nan")
    if sharpe_variance is None:
        sharpe_variance = 1.0 / (r.size - 1)
    sr_star = expected_max_sharpe(n_trials, sharpe_variance)
    return probabilistic_sharpe(r, benchmark_sr=sr_star)


def summary(
    returns: pd.Series,
    positions: pd.DataFrame | None = None,
    periods_per_year: float = 252.0,
    n_trials: int = 1,
) -> dict[str, float]:
    """Standard metric block for one return series. All figures net of costs."""
    out = {
        "sharpe_ann": sharpe(returns, periods_per_year),
        "sharpe_per_period": sharpe(returns, periods_per_year, annualize=False),
        "sortino_ann": sortino(returns, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "calmar": calmar(returns, periods_per_year),
        "hit_rate": hit_rate(returns),
        "total_return": float(np.prod(1.0 + _as_array(returns)) - 1.0),
        "n_periods": int(len(returns)),
        "deflated_sharpe": deflated_sharpe(returns, n_trials),
        "n_trials": int(n_trials),
    }
    out["turnover_ann"] = turnover(positions, periods_per_year) if positions is not None else float("nan")
    return out


#: Kept for backward compatibility with Phase 1 internal callers.
_newey_west_mean_se = newey_west_mean_se
