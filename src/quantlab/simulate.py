"""Synthetic market generators.

Every generator returns a ``pd.DataFrame`` of prices indexed by timestamp with
one column per asset (``asset_0`` .. ``asset_{n-1}``).

Parameterisation convention
---------------------------
All drift and volatility parameters are supplied in **annualised** units and
scaled internally by ``dt`` (the length of one bar in years, default
``1/252``).  This keeps parameters comparable across sampling frequencies: a
20% annual volatility means the same thing whether bars are daily or hourly.
Getting this wrong is the most common source of nonsense in simulated
backtests, so the scaling is done in exactly one place per generator.

The four generators form a ladder of falsifiability:

- :func:`gbm` is the null.  Independent increments, no exploitable structure.
  The correct answer is known to be zero, so any strategy that appears
  profitable here is measuring its own bias.
- :func:`ornstein_uhlenbeck` is a positive control with a known half-life.
  A pipeline that cannot find this edge will not find a real one.
- :func:`garch_t` adds fat tails and volatility clustering without adding
  predictable *direction*.  It is still a null for directional signals, but a
  genuine test for volatility forecasting and position sizing.
- :func:`regime_switching` is the generator the regime logic is tested
  against, because the true latent state is known and returned on request.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import lfilter

__all__ = [
    "gbm",
    "ornstein_uhlenbeck",
    "garch_t",
    "regime_switching",
    "TREND",
    "CHOP",
]

# Latent state codes for :func:`regime_switching`.
TREND = 0
CHOP = 1

_DEFAULT_START = "2015-01-01"


def _index(n_steps: int, start: str, freq: str) -> pd.DatetimeIndex:
    """Build the timestamp index shared by every generator."""
    return pd.date_range(start=start, periods=n_steps, freq=freq)


def _columns(n_assets: int) -> list[str]:
    return [f"asset_{i}" for i in range(n_assets)]


def _correlated_normals(
    n_steps: int,
    n_assets: int,
    correlation: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw ``(n_steps, n_assets)`` standard normals with equicorrelation.

    The target correlation matrix is ``R = (1 - rho) I + rho 11'``.  This is
    positive definite iff ``-1/(n-1) < rho <= 1``; a uniform correlation more
    negative than that is not achievable by any set of random variables,
    because you cannot have many things all strongly disagree with each other
    at once.  We reject such inputs rather than silently repairing them.
    """
    if n_assets < 1:
        raise ValueError("n_assets must be >= 1")
    if not -1.0 < correlation <= 1.0:
        raise ValueError("correlation must lie in (-1, 1]")
    if n_assets > 1:
        lower_bound = -1.0 / (n_assets - 1)
        if correlation <= lower_bound:
            raise ValueError(
                f"equicorrelation {correlation} is not positive definite for "
                f"{n_assets} assets; requires correlation > {lower_bound:.4f}"
            )

    shocks = rng.standard_normal((n_steps, n_assets))
    if n_assets == 1 or correlation == 0.0:
        return shocks

    corr = np.full((n_assets, n_assets), correlation)
    np.fill_diagonal(corr, 1.0)
    chol = np.linalg.cholesky(corr)
    return shocks @ chol.T


def _to_prices(
    log_prices: np.ndarray,
    n_steps: int,
    n_assets: int,
    start: str,
    freq: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        np.exp(log_prices),
        index=_index(n_steps, start, freq),
        columns=_columns(n_assets),
    )


def gbm(
    n_steps: int = 1000,
    n_assets: int = 1,
    mu_ann: float = 0.0,
    sigma_ann: float = 0.20,
    correlation: float = 0.0,
    s0: float = 100.0,
    dt: float = 1.0 / 252.0,
    seed: int | None = None,
    start: str = _DEFAULT_START,
    freq: str = "B",
) -> pd.DataFrame:
    """Geometric Brownian motion -- the null hypothesis.

    Log prices follow ``d log S = (mu - sigma^2 / 2) dt + sigma dW``, so that
    ``E[S_t] = S_0 exp(mu t)``.  The ``-sigma^2 / 2`` Ito correction is what
    makes ``mu`` the *arithmetic* drift rather than the log drift; omitting it
    would make a high-volatility asset silently drift upward.

    Increments are independent, so the conditional expectation of any future
    return given any function of the past is zero.  No causal strategy has
    positive expected gross return here, and every causal strategy has
    negative expected net return once costs are paid.

    Parameters
    ----------
    mu_ann, sigma_ann
        Annualised arithmetic drift and volatility.
    correlation
        Equicorrelation between asset log-return shocks.
    """
    rng = np.random.default_rng(seed)
    shocks = _correlated_normals(n_steps, n_assets, correlation, rng)
    increments = (mu_ann - 0.5 * sigma_ann**2) * dt + sigma_ann * np.sqrt(dt) * shocks
    increments[0] = 0.0
    log_prices = np.log(s0) + np.cumsum(increments, axis=0)
    return _to_prices(log_prices, n_steps, n_assets, start, freq)


def ornstein_uhlenbeck(
    n_steps: int = 1000,
    n_assets: int = 1,
    half_life: float = 10.0,
    sigma_ann: float = 0.20,
    correlation: float = 0.0,
    s0: float = 100.0,
    dt: float = 1.0 / 252.0,
    seed: int | None = None,
    start: str = _DEFAULT_START,
    freq: str = "B",
) -> pd.DataFrame:
    """Mean-reverting log price -- a positive control with a known half-life.

    Log price follows ``d X = -theta (X - X_0) dt + sigma dW`` with
    ``theta = ln(2) / half_life`` where ``half_life`` is measured in **bars**,
    so a deviation from the long-run level decays to half its size in
    ``half_life`` bars.

    The exact (not Euler) discretisation is used::

        X_{t+1} - X_0 = phi (X_t - X_0) + sqrt(v) z_t
        phi = exp(-theta_bar)
        v   = sigma_bar^2 (1 - phi^2) / (2 theta_bar)

    Euler discretisation would bias the realised half-life whenever
    ``theta * dt`` is not small; the exact form is correct at any step size.

    This is an AR(1) recursion.  Rather than a Python loop we use
    ``scipy.signal.lfilter``, which evaluates the same recursion in compiled
    code -- the recursion is genuinely sequential, but it need not be
    sequential *in Python*.
    """
    if half_life <= 0:
        raise ValueError("half_life must be positive")
    rng = np.random.default_rng(seed)

    theta_bar = np.log(2.0) / half_life          # per-bar mean-reversion rate
    sigma_bar = sigma_ann * np.sqrt(dt)          # per-bar volatility
    phi = np.exp(-theta_bar)
    stationary_var = sigma_bar**2 / (2.0 * theta_bar)
    innov_std = np.sqrt(stationary_var * (1.0 - phi**2))

    shocks = _correlated_normals(n_steps, n_assets, correlation, rng) * innov_std
    shocks[0] = 0.0
    # AR(1): y_t = phi y_{t-1} + e_t, evaluated column-wise in compiled code.
    deviations = lfilter([1.0], [1.0, -phi], shocks, axis=0)
    log_prices = np.log(s0) + deviations
    return _to_prices(log_prices, n_steps, n_assets, start, freq)


def garch_t(
    n_steps: int = 1000,
    n_assets: int = 1,
    target_ann_vol: float = 0.20,
    alpha: float = 0.08,
    beta: float = 0.90,
    nu: float = 5.0,
    mu_ann: float = 0.0,
    correlation: float = 0.0,
    s0: float = 100.0,
    dt: float = 1.0 / 252.0,
    seed: int | None = None,
    start: str = _DEFAULT_START,
    freq: str = "B",
) -> pd.DataFrame:
    """GARCH(1,1) variance with standardised Student-t shocks.

    The conditional variance follows
    ``sigma_t^2 = omega + alpha eps_{t-1}^2 + beta sigma_{t-1}^2`` with
    ``eps_t = sigma_t z_t`` and ``z_t`` iid standardised Student-t.

    Parameterisation: ``target_ann_vol`` sets the *long-run* annualised
    volatility, from which ``omega = target_ann_vol^2 dt (1 - alpha - beta)``.
    This is the annual-units convention -- you specify the volatility level you
    want and ``alpha``/``beta`` control only its dynamics, not its level.

    ``z_t`` is a Student-t with ``nu`` degrees of freedom divided by
    ``sqrt(nu / (nu - 2))`` so that ``Var(z_t) = 1``.  Without this rescaling
    the realised volatility would exceed ``target_ann_vol`` by that factor,
    which for ``nu = 5`` is 29%.  ``nu > 2`` is required for the variance to
    exist at all.

    Volatility clusters and tails are fat, but the conditional *mean* is
    constant.  Directionally this is still a null; it is a real test only for
    volatility forecasting and position sizing.
    """
    if not 0 <= alpha or not 0 <= beta:
        raise ValueError("alpha and beta must be non-negative")
    if alpha + beta >= 1.0:
        raise ValueError(
            f"alpha + beta = {alpha + beta} >= 1: variance is not stationary "
            "and the long-run level is undefined"
        )
    if nu <= 2.0:
        raise ValueError("nu must exceed 2 for the Student-t variance to exist")

    rng = np.random.default_rng(seed)
    long_run_var = (target_ann_vol**2) * dt
    omega = long_run_var * (1.0 - alpha - beta)

    # Correlate the *standardised* shocks across assets, then let each asset
    # run its own independent variance process.
    gauss = _correlated_normals(n_steps, n_assets, correlation, rng)
    chi2 = rng.chisquare(nu, size=(n_steps, n_assets))
    student_t = gauss / np.sqrt(chi2 / nu)
    z = student_t / np.sqrt(nu / (nu - 2.0))

    variance = np.empty((n_steps, n_assets))
    eps = np.empty((n_steps, n_assets))
    variance[0] = long_run_var
    eps[0] = 0.0
    # Genuinely recursive: sigma_t^2 depends on the *realised* shock eps_{t-1},
    # which is itself a function of sigma_{t-1}. There is no closed form and no
    # linear filter that produces this, so a Python loop is required.
    for t in range(1, n_steps):
        variance[t] = omega + alpha * eps[t - 1] ** 2 + beta * variance[t - 1]
        eps[t] = np.sqrt(variance[t]) * z[t]

    increments = (mu_ann - 0.5 * target_ann_vol**2) * dt + eps
    increments[0] = 0.0
    log_prices = np.log(s0) + np.cumsum(increments, axis=0)
    return _to_prices(log_prices, n_steps, n_assets, start, freq)


def regime_switching(
    n_steps: int = 1000,
    n_assets: int = 1,
    p_stay_trend: float = 0.97,
    p_stay_chop: float = 0.97,
    trend_drift_ann: float = 0.60,
    chop_half_life: float = 5.0,
    sigma_ann: float = 0.20,
    correlation: float = 0.0,
    s0: float = 100.0,
    dt: float = 1.0 / 252.0,
    seed: int | None = None,
    start: str = _DEFAULT_START,
    freq: str = "B",
    return_states: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.Series]:
    """Two-state Markov switching between a trending and a choppy regime.

    The latent state follows a Markov chain with transition matrix::

        P = [[p_stay_trend,     1 - p_stay_trend],
             [1 - p_stay_chop,  p_stay_chop     ]]

    In the trend state (:data:`TREND`) log price carries a persistent drift of
    magnitude ``trend_drift_ann`` whose **sign is drawn once per episode** --
    per asset -- so that trends go both ways.  A generator whose trends are
    always upward would let a permanently-long strategy masquerade as a trend
    follower.

    In the chop state (:data:`CHOP`) log price mean-reverts, with half-life
    ``chop_half_life`` bars, toward the level at which the chop episode began.
    Re-anchoring at each episode start is what makes chop *local*: without it
    the series would revert to a fixed global level and become one long
    tradeable mean-reversion, which is not what chop means.

    The latent state is shared across assets -- it is a market-wide condition,
    which is the premise the regime logic is built on -- while shocks and
    trend directions are asset-specific.

    Returns
    -------
    prices, or ``(prices, states)`` when ``return_states=True``.  ``states`` is
    an integer Series of the **true** latent regime.  It is the ground truth
    that :mod:`quantlab.regime` is scored against and must never be fed to a
    strategy.
    """
    for name, p in (("p_stay_trend", p_stay_trend), ("p_stay_chop", p_stay_chop)):
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"{name} must be a probability in [0, 1]")

    rng = np.random.default_rng(seed)
    theta_bar = np.log(2.0) / chop_half_life
    sigma_bar = sigma_ann * np.sqrt(dt)
    trend_step = trend_drift_ann * dt

    shocks = _correlated_normals(n_steps, n_assets, correlation, rng) * sigma_bar
    switch_draws = rng.random(n_steps)
    sign_draws = rng.choice([-1.0, 1.0], size=(n_steps, n_assets))

    states = np.empty(n_steps, dtype=np.int64)
    log_prices = np.empty((n_steps, n_assets))
    log_prices[0] = np.log(s0)

    # Stationary distribution of the two-state chain, used to draw t=0.
    p_trend_to_chop = 1.0 - p_stay_trend
    p_chop_to_trend = 1.0 - p_stay_chop
    denom = p_trend_to_chop + p_chop_to_trend
    pi_trend = p_chop_to_trend / denom if denom > 0 else 0.5
    states[0] = TREND if switch_draws[0] < pi_trend else CHOP

    drift_sign = sign_draws[0].copy()
    anchor = log_prices[0].copy()

    # Genuinely recursive on three counts: the Markov state depends on the
    # previous state, the chop anchor depends on where the previous episode
    # ended, and the price level feeds its own mean reversion. None of these
    # can be expressed as a fixed linear filter over the shock sequence.
    for t in range(1, n_steps):
        prev = states[t - 1]
        stay = p_stay_trend if prev == TREND else p_stay_chop
        states[t] = prev if switch_draws[t] < stay else (CHOP if prev == TREND else TREND)

        if states[t] != prev:  # episode boundary: re-draw regime-local state
            if states[t] == TREND:
                drift_sign = sign_draws[t].copy()
            else:
                anchor = log_prices[t - 1].copy()

        if states[t] == TREND:
            log_prices[t] = log_prices[t - 1] + drift_sign * trend_step + shocks[t]
        else:
            pull = theta_bar * (anchor - log_prices[t - 1])
            log_prices[t] = log_prices[t - 1] + pull + shocks[t]

    index = _index(n_steps, start, freq)
    prices = pd.DataFrame(np.exp(log_prices), index=index, columns=_columns(n_assets))
    if return_states:
        return prices, pd.Series(states, index=index, name="true_regime")
    return prices
