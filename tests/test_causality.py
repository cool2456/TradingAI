"""Causality property tests -- the most important file in the repository.

Every result the library produces is conditional on this file passing. A
strategy that reads the future does not have a small bias; it has an
unbounded one, and no downstream statistic can detect or correct it.

The property
------------
For a causal function ``f`` and any index ``t``::

    f(prices)[:t]  ==  f(prices with everything after t replaced)[:t]

We check it by generating a price path, replacing the entire tail after a
randomly chosen ``t`` with a different random path, recomputing, and comparing
the head. Repeated over at least 20 random values of ``t`` per function.

Two things make this a real test rather than a ritual:

1. It verifies the perturbation actually mattered -- a function that ignores
   its input passes causality vacuously, so for price-dependent functions we
   assert the tail *did* change.
2. It is applied to two deliberately leaky signals that must fail. A test that
   has never rejected anything is not evidence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from quantlab.signals import (
    CONTROLS,
    PRICE_DEPENDENT,
    SIGNALS,
    BoundSignal,
    default_signal_set,
    ma_crossover,
    zscore_reversion,
)
from quantlab.simulate import gbm

N_TRIALS = 25


def _perturb_after(prices: pd.DataFrame, t: int, rng: np.random.Generator) -> pd.DataFrame:
    """Replace every row strictly after position ``t`` with a different path."""
    perturbed = prices.copy()
    tail = len(prices) - (t + 1)
    if tail <= 0:
        return perturbed
    shock = np.exp(np.cumsum(rng.normal(0.0, 0.05, size=(tail, prices.shape[1])), axis=0))
    perturbed.iloc[t + 1 :] = prices.iloc[t + 1 :].to_numpy() * shock
    return perturbed


def _as_frame(out) -> pd.DataFrame:
    """Regime classifiers return a Series; signals return a DataFrame."""
    if isinstance(out, pd.Series):
        return out.to_frame("value")
    return out


def assert_causal(
    fn,
    prices: pd.DataFrame,
    *,
    lookback: int = 0,
    price_dependent: bool = True,
    n_trials: int = N_TRIALS,
    seed: int = 0,
    label: str = "function",
) -> None:
    """Assert ``fn(prices)`` at every index ``<= t`` is unaffected by data after ``t``.

    Shared by the signal tests here and the regime-classifier tests, so that
    both are held to exactly the same standard.
    """
    rng = np.random.default_rng(seed)
    baseline = _as_frame(fn(prices))
    n = len(prices)
    low, high = max(lookback + 5, 10), n - 10
    assert high > low, "price series too short for the requested lookback"

    tail_changed = False
    for _ in range(n_trials):
        t = int(rng.integers(low, high))
        recomputed = _as_frame(fn(_perturb_after(prices, t, rng)))

        pd.testing.assert_frame_equal(
            baseline.iloc[: t + 1],
            recomputed.iloc[: t + 1],
            check_exact=False,
            rtol=0,
            atol=0,
            obj=f"{label} at t={t}",
        )
        if not baseline.iloc[t + 1 :].equals(recomputed.iloc[t + 1 :]):
            tail_changed = True

    if price_dependent:
        assert tail_changed, (
            f"{label} produced identical output after the perturbation point in all "
            f"{n_trials} trials; it does not depend on prices and the causality "
            "check is vacuous"
        )


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    return gbm(n_steps=500, n_assets=3, correlation=0.3, sigma_ann=0.25, seed=42)


# ------------------------------------------------------------ the property


@pytest.mark.parametrize("bound", default_signal_set(), ids=lambda b: b.name)
def test_signal_is_causal(bound: BoundSignal, prices: pd.DataFrame) -> None:
    assert_causal(
        bound,
        prices,
        lookback=bound.lookback,
        price_dependent=bound.name in PRICE_DEPENDENT,
        label=bound.label,
    )


@pytest.mark.parametrize("bound", default_signal_set(), ids=lambda b: b.name)
def test_signal_is_stable_under_appended_data(bound: BoundSignal, prices: pd.DataFrame) -> None:
    """A second form of causality: yesterday's signal must not change tomorrow.

    Recomputing on a longer history must reproduce the earlier values exactly.
    A function that fails this is using a full-sample statistic somewhere --
    the single most common source of accidental look-ahead in research code.
    """
    cut = 300
    short = _as_frame(bound(prices.iloc[:cut]))
    full = _as_frame(bound(prices)).iloc[:cut]
    pd.testing.assert_frame_equal(short, full, check_exact=True, obj=bound.label)


@pytest.mark.parametrize("bound", default_signal_set(), ids=lambda b: b.name)
def test_signal_respects_position_bounds(bound: BoundSignal, prices: pd.DataFrame) -> None:
    out = bound(prices)
    finite = out.to_numpy()[np.isfinite(out.to_numpy())]
    assert finite.size > 0
    assert finite.min() >= -1.0 - 1e-12
    assert finite.max() <= 1.0 + 1e-12


@pytest.mark.parametrize("bound", default_signal_set(), ids=lambda b: b.name)
def test_lookback_attribute_is_honest(bound: BoundSignal, prices: pd.DataFrame) -> None:
    """``.lookback`` must cover every leading bar the signal cannot define.

    The engine discards this many bars as warm-up. If the attribute understates
    the true warm-up, undefined values leak into the evaluated sample.
    """
    out = bound(prices)
    if bound.lookback == 0:
        assert out.iloc[0].notna().all()
        return
    defined_from = int(out.notna().all(axis=1).to_numpy().argmax())
    assert defined_from <= bound.lookback, (
        f"{bound.label} declares lookback={bound.lookback} but is not fully "
        f"defined until bar {defined_from}"
    )


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(window=st.integers(min_value=3, max_value=80))
def test_zscore_reversion_is_causal_at_any_window(window: int) -> None:
    """Causality must not depend on a lucky parameter choice."""
    path = gbm(n_steps=260, n_assets=2, seed=7)
    bound = zscore_reversion.bind(window=window)
    assert_causal(bound, path, lookback=bound.lookback, n_trials=5, label=bound.label)


# ------------------------------------------------- the test must have teeth


def _blatant_leak(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Look-ahead by construction: compares today's price to the future mean."""
    future_mean = prices.rolling(window).mean().shift(-window)
    return np.sign(future_mean - prices).clip(-1.0, 1.0)


def _subtle_leak(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Look-ahead via full-sample normalisation.

    The rolling mean is causal, but dividing by ``prices.std()`` -- computed
    over the whole sample, including the future -- is not. This is the leak
    that survives code review, because every individual line looks reasonable.
    """
    deviation = prices - prices.rolling(window).mean()
    return (deviation / prices.std()).clip(-1.0, 1.0)


@pytest.mark.parametrize("leaky", [_blatant_leak, _subtle_leak], ids=["blatant", "subtle"])
def test_causality_check_rejects_leaky_signals(leaky, prices: pd.DataFrame) -> None:
    with pytest.raises(AssertionError):
        assert_causal(leaky, prices, lookback=20, label=leaky.__name__)


def test_appended_data_check_rejects_full_sample_normalisation(prices: pd.DataFrame) -> None:
    """The subtle leak is also caught by the stability-under-append test."""
    cut = 300
    short = _subtle_leak(prices.iloc[:cut])
    full = _subtle_leak(prices).iloc[:cut]
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(short, full, check_exact=True)


def test_controls_are_not_exempt_from_causality(prices: pd.DataFrame) -> None:
    """Both controls obey the same contract; they are held to it explicitly."""
    for name in CONTROLS:
        bound = SIGNALS[name].bind()
        assert_causal(bound, prices, price_dependent=False, label=bound.label)


def test_ma_crossover_rejects_inverted_windows() -> None:
    with pytest.raises(ValueError, match="must be shorter than"):
        ma_crossover(gbm(n_steps=100, seed=0), fast=50, slow=10)


# ==========================================================================
# Regime classifiers -- held to exactly the same standard as the signals.
# ==========================================================================

from functools import partial  # noqa: E402

from quantlab.regime import (  # noqa: E402
    DEFAULT_MIN_PERIODS,
    autocorr_regime,
    classify,
    efficiency_ratio,
    realized_vol,
    vol_regime,
)

REGIME_ESTIMATORS = {
    "realized_vol": partial(realized_vol, window=20),
    "efficiency_ratio": partial(efficiency_ratio, window=20),
    "autocorr_regime": partial(autocorr_regime, window=60),
    "vol_regime": partial(vol_regime, window=20, min_periods=60),
    "classify_tier1": partial(classify, tier=1, min_periods=60),
    "classify_tier2": partial(classify, tier=2, min_periods=60),
    "classify_tier3": partial(classify, tier=3, min_periods=60, allow_tier3=True),
}


@pytest.fixture(scope="module")
def long_prices() -> pd.DataFrame:
    """Long enough for the expanding quantiles to become defined."""
    return gbm(n_steps=800, n_assets=2, correlation=0.3, sigma_ann=0.25, seed=11)


@pytest.mark.parametrize("name", list(REGIME_ESTIMATORS), ids=list(REGIME_ESTIMATORS))
def test_regime_estimator_is_causal(name: str, long_prices: pd.DataFrame) -> None:
    assert_causal(REGIME_ESTIMATORS[name], long_prices, lookback=120, label=name)


@pytest.mark.parametrize("name", list(REGIME_ESTIMATORS), ids=list(REGIME_ESTIMATORS))
def test_regime_estimator_is_stable_under_appended_data(
    name: str, long_prices: pd.DataFrame
) -> None:
    """The expanding-quantile check the project brief asks for by name.

    ``vol_regime`` must bucket volatility using quantiles of the history so far,
    not of the whole sample. The two are indistinguishable on a single pass and
    differ exactly here: a full-sample quantile means today's label depends on
    volatility that has not happened yet, so appending future data silently
    rewrites the past.
    """
    fn = REGIME_ESTIMATORS[name]
    cut = 600
    short = _as_frame(fn(long_prices.iloc[:cut]))
    full = _as_frame(fn(long_prices)).iloc[:cut]
    pd.testing.assert_frame_equal(short, full, check_exact=True, obj=name)


def _full_sample_vol_regime(prices: pd.DataFrame, window: int = 20) -> pd.Series:
    """The look-ahead version of ``vol_regime``: quantiles over the whole sample.

    Every line is defensible on its own, which is why this bug ships. In 2015
    this classifier already knows what will count as a high-volatility day in
    2020.
    """
    vol = realized_vol(prices, window).mean(axis=1)
    lo, hi = vol.quantile(0.33), vol.quantile(0.67)      # <- full sample
    labels = pd.Series("normal_vol", index=prices.index, dtype="object")
    labels[vol <= lo] = "low_vol"
    labels[vol >= hi] = "high_vol"
    return labels.where(vol.notna())


def test_full_sample_quantiles_are_detected_as_look_ahead(long_prices: pd.DataFrame) -> None:
    """The expanding-quantile requirement has teeth: the naive version fails."""
    cut = 600
    short = _full_sample_vol_regime(long_prices.iloc[:cut]).to_frame()
    full = _full_sample_vol_regime(long_prices).iloc[:cut].to_frame()
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(short, full, check_exact=True)

    with pytest.raises(AssertionError):
        assert_causal(_full_sample_vol_regime, long_prices, lookback=120, label="full_sample")


def test_vol_regime_buckets_are_populated_and_ordered(long_prices: pd.DataFrame) -> None:
    """Expanding quantiles should still produce roughly balanced buckets."""
    labels = vol_regime(long_prices, window=20, min_periods=60)
    shares = labels.value_counts(normalize=True)
    assert set(shares.index) == {"low_vol", "normal_vol", "high_vol"}
    assert shares.min() > 0.15, f"a bucket is nearly empty: {shares.to_dict()}"

    vol = realized_vol(long_prices, 20).mean(axis=1)
    means = vol.groupby(labels.astype("object")).mean()
    assert means["low_vol"] < means["normal_vol"] < means["high_vol"]


def test_efficiency_ratio_is_bounded_in_unit_interval(long_prices: pd.DataFrame) -> None:
    er = efficiency_ratio(long_prices, window=20).dropna()
    assert er.min() >= 0.0 and er.max() <= 1.0


def test_tier3_requires_explicit_opt_in(long_prices: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="allow_tier3"):
        classify(long_prices, tier=3)
    assert classify(long_prices, tier=3, allow_tier3=True, min_periods=60).notna().any()
