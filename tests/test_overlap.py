"""The overlapping-window problem, measured rather than asserted.

Forward returns computed at every bar over ``h`` bars share ``h - k`` of their
``h`` periods with the observation ``k`` bars earlier. This file checks the
structure that creates, the standard-error correction it requires, and the
condition under which it does *not* apply -- which the project brief's framing
omits and which changes how every number in the IC report should be read.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import synthetic_panel
from quantlab.live.ic import compute_ic
from quantlab.live.featurestore import forward_returns


# ------------------------------------------------- the structure of overlap


@pytest.mark.parametrize("horizon", [5, 30, 130])
def test_forward_return_autocorrelation_is_h_minus_k_over_h(horizon: int) -> None:
    """Overlapping sums of iid increments have autocorrelation ``(h-k)/h``.

    At ``h = 130`` the lag-1 autocorrelation is 0.992. Every IID standard error
    computed on such a series is claiming those observations are independent.
    """
    rng = np.random.default_rng(0)
    index = pd.date_range("2024-01-01", periods=60000, freq="min", tz="UTC")
    returns = pd.Series(rng.normal(size=60000), index=index)
    overlapping = returns.shift(-1).rolling(horizon).sum().shift(-(horizon - 1)).dropna()

    for lag in (1, horizon // 4, horizon // 2):
        if lag < 1:
            continue
        empirical = overlapping.autocorr(lag)
        expected = (horizon - lag) / horizon
        assert empirical == pytest.approx(expected, abs=0.03), (
            f"h={horizon} lag={lag}: got {empirical:.4f}, expected {expected:.4f}"
        )


def test_lag_one_autocorrelation_at_h_130_is_0992() -> None:
    """The specific figure the brief quotes, confirmed."""
    rng = np.random.default_rng(1)
    returns = pd.Series(rng.normal(size=120000))
    overlapping = returns.shift(-1).rolling(130).sum().shift(-129).dropna()
    assert overlapping.autocorr(1) == pytest.approx(0.992, abs=0.004)


# ------------------------------- HAC exceeds IID, and the gap grows with h


def test_hac_standard_error_exceeds_iid_on_overlapping_data() -> None:
    signal, forward = synthetic_panel(40000, 10, horizon=130, signal_window=130, seed=3)
    result = compute_ic(signal, forward, horizon_bars=130, residualise=False)
    assert result.se_driscoll_kraay > result.se_iid
    assert result.se_driscoll_kraay / result.se_iid > 3


def test_the_correction_grows_with_horizon() -> None:
    """The ratio of HAC to IID standard error must increase with the horizon.

    This is the property that makes long-horizon claims so much harder to
    support than short-horizon ones: the effective sample shrinks in ``h``
    while the calendar cost of collecting it grows in ``h``.
    """
    ratios = []
    for horizon in (5, 30, 130):
        signal, forward = synthetic_panel(60000, 8, horizon=horizon,
                                          signal_window=horizon, seed=5)
        result = compute_ic(signal, forward, horizon_bars=horizon, residualise=False)
        ratios.append(result.se_driscoll_kraay / result.se_iid)

    assert ratios == sorted(ratios), f"ratio did not increase with horizon: {ratios}"
    assert ratios[-1] > 2 * ratios[0]


def test_variance_inflation_grows_with_horizon() -> None:
    inflations = []
    for horizon in (5, 30, 130):
        signal, forward = synthetic_panel(60000, 8, horizon=horizon,
                                          signal_window=horizon, seed=7)
        inflations.append(
            compute_ic(signal, forward, horizon, residualise=False).variance_inflation
        )
    assert inflations == sorted(inflations)
    assert inflations[-1] > 20


# ------------------------------------- the condition the brief's framing omits


def test_overlap_alone_does_not_inflate_anything() -> None:
    """An iid signal against heavily overlapping returns shows no inflation.

    ``Cov(x_t, x_{t-k}) = Cov(s_t, s_{t-k}) * Cov(r_t, r_{t-k})`` for
    independent mean-zero series, so if either factor is zero the product is
    uncorrelated no matter how severe the overlap. The brief attributes the
    inflation to the forward returns alone; it takes both.

    This matters practically: it means ``n/h`` is an upper bound on the damage,
    reached only when the signal is as persistent as the return window, and the
    measured ``n_effective`` is the honest figure to report.
    """
    for horizon in (30, 130, 390):
        signal, forward = synthetic_panel(40000, 5, horizon=horizon,
                                          signal_window=1, seed=11)
        result = compute_ic(signal, forward, horizon, residualise=False)
        assert result.variance_inflation == pytest.approx(1.0, abs=0.4), (
            f"h={horizon} with an iid signal inflated by {result.variance_inflation:.2f}"
        )


@pytest.mark.parametrize("signal_window", [1, 30, 130])
def test_inflation_is_driven_by_signal_persistence(signal_window: int) -> None:
    """Holding the horizon fixed, inflation rises with signal persistence."""
    signal, forward = synthetic_panel(40000, 5, horizon=130,
                                      signal_window=signal_window, seed=13)
    inflation = compute_ic(signal, forward, 130, residualise=False).variance_inflation
    if signal_window == 1:
        assert inflation < 2
    else:
        assert inflation > 10


# ------------------------------------------------------- session boundaries


def test_within_session_masking_removes_overnight_windows() -> None:
    """A forward window that crosses a session prices a gap nobody held."""
    index = pd.DatetimeIndex(
        np.concatenate([
            pd.date_range(f"2024-03-{day:02d} 14:30", periods=390, freq="min", tz="UTC")
            for day in (1, 4, 5)
        ])
    )
    rng = np.random.default_rng(17)
    panel = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.0004, size=(len(index), 2)), axis=0)),
        index=index, columns=["AAA", "BBB"],
    )
    for horizon in (30, 130):
        masked = forward_returns(panel, horizon, within_session=True).notna().sum().sum()
        unmasked = forward_returns(panel, horizon, within_session=False).notna().sum().sum()
        assert masked == 3 * (390 - horizon) * 2
        assert unmasked > masked


def test_horizon_at_session_length_has_no_within_session_windows() -> None:
    """h=390 on 390-bar sessions is empty by construction, and says so.

    Not a bug: a 390-bar forward return starting anywhere in a 390-bar session
    always lands in the next one. The horizon must either shorten or be
    reported as an overnight-inclusive return, which is a different quantity
    driven by gap risk rather than by microstructure.
    """
    index = pd.DatetimeIndex(
        np.concatenate([
            pd.date_range(f"2024-03-{day:02d} 14:30", periods=390, freq="min", tz="UTC")
            for day in (1, 4)
        ])
    )
    panel = pd.DataFrame(100.0, index=index, columns=["AAA"])
    panel["AAA"] = np.linspace(100, 110, len(index))
    with pytest.raises(ValueError, match="OVERNIGHT-INCLUSIVE"):
        forward_returns(panel, 390, within_session=True)
    assert forward_returns(panel, 390, within_session=False).notna().sum().sum() > 0
