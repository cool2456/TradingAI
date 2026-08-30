"""Calibration of the IC estimator against synthetic data with known answers.

The estimator is the deliverable of Phase 2, so it is validated the same way
the Phase 1 metrics were: by Monte Carlo against a known null and a known
planted effect, rather than against a published arithmetic example. A published
example checks one evaluation; a calibration check exercises the whole sampling
distribution, which is the thing actually being claimed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import synthetic_panel
from quantlab.live.ic import (
    ICReport,
    compute_ic,
    cross_sectional_demean,
    rolling_beta_residual,
)


# ------------------------------------------------------------ recovery


@pytest.mark.parametrize("planted", [0.0, 0.03, 0.08])
def test_planted_ic_is_recovered(planted: float) -> None:
    signal, forward = synthetic_panel(20000, 20, horizon=30, signal_window=30,
                                      seed=7, planted_ic=planted)
    result = compute_ic(signal, forward, horizon_bars=30, residualise=False)
    assert result.ic == pytest.approx(planted, abs=0.008)


def test_a_real_effect_still_clears_the_corrected_bar() -> None:
    """The correction must not be so blunt that it destroys genuine signal."""
    signal, forward = synthetic_panel(20000, 20, horizon=30, signal_window=30,
                                      seed=11, planted_ic=0.05)
    result = compute_ic(signal, forward, horizon_bars=30, residualise=False)
    assert result.t_hac > 4
    assert result.p_value < 1e-4


def test_sign_of_the_ic_is_recovered() -> None:
    signal, forward = synthetic_panel(15000, 15, horizon=30, signal_window=30,
                                      seed=13, planted_ic=-0.05)
    assert compute_ic(signal, forward, 30, residualise=False).ic < -0.03


# --------------------------------------------------- effective sample size


def test_effective_sample_size_collapses_toward_n_over_h() -> None:
    """``n_effective`` must land near ``n/h``, not near ``n``.

    The exact relationship is not ``n/h``. For a signal of persistence ``w``
    and a horizon ``h``, the product's autocorrelation is the product of the
    two triangular kernels, ``((w-k)/w)((h-k)/h)``, giving a variance inflation
    of about ``1 + 2h/3`` when ``w = h`` rather than ``h``. The brief's ``n/h``
    is therefore a conservative bound of the right order, and both are
    reported. What must not happen is ``n_effective`` staying near ``n_raw``.
    """
    signal, forward = synthetic_panel(40000, 10, horizon=30, signal_window=30, seed=3)
    result = compute_ic(signal, forward, horizon_bars=30, residualise=False)

    assert result.n_effective < result.n_raw / 5, "overlap correction is not biting"
    ratio = result.n_effective / result.n_effective_naive
    assert 0.5 < ratio < 3.0, f"n_effective is {ratio:.2f}x the n/h bound"
    assert result.variance_inflation > 5


def test_effective_sample_size_falls_as_horizon_rises() -> None:
    previous = np.inf
    for horizon in (10, 30, 130):
        signal, forward = synthetic_panel(40000, 10, horizon=horizon,
                                          signal_window=horizon, seed=5)
        result = compute_ic(signal, forward, horizon_bars=horizon, residualise=False)
        assert result.n_effective < previous
        previous = result.n_effective


def test_iid_signal_shows_no_inflation_even_at_long_horizons() -> None:
    """The mechanism, stated as a test.

    Overlap alone does not inflate anything. ``Cov(x_t, x_{t-k})`` factorises
    into the signal's autocovariance times the return's, so an iid signal
    against 130-bar overlapping returns has a variance inflation of 1. This is
    why ``n/h`` is a bound rather than a description, and why the measured
    figure is the one reported.
    """
    signal, forward = synthetic_panel(40000, 5, horizon=130, signal_window=1, seed=17)
    result = compute_ic(signal, forward, horizon_bars=130, residualise=False)
    assert result.variance_inflation == pytest.approx(1.0, abs=0.4)
    assert result.n_effective > result.n_raw / 3


# --------------------------------------------------------- null calibration


def _rejection_rates(reps: int, horizon: int, signal_window: int,
                     n_symbols: int, n_bars: int, correlation: float = 0.0,
                     seed0: int = 900) -> tuple[float, float, float]:
    iid = hac = cluster = 0
    for i in range(reps):
        signal, forward = synthetic_panel(n_bars, n_symbols, horizon, signal_window,
                                          seed=seed0 + i, return_correlation=correlation)
        result = compute_ic(signal, forward, horizon, residualise=False)
        iid += abs(result.t_iid) > 1.96
        hac += abs(result.t_hac) > 1.96
        if np.isfinite(result.se_cluster_asset) and result.se_cluster_asset > 0:
            cluster += abs(result.ic / result.se_cluster_asset) > 1.96
    return iid / reps, hac / reps, cluster / reps


def test_iid_standard_error_is_catastrophically_wrong_under_the_null() -> None:
    """The headline claim of this phase, as an executable assertion.

    True IC is zero by construction. The IID standard error -- the one the
    previous brief specified -- rejects on well over half of samples at a
    nominal 5%. This is the single number that justifies the whole module.
    """
    iid_rate, hac_rate, _ = _rejection_rates(150, horizon=30, signal_window=30,
                                             n_symbols=10, n_bars=3000)
    assert iid_rate > 0.40, f"expected the IID SE to over-reject badly, got {iid_rate:.1%}"
    assert hac_rate < iid_rate / 3
    assert hac_rate < 0.16, f"Driscoll-Kraay rejection {hac_rate:.1%} is too far above nominal 5%"


def test_driscoll_kraay_holds_up_under_cross_sectional_correlation() -> None:
    """Correlated returns must not break the correction.

    Real intraday equity returns run pairwise correlations of 0.3 to 0.5, which
    is the reason breadth buys so much less than the raw asset count suggests.
    """
    iid_rate, hac_rate, _ = _rejection_rates(120, horizon=30, signal_window=30,
                                             n_symbols=30, n_bars=3000,
                                             correlation=0.5, seed0=1300)
    assert iid_rate > 0.40
    assert hac_rate < 0.18, f"rejection under rho=0.5 was {hac_rate:.1%}"


def test_estimator_does_not_lose_all_power() -> None:
    """A correction that never rejects anything would also be useless."""
    detections = 0
    for i in range(30):
        signal, forward = synthetic_panel(8000, 20, 30, 30, seed=2000 + i, planted_ic=0.06)
        detections += compute_ic(signal, forward, 30, residualise=False).p_value < 0.05
    assert detections / 30 > 0.7, "estimator has no power against a real 0.06 IC"


# ------------------------------------------------------- residualisation


def test_cross_sectional_demeaning_is_causal() -> None:
    """Perturbing the future must not change any past residual.

    The same property test the Phase 1 signals face, applied to the
    residualisation step -- which is where a fitted full-sample beta would
    reintroduce look-ahead one layer above where those tests can see it.
    """
    rng = np.random.default_rng(23)
    index = pd.date_range("2024-01-01", periods=500, freq="min", tz="UTC")
    returns = pd.DataFrame(rng.normal(size=(500, 8)), index=index,
                           columns=[f"S{i}" for i in range(8)])
    baseline = cross_sectional_demean(returns)

    for cut in (100, 250, 400):
        perturbed = returns.copy()
        perturbed.iloc[cut + 1:] += rng.normal(0, 5, size=(500 - cut - 1, 8))
        pd.testing.assert_frame_equal(
            baseline.iloc[: cut + 1],
            cross_sectional_demean(perturbed).iloc[: cut + 1],
        )


def test_full_sample_beta_residualisation_would_not_be_causal() -> None:
    """Why the default is demeaning: the fitted alternative fails the same test."""
    rng = np.random.default_rng(29)
    index = pd.date_range("2024-01-01", periods=500, freq="min", tz="UTC")
    returns = pd.DataFrame(rng.normal(size=(500, 8)), index=index,
                           columns=[f"S{i}" for i in range(8)])

    def full_sample_beta_residual(frame: pd.DataFrame) -> pd.DataFrame:
        market = frame.mean(axis=1)
        beta = frame.apply(lambda col: col.cov(market) / market.var())
        return frame - np.outer(market, beta)

    perturbed = returns.copy()
    perturbed.iloc[251:] += rng.normal(0, 5, size=(249, 8))
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(
            full_sample_beta_residual(returns).iloc[:251],
            full_sample_beta_residual(perturbed).iloc[:251],
        )


def test_rolling_beta_residual_is_causal() -> None:
    rng = np.random.default_rng(31)
    index = pd.date_range("2024-01-01", periods=800, freq="min", tz="UTC")
    returns = pd.DataFrame(rng.normal(size=(800, 6)), index=index,
                           columns=[f"S{i}" for i in range(6)])
    baseline = rolling_beta_residual(returns, window=100)
    perturbed = returns.copy()
    perturbed.iloc[501:] += rng.normal(0, 5, size=(299, 6))
    pd.testing.assert_frame_equal(
        baseline.iloc[:501], rolling_beta_residual(perturbed, window=100).iloc[:501]
    )


def test_residualisation_removes_a_planted_market_factor() -> None:
    rng = np.random.default_rng(37)
    index = pd.date_range("2024-01-01", periods=2000, freq="min", tz="UTC")
    market = rng.normal(0, 1, size=(2000, 1))
    returns = pd.DataFrame(0.9 * market + 0.4 * rng.normal(size=(2000, 10)),
                           index=index, columns=[f"S{i}" for i in range(10)])
    assert returns.corr().to_numpy()[np.triu_indices(10, 1)].mean() > 0.7
    residual = cross_sectional_demean(returns)
    assert abs(residual.corr().to_numpy()[np.triu_indices(10, 1)].mean()) < 0.2


# ------------------------------------------------------- degenerate cases


def test_constant_signal_reports_a_note_rather_than_crashing() -> None:
    """``always_long`` is a registered control and must not break the grid."""
    _, forward = synthetic_panel(3000, 5, 30, 30, seed=41)
    constant = pd.DataFrame(1.0, index=forward.index, columns=forward.columns)
    result = compute_ic(constant, forward, 30, signal_name="always_long", residualise=False)
    assert np.isnan(result.ic)
    assert "zero variance" in result.note


def test_empty_overlap_raises_rather_than_returning_nonsense() -> None:
    signal, forward = synthetic_panel(500, 3, 30, 30, seed=43)
    with pytest.raises(ValueError, match="no overlapping"):
        compute_ic(signal, forward.rename(columns=lambda c: c + "_other"), 30)


def test_report_refuses_a_ranking_without_corrections() -> None:
    """There is no code path to a 'best signal' without an FDR correction."""
    report = ICReport()
    for i in range(6):
        signal, forward = synthetic_panel(4000, 8, 30, 30, seed=50 + i)
        report.add(compute_ic(signal, forward, 30, signal_name=f"sig{i}",
                              hypothesis_id=f"H{i:03d}", residualise=False))
    ranking = report.ranking()
    assert "p_value_bh" in ranking.columns and "p_value_by" in ranking.columns
    assert ranking["p_value_bh"].notna().all()
    assert ranking["n_effective"].notna().all()
    # Under the null almost nothing should survive.
    assert len(report.survivors()) <= 1
