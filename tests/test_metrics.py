"""Metric correctness tests.

Two of these are calibration tests rather than worked examples: they generate
data under a known null and check that the metric's own error rate matches its
advertised error rate. That is a stronger check than reproducing a single
arithmetic case, because it exercises the metric's *sampling distribution*,
which is the thing that is actually claimed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from quantlab.metrics import (
    annualize_sharpe,
    deannualize_sharpe,
    deflated_sharpe,
    expected_max_sharpe,
    hit_rate,
    information_coefficient,
    max_drawdown,
    probabilistic_sharpe,
    sharpe,
    sharpe_variance_across_trials,
    sortino,
    turnover,
)

import pandas as pd


# ---------------------------------------------------------------- hand cases


def test_sharpe_matches_hand_computation():
    """Sharpe annualisation against arithmetic done in pure Python."""
    returns = [0.01, -0.005, 0.02, 0.0, -0.01]

    n = len(returns)
    mean = sum(returns) / n
    var = sum((x - mean) ** 2 for x in returns) / (n - 1)  # ddof=1
    std = math.sqrt(var)
    expected_per_period = mean / std
    expected_annual = expected_per_period * math.sqrt(252)

    assert mean == pytest.approx(0.003)
    assert var == pytest.approx(1.45e-4)
    assert expected_per_period == pytest.approx(0.24913644, rel=1e-8)
    assert expected_annual == pytest.approx(3.95491837, rel=1e-8)

    assert sharpe(returns, annualize=False) == pytest.approx(expected_per_period, rel=1e-12)
    assert sharpe(returns, periods_per_year=252) == pytest.approx(expected_annual, rel=1e-12)


def test_sharpe_annualisation_roundtrip():
    for ppy in (12, 52, 252, 6 * 252):
        assert deannualize_sharpe(annualize_sharpe(0.05, ppy), ppy) == pytest.approx(0.05)


def test_sharpe_scales_with_sqrt_periods_per_year():
    """Sharpe must scale as sqrt(ppy); a linear scaling is a classic bug."""
    rng = np.random.default_rng(0)
    r = rng.normal(0.0004, 0.01, 5000)
    assert sharpe(r, 252) / sharpe(r, 12) == pytest.approx(math.sqrt(252 / 12), rel=1e-12)


def test_max_drawdown_hand_case():
    # equity: 1.5, 0.75, 0.75 -> peak 1.5 -> trough 0.75 -> -50%
    assert max_drawdown([0.5, -0.5, 0.0]) == pytest.approx(-0.5)
    # monotonically rising equity has no drawdown
    assert max_drawdown([0.01, 0.02, 0.03]) == pytest.approx(0.0)


def test_max_drawdown_simple_vs_compound():
    r = [0.5, -0.5]
    assert max_drawdown(r, compound=True) == pytest.approx(-0.5)
    assert max_drawdown(r, compound=False) == pytest.approx(-0.5)


def test_hit_rate_excludes_flat_periods():
    assert hit_rate([0.01, -0.01, 0.0, 0.0, 0.01]) == pytest.approx(2 / 3)


def test_sortino_penalises_only_downside():
    """A series with the same mean but no losses has an undefined Sortino."""
    assert math.isnan(sortino([0.01, 0.02, 0.03]))
    # downside deviation divides by n, not by the count of losses
    r = [0.1, 0.1, 0.1, -0.1]
    dd = math.sqrt(((-0.1) ** 2) / 4)
    expected = (sum(r) / 4) / dd
    assert sortino(r, annualize=False) == pytest.approx(expected)


def test_turnover_hand_case():
    pos = pd.DataFrame({"a": [0.0, 1.0, 1.0, 0.0], "b": [0.0, 0.0, -1.0, -1.0]})
    # per-period |diff| sums: nan, 1, 1, 1 -> mean over 3 valid = 1.0
    assert turnover(pos, periods_per_year=252) == pytest.approx(252.0)


# --------------------------------------------------- expected max / deflation


def test_expected_max_sharpe_matches_monte_carlo():
    """The BLdP approximation against the empirical mean of the maximum."""
    rng = np.random.default_rng(7)
    variance = 1.0 / 1000.0
    for n_trials in (5, 20, 100, 1000):
        draws = rng.normal(0.0, math.sqrt(variance), size=(4000, n_trials))
        empirical = draws.max(axis=1).mean()
        approx = expected_max_sharpe(n_trials, variance)
        assert approx == pytest.approx(empirical, rel=0.05), (
            f"n_trials={n_trials}: approx {approx:.5f} vs empirical {empirical:.5f}"
        )


def test_expected_max_sharpe_grows_with_trials():
    v = 1e-3
    values = [expected_max_sharpe(n, v) for n in (2, 10, 100, 1000, 10000)]
    assert values == sorted(values)
    assert expected_max_sharpe(1, v) == 0.0


def test_regime_conditioning_raises_the_luck_threshold():
    """The cost of regime conditioning, measured against Monte Carlo truth.

    The project brief estimates this with the asymptotic ``sqrt(2 ln N)`` and
    concludes that going from S=5 strategies to S=5 across R=4 regimes raises
    the luck threshold by "about 37%" (``sqrt(ln 20 / ln 5) = 1.364``).

    That understates it. ``sqrt(2 ln N)`` is the *limiting* form and converges
    slowly: at N=5 it gives 1.794 against a true E[max] of 1.163, and at N=20
    it gives 2.448 against 1.866 -- overstating the absolute threshold by ~50%
    while understating the ratio between the two. The true multiplier is
    ``1.866 / 1.163 = 1.605``.

    So regime conditioning at this scale costs about **60%**, not 37%. The
    library reports the BLdP expectation, which lands within 1% of Monte Carlo.
    """
    rng = np.random.default_rng(101)
    draws = rng.normal(size=(200_000, 20))
    mc_plain = draws[:, :5].max(axis=1).mean()
    mc_conditioned = draws.max(axis=1).mean()
    true_ratio = mc_conditioned / mc_plain
    assert true_ratio == pytest.approx(1.605, abs=0.02)

    v = 1.0 / 1000.0
    approx_ratio = expected_max_sharpe(5 * 4, v) / expected_max_sharpe(5, v)
    assert approx_ratio == pytest.approx(true_ratio, rel=0.02)

    # And the brief's asymptotic is materially below the truth.
    brief_ratio = math.sqrt(math.log(20) / math.log(5))
    assert brief_ratio < true_ratio * 0.90


def test_deflated_sharpe_is_calibrated_under_the_null():
    """Under H0 the DSR of the best of N trials should be ~Uniform(0, 1).

    This is the test that matters. If the best of 50 worthless strategies
    routinely scores DSR > 0.95, the deflation is not doing its job and every
    parameter sweep in the library reports fiction.
    """
    rng = np.random.default_rng(11)
    reps, n_trials, n_periods = 400, 50, 1000

    exceedances = 0
    for _ in range(reps):
        panel = rng.normal(0.0, 0.01, size=(n_trials, n_periods))
        sharpes = panel.mean(axis=1) / panel.std(axis=1, ddof=1)
        best = int(np.argmax(sharpes))
        dsr = deflated_sharpe(
            panel[best],
            n_trials=n_trials,
            sharpe_variance=sharpe_variance_across_trials(sharpes),
        )
        exceedances += dsr > 0.95

    rate = exceedances / reps
    # Nominal 5%. Allow generous slack for the BLdP approximation and MC noise,
    # but a rate near 50% (what an undeflated PSR would give) must fail.
    assert rate < 0.15, f"DSR>0.95 fired on {rate:.1%} of null experiments (nominal 5%)"


def test_deflation_actually_bites():
    """An undeflated PSR passes where the deflated one fails, on the same data."""
    rng = np.random.default_rng(3)
    panel = rng.normal(0.0, 0.01, size=(200, 1000))
    sharpes = panel.mean(axis=1) / panel.std(axis=1, ddof=1)
    best = panel[int(np.argmax(sharpes))]

    assert probabilistic_sharpe(best) > 0.95  # looks significant on its own
    assert deflated_sharpe(best, n_trials=200,
                           sharpe_variance=sharpe_variance_across_trials(sharpes)) < 0.95


def test_probabilistic_sharpe_penalises_negative_skew():
    """Same Sharpe, uglier distribution, weaker evidence."""
    rng = np.random.default_rng(5)
    symmetric = rng.normal(0.0005, 0.01, 2000)

    skewed = -rng.gamma(shape=1.0, scale=1.0, size=2000)
    skewed = (skewed - skewed.mean()) / skewed.std(ddof=1)
    skewed = skewed * symmetric.std(ddof=1) + symmetric.mean()

    assert sharpe(symmetric, annualize=False) == pytest.approx(
        sharpe(skewed, annualize=False), rel=1e-9
    )
    assert probabilistic_sharpe(skewed) < probabilistic_sharpe(symmetric)


def test_probabilistic_sharpe_gaussian_closed_form():
    """For near-Gaussian returns PSR reduces to Phi(SR sqrt(n-1) / sqrt(1 + SR^2/2))."""
    from scipy import stats as st

    rng = np.random.default_rng(13)
    r = rng.normal(0.0003, 0.01, 20000)
    sr = sharpe(r, annualize=False)
    expected = st.norm.cdf(sr * math.sqrt(len(r) - 1) / math.sqrt(1 + sr**2 / 2))
    assert probabilistic_sharpe(r) == pytest.approx(expected, abs=0.02)


# ------------------------------------------------------ information coefficient


def test_ic_standard_error_matches_simulation_iid():
    """With iid data the naive and HAC standard errors both match reality."""
    rng = np.random.default_rng(17)
    n, reps = 500, 800
    ics, hac_ses = [], []
    for _ in range(reps):
        res = information_coefficient(rng.normal(size=n), rng.normal(size=n))
        ics.append(res.ic)
        hac_ses.append(res.se_hac)

    empirical_se = float(np.std(ics, ddof=1))
    assert empirical_se == pytest.approx(1 / math.sqrt(n - 2), rel=0.10)
    assert float(np.mean(hac_ses)) == pytest.approx(empirical_se, rel=0.10)


def test_ic_test_size_is_correct_for_iid_data():
    rng = np.random.default_rng(19)
    n, reps = 400, 1000
    rejections = sum(
        information_coefficient(rng.normal(size=n), rng.normal(size=n)).p_value < 0.05
        for _ in range(reps)
    )
    assert 0.03 < rejections / reps < 0.08


def test_naive_ic_standard_error_is_badly_wrong_under_overlap():
    """The headline claim: overlapping forward returns break the naive SE.

    Signal and forward return are built from *independent* noise, so the true
    IC is zero. The signal is a 20-bar rolling mean and the forward return is a
    20-bar forward sum, both of which induce strong autocorrelation in the
    per-period product. The naive standard error assumes independence and
    therefore rejects far too often; the HAC standard error should be close to
    nominal.
    """
    rng = np.random.default_rng(23)
    n, horizon, reps = 1500, 20, 400
    naive_rejections = 0
    hac_rejections = 0

    for _ in range(reps):
        a = pd.Series(rng.normal(size=n))
        b = pd.Series(rng.normal(size=n))
        signal = a.rolling(horizon).mean()
        forward = b[::-1].rolling(horizon).sum()[::-1]  # h-period forward sum

        res = information_coefficient(signal, forward, overlap=horizon)
        naive_t = res.ic / res.se_naive
        naive_rejections += abs(naive_t) > 1.96
        hac_rejections += res.p_value < 0.05

    naive_rate = naive_rejections / reps
    hac_rate = hac_rejections / reps

    assert naive_rate > 0.30, (
        f"expected the naive SE to over-reject badly, got {naive_rate:.1%}"
    )
    assert hac_rate < naive_rate / 2
    assert hac_rate < 0.25, f"HAC rejection rate {hac_rate:.1%} still far above nominal 5%"


def test_ic_recovers_a_planted_correlation():
    rng = np.random.default_rng(29)
    n = 5000
    signal = rng.normal(size=n)
    noise = rng.normal(size=n)
    forward = 0.1 * signal + noise  # true corr ~ 0.1/sqrt(1.01) ~ 0.0995

    res = information_coefficient(signal, forward)
    assert res.ic == pytest.approx(0.0995, abs=0.03)
    assert res.t_stat > 4


def test_ic_spearman_is_robust_to_monotone_transform():
    rng = np.random.default_rng(31)
    signal = rng.normal(size=2000)
    forward = 0.2 * signal + rng.normal(size=2000)

    plain = information_coefficient(signal, forward, method="spearman")
    warped = information_coefficient(np.exp(signal), forward, method="spearman")
    assert plain.ic == pytest.approx(warped.ic, rel=1e-9)
