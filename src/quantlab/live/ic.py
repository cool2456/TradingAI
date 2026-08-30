"""Information coefficient measurement. **This module is the deliverable.**

The question Phase 2 exists to answer is whether any signal in
:mod:`quantlab.signals` has an information coefficient reliably above zero on
real data at a horizon where costs do not eat it. Everything else in
``quantlab.live`` is plumbing that only matters if the answer here is yes.

Three ways to get this wrong, and what is done instead
-----------------------------------------------------

**1. Overlap.** A forward return computed at every bar over ``h`` bars shares
``h - k`` of its ``h`` periods with the observation ``k`` bars earlier, so the
product series has autocorrelation ``(h - k)/h`` -- 0.992 at lag 1 when
``h = 130``. The IID standard error assumes those observations are independent.
They are not, and the resulting t-statistic is inflated by roughly ``sqrt(h)``:
a factor of 11 at ``h = 130``. Newey-West with a lag length of at least ``h``
is used, and the IID value is reported alongside so the size of the correction
is visible rather than merely asserted.

**2. Cross-sectional dependence.** Pooling 50 names and computing one
correlation treats 50 observations at the same timestamp as 50 independent
draws. Intraday equity returns run pairwise correlations of 0.3 to 0.5, so the
true independent count at each timestamp is closer to two or three.

The brief calls for "asset-clustered standard errors". That is the wrong
cluster. Clustering **by asset** permits arbitrary dependence within an asset
over time while assuming independence *across* assets -- which is precisely the
assumption that fails here. The fix is to cluster by **time period**, which
combined with a Newey-West correction along the time axis is the Driscoll-Kraay
estimator. That is what :func:`compute_ic` uses. A cluster-by-asset standard
error is reported beside it so the difference is visible; on real data it is
the optimistic one.

**3. Look-ahead in the residualisation.** Residualising returns against a market
factor with a beta fitted on the whole sample re-introduces exactly the
full-sample normalisation that ``tests/test_causality.py`` was written to
catch, one layer up where those tests cannot see it. Cross-sectional demeaning
-- subtracting the equal-weight mean return across the universe at each
timestamp -- needs no estimated parameter, uses no future data, and is
equivalent to assuming a beta of 1 for every name. It is the default.
:func:`rolling_beta_residual` offers a causal, trailing-window alternative.

When overlap actually bites, and when it does not
-------------------------------------------------
A detail the brief's framing omits, which changes how the numbers should be
read. Overlapping forward returns are autocorrelated, but the quantity whose
autocorrelation inflates the standard error is the *product*
``x = signal x forward_return``, and for independent mean-zero series
``Cov(x_t, x_{t-k}) = Cov(s_t, s_{t-k}) * Cov(r_t, r_{t-k})``.

So an **iid signal against 130-bar overlapping returns has no variance
inflation at all** -- measured at 1.0 -- because the product's autocovariance
is the product of the two, and one factor is zero. Overlap only bites when the
signal is persistent as well. Measured here on synthetic nulls:

===================  ===============  ===============
signal persistence   horizon h=30     horizon h=130
===================  ===============  ===============
w = 1 (iid)          VIF 0.9          VIF 1.0
w = 30               VIF 15.0         VIF 27.0
w = 130              VIF 19.2         VIF 66.5
===================  ===============  ===============

Every signal in :mod:`quantlab.signals` is a trailing rolling statistic and is
therefore strongly persistent, so the inflation is real here -- but ``n / h``
is a worst-case bound rather than the actual figure, which is why the measured
``n_effective`` is the headline.

Residual over-rejection, disclosed
----------------------------------
Newey-West is downward-biased in finite samples. On synthetic nulls with
``L = 2h`` this estimator rejects on roughly 5-9% of samples against a nominal
5%, versus 63-71% for the IID standard error. The correction removes almost
all of the error and not quite all of it, so a p-value near the threshold
should be read as "not established" rather than "just significant".

What ``n_effective`` means here
-------------------------------
The brief defines it as ``n / horizon_bars``. That is correct for a single
asset and wrong for a pooled panel, where it ignores the cross-sectional
dependence above and can overstate the independent count by up to a factor of
``N``. Both are reported. The headline figure is measured rather than assumed::

    n_effective = (sd(x) / SE_driscoll_kraay) ** 2

which is the number of genuinely independent observations that would produce
the standard error actually obtained. It accounts for overlap and
cross-sectional correlation together, and it is estimated from the data instead
of from an assumption about it.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from ..metrics import newey_west_mean_se

__all__ = [
    "ICResult",
    "ICReport",
    "compute_ic",
    "cross_sectional_demean",
    "rolling_beta_residual",
    "benjamini_hochberg",
    "benjamini_yekutieli",
    "load_hypotheses",
    "predicted_direction",
    "mark_tested",
    "require_registered",
    "HYPOTHESES_PATH",
]

HYPOTHESES_PATH = Path("hypotheses.yaml")


# --------------------------------------------------------------------------
# Residualisation
# --------------------------------------------------------------------------


def cross_sectional_demean(returns: pd.DataFrame) -> pd.DataFrame:
    """Subtract the equal-weight cross-sectional mean return at each timestamp.

    ``resid[i, t] = r[i, t] - mean_j r[j, t]``

    This is residualisation against an equal-weight market factor with the beta
    of every name fixed at 1. No parameter is estimated, so no future data can
    enter -- which is the entire point. A fitted beta, even a "sensible"
    full-sample one, makes today's residual depend on next year's covariance.

    The cost of the assumption is that names whose true beta is far from 1 are
    imperfectly hedged and retain some market exposure. That is a bias toward
    finding *market* predictability rather than *residual* predictability, so it
    is conservative in the direction that matters: it can only make a signal
    look better than it is by leaving beta in, and the raw-versus-residual
    comparison in the report is what exposes that.
    """
    return returns.sub(returns.mean(axis=1), axis=0)


def rolling_beta_residual(
    returns: pd.DataFrame, window: int = 390, min_periods: int | None = None
) -> pd.DataFrame:
    """Residualise against a trailing-window beta on the equal-weight factor.

    ``beta_i(t)`` is estimated on the window **ending at ``t - 1``**, so the
    residual at ``t`` uses no contemporaneous or future information. Offered as
    an alternative to :func:`cross_sectional_demean` when betas genuinely differ
    across the universe; it costs ``window`` bars of warm-up and adds an
    estimated parameter per name, which is a real statistical cost and not a
    free improvement.
    """
    min_periods = min_periods or window // 2
    market = returns.mean(axis=1)
    cov = returns.rolling(window, min_periods=min_periods).cov(market).shift(1)
    var = market.rolling(window, min_periods=min_periods).var().shift(1)
    beta = cov.div(var.where(var > 0), axis=0)
    return returns - beta.mul(market, axis=0)


# --------------------------------------------------------------------------
# The estimator
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ICResult:
    """One information-coefficient measurement, with every caveat attached.

    ``ic`` alone is meaningless. It is reported only together with
    ``n_effective``, ``se_driscoll_kraay`` and the false-discovery-corrected
    p-value, and :class:`ICReport` refuses to rank signals without them.
    """

    hypothesis_id: str
    signal: str
    horizon_bars: int
    residualised: bool

    ic: float
    se_iid: float
    se_newey_west: float
    se_driscoll_kraay: float
    se_cluster_asset: float

    t_iid: float
    t_hac: float
    p_value: float

    n_raw: int
    n_symbols: int
    n_periods: int
    n_effective: float
    n_effective_naive: float
    variance_inflation: float
    mean_pairwise_corr: float
    hac_lags: int
    within_session: bool
    note: str = ""

    p_value_bh: float | None = None
    p_value_by: float | None = None
    #: The same IC measured with entry_lag=0, where the signal's price is also
    #: the return's entry price. Diagnostic only -- it is contaminated by the
    #: bid-ask bounce and is never counted as a trial or reported as a finding.
    ic_entry_lag0_DIAGNOSTIC: float | None = None
    #: The same IC measured WITHOUT residualising, against raw returns. Reported
    #: beside the residual figure because the gap between them is how much of
    #: any apparent signal is market beta rather than stock-specific
    #: information. A signal whose raw IC greatly exceeds its residual IC is
    #: mostly a market-timing bet wearing a cross-sectional costume.
    ic_raw: float | None = None
    t_hac_raw: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _iid_correlation_se(r: float, n: int) -> float:
    """Exact IID standard error of a Pearson correlation: ``sqrt((1-r^2)/(n-2))``.

    Note this is the *general* form. Under the null ``r = 0`` it reduces to
    ``1/sqrt(n-2)``. Both are still wrong on overlapping data -- they are
    reported only to show how much the dependence correction moves the answer.
    """
    if n < 3:
        return float("nan")
    return float(math.sqrt(max(1.0 - r**2, 0.0) / (n - 2)))


def _mean_pairwise_correlation(panel: pd.DataFrame) -> float:
    """Average off-diagonal pairwise correlation. Drives the breadth calculation."""
    if panel.shape[1] < 2:
        return 0.0
    corr = panel.corr().to_numpy()
    off_diagonal = corr[~np.eye(corr.shape[0], dtype=bool)]
    finite = off_diagonal[np.isfinite(off_diagonal)]
    return float(finite.mean()) if finite.size else 0.0


def _newey_west_panel_se(panel: np.ndarray, lags: int) -> float:
    """Newey-West SE of the pooled mean, correcting time but not the cross-section.

    The middle rung of the three standard errors: it allows arbitrary serial
    dependence within each symbol and assumes independence **across** symbols.
    That is what "cluster by asset" buys, and reporting it beside the
    Driscoll-Kraay figure shows what the independence assumption is worth.

    Autocovariances for all lags are obtained by FFT rather than by looping.
    At ``h = 390`` the truncation is 780 lags over ~98,000 timestamps and 100
    symbols; the loop is 7.6e9 multiply-adds, the FFT is ``O(T log T)`` per
    symbol and finishes in well under a second.
    """
    mask = np.isfinite(panel)
    counts = mask.sum(axis=0).astype(float)
    keep = counts > max(lags + 2, 10)
    if not keep.any():
        return float("nan")
    panel, mask, counts = panel[:, keep], mask[:, keep], counts[keep]

    means = np.where(mask, panel, 0.0).sum(axis=0) / counts
    centered = np.where(mask, panel - means, 0.0)

    n_rows = centered.shape[0]
    n_fft = 1 << int(np.ceil(np.log2(max(2 * n_rows, 2))))
    spectrum = np.fft.rfft(centered, n_fft, axis=0)
    autocov = np.fft.irfft(spectrum * np.conj(spectrum), n_fft, axis=0)[: lags + 1]
    autocov = autocov / counts

    weights = 1.0 - np.arange(1, lags + 1) / (lags + 1.0)
    long_run = autocov[0] + 2.0 * (weights[:, None] * autocov[1:]).sum(axis=0)
    long_run = np.maximum(long_run, autocov[0])       # Bartlett can go negative

    variance_of_symbol_mean = long_run / counts
    share = counts / counts.sum()
    return float(np.sqrt((share**2 * variance_of_symbol_mean).sum()))


def compute_ic(
    signal: pd.DataFrame | pd.Series,
    forward_return: pd.DataFrame | pd.Series,
    horizon_bars: int,
    method: Literal["newey_west"] = "newey_west",
    *,
    hypothesis_id: str = "",
    signal_name: str = "",
    residualise: bool = True,
    hac_lags: int | None = None,
    lag_multiple: float = 2.0,
    within_session: bool = True,
    min_symbols_per_period: int = 2,
) -> ICResult:
    """Pooled information coefficient with Driscoll-Kraay standard errors.

    Parameters
    ----------
    signal, forward_return
        ``timestamp x symbol`` panels, aligned. A Series is treated as a
        one-column panel.
    horizon_bars
        The forward-return horizon. Sets the minimum Newey-West lag length and
        the naive effective-sample divisor.
    residualise
        Cross-sectionally demean the forward returns before measuring. See
        :func:`cross_sectional_demean`.
    lag_multiple
        Newey-West truncation as a multiple of ``horizon_bars``. The brief
        requires "at least h"; 2h is the default because it calibrates better.
        Under a null with signal persistence equal to the horizon, ``L = h``
        rejects on 6.0-10.5% of samples against a nominal 5%, while ``L = 2h``
        rejects on 5.0-9.5%. The Bartlett weight at lag ``k`` is
        ``1 - k/(L+1)``, so a truncation at exactly ``h`` gives the
        autocovariance at lag ``h-1`` a weight of ``1/(h+1)`` -- it is counted
        at roughly 3% of its true size. Extending the window restores it.

    Method
    ------
    The signal and the (optionally residualised) forward return are
    standardised over the pooled panel, giving per-observation products
    ``x[i, t] = z_s[i, t] * z_r[i, t]`` whose pooled mean is the correlation.

    Those products are then averaged across symbols at each timestamp, which
    collapses the cross-section and makes cross-sectional dependence
    irrelevant to the standard error, and a Newey-West correction with
    ``lags >= horizon_bars`` is applied along the time axis. That combination is
    the Driscoll-Kraay estimator, and it is the headline standard error.
    """
    if method != "newey_west":
        raise ValueError(f"unsupported method {method!r}; only 'newey_west' is implemented")
    if horizon_bars < 1:
        raise ValueError("horizon_bars must be >= 1")

    sig = signal.to_frame() if isinstance(signal, pd.Series) else signal.copy()
    fwd = forward_return.to_frame() if isinstance(forward_return, pd.Series) else forward_return.copy()

    common_cols = sig.columns.intersection(fwd.columns)
    common_idx = sig.index.intersection(fwd.index)
    sig, fwd = sig.loc[common_idx, common_cols], fwd.loc[common_idx, common_cols]
    if sig.empty:
        raise ValueError("signal and forward_return share no overlapping (timestamp, symbol) cells")

    if residualise:
        fwd = cross_sectional_demean(fwd)

    valid = sig.notna() & fwd.notna()
    sig, fwd = sig.where(valid), fwd.where(valid)

    sig_values = sig.to_numpy(dtype=float)
    fwd_values = fwd.to_numpy(dtype=float)
    finite = np.isfinite(sig_values) & np.isfinite(fwd_values)
    n_raw = int(finite.sum())

    degenerate = ""
    if n_raw < 10:
        degenerate = f"only {n_raw} usable observations"
    elif np.nanstd(sig_values[finite]) == 0:
        degenerate = "signal has zero variance; a constant carries no information by construction"
    elif np.nanstd(fwd_values[finite]) == 0:
        degenerate = "forward returns have zero variance"

    if degenerate:
        nan = float("nan")
        return ICResult(
            hypothesis_id, signal_name or "?", horizon_bars, residualise,
            nan, nan, nan, nan, nan, nan, nan, nan,
            n_raw, int(sig.shape[1]), int(sig.shape[0]), nan, nan, nan, nan,
            hac_lags or horizon_bars, within_session, note=degenerate,
        )

    # Pooled standardisation: the mean of the products is then the pooled
    # Pearson correlation between signal and forward return.
    s_mean, s_std = np.nanmean(sig_values[finite]), np.nanstd(sig_values[finite])
    r_mean, r_std = np.nanmean(fwd_values[finite]), np.nanstd(fwd_values[finite])
    products = np.full(sig_values.shape, np.nan)
    products[finite] = ((sig_values[finite] - s_mean) / s_std) * (
        (fwd_values[finite] - r_mean) / r_std
    )

    ic = float(np.nanmean(products))
    product_sd = float(np.nanstd(products[finite], ddof=1))

    # --- Driscoll-Kraay: collapse the cross-section, then Newey-West in time.
    per_period = pd.DataFrame(products, index=sig.index, columns=sig.columns)
    counts = per_period.notna().sum(axis=1)
    period_mean = per_period.mean(axis=1).where(counts >= min(min_symbols_per_period, sig.shape[1]))
    period_mean = period_mean.dropna()
    n_periods = int(len(period_mean))

    lags = int(hac_lags if hac_lags is not None else round(lag_multiple * horizon_bars))
    lags = max(1, min(lags, max(n_periods - 2, 1)))

    if n_periods < 5:
        se_dk = float("nan")
    else:
        # SE of the mean of the collapsed series. Because the pooled IC equals
        # the (roughly balanced) mean of the per-period means, this is the SE of
        # the IC itself.
        se_dk = newey_west_mean_se(period_mean.to_numpy(dtype=float), lags)

    # --- Newey-West within each symbol, independence assumed across symbols.
    # Applying a lag to the *flattened* panel would be meaningless: consecutive
    # elements there are different symbols at the same timestamp, not the same
    # symbol one bar apart.
    se_nw = _newey_west_panel_se(products, lags)

    # --- Cluster by asset: allows serial dependence within a name, assumes
    # independence across names. Reported to show what that assumption buys.
    asset_means = per_period.mean(axis=0).dropna()
    n_symbols_used = int(len(asset_means))
    se_cluster_asset = (
        float(asset_means.std(ddof=1) / math.sqrt(n_symbols_used))
        if n_symbols_used > 1
        else float("nan")
    )

    se_iid = _iid_correlation_se(ic, n_raw)
    t_iid = ic / se_iid if se_iid and np.isfinite(se_iid) and se_iid > 0 else float("nan")
    t_hac = ic / se_dk if se_dk and np.isfinite(se_dk) and se_dk > 0 else float("nan")
    p_value = float(2.0 * stats.norm.sf(abs(t_hac))) if np.isfinite(t_hac) else float("nan")

    # Measured effective sample size: the iid count that would give this SE.
    n_effective = (
        float((product_sd / se_dk) ** 2)
        if se_dk and np.isfinite(se_dk) and se_dk > 0
        else float("nan")
    )
    n_effective_naive = float(n_raw / horizon_bars)
    variance_inflation = (
        float((se_dk / (product_sd / math.sqrt(n_raw))) ** 2)
        if se_dk and np.isfinite(se_dk) and se_dk > 0
        else float("nan")
    )

    return ICResult(
        hypothesis_id=hypothesis_id,
        signal=signal_name or "?",
        horizon_bars=horizon_bars,
        residualised=residualise,
        ic=ic,
        se_iid=se_iid,
        se_newey_west=se_nw,
        se_driscoll_kraay=se_dk,
        se_cluster_asset=se_cluster_asset,
        t_iid=float(t_iid),
        t_hac=float(t_hac),
        p_value=p_value,
        n_raw=n_raw,
        n_symbols=int(sig.shape[1]),
        n_periods=n_periods,
        n_effective=n_effective,
        n_effective_naive=n_effective_naive,
        variance_inflation=variance_inflation,
        mean_pairwise_corr=_mean_pairwise_correlation(fwd),
        hac_lags=lags,
        within_session=within_session,
    )


# --------------------------------------------------------------------------
# Multiplicity
# --------------------------------------------------------------------------


def benjamini_hochberg(p_values: Sequence[float], alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, controlling the false discovery rate.

    Valid under independence and under positive regression dependence. It is
    **not** guaranteed under arbitrary dependence, which matters here:
    ``zscore_momentum`` is the exact negative of ``zscore_reversion`` at
    matching parameters, so those tests are negatively dependent. Use
    :func:`benjamini_yekutieli` when that matters.
    """
    p = np.asarray(p_values, dtype=float)
    finite = np.isfinite(p)
    out = np.full(p.shape, np.nan)
    if not finite.any():
        return out
    values = p[finite]
    m = values.size
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.minimum.accumulate((ranked * m / np.arange(m, 0, -1))[::-1])[::-1]
    result = np.empty(m)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    out[finite] = result
    return out


def benjamini_yekutieli(p_values: Sequence[float], alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Yekutieli adjusted p-values. Valid under **arbitrary** dependence.

    Identical to Benjamini-Hochberg but scaled by the harmonic number
    ``c(m) = sum_{i=1..m} 1/i``, which is the price of not having to assume
    anything about how the tests depend on one another. At ``m = 21`` that
    factor is 3.65, so it is a substantial price -- and the correct one to pay
    when the grid contains signals that are exact negatives of each other.
    """
    p = np.asarray(p_values, dtype=float)
    finite = np.isfinite(p)
    m = int(finite.sum())
    if m == 0:
        return np.full(p.shape, np.nan)
    harmonic = float(np.sum(1.0 / np.arange(1, m + 1)))
    out = benjamini_hochberg(p)
    return np.clip(out * harmonic, 0.0, 1.0)


@dataclass
class ICReport:
    """The full grid of measurements, with multiplicity corrections attached."""

    results: list[ICResult] = field(default_factory=list)
    alpha: float = 0.05
    _corrected: bool = False

    def add(self, result: ICResult) -> None:
        self.results.append(result)
        self._corrected = False

    def apply_corrections(self) -> "ICReport":
        """Attach BH and BY adjusted p-values across the whole grid."""
        p_values = [r.p_value for r in self.results]
        bh = benjamini_hochberg(p_values, self.alpha)
        by = benjamini_yekutieli(p_values, self.alpha)
        self.results = [
            ICResult(**{**r.to_dict(), "p_value_bh": None if not np.isfinite(b) else float(b),
                        "p_value_by": None if not np.isfinite(y) else float(y)})
            for r, b, y in zip(self.results, bh, by)
        ]
        self._corrected = True
        return self

    def to_frame(self) -> pd.DataFrame:
        if not self._corrected:
            self.apply_corrections()
        return pd.DataFrame([r.to_dict() for r in self.results])

    def ranking(self) -> pd.DataFrame:
        """Signals ordered by evidence, with corrections always attached.

        There is no method that returns a ranking without them. A "best signal"
        chosen from a grid of 21 tests and reported with an uncorrected p-value
        is the multiple-comparisons error in its purest form, and the way to
        prevent it is to make the uncorrected version unavailable rather than
        discouraged.
        """
        frame = self.to_frame()
        if frame.empty:
            return frame
        columns = [
            "hypothesis_id", "signal", "horizon_bars", "ic_raw", "ic", "t_hac_raw",
            "t_iid", "t_hac",
            "n_raw", "n_effective", "n_effective_naive", "variance_inflation",
            "p_value", "p_value_bh", "p_value_by", "ic_entry_lag0_DIAGNOSTIC", "note",
        ]
        return frame[columns].sort_values("p_value_by", na_position="last").reset_index(drop=True)

    def survivors(self, alpha: float | None = None, method: str = "by") -> pd.DataFrame:
        """Rows clearing the FDR-corrected bar. Empty is a valid and common answer."""
        alpha = alpha if alpha is not None else self.alpha
        column = {"by": "p_value_by", "bh": "p_value_bh"}[method]
        frame = self.to_frame()
        if frame.empty:
            return frame
        return frame[frame[column].notna() & (frame[column] < alpha)]

    def summary_line(self, method: str = "by") -> str:
        """One sentence, suitable for the top of the README."""
        frame = self.to_frame()
        survivors = self.survivors(method=method)
        n_tests = len(frame)
        if survivors.empty:
            return (
                f"No signal cleared the FDR-corrected bar. {n_tests} tests "
                f"({frame['signal'].nunique()} signals x "
                f"{frame['horizon_bars'].nunique()} horizons), "
                f"alpha={self.alpha}, Benjamini-Yekutieli corrected. "
                "The answer to this phase's question is no."
            )
        best = survivors.sort_values("p_value_by").iloc[0]
        return (
            f"{len(survivors)} of {n_tests} tests cleared the FDR-corrected bar. "
            f"Strongest: {best['signal']} at h={int(best['horizon_bars'])}, "
            f"IC={best['ic']:.4f}, HAC t={best['t_hac']:.2f}, "
            f"n_effective={best['n_effective']:.0f}, "
            f"BY-corrected p={best['p_value_by']:.4f}."
        )


# --------------------------------------------------------------------------
# Pre-registration enforcement
# --------------------------------------------------------------------------


def load_hypotheses(path: Path | str = HYPOTHESES_PATH) -> dict[str, Any]:
    """Read and validate ``hypotheses.yaml``.

    Rejects any hypothesis whose ``mechanism`` fails to name a loser. A
    mechanism of "the indicator works" is a restatement of the prediction, not
    an explanation of it, and is refused at load time rather than tested.
    """
    import yaml

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. No hypothesis may be tested before it is "
            "pre-registered and committed."
        )
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    hypotheses = document.get("hypotheses", [])
    if not hypotheses:
        raise ValueError(f"{path} contains no hypotheses")

    for entry in hypotheses:
        for required in ("id", "signal", "horizons_bars", "prediction", "mechanism", "status"):
            if required not in entry:
                raise ValueError(f"hypothesis {entry.get('id', '?')} is missing {required!r}")
        mechanism = str(entry["mechanism"]).strip()
        if len(mechanism) < 80:
            raise ValueError(
                f"hypothesis {entry['id']} has a mechanism of {len(mechanism)} characters. "
                "It must name who loses money and why they keep doing it; a mechanism "
                "that short cannot."
            )
        upper = mechanism.upper()
        if "WHO LOSES" not in upper and "CONTROL" not in upper:
            raise ValueError(
                f"hypothesis {entry['id']} does not identify who loses money. "
                "'The indicator works' is a prediction, not a mechanism."
            )
    return document


def require_registered(
    signal: str,
    horizon_bars: int,
    params: Mapping[str, Any] | None = None,
    document: dict[str, Any] | None = None,
    path: Path | str = HYPOTHESES_PATH,
) -> dict[str, Any]:
    """Return the registered hypothesis for this test, or raise.

    The gate that makes pre-registration real: a measurement that was not
    registered cannot be run, so the grid cannot quietly grow after the data
    has been seen.
    """
    document = document or load_hypotheses(path)
    for entry in document["hypotheses"]:
        if entry["signal"] != signal:
            continue
        if horizon_bars not in entry["horizons_bars"]:
            continue
        if params is not None and dict(entry.get("params", {})) != dict(params):
            raise ValueError(
                f"{signal} is registered as {entry['id']} with params "
                f"{entry.get('params')}, but this test uses {dict(params)}. "
                "Parameter variations are separate hypotheses and separate trials; "
                "register them before running them."
            )
        return entry
    raise ValueError(
        f"no registered hypothesis for signal={signal!r} at horizon={horizon_bars}. "
        f"Add it to {path}, commit it, and only then run the test."
    )


def budget_remaining(document: dict[str, Any] | None = None,
                     path: Path | str = HYPOTHESES_PATH) -> dict[str, int]:
    """Trial budget accounting for this phase."""
    document = document or load_hypotheses(path)
    registered = sum(len(h["horizons_bars"]) for h in document["hypotheses"])
    budget = int(document.get("meta", {}).get("trial_budget", 0))
    return {"budget": budget, "registered": registered, "remaining": budget - registered}


# --------------------------------------------------------------------------
# The experiment
# --------------------------------------------------------------------------


def run_ic_grid(
    prices: pd.DataFrame,
    document: dict[str, Any] | None = None,
    path: Path | str = HYPOTHESES_PATH,
    residualise: bool = True,
    log: bool = True,
    session_length: int | None = 390,
    entry_lag: int = 1,
    diagnose_bounce: bool = True,
    verbose: bool = True,
) -> ICReport:
    """Run every pre-registered hypothesis against a price panel. **Step 5.**

    For each registered (signal, horizon) pair this computes the signal from
    prices, the forward return over the horizon, and the information
    coefficient with Driscoll-Kraay standard errors, then applies
    false-discovery-rate corrections across the whole grid at once.

    Horizons at or beyond the session length have no within-session forward
    window, so they are measured as overnight-inclusive returns and flagged as
    such in the ``note`` column. That is a different quantity -- dominated by
    gap risk and overnight news rather than by the microstructure effects the
    hypotheses name -- and it is labelled rather than quietly mixed in.

    Every measurement appends one trial to ``research_log.jsonl``, so the grid
    is charged for at the same rate as any other parameter sweep.
    """
    from ..signals import SIGNALS
    from ..validate import log_trial
    from .featurestore import forward_returns

    document = document or load_hypotheses(path)
    report = ICReport()

    for entry in document["hypotheses"]:
        name = entry["signal"]
        params = dict(entry.get("params", {}))
        if name not in SIGNALS:
            raise ValueError(f"{entry['id']} names signal {name!r}, which is not in quantlab.signals")
        bound = SIGNALS[name].bind(**params)
        signal_panel = bound(prices)

        for horizon in entry["horizons_bars"]:
            require_registered(name, horizon, params, document=document)

            within_session = (
                session_length is None or horizon + entry_lag <= session_length - 1
            )
            forward = forward_returns(prices, horizon, within_session=within_session,
                                      entry_lag=entry_lag)
            result = compute_ic(
                signal_panel, forward, horizon,
                hypothesis_id=entry["id"], signal_name=name,
                residualise=residualise, within_session=within_session,
            )
            if not within_session:
                result = ICResult(**{
                    **result.to_dict(),
                    "note": (result.note + " | " if result.note else "")
                    + "OVERNIGHT-INCLUSIVE: horizon >= session length, so this "
                      "measures a gap the intraday mechanism does not describe",
                })
            # Diagnostic only, never a trial and never eligible to be a finding:
            # the same IC measured with entry_lag=0, where the signal's own
            # print is also the return's entry price. The gap between the two
            # is the bid-ask bounce artifact.
            if diagnose_bounce and entry_lag > 0:
                bounce_forward = forward_returns(prices, horizon,
                                                 within_session=within_session, entry_lag=0)
                bounce = compute_ic(signal_panel, bounce_forward, horizon,
                                    residualise=residualise)
                result = ICResult(**{**result.to_dict(),
                                     "ic_entry_lag0_DIAGNOSTIC": bounce.ic})

            # Raw (unresidualised) IC, reported beside the residual figure. The
            # gap is the share of any apparent signal that is market beta.
            if residualise:
                raw = compute_ic(signal_panel, forward, horizon, residualise=False)
                result = ICResult(**{**result.to_dict(),
                                     "ic_raw": raw.ic, "t_hac_raw": raw.t_hac})
            report.add(result)

            if verbose:
                print(
                    f"  {entry['id']} {name:22s} h={horizon:4d} "
                    f"IC={result.ic:+.4f} t_iid={result.t_iid:+7.2f} "
                    f"t_hac={result.t_hac:+6.2f} n_eff={result.n_effective:10.0f}"
                    + (f"  [{result.note[:40]}]" if result.note else "")
                )
            if log:
                log_trial(
                    {
                        "phase": 2, "hypothesis_id": entry["id"], "signal": name,
                        "params": params, "horizon_bars": horizon,
                        "residualised": residualise, "within_session": within_session,
                        "n_symbols": result.n_symbols,
                    },
                    {
                        "ic": result.ic, "t_hac": result.t_hac, "p_value": result.p_value,
                        "n_effective": result.n_effective, "n_raw": result.n_raw,
                        # Phase 1's DSR machinery reads sharpe_per_period; an IC
                        # measurement has no Sharpe, so it is left null rather
                        # than fabricated from a return series that does not exist.
                        "sharpe_per_period": None,
                    },
                    kind="phase2_ic",
                )
    return report.apply_corrections()


def predicted_direction(prediction: str) -> Literal["positive", "non_positive", "null"]:
    """Extract the registered directional claim from a prediction string.

    Deliberately crude and auditable rather than clever: the prediction strings
    were written by hand at registration time and use three unambiguous forms.
    Getting this wrong in the permissive direction is how a sign reversal gets
    reported as a confirmation.
    """
    text = prediction.lower()
    if "indistinguishable from zero" in text or "= 0 within" in text or "= 0 by construction" in text:
        return "null"
    if "<= 0" in text or "ic <= 0" in text:
        return "non_positive"
    if "ic > " in text or "> 0" in text:
        return "positive"
    return "null"


def mark_tested(
    report: ICReport, path: Path | str = HYPOTHESES_PATH
) -> dict[str, str]:
    """Flip each tested hypothesis's ``status`` to a factual verdict.

    Verdicts are assigned mechanically from the corrected p-values **and the
    registered direction**, so the author cannot grade their own homework after
    seeing the numbers.

    The direction check is not optional. A two-sided test rejects when the
    effect is significant in *either* direction, so a hypothesis predicting
    ``IC > 0.02`` that measures a significant ``-0.008`` would otherwise be
    recorded as "supported" -- when in fact its prediction is falsified and the
    opposite effect is present. That is the single easiest way to turn a
    refutation into a confirmation, and it is checked explicitly.
    """
    import yaml

    path = Path(path)
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    frame = report.to_frame()
    verdicts: dict[str, str] = {}

    for entry in document["hypotheses"]:
        rows = frame[frame["hypothesis_id"] == entry["id"]]
        if rows.empty:
            continue
        direction = predicted_direction(str(entry.get("prediction", "")))
        significant = rows[rows["p_value_by"].notna() & (rows["p_value_by"] < report.alpha)]

        if significant.empty:
            if direction == "null":
                verdict = (
                    "PREDICTION HELD: no horizon showed a significant IC, which is "
                    "what was registered"
                )
            else:
                verdict = "FALSIFIED: no horizon cleared the FDR-corrected bar"
        else:
            best = significant.reindex(significant["p_value_by"].sort_values().index).iloc[0]
            ic, horizon = float(best["ic"]), int(best["horizon_bars"])
            detail = (
                f"IC={ic:+.4f}, HAC t={best['t_hac']:.2f}, BY p={best['p_value_by']:.4f}, "
                f"n_effective={best['n_effective']:.0f} at h={horizon}"
            )
            sign_ok = (
                (direction == "positive" and ic > 0)
                or (direction == "non_positive" and ic <= 0)
            )
            if direction == "null":
                verdict = f"FALSIFIED: registered as null, but a significant IC was found -- {detail}"
            elif sign_ok:
                verdict = f"SUPPORTED in sign -- {detail}"
            else:
                verdict = (
                    f"FALSIFIED WITH SIGN REVERSAL: the prediction was "
                    f"{entry['prediction'].strip()[:60]!r}, and the measured effect is "
                    f"significant in the OPPOSITE direction -- {detail}"
                )
        entry["status"] = verdict
        verdicts[entry["id"]] = verdict

    path.write_text(yaml.safe_dump(document, sort_keys=False, width=88), encoding="utf-8")
    return verdicts
