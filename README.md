# quantlab

A research system for measuring whether trading signals contain information,
and for evaluating a multi-strategy book honestly when they do.

---

# PHASE 2 RESULT: no tradeable signal

**Did step 5 find any signal clearing the FDR-corrected bar?**
Four of 21 pre-registered tests cleared it. **None of them is tradeable, and
none is stable across the sample.** The answer to the phase question -- *is
there an information coefficient reliably above zero at a horizon where costs
do not eat it* -- is **no**.

| | signal | h (bars) | raw IC | raw HAC t | residual IC | IID t | **HAC t** | n_effective | BY p |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H003 | timeseries_momentum | 30 | 0.0033 | +0.79 | +0.0088 | +26.27 | **+3.14** | 128,669 | 0.0058 |
| H001 | zscore_reversion | 130 | −0.0050 | −1.25 | **−0.0082** | −20.74 | **−3.00** | 142,112 | 0.0099 |
| H005 | vol_breakout | 130 | 0.0045 | +1.00 | +0.0089 | +22.72 | **+2.85** | 101,854 | 0.0172 |
| H002 | zscore_momentum | 130 | 0.0038 | +0.61 | +0.0124 | +31.46 | **+2.75** | 55,114 | 0.0249 |

Four reasons the four survivors do not constitute a finding:

**0. Nothing is significant before residualising.** Not one raw IC clears
significance — the largest is |t| = 1.25 across all 21 tests. The effect exists
only in the market-neutral residual, so capturing it requires a dollar-neutral
long-short book across the full universe. Run long-only, there is no
statistically detectable signal here at all.

The brief expected residualising to *shrink* the IC by stripping out market
beta. It does the opposite: the residual IC is larger than the raw IC in every
case. Cross-sectional demeaning removes the dominant common factor, which for a
cross-sectional signal is mostly noise, so it raises signal-to-noise rather
than removing contamination. That makes the result more fragile, not less —
it is conditional on being able to run market-neutral.

**1. They vanish in the second half of the sample.** Every one is significant
in the first six months and none in the second:

| signal | h | full year | H1 (Sep–Feb) | H2 (Mar–Aug) |
|---|---:|---:|---:|---:|
| timeseries_momentum | 30 | +0.0088 (t 3.14) | +0.0149 (t **4.51**) | +0.0030 (t 0.68) |
| zscore_reversion | 130 | −0.0082 (t −3.00) | −0.0100 (t **−3.07**) | −0.0064 (t −1.49) |
| vol_breakout | 130 | +0.0089 (t 2.85) | +0.0122 (t **3.29**) | +0.0058 (t 1.16) |
| zscore_momentum | 130 | +0.0124 (t 2.75) | +0.0164 (t **3.15**) | +0.0086 (t 1.18) |

By quarter the effect is concentrated almost entirely in Q2. The *sign* agrees
with the full sample in 4 of 4 quarters for all four signals, which is more
than noise would usually give, so this is not certainly nothing — but a full
-sample t of 3.1 that is 4.5 in one half and 0.7 in the other is not an
estimate of a stable quantity.

**2. The break-even cost sits inside the realistic cost band.** A dollar
-neutral book sized on the standardised signal earns `IC × σ_r` gross per round
trip and pays `2 × c × E|z|`, so it breaks even at `c* = IC·σ_r / (2·E|z|)`:

| signal | h | σ_r | gross / round trip | break-even one-way |
|---|---:|---:|---:|---:|
| zscore_momentum | 130 | 1.071% | 1.33 bp | **0.76 bp** |
| zscore_reversion | 130 | 1.071% | 0.87 bp | 0.50 bp |
| vol_breakout | 130 | 1.071% | 0.96 bp | 0.48 bp |
| timeseries_momentum | 30 | 0.566% | 0.50 bp | 0.25 bp |

Realistic all-in one-way cost for US large caps is 0.5–1.5 bp. The best
break-even is 0.76 bp: it clears an optimistic 0.5 bp and fails at 1.0 bp. That
is before market impact, implementation shortfall, and any capacity constraint.

**3. It cannot be expressed under the trade budget.** Capturing an h=130 signal
across 100 names needs roughly 300 order submissions per day. The system's
budget is 3.

### What *was* established

The pre-registered mechanism for H001 is **refuted with a sign reversal**.
H001 predicted intraday reversion from liquidity provision, `IC > +0.02`; the
measured residual IC is `−0.0082` with HAC t = −3.00. H002, registered as a
prediction of failure (`IC ≤ 0`), came in significantly *positive*. Both point
the same way, because reversion is the negative of momentum: over this year,
these names showed weak intraday **momentum**, not mean reversion. That is a
genuine pre-registered refutation, and it is the phase's real output.

Registered verdicts, graded mechanically against the registered direction:

| | prediction | verdict |
|---|---|---|
| H001 | IC > 0.02 at h=130 | **falsified, sign reversal** |
| H002 | IC ≤ 0 (registered failure) | **falsified, sign reversal** — a positive IC here falsifies the H001 mechanism, exactly as pre-registered |
| H003 | null at all horizons | **falsified** — significant at h=30 |
| H004 | tracks H003 within 0.01, no independent signal | **held** — h=130 IC 0.0089 vs H003's 0.0105, gap 0.0016 |
| H005 | IC > 0 at h=30 only, decaying by h=390 | **sign held, pattern falsified** — the effect appears at h=130, not h=30 |
| H006 | random_signal, IC = 0 | **held** — IC ≤ 0.0004, \|t\| ≤ 0.94 at every horizon |
| H007 | always_long, IC undefined | **held** — zero-variance signal, no IC exists |

**Effective sample size behind the claims: 47,901 to 142,112** against 6.5–9.0
million raw observations. The measured variance inflation runs 40× to 158×.
Both controls behaved: `random_signal` returned IC ≤ 0.0004 with n_effective
≈ 9,000,000 — essentially its raw count, because an iid signal suffers no
inflation, which is the mechanism check.

**Trial count: 29. Deflated-Sharpe luck threshold: 3.37 annualised Sharpe.**
Read from `research_log.jsonl`; refresh with
`python -c "from quantlab.validate import trial_count, current_threshold; print(trial_count(), current_threshold(2500)['luck_threshold_ann'])"`.

### Caveats that bound every number above

- **Data is historical SIP; live data would be IEX.** Alpaca's free tier serves
  real time from IEX only — roughly 2% of US equity volume with quotes wider
  than the NBBO. Nothing here was measured on IEX, and nothing should be.
- **Alpaca's paper engine fills optimistically**: no queue position, no partial
  fills, no market impact. Paper slippage is a *lower bound* on real slippage.
  Passing paper is necessary and not sufficient.
- **The universe is survivorship-biased.** It was ranked by liquidity as of the
  backfill start date, which removes look-ahead from the *ranking*, but the
  candidate pool is Alpaca's current membership. Any name delisted between
  2025-08-30 and today is absent. Not fixable with this data source.
- **This system has never traded real money.** There is no live endpoint, no
  live keys, and no configuration path that reaches one; `config.py` refuses
  the live URL by validator. A backtest is a hypothesis, not a result.

### The measurement corrections, and what each was worth

| correction | effect on the answer |
|---|---|
| **Entry lag** (`P_{t+1+h}/P_{t+1}`, not `P_{t+h}/P_t`) | On a driftless random walk seen through an alternating half-spread — true reversion exactly zero — the naive form gives **IC +0.2731 at HAC t +69.50**. With entry_lag=1: −0.0013, t −0.32. Largest correction in the project, and no standard error fixes it: it is a bias in the estimate, not its variance. On these very liquid names at 1-minute bars it happened to change little (spreads are tiny relative to 1-minute volatility), but that is an empirical accident, not a reason to omit it. |
| **HAC vs IID standard errors** | IID t-statistics run **8–11× the HAC values** on this data (zscore_momentum h=130: IID +31.46 vs HAC +2.75). Under a synthetic null the IID SE rejects on **63–71%** of samples at a nominal 5%; Driscoll-Kraay brings that to 5–9%. |
| **Residualising** | Raw returns have mean pairwise correlation +0.131, giving N_eff of **7.2** of 100 names. Cross-sectional demeaning takes that to +0.0037 and N_eff **72.9**. Breadth is worth 10× more after residualising — this is what made a one-year study feasible at all. |
| **Fund exclusion** | The unfiltered liquidity ranking was ~40% ETFs including TQQQ, SQQQ, SOXL, SOXS. A 3× daily ETF's "reversion" is volatility-decay rebalancing, and SPY/VOO/IVV *are* the factor being residualised against. |

---

# PHASE 1: regime-conditional multi-strategy evaluation

Evaluating a multi-strategy book under **a hard budget of 3 trades per day
across the entire portfolio** and **regime-conditional strategy allocation**
computed from causally available information only.

> **Status after Phase 2: `regime_tier` now defaults to 0 (fixed weights).**
> Tier 2 remains implemented and tested but is not deployed. See the Phase 1
> headline finding below for why.

---

## Read this before any number below

**Every reported figure is net of transaction costs.** Gross figures are
computed for diagnosis and are labelled `_DIAGNOSTIC`; they are never a
headline. With zero skill a strategy does not break even, it bleeds cost, and
the tests assert exactly that.

**Every reported figure is evaluated on data not used to choose it.** Parameter
selection happens only inside `walk_forward`, on a training window, and is
scored on the window that follows it. There is deliberately no
`best_parameters()` convenience function.

**Trial count: 8. Deflated-Sharpe luck threshold: 2.39 annualised Sharpe.**

Those come from `research_log.jsonl`, which every evaluation entry point appends
to. The threshold is the Sharpe the best of 8 worthless strategies would be
expected to reach by luck alone. It rises every time anyone runs anything.
Refresh it with:

```bash
python -c "from quantlab.validate import trial_count, current_threshold; \
print(trial_count(), current_threshold(2500)['luck_threshold_ann'])"
```

The threshold is currently high because the logged trials scatter widely (from
−3.6 to +0.9 annualised). The deflation uses the observed variance of trial
Sharpes, so a search that ranged over genuinely different strategies is charged
more than one that nudged a single parameter. That is the intended behaviour and
it is uncomfortable, which is the point.

**This is a research tool. It has never traded real money and has no broker
connectivity, order routing, or exchange interface of any kind. A backtest is a
hypothesis, not a result.**

---

## The headline finding: the regime logic does not earn its keep

The brief asked that this be reported plainly if true. It is true.

Regime-conditional allocation was compared against an identical fixed
equal-weight allocation — same signals, same sizing, same budget, same costs,
same lag, differing *only* in whether strategy weights vary with the detected
regime. 40 seeds per generator, 2500 bars, 2bp costs:

| Market generator | Regime-conditional | Fixed weight | Uplift | t | Win rate |
|---|---:|---:|---:|---:|---:|
| `regime_switching` | **+0.89** | +0.01 | **+0.89** | 21.1 | 100% |
| `gbm` (null) | −0.09 | +0.01 | **−0.11** | −3.7 | 35% |
| `garch_t` | −0.11 | −0.03 | **−0.07** | −2.4 | 35% |
| `ornstein_uhlenbeck` | −3.08 | −3.61 | +0.53 | 18.3 | 100% |

Read across the rows rather than down the first column:

1. **It works decisively on the one generator built to contain the structure it
   assumes.** `regime_switching` has a real, persistent latent state, and the
   classifier finds it. This is a sanity check that the machinery functions, not
   evidence that regimes exist in markets.

2. **On a random walk and on GARCH it actively loses money**, significantly, in
   ~65% of seeds. Regime conditioning is not free and not neutral: chasing
   spurious regime transitions raises turnover, and turnover is charged. A
   fixed allocation beats it.

3. **On Ornstein-Uhlenbeck the "uplift" is meaningless** because both books are
   catastrophic. Standalone `zscore_reversion` earns **+2.73** on the same data;
   the portfolio turns it into **−3.08**. The default weight map says "trend
   regime → weight momentum," the efficiency ratio labels roughly half of a
   mean-reverting series as trending, and the resulting book is net-momentum on
   mean-reverting data. Regime-conditional allocation with hand-set priors
   destroyed a real edge that a single unconditional strategy captured.

The one-line verdict: **on this evidence the Tier-2 regime layer is not
supported.** It helps where regimes were planted, hurts where they were not, and
its cost is paid in every case.

### The label itself is not significant

Information coefficient of the trend/chop label against next-day portfolio
return, one seed per generator:

| Generator | IC | naive t | **HAC t** | p |
|---|---:|---:|---:|---:|
| `regime_switching` | +0.040 | 1.99 | **1.81** | 0.071 |
| `gbm` | +0.023 | 1.12 | **1.08** | 0.278 |
| `garch_t` | +0.019 | 0.96 | **0.90** | 0.369 |
| `ornstein_uhlenbeck` | +0.020 | 1.00 | **1.02** | 0.309 |

Note the first row. The naive standard error puts it at t = 1.99, just over the
1.96 threshold; the autocorrelation-corrected standard error puts it at 1.81,
just under. That is the difference between a finding and a non-finding, on the
generator most favourable to the hypothesis, produced entirely by which standard
error you use.

### Nothing reliably beats the controls

Mean net annualised Sharpe, 40 seeds. Controls in **bold**:

| | `regime_switching` | `gbm` | `garch_t` | `ornstein_uhlenbeck` |
|---|---:|---:|---:|---:|
| PORTFOLIO regime-conditional | 0.89 | −0.09 | −0.11 | −3.08 |
| PORTFOLIO fixed-weight | 0.01 | 0.01 | −0.03 | −3.61 |
| zscore_reversion | 0.84 | −0.19 | −0.17 | 2.73 |
| zscore_momentum | −0.57 | −0.04 | −0.08 | −4.25 |
| timeseries_momentum | 0.12 | 0.02 | −0.01 | −3.13 |
| ma_crossover | 0.08 | 0.01 | −0.04 | −2.88 |
| vol_breakout | −0.39 | −0.04 | −0.08 | −2.77 |
| **random_signal** | **−0.58** | **−0.60** | **−0.68** | **−0.61** |
| **always_long** | **0.13** | **−0.01** | **−0.01** | **0.15** |

On `garch_t`, **nothing beats both controls.** On `gbm` the three that nominally
clear the bar do so by 0.02–0.03 Sharpe against a control bar of −0.01, which is
noise. `random_signal` sits at −0.6 in every column — that is the cost of
trading with no information, and it is remarkably stable.

---

## Non-negotiable conventions

**Causality.** A position at `t` depends only on information available at `t`.
`tests/test_causality.py` perturbs all prices strictly after a random index `t`,
recomputes, and asserts nothing at or before `t` moved — over 25 random `t`
values, for every signal and every regime classifier. It also carries two
deliberately leaky functions that must fail, including full-sample
normalisation, so the test demonstrably has teeth.

**The controls are mandatory.** `random_signal` (noise floor) and `always_long`
(buy and hold) appear in every table with the same prominence as the
strategies, and the frontend cannot render a ledger without them.

**Honest trial accounting.** Every configuration evaluated appends one line to
`research_log.jsonl`; `deflated_sharpe` reads its trial count from that file. A
9-point parameter sweep costs 9 trials and raises the bar for every later
result. The log is append-only.

**Vectorised.** Python loops appear in exactly three places — GARCH variance,
the Markov chain, and the trade-budget day loop — each genuinely sequential and
each carrying a comment explaining why.

---

## The three tiers

Regime conditioning multiplies the hypothesis space. With `S` strategies and `R`
regimes you fit `S x R` decisions instead of `S`.

| Tier | What it drives | Status |
|---|---|---|
| **1** | Volatility regime → position sizing | **Well supported.** Volatility is genuinely autocorrelated and forecastable. Standard practice, not a research bet. |
| **2** | Trend/chop regime → strategy weights | **Testable, and on the evidence above not supported.** Charged for in the DSR. |
| **3** | 12 fine-grained regimes → strategy selection | **Dangerous.** Requires `allow_tier3=True`. |

Default configuration uses Tier 1 and Tier 2 only.

`quantlab.regime.hypothesis_cost` makes the cost visible:

```python
>>> hypothesis_cost(n_strategies=5, n_regimes=4, n_periods=2500)
{'luck_threshold_ann': 0.379, 'luck_threshold_conditioned_ann': 0.604,
 'multiplier': 1.594, 'bars_per_regime': 625.0, ...}
```

Moving from 5 strategies to 5 across 4 regimes raises the luck threshold by
**59%**, not the 37% the crude `sqrt(2 ln N)` asymptotic suggests — that
approximation converges slowly and understates the ratio at these sample sizes.
Monte Carlo puts the true multiplier at 1.605; the Bailey–López de Prado
expectation used here lands within 1%.

`regime_sample_counts` reports the second, separate cost — that each
regime-conditional weight is estimated on only the bars in its regime. Under
Tier 3 the rarest of 12 regimes held **73 bars**, implying a Sharpe standard
error of **1.86 annualised**. Nothing measured on that is real.

### Tier 3, measured

The twelve Tier-3 weight vectors are derived mechanically from the two
trend/chop priors by two rules — an autocorrelation tilt and a
volatility-driven concentration exponent — rather than hand-written, because
twelve hand-written vectors are twelve chances to encode a result already seen
in the data. Nothing is fitted. On one seed, 2500 bars:

| | `regime_switching` | `gbm` (null) |
|---|---:|---:|
| Fixed weight | +0.61 | **+0.36** |
| Tier 2 (2 regimes) | +1.22 | −0.08 |
| Tier 3 (12 regimes) | **+1.52** | **−0.34** |

Tier 3 is the best book where regimes genuinely exist and **four times worse
than Tier 2** where they do not, against a fixed allocation that beats both.
Finer conditioning amplifies whatever is present — signal where there is
signal, damage where there is none. You cannot know ex ante which case you are
in. That asymmetry is the entire argument against Tier 3, and it is why it
requires `allow_tier3=True`.

A bug worth recording, because it is the failure mode this whole design is
arranged to catch: the first working version of Tier 3 produced a book
*identical* to the fixed-weight one. Tier-3 labels are compound
(`high_vol|trend|ac-`) while the priors were keyed `trend`/`chop`, so every
lookup missed, every bar fell back to equal weight, and the result was a
fixed-weight portfolio reported as regime-conditional. Every number was
arithmetically valid; only the label was wrong. `resolve_weight_map` now
resolves compound labels to their components and `run_regime_portfolio` raises
when coverage is zero rather than silently degrading.

---

## Falsification tools, in order of usefulness

**`information_coefficient(signal, forward_return, overlap=h)`** is the fastest
way to kill a bad idea. It reports both a naive standard error and a
Newey–West HAC standard error. With overlapping forward returns the naive
version inflates the t-statistic by roughly `sqrt(h)`; at `h = 20` that is a
factor of 4.5, which is more than enough to manufacture a publishable result
from nothing. `tests/test_metrics.py` demonstrates this directly: under a
constructed null with 20-bar overlap, where the true IC is zero by construction,
the naive standard error rejects on **55.5%** of samples against a nominal 5%.
The HAC version rejects on 10.0%. Eleven of every twenty "discoveries" the naive
test would report on such data are the standard error, not the signal.

**`deflated_sharpe(returns, n_trials, sharpe_variance)`** — Bailey and López de
Prado. All Sharpes are per-period internally; the annualised/per-period
distinction is enforced at every boundary because converting inside the formula
inflates the statistic by `sqrt(252)`.

The DSR is verified by calibration rather than by reproducing a published
arithmetic example: under the null, the DSR of the best of N trials should be
approximately uniform, so `DSR > 0.95` should fire on about 5% of null
experiments. It does.

The comparison is stark. Take the best of 200 pure-noise strategies and ask
whether its Sharpe is significant: the undeflated PSR says yes on **100%** of
such experiments. The deflated Sharpe says yes on **0%**. Both are looking at
the same returns; only one of them knows how many strategies were discarded to
find them.

---

## Install and run

```bash
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
uv pip install --python .venv/bin/python "uvicorn[standard]" fastapi pydantic

.venv/bin/python -m pytest tests/ -q          # 129 tests, ~16s
```

Backend and frontend:

```bash
.venv/bin/python -m uvicorn quantlab.server:app --port 8000
cd frontend && npm install && npm run dev     # http://localhost:5173
```

Minimal use:

```python
from quantlab.simulate import regime_switching
from quantlab.engine import run_regime_portfolio

prices = regime_switching(n_steps=2500, n_assets=4, seed=7)
result = run_regime_portfolio(prices, tier=2, budget=3, cost_bps=2.0)

print(result.ledger())          # every strategy, both controls, both books
print(result.regime_verdict())  # did regime conditioning beat fixed weights?
```

Walk-forward, which is the only place parameters may be chosen:

```python
from quantlab.validate import walk_forward
from quantlab.signals import zscore_reversion

wf = walk_forward(prices, zscore_reversion.bind, {"window": [5, 10, 20, 40, 80]},
                  train=504, test=126, cost_bps=2.0, warmup=80)
print(wf.summary())
print(wf.stability["verdict"])   # parameter instability is itself a finding
```

On Ornstein-Uhlenbeck the selection concentrates on one window (entropy 0.15)
and out-of-sample Sharpe is 2.21. On GBM the selection scatters near-uniformly
across the grid (entropy 0.93) and out-of-sample Sharpe is −0.32. That scatter
is what selecting on noise looks like, and `walk_forward` reports it.

---

## Corrections made to the original specification

Six places where the brief as written would have produced wrong numbers.

1. **The null test contradicted the "no neutral outcome" rule.** The brief asked
   for mean out-of-sample Sharpe within two standard errors of zero. That is
   right for gross returns and wrong for net: a system whose net Sharpe on
   Brownian motion were genuinely zero would not be charging for its own
   trading. The test now asserts gross ≈ 0 **and** net significantly negative.

2. **The IC standard error `1/sqrt(n-1)` was wrong twice.** Under H₀ the exact
   test uses `1/sqrt(n-2)`; far more importantly, overlapping forward returns
   make the observations dependent and the effective sample far smaller than
   `n`. Newey–West HAC is now the headline, with the naive value shown beside
   it.

3. **The trade-budget value function was dimensionally inconsistent.**
   `|Δ|·edge − cost_bps/1e4` scales the benefit with trade size but not the
   cost, so a 0.02 tweak is charged the same as a full flip. It is now
   `|Δ|·(edge − c) − f`. What the brief wrote is a *ticket charge* wearing the
   name of a proportional cost. A related non-obvious property, tested: a
   uniform fixed cost can never reorder the queue (it subtracts a constant), it
   only imposes a minimum viable trade size; the proportional cost *does*
   reorder, but only when edges differ across assets.

4. **The engine applied the lag twice.** `position_t = target.shift(lag)` and
   `gross_{t+1} = position_t · ret_{t+1}` compose to `lag+1` in pandas, since
   `pct_change()` at `t` already covers `(t−1, t]`. One shift, applied in one
   place.

5. **"3 trades per day" was undefined under volatility targeting**, which nudges
   every position on every bar. Without a dead band the trade count is unbounded
   and the budget is exhausted by rounding noise. `min_trade_size` is now
   explicit and defaults to 0.05.

6. **`sqrt(2·ln(S·R)/T)` understates the cost of regime conditioning.** The
   brief's 37% arithmetic is internally correct but the asymptotic is not
   accurate at these `N`; the true figure is ~60%. See the tiers section.

Two further deviations, made for correctness rather than to fix an error:

- **`apply_trade_budget` takes `current` as an initial condition, not a panel.**
  The brief types it as a DataFrame alongside `desired`, implying the panel can
  be evaluated at once. It cannot: a trade the budget blocks on Monday changes
  Tuesday's book, its deltas, its ranking and its budget consumption. The
  process is simulated forward. A DataFrame is still accepted and its first row
  seeds the book.

- **`walk_forward` feeds each test block the preceding `warmup` bars as
  context.** Recomputing a 60-bar-lookback signal from scratch on a 126-bar test
  block discards half the out-of-sample data and measures cold-start behaviour.
  Using prior bars is causal — they are past data throughout — and is what
  happens live.

One request not fulfilled: the brief asked that `deflated_sharpe` reproduce "the
published worked example." That paper was not available here and its numbers
were not invented. The Monte Carlo calibration test described above is used
instead, and is a stronger check.

---

## Anti-goals

No broker connectivity, order routing, or real-money execution. No machine
learning — the bottleneck is measurement honesty, not model capacity. No
parameter optimisation outside `walk_forward`, and no "best parameters"
convenience function: it would be used, and it would produce numbers that mean
nothing.

## Layout

```
src/quantlab/
  simulate.py   gbm · ornstein_uhlenbeck · garch_t · regime_switching
  signals.py    5 strategies + 2 mandatory controls, each with .lookback
  regime.py     causal classification, expanding quantiles, 3 tiers
  allocator.py  regime weights, trade-budget selection and enforcement
  sizing.py     volatility targeting with a hard leverage cap, fractional Kelly
  engine.py     the accounting, and regime-portfolio composition
  metrics.py    Sharpe · IC with HAC · DSR · PSR · expected max Sharpe
  validate.py   walk-forward, purged k-fold with embargo, trial logging
  server.py     FastAPI backend
tests/          129 tests; test_causality.py gates everything downstream
frontend/       React + Vite + TypeScript
research_log.jsonl   append-only; never edit by hand
```
