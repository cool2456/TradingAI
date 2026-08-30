import { LineChart, Legend, seriesColor, type Series } from './components/Chart'
import { fmt, type Tick } from './api'

/**
 * Strategies tab: every strategy and both controls on one axis, a ranked
 * ledger, and the tape of closed round trips.
 *
 * Win rate is placed immediately beside P&L because the two diverge constantly
 * -- a strategy can win 55% of bars and still lose money, and a strategy can
 * win 40% and make a fortune. Seeing them apart invites the reader to treat
 * either one as a summary of the other.
 */
export function StrategiesTab({ tick }: { tick: Tick | null }) {
  if (!tick) return <div className="empty">starting session…</div>

  let strategyIndex = 0
  let controlIndex = 0
  let portfolioIndex = 0
  const series: Series[] = tick.strategies.map((s) => {
    const colorIndex =
      s.kind === 'control' ? controlIndex++ : s.kind === 'portfolio' ? portfolioIndex++ : strategyIndex++
    return { name: s.name, kind: s.kind, colorIndex, points: s.equity }
  })

  const ranked = [...tick.strategies].sort(
    (a, b) => (b.stats.sharpe_ann ?? -Infinity) - (a.stats.sharpe_ann ?? -Infinity),
  )
  const colorOf = (name: string) => {
    const match = series.find((s) => s.name === name)
    return match ? seriesColor(match.kind, match.colorIndex ?? 0) : 'var(--text-dim)'
  }

  const controls = tick.strategies.filter((s) => s.kind === 'control')
  const best = ranked.find((s) => s.kind !== 'control')
  const beatsControls =
    best != null &&
    controls.every((c) => (best.stats.sharpe_ann ?? -Infinity) > (c.stats.sharpe_ann ?? -Infinity))

  const signClass = (v: number | null | undefined) =>
    v == null || !isFinite(v) ? '' : v > 0 ? 'pos' : v < 0 ? 'neg' : ''

  return (
    <div className="grid" style={{ gridTemplateColumns: '1fr' }}>
      <section className="panel">
        <header>
          <h2>Equity curves</h2>
          <span className="sub">net of costs · controls drawn dashed</span>
        </header>
        <LineChart series={series} height={300} yFormat={(v) => v.toFixed(2)} />
        <Legend series={series} />
      </section>

      <section className="panel">
        <header>
          <h2>Ledger</h2>
          <span className="sub">
            ranked by net Sharpe · a strategy that does not beat both controls has failed
          </span>
        </header>
        <table>
          <thead>
            <tr>
              <th>Strategy</th>
              <th className="num">P&amp;L</th>
              <th className="num paired">Win rate</th>
              <th className="num">Sharpe</th>
              <th className="num">Max DD</th>
              <th className="num">Trades</th>
              <th className="num">Cost paid</th>
              <th className="num">Bars</th>
            </tr>
          </thead>
          <tbody>
            {ranked.map((s) => (
              <tr
                key={s.name}
                className={
                  s.kind === 'control' ? 'control-row' : s.kind === 'portfolio' ? 'portfolio-row' : ''
                }
              >
                <td>
                  <span className="row-name">
                    <span
                      className={`row-rule ${s.kind === 'control' ? 'dashed' : ''}`}
                      style={{ color: colorOf(s.name) }}
                    />
                    {s.name}
                    {s.kind === 'control' && <span className="badge-control">CONTROL</span>}
                    {s.kind === 'portfolio' && <span className="badge-portfolio">BOOK</span>}
                  </span>
                </td>
                <td className={`num ${signClass(s.stats.pnl)}`}>{fmt.pct(s.stats.pnl)}</td>
                <td className="num paired">{fmt.pct(s.stats.win_rate, 1)}</td>
                <td className={`num ${signClass(s.stats.sharpe_ann)}`}>
                  {fmt.num(s.stats.sharpe_ann)}
                </td>
                <td className={`num ${signClass(s.stats.max_drawdown)}`}>{fmt.pct(s.stats.max_drawdown)}</td>
                <td className="num">{fmt.int(s.stats.trades)}</td>
                <td className="num">{fmt.pct(s.stats.cost_paid)}</td>
                <td className="num">{fmt.int(s.stats.n_periods)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="notice">
          Every figure is net of transaction costs and computed over the bars revealed so far, so
          early in a session they are estimated on very little data. Win rate sits beside P&amp;L
          deliberately: they diverge constantly, and neither one summarises the other.
          {best && (
            <>
              {' '}
              Best non-control strategy is <strong>{best.name}</strong>, which{' '}
              {beatsControls ? 'beats' : 'does NOT beat'} both controls.
            </>
          )}
        </p>
      </section>

      <section className="panel">
        <header>
          <h2>Trade tape</h2>
          <span className="sub">
            closed round trips in the regime-conditional book · most recent first
          </span>
        </header>
        {tick.trade_tape.length === 0 ? (
          <div className="empty">no closed round trips yet</div>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Asset</th>
                <th>Side</th>
                <th>Opened</th>
                <th>Closed</th>
                <th className="num">Bars held</th>
                <th className="num">P&amp;L</th>
              </tr>
            </thead>
            <tbody>
              {tick.trade_tape.map((trip, i) => (
                <tr key={`${trip.asset}-${trip.closed}-${i}`}>
                  <td>{trip.asset}</td>
                  <td>{trip.side}</td>
                  <td className="num">{fmt.date(trip.opened)}</td>
                  <td className="num">{fmt.date(trip.closed)}</td>
                  <td className="num">{fmt.int(trip.bars)}</td>
                  <td className={`num ${signClass(trip.pnl)}`}>{fmt.pct(trip.pnl, 3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </div>
  )
}
