import { LineChart, seriesColor, type Series } from './components/Chart'
import { RegimeStrip } from './components/Header'
import { fmt, type Tick } from './api'

/**
 * Market tab: what the simulated market is doing right now, what regime the
 * causal classifier believes it is in, and where every strategy is positioned.
 */
export function MarketTab({ tick, assets }: { tick: Tick | null; assets: string[] }) {
  if (!tick) return <div className="empty">starting session…</div>

  const priceSeries: Series[] = assets.map((asset, i) => ({
    name: asset,
    kind: 'strategy',
    colorIndex: i,
    points: tick.prices.map((row) => ({
      t: String(row.t),
      v: typeof row[asset] === 'number' ? (row[asset] as number) : null,
    })),
  }))

  const volSeries: Series[] = [
    { name: 'realised vol (20d, annualised)', kind: 'portfolio', colorIndex: 0, points: tick.realized_vol },
  ]

  const latestVol = [...tick.realized_vol].reverse().find((p) => p.v != null)?.v ?? null
  const strategies = tick.strategies.filter((s) => s.kind !== 'portfolio')
  const portfolio = tick.strategies.find((s) => s.name === 'PORTFOLIO regime-conditional')

  return (
    <div className="grid" style={{ gridTemplateColumns: '1fr' }}>
      <section className="panel">
        <header>
          <h2>Price</h2>
          <span className="sub">
            bar {fmt.int(tick.cursor)} of {fmt.int(tick.n_steps)} · shaded bands are the detected
            regime, classified causally from trailing data only
          </span>
        </header>
        <LineChart
          series={priceSeries}
          bands={tick.regime.bands}
          height={280}
          yFormat={(v) => v.toFixed(0)}
          showBandLabels
        />
        <div style={{ marginTop: 10 }}>
          <RegimeStrip tick={tick} />
        </div>
      </section>

      <div className="grid" style={{ gridTemplateColumns: 'minmax(0, 1.4fr) minmax(0, 1fr)' }}>
        <section className="panel">
          <header>
            <h2>Realised volatility</h2>
            <span className="sub">
              20-bar trailing, annualised · drives Tier 1 position sizing
            </span>
          </header>
          <LineChart
            series={volSeries}
            height={150}
            yFormat={(v) => `${(v * 100).toFixed(0)}%`}
          />
          <div className="stat-row" style={{ marginTop: 10 }}>
            <div className="stat">
              <span className="stat-label">Current</span>
              <span className="stat-value">{fmt.pct(latestVol, 1)}</span>
            </div>
            <div className="stat">
              <span className="stat-label">Regime</span>
              <span className="stat-value">
                {tick.regime.current ? tick.regime.current.replace(/_/g, ' ') : '--'}
              </span>
            </div>
            <div className="stat">
              <span className="stat-label">Budget used today</span>
              <span className="stat-value">
                {tick.budget.used_today} / {tick.budget.limit}
              </span>
            </div>
          </div>
        </section>

        <section className="panel">
          <header>
            <h2>Portfolio book</h2>
            <span className="sub">post-budget, volatility-targeted</span>
          </header>
          <div className="positions">
            {Object.entries(portfolio?.positions ?? tick.positions).map(([asset, value]) => {
              const magnitude = Math.min(Math.abs(value ?? 0), 1)
              return (
                <div className="position" key={asset}>
                  <div className="asset">{asset}</div>
                  <div className="value num">{fmt.num(value, 3)}</div>
                  <div className="bar">
                    <span
                      style={{
                        left: (value ?? 0) >= 0 ? '50%' : `${50 - magnitude * 50}%`,
                        width: `${magnitude * 50}%`,
                      }}
                    />
                  </div>
                </div>
              )
            })}
          </div>
        </section>
      </div>

      <section className="panel">
        <header>
          <h2>Strategy positions</h2>
          <span className="sub">
            each strategy standalone and unbudgeted · the two controls are set apart
          </span>
        </header>
        <table>
          <thead>
            <tr>
              <th>Strategy</th>
              {assets.map((a) => (
                <th key={a} className="num">
                  {a.replace('asset_', 'A')}
                </th>
              ))}
              <th className="num">Gross exposure</th>
            </tr>
          </thead>
          <tbody>
            {strategies.map((s, i) => {
              const gross = Object.values(s.positions ?? {}).reduce(
                (sum, v) => sum + Math.abs(v ?? 0),
                0,
              )
              return (
                <tr key={s.name} className={s.kind === 'control' ? 'control-row' : ''}>
                  <td>
                    <span className="row-name">
                      <span
                        className={`row-rule ${s.kind === 'control' ? 'dashed' : ''}`}
                        style={{ color: seriesColor(s.kind, i) }}
                      />
                      {s.name}
                      {s.kind === 'control' && <span className="badge-control">CONTROL</span>}
                    </span>
                  </td>
                  {assets.map((a) => (
                    <td key={a} className="num">
                      {fmt.num(s.positions?.[a], 2)}
                    </td>
                  ))}
                  <td className="num">{fmt.num(gross, 2)}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </section>
    </div>
  )
}
