import type { Tick } from '../api'

/**
 * Persistent chrome. The regime badge and the trade budget are shown on every
 * tab without exception: both are properties of the whole system, and a reader
 * who cannot see how much budget remains cannot interpret why a signal did not
 * result in a trade.
 */

export function RegimeBadge({ tick }: { tick: Tick | null }) {
  const current = tick?.regime.current
  const truth = tick?.regime.true_state
  return (
    <div className="regime-badge">
      <span className="label">Regime</span>
      <span className="value">{current ? current.replace(/_/g, ' ') : 'warming up'}</span>
      <span className="tier">tier {tick?.regime.tier ?? '-'}</span>
      {truth && (
        <span className="tier" title="The generator's true latent state. Available only in simulation and never fed to a strategy.">
          truth: {truth}
        </span>
      )}
    </div>
  )
}

export function BudgetIndicator({ tick }: { tick: Tick | null }) {
  const used = tick?.budget.used_today ?? 0
  const limit = tick?.budget.limit ?? 0
  const exhausted = limit > 0 && used >= limit
  return (
    <div className={`budget ${exhausted ? 'exhausted' : ''}`} title="Trades used today across the entire portfolio. Resets each session.">
      <span className="label">Budget</span>
      <span className="budget-pips">
        {Array.from({ length: limit }, (_, i) => (
          <span key={i} className={`pip ${i < used ? 'used' : ''}`} />
        ))}
      </span>
      <span className="budget-count">
        {used} / {limit} trades used today
      </span>
    </div>
  )
}

export function RegimeStrip({ tick }: { tick: Tick | null }) {
  const bands = tick?.regime.bands ?? []
  if (!bands.length) return null
  const total = bands.reduce((sum, b) => sum + b.bars, 0)
  const order = ['low_vol', 'normal_vol', 'high_vol', 'chop', 'trend']
  const names = Array.from(new Set(bands.map((b) => b.regime))).sort((a, b) => {
    const ia = order.indexOf(a)
    const ib = order.indexOf(b)
    if (ia !== -1 && ib !== -1) return ia - ib
    return a.localeCompare(b)
  })
  const alphas = [0.1, 0.2, 0.32, 0.44, 0.56]

  return (
    <div>
      <div style={{ display: 'flex', height: 14, borderRadius: 3, overflow: 'hidden', border: '1px solid var(--border)' }}>
        {bands.map((band, i) => (
          <div
            key={`${band.start}-${i}`}
            title={`${band.regime} - ${band.bars} bars from ${band.start.slice(0, 10)}`}
            style={{
              width: `${(band.bars / total) * 100}%`,
              background: `rgba(207, 216, 230, ${alphas[names.indexOf(band.regime) % alphas.length]})`,
            }}
          />
        ))}
      </div>
      <div className="legend" style={{ marginTop: 6 }}>
        {names.map((name, i) => (
          <div key={name} className="legend-item">
            <span
              style={{
                width: 12,
                height: 12,
                borderRadius: 2,
                border: '1px solid var(--border)',
                background: `rgba(207, 216, 230, ${alphas[i % alphas.length]})`,
              }}
            />
            <span>{name.replace(/_/g, ' ')}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
