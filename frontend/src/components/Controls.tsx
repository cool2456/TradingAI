import type { SessionConfig } from '../api'

/**
 * The control bar is persistent across both tabs.
 *
 * Tier 3 is gated behind an explicit acknowledgement, mirroring the
 * ``allow_tier3=True`` argument the Python API requires. A twelve-regime
 * classifier fitted on a few hundred bars per cell is not a setting like the
 * others and is not presented as one.
 */

interface Props {
  config: SessionConfig
  onChange: (patch: Partial<SessionConfig>) => void
  running: boolean
  onToggleRun: () => void
  onRestart: () => void
  speed: number
  onSpeed: (v: number) => void
  busy: boolean
}

export function Controls({
  config,
  onChange,
  running,
  onToggleRun,
  onRestart,
  speed,
  onSpeed,
  busy,
}: Props) {
  return (
    <div className="controlbar">
      <button className="btn primary" onClick={onToggleRun} disabled={busy}>
        {running ? 'Pause' : 'Play'}
      </button>
      <button className="btn" onClick={onRestart} disabled={busy}>
        Restart
      </button>

      <div className="control">
        <label>
          Cost <span className="val">{config.cost_bps.toFixed(1)} bps</span>
        </label>
        <input
          type="range"
          min={0}
          max={20}
          step={0.5}
          value={config.cost_bps}
          onChange={(e) => onChange({ cost_bps: Number(e.target.value) })}
        />
      </div>

      <div className="control">
        <label>
          Vol target <span className="val">{(config.target_ann_vol * 100).toFixed(0)}%</span>
        </label>
        <input
          type="range"
          min={0.02}
          max={0.4}
          step={0.01}
          value={config.target_ann_vol}
          onChange={(e) => onChange({ target_ann_vol: Number(e.target.value) })}
        />
      </div>

      <div className="control">
        <label>
          Trade budget <span className="val">{config.budget}/day</span>
        </label>
        <input
          type="range"
          min={0}
          max={12}
          step={1}
          value={config.budget}
          onChange={(e) => onChange({ budget: Number(e.target.value) })}
        />
      </div>

      <div className="control">
        <label>
          Speed <span className="val">{speed}x</span>
        </label>
        <input
          type="range"
          min={1}
          max={20}
          step={1}
          value={speed}
          onChange={(e) => onSpeed(Number(e.target.value))}
        />
      </div>

      <div className="control">
        <label>Regime tier</label>
        <select
          value={config.tier}
          onChange={(e) => onChange({ tier: Number(e.target.value) as 1 | 2 | 3 })}
        >
          <option value={1}>1 — vol sizing (supported)</option>
          <option value={2}>2 — trend/chop weights (unproven)</option>
          <option value={3} disabled={!config.allow_tier3}>
            3 — 12 regimes (risky)
          </option>
        </select>
      </div>

      <div className="control" style={{ minWidth: 0 }}>
        <label style={{ whiteSpace: 'nowrap' }}>Allow tier 3</label>
        <input
          type="checkbox"
          checked={config.allow_tier3}
          style={{ accentColor: 'var(--control)', width: 14, height: 14 }}
          onChange={(e) =>
            onChange({
              allow_tier3: e.target.checked,
              ...(e.target.checked ? {} : config.tier === 3 ? { tier: 2 as const } : {}),
            })
          }
        />
      </div>

      <div className="control">
        <label>Market</label>
        <select
          value={config.generator}
          onChange={(e) => onChange({ generator: e.target.value as SessionConfig['generator'] })}
        >
          <option value="regime_switching">regime switching</option>
          <option value="gbm">GBM (null)</option>
          <option value="ornstein_uhlenbeck">Ornstein-Uhlenbeck</option>
          <option value="garch_t">GARCH-t</option>
        </select>
      </div>

      {config.tier === 3 && (
        <p className="tier3-warning">
          Tier 3 splits the sample into 12 regimes. Each strategy weight is then estimated on a
          few hundred bars, and the luck threshold rises with the hypothesis count. Treat every
          number on screen as decoration.
        </p>
      )}
      {config.generator === 'gbm' && (
        <p className="tier3-warning">
          GBM has independent increments. The correct result here is a gross Sharpe of zero and a
          net Sharpe of minus the cost. Anything else is a bug.
        </p>
      )}
    </div>
  )
}
