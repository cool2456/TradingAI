import { useLayoutEffect, useMemo, useRef, useState } from 'react'
import type { Point, RegimeBand } from '../api'

/**
 * Palette mirror of styles.css.
 *
 * Series identity is a lightness ramp of one neutral hue. Hue is reserved for
 * P&L sign and is never used to tell one strategy from another, so these are
 * deliberately all the same colour at different brightnesses.
 */
export const SERIES_COLORS = ['#cfd8e6', '#a9b6c9', '#8794ab', '#6b788f', '#566274']
export const CONTROL_COLOR = '#7d7566'
export const CONTROL_COLOR_ALT = '#a09378'
export const PORTFOLIO_COLORS = ['#f2f5fa', '#8b93a3']

export type SeriesKind = 'strategy' | 'control' | 'portfolio' | 'price'

export interface Series {
  name: string
  points: Point[]
  kind: SeriesKind
  colorIndex?: number
}

export function seriesColor(kind: SeriesKind, index = 0): string {
  if (kind === 'control') return index === 0 ? CONTROL_COLOR : CONTROL_COLOR_ALT
  if (kind === 'portfolio') return PORTFOLIO_COLORS[index % PORTFOLIO_COLORS.length]
  return SERIES_COLORS[index % SERIES_COLORS.length]
}

/** Regimes are shaded by opacity, never by hue -- see the note in styles.css. */
const REGIME_ORDER = ['low_vol', 'normal_vol', 'high_vol', 'chop', 'trend']
const BAND_ALPHAS = [0.028, 0.062, 0.1, 0.14, 0.185]

function bandFills(bands: RegimeBand[]): Map<string, string> {
  const names = Array.from(new Set(bands.map((b) => b.regime)))
  names.sort((a, b) => {
    const ia = REGIME_ORDER.indexOf(a)
    const ib = REGIME_ORDER.indexOf(b)
    if (ia !== -1 && ib !== -1) return ia - ib
    if (ia !== -1) return -1
    if (ib !== -1) return 1
    return a.localeCompare(b)
  })
  const map = new Map<string, string>()
  names.forEach((name, i) => {
    map.set(name, `rgba(207, 216, 230, ${BAND_ALPHAS[i % BAND_ALPHAS.length]})`)
  })
  return map
}

function useWidth<T extends HTMLElement>() {
  const ref = useRef<T | null>(null)
  const [width, setWidth] = useState(720)
  useLayoutEffect(() => {
    const node = ref.current
    if (!node) return
    const observer = new ResizeObserver((entries) => {
      const next = entries[0]?.contentRect.width
      if (next && next > 0) setWidth(next)
    })
    observer.observe(node)
    return () => observer.disconnect()
  }, [])
  return { ref, width }
}

interface Props {
  series: Series[]
  bands?: RegimeBand[]
  height?: number
  yFormat?: (v: number) => string
  zeroLine?: boolean
  showBandLabels?: boolean
}

const MARGIN = { top: 8, right: 8, bottom: 20, left: 52 }

export function LineChart({
  series,
  bands = [],
  height = 220,
  yFormat = (v) => v.toFixed(2),
  zeroLine = false,
  showBandLabels = false,
}: Props) {
  const { ref, width } = useWidth<HTMLDivElement>()
  const [hoverX, setHoverX] = useState<number | null>(null)

  const model = useMemo(() => {
    const all = series.flatMap((s) =>
      s.points
        .filter((p) => p.v != null && isFinite(p.v))
        .map((p) => ({ x: Date.parse(p.t), y: p.v as number })),
    )
    if (all.length === 0) return null

    let xMin = Infinity
    let xMax = -Infinity
    let yMin = Infinity
    let yMax = -Infinity
    for (const p of all) {
      if (p.x < xMin) xMin = p.x
      if (p.x > xMax) xMax = p.x
      if (p.y < yMin) yMin = p.y
      if (p.y > yMax) yMax = p.y
    }
    if (zeroLine) {
      yMin = Math.min(yMin, 0)
      yMax = Math.max(yMax, 0)
    }
    const pad = (yMax - yMin || Math.abs(yMax) || 1) * 0.08
    yMin -= pad
    yMax += pad
    if (xMax === xMin) xMax = xMin + 1
    return { xMin, xMax, yMin, yMax }
  }, [series, zeroLine])

  if (!model) {
    return (
      <div ref={ref}>
        <div className="empty">no data yet</div>
      </div>
    )
  }

  const innerW = Math.max(width - MARGIN.left - MARGIN.right, 10)
  const innerH = height - MARGIN.top - MARGIN.bottom
  const sx = (t: number) => ((t - model.xMin) / (model.xMax - model.xMin)) * innerW
  const sy = (v: number) => innerH - ((v - model.yMin) / (model.yMax - model.yMin)) * innerH

  const yTicks = 4
  const ticks = Array.from({ length: yTicks + 1 }, (_, i) => model.yMin + ((model.yMax - model.yMin) * i) / yTicks)
  const fills = bandFills(bands)

  const path = (points: Point[]) => {
    let d = ''
    let pen = false
    for (const p of points) {
      if (p.v == null || !isFinite(p.v)) {
        pen = false
        continue
      }
      const x = sx(Date.parse(p.t))
      const y = sy(p.v)
      d += `${pen ? 'L' : 'M'}${x.toFixed(2)},${y.toFixed(2)}`
      pen = true
    }
    return d
  }

  // Nearest-point lookup for the hover readout, using the longest series as the
  // reference axis.
  const reference = series.reduce((best, s) => (s.points.length > best.points.length ? s : best), series[0])
  let hoverIndex: number | null = null
  if (hoverX != null && reference && reference.points.length) {
    const target = model.xMin + ((hoverX - MARGIN.left) / innerW) * (model.xMax - model.xMin)
    let bestDist = Infinity
    reference.points.forEach((p, i) => {
      const dist = Math.abs(Date.parse(p.t) - target)
      if (dist < bestDist) {
        bestDist = dist
        hoverIndex = i
      }
    })
  }
  const hoverTime = hoverIndex != null ? Date.parse(reference.points[hoverIndex].t) : null

  return (
    <div ref={ref}>
      <svg
        className="chart"
        width="100%"
        height={height}
        onMouseMove={(e) => setHoverX(e.nativeEvent.offsetX)}
        onMouseLeave={() => setHoverX(null)}
      >
        <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
          {bands.map((band, i) => {
            const x0 = sx(Date.parse(band.start))
            const x1 = sx(Date.parse(band.end))
            const w = Math.max(x1 - x0, 0.5)
            return (
              <g key={`${band.start}-${i}`}>
                <rect x={x0} y={0} width={w} height={innerH} fill={fills.get(band.regime)} />
                {showBandLabels && w > 42 && (
                  <text className="band-label" x={x0 + 4} y={11}>
                    {band.regime.replace('_', ' ')}
                  </text>
                )}
              </g>
            )
          })}

          {ticks.map((t) => (
            <g key={t}>
              <line className="grid-line" x1={0} x2={innerW} y1={sy(t)} y2={sy(t)} />
              <text className="axis-text" x={-7} y={sy(t) + 3} textAnchor="end">
                {yFormat(t)}
              </text>
            </g>
          ))}

          {zeroLine && model.yMin < 0 && model.yMax > 0 && (
            <line className="zero-line" x1={0} x2={innerW} y1={sy(0)} y2={sy(0)} />
          )}

          {series.map((s) => (
            <path
              key={s.name}
              className={`series ${s.kind}`}
              d={path(s.points)}
              stroke={seriesColor(s.kind, s.colorIndex ?? 0)}
            />
          ))}

          {hoverTime != null && (
            <line className="cursor-line" x1={sx(hoverTime)} x2={sx(hoverTime)} y1={0} y2={innerH} />
          )}

          <line className="axis-line" x1={0} x2={innerW} y1={innerH} y2={innerH} />
          <text className="axis-text" x={0} y={innerH + 14}>
            {new Date(model.xMin).toISOString().slice(0, 10)}
          </text>
          <text className="axis-text" x={innerW} y={innerH + 14} textAnchor="end">
            {new Date(model.xMax).toISOString().slice(0, 10)}
          </text>
          {hoverTime != null && (
            <text className="axis-text" x={sx(hoverTime)} y={innerH + 14} textAnchor="middle">
              {new Date(hoverTime).toISOString().slice(0, 10)}
            </text>
          )}
        </g>
      </svg>
    </div>
  )
}

interface LegendProps {
  series: Series[]
}

export function Legend({ series }: LegendProps) {
  return (
    <div className="legend">
      {series.map((s) => (
        <div key={s.name} className={`legend-item ${s.kind === 'control' ? 'control' : ''}`}>
          <span
            className="legend-swatch"
            style={{
              borderTopColor: seriesColor(s.kind, s.colorIndex ?? 0),
              borderTopStyle: s.kind === 'control' ? 'dashed' : 'solid',
              borderTopWidth: s.kind === 'portfolio' ? 3 : 2,
            }}
          />
          <span>{s.name}</span>
          {s.kind === 'control' && <span className="badge">CONTROL</span>}
        </div>
      ))}
    </div>
  )
}
