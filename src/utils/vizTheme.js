/**
 * Dark-theme visual constants for D3.js charts.
 * Import in any component that builds SVG charts.
 *
 * Usage:
 *   import { VIZ, brightenScale, darkColors } from '../utils/vizTheme'
 *   svg.selectAll('.tick text').attr('fill', VIZ.axis)
 */
export const VIZ = {
  grid:      '#252840',
  axis:      '#5e6388',
  text:      '#9296b0',
  textEm:    '#e2e4f0',
  bg:        '#181b26',
  tooltipBg: '#1f2235',
}

/** Category color palette — high saturation, dark-bg optimized */
export const darkColors = [
  '#6366f1', '#34d399', '#fbbf24', '#f87171', '#a78bfa',
  '#60a5fa', '#f472b6', '#fb923c', '#2dd4bf', '#e2e4f0',
]

/**
 * Shift a sequential interpolator's domain so the dark-end doesn't
 * disappear into a dark background.
 *
 * Usage:
 *   const hot = d3.scaleSequential(brightenScale(d3.interpolateOrRd)).domain([0, 1])
 */
export function brightenScale(interpolator, min = 0.12) {
  return (t) => interpolator(min + (1 - min) * Math.max(0, Math.min(1, t)))
}

/**
 * Post-process a D3 axis group to use dark-theme colours.
 *
 * Usage:
 *   svg.append('g').call(d3.axisBottom(x)).call(applyDarkAxis)
 */
export function applyDarkAxis(axisGroup) {
  axisGroup.selectAll('.domain, .tick line').attr('stroke', VIZ.grid)
  axisGroup.selectAll('.tick text').attr('fill', VIZ.axis).style('font-size', '11px')
}
