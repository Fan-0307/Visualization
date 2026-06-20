// Precompute system-level attention-diffusion statistics from layer_evo result.json files.
// Output: src/data/layer_evo_stats.json  (a few KB; avoids shipping/parsing the 259MB raw grids)
//
// For each model, aggregates four per-layer metrics across ALL samples, split by correctness:
//   entropy (0=concentrated, 1=diffuse), concentration (1-entropy), sparsity (Gini), top5 mass.
// The metric formulas are copied verbatim from LayerEvolution.vue's compute() so the
// fingerprint view and Tab6 report identical numbers.
//
// Usage:  node data/compute_diffusion_stats.js

const fs = require('fs')
const path = require('path')

const ROOT = path.join(__dirname, '..', 'public', 'data', 'layer_evo')
const OUT = path.join(__dirname, '..', 'src', 'data', 'layer_evo_stats.json')

const DISPLAY = { qwen: 'Qwen2-VL', llava: 'LLaVA', blip: 'BLIP2', clip: 'CLIP' }
const MODEL_ORDER = ['qwen', 'llava', 'blip', 'clip']
const METRICS = ['entropy', 'concentration', 'sparsity', 'top5']

// Verbatim from LayerEvolution.vue:658-670
function compute(grid, metric) {
  const g = grid
  if (!g?.length) return 0
  const flat = g.flat(); const total = flat.reduce((a, b) => a + b, 0) || 1
  const norm = flat.map(v => v / total).sort((a, b) => b - a); const n = norm.length
  switch (metric) {
    case 'concentration': { const ent = -norm.reduce((s, v) => s + (v > 0 ? v * Math.log(v) : 0), 0); return 1 - ent / Math.log(n) }
    case 'entropy': return -norm.reduce((s, v) => s + (v > 0 ? v * Math.log(v) : 0), 0) / Math.log(n)
    case 'sparsity': { let cum = 0, gini = 0; for (let i = 0; i < n; i++) { gini += (i + 1) * norm[i]; cum += norm[i] } return cum > 0 ? (2 * gini / (n * cum) - (n + 1) / n) : 0 }
    case 'top5': { const k = Math.max(1, Math.ceil(n * 0.05)); return norm.slice(0, k).reduce((a, b) => a + b, 0) }
    default: return 0
  }
}

const mean = a => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : null)

const out = { metric_note: '0=concentrated/focused, 1=diffuse (for entropy)', models: {} }

for (const m of MODEL_ORDER) {
  const dir = path.join(ROOT, m)
  if (!fs.existsSync(dir)) { console.warn('skip (no dir):', m); continue }

  // acc[layerIndex][metric] = { all:[], correct:[], wrong:[] }; comp[layerIndex] = {component: count}
  const acc = {}
  const comp = {}
  let sampleCount = 0

  for (const sid of fs.readdirSync(dir)) {
    const fp = path.join(dir, sid, 'result.json')
    if (!fs.existsSync(fp)) continue
    let j
    try { j = JSON.parse(fs.readFileSync(fp, 'utf8')) } catch (e) { continue }
    const layers = j.layers
    if (!layers?.length) continue
    const correct = j.correct === 1 ? 'correct' : (j.correct === 0 ? 'wrong' : null)
    sampleCount++

    layers.forEach((l, i) => {
      if (!acc[i]) { acc[i] = {}; METRICS.forEach(k => acc[i][k] = { all: [], correct: [], wrong: [] }) }
      if (!comp[i]) comp[i] = {}
      const c = l.component || 'unknown'
      comp[i][c] = (comp[i][c] || 0) + 1
      for (const k of METRICS) {
        const v = compute(l.raw_visual_grid, k)
        if (v == null || Number.isNaN(v)) continue
        acc[i][k].all.push(v)
        if (correct) acc[i][k][correct].push(v)
      }
    })
  }

  const layerIdxs = Object.keys(acc).map(Number).sort((a, b) => a - b)
  const layers = layerIdxs.map(i => {
    const majorityComp = Object.entries(comp[i]).sort((a, b) => b[1] - a[1])[0][0]
    const row = { i, component: majorityComp, n: acc[i].entropy.all.length }
    for (const k of METRICS) {
      row[k] = round(mean(acc[i][k].all))
      row[k + '_correct'] = round(mean(acc[i][k].correct))
      row[k + '_wrong'] = round(mean(acc[i][k].wrong))
    }
    return row
  })

  // Component boundaries (where the majority component changes between adjacent layers)
  const boundaries = []
  for (let i = 1; i < layers.length; i++) {
    if (layers[i].component !== layers[i - 1].component) {
      boundaries.push({ at: layers[i].i, from: layers[i - 1].component, to: layers[i].component })
    }
  }

  out.models[m] = {
    key: m,
    display_name: DISPLAY[m] || m,
    sample_count: sampleCount,
    layer_count: layers.length,
    components: [...new Set(layers.map(l => l.component))],
    component_boundaries: boundaries,
    layers,
  }
  console.log(`${m}: ${sampleCount} samples, ${layers.length} layers, entropy ${layers[0]?.entropy}→${layers[layers.length - 1]?.entropy}`)
}

function round(v) { return v == null ? null : Math.round(v * 1e4) / 1e4 }

fs.writeFileSync(OUT, JSON.stringify(out))
console.log('wrote', OUT, '(' + (fs.statSync(OUT).size / 1024).toFixed(1) + ' KB)')
