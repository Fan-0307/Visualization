<template>
  <div class="sc-root">
    <!-- Header -->
    <div class="sc-header card">
      <div class="sc-header-top">
        <div>
          <h2 class="sc-title">Attention Sink 浓度与幻觉相关性</h2>
          <p class="sc-subtitle">
            假设：当模型对某张图的注意力高度集中在极少数 patch（Attention Sink 浓度异常偏高）时，
            它实际上没有有效利用图像内容，更容易产生错误答案（幻觉）。
          </p>
        </div>
        <div class="sc-sink-formula">
          <div class="formula-label">Sink Score</div>
          <div class="formula-body">
            <span class="formula-num">Σ top-1% patches</span>
            <div class="formula-divider"></div>
            <span class="formula-den">Σ all patches</span>
          </div>
          <div class="formula-note">均值跨最后 1/3 层、所有 token</div>
        </div>
      </div>

      <!-- Global finding -->
      <div class="sc-finding-bar" v-if="globalFinding">
        <span class="finding-icon">{{ globalFinding.supports ? '🔍' : '❌' }}</span>
        <span class="finding-text">{{ globalFinding.text }}</span>
        <span class="finding-n">(n={{ globalFinding.n }})</span>
      </div>
    </div>

    <!-- Per-model cards -->
    <div class="sc-model-grid">
      <div v-for="m in modelStats" :key="m.model" class="sc-model-card card">
        <div class="sc-model-header">
          <span class="sc-model-name">{{ m.model }}</span>
          <span class="sc-model-badge" :class="m.supports ? 'badge-yes' : 'badge-no'">
            {{ m.supports ? '✓ 假设成立' : '✗ 假设不成立' }}
          </span>
        </div>

        <!-- Stats row -->
        <div class="sc-stats-row">
          <div class="sc-stat">
            <div class="sc-stat-label">错误样本 Sink 均值</div>
            <div class="sc-stat-val err">{{ fmt(m.meanWrong) }}</div>
          </div>
          <div class="sc-arrow">→</div>
          <div class="sc-stat">
            <div class="sc-stat-label">正确样本 Sink 均值</div>
            <div class="sc-stat-val ok">{{ fmt(m.meanRight) }}</div>
          </div>
          <div class="sc-stat sc-stat-delta">
            <div class="sc-stat-label">差值 Δ</div>
            <div class="sc-stat-val" :class="m.delta > 0 ? 'err' : 'ok'">
              {{ m.delta > 0 ? '+' : '' }}{{ fmt(m.delta) }}
            </div>
          </div>
          <div class="sc-stat sc-stat-corr">
            <div class="sc-stat-label">点二列相关 r</div>
            <div class="sc-stat-val corr">{{ fmtCorr(m.corr) }}</div>
          </div>
          <div class="sc-stat">
            <div class="sc-stat-label">样本数 (✓/✗)</div>
            <div class="sc-stat-val mono">{{ m.nRight }}/{{ m.nWrong }}</div>
          </div>
        </div>

        <!-- Strip chart -->
        <svg :ref="el => svgRefs[m.model] = el" class="sc-strip-svg"></svg>

        <!-- Caveat if n is small -->
        <div v-if="m.nRight < 5 || m.nWrong < 5" class="sc-caveat">
          ⚠ 样本量偏小（正确 {{ m.nRight }}，错误 {{ m.nWrong }}），结论需谨慎
        </div>
      </div>
    </div>

    <!-- All-model combined scatter -->
    <div class="sc-combined card">
      <div class="sc-section-title">全模型样本分布（Sink Score vs. 正确性）</div>
      <svg ref="combinedSvg" class="sc-combined-svg"></svg>
      <div class="sc-combined-legend">
        <span class="leg-dot ok"></span><span>正确</span>
        <span class="leg-dot err"></span><span>错误</span>
        <span class="leg-model" v-for="m in modelColors" :key="m.model"
              :style="{ borderColor: m.color }">{{ m.model }}</span>
      </div>
    </div>

    <!-- Discussion -->
    <div class="sc-discussion card">
      <div class="sc-section-title">方法局限性与解读</div>
      <ul class="sc-disc-list">
        <li><strong>相关 ≠ 因果：</strong>Sink Score 高可能是图像内容稀少（纯色背景、遮挡）导致，而非模型能力问题。</li>
        <li><strong>VQA 正确率是代理指标：</strong>Sink 高但答案正确（知识驱动），或 Sink 低但答案错误（其它失败模式），均可能存在。</li>
        <li><strong>BLIP 的 Sink 性质不同：</strong>BLIP 用显式 cross-attention，CLS token 是结构性 Sink，与 decoder-only 的 Sink 机制不可直接比较。</li>
        <li><strong>如果相关性显著（|r| > 0.3）：</strong>Sink Score 可作为无需 ground-truth 的推理时幻觉先验信号，用于自我纠错（Self-Correction）或 uncertainty 估计。</li>
      </ul>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, nextTick } from 'vue'
import * as d3 from 'd3'
import attnIndex from '../data/attn_index.json'

const samples = attnIndex.samples || []
const svgRefs = ref({})
const combinedSvg = ref(null)

// ── Point-biserial correlation ────────────────────────────────────────
// r_pb = (M1 - M0) / S_total * sqrt(n1 * n0 / n^2)
// M1 = mean score for correct (1), M0 = mean score for incorrect (0)
// Negative r means higher sink → more errors (supports hypothesis)
function pointBiserial(scores, labels) {
  const n = scores.length
  if (n < 3) return 0
  const n1 = labels.filter(l => l === 1).length
  const n0 = n - n1
  if (n1 === 0 || n0 === 0) return 0
  const mean = scores.reduce((a, b) => a + b, 0) / n
  const variance = scores.reduce((a, b) => a + (b - mean) ** 2, 0) / n
  const std = Math.sqrt(variance) || 1e-12
  const mean1 = scores.filter((_, i) => labels[i] === 1).reduce((a, b) => a + b, 0) / n1
  const mean0 = scores.filter((_, i) => labels[i] === 0).reduce((a, b) => a + b, 0) / n0
  return (mean1 - mean0) / std * Math.sqrt((n1 * n0) / (n * n))
}

// ── Per-model stats ───────────────────────────────────────────────────
const modelStats = computed(() => {
  const byModel = {}
  for (const s of samples) {
    if (!byModel[s.model]) byModel[s.model] = []
    byModel[s.model].push(s)
  }
  return Object.entries(byModel).map(([model, items]) => {
    const hasSink = items.some(s => s.sink_score != null)
    if (!hasSink) return null
    const right = items.filter(s => s.correct === 1)
    const wrong = items.filter(s => s.correct === 0)
    const meanRight = right.length ? right.reduce((a, b) => a + (b.sink_score || 0), 0) / right.length : 0
    const meanWrong = wrong.length ? wrong.reduce((a, b) => a + (b.sink_score || 0), 0) / wrong.length : 0
    const scores = items.map(s => s.sink_score || 0)
    const labels = items.map(s => s.correct)
    const corr = pointBiserial(scores, labels)
    const delta = meanWrong - meanRight  // positive = wrong has higher sink
    return {
      model,
      items,
      meanRight,
      meanWrong,
      delta,
      corr,
      nRight: right.length,
      nWrong: wrong.length,
      // hypothesis supported if wrong > right sink (delta > 0)
      supports: delta > 0.002,
    }
  }).filter(Boolean)
})

const globalFinding = computed(() => {
  const all = modelStats.value
  if (!all.length) return null
  const n = samples.filter(s => s.sink_score != null).length
  const supporting = all.filter(m => m.supports).length
  const scores = samples.filter(s => s.sink_score != null).map(s => s.sink_score || 0)
  const labels = samples.filter(s => s.sink_score != null).map(s => s.correct)
  const globalCorr = pointBiserial(scores, labels)
  const supports = supporting >= Math.ceil(all.length / 2)
  return {
    n,
    supports,
    corr: globalCorr,
    text: supports
      ? `在 ${supporting}/${all.length} 个模型中，错误答案的 Sink 浓度高于正确答案（全局 r = ${fmtCorr(globalCorr)}）`
      : `假设在多数模型中不成立，或样本量不足（全局 r = ${fmtCorr(globalCorr)}）——数据量更大时结论可能改变`,
  }
})

const MODEL_COLORS = ['#6366f1', '#f59e0b', '#10b981', '#f43f5e', '#06b6d4']
const modelColors = computed(() =>
  modelStats.value.map((m, i) => ({ model: m.model, color: MODEL_COLORS[i % MODEL_COLORS.length] }))
)

// ── Formatting ────────────────────────────────────────────────────────
function fmt(v) {
  if (v == null || isNaN(v)) return '—'
  return v.toFixed(4)
}
function fmtCorr(v) {
  if (v == null || isNaN(v)) return '—'
  return (v >= 0 ? '+' : '') + v.toFixed(3)
}

// ── Draw strip charts ─────────────────────────────────────────────────
function drawStrip(model, svgEl) {
  if (!svgEl) return
  const stat = modelStats.value.find(m => m.model === model)
  if (!stat) return

  d3.select(svgEl).selectAll('*').remove()

  const W = svgEl.clientWidth || 600
  const H = 80
  const pad = { left: 60, right: 20, top: 14, bottom: 24 }
  const iw = W - pad.left - pad.right

  const allScores = stat.items.map(s => s.sink_score || 0)
  const xScale = d3.scaleLinear()
    .domain([0, Math.max(d3.max(allScores) * 1.1, 0.5)])
    .range([0, iw])

  const svg = d3.select(svgEl)
    .attr('width', W).attr('height', H)
  const g = svg.append('g').attr('transform', `translate(${pad.left},${pad.top})`)

  // X axis
  g.append('g')
    .attr('transform', `translate(0,${H - pad.top - pad.bottom})`)
    .call(d3.axisBottom(xScale).ticks(5).tickFormat(d3.format('.2f')))
    .call(ax => {
      ax.select('.domain').attr('stroke', '#353a54')
      ax.selectAll('.tick line').attr('stroke', '#353a54')
      ax.selectAll('.tick text').attr('fill', '#9296b0').attr('font-size', 10)
    })

  // Labels
  const yRight = 18, yWrong = 36
  svg.append('text').attr('x', 8).attr('y', pad.top + yRight + 4)
    .attr('fill', '#34d399').attr('font-size', 11).attr('font-weight', 600).text('✓ 正确')
  svg.append('text').attr('x', 8).attr('y', pad.top + yWrong + 4)
    .attr('fill', '#f87171').attr('font-size', 11).attr('font-weight', 600).text('✗ 错误')

  // Mean lines
  const meanRightX = xScale(stat.meanRight)
  const meanWrongX = xScale(stat.meanWrong)
  g.append('line').attr('x1', meanRightX).attr('x2', meanRightX)
    .attr('y1', 0).attr('y2', yRight + 8)
    .attr('stroke', '#34d399').attr('stroke-width', 2).attr('stroke-dasharray', '4,2')
  g.append('line').attr('x1', meanWrongX).attr('x2', meanWrongX)
    .attr('y1', yWrong - 8).attr('y2', yWrong + 8)
    .attr('stroke', '#f87171').attr('stroke-width', 2).attr('stroke-dasharray', '4,2')

  // Dots with jitter
  const rng = d3.randomNormal(0, 3)
  for (const s of stat.items) {
    const x = xScale(s.sink_score || 0)
    const isRight = s.correct === 1
    const y = isRight ? yRight : yWrong
    g.append('circle')
      .attr('cx', x).attr('cy', y + rng())
      .attr('r', 5)
      .attr('fill', isRight ? '#34d399' : '#f87171')
      .attr('fill-opacity', 0.7)
      .attr('stroke', isRight ? '#10b981' : '#ef4444')
      .attr('stroke-width', 1)
  }
}

// ── Draw combined scatter ─────────────────────────────────────────────
function drawCombined() {
  const svgEl = combinedSvg.value
  if (!svgEl) return

  d3.select(svgEl).selectAll('*').remove()

  const W = svgEl.clientWidth || 700
  const H = 220
  const pad = { left: 50, right: 20, top: 14, bottom: 34 }
  const iw = W - pad.left - pad.right
  const ih = H - pad.top - pad.bottom

  const allItems = samples.filter(s => s.sink_score != null)
  const allScores = allItems.map(s => s.sink_score)
  const xScale = d3.scaleLinear()
    .domain([0, Math.max(d3.max(allScores) * 1.05, 0.3)])
    .range([0, iw])
  const yScale = d3.scaleLinear().domain([-0.3, 1.3]).range([ih, 0])

  const colorByModel = {}
  modelStats.value.forEach((m, i) => { colorByModel[m.model] = MODEL_COLORS[i % MODEL_COLORS.length] })

  const svg = d3.select(svgEl).attr('width', W).attr('height', H)
  const g = svg.append('g').attr('transform', `translate(${pad.left},${pad.top})`)

  // Grid
  g.append('line').attr('x1', 0).attr('x2', iw).attr('y1', yScale(0)).attr('y2', yScale(0))
    .attr('stroke', '#f87171').attr('stroke-width', 0.5).attr('stroke-dasharray', '4,3').attr('opacity', 0.4)
  g.append('line').attr('x1', 0).attr('x2', iw).attr('y1', yScale(1)).attr('y2', yScale(1))
    .attr('stroke', '#34d399').attr('stroke-width', 0.5).attr('stroke-dasharray', '4,3').attr('opacity', 0.4)

  // Y axis
  g.append('g').call(
    d3.axisLeft(yScale).tickValues([0, 1]).tickFormat(d => d === 0 ? '错误' : '正确')
  ).call(ax => {
    ax.select('.domain').remove()
    ax.selectAll('.tick line').remove()
    ax.selectAll('.tick text').attr('fill', '#9296b0').attr('font-size', 11)
  })

  // X axis
  g.append('g').attr('transform', `translate(0,${ih})`).call(
    d3.axisBottom(xScale).ticks(6).tickFormat(d3.format('.2f'))
  ).call(ax => {
    ax.select('.domain').attr('stroke', '#353a54')
    ax.selectAll('.tick line').attr('stroke', '#353a54')
    ax.selectAll('.tick text').attr('fill', '#9296b0').attr('font-size', 10)
  })
  g.append('text').attr('x', iw / 2).attr('y', ih + 28)
    .attr('text-anchor', 'middle').attr('fill', '#5e6388').attr('font-size', 11)
    .text('Sink Score (top-1% patch 注意力占比)')

  // Regression line per model
  for (const m of modelStats.value) {
    const pts = m.items.filter(s => s.sink_score != null)
    if (pts.length < 3) continue
    const xs = pts.map(s => s.sink_score)
    const ys = pts.map(s => s.correct)
    const xm = xs.reduce((a, b) => a + b, 0) / xs.length
    const ym = ys.reduce((a, b) => a + b, 0) / ys.length
    const num = xs.reduce((a, x, i) => a + (x - xm) * (ys[i] - ym), 0)
    const den = xs.reduce((a, x) => a + (x - xm) ** 2, 0)
    if (Math.abs(den) < 1e-12) continue
    const slope = num / den
    const intercept = ym - slope * xm
    const x0 = xScale.domain()[0], x1 = xScale.domain()[1]
    g.append('line')
      .attr('x1', xScale(x0)).attr('x2', xScale(x1))
      .attr('y1', yScale(slope * x0 + intercept)).attr('y2', yScale(slope * x1 + intercept))
      .attr('stroke', colorByModel[m.model] || '#6366f1')
      .attr('stroke-width', 1.5).attr('stroke-dasharray', '6,3').attr('opacity', 0.5)
  }

  // Dots with jitter
  const jitter = d3.randomNormal(0, 0.05)
  for (const s of allItems) {
    const x = xScale(s.sink_score)
    const y = yScale(Math.max(-0.25, Math.min(1.25, (s.correct || 0) + jitter())))
    const color = colorByModel[s.model] || '#6366f1'
    g.append('circle')
      .attr('cx', x).attr('cy', y)
      .attr('r', 5)
      .attr('fill', s.correct ? '#34d399' : '#f87171')
      .attr('fill-opacity', 0.65)
      .attr('stroke', color).attr('stroke-width', 1.5)
  }
}

// ── Lifecycle ─────────────────────────────────────────────────────────
onMounted(() => {
  nextTick(() => {
    for (const m of modelStats.value) {
      drawStrip(m.model, svgRefs.value[m.model])
    }
    drawCombined()
  })
})
</script>

<style scoped>
.sc-root {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

/* Header */
.sc-header-top {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 24px;
  flex-wrap: wrap;
}
.sc-title {
  font-size: 16px;
  font-weight: 700;
  color: var(--text-primary);
  margin-bottom: 6px;
}
.sc-subtitle {
  font-size: 13px;
  color: var(--text-secondary);
  line-height: 1.6;
  max-width: 640px;
}

/* Formula box */
.sc-sink-formula {
  background: var(--bg-elevated);
  border: 1px solid var(--border-default);
  border-radius: var(--radius-md);
  padding: 10px 16px;
  text-align: center;
  flex-shrink: 0;
}
.formula-label {
  font-size: 11px;
  font-weight: 600;
  color: var(--text-tertiary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 6px;
}
.formula-body {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
}
.formula-num, .formula-den {
  font-size: 12px;
  color: var(--text-primary);
  font-family: var(--font-mono);
}
.formula-divider {
  width: 120px;
  height: 1px;
  background: var(--border-default);
  margin: 3px 0;
}
.formula-note {
  font-size: 10px;
  color: var(--text-tertiary);
  margin-top: 4px;
}

/* Finding bar */
.sc-finding-bar {
  margin-top: 14px;
  padding: 10px 14px;
  background: var(--bg-elevated);
  border-radius: var(--radius-md);
  border-left: 3px solid var(--accent-primary);
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
  color: var(--text-primary);
  line-height: 1.5;
}
.finding-icon { font-size: 16px; }
.finding-text { flex: 1; }
.finding-n { color: var(--text-tertiary); font-family: var(--font-mono); font-size: 11px; }

/* Model grid */
.sc-model-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(520px, 1fr));
  gap: 16px;
}

.sc-model-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;
}
.sc-model-name {
  font-size: 14px;
  font-weight: 700;
  color: var(--text-primary);
  font-family: var(--font-mono);
}
.sc-model-badge {
  padding: 2px 10px;
  border-radius: 10px;
  font-size: 11px;
  font-weight: 600;
}
.badge-yes { background: var(--color-correct-bg); color: var(--color-correct); }
.badge-no  { background: var(--color-error-bg);   color: var(--color-error); }

/* Stats row */
.sc-stats-row {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
  margin-bottom: 12px;
  padding: 10px 12px;
  background: var(--bg-elevated);
  border-radius: var(--radius-md);
}
.sc-stat {
  display: flex;
  flex-direction: column;
  gap: 3px;
}
.sc-stat-label {
  font-size: 10px;
  color: var(--text-tertiary);
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.4px;
  white-space: nowrap;
}
.sc-stat-val {
  font-size: 16px;
  font-weight: 700;
  font-family: var(--font-mono);
  font-variant-numeric: tabular-nums;
}
.sc-stat-val.ok   { color: var(--color-correct); }
.sc-stat-val.err  { color: var(--color-error); }
.sc-stat-val.corr { color: var(--accent-primary); }
.sc-stat-val.mono { color: var(--text-primary); font-size: 13px; }
.sc-arrow { color: var(--text-tertiary); font-size: 16px; margin: 0 4px; padding-top: 14px; }

/* Strip SVG */
.sc-strip-svg {
  width: 100%;
  height: 80px;
  display: block;
}

/* Caveat */
.sc-caveat {
  margin-top: 8px;
  font-size: 11px;
  color: var(--color-warning);
  opacity: 0.8;
}

/* Combined */
.sc-section-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--text-secondary);
  margin-bottom: 12px;
  text-transform: uppercase;
  letter-spacing: 0.4px;
}
.sc-combined-svg {
  width: 100%;
  height: 220px;
  display: block;
}
.sc-combined-legend {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-top: 10px;
  flex-wrap: wrap;
  font-size: 12px;
  color: var(--text-secondary);
}
.leg-dot {
  width: 10px; height: 10px; border-radius: 50%; display: inline-block;
}
.leg-dot.ok  { background: #34d399; }
.leg-dot.err { background: #f87171; }
.leg-model {
  padding: 1px 8px;
  border-radius: 10px;
  border: 1.5px solid currentColor;
  font-size: 11px;
  font-family: var(--font-mono);
}

/* Discussion */
.sc-disc-list {
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.sc-disc-list li {
  font-size: 13px;
  color: var(--text-secondary);
  line-height: 1.6;
  padding-left: 14px;
  position: relative;
}
.sc-disc-list li::before {
  content: '·';
  position: absolute;
  left: 0;
  color: var(--accent-primary);
  font-weight: 700;
}
.sc-disc-list strong {
  color: var(--text-primary);
}
</style>
