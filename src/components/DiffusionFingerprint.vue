<template>
  <div>
    <div class="card">
      <div class="section-title">注意力扩散指纹
        <span class="df-sub">每模型 · 每层 · 全样本平均（{{ metricDef.label }}）</span>
      </div>

      <!-- headline finding -->
      <div class="df-callout">
        <b>核心现象：</b>经过成堆 self-attention 后，视觉 token 感受野极大、注意力<b>高度扩散</b>——四个模型的视觉编码器归一化熵都在 0.92–0.98。
        唯一出现空间<b>聚焦</b>的是 <b>BLIP2 的 Q-Former 跨注意力瓶颈</b>（一路收敛到 0.42）。
        说明现代 VLM 的注意力扩散与否<b>由注意力机制决定，而非 encoder/decoder</b>。
      </div>

      <!-- metric toggle -->
      <div class="df-metrics">
        <button v-for="m in metricDefs" :key="m.key" @click="metric = m.key"
          :class="['df-mbtn', metric === m.key && 'df-mbtn-on']">{{ m.label }}</button>
        <span class="df-info">{{ metricDef.info }}</span>
      </div>

      <!-- heatmap overview -->
      <div class="df-heatmap"><svg ref="hmRef"></svg></div>
      <div class="df-legend">
        <span class="df-legend-label">聚焦</span>
        <div class="df-legend-bar"></div>
        <span class="df-legend-label">扩散</span>
        <span class="df-hint">点击模型行 = 在下方折线高亮；点击单元格 → 在「逐层演化」打开该模型</span>
      </div>
    </div>

    <!-- per-model line detail: ALL models overlaid, selected one emphasized -->
    <div class="card df-detail">
      <div class="section-title">
        逐层 {{ metricDef.label }} 演化 · 四模型对比（X 轴 = 归一化层深）
        <button class="df-open" @click="openInEvolution(selected)">在「逐层演化」中打开 {{ selectedDisplay }} →</button>
      </div>
      <div class="df-model-legend">
        <button v-for="m in models" :key="m" class="df-mleg"
          :class="{ on: selected===m }" @click="selected=m">
          <i class="sw" :style="{background: MODEL_COLORS[m]}"></i>{{ stats.models[m].display_name }}
        </button>
        <span class="df-comp-hint">点击模型 = 高亮并显示其正确/错误分线</span>
      </div>
      <div class="df-detail-note">{{ detailNote }}</div>
      <div class="df-linechart"><svg ref="lineRef"></svg></div>
      <div class="df-cvw-note">{{ correctWrongNote }}</div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch, nextTick, inject } from 'vue'
import * as d3 from 'd3'
import stats from '../data/layer_evo_stats.json'

const broadcast = inject('broadcast', () => {})
const navigate = inject('navigate', () => {})

const MODEL_ORDER = ['qwen', 'llava', 'blip', 'clip']
const models = MODEL_ORDER.filter(m => stats.models[m])
const MODEL_COLORS = { qwen: '#3b82f6', llava: '#10b981', blip: '#f59e0b', clip: '#ec4899' }

const metricDefs = [
  { key: 'entropy', label: 'Entropy', diffuseHigh: true, info: '归一化熵：接近 1 = 注意力均匀散布（扩散），接近 0 = 极度集中' },
  { key: 'top5', label: 'Top-5%', diffuseHigh: false, info: '最热的 5% patch 占据的注意力质量占比，越高越聚焦' },
]
const metric = ref('entropy')
const metricDef = computed(() => metricDefs.find(m => m.key === metric.value))

const selected = ref('blip')   // emphasized model (BLIP2 = the one that actually concentrates)
const selectedDisplay = computed(() => stats.models[selected.value]?.display_name || selected.value)

const COMP_LABEL = {
  vision_encoder: '视觉编码器', vision_tower: '视觉编码器',
  qformer_self_attention: 'Q-Former 自注意力', qformer_cross_attention: 'Q-Former 跨注意力',
  language_model: '语言模型', unknown: '视觉编码器(ViT)',
}
const compLabel = c => COMP_LABEL[c] || c

const hmRef = ref(null)
const lineRef = ref(null)

// metric domain across all models' layers (for a stable color scale)
const domain = computed(() => {
  const vals = []
  for (const m of models) for (const l of stats.models[m].layers) {
    const v = l[metric.value]; if (v != null) vals.push(v)
  }
  return d3.extent(vals)
})

function diffuseScore(v) {
  const [lo, hi] = domain.value
  const t = hi === lo ? 0.5 : (v - lo) / (hi - lo)
  return metricDef.value.diffuseHigh ? t : 1 - t
}
// diffuse → red, focused → blue
const colorFor = v => d3.interpolateRdYlBu(1 - diffuseScore(v))

function drawHeatmap() {
  const svg = d3.select(hmRef.value)
  svg.selectAll('*').remove()
  const ml = 96, mt = 10, mr = 16, mb = 26
  const rowH = 46, gap = 8
  const W = Math.max(640, (hmRef.value?.parentElement?.clientWidth || 720))
  const w = W - ml - mr
  const H = mt + mb + models.length * (rowH + gap)
  svg.attr('width', W).attr('height', H)
  const g = svg.append('g').attr('transform', `translate(${ml},${mt})`)

  const tip = d3.select('body').selectAll('.df-tip').data([0]).join('div')
    .attr('class', 'df-tip')
    .style('position', 'absolute').style('background', 'rgba(15,23,42,0.92)').style('color', '#fff')
    .style('padding', '7px 11px').style('border-radius', '8px').style('font-size', '12px')
    .style('line-height', '1.5').style('pointer-events', 'none').style('opacity', 0)

  models.forEach((m, ri) => {
    const M = stats.models[m]
    const L = M.layers
    const y0 = ri * (rowH + gap)
    const cellW = w / L.length

    // model label (clickable → expand)
    svg.append('text').attr('x', ml - 12).attr('y', mt + y0 + rowH / 2)
      .attr('text-anchor', 'end').attr('dominant-baseline', 'middle')
      .attr('font-size', 13).attr('font-weight', 700)
      .attr('fill', selected.value === m ? '#1d4ed8' : '#1e293b')
      .style('cursor', 'pointer').text(M.display_name)
      .on('click', () => { selected.value = m })

    // cells
    g.selectAll(`rect.r${ri}`).data(L).join('rect').attr('class', `r${ri}`)
      .attr('x', (_, i) => i * cellW).attr('y', y0)
      .attr('width', Math.max(1, cellW - 0.5)).attr('height', rowH)
      .attr('fill', d => colorFor(d[metric.value]))
      .style('cursor', 'pointer')
      .on('mouseover', function (e, d) {
        d3.select(this).attr('stroke', '#0f172a').attr('stroke-width', 1.2)
        tip.style('opacity', 1).html(
          `<b>${M.display_name}</b> · 层 ${d.i + 1}/${L.length}<br/>` +
          `组件：${compLabel(d.component)}<br/>` +
          `${metricDef.value.label}：<b>${d[metric.value]?.toFixed(3)}</b> · n=${d.n}<br/>` +
          `<span style="color:#93c5fd">▶ 点击 → 在「逐层演化」打开 ${M.display_name}</span>`)
          .style('left', (e.pageX + 12) + 'px').style('top', (e.pageY - 30) + 'px')
      })
      .on('mouseout', function () { d3.select(this).attr('stroke', 'none'); tip.style('opacity', 0) })
      .on('click', (_, d) => openInEvolution(m, d.i))

    // component boundary separators + labels
    M.component_boundaries.forEach(b => {
      const bx = b.at * cellW
      g.append('line').attr('x1', bx).attr('x2', bx).attr('y1', y0).attr('y2', y0 + rowH)
        .attr('stroke', '#fff').attr('stroke-width', 2)
    })
    // component name band under each row
    const segs = []
    let start = 0
    for (let i = 1; i <= L.length; i++) {
      if (i === L.length || L[i].component !== L[start].component) {
        segs.push({ c: L[start].component, x0: start, x1: i }); start = i
      }
    }
    segs.forEach(s => {
      const cx = ((s.x0 + s.x1) / 2) * cellW
      if ((s.x1 - s.x0) * cellW < 34) return
      g.append('text').attr('x', cx).attr('y', y0 + rowH - 5).attr('text-anchor', 'middle')
        .attr('font-size', 9).attr('fill', 'rgba(255,255,255,0.85)').text(compLabel(s.c))
    })

    // highlight expanded row
    if (selected.value === m) {
      g.append('rect').attr('x', -2).attr('y', y0 - 2).attr('width', w + 4).attr('height', rowH + 4)
        .attr('fill', 'none').attr('stroke', '#1d4ed8').attr('stroke-width', 2).attr('rx', 3)
    }
  })

  // depth axis label
  svg.append('text').attr('x', ml).attr('y', H - 8).attr('font-size', 11).attr('fill', '#94a3b8').text('浅层')
  svg.append('text').attr('x', W - mr).attr('y', H - 8).attr('text-anchor', 'end').attr('font-size', 11).attr('fill', '#94a3b8').text('深层 →')
}

function drawLine() {
  if (!lineRef.value) return
  const svg = d3.select(lineRef.value)
  svg.selectAll('*').remove()
  const ml = 46, mr = 120, mt = 14, mb = 40
  const W = Math.max(560, (lineRef.value?.parentElement?.clientWidth || 700))
  const H = 260
  const w = W - ml - mr, h = H - mt - mb
  svg.attr('width', W).attr('height', H)
  const g = svg.append('g').attr('transform', `translate(${ml},${mt})`)

  // X = normalized layer depth (0..1) so models with different layer counts align.
  const x = d3.scaleLinear().domain([0, 1]).range([0, w])
  const frac = (i, L) => L.length > 1 ? i / (L.length - 1) : 0

  // Y domain across all models' main metric (+ selected model's correct/wrong).
  const vals = []
  for (const m of models) for (const l of stats.models[m].layers) { const v = l[metric.value]; if (v != null) vals.push(v) }
  const sel = stats.models[selected.value]
  for (const l of sel.layers) {[l[metric.value + '_correct'], l[metric.value + '_wrong']].forEach(v => { if (v != null) vals.push(v) })}
  const y = d3.scaleLinear().domain(d3.extent(vals)).nice().range([h, 0])

  // component bands for the SELECTED model only (background context)
  const compColors = { vision_encoder: '#eff6ff', vision_tower: '#eff6ff', qformer_self_attention: '#f0fdf4', qformer_cross_attention: '#fef3c7', language_model: '#faf5ff', unknown: '#f8fafc' }
  const SL = sel.layers
  let start = 0
  for (let i = 1; i <= SL.length; i++) {
    if (i === SL.length || SL[i].component !== SL[start].component) {
      const x0 = x(frac(start, SL)), x1 = x(frac(i - 1, SL))
      g.append('rect').attr('x', x0).attr('y', 0).attr('width', Math.max(0, x1 - x0)).attr('height', h)
        .attr('fill', compColors[SL[start].component] || '#f8fafc').attr('opacity', 0.7)
      if (x1 - x0 > 30)
        g.append('text').attr('x', (x0 + x1) / 2).attr('y', 12).attr('text-anchor', 'middle')
          .attr('font-size', 9).attr('fill', '#94a3b8').text(compLabel(SL[start].component))
      start = i
    }
  }

  g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(6).tickFormat(d3.format('.0%')))
    .selectAll('text').style('font-size', '10px').style('fill', '#64748b')
  g.append('g').call(d3.axisLeft(y).ticks(5)).selectAll('text').style('font-size', '10px').style('fill', '#64748b')
  g.append('text').attr('x', w / 2).attr('y', h + 34).attr('text-anchor', 'middle').attr('font-size', 11).attr('fill', '#94a3b8').text('归一化层深（浅 → 深）')

  // all four models, main metric
  const mkLine = L => d3.line().defined(d => d[metric.value] != null).x((d, i) => x(frac(i, L))).y(d => y(d[metric.value]))
  models.forEach(m => {
    const L = stats.models[m].layers
    const isSel = m === selected.value
    g.append('path').datum(L).attr('fill', 'none').attr('stroke', MODEL_COLORS[m])
      .attr('stroke-width', isSel ? 3 : 1.5).attr('opacity', isSel ? 1 : 0.5).attr('d', mkLine(L))
    // end label
    const last = L[L.length - 1]
    g.append('text').attr('x', w + 6).attr('y', y(last[metric.value]))
      .attr('dominant-baseline', 'middle').attr('font-size', 10)
      .attr('font-weight', isSel ? 700 : 400).attr('fill', MODEL_COLORS[m])
      .text(stats.models[m].display_name)
  })

  // selected model's correct / wrong split (dashed)
  ;[['_correct', '#16a34a'], ['_wrong', '#dc2626']].forEach(([suf, col]) => {
    const key = metric.value + suf
    g.append('path').datum(SL).attr('fill', 'none').attr('stroke', col).attr('stroke-width', 1.4)
      .attr('stroke-dasharray', '3,3').attr('opacity', 0.9)
      .attr('d', d3.line().defined(d => d[key] != null).x((d, i) => x(frac(i, SL))).y(d => y(d[key])))
  })
}

function openInEvolution(modelKey, layerIdx) {
  broadcast(null, modelKey, 'diffusion')
  navigate('layer')
}

const detailNote = computed(() => {
  const M = stats.models[selected.value]
  if (!M) return ''
  const a = M.layers[0]?.entropy, b = M.layers[M.layers.length - 1]?.entropy
  if (selected.value === 'blip')
    return `BLIP2 注意力沿 视觉编码器 → Q-Former 自注意力 → Q-Former 跨注意力 → 语言模型 逐级收敛：熵从 ${a?.toFixed(2)} 降到 ${b?.toFixed(2)}，是四模型中唯一形成明显空间聚焦的。`
  return `${M.display_name} 的注意力在 self-attention 管线中始终高度扩散：熵从 ${a?.toFixed(2)} 仅变化到 ${b?.toFixed(2)}，末层仍未聚焦到具体 patch。`
})

const correctWrongNote = computed(() => {
  const M = stats.models[selected.value]
  if (!M) return ''
  const last = M.layers[M.layers.length - 1]
  const c = last?.[metric.value + '_correct'], w = last?.[metric.value + '_wrong']
  if (c == null || w == null) return ''
  const gap = Math.abs(c - w)
  return `诚实说明：正确(${c.toFixed(3)})与错误(${w.toFixed(3)})样本的末层差异仅 ${gap.toFixed(3)}，几乎重合——注意力集中度并不能可靠区分对错，这恰好印证「注意力不等于解释」，正确性主要由行为层而非热力图尖锐程度决定。`
})

onMounted(() => { nextTick(drawHeatmap); nextTick(drawLine) })
watch([metric, selected], () => { nextTick(drawHeatmap); nextTick(drawLine) })
</script>

<style scoped>
.df-sub { font-weight: 400; color: #94a3b8; font-size: 12px; margin-left: 10px; }
.df-callout { background: #f8fafc; border: 1px solid #e2e8f0; border-left: 4px solid #3b82f6; border-radius: 8px; padding: 12px 14px; font-size: 13px; line-height: 1.7; color: #334155; margin-bottom: 14px; }
.df-callout b { color: #1d4ed8; }
.df-metrics { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; margin-bottom: 12px; }
.df-mbtn { padding: 5px 12px; border: 1px solid #dbe3ef; background: #fff; color: #475569; border-radius: 6px; font-size: 12px; cursor: pointer; }
.df-mbtn-on { background: #1d4ed8; border-color: #1d4ed8; color: #fff; font-weight: 600; }
.df-info { font-size: 12px; color: #94a3b8; margin-left: 6px; }
.df-heatmap { overflow-x: auto; }
.df-legend { display: flex; align-items: center; gap: 8px; margin-top: 10px; font-size: 12px; color: #64748b; }
.df-legend-label { font-size: 11px; }
.df-legend-bar { width: 120px; height: 8px; border-radius: 4px; background: linear-gradient(to right, #4575b4, #ffffbf, #d73027); }
.df-hint { margin-left: auto; color: #94a3b8; }
.df-model-legend { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-bottom: 10px; }
.df-mleg { display: inline-flex; align-items: center; gap: 6px; border: 1px solid #e2e8f0; background: #fff; color: #475569; border-radius: 999px; padding: 4px 12px; font-size: 12px; cursor: pointer; }
.df-mleg.on { border-color: #1d4ed8; background: #eff6ff; color: #1d4ed8; font-weight: 700; }
.df-mleg .sw { width: 14px; height: 4px; border-radius: 2px; }
.df-detail-note { font-size: 13px; color: #334155; line-height: 1.6; margin-bottom: 10px; }
.df-open { float: right; border: 1px solid #bfdbfe; background: #eff6ff; color: #1d4ed8; border-radius: 6px; padding: 4px 10px; font-size: 12px; cursor: pointer; }
.df-open:hover { background: #dbeafe; }
.df-linechart { overflow-x: auto; }
.df-line-legend { display: flex; gap: 16px; align-items: center; font-size: 12px; color: #475569; margin-top: 8px; }
.df-line-legend .sw { display: inline-block; width: 14px; height: 4px; border-radius: 2px; vertical-align: middle; margin-right: 5px; }
.df-comp-hint { color: #94a3b8; }
.df-cvw-note { margin-top: 10px; font-size: 12px; color: #92400e; background: #fffbeb; border: 1px solid #fde68a; border-radius: 8px; padding: 9px 12px; line-height: 1.6; }
</style>
