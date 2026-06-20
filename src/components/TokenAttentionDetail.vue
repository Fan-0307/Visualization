<template>
  <div class="tad-root">
    <!-- ===== Controls Bar ===== -->
    <div class="card tad-card">
      <div class="tad-ctrls">
        <div class="tad-ctrl-group">
          <label class="tad-label">样本</label>
          <select v-model="sid" @change="onSampleChange" class="tad-select">
            <option v-for="s in samples" :key="s.file" :value="s.file">{{ s.file }}</option>
          </select>
        </div>

        <div class="tad-ctrl-group" v-if="!rollout">
          <label class="tad-label">层</label>
          <div class="tad-layer-nav">
            <button class="tad-btn-icon" @click="decLayer">◀</button>
            <span class="tad-layer-num">{{ layerIdx }} / {{ maxLayer }}</span>
            <button class="tad-btn-icon" @click="incLayer">▶</button>
          </div>
        </div>

        <div class="tad-spacer"></div>

        <button class="tad-btn-rollout" :class="{ on: rollout && hasRollout, dim: rollout && !hasRollout }"
                @click="rollout = !rollout"
                :title="!hasRollout ? '当前模型无 Rollout 数据（BLIP 不支持）' : rollout ? 'Attention Rollout ON (Abnar & Zuidema 2020)' : '切换到 Rollout 视图'">
          ⟲ Rollout{{ !hasRollout ? ' ✕' : '' }}
        </button>

        <button class="tad-btn-agg" :class="{ on: aggMode }" @click="aggMode = !aggMode">
          {{ aggMode ? '⊞ 聚合视图' : '⊟ 逐 token' }}
        </button>

        <button class="tad-btn-baseline" :class="{ on: baseline }" @click="baseline = !baseline"
                :title="hasGlobalStats ? '全局跨样本基线' : '当前样本内基线'">
          {{ baseline ? (hasGlobalStats ? '⊘ 全局基线' : '⊘ 已去基线') : '◌ 去基线' }}
        </button>

        <span v-if="sampleData" class="tad-badge" :class="sampleData.correct ? 'ok' : 'err'">
          {{ sampleData.correct ? '✓ 正确' : '✗ 错误' }}
        </span>

        <label class="tad-check">
          <input type="checkbox" v-model="smooth" /> 平滑
        </label>

        <button v-if="!aggMode" class="tad-btn-play" :class="{ playing }" @click="togglePlay">
          {{ playing ? '⏸ 暂停' : '▶ 播放' }}
        </button>
      </div>

      <!-- Layer attention sparkline -->
      <svg v-if="sampleData" ref="sparkSvg" class="tad-sparkline" @click="onSparkClick"></svg>
    </div>

    <!-- ===== Empty State ===== -->
    <div v-if="!sampleData && samples.length === 0" class="tad-empty card">
      <div class="tad-empty-icon">🔬</div>
      <div class="tad-empty-title">暂无逐 Token 注意力数据</div>
      <div class="tad-empty-desc">
        请先运行 <code>data/qwen/run.py</code> 或 <code>data/blip/run.py</code> 生成数据，<br />
        然后执行 <code>python data/build_index.py</code> 构建索引。
      </div>
    </div>

    <div v-else-if="!sampleData && samples.length > 0" class="tad-empty card">
      <div class="tad-empty-icon">📂</div>
      <div class="tad-empty-title">选择样本以开始分析</div>
    </div>

    <!-- ===== Main View ===== -->
    <div v-if="sampleData" class="tad-main">
      <!-- Left: Image + heatmap -->
      <div class="tad-img-panel card">
        <div class="tad-img-frame">
          <img
            :src="`/img/${sampleData.image.img_id}.jpeg`"
            :width="imgDisplayW"
            :height="imgDisplayH"
            class="tad-img"
            alt="input image"
            @error="e => e.target.style.display = 'none'"
          />
          <canvas ref="hmCanvas" class="tad-hm" :width="imgDisplayW" :height="imgDisplayH"></canvas>
        </div>
        <!-- Color legend -->
        <div class="tad-legend">
          <span class="tad-legend-label">低</span>
          <svg ref="legendSvg" class="tad-legend-bar"></svg>
          <span class="tad-legend-label">高</span>
        </div>
      </div>

      <!-- Right: Token panel -->
      <div class="tad-tok-panel">
        <!-- Question tokens -->
        <div class="tad-tok-section">
          <div class="tad-tok-header">
            <span class="tad-tok-header-icon">📝</span>
            <span>Question</span>
          </div>
          <div class="tad-tok-row">
            <div
              v-for="(t, i) in questionList"
              :key="'q' + i"
              class="tad-tok"
              :class="{ active: selType === 'q' && selIdx === i, pulse: playing && selType === 'q' && selIdx === i }"
              @click="selectToken('q', i)"
            >
              <span class="tad-tok-text">{{ t.token }}</span>
              <span
                class="tad-tok-bar"
                :style="{ background: tokenBarColor(t), width: tokenBarPct(t) + '%' }"
              ></span>
            </div>
          </div>
        </div>

        <!-- Answer tokens -->
        <div class="tad-tok-section">
          <div class="tad-tok-header">
            <span class="tad-tok-header-icon">💬</span>
            <span>Answer</span>
          </div>
          <div class="tad-tok-row">
            <div
              v-for="(t, i) in answerList"
              :key="'a' + i"
              class="tad-tok"
              :class="{ active: selType === 'a' && selIdx === i, pulse: playing && selType === 'a' && selIdx === i }"
              @click="selectToken('a', i)"
            >
              <span class="tad-tok-text">{{ t.token }}</span>
              <span
                class="tad-tok-bar"
                :style="{ background: tokenBarColor(t), width: tokenBarPct(t) + '%' }"
              ></span>
            </div>
          </div>
        </div>

        <!-- Selected token info -->
        <div v-if="selToken" class="tad-tok-info card">
          <div class="tad-tok-info-row">
            <span class="tad-tok-info-label">选中 Token</span>
            <span class="tad-tok-info-token">「{{ selToken.token }}」</span>
          </div>
          <div class="tad-tok-info-row">
            <span class="tad-tok-info-label">视觉注意力均值</span>
            <span class="tad-tok-info-val">{{ fmtScore(visionScore(selToken)) }}</span>
            <span class="tad-tok-info-label" style="margin-left:16px">文本注意力均值</span>
            <span class="tad-tok-info-val">{{ fmtScore(textScore(selToken)) }}</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, nextTick, onBeforeUnmount, inject } from 'vue'
import * as d3 from 'd3'
import attnIndex from '../data/attn_index.json'

const broadcast = inject('broadcast', () => {})
const consume = inject('consume', () => null)

// ===== Data =====
const samples = attnIndex.samples || []
const sid = ref(samples[0]?.file || '')

// Map an incoming linked sample_id (== question_id) to one of this view's files.
function fileForQid(qid, preferModel) {
  if (qid == null) return null
  const matches = samples.filter(s => String(s.question_id) === String(qid))
  if (!matches.length) return null
  const preferred = preferModel && matches.find(s => s.model === preferModel)
  return (preferred || matches[0]).file
}
const currentQid = computed(() =>
  samples.find(s => s.file === sid.value)?.question_id ?? null)
const layerIdx = ref(0)
const maxLayer = ref(11)
const sampleData = ref(null)
const selType = ref('q')
const selIdx = ref(0)
const hmCanvas = ref(null)
const sparkSvg = ref(null)
const legendSvg = ref(null)
const smooth = ref(true)
const playing = ref(false)
const baseline = ref(true)   // default on: removes attention sinks
const aggMode = ref(false)   // aggregate all answer tokens across all layers
const rollout = ref(false)   // Rollout is near-zero for causal LMs; off by default
const imgDisplayW = 360
const imgDisplayH = ref(270)
let playTimer = null

// ===== Layout helpers =====
const gridW = computed(() => sampleData.value?.image.grid.w || 1)
const gridH = computed(() => sampleData.value?.image.grid.h || 1)

// Whether current sample actually has rollout data
const hasRollout = computed(() => {
  const q = sampleData.value?.question || {}
  const first = Object.values(q)[0]
  return !!first?.vis_attn?.rollout
})

// Current layer key: 'rollout' or 'layer_N'
// If rollout toggled but no rollout data (e.g. BLIP), fall back to last layer
const layerKey = computed(() => {
  if (rollout.value && hasRollout.value) return 'rollout'
  return `layer_${layerIdx.value}`
})

// Tokens that are pure template noise (chat format boilerplate / special tokens)
const SKIP_TOKENS = new Set([
  '<s>', '</s>', '<|im_start|>', '<|im_end|>', '<|endoftext|>',
  'USER', 'ASSISTANT', 'user', 'assistant',
  'USER:', 'ASSISTANT:',
  'US', 'ER', 'ASS', 'IST', 'ANT',   // LLaVA sub-word splits of USER/ASSISTANT
  '\n', '\r\n', '', ' ',
  ':', ':\n', ' \n',
])

function tokenList(obj) {
  const lk = layerKey.value
  return Object.entries(obj || {})
    .map(([k, v]) => ({
      key: k,
      token: v.token,
      vis: v.vis_attn?.[lk] || null,
      que: v.que_attn?.[lk] || null,
    }))
    .filter(t => !SKIP_TOKENS.has(t.token))
}

// Global stats from attn_index (computed once per model)
const globalStats = computed(() => {
  if (!sampleData.value) return null
  const model = sampleData.value.model
  return attnIndex.global_stats?.[model] || null
})
const hasGlobalStats = computed(() => !!globalStats.value)

const questionList = computed(() => tokenList(sampleData.value?.question))
const answerList = computed(() => tokenList(sampleData.value?.answer))

// ===== Selected token =====
const selToken = computed(() => {
  const list = selType.value === 'q' ? questionList.value : answerList.value
  return list[selIdx.value] || null
})

// Full raw entry for sparkline (access all layers)
const selTokenFull = computed(() => {
  if (!sampleData.value) return null
  const dict = selType.value === 'q' ? sampleData.value.question : sampleData.value.answer
  if (!dict) return null
  const keys = Object.keys(dict)
  if (selIdx.value >= keys.length) return null
  return dict[keys[selIdx.value]]
})

// ===== Attention scores =====
function meanVec(vec) {
  if (!vec || !vec.length) return 0
  return vec.reduce((a, b) => a + b, 0) / vec.length
}

const visionScore = (t) => meanVec(t.vis)
const textScore = (t) => meanVec(t.que)

const allTokens = computed(() => [...questionList.value, ...answerList.value])

const maxVisScore = computed(() => {
  let m = 0
  for (const t of allTokens.value) m = Math.max(m, visionScore(t))
  return m || 1
})

function tokenBarColor(t) {
  const v = maxVisScore.value > 0 ? visionScore(t) / maxVisScore.value : 0
  return d3.interpolateOrRd(v)
}

function tokenBarPct(t) {
  const v = maxVisScore.value > 0 ? visionScore(t) / maxVisScore.value : 0
  return Math.round(v * 100)
}

function fmtScore(v) {
  if (v == null || isNaN(v)) return '—'
  if (v === 0) return '0'
  if (v < 0.0001) return v.toExponential(2)
  return v.toFixed(5)
}

// ===== Sparkline data =====
const sparklineData = computed(() => {
  if (!selTokenFull.value) return []
  const vis = selTokenFull.value.vis_attn || {}
  const out = []
  for (let i = 0; i <= maxLayer.value; i++) {
    const vec = vis[`layer_${i}`]
    out.push(vec && vec.length ? vec.reduce((a, b) => a + b, 0) / vec.length : 0)
  }
  return out
})

// ===== Navigation =====
function selectToken(type, idx) {
  selType.value = type
  selIdx.value = idx
}

function incLayer() {
  if (layerIdx.value < maxLayer.value) layerIdx.value++
}
function decLayer() {
  if (layerIdx.value > 0) layerIdx.value--
}

function nextToken() {
  const ql = questionList.value
  const al = answerList.value
  const total = ql.length + al.length
  if (total === 0) return
  const cur = selType.value === 'q' ? selIdx.value : ql.length + selIdx.value
  const next = (cur + 1) % total
  if (next < ql.length) {
    selType.value = 'q'
    selIdx.value = next
  } else {
    selType.value = 'a'
    selIdx.value = next - ql.length
  }
}

function togglePlay() {
  playing.value = !playing.value
  if (playing.value) {
    playTimer = setInterval(nextToken, 2500)
  } else {
    clearInterval(playTimer)
    playTimer = null
  }
}

onBeforeUnmount(() => clearInterval(playTimer))

// ===== Sparkline click → layer navigation =====
function onSparkClick(e) {
  const svg = sparkSvg.value
  if (!svg || !sparklineData.value.length) return
  const rect = svg.getBoundingClientRect()
  const x = e.clientX - rect.left
  const frac = x / rect.width
  layerIdx.value = Math.round(frac * (sparklineData.value.length - 1))
}

// ===== Baseline: Z-score per patch (token_value - mean) / std =====
// Prefers global cross-sample stats (more stable, captures true attention sinks).
// Falls back to per-sample stats if no global stats available.
const baselineStats = computed(() => {
  // --- Global stats path ---
  if (globalStats.value) {
    const { mean, std } = globalStats.value
    return {
      mean: new Float64Array(mean),
      std: new Float64Array(std),
      isGlobal: true,
    }
  }
  // --- Per-sample fallback ---
  const all = allTokens.value.filter(t => t.vis && t.vis.length > 0)
  if (all.length < 2) return null
  const n = all[0].vis.length
  const mean = new Float64Array(n)
  const m2 = new Float64Array(n)
  // Two-pass for numerical stability
  for (const t of all) for (let i = 0; i < n; i++) mean[i] += t.vis[i]
  for (let i = 0; i < n; i++) mean[i] /= all.length
  for (const t of all) {
    for (let i = 0; i < n; i++) {
      const d = t.vis[i] - mean[i]
      m2[i] += d * d
    }
  }
  const std = new Float64Array(n)
  for (let i = 0; i < n; i++) std[i] = Math.sqrt(m2[i] / all.length) + 1e-12
  return { mean, std, isGlobal: false }
})

// ===== Aggregate vec: element-wise max across all tokens, √-transformed =====
function computeAggVec() {
  if (!sampleData.value) return null
  const allToks = [
    ...Object.values(sampleData.value.question || {}),
    ...Object.values(sampleData.value.answer   || {}),
  ]
  const valid = allToks.filter(t => t.vis_attn && Object.keys(t.vis_attn).length)
  if (!valid.length) return null

  // Average across all layers per token, then max across tokens
  const tokenVecs = valid.map(t => {
    const vecs = Object.values(t.vis_attn).filter(v => v?.length)
    if (!vecs.length) return null
    const n = vecs[0].length
    const avg = new Float64Array(n)
    for (const v of vecs) for (let i = 0; i < n; i++) avg[i] += v[i] / vecs.length
    return avg
  }).filter(Boolean)

  if (!tokenVecs.length) return null
  const n = tokenVecs[0].length
  const result = new Float64Array(n)
  for (const v of tokenVecs) for (let i = 0; i < n; i++) result[i] = Math.max(result[i], v[i])
  return result
}

// ===== D3: Heatmap on canvas =====
function drawHeatmap() {
  const canvas = hmCanvas.value
  if (!canvas) return
  const ctx = canvas.getContext('2d')
  ctx.clearRect(0, 0, canvas.width, canvas.height)

  const gW = gridW.value
  const gH = gridH.value
  const nPatches = gW * gH

  // Aggregate mode: show all-token max instead of selected token
  if (aggMode.value) {
    const aggVec = computeAggVec()
    if (!aggVec) return
    const sliceOffset = aggVec.length > nPatches ? aggVec.length - nPatches : 0
    const raw = Array.from(aggVec).slice(sliceOffset, sliceOffset + nPatches)
    const transformed = raw.map(v => Math.sqrt(v))
    const mx = Math.max(...transformed) || 1
    const off = document.createElement('canvas')
    off.width = gW; off.height = gH
    const offCtx = off.getContext('2d')
    const imgData = offCtx.createImageData(gW, gH)
    const d = imgData.data
    for (let i = 0; i < nPatches && i < transformed.length; i++) {
      const c = d3.rgb(d3.interpolateOrRd(transformed[i] / mx))
      d[i*4]=c.r; d[i*4+1]=c.g; d[i*4+2]=c.b; d[i*4+3]=185
    }
    offCtx.putImageData(imgData, 0, 0)
    ctx.imageSmoothingEnabled = smooth.value
    ctx.imageSmoothingQuality = 'high'
    ctx.drawImage(off, 0, 0, canvas.width, canvas.height)
    return
  }

  if (!selToken.value?.vis) return
  let vec = selToken.value.vis
  // sliceOffset: how many leading elements were stripped (e.g. BLIP's CLS token)
  const sliceOffset = vec.length > nPatches ? vec.length - nPatches : 0
  let patches = Array.from(sliceOffset > 0 ? vec.slice(sliceOffset) : vec)

  // Apply Z-score baseline to remove attention sinks
  let useFixedScale = false
  if (baseline.value && baselineStats.value) {
    const { mean, std } = baselineStats.value
    const maxZ = 3
    patches = patches.map((v, i) => {
      const z = (v - mean[sliceOffset + i]) / std[sliceOffset + i]
      return Math.max(0, z) / maxZ
    })
    useFixedScale = true
  }

  // √ transform: compress high values, boost low values → better contrast
  patches = patches.map(v => Math.sqrt(Math.max(0, v)))

  // Top-K threshold: zero out bottom 70%, only show top 30% patches
  // This creates a spotlight effect that highlights the most attended regions
  const sorted = [...patches].sort((a, b) => a - b)
  const threshold = sorted[Math.floor(sorted.length * 0.70)] || 0
  patches = patches.map(v => v > threshold ? v - threshold : 0)

  const mx = useFixedScale
    ? (d3.max(patches) || 1)   // rescale after threshold
    : (d3.max(patches) || 1)

  const off = document.createElement('canvas')
  off.width = gW
  off.height = gH
  const offCtx = off.getContext('2d')
  const imgData = offCtx.createImageData(gW, gH)
  const data = imgData.data

  for (let i = 0; i < gW * gH && i < patches.length; i++) {
    const v = patches[i] / mx
    const c = d3.rgb(d3.interpolateOrRd(v))
    data[i * 4] = c.r
    data[i * 4 + 1] = c.g
    data[i * 4 + 2] = c.b
    data[i * 4 + 3] = 200
  }
  offCtx.putImageData(imgData, 0, 0)

  ctx.imageSmoothingEnabled = smooth.value
  ctx.imageSmoothingQuality = 'high'
  ctx.drawImage(off, 0, 0, canvas.width, canvas.height)
}

// ===== D3: Sparkline SVG =====
function drawSparkline() {
  const svg = sparkSvg.value
  if (!svg) return
  d3.select(svg).selectAll('*').remove()

  // In rollout mode, the sparkline is not applicable (rollout already integrates all layers)
  if (rollout.value) {
    const W = svg.clientWidth || 640
    d3.select(svg).append('text')
      .attr('x', W / 2).attr('y', 26)
      .attr('text-anchor', 'middle')
      .attr('font-size', '11px')
      .attr('fill', '#94a3b8')
      .text('Rollout 模式：已跨所有层累积注意力，无需逐层导航')
    return
  }

  const data = sparklineData.value
  if (!data.length) return

  const W = svg.clientWidth || 640
  const H = 42
  const pad = { top: 6, right: 8, bottom: 6, left: 2 }
  const iw = W - pad.left - pad.right
  const ih = H - pad.top - pad.bottom

  const g = d3.select(svg).append('g').attr('transform', `translate(${pad.left},${pad.top})`)

  // Background grid lines
  const yLo = d3.min(data) || 0
  const yHi = d3.max(data) || 1
  const yRange = yHi - yLo || 1
  const x = d3.scaleLinear().domain([0, data.length - 1]).range([0, iw])
  const y = d3.scaleLinear().domain([yLo - yRange * 0.05, yHi + yRange * 0.05]).range([ih, 0])

  // Area fill
  const area = d3.area()
    .x((_, i) => x(i))
    .y0(ih)
    .y1(d => y(d))
    .curve(d3.curveMonotoneX)
  g.append('path').datum(data).attr('d', area).attr('fill', 'rgba(59,130,246,0.08)')

  // Line
  const line = d3.line()
    .x((_, i) => x(i))
    .y(d => y(d))
    .curve(d3.curveMonotoneX)
  g.append('path').datum(data).attr('d', line)
    .attr('fill', 'none').attr('stroke', '#94a3b8').attr('stroke-width', 1.2)

  // Current layer indicator
  const cx = x(layerIdx.value)
  const cy = y(data[layerIdx.value] || 0)
  g.append('line')
    .attr('x1', cx).attr('x2', cx)
    .attr('y1', 0).attr('y2', ih)
    .attr('stroke', '#3b82f6').attr('stroke-width', 1).attr('stroke-dasharray', '3,2')

  g.append('circle')
    .attr('cx', cx).attr('cy', cy)
    .attr('r', 4).attr('fill', '#3b82f6').attr('stroke', '#fff').attr('stroke-width', 1.5)
}

// ===== D3: Color legend SVG =====
function drawLegend() {
  const svg = legendSvg.value
  if (!svg) return
  d3.select(svg).selectAll('*').remove()

  const W = 120, H = 14
  d3.select(svg).attr('viewBox', `0 0 ${W} ${H}`).attr('width', W).attr('height', H)

  const defs = d3.select(svg).append('defs')
  const grad = defs.append('linearGradient').attr('id', 'tad-legend-grad')
  for (let i = 0; i <= 10; i++) {
    grad.append('stop').attr('offset', `${i * 10}%`).attr('stop-color', d3.interpolateOrRd(i / 10))
  }

  d3.select(svg).append('rect')
    .attr('x', 0).attr('y', 0)
    .attr('width', W).attr('height', H)
    .attr('rx', 3)
    .attr('fill', 'url(#tad-legend-grad)')
}

// ===== Load sample =====
async function loadSample(file) {
  try {
    const resp = await fetch(`/data/attn/${file}`)
    const json = await resp.json()
    sampleData.value = json

    // Count only layer_N keys (exclude 'rollout')
    const countLayerKeys = (t) =>
      Object.keys(t.vis_attn || {}).filter(k => k.startsWith('layer_')).length
    const nLayers = Math.max(
      ...Object.values(json.question || {}).map(countLayerKeys),
      ...Object.values(json.answer || {}).map(countLayerKeys),
      1
    )
    maxLayer.value = nLayers - 1
    // Default to ~70% depth: semantic attention is strongest in middle-late layers
    layerIdx.value = Math.floor(nLayers * 0.70)
    selType.value = 'q'
    selIdx.value = 0

    const ratio = json.image.h / json.image.w
    imgDisplayH.value = Math.round(imgDisplayW * ratio)
  } catch (e) {
    console.error('Failed to load sample:', file, e)
    sampleData.value = null
  }
}

function onSampleChange() {
  loadSample(sid.value)
  if (currentQid.value != null) broadcast(currentQid.value, null, 'token')
}

// ===== Watchers =====
watch([selType, selIdx, layerIdx, sampleData, smooth, baseline, aggMode, rollout], () => {
  nextTick(drawHeatmap)
})

watch([selTokenFull, layerIdx, sampleData, rollout], () => {
  nextTick(drawSparkline)
})

watch([sampleData], () => {
  nextTick(drawLegend)
})

// ===== Init =====
// Consume a pending cross-view selection (e.g. a sample chosen in the diagnosis view).
const pending = consume('token')
if (pending?.sampleId != null) {
  const f = fileForQid(pending.sampleId, pending.model)
  if (f) sid.value = f
}
if (samples.length) loadSample(sid.value)
</script>

<style scoped>
/* ===== Root ===== */
.tad-root {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

/* ===== Card override ===== */
.tad-card {
  padding: 16px 20px;
}

/* ===== Controls ===== */
.tad-ctrls {
  display: flex;
  gap: 20px;
  align-items: center;
  flex-wrap: wrap;
}

.tad-ctrl-group {
  display: flex;
  align-items: center;
  gap: 8px;
}

.tad-label {
  font-size: 12px;
  font-weight: 600;
  color: #6b7280;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  white-space: nowrap;
}

.tad-select {
  padding: 5px 10px;
  border: 1px solid #d1d5db;
  border-radius: 6px;
  font-size: 13px;
  background: #fff;
  color: #1f2937;
  cursor: pointer;
  max-width: 320px;
  transition: border-color 0.15s;
}
.tad-select:focus {
  outline: none;
  border-color: #3b82f6;
  box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.15);
}

.tad-layer-nav {
  display: flex;
  align-items: center;
  gap: 4px;
}

.tad-btn-icon {
  width: 28px;
  height: 28px;
  border: 1px solid #d1d5db;
  border-radius: 6px;
  background: #fff;
  cursor: pointer;
  font-size: 11px;
  color: #4b5563;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.15s;
}
.tad-btn-icon:hover {
  background: #f3f4f6;
  border-color: #9ca3af;
}

.tad-layer-num {
  font-size: 13px;
  font-weight: 600;
  color: #1f2937;
  min-width: 56px;
  text-align: center;
  font-variant-numeric: tabular-nums;
}

.tad-spacer {
  flex: 1;
}

/* Badge */
.tad-badge {
  padding: 3px 12px;
  border-radius: 10px;
  font-size: 12px;
  font-weight: 600;
}
.tad-badge.ok {
  background: #dcfce7;
  color: #16a34a;
}
.tad-badge.err {
  background: #fee2e2;
  color: #dc2626;
}

/* Check */
.tad-check {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 12px;
  color: #6b7280;
  cursor: pointer;
  user-select: none;
}
.tad-check input {
  margin: 0;
}

/* Play button */
.tad-btn-play {
  padding: 5px 16px;
  font-size: 13px;
  cursor: pointer;
  border: 1px solid #d1d5db;
  border-radius: 6px;
  background: #fff;
  color: #374151;
  font-weight: 500;
  transition: all 0.15s;
}
.tad-btn-play:hover {
  background: #f9fafb;
  border-color: #9ca3af;
}
.tad-btn-play.playing {
  background: #ecfdf5;
  border-color: #10b981;
  color: #059669;
}

/* Rollout toggle */
.tad-btn-rollout {
  padding: 5px 14px;
  font-size: 12px;
  cursor: pointer;
  border: 1px solid #d1d5db;
  border-radius: 6px;
  background: #fff;
  color: #6b7280;
  font-weight: 500;
  transition: all 0.2s;
  white-space: nowrap;
}
.tad-btn-rollout:hover { background: #f9fafb; border-color: #9ca3af; }
.tad-btn-rollout.on {
  background: #f0fdf4;
  border-color: #22c55e;
  color: #15803d;
  font-weight: 600;
}
.tad-btn-rollout.dim {
  opacity: 0.45;
  cursor: not-allowed;
}

/* Aggregate toggle */
.tad-btn-agg {
  padding: 5px 14px;
  font-size: 12px;
  cursor: pointer;
  border: 1px solid #d1d5db;
  border-radius: 6px;
  background: #fff;
  color: #6b7280;
  font-weight: 500;
  transition: all 0.2s;
  white-space: nowrap;
}
.tad-btn-agg:hover { background: #f9fafb; border-color: #9ca3af; }
.tad-btn-agg.on {
  background: #fef3c7;
  border-color: #d97706;
  color: #92400e;
  font-weight: 600;
}

/* Baseline toggle */
.tad-btn-baseline {
  padding: 5px 14px;
  font-size: 12px;
  cursor: pointer;
  border: 1px solid #d1d5db;
  border-radius: 6px;
  background: #fff;
  color: #6b7280;
  font-weight: 500;
  transition: all 0.2s;
  white-space: nowrap;
}
.tad-btn-baseline:hover {
  background: #f9fafb;
  border-color: #9ca3af;
}
.tad-btn-baseline.on {
  background: #eff6ff;
  border-color: #3b82f6;
  color: #2563eb;
  font-weight: 600;
}

/* ===== Sparkline ===== */
.tad-sparkline {
  width: 100%;
  height: 42px;
  margin-top: 10px;
  cursor: pointer;
  border-radius: 6px;
  background: #fafbfc;
  border: 1px solid #f0f0f0;
}
.tad-sparkline:hover {
  background: #f5f7fa;
}

/* ===== Empty state ===== */
.tad-empty {
  text-align: center;
  padding: 56px 24px;
  border-radius: 12px;
}
.tad-empty-icon {
  font-size: 40px;
  margin-bottom: 12px;
}
.tad-empty-title {
  font-size: 16px;
  font-weight: 600;
  color: #6b7280;
  margin-bottom: 8px;
}
.tad-empty-desc {
  font-size: 13px;
  color: #9ca3af;
  line-height: 1.7;
}
.tad-empty-desc code {
  background: #f3f4f6;
  padding: 1px 6px;
  border-radius: 3px;
  font-size: 12px;
  color: #6b7280;
}

/* ===== Main layout ===== */
.tad-main {
  display: flex;
  gap: 20px;
  align-items: flex-start;
}

/* ===== Image panel ===== */
.tad-img-panel {
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
  padding: 16px;
}

.tad-img-frame {
  position: relative;
  line-height: 0;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.12);
}

.tad-img {
  display: block;
}

.tad-hm {
  position: absolute;
  top: 0;
  left: 0;
  pointer-events: none;
  transition: opacity 0.25s;
}

/* Legend */
.tad-legend {
  display: flex;
  align-items: center;
  gap: 6px;
}
.tad-legend-label {
  font-size: 10px;
  color: #9ca3af;
  font-weight: 500;
}
.tad-legend-bar {
  flex-shrink: 0;
}

/* ===== Token panel ===== */
.tad-tok-panel {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.tad-tok-section {
  background: #fff;
  border-radius: 10px;
  padding: 14px 16px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
  border: 1px solid #f0f0f0;
}

.tad-tok-header {
  font-size: 12px;
  font-weight: 600;
  color: #6b7280;
  margin-bottom: 10px;
  display: flex;
  align-items: center;
  gap: 6px;
}
.tad-tok-header-icon {
  font-size: 14px;
}

.tad-tok-row {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
}

/* Token pill */
.tad-tok {
  display: inline-flex;
  flex-direction: column;
  align-items: center;
  padding: 4px 9px;
  font-size: 13px;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  border: 1.5px solid #e5e7eb;
  border-radius: 6px;
  cursor: pointer;
  background: #fafbfc;
  color: #374151;
  transition: transform 0.15s, box-shadow 0.15s, border-color 0.15s, background 0.15s;
  min-width: 28px;
  user-select: none;
}
.tad-tok:hover {
  transform: translateY(-1px);
  border-color: #b0b7c3;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);
}
.tad-tok.active {
  border-color: #f97316;
  box-shadow: 0 0 0 2px rgba(249, 115, 22, 0.25);
  background: #fff;
  transform: translateY(-2px);
}
.tad-tok.pulse {
  animation: tok-pulse 0.6s ease-in-out;
}

.tad-tok-text {
  line-height: 1.3;
  white-space: nowrap;
}

/* Token attention bar */
.tad-tok-bar {
  height: 3px;
  border-radius: 1.5px;
  margin-top: 3px;
  transition: width 0.3s, background 0.3s;
  min-width: 4px;
  align-self: stretch;
}

/* Selected token info */
.tad-tok-info {
  padding: 12px 16px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.tad-tok-info-row {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
}
.tad-tok-info-label {
  color: #9ca3af;
  font-weight: 500;
}
.tad-tok-info-token {
  font-weight: 700;
  color: #1f2937;
  font-size: 14px;
}
.tad-tok-info-val {
  font-weight: 600;
  color: #3b82f6;
  font-variant-numeric: tabular-nums;
  font-family: 'SF Mono', 'Fira Code', monospace;
}

/* ===== Animations ===== */
@keyframes tok-pulse {
  0%   { transform: scale(1); }
  30%  { transform: scale(1.12); }
  100% { transform: scale(1); }
}
</style>
