<template>
  <div class="levo-root">
    <!-- ═══ Controls ═══ -->
    <div class="card levo-ctrl">
      <div class="levo-ctrl-row">
        <div class="levo-cg">
          <label class="levo-lbl">模型 A</label>
          <select v-model="modelA" @change="onModelAChange" class="levo-sel">
            <option v-for="m in models" :key="m" :value="m">{{ m }}</option>
          </select>
        </div>
        <div class="levo-cg">
          <label class="levo-lbl">样本</label>
          <select v-model="sampleId" @change="onSampleChange" class="levo-sel levo-sel-wide">
            <option v-for="s in sampleList" :key="s.id" :value="s.id">
              {{ s.correct ? '✓' : '✗' }} 图{{ s.image_id }} · {{ s.question?.slice(0, 50) }}...
            </option>
          </select>
        </div>
        <div class="levo-cg">
          <label class="levo-lbl">
            <input type="checkbox" v-model="compareMode" @change="onCompareToggle" />
            双模型对比
          </label>
        </div>
        <div class="levo-cg" v-if="compareMode">
          <label class="levo-lbl">模型 B</label>
          <select v-model="modelB" @change="onModelBChange" class="levo-sel">
            <option v-for="m in models.filter(x => x !== modelA)" :key="m" :value="m">{{ m }}</option>
          </select>
          <label class="levo-lbl" style="margin-left:8px">
            <input type="checkbox" v-model="syncMode" /> 同步推进
          </label>
        </div>
      </div>
      <div v-if="sampleInfo" class="levo-qt">
        <span class="levo-qt-lbl">Q:</span> {{ sampleInfo.question }}
        <span class="levo-gt">{{ sampleInfo.gt }}</span>
      </div>
    </div>

    <!-- ═══ Empty ═══ -->
    <div v-if="!sampleId" class="card levo-empty">
      <div class="levo-empty-icon">🧬</div>
      <div>选择一个样本查看逐层注意力演化</div>
      <div style="font-size:12px;color:#999;margin-top:4px">
        数据: {{ totalSamples }} 样本 / {{ models.length }} 模型
      </div>
    </div>

    <template v-if="sampleId && dataA">
      <!-- ═══ Single model: layer nav + sparkline ═══ -->
      <div class="card levo-nav-card" v-if="!compareMode">
        <div class="levo-nav-row">
          <button class="levo-btn" @click="decLayer('A')" :disabled="layerIdxA <= 0">◀</button>
          <input type="range" :min="0" :max="maxLayerA" v-model.number="layerIdxA"
            class="levo-slider" @input="onSlide('A')" />
          <button class="levo-btn" @click="incLayer('A')" :disabled="layerIdxA >= maxLayerA">▶</button>
          <span class="levo-layer-num">{{ modelA }} 层 {{ layerIdxA }} / {{ maxLayerA }}</span>
          <span class="levo-comp-tag" :style="{background: compColor(curComponent(dataA, layerIdxA))}">
            {{ curComponent(dataA, layerIdxA)?.replace(/_/g, ' ') || 'unknown' }}
          </span>
          <span class="levo-semantic-sm">{{ semA?.text }}</span>
          <button class="levo-btn-play" @click="togglePlay"> {{ playing ? '⏸' : '▶' }} </button>
          <label class="levo-speed-lbl">
            速度 <input type="range" min="100" max="2000" step="100" v-model.number="playSpeed"
              style="width:60px" /> {{ playSpeed }}ms
          </label>
        </div>
        <svg ref="sparkSvgA" class="levo-spark" @click="onSparkClick($event, 'A')"></svg>
      </div>

      <!-- ═══ Comparison mode: dual layer nav + dual sparklines ═══ -->
      <div v-if="compareMode && dataB" class="card levo-nav-card">
        <!-- Model A timeline -->
        <div class="levo-timeline-label">{{ modelA }} ({{ totalLayersA }} 层, {{ componentsA.join(' → ') }})</div>
        <div class="levo-nav-row">
          <button class="levo-btn" @click="decLayer('A')" :disabled="layerIdxA <= 0">◀</button>
          <input type="range" :min="0" :max="maxLayerA" v-model.number="layerIdxA"
            class="levo-slider" @input="onSlide('A')" />
          <button class="levo-btn" @click="incLayer('A')" :disabled="layerIdxA >= maxLayerA">▶</button>
          <span class="levo-layer-num-sm">层 {{ layerIdxA }}/{{ maxLayerA }}</span>
          <span class="levo-comp-tag" :style="{background: compColor(curComponent(dataA, layerIdxA))}">
            {{ curComponent(dataA, layerIdxA)?.replace(/_/g, ' ') || 'unknown' }}
          </span>
          <span class="levo-semantic-sm">{{ semA?.text }}</span>
        </div>
        <svg ref="sparkSvgA" class="levo-spark" @click="onSparkClick($event, 'A')"></svg>

        <!-- Model B timeline -->
        <div class="levo-timeline-label" style="margin-top:10px">{{ modelB }} ({{ totalLayersB }} 层, {{ componentsB.join(' → ') }})</div>
        <div class="levo-nav-row">
          <button class="levo-btn" @click="decLayer('B')" :disabled="layerIdxB <= 0">◀</button>
          <input type="range" :min="0" :max="maxLayerB" v-model.number="layerIdxB"
            class="levo-slider" @input="onSlide('B')" />
          <button class="levo-btn" @click="incLayer('B')" :disabled="layerIdxB >= maxLayerB">▶</button>
          <span class="levo-layer-num-sm">层 {{ layerIdxB }}/{{ maxLayerB }}</span>
          <span class="levo-comp-tag" :style="{background: compColor(curComponent(dataB, layerIdxB))}">
            {{ curComponent(dataB, layerIdxB)?.replace(/_/g, ' ') || 'unknown' }}
          </span>
          <span class="levo-semantic-sm">{{ semB?.text }}</span>
        </div>
        <svg ref="sparkSvgB" class="levo-spark" @click="onSparkClick($event, 'B')"></svg>

        <!-- Shared playback -->
        <div class="levo-play-shared">
          <button class="levo-btn-play" @click="togglePlay"> {{ playing ? '⏸' : '▶' }} </button>
          <label class="levo-speed-lbl">
            速度 <input type="range" min="100" max="2000" step="100" v-model.number="playSpeed"
              style="width:60px" /> {{ playSpeed }}ms
          </label>
        </div>
      </div>

      <!-- ═══ Heatmap panels ═══ -->
      <div :class="['levo-panels', compareMode && dataB && 'levo-panels-dual']">
        <!-- Panel A -->
        <div class="card levo-panel">
          <div class="levo-panel-hdr">
            <span class="levo-panel-model">{{ modelA }}</span>
            <span class="levo-panel-badge" :class="infoA?.correct ? 'badge-ok' : 'badge-err'">
              {{ infoA?.correct ? '✓ ' + infoA.pred : '✗ ' + (infoA?.pred || '(无预测)') }}
            </span>
          </div>
          <div class="levo-heatmap-wrap">
            <canvas ref="canvasA" class="levo-canvas"
              @mousemove="onCanvasHover($event, 'A')" @mouseleave="hoverCell = null"></canvas>
            <img :src="imgPathA" style="display:none"
              ref="imgA" @load="drawHeatmap('A')" />
          </div>
          <!-- Layer semantic info -->
          <div class="levo-semantic">{{ semA?.text }}</div>
          <div v-if="hoverCell" class="levo-hover-info">
            cell [{{ hoverCell.row }},{{ hoverCell.col }}] attn={{ hoverCell.val?.toFixed(5) }}
          </div>
          <div class="levo-layer-info">
            <span class="levo-grid-badge" :class="semA?.isAbstract ? 'abstract' : (semA?.isCoarse ? 'coarse' : 'fine')">
              {{ gridLabelA }}
            </span>
            <span>heads: {{ layerA?.head_mode }}</span>
            <span>tokens: {{ layerA?.visual_token_count }}</span>
          </div>
        </div>

        <!-- Panel B -->
        <div class="card levo-panel" v-if="compareMode && dataB">
          <div class="levo-panel-hdr">
            <span class="levo-panel-model">{{ modelB }}</span>
            <span class="levo-panel-badge" :class="infoB?.correct ? 'badge-ok' : 'badge-err'">
              {{ infoB?.correct ? '✓ ' + infoB.pred : '✗ ' + (infoB?.pred || '(无预测)') }}
            </span>
          </div>
          <div class="levo-heatmap-wrap">
            <canvas ref="canvasB" class="levo-canvas"
              @mousemove="onCanvasHover($event, 'B')" @mouseleave="hoverCell = null"></canvas>
            <img :src="imgPathB" style="display:none"
              ref="imgB" @load="drawHeatmap('B')" />
          </div>
          <!-- Layer semantic info -->
          <div class="levo-semantic">{{ semB?.text }}</div>
          <div class="levo-layer-info">
            <span class="levo-grid-badge" :class="semB?.isAbstract ? 'abstract' : (semB?.isCoarse ? 'coarse' : 'fine')">
              {{ gridLabelB }}
            </span>
            <span>heads: {{ layerB?.head_mode }}</span>
            <span>tokens: {{ layerB?.visual_token_count }}</span>
          </div>
        </div>
      </div>

      <!-- ═══ Analytics: evolution + adj diff ═══ -->
      <div class="card levo-analytics">
        <div class="levo-analytics-header">
          <div class="section-title">注意力演化分析</div>
          <div class="levo-analytics-desc">
            追踪注意力从浅层到深层的聚焦/扩散过程。组件交界处（色带变化）通常伴随注意力模式的剧烈变化。
          </div>
        </div>

        <!-- Metric selector -->
        <div class="levo-metric-row">
          <button v-for="m in evoMetrics" :key="m.key" @click="evoMetric = m.key"
            :class="['levo-metric-btn', evoMetric === m.key && 'levo-metric-btn-on']">
            {{ m.label }}
          </button>
          <span class="levo-metric-info">{{ metricInfo }}</span>
        </div>

        <!-- Evolution chart -->
        <div class="levo-chart-section">
          <div class="levo-chart-title">演化曲线 {{ compareMode ? '（X轴归一化）' : '' }}</div>
          <svg ref="evoSvg" class="levo-chart-svg"></svg>
          <div class="levo-comp-legend">
            <template v-for="(comps, modelName) in allComponentLegends" :key="modelName">
              <span class="levo-legend-model">{{ modelName }}</span>
              <span v-for="c in comps" :key="c" class="levo-comp-dot" :style="{background: compColor(c)}"></span>
              <span v-for="c in comps" :key="c" class="levo-legend-comp">{{ c.replace(/_/g, ' ') }}</span>
            </template>
          </div>
        </div>

        <!-- Adjacent layer diff -->
        <div class="levo-chart-section">
          <div class="levo-chart-title">相邻层差异（L1 距离 + KL 散度）</div>
          <div class="levo-adj-desc">柱高 = 注意力分布变化幅度 | 红线 = KL 散度峰值 | 组件边界通常有最大变化</div>
          <svg ref="adjSvgA" class="levo-chart-svg"></svg>
          <svg v-if="compareMode && dataB" ref="adjSvgB" class="levo-chart-svg" style="margin-top:16px"></svg>
        </div>
      </div>
    </template>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import * as d3 from 'd3'

// ── state ──────────────────────────────────────────────────────────
const models = ref([]); const totalSamples = ref(0)
const modelA = ref(''); const modelB = ref('')
const sampleId = ref(''); const sampleList = ref([])
const sampleInfo = ref(null)
const dataA = ref(null); const dataB = ref(null)
const layerIdxA = ref(0); const layerIdxB = ref(0)
const playing = ref(false); const playSpeed = ref(500)
const compareMode = ref(false); const syncMode = ref(true)
const evoMetric = ref('concentration')
const hoverCell = ref(null)

const sparkSvgA = ref(null); const sparkSvgB = ref(null)
const evoSvg = ref(null); const adjSvgA = ref(null); const adjSvgB = ref(null)
const canvasA = ref(null); const canvasB = ref(null)
const imgA = ref(null); const imgB = ref(null)

let playTimer = null; let indexData = null

const evoMetrics = [
  { key: 'concentration', label: '集中度 ↑', info: '接近 1 = 注意力锁死在少数 patch，接近 0 = 均匀分布' },
  { key: 'entropy', label: 'Entropy ↓', info: '接近 1 = 均匀散布（没有重点），接近 0 = 极度集中' },
  { key: 'sparsity', label: 'Gini ↑', info: '接近 1 = 极少数 patch 垄断注意力（类似贫富差距）' },
  { key: 'top5', label: 'Top-5% ↑', info: '前 5% patch 占据的注意力质量占比，越高越集中' },
]
const metricInfo = computed(() => evoMetrics.find(m => m.key === evoMetric.value)?.info || '')

// ── computed ────────────────────────────────────────────────────────
const maxLayerA = computed(() => (dataA.value?.layers?.length || 1) - 1)
const maxLayerB = computed(() => (dataB.value?.layers?.length || 1) - 1)
const totalLayersA = computed(() => dataA.value?.layers?.length || 0)
const totalLayersB = computed(() => dataB.value?.layers?.length || 0)

function getLayer(data, idx) {
  if (!data?.layers) return null
  return data.layers[Math.min(idx, data.layers.length - 1)] || null
}

function curComponent(data, idx) { return getLayer(data, idx)?.component }

const infoA = computed(() => dataA.value ? { correct: dataA.value.correct, pred: dataA.value.prediction } : null)
const infoB = computed(() => dataB.value ? { correct: dataB.value.correct, pred: dataB.value.prediction } : null)
const layerA = computed(() => getLayer(dataA.value, layerIdxA.value))
const layerB = computed(() => getLayer(dataB.value, layerIdxB.value))
const imgPathA = computed(() => dataA.value?.image_path || '')
const imgPathB = computed(() => dataB.value?.image_path || '')
const semA = computed(() => compSemantic(layerA.value))
const semB = computed(() => compSemantic(layerB.value))
const gridLabelA = computed(() => gridLabel(layerA.value))
const gridLabelB = computed(() => gridLabel(layerB.value))

const componentsA = computed(() => {
  if (!dataA.value?.layers) return []
  return [...new Set(dataA.value.layers.map(l => l.component || 'unknown'))]
})
const componentsB = computed(() => {
  if (!dataB.value?.layers) return []
  return [...new Set(dataB.value.layers.map(l => l.component || 'unknown'))]
})
const allComponentLegends = computed(() => {
  const r = {}
  if (dataA.value) r[modelA.value] = componentsA.value
  if (dataB.value && compareMode.value) r[modelB.value] = componentsB.value
  return r
})

// ── load index ─────────────────────────────────────────────────────
async function loadIndex() {
  try {
    const r = await fetch('/src/data/layer_evo_index.json')
    indexData = await r.json()
    models.value = Object.keys(indexData.models).sort()
    totalSamples.value = indexData.total_samples
    if (models.value.length > 0) {
      modelA.value = models.value.includes('llava') ? 'llava' : models.value[0]
      modelB.value = models.value.includes('blip') ? 'blip' : models.value[models.value.length > 1 ? 1 : 0]
      populateSamples(modelA.value)
    }
  } catch (e) { console.error('Failed to load index', e) }
}

function populateSamples(model) {
  if (!indexData?.models[model]) { sampleList.value = []; return }
  const smps = indexData.models[model].samples
  sampleList.value = Object.entries(smps).map(([id, s]) => ({ id, ...s }))
  sampleList.value.sort((a, b) => (a.correct || 0) - (b.correct || 0) || String(a.image_id).localeCompare(String(b.image_id)))
}

function onModelAChange() { populateSamples(modelA.value); sampleId.value = ''; dataA.value = null }
function onModelBChange() { if (sampleId.value) loadSampleB() }
function onSampleChange() { loadSampleA(); if (compareMode.value) loadSampleB() }
function onCompareToggle() {
  if (compareMode.value && sampleId.value) { loadSampleB(); layerIdxB.value = layerIdxA.value }
  else dataB.value = null
}

// ── load sample ────────────────────────────────────────────────────
async function loadSampleA() {
  if (!sampleId.value || !modelA.value) return
  const r = await fetch(`/data/layer_evo/${modelA.value}/${sampleId.value}/result.json`)
  dataA.value = await r.json()
  layerIdxA.value = 0
  sampleInfo.value = { question: dataA.value.question, gt: dataA.value.ground_truth }
  await nextTick(); drawAll()
}
async function loadSampleB() {
  if (!sampleId.value || !modelB.value) return
  try {
    const r = await fetch(`/data/layer_evo/${modelB.value}/${sampleId.value}/result.json`)
    dataB.value = await r.json()
    layerIdxB.value = 0
    await nextTick(); drawAll()
  } catch (e) { console.error('load B failed', e); dataB.value = null }
}

function nextTick() { return new Promise(r => setTimeout(r, 0)) }

// ── color ──────────────────────────────────────────────────────────
function compColor(comp) {
  const m = { vision_tower: '#3b82f6', vision_encoder: '#3b82f6', language_model: '#ef4444',
    qformer_cross_attention: '#f59e0b', qformer_self_attention: '#8b5cf6', unknown: '#999' }
  return m[comp] || '#999'
}

// Determine if attention is image-mappable based on visual_representation
function isImageSpace(layer) {
  const vis = (layer?.visual_representation) || ''
  // These are abstract tokens, not spatially mapped to the image
  const abstract = ['qformer_query_tokens', 'language_query_prefix_tokens']
  if (abstract.some(kw => vis.includes(kw))) return false
  // These are image-space: patches, projected/merged vision tokens, vision encoder outputs
  if (vis.includes('patch') || vis.includes('vision_encoder_token') ||
      vis.includes('projected') || vis.includes('merged')) return true
  // Default: if it has a raw_visual_grid, assume image-space
  return !!(layer?.raw_visual_grid)
}

// Per-component semantic meaning (what this attention actually represents)
function compSemantic(layer) {
  const comp = (layer?.component) || 'unknown'
  const vis = layer?.visual_representation
  const agg = layer?.token_aggregation || ''
  const n = layer?.visual_token_count || 0
  const grid = layer?.raw_visual_grid
  const gh = grid?.length || 0
  const gw = grid?.[0]?.length || 0
  const imgSpace = isImageSpace(layer)

  const parts = []

  if (comp === 'vision_tower' || comp === 'vision_encoder') {
    if (agg.includes('cls')) parts.push('CLS token 自注意力')
    else parts.push('视觉 patch 自注意力')
    parts.push(`→ ${n} 个图像 patch`)
  } else if (comp === 'language_model') {
    if (imgSpace) parts.push('生成 token 跨注意力')
    else parts.push('语言 token 自注意力')
    parts.push(imgSpace ? `→ ${n} 个投影视觉特征` : `→ ${n} 个语言前缀 token`)
  } else if (comp?.includes('qformer_cross')) {
    parts.push('Q-Former query 跨注意力 → 视觉特征')
  } else if (comp?.includes('qformer_self')) {
    parts.push('Q-Former 自注意力 → learned queries')
  } else {
    parts.push('自注意力')
  }

  const isAbstract = !imgSpace

  return {
    text: parts.join(' '),
    isCoarse: gh <= 7 && comp !== 'vision_encoder' && comp !== 'vision_tower',
    isAbstract,
    isImageSpace: imgSpace,
  }
}

// Grid label with semantic context
function gridLabel(layer) {
  const grid = layer?.raw_visual_grid
  if (!grid?.length) return ''
  const gh = grid.length, gw = grid[0]?.length || 0
  const n = gh * gw
  const imgSpace = isImageSpace(layer)
  if (!imgSpace) return `抽象  ${gh}×${gw} (${n} tokens)`
  if (n <= 49) return `粗粒度 ${gh}×${gw} (${n} cells)`
  if (n <= 400) return `中粒度 ${gh}×${gw} (${n} cells)`
  return `细粒度 ${gh}×${gw} (${n} cells)`
}

// ── heatmap ────────────────────────────────────────────────────────
function drawHeatmap(which) {
  const canvas = which === 'A' ? canvasA.value : canvasB.value
  const img = which === 'A' ? imgA.value : imgB.value
  const layer = which === 'A' ? layerA.value : layerB.value
  if (!canvas || !img || !layer) return
  const grid = layer.raw_visual_grid
  if (!grid?.length) return
  const gh = grid.length, gw = grid[0]?.length || 0
  if (!gw) return

  const imgSpace = isImageSpace(layer)
  const maxW = compareMode.value ? 380 : 440
  const scale = Math.min(maxW / img.naturalWidth, 1)
  const w = img.naturalWidth * scale, h = img.naturalHeight * scale

  if (imgSpace) {
    // ── Image-space: overlay heatmap on image ──
    canvas.width = w; canvas.height = h
    const ctx = canvas.getContext('2d')
    ctx.drawImage(img, 0, 0, w, h)

    const flat = grid.flat()
    const gmax = Math.max(...flat), gmin = Math.min(...flat), range = gmax - gmin || 1
    const cellW = w / gw, cellH = h / gh

    for (let r = 0; r < gh; r++) {
      for (let c = 0; c < gw; c++) {
        const alpha = Math.pow((grid[r][c] - gmin) / range, 0.6) * 0.75
        ctx.fillStyle = `rgba(255, 50, 30, ${alpha})`
        ctx.fillRect(c * cellW, r * cellH, cellW, cellH)
      }
    }
    ctx.strokeStyle = 'rgba(255,255,255,0.1)'; ctx.lineWidth = 0.5
    for (let r = 0; r <= gh; r++) { ctx.beginPath(); ctx.moveTo(0, r * cellH); ctx.lineTo(w, r * cellH); ctx.stroke() }
    for (let c = 0; c <= gw; c++) { ctx.beginPath(); ctx.moveTo(c * cellW, 0); ctx.lineTo(c * cellW, h); ctx.stroke() }
  } else {
    // ── Abstract space: render as standalone grid, image as dimmed context ──
    const gridW = Math.min(w, 320)
    const cellSize = gridW / Math.max(gw, gh)
    const gridH = cellSize * gh
    const totalW = gridW + 24  // grid + padding
    const totalH = Math.max(gridH + 40, 200)

    canvas.width = totalW; canvas.height = totalH
    const ctx = canvas.getContext('2d')
    ctx.fillStyle = '#1a1a2e'; ctx.fillRect(0, 0, totalW, totalH)

    const flat = grid.flat()
    const gmax = Math.max(...flat), gmin = Math.min(...flat), range = gmax - gmin || 1

    for (let r = 0; r < gh; r++) {
      for (let c = 0; c < gw; c++) {
        const val = (grid[r][c] - gmin) / range
        const alpha = 0.15 + val * 0.85
        ctx.fillStyle = `rgba(255, 50, 30, ${alpha})`
        ctx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize)
      }
    }

    // Grid lines + labels
    ctx.strokeStyle = 'rgba(255,255,255,0.3)'; ctx.lineWidth = 1
    for (let r = 0; r <= gh; r++) { ctx.beginPath(); ctx.moveTo(0, r * cellSize); ctx.lineTo(gridW, r * cellSize); ctx.stroke() }
    for (let c = 0; c <= gw; c++) { ctx.beginPath(); ctx.moveTo(c * cellSize, 0); ctx.lineTo(c * cellSize, gridH); ctx.stroke() }

    ctx.fillStyle = 'rgba(255,255,255,0.7)'; ctx.font = `${cellSize * 0.32}px sans-serif`
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle'
    for (let r = 0; r < gh; r++) {
      for (let c = 0; c < gw; c++) {
        ctx.fillText(`${r},${c}`, c * cellSize + cellSize/2, r * cellSize + cellSize/2)
      }
    }

    // Warning label
    ctx.fillStyle = 'rgba(255,255,255,0.5)'; ctx.font = '11px sans-serif'
    ctx.textAlign = 'center'
    ctx.fillText('⚠ 抽象嵌入空间 — 非图像坐标', gridW / 2, gridH + 20)
  }
}

function onCanvasHover(e, which) {
  const canvas = which === 'A' ? canvasA.value : canvasB.value
  const layer = which === 'A' ? layerA.value : layerB.value
  if (!canvas || !layer) return
  const grid = layer.raw_visual_grid
  if (!grid?.length) return
  const gh = grid.length, gw = grid[0].length
  const rect = canvas.getBoundingClientRect()
  const col = Math.floor(((e.clientX - rect.left) / rect.width) * gw)
  const row = Math.floor(((e.clientY - rect.top) / rect.height) * gh)
  if (row >= 0 && row < gh && col >= 0 && col < gw) hoverCell.value = { row, col, val: grid[row][col] }
}

// ── sparkline ──────────────────────────────────────────────────────
function drawSparkline(svgRef, data, layerIdx, modelLabel) {
  if (!svgRef || !data?.layers) return
  const layers = data.layers
  const svg = d3.select(svgRef); svg.selectAll('*').remove()
  const W = svgRef.clientWidth || 700
  const H = 40, ml = 4, mr = 4, mt = 3, mb = 3
  const w = W - ml - mr, h = H - mt - mb

  const vals = layers.map(l => {
    const g = l.raw_visual_grid
    if (!g?.length) return 0
    const flat = g.flat(); const total = flat.reduce((a, b) => a + b, 0) || 1
    const norm = flat.map(v => v / total)
    const ent = -norm.reduce((s, v) => s + (v > 0 ? v * Math.log(v) : 0), 0)
    return 1 - ent / Math.log(flat.length)
  })

  const x = d3.scaleLinear().domain([0, layers.length - 1]).range([0, w])
  const y = d3.scaleLinear().domain([d3.min(vals) * 0.9 || 0, d3.max(vals) * 1.05 || 1]).range([h, 0])
  const g = svg.append('g').attr('transform', `translate(${ml},${mt})`)

  // Component bands
  let prevComp = null, bandStart = 0
  layers.forEach((l, i) => {
    const comp = l.component || 'unknown'
    if (comp !== prevComp) {
      if (prevComp !== null) {
        g.append('rect').attr('x', x(bandStart)).attr('y', 0)
          .attr('width', x(i) - x(bandStart)).attr('height', h)
          .attr('fill', compColor(prevComp)).attr('opacity', 0.12).attr('rx', 2)
        // Label
        const bw = x(i) - x(bandStart)
        if (bw > 30) {
          g.append('text').attr('x', x(bandStart) + bw / 2).attr('y', h / 2)
            .attr('text-anchor', 'middle').attr('dominant-baseline', 'middle')
            .attr('font-size', 9).attr('fill', compColor(prevComp)).attr('opacity', 0.6)
            .text(prevComp.replace(/_/g, ' '))
        }
      }
      bandStart = i; prevComp = comp
    }
  })
  if (prevComp !== null) {
    g.append('rect').attr('x', x(bandStart)).attr('y', 0)
      .attr('width', x(layers.length - 1) - x(bandStart) + 2).attr('height', h)
      .attr('fill', compColor(prevComp)).attr('opacity', 0.12).attr('rx', 2)
  }

  // Line
  const line = d3.line().x((d, i) => x(i)).y(d => y(d)).curve(d3.curveMonotoneX)
  g.append('path').datum(vals).attr('d', line).attr('fill', 'none')
    .attr('stroke', '#3b82f6').attr('stroke-width', 1.5)

  // Current marker
  const ci = Math.round(layerIdx)
  g.append('circle').attr('cx', x(ci)).attr('cy', y(vals[ci] || 0))
    .attr('r', 4).attr('fill', '#ef4444').attr('stroke', '#fff').attr('stroke-width', 1.5)
}

function onSparkClick(e, which) {
  const svgRef = which === 'A' ? sparkSvgA.value : sparkSvgB.value
  const data = which === 'A' ? dataA.value : dataB.value
  if (!svgRef || !data?.layers) return
  const rect = svgRef.getBoundingClientRect()
  const ratio = (e.clientX - rect.left) / rect.width
  const idx = Math.round(ratio * (data.layers.length - 1))
  if (which === 'A') layerIdxA.value = Math.max(0, Math.min(data.layers.length - 1, idx))
  else layerIdxB.value = Math.max(0, Math.min(data.layers.length - 1, idx))
  drawAll()
}

// ── evolution chart (richer, larger, with gradient and annotations) ─
function drawEvoChart() {
  if (!evoSvg.value || !dataA.value?.layers) return
  const layersA = dataA.value.layers
  const layersB = dataB.value?.layers
  const svg = d3.select(evoSvg.value); svg.selectAll('*').remove()

  const W = evoSvg.value.clientWidth || 700
  const H = 280, ml = 52, mr = 20, mt = 14, mb = 48
  const w = W - ml - mr, h = H - mt - mb

  const compute = (l, metric) => {
    const g = l.raw_visual_grid
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

  const valsA = layersA.map(l => compute(l, evoMetric.value))
  const valsB = layersB?.map(l => compute(l, evoMetric.value)) || null
  const useNorm = compareMode.value && valsB

  const xDomain = useNorm ? [0, 100] : [0, layersA.length - 1]
  const x = d3.scaleLinear().domain(xDomain).range([0, w])
  const allVals = valsB ? [...valsA, ...valsB] : valsA
  const yMin = d3.min(allVals) * 0.9, yMax = d3.max(allVals) * 1.08
  const y = d3.scaleLinear().domain([yMin, yMax]).range([h, 0])
  const g = svg.append('g').attr('transform', `translate(${ml},${mt})`)

  // ── Component background bands for model A ──
  let prevComp = null, bandStart = 0
  const xForBand = (i) => useNorm ? x((i / (layersA.length - 1)) * 100) : x(i)
  layersA.forEach((l, i) => {
    const comp = l.component || 'unknown'
    if (comp !== prevComp) {
      if (prevComp !== null) {
        const bx = xForBand(bandStart), bw = xForBand(i) - bx
        g.append('rect').attr('x', bx).attr('y', 0).attr('width', bw).attr('height', h)
          .attr('fill', compColor(prevComp)).attr('opacity', 0.07).attr('rx', 3)
        if (bw > 50) {
          g.append('text').attr('x', bx + bw/2).attr('y', 12)
            .attr('text-anchor', 'middle').attr('font-size', 10).attr('fill', compColor(prevComp))
            .attr('opacity', 0.7).attr('font-weight', 600)
            .text(prevComp.replace(/_/g, ' '))
        }
      }
      bandStart = i; prevComp = comp
    }
  })
  if (prevComp !== null) {
    const bx = xForBand(bandStart), bw = xForBand(layersA.length - 1) - bx + 3
    g.append('rect').attr('x', bx).attr('y', 0).attr('width', bw).attr('height', h)
      .attr('fill', compColor(prevComp)).attr('opacity', 0.07).attr('rx', 3)
  }

  // ── Reference line for entropy/concentration ──
  if (evoMetric.value === 'entropy' || evoMetric.value === 'concentration') {
    const refY = evoMetric.value === 'entropy' ? y(1.0) : y(0)
    if (refY > 0 && refY < h) {
      g.append('line').attr('x1', 0).attr('x2', w).attr('y1', refY).attr('y2', refY)
        .attr('stroke', '#999').attr('stroke-width', 0.8).attr('stroke-dasharray', '5,5')
      g.append('text').attr('x', w).attr('y', refY - 4).attr('text-anchor', 'end')
        .attr('font-size', 9).attr('fill', '#999')
        .text(evoMetric.value === 'entropy' ? '均匀' : '零集中')
    }
  }

  // ── Axes ──
  g.append('g').attr('transform', `translate(0,${h})`)
    .call(d3.axisBottom(x).ticks(useNorm ? 10 : Math.min(layersA.length, 12)).tickFormat(d => useNorm ? d + '%' : d))
    .selectAll('text').style('font-size', '10px').style('fill', '#999')
  g.append('g').call(d3.axisLeft(y).ticks(5).tickFormat(d3.format('.3f')))
    .selectAll('text').style('font-size', '10px').style('fill', '#999')
  g.selectAll('.domain').style('stroke', '#e0e0e0')
  g.selectAll('.tick line').style('stroke', '#f0f0f0')

  // ── X/Y axis labels ──
  g.append('text').attr('x', w/2).attr('y', h + 34).attr('text-anchor', 'middle')
    .attr('font-size', 11).attr('fill', '#888')
    .text(useNorm ? '层深度（归一化 %）' : '采样层索引')
  g.append('text').attr('x', -38).attr('y', h/2).attr('text-anchor', 'middle')
    .attr('transform', `rotate(-90, -38, ${h/2})`).attr('font-size', 11).attr('fill', '#888')
    .text(evoMetrics.find(m => m.key === evoMetric.value)?.label || '')

  // ── Gradient fill under line A ──
  const xA = useNorm ? (d, i) => x((i / (layersA.length - 1)) * 100) : (d, i) => x(i)
  const xB = useNorm && layersB ? (d, i) => x((i / (layersB.length - 1)) * 100) : (d, i) => x(i)

  const areaA = d3.area().x(xA).y0(h).y1(d => y(d)).curve(d3.curveMonotoneX)
  const gradA = svg.append('defs').append('linearGradient').attr('id', 'gradA')
  gradA.append('stop').attr('offset', '0%').attr('stop-color', '#3b82f6').attr('stop-opacity', 0.25)
  gradA.append('stop').attr('offset', '100%').attr('stop-color', '#3b82f6').attr('stop-opacity', 0.02)
  g.append('path').datum(valsA).attr('d', areaA).attr('fill', 'url(#gradA)')

  // Line A
  const lineA = d3.line().x(xA).y(d => y(d)).curve(d3.curveMonotoneX)
  g.append('path').datum(valsA).attr('d', lineA).attr('fill', 'none')
    .attr('stroke', '#3b82f6').attr('stroke-width', 2.2)
  g.selectAll('.dot-a').data(valsA).enter()
    .append('circle').attr('cx', xA).attr('cy', d => y(d)).attr('r', 2.5).attr('fill', '#3b82f6')

  // Line B
  if (valsB && layersB) {
    const areaB = d3.area().x(xB).y0(h).y1(d => y(d)).curve(d3.curveMonotoneX)
    const gradB = svg.append('defs').append('linearGradient').attr('id', 'gradB')
    gradB.append('stop').attr('offset', '0%').attr('stop-color', '#f59e0b').attr('stop-opacity', 0.18)
    gradB.append('stop').attr('offset', '100%').attr('stop-color', '#f59e0b').attr('stop-opacity', 0.02)
    g.append('path').datum(valsB).attr('d', areaB).attr('fill', 'url(#gradB)')
    const lineB = d3.line().x(xB).y(d => y(d)).curve(d3.curveMonotoneX)
    g.append('path').datum(valsB).attr('d', lineB).attr('fill', 'none')
      .attr('stroke', '#f59e0b').attr('stroke-width', 2.2).attr('stroke-dasharray', '5,4')
    g.selectAll('.dot-b').data(valsB).enter()
      .append('circle').attr('cx', xB).attr('cy', d => y(d)).attr('r', 2.5).attr('fill', '#f59e0b')
  }

  // ── Current layer markers ──
  const mxA = useNorm ? x((layerIdxA.value / (layersA.length - 1)) * 100) : x(layerIdxA.value)
  g.append('line').attr('x1', mxA).attr('x2', mxA).attr('y1', 0).attr('y2', h)
    .attr('stroke', '#ef4444').attr('stroke-width', 2).attr('stroke-dasharray', '4,3')
  g.append('circle').attr('cx', mxA).attr('cy', y(valsA[layerIdxA.value] || 0))
    .attr('r', 5).attr('fill', '#ef4444').attr('stroke', '#fff').attr('stroke-width', 2)
  if (valsB && layersB) {
    const mxB = useNorm ? x((layerIdxB.value / (layersB.length - 1)) * 100) : x(layerIdxB.value)
    g.append('line').attr('x1', mxB).attr('x2', mxB).attr('y1', 0).attr('y2', h)
      .attr('stroke', '#f59e0b').attr('stroke-width', 1.5).attr('stroke-dasharray', '4,3')
  }

  // ── Legend ──
  const ly = h + 24
  g.append('rect').attr('x', 0).attr('y', ly - 8).attr('width', 14).attr('height', 3).attr('fill', '#3b82f6').attr('rx', 1)
  g.append('text').attr('x', 18).attr('y', ly).attr('font-size', 11).attr('fill', '#3b82f6').attr('font-weight', 600).text(modelA.value)
  if (valsB) {
    g.append('line').attr('x1', 100).attr('x2', 114).attr('y1', ly - 6).attr('y2', ly - 6)
      .attr('stroke', '#f59e0b').attr('stroke-width', 2).attr('stroke-dasharray', '4,3')
    g.append('text').attr('x', 118).attr('y', ly).attr('font-size', 11).attr('fill', '#f59e0b').attr('font-weight', 600).text(modelB.value)
  }
}

// ── adjacent diff (improved: L1 bars + KL line) ────────────────────
function drawAdjChart(svgRef, data, modelLabel) {
  if (!svgRef || !data) return
  const diffs = data.adjacent_layer_differences
  if (!diffs?.length) return
  const svg = d3.select(svgRef); svg.selectAll('*').remove()
  const W = svgRef.clientWidth || 700
  const H = 160, ml = 48, mr = 48, mt = 12, mb = 36
  const w = W - ml - mr, h = H - mt - mb

  const x = d3.scaleBand().domain(d3.range(diffs.length)).range([0, w]).padding(0.25)
  const yL1 = d3.scaleLinear().domain([0, d3.max(diffs, d => d.l1_sum || 0) * 1.15]).range([h, 0])
  const yKL = d3.scaleLinear().domain([0, d3.max(diffs, d => d.symmetric_kl || 0) * 1.2]).range([h, 0])

  const g = svg.append('g').attr('transform', `translate(${ml},${mt})`)

  // ── Component background ──
  let prevComp = null, bandStart = 0
  const layers = data.layers
  diffs.forEach((d, i) => {
    const comp = layers[d.left_layer_index]?.component || 'unknown'
    if (comp !== prevComp) {
      if (prevComp !== null) {
        const bx = x(bandStart) - x.bandwidth()/2, bw = x(i) - x(bandStart)
        g.append('rect').attr('x', bx).attr('y', 0).attr('width', bw).attr('height', h)
          .attr('fill', compColor(prevComp)).attr('opacity', 0.06).attr('rx', 2)
        if (bw > 40) {
          g.append('text').attr('x', bx + bw/2).attr('y', 10).attr('text-anchor', 'middle')
            .attr('font-size', 9).attr('fill', compColor(prevComp)).attr('opacity', 0.6)
            .text(prevComp.replace(/_/g, ' '))
        }
      }
      bandStart = i; prevComp = comp
    }
  })

  // ── Axes ──
  g.append('g').attr('transform', `translate(0,${h})`)
    .call(d3.axisBottom(x).tickValues(d3.range(0, diffs.length, Math.max(1, Math.ceil(diffs.length / 10)))))
    .selectAll('text').style('font-size', '9px').style('fill', '#aaa')
  g.append('g').call(d3.axisLeft(yL1).ticks(4).tickFormat(d3.format('.2f')))
    .selectAll('text').style('font-size', '9px').style('fill', '#3b82f6')
  g.append('g').call(d3.axisRight(yKL).ticks(4).tickFormat(d3.format('.2f')))
    .attr('transform', `translate(${w},0)`)
    .selectAll('text').style('font-size', '9px').style('fill', '#ef4444')
  g.selectAll('.domain').style('stroke', '#e0e0e0')
  g.selectAll('.tick line').style('stroke', '#f5f5f5')

  // ── L1 bars ──
  g.selectAll('.adj-bar').data(diffs).enter().append('rect')
    .attr('x', (d, i) => x(i)).attr('y', d => yL1(d.l1_sum || 0))
    .attr('width', x.bandwidth()).attr('height', d => h - yL1(d.l1_sum || 0))
    .attr('fill', (d) => compColor(layers[d.left_layer_index]?.component))
    .attr('opacity', 0.65).attr('rx', 2)
    .append('title').text(d => `层${d.left_layer_index}→${d.right_layer_index}\nL1: ${(d.l1_sum||0).toFixed(4)}\nKL: ${(d.symmetric_kl||0).toFixed(4)}`)

  // ── KL line ──
  const klLine = d3.line().x((d, i) => x(i) + x.bandwidth()/2).y(d => yKL(d.symmetric_kl || 0)).curve(d3.curveMonotoneX)
  g.append('path').datum(diffs).attr('d', klLine).attr('fill', 'none')
    .attr('stroke', '#ef4444').attr('stroke-width', 1.8)
  g.selectAll('.kl-dot').data(diffs).enter()
    .append('circle').attr('cx', (d, i) => x(i) + x.bandwidth()/2).attr('cy', d => yKL(d.symmetric_kl || 0))
    .attr('r', 2.5).attr('fill', '#ef4444')
    .append('title').text(d => `KL sym: ${(d.symmetric_kl||0).toFixed(4)}`)

  // ── Annotate max L1 ──
  const maxL1 = d3.max(diffs, d => d.l1_sum || 0)
  const maxIdx = diffs.findIndex(d => (d.l1_sum || 0) === maxL1)
  if (maxIdx >= 0) {
    const cx = x(maxIdx) + x.bandwidth()/2, cy = yL1(maxL1)
    g.append('text').attr('x', cx).attr('y', cy - 8).attr('text-anchor', 'middle')
      .attr('font-size', 9).attr('fill', '#ef4444').attr('font-weight', 600)
      .text(`max ${maxL1.toFixed(2)}`)
  }

  // ── Labels ──
  g.append('text').attr('x', w/2).attr('y', h + 26).attr('text-anchor', 'middle')
    .attr('font-size', 11).attr('fill', '#888').text('相邻层对（左→右）')
  g.append('text').attr('x', -34).attr('y', -4).attr('font-size', 10).attr('fill', '#3b82f6').text('L1')
  g.append('text').attr('x', w + 36).attr('y', -4).attr('font-size', 10).attr('fill', '#ef4444').text('KL')
  g.append('text').attr('x', 0).attr('y', -4).attr('font-size', 10).attr('fill', '#888').text(modelLabel)
}

// ── navigation ─────────────────────────────────────────────────────
function incLayer(which) {
  if (which === 'A' && layerIdxA.value < maxLayerA.value) { layerIdxA.value++; syncBfromA() }
  else if (which === 'B' && layerIdxB.value < maxLayerB.value) { layerIdxB.value++; syncAfromB() }
  drawAll()
}
function decLayer(which) {
  if (which === 'A' && layerIdxA.value > 0) { layerIdxA.value--; syncBfromA() }
  else if (which === 'B' && layerIdxB.value > 0) { layerIdxB.value--; syncAfromB() }
  drawAll()
}
function onSlide(which) {
  if (which === 'A') syncBfromA()
  else syncAfromB()
  drawAll()
}

function syncBfromA() {
  if (!syncMode.value || !dataB.value?.layers) return
  const pct = maxLayerA.value > 0 ? layerIdxA.value / maxLayerA.value : 0
  layerIdxB.value = Math.round(pct * maxLayerB.value)
}
function syncAfromB() {
  if (!syncMode.value || !dataA.value?.layers) return
  const pct = maxLayerB.value > 0 ? layerIdxB.value / maxLayerB.value : 0
  layerIdxA.value = Math.round(pct * maxLayerA.value)
}

function togglePlay() {
  playing.value = !playing.value
  if (playing.value) playLoop()
  else clearTimeout(playTimer)
}
function playLoop() {
  if (!playing.value) return
  if (layerIdxA.value >= maxLayerA.value) { layerIdxA.value = 0; layerIdxB.value = 0 }
  else { layerIdxA.value++; syncBfromA() }
  drawAll()
  playTimer = setTimeout(playLoop, playSpeed.value)
}

// ── draw all ───────────────────────────────────────────────────────
function drawAll() {
  drawHeatmap('A')
  if (compareMode.value && dataB.value) drawHeatmap('B')
  drawSparkline(sparkSvgA.value, dataA.value, layerIdxA.value, modelA.value)
  if (compareMode.value && dataB.value) {
    drawSparkline(sparkSvgB.value, dataB.value, layerIdxB.value, modelB.value)
  }
  drawEvoChart()
  drawAdjChart(adjSvgA.value, dataA.value, modelA.value)
  if (compareMode.value && dataB.value) drawAdjChart(adjSvgB.value, dataB.value, modelB.value)
}

function onKey(e) {
  if (e.key === 'ArrowLeft') decLayer('A')
  if (e.key === 'ArrowRight') incLayer('A')
  if (e.key === ' ') { e.preventDefault(); togglePlay() }
}

// ── lifecycle ──────────────────────────────────────────────────────
onMounted(() => { loadIndex(); window.addEventListener('keydown', onKey) })
onUnmounted(() => { clearTimeout(playTimer); window.removeEventListener('keydown', onKey) })
</script>

<style scoped>
.levo-root { display: flex; flex-direction: column; gap: 12px; }
.levo-ctrl { padding: 14px 18px; }
.levo-ctrl-row { display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }
.levo-cg { display: flex; align-items: center; gap: 6px; }
.levo-lbl { font-size: 12px; color: #666; white-space: nowrap; }
.levo-sel { font-size: 12px; padding: 3px 8px; border: 1px solid #ddd; border-radius: 4px; max-width: 150px; }
.levo-sel-wide { max-width: 320px; }
.levo-qt { margin-top: 10px; font-size: 13px; }
.levo-qt-lbl { font-weight: 600; color: #555; }
.levo-gt { margin-left: 12px; font-size: 12px; color: #fff; background: #8b5cf6; padding: 1px 8px; border-radius: 3px; }

.levo-empty { text-align: center; padding: 60px 20px; color: #999; }
.levo-empty-icon { font-size: 48px; margin-bottom: 12px; }

.levo-nav-card { padding: 12px 18px; }
.levo-nav-row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.levo-timeline-label { font-size: 12px; font-weight: 600; color: #555; margin-bottom: 4px; }
.levo-btn { padding: 4px 10px; border: 1px solid #ddd; border-radius: 4px; background: #fff; cursor: pointer; font-size: 13px; }
.levo-btn:disabled { opacity: 0.3; cursor: default; }
.levo-btn-play { padding: 4px 10px; border: 1px solid #ef4444; border-radius: 4px; background: #fff; color: #ef4444; cursor: pointer; font-size: 14px; }
.levo-slider { flex: 1; min-width: 80px; accent-color: #3b82f6; }
.levo-layer-num { font-size: 13px; font-weight: 600; color: #333; min-width: 100px; text-align: center; }
.levo-layer-num-sm { font-size: 12px; font-weight: 600; color: #555; min-width: 60px; }
.levo-comp-tag { font-size: 11px; color: #fff; padding: 1px 8px; border-radius: 3px; white-space: nowrap; }
.levo-speed-lbl { font-size: 11px; color: #999; display: flex; align-items: center; gap: 4px; }
.levo-spark { width: 100%; height: 40px; cursor: pointer; margin-top: 4px; }
.levo-play-shared { display: flex; gap: 10px; align-items: center; margin-top: 8px; padding-top: 8px; border-top: 1px dashed #eee; }

.levo-panels { display: grid; grid-template-columns: 1fr; gap: 12px; }
.levo-panels-dual { grid-template-columns: 1fr 1fr; }

.levo-panel { padding: 14px; }
.levo-panel-hdr { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
.levo-panel-model { font-weight: 700; font-size: 14px; }
.levo-panel-badge { font-size: 12px; padding: 2px 8px; border-radius: 4px; }
.badge-ok { background: #dcfce7; color: #16a34a; }
.badge-err { background: #fef2f2; color: #dc2626; }

.levo-heatmap-wrap { position: relative; min-height: 120px; }
.levo-canvas { display: block; border-radius: 6px; border: 1px solid #eee; max-width: 100%; }
.levo-hover-info { font-size: 11px; color: #666; margin-top: 4px; }
.levo-layer-info { display: flex; gap: 14px; margin-top: 6px; font-size: 11px; color: #999; flex-wrap: wrap; align-items: center; }
.levo-semantic { font-size: 12px; color: #555; margin-top: 6px; font-style: italic; }
.levo-semantic-sm { font-size: 11px; color: #888; max-width: 260px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.levo-grid-badge { font-size: 10px; padding: 2px 8px; border-radius: 3px; font-weight: 600; }
.levo-grid-badge.fine { background: #dbeafe; color: #1d4ed8; }
.levo-grid-badge.coarse { background: #fef3c7; color: #92400e; }
.levo-grid-badge.abstract { background: #fce7f3; color: #9d174d; }

.levo-analytics { padding: 20px; }
.levo-analytics-header { margin-bottom: 14px; }
.levo-analytics-desc { font-size: 12px; color: #999; margin-top: 4px; line-height: 1.5; }
.levo-metric-row { display: flex; gap: 8px; align-items: center; margin-bottom: 18px; flex-wrap: wrap; }
.levo-metric-info { font-size: 11px; color: #aaa; margin-left: 8px; }
.levo-metric-btn { padding: 5px 14px; border: 1px solid #ddd; border-radius: 5px; background: #fff; cursor: pointer; font-size: 12px; color: #555; transition: all 0.15s; }
.levo-metric-btn:hover { border-color: #3b82f6; color: #3b82f6; }
.levo-metric-btn-on { background: #3b82f6; border-color: #3b82f6; color: #fff; font-weight: 600; }

.levo-chart-section { margin-top: 20px; padding-top: 16px; border-top: 1px solid #f0f0f0; }
.levo-chart-title { font-size: 13px; font-weight: 600; color: #555; margin-bottom: 8px; }
.levo-chart-svg { width: 100%; }
.levo-adj-desc { font-size: 11px; color: #aaa; margin-bottom: 8px; }

.levo-comp-legend { display: flex; gap: 6px; align-items: center; margin-top: 8px; flex-wrap: wrap; }
.levo-legend-model { font-size: 10px; color: #999; margin-right: 2px; font-weight: 600; }
.levo-legend-comp { font-size: 11px; color: #666; margin-right: 14px; }
.levo-comp-dot { width: 10px; height: 10px; border-radius: 2px; display: inline-block; margin-right: 2px; }

.levo-adj { padding: 16px; display: none; }
</style>
