<template>
  <div class="aac-root">
    <!-- Controls -->
    <div class="card aac-ctrl">
      <div class="aac-ctrl-row">
        <div class="aac-cg">
          <label class="aac-lbl">样本</label>
          <select v-model="selQid" class="aac-sel">
            <option v-for="q in questions" :key="q.qid" :value="q.qid">
              {{ q.hasDiff ? '★ ' : '' }}图{{ q.imgId }} · {{ q.qid }} · {{ q.entries.length }}模型
            </option>
          </select>
          <span v-if="curQ?.hasDiff" class="aac-diff-hint">有正确/错误对比</span>
        </div>
        <div class="aac-cg">
          <label class="aac-lbl">聚合来源</label>
          <div class="aac-rg">
            <label class="aac-rl"><input type="radio" v-model="aggType" value="both" /> 全部 token</label>
            <label class="aac-rl"><input type="radio" v-model="aggType" value="answer" /> 回答 token</label>
            <label class="aac-rl"><input type="radio" v-model="aggType" value="question" /> 问题 token</label>
          </div>
        </div>
        <div class="aac-cg">
          <label class="aac-lbl">层范围</label>
          <div class="aac-rg">
            <label class="aac-rl"><input type="radio" v-model="layerRange" value="last" /> 后 1/3 层</label>
            <label class="aac-rl"><input type="radio" v-model="layerRange" value="all" /> 全部层</label>
            <label class="aac-rl"><input type="radio" v-model="layerRange" value="top" /> 最后层</label>
          </div>
        </div>
        <div class="aac-cg">
          <label class="aac-lbl">亮度变换</label>
          <div class="aac-rg">
            <label class="aac-rl"><input type="radio" v-model="xform" value="sqrt" /> √</label>
            <label class="aac-rl"><input type="radio" v-model="xform" value="log" /> log</label>
            <label class="aac-rl"><input type="radio" v-model="xform" value="linear" /> 线性</label>
          </div>
        </div>
      </div>
      <div v-if="questionText" class="aac-qt">
        <span class="aac-qt-lbl">Q: </span>{{ questionText }}
      </div>
    </div>

    <!-- Panels -->
    <div v-if="selQid" class="aac-panels">
      <!-- Model panels -->
      <div
        v-for="p in panels"
        :key="p.model"
        class="card aac-panel"
        :class="p.data ? (p.correct ? 'aac-panel-ok' : 'aac-panel-err') : 'aac-panel-na'"
      >
        <div class="aac-ph">
          <span class="aac-pname">{{ p.model }}</span>
          <span v-if="p.data" class="aac-pbadge" :class="p.correct ? 'badge-ok' : 'badge-err'">
            {{ p.correct ? '✓ 正确' : '✗ 错误' }}
          </span>
        </div>
        <div v-if="p.loading" class="aac-load">加载中…</div>
        <template v-else-if="p.data && p.agg">
          <div class="aac-frame">
            <img
              :src="`/img/${p.data.image.img_id}.jpeg`"
              :width="PW" :height="p.dispH"
              class="aac-img"
              @error="e => e.target.style.display = 'none'"
            />
            <canvas
              :ref="el => { if (el) cmaps[p.model] = el }"
              class="aac-hm"
              :width="PW" :height="p.dispH"
            ></canvas>
          </div>
          <div class="aac-metrics">
            <div class="aac-metric">
              <span class="aac-ml">集中度</span>
              <span class="aac-mv">{{ p.concentration }}</span>
            </div>
            <div class="aac-metric">
              <span class="aac-ml">峰均比</span>
              <span class="aac-mv">{{ p.peakRatio }}</span>
            </div>
          </div>
        </template>
        <div v-else-if="!p.loading" class="aac-nodata">无视觉注意力数据</div>
      </div>

      <!-- Diff panel -->
      <div v-if="diffPair" class="card aac-panel aac-panel-diff">
        <div class="aac-ph">
          <span class="aac-pname">差异图</span>
          <span class="aac-psub">正确 − 错误（已归一化）</span>
        </div>
        <div class="aac-frame">
          <img
            :src="`/img/${diffPair.imgId}.jpeg`"
            :width="PW" :height="diffPair.dispH"
            class="aac-img"
            @error="e => e.target.style.display = 'none'"
          />
          <canvas
            ref="diffCv"
            class="aac-hm"
            :width="PW" :height="diffPair.dispH"
          ></canvas>
        </div>
        <div class="aac-diff-legend">
          <span style="color:#1d4ed8">◀ 错误模型更关注</span>
          <span style="color:#b91c1c">正确模型更关注 ▶</span>
        </div>
      </div>
    </div>

    <div v-if="!selQid" class="card aac-empty">
      <div class="aac-empty-icon">🔍</div>
      <div>选择样本以查看聚合注意力对比</div>
      <div class="aac-empty-hint">★ 标记的样本有跨模型正确/错误对比</div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, nextTick } from 'vue'
import * as d3 from 'd3'
import attnIndex from '../data/attn_index.json'

const PW = 260  // heatmap display width (px)

// ===== Build question index =====
const qidMap = new Map()
for (const s of attnIndex.samples) {
  if (!qidMap.has(s.question_id)) {
    qidMap.set(s.question_id, { qid: s.question_id, imgId: s.image_id, entries: [] })
  }
  qidMap.get(s.question_id).entries.push({
    model: s.model,
    file: s.file,
    correct: !!s.correct,
  })
}
// Annotate hasDiff, filter to multi-model only, sort diff-first
const questions = [...qidMap.values()]
  .filter(q => q.entries.length >= 2)
  .map(q => ({
    ...q,
    hasDiff: q.entries.some(e => e.correct) && q.entries.some(e => !e.correct),
  }))
  .sort((a, b) => b.hasDiff - a.hasDiff)

// ===== Reactive state =====
const selQid = ref(questions[0]?.qid || '')
const aggType = ref('both')
const layerRange = ref('last')
const xform = ref('sqrt')

const curQ = computed(() => qidMap.get(selQid.value))

// ===== File cache =====
const fileCache = {}
async function loadFile(file) {
  if (!fileCache[file]) {
    const r = await fetch(`/data/attn/${file}`)
    fileCache[file] = await r.json()
  }
  return fileCache[file]
}

// ===== Panels =====
const panels = ref([])
const cmaps = {}          // model → canvas element
const diffCv = ref(null)

// ===== Aggregation =====
function computeAggVec(data) {
  const tokens = []
  if (aggType.value !== 'question') tokens.push(...Object.values(data.answer || {}))
  if (aggType.value !== 'answer')   tokens.push(...Object.values(data.question || {}))

  // Keep only tokens that actually have vis_attn entries
  const valid = tokens.filter(t => t.vis_attn && Object.keys(t.vis_attn).length > 0)
  if (!valid.length) return null

  const allLayers = Object.keys(valid[0].vis_attn).sort(
    (a, b) => +a.split('_')[1] - +b.split('_')[1]
  )
  const nL = allLayers.length
  const useLayers =
    layerRange.value === 'last' ? allLayers.slice(Math.floor(nL * 2 / 3)) :
    layerRange.value === 'top'  ? allLayers.slice(-1) :
    allLayers

  // Per-token: average vis_attn across used layers
  const tokenVecs = valid.map(t => {
    const vecs = useLayers.map(l => t.vis_attn[l]).filter(v => v?.length)
    if (!vecs.length) return null
    const n = vecs[0].length
    const avg = new Float64Array(n)
    for (const v of vecs) for (let i = 0; i < n; i++) avg[i] += v[i] / vecs.length
    return avg
  }).filter(Boolean)

  if (!tokenVecs.length) return null

  // Aggregate across tokens: element-wise max
  const n = tokenVecs[0].length
  const result = new Float64Array(n)
  for (const v of tokenVecs) for (let i = 0; i < n; i++) result[i] = Math.max(result[i], v[i])
  return result
}

// ===== Metrics =====
function calcConcentration(raw) {
  const s = raw.reduce((a, b) => a + b, 0)
  if (!s) return '—'
  const ent = raw.reduce((acc, v) => {
    const p = v / s
    return acc - (p > 1e-12 ? p * Math.log(p) : 0)
  }, 0)
  return (1 - ent / Math.log(raw.length)).toFixed(3)
}

function calcPeakRatio(raw) {
  const mx = Math.max(...raw)
  const mn = raw.reduce((a, b) => a + b, 0) / raw.length
  return mn > 0 ? (mx / mn).toFixed(1) + 'x' : '—'
}

// ===== Rendering =====
function applyXform(arr) {
  if (xform.value === 'sqrt')   return arr.map(v => Math.sqrt(v))
  if (xform.value === 'log')    return arr.map(v => Math.log1p(v * 1000))
  return arr
}

function patchesFromAgg(agg, grid) {
  const { w: gW, h: gH } = grid
  const nP = gW * gH
  const full = Array.from(agg)
  const off = full.length > nP ? full.length - nP : 0
  return full.slice(off, off + nP)
}

function renderHeatmap(canvas, raw, colorFn) {
  if (!canvas || !raw.length) return
  const gW = canvas._gW, gH = canvas._gH
  const transformed = applyXform(raw)
  const mx = Math.max(...transformed) || 1

  const off = document.createElement('canvas')
  off.width = gW; off.height = gH
  const oc = off.getContext('2d')
  const img = oc.createImageData(gW, gH)
  const d = img.data
  for (let i = 0; i < gW * gH && i < transformed.length; i++) {
    const c = d3.rgb(colorFn(transformed[i] / mx))
    d[i*4]=c.r; d[i*4+1]=c.g; d[i*4+2]=c.b; d[i*4+3]=185
  }
  oc.putImageData(img, 0, 0)
  const ctx = canvas.getContext('2d')
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  ctx.imageSmoothingEnabled = true
  ctx.imageSmoothingQuality = 'high'
  ctx.drawImage(off, 0, 0, canvas.width, canvas.height)
}

function renderDiff(canvas, rawCorrect, rawWrong, gW, gH) {
  if (!canvas) return
  const mxC = Math.max(...rawCorrect) || 1
  const mxW = Math.max(...rawWrong) || 1
  const diff = rawCorrect.map((v, i) => v / mxC - rawWrong[i] / mxW)  // [-1, 1]
  const scale = d3.scaleSequential(d3.interpolateRdBu).domain([1, -1])  // red=correct>wrong, blue=wrong>correct

  const off = document.createElement('canvas')
  off.width = gW; off.height = gH
  const oc = off.getContext('2d')
  const img = oc.createImageData(gW, gH)
  const d = img.data
  for (let i = 0; i < gW * gH && i < diff.length; i++) {
    const c = d3.rgb(scale(diff[i]))
    d[i*4]=c.r; d[i*4+1]=c.g; d[i*4+2]=c.b; d[i*4+3]=210
  }
  oc.putImageData(img, 0, 0)
  const ctx = canvas.getContext('2d')
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  ctx.imageSmoothingEnabled = true
  ctx.imageSmoothingQuality = 'high'
  ctx.drawImage(off, 0, 0, canvas.width, canvas.height)
}

// ===== Diff pair computation =====
const diffPair = computed(() => {
  const loaded = panels.value.filter(p => p.data && p.agg)
  const correct = loaded.filter(p => p.correct)
  const wrong   = loaded.filter(p => !p.correct)
  if (!correct.length || !wrong.length) return null

  // Find a correct+wrong pair with matching grid size
  for (const c of correct) {
    for (const w of wrong) {
      const gc = c.data.image.grid, gw = w.data.image.grid
      if (gc.w === gw.w && gc.h === gw.h) {
        return {
          correct: c, wrong: w,
          imgId: c.data.image.img_id,
          dispH: c.dispH,
          grid: gc,
        }
      }
    }
  }
  return null  // no matching grid pair
})

const questionText = computed(() => {
  const p = panels.value.find(p => p.data)
  if (!p?.data) return ''
  return Object.values(p.data.question || {})
    .map(t => t.token)
    .join('')
    .replace(/\[CLS\]/g, '')
    .replace(/\[SEP\]/g, '')
    .trim()
})

// ===== Draw loop =====
function drawAll() {
  for (const p of panels.value) {
    if (!p.data || !p.agg) continue
    const canvas = cmaps[p.model]
    if (!canvas) continue
    const { w: gW, h: gH } = p.data.image.grid
    canvas._gW = gW
    canvas._gH = gH
    const raw = patchesFromAgg(p.agg, p.data.image.grid)
    renderHeatmap(canvas, raw, d3.interpolateYlOrRd)
  }

  if (diffPair.value && diffCv.value) {
    const { correct: cp, wrong: wp, grid } = diffPair.value
    const rawC = patchesFromAgg(cp.agg, cp.data.image.grid)
    const rawW = patchesFromAgg(wp.agg, wp.data.image.grid)
    diffCv.value._gW = grid.w
    diffCv.value._gH = grid.h
    renderDiff(diffCv.value, rawC, rawW, grid.w, grid.h)
  }
}

// ===== Load =====
async function loadAndRender() {
  const entries = curQ.value?.entries || []
  panels.value = entries.map(e => ({
    ...e, data: null, agg: null, loading: true,
    dispH: 200, concentration: '—', peakRatio: '—',
  }))

  await Promise.all(entries.map(async (e, idx) => {
    try {
      const data = await loadFile(e.file)
      const agg  = computeAggVec(data)
      const ratio = data.image.h / data.image.w
      const dispH = Math.round(PW * ratio)
      const raw = agg ? patchesFromAgg(agg, data.image.grid) : []
      panels.value[idx] = {
        ...panels.value[idx],
        data, agg, loading: false, dispH,
        concentration: raw.length ? calcConcentration(raw) : '—',
        peakRatio:     raw.length ? calcPeakRatio(raw)     : '—',
      }
    } catch {
      panels.value[idx] = { ...panels.value[idx], loading: false }
    }
  }))

  await nextTick()
  drawAll()
}

watch(selQid, loadAndRender, { immediate: true })
watch([aggType, layerRange, xform], async () => {
  // Recompute agg vecs from cached data, no re-fetch needed
  for (let i = 0; i < panels.value.length; i++) {
    const p = panels.value[i]
    if (!p.data) continue
    const agg = computeAggVec(p.data)
    const raw = agg ? patchesFromAgg(agg, p.data.image.grid) : []
    panels.value[i] = {
      ...p, agg,
      concentration: raw.length ? calcConcentration(raw) : '—',
      peakRatio:     raw.length ? calcPeakRatio(raw)     : '—',
    }
  }
  await nextTick()
  drawAll()
})

// Re-draw if diffPair becomes available after async load
watch(diffPair, async () => {
  await nextTick()
  if (diffPair.value && diffCv.value) {
    const { correct: cp, wrong: wp, grid } = diffPair.value
    const rawC = patchesFromAgg(cp.agg, cp.data.image.grid)
    const rawW = patchesFromAgg(wp.agg, wp.data.image.grid)
    diffCv.value._gW = grid.w
    diffCv.value._gH = grid.h
    renderDiff(diffCv.value, rawC, rawW, grid.w, grid.h)
  }
})
</script>

<style scoped>
.aac-root {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

/* Controls */
.aac-ctrl {
  padding: 16px 20px;
}
.aac-ctrl-row {
  display: flex;
  gap: 24px;
  align-items: flex-start;
  flex-wrap: wrap;
}
.aac-cg {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.aac-lbl {
  font-size: 11px;
  font-weight: 600;
  color: #6b7280;
  text-transform: uppercase;
  letter-spacing: 0.4px;
}
.aac-sel {
  padding: 5px 10px;
  border: 1px solid #d1d5db;
  border-radius: 6px;
  font-size: 13px;
  background: #fff;
  max-width: 340px;
  cursor: pointer;
}
.aac-rg {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}
.aac-rl {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 13px;
  color: #374151;
  cursor: pointer;
  user-select: none;
}
.aac-diff-hint {
  font-size: 11px;
  color: #d97706;
  font-weight: 600;
  margin-top: 2px;
}
.aac-qt {
  margin-top: 12px;
  font-size: 13px;
  color: #374151;
  padding-top: 10px;
  border-top: 1px solid #f0f0f0;
}
.aac-qt-lbl {
  font-weight: 600;
  color: #6b7280;
  margin-right: 4px;
}

/* Panels row */
.aac-panels {
  display: flex;
  gap: 16px;
  align-items: flex-start;
  flex-wrap: wrap;
}

.aac-panel {
  flex: 0 0 auto;
  padding: 14px;
  display: flex;
  flex-direction: column;
  gap: 10px;
  border-top: 3px solid transparent;
}
.aac-panel-ok  { border-top-color: #22c55e; }
.aac-panel-err { border-top-color: #ef4444; }
.aac-panel-diff { border-top-color: #8b5cf6; }
.aac-panel-na  { border-top-color: #d1d5db; opacity: 0.6; }

.aac-ph {
  display: flex;
  align-items: center;
  gap: 8px;
}
.aac-pname {
  font-size: 13px;
  font-weight: 700;
  color: #1f2937;
}
.aac-psub {
  font-size: 11px;
  color: #9ca3af;
}
.aac-pbadge {
  font-size: 11px;
  font-weight: 600;
  padding: 2px 8px;
  border-radius: 8px;
}
.badge-ok  { background: #dcfce7; color: #15803d; }
.badge-err { background: #fee2e2; color: #b91c1c; }

.aac-frame {
  position: relative;
  line-height: 0;
  border-radius: 6px;
  overflow: hidden;
  box-shadow: 0 1px 3px rgba(0,0,0,0.12);
}
.aac-img { display: block; }
.aac-hm {
  position: absolute;
  top: 0; left: 0;
  pointer-events: none;
}

.aac-metrics {
  display: flex;
  gap: 16px;
}
.aac-metric {
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.aac-ml {
  font-size: 10px;
  color: #9ca3af;
  font-weight: 500;
}
.aac-mv {
  font-size: 13px;
  font-weight: 600;
  color: #3b82f6;
  font-variant-numeric: tabular-nums;
}

.aac-diff-legend {
  display: flex;
  justify-content: space-between;
  font-size: 11px;
  font-weight: 500;
  padding: 0 2px;
}

.aac-load {
  font-size: 13px;
  color: #9ca3af;
  padding: 20px 0;
  text-align: center;
}
.aac-nodata {
  font-size: 12px;
  color: #d1d5db;
  padding: 12px 0;
  text-align: center;
}

/* Empty state */
.aac-empty {
  text-align: center;
  padding: 60px 24px;
  color: #9ca3af;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
}
.aac-empty-icon {
  font-size: 36px;
  margin-bottom: 4px;
}
.aac-empty-hint {
  font-size: 12px;
  color: #d97706;
}
</style>
