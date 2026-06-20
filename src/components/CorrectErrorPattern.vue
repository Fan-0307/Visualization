<template>
  <div>
    <div class="card">
      <div class="section-title">正确 vs 错误时的注意力模式</div>

      <!-- Model selector -->
      <div style="display:flex;gap:8px;margin-bottom:16px">
        <button v-for="m in models" :key="m" @click="selectedModel=m"
          :data-full-name="modelFullName(m)"
          :class="['model-btn', 'model-tooltip', selectedModel===m && 'model-btn-active']">{{ m }}</button>
      </div>

      <!-- Question type filter -->
      <div class="qt-filter-row">
        <button v-for="qt in qtOptions" :key="qt.value"
          @click="activeQt = qt.value"
          :class="['qt-chip', activeQt === qt.value && 'qt-chip-active']">
          {{ qt.label }} <span class="qt-chip-n">{{ qt.count }}</span>
        </button>
      </div>

      <div v-if="selectedModel" class="pattern-panels">
        <!-- Correct samples -->
        <div class="pattern-section">
          <div class="pattern-header">
            <span class="pattern-badge badge-correct">✓ 答对</span>
            <span class="pattern-n">{{ filteredCorrect.length }} 个样本</span>
          </div>
          <div class="thumb-scroll" @wheel.prevent="onWheel($event, 'correct')">
            <div v-for="sid in paginatedCorrect" :key="sid" class="thumb-wrap thumb-correct"
              @mouseenter="onHover(sid, $event)"
              @mouseleave="onLeave">
              <img :src="sampleImage(sid)" class="thumb"
                @error="e => e.target.style.opacity='0.15'" />
            </div>
            <div v-if="filteredCorrect.length === 0" class="thumb-empty">无</div>
          </div>
          <div v-if="filteredCorrect.length > pageSize" class="scroll-hint">
            滚轮翻页 · {{ correctPage.start + 1 }}-{{ Math.min(correctPage.end, filteredCorrect.length) }} / {{ filteredCorrect.length }}
          </div>
        </div>

        <!-- Wrong samples -->
        <div class="pattern-section">
          <div class="pattern-header">
            <span class="pattern-badge badge-wrong">✗ 答错</span>
            <span class="pattern-n">{{ filteredWrong.length }} 个样本</span>
          </div>
          <div class="thumb-scroll" @wheel.prevent="onWheel($event, 'wrong')">
            <div v-for="sid in paginatedWrong" :key="sid" class="thumb-wrap thumb-wrong"
              @mouseenter="onHover(sid, $event)"
              @mouseleave="onLeave">
              <img :src="sampleImage(sid)" class="thumb"
                @error="e => e.target.style.opacity='0.15'" />
            </div>
            <div v-if="filteredWrong.length === 0" class="thumb-empty">无</div>
          </div>
          <div v-if="filteredWrong.length > pageSize" class="scroll-hint">
            滚轮翻页 · {{ wrongPage.start + 1 }}-{{ Math.min(wrongPage.end, filteredWrong.length) }} / {{ filteredWrong.length }}
          </div>
        </div>
      </div>

      <!-- Hover preview popup: heatmap overlaid on original -->
      <Teleport to="body">
        <div v-if="hovered" class="thumb-preview" :style="previewStyle">
          <div class="preview-dual">
            <div class="preview-col">
              <div class="preview-label">原图</div>
              <img :src="sampleImage(hovered.sid)" class="preview-img"
                @error="e => e.target.style.opacity='0.15'" />
            </div>
            <div class="preview-col">
              <div class="preview-label">末层视觉证据</div>
              <img :src="modelPhoto(hovered.sid)" class="preview-img"
                @error="e => { e.target.style.display='none'; e.target.nextElementSibling.style.display='flex' }" />
              <div class="preview-hm-empty">暂无证据图</div>
            </div>
          </div>
          <div class="thumb-preview-info">
            <div class="thumb-preview-q">{{ hovered.question }}</div>
            <div class="thumb-preview-meta">
              答案: {{ hovered.gt }} · {{ hovered.model }} · {{ hovered.qt }}
            </div>
          </div>
        </div>
      </Teleport>
    </div>

    <!-- Distribution chart -->
    <div class="card" style="margin-top:16px">
      <div class="section-title">各问题类型答对/答错分布</div>
      <div style="overflow-x:auto"><svg ref="svgRef"></svg></div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, reactive } from 'vue'
import * as d3 from 'd3'
import { getPhotoUrl } from '../data/processedData'

const props = defineProps({ data: Array, models: Array })
const svgRef = ref(null)
const selectedModel = ref(null)
const activeQt = ref('all')
const pageSize = 48
const offsetC = ref(0)
const offsetW = ref(0)
const hovered = ref(null)

const previewX = ref(0)
const previewY = ref(0)

watch(() => props.models, ms => { if (ms?.length) selectedModel.value = ms[0] }, { immediate: true })

const modelKey = (label) =>
  props.data.find(d => d.model === label)?.model_key || label.toLowerCase()
const modelFullName = (label) =>
  props.data.find(d => d.model === label)?.model_full_name || label
const modelPhoto = (sampleId) => getPhotoUrl(modelKey(selectedModel.value), sampleId)
const sampleImage = (sampleId) => {
  const row = props.data.find(d => d.sample_id === sampleId && d.model === selectedModel.value)
  return row?.image_src || row?.image_src_fallback || ''
}

// Question type options with counts
const qtOptions = computed(() => {
  const rows = props.data.filter(d => d.model === selectedModel.value)
  const counts = {}
  rows.forEach(d => {
    const qt = d.question_type || 'other'
    counts[qt] = (counts[qt] || 0) + 1
  })
  const opts = [{ value: 'all', label: '全部', count: rows.length }]
  for (const [qt, n] of Object.entries(counts).sort((a, b) => b[1] - a[1])) {
    opts.push({ value: qt, label: qt, count: n })
  }
  return opts
})

// Filtered samples
const filteredCorrect = computed(() =>
  props.data.filter(d =>
    d.model === selectedModel.value &&
    d.correct &&
    (activeQt.value === 'all' || d.question_type === activeQt.value)
  ).map(d => d.sample_id)
)
const filteredWrong = computed(() =>
  props.data.filter(d =>
    d.model === selectedModel.value &&
    !d.correct &&
    (activeQt.value === 'all' || d.question_type === activeQt.value)
  ).map(d => d.sample_id)
)

// Pagination
const correctPage = computed(() => ({
  start: offsetC.value,
  end: offsetC.value + pageSize
}))
const wrongPage = computed(() => ({
  start: offsetW.value,
  end: offsetW.value + pageSize
}))
const paginatedCorrect = computed(() => filteredCorrect.value.slice(correctPage.value.start, correctPage.value.end))
const paginatedWrong = computed(() => filteredWrong.value.slice(wrongPage.value.start, wrongPage.value.end))

// Reset offset when filter changes
watch([selectedModel, activeQt], () => { offsetC.value = 0; offsetW.value = 0 })

function onWheel(e, type) {
  const list = type === 'correct' ? filteredCorrect.value : filteredWrong.value
  const cur = type === 'correct' ? offsetC.value : offsetW.value
  const maxOff = Math.max(0, list.length - pageSize)
  const delta = e.deltaY > 0 ? pageSize : -pageSize
  const next = Math.max(0, Math.min(maxOff, cur + delta))
  if (type === 'correct') offsetC.value = next
  else offsetW.value = next
}

// Hover preview
const previewStyle = computed(() => ({
  left: previewX.value + 'px',
  top: previewY.value + 'px',
}))

function onHover(sid, e) {
  const row = props.data.find(d => d.sample_id === sid && d.model === selectedModel.value)
  hovered.value = {
    sid,
    question: row?.question || '',
    gt: row?.ground_truth || '',
    model: row?.model || selectedModel.value,
    qt: row?.question_type || '',
  }
  previewX.value = Math.min(e.clientX + 16, window.innerWidth - 280)
  previewY.value = Math.min(e.clientY + 16, window.innerHeight - 260)
}

function onLeave() { hovered.value = null }

// Chart
const qtypes = computed(() => [...new Set(props.data.map(d => d.question_type))].sort())

function drawChart() {
  if (!svgRef.value || !selectedModel.value) return
  const qts = qtypes.value
  const W = Math.max(480, qts.length * 72 + 80)
  const H = 200, ml = 44, mb = 56, mt = 8, mr = 16
  const svg = d3.select(svgRef.value).attr('width', W).attr('height', H)
  svg.selectAll('*').remove()
  const g = svg.append('g').attr('transform', `translate(${ml},${mt})`)
  const w = W-ml-mr, h = H-mt-mb

  const maxN = Math.max(...qts.map(qt =>
    props.data.filter(d => d.model === selectedModel.value && d.question_type === qt).length
  ))
  const x = d3.scaleBand().domain(qts).range([0,w]).padding(0.3)
  const y = d3.scaleLinear().domain([0, maxN]).range([h,0]).nice()

  g.append('g').attr('transform',`translate(0,${h})`).call(d3.axisBottom(x).tickSize(0))
    .selectAll('text').attr('transform','rotate(-30)').style('text-anchor','end').style('font-size','11px').style('fill','#666')
  g.append('g').call(d3.axisLeft(y).ticks(4).tickSize(-w))
    .selectAll('line').style('stroke','#f0f0f0')
  g.selectAll('.domain').remove()

  qts.forEach(qt => {
    const rows = props.data.filter(d => d.model === selectedModel.value && d.question_type === qt)
    const correct = rows.filter(d => d.correct).length
    const wrong = rows.length - correct
    const bw = x.bandwidth()
    if (correct > 0)
      g.append('rect').attr('x',x(qt)).attr('y',y(correct)).attr('width',bw).attr('height',h-y(correct))
        .attr('fill','#22c55e').attr('opacity',0.8).attr('rx',3)
    if (wrong > 0)
      g.append('rect').attr('x',x(qt)).attr('y',y(correct+wrong)).attr('width',bw).attr('height',y(correct)-y(correct+wrong))
        .attr('fill','#ef4444').attr('opacity',0.8).attr('rx',3)
    g.append('text').attr('x',x(qt)+bw/2).attr('y',y(correct+wrong)-4)
      .attr('text-anchor','middle').attr('font-size',10).attr('fill','#555').text(rows.length)
  })

  svg.append('rect').attr('x',ml).attr('y',H-14).attr('width',10).attr('height',10).attr('fill','#22c55e').attr('rx',2)
  svg.append('text').attr('x',ml+14).attr('y',H-5).attr('font-size',11).attr('fill','#555').text('正确')
  svg.append('rect').attr('x',ml+52).attr('y',H-14).attr('width',10).attr('height',10).attr('fill','#ef4444').attr('rx',2)
  svg.append('text').attr('x',ml+66).attr('y',H-5).attr('font-size',11).attr('fill','#555').text('错误')
}

onMounted(drawChart)
watch([selectedModel, () => props.data], drawChart)
</script>

<style scoped>
.model-btn { padding: 5px 14px; border: 1px solid #e5e7eb; border-radius: 6px; cursor: pointer; background: #fff; color: #555; font-size: 13px; transition: all 0.15s; }
.model-btn:hover { border-color: #3b82f6; color: #3b82f6; }
.model-btn-active { background: #3b82f6; border-color: #3b82f6; color: #fff; font-weight: 600; }

/* Question type chips */
.qt-filter-row { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 18px; }
.qt-chip { padding: 3px 12px; border: 1px solid #e5e7eb; border-radius: 12px; cursor: pointer; background: #fff; color: #64748b; font-size: 12px; transition: all 0.15s; }
.qt-chip:hover { border-color: #94a3b8; color: #334155; }
.qt-chip-active { background: #1e293b; border-color: #1e293b; color: #fff; }
.qt-chip-n { opacity: 0.6; margin-left: 4px; }

/* Pattern panels */
.pattern-panels { display: flex; gap: 24px; flex-wrap: wrap; }
.pattern-section { flex: 1; min-width: 260px; }
.pattern-header { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
.pattern-badge { padding: 2px 10px; border-radius: 8px; font-size: 12px; font-weight: 600; }
.badge-correct { background: #dcfce7; color: #16a34a; }
.badge-wrong { background: #fee2e2; color: #dc2626; }
.pattern-n { font-size: 12px; color: #94a3b8; }

/* Thumbnail grid */
.thumb-scroll { display: flex; flex-wrap: wrap; gap: 4px; max-height: 220px; overflow-y: auto; padding: 2px; }
.thumb-wrap { border-radius: 5px; cursor: pointer; transition: transform 0.12s; line-height: 0; }
.thumb-wrap:hover { transform: scale(1.15); z-index: 5; position: relative; }
.thumb-wrap.thumb-correct { border: 2px solid #22c55e; }
.thumb-wrap.thumb-wrong { border: 2px solid #ef4444; }
.thumb { width: 48px; height: 48px; object-fit: cover; border-radius: 3px; display: block; }
.thumb-empty { font-size: 12px; color: #d1d5db; padding: 8px; }
.scroll-hint { font-size: 11px; color: #94a3b8; margin-top: 6px; text-align: right; }

/* Hover preview */
.thumb-preview {
  position: fixed;
  z-index: 100;
  background: rgba(15, 23, 42, 0.95);
  border: 1px solid rgba(255,255,255,0.1);
  border-radius: 12px;
  padding: 12px;
  width: 420px;
  box-shadow: 0 12px 32px rgba(0,0,0,0.4);
  pointer-events: none;
}
.preview-dual { display: flex; gap: 8px; margin-bottom: 8px; }
.preview-col { flex: 1; min-width: 0; }
.preview-label { font-size: 10px; color: #94a3b8; letter-spacing: 0.5px; margin-bottom: 4px; }
.preview-img { width: 100%; aspect-ratio: 1; object-fit: cover; border-radius: 6px; display: block; background: #0f172a; }
.preview-hm-empty { display: none; width: 100%; aspect-ratio: 1; border-radius: 6px; align-items: center; justify-content: center; font-size: 11px; color: #64748b; background: rgba(255,255,255,0.05); }
.thumb-preview-info { display: flex; flex-direction: column; gap: 4px; }
.thumb-preview-q { font-size: 13px; color: #f1f5f9; line-height: 1.4; font-weight: 500; }
.thumb-preview-meta { font-size: 11px; color: #94a3b8; }
</style>
