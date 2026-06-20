<template>
  <div>
    <!-- top: error matrix -->
    <div class="card">
      <div class="section-title">错误矩阵
        <span style="font-weight:400;color:#999;font-size:12px;margin-left:8px">点击格子查看末层视觉证据</span>
      </div>
      <div style="display:flex;gap:24px;align-items:flex-start">
        <div class="em-left">
          <div v-if="linkSource" class="link-breadcrumb" style="margin-top:0">
            <span class="lb-arrow">←</span>
            来自「模型对比」：
            <b v-if="linkModel">{{ linkModel }}</b>
            <b v-if="linkModel && questionTypeFilter !== 'all'"> · </b>
            <b v-if="questionTypeFilter !== 'all'">{{ questionTypeFilter }}</b>
            <button class="lb-clear" @click="clearLink">清除联动 ✕</button>
          </div>
          <div class="filter-row" style="flex-direction:column;gap:14px;margin-bottom:0">
            <label class="filter-field filter-field-lg">
              <span>问题类型</span>
              <select v-model="questionTypeFilter">
                <option value="all">All</option>
                <option v-for="qt in questionTypeOptions" :key="qt" :value="qt">{{ qt }}</option>
              </select>
            </label>
            <label class="filter-field filter-field-lg">
              <span>错误筛选</span>
              <select v-model="errorFilter">
                <option value="all">全部样本</option>
                <option value="any_error">至少一个模型错误</option>
                <option value="all_wrong">全部模型错误</option>
                <option value="partial_wrong">仅部分模型错误</option>
              </select>
            </label>
            <label class="filter-field filter-field-lg">
              <span>排序</span>
              <select v-model="sortMode">
                <option value="default">默认排序</option>
                <option value="wrong_desc">错误模型数量从高到低</option>
                <option value="divergence_desc">模型预测分歧度从高到低</option>
              </select>
            </label>
          </div>
        </div>
        <div class="em-right">
          <div style="display:flex;gap:12px;margin-bottom:10px;font-size:14px;color:#666">
            <span><i style="display:inline-block;width:14px;height:14px;background:#22c55e;border-radius:2px;vertical-align:middle;margin-right:4px"></i>正确</span>
            <span><i style="display:inline-block;width:14px;height:14px;background:#ef4444;border-radius:2px;vertical-align:middle;margin-right:4px"></i>错误</span>
            <span><i style="display:inline-block;width:14px;height:14px;border:2px solid #f97316;border-radius:2px;vertical-align:middle;margin-right:4px"></i>全错</span>
            <span><i style="display:inline-block;width:14px;height:14px;border:2px solid #3b82f6;border-radius:2px;vertical-align:middle;margin-right:4px"></i>选中</span>
            <span style="margin-left:auto;color:#999">{{ filteredSamples.length }} / {{ allSamples.length }} 个样本</span>
          </div>
          <div style="overflow:auto;max-height:420px"><svg ref="svgRef"></svg></div>
        </div>
      </div>
    </div>

    <!-- bottom: attention detail for selected sample -->
    <div class="card" style="margin-top:16px" ref="detailCard">
      <div class="section-title">
        末层视觉证据对比
      </div>
      <div v-if="!selected" style="color:#bbb;text-align:center;padding:40px;font-size:13px">
        点击上方矩阵中的格子查看末层视觉证据图
      </div>
      <template v-else>
        <div class="method-note">
          说明：这里展示的是最后一层视觉表征上的响应/证据代理，并非所有模型都等价于对原图像素的原始注意力。
          它适合辅助定位模型差异与典型错误案例，不作为严格因果归因。
        </div>
        <div class="sample-hero">
          <div class="sample-meta">
            <span>样本 {{ selected.sample_id }}</span>
            <span>{{ selectedStats?.questionType || 'unknown' }}</span>
          </div>
          <div class="sample-question">{{ selectedEntry?.question }}</div>
          <div class="sample-answer">
            标准答案 <b>{{ selectedEntry?.ground_truth || '—' }}</b>
          </div>
        </div>
        <div class="diagnosis-box">
          <div class="diagnosis-title">诊断摘要</div>
          <div class="diagnosis-text">
            样本 <b>{{ selected.sample_id }}</b> 属于 <b>{{ selectedStats?.questionType || 'unknown' }}</b> 类型。
            <template v-if="correctModels.length">
              {{ correctModels.join('、') }} 回答正确。
            </template>
            <template v-if="wrongModels.length">
              {{ wrongModels.join('、') }} 回答错误。
            </template>
            该样本有 <b>{{ selectedStats?.wrong || 0 }}</b> 个错误模型，预测分歧度为
            <b>{{ selectedStats?.divergence || 0 }}/{{ selectedStats?.total || models.length }}</b>。
            {{ diagnosisHint }}
            证据图用于辅助观察末层表征差异，不能直接解释为模型“真实看见”的像素区域。
          </div>
        </div>
        <div style="display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start">
          <!-- original image -->
          <div style="text-align:center">
            <div class="img-label">原图</div>
            <div class="img-pred">{{ selectedEntry?.ground_truth || '—' }}</div>
            <img :src="selectedEntry?.image_src"
              class="attn-img" style="border:2px solid #e5e7eb"
              @error="e => { if (!e.target._tried && selectedEntry?.image_src_fallback) { e.target._tried=true; e.target.src=selectedEntry.image_src_fallback } else { e.target.style.opacity='0.15' } }" />
            <div style="height:18px"></div>
          </div>
          <!-- per-model -->
          <div v-for="m in models" :key="m" style="text-align:center">
            <div class="img-label model-tooltip" :data-full-name="modelFullName(m)" :style="{color: modelCorrect(m) ? '#16a34a' : '#dc2626'}">
              {{ m }} {{ modelCorrect(m) ? '✓' : '✗' }}
            </div>
            <div class="img-pred">{{ modelPred(m) }}</div>
            <img :src="modelPhoto(m)"
              class="attn-img" :style="{border: selected.model===m ? '2px solid #3b82f6' : '2px solid #e5e7eb'}"
              @error="e => { e.target.style.display='none'; e.target.nextElementSibling.style.display='flex' }" />
            <div class="attn-placeholder">暂无证据图</div>
            <div class="hm-legend">
              <span class="hm-legend-label">低</span>
              <div class="hm-legend-bar"></div>
              <span class="hm-legend-label">高</span>
            </div>
          </div>
        </div>
      </template>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch, inject, nextTick } from 'vue'
import * as d3 from 'd3'
import { attentionData, getPhotoUrl } from '../data/processedData'

const broadcast = inject('broadcast', () => {})
const consume = inject('consume', () => null)

const props = defineProps({ data: Array, models: Array })
const svgRef = ref(null)
const detailCard = ref(null)
const selected = ref(null)
const questionTypeFilter = ref('all')
const errorFilter = ref('any_error')
const sortMode = ref('default')

// Linked drill-down state: the model/source carried over from the overview view.
const linkModel = ref(null)
const linkSource = ref(null)

function clearLink() {
  linkModel.value = null
  linkSource.value = null
}

const modelKey = (label) =>
  props.data.find(d => d.model === label)?.model_key || label.toLowerCase()

const allSamples = computed(() => [...new Set(props.data.map(d => d.sample_id))])
const questionTypeOptions = computed(() => {
  const preferred = ['how many', 'what', 'what color', 'where', 'yes/no', 'other']
  const actual = new Set(props.data.map(d => d.question_type).filter(Boolean))
  const ordered = preferred.filter(qt => actual.has(qt))
  const rest = [...actual].filter(qt => !preferred.includes(qt)).sort()
  return [...ordered, ...rest]
})

function normalizedAnswer(v) {
  return String(v || '').trim().toLowerCase()
}

const sampleStats = computed(() => {
  const stats = {}
  for (const sampleId of allSamples.value) {
    const rows = props.data.filter(d => d.sample_id === sampleId)
    const predictions = rows.map(d => normalizedAnswer(d.prediction)).filter(Boolean)
    stats[sampleId] = {
      sampleId,
      rows,
      total: rows.length,
      wrong: rows.filter(d => !d.correct).length,
      correct: rows.filter(d => d.correct).length,
      divergence: new Set(predictions).size,
      questionType: rows[0]?.question_type || 'unknown',
    }
  }
  return stats
})

const allWrongSamples = computed(() =>
  new Set(Object.values(sampleStats.value).filter(s => s.wrong === s.total).map(s => s.sampleId))
)

const filteredSamples = computed(() => {
  let samples = allSamples.value.filter(sampleId => {
    const s = sampleStats.value[sampleId]
    if (!s) return false
    if (questionTypeFilter.value !== 'all' && s.questionType !== questionTypeFilter.value) return false
    if (errorFilter.value === 'any_error' && s.wrong === 0) return false
    if (errorFilter.value === 'all_wrong' && s.wrong !== s.total) return false
    if (errorFilter.value === 'partial_wrong' && (s.wrong === 0 || s.wrong === s.total)) return false
    return true
  })

  if (sortMode.value === 'wrong_desc') {
    samples = samples.slice().sort((a, b) =>
      sampleStats.value[b].wrong - sampleStats.value[a].wrong || String(a).localeCompare(String(b))
    )
  } else if (sortMode.value === 'divergence_desc') {
    samples = samples.slice().sort((a, b) =>
      sampleStats.value[b].divergence - sampleStats.value[a].divergence ||
      sampleStats.value[b].wrong - sampleStats.value[a].wrong ||
      String(a).localeCompare(String(b))
    )
  }
  return samples
})

const selectedEntry = computed(() => selected.value
  ? attentionData[`${selected.value.model}__${selected.value.sample_id}`]
  : null)
const selectedRows = computed(() =>
  selected.value ? props.data.filter(d => d.sample_id === selected.value.sample_id) : []
)
const selectedStats = computed(() =>
  selected.value ? sampleStats.value[selected.value.sample_id] : null
)
const correctModels = computed(() => selectedRows.value.filter(d => d.correct).map(d => d.model))
const wrongModels = computed(() => selectedRows.value.filter(d => !d.correct).map(d => d.model))
const diagnosisHint = computed(() => {
  const s = selectedStats.value
  if (!s) return ''
  if (s.wrong === s.total) return '这是全错样本，适合作为困难样本案例。'
  if (s.wrong > 0 && s.divergence >= Math.max(3, s.total - 1)) {
    return '该样本模型分歧度较高，适合观察正确模型与错误模型的注意力区域差异。'
  }
  if (s.wrong > 0) return '这是部分模型出错样本，适合比较模型能力差异。'
  return '所有模型均回答正确，可作为稳定成功案例。'
})

const modelCorrect = (m) => props.data.find(d => d.model === m && d.sample_id === selected.value?.sample_id)?.correct
const modelPred = (m) => attentionData[`${m}__${selected.value?.sample_id}`]?.prediction || ''
const modelPhoto = (m) => getPhotoUrl(modelKey(m), selected.value?.sample_id)
const modelFullName = (m) => props.data.find(d => d.model === m)?.model_full_name || m

function redraw() {
  const samples = filteredSamples.value
  const ms = props.models
  const cell = 30, ml = 100, mt = 28, mb = 48, mr = 20
  const W = samples.length * cell + ml + mr
  const H = ms.length * cell + mt + mb

  const svg = d3.select(svgRef.value).attr('width', W).attr('height', H)
  svg.selectAll('*').remove()
  const g = svg.append('g').attr('transform', `translate(${ml},${mt})`)

  const x = d3.scaleBand().domain(samples).range([0, samples.length * cell]).padding(0.06)
  const y = d3.scaleBand().domain(ms).range([0, ms.length * cell]).padding(0.06)

  g.append('g').attr('transform', `translate(0,${ms.length * cell})`)
    .call(d3.axisBottom(x).tickSize(0))
    .selectAll('text').attr('transform','rotate(-35)').style('text-anchor','end').style('font-size','10px').style('fill','#555')
  g.append('g').call(d3.axisLeft(y).tickSize(0))
    .selectAll('text').style('font-size','14px')
    .style('fill', d => d === linkModel.value ? '#1d4ed8' : '#222')
    .style('font-weight', d => d === linkModel.value ? 700 : 400)
  g.selectAll('.domain').style('display','none')

  // Highlight the linked model's row (drilled in from the overview view).
  if (linkModel.value && y(linkModel.value) != null) {
    g.insert('rect', ':first-child')
      .attr('x', -ml + 4).attr('y', y(linkModel.value) - 1)
      .attr('width', samples.length * cell + ml - 8).attr('height', y.bandwidth() + 2)
      .attr('fill', 'rgba(59,130,246,0.10)').attr('rx', 3)
      .attr('stroke', 'rgba(59,130,246,0.45)').attr('stroke-width', 1)
  }

  const tooltip = d3.select('body').selectAll('.em-tip').data([0]).join('div')
    .attr('class','em-tip')
    .style('position','absolute').style('background','rgba(15,15,30,0.88)').style('color','#fff')
    .style('padding','7px 12px').style('border-radius','8px').style('font-size','12px')
    .style('pointer-events','none').style('opacity',0).style('line-height','1.6')

  const filtered = props.data.filter(d => samples.includes(d.sample_id))
  g.selectAll('rect').data(filtered).join('rect')
    .attr('x', d => x(d.sample_id)).attr('y', d => y(d.model))
    .attr('width', x.bandwidth()).attr('height', y.bandwidth())
    .attr('fill', d => d.correct ? '#22c55e' : '#ef4444')
    .attr('opacity', 0.82).attr('rx', 2)
    .attr('stroke', d => d.sample_id === selected.value?.sample_id ? '#3b82f6' : (allWrongSamples.value.has(d.sample_id) ? '#f97316' : 'none'))
    .attr('stroke-width', d => d.sample_id === selected.value?.sample_id ? 2 : 1.5)
    .style('cursor','pointer')
    .on('mouseover', function(e, d) {
      d3.select(this).attr('opacity', 1).attr('stroke','#1d4ed8').attr('stroke-width', 2)
      const s = sampleStats.value[d.sample_id]
      tooltip.style('opacity',1)
        .html(`<b>${d.model}</b> · ${d.sample_id}<br/><span style="color:#cbd5e1">${d.model_full_name || modelFullName(d.model)}</span><br/>${d.correct ? '✓ 正确' : '✗ 错误'}<br/>类型：${s?.questionType || 'unknown'} · 错误 ${s?.wrong || 0}/${s?.total || 0} · 分歧 ${s?.divergence || 0}<br/><span style="color:#93c5fd">▶ 点击查看四模型证据图对比</span>`)
        .style('left',(e.pageX+12)+'px').style('top',(e.pageY-28)+'px')
    })
    .on('mouseout', function(e, d) {
      d3.select(this).attr('opacity', 0.82)
        .attr('stroke', d.sample_id === selected.value?.sample_id ? '#3b82f6' : (allWrongSamples.value.has(d.sample_id) ? '#f97316' : 'none'))
        .attr('stroke-width', d.sample_id === selected.value?.sample_id ? 2 : 1.5)
      tooltip.style('opacity',0)
    })
    .on('click', (_, d) => {
      selected.value = d; broadcast(d.sample_id, d.model, 'matrix'); redraw()
      nextTick(() => detailCard.value?.scrollIntoView({ behavior: 'smooth', block: 'start' }))
    })
}

// Consume a pending cross-view selection (e.g. drilled in from the overview).
const pending = consume('matrix')
if (pending) {
  linkSource.value = 'compare'
  if (pending.questionType && questionTypeOptions.value.includes(pending.questionType)) {
    questionTypeFilter.value = pending.questionType
    errorFilter.value = 'all'   // overview drill-down should not be hidden by the error filter
  }
  if (pending.model) linkModel.value = pending.model
  if (pending.sampleId && sampleStats.value[pending.sampleId]) {
    selected.value = { sample_id: pending.sampleId, model: pending.model || props.models?.[0] }
    errorFilter.value = 'all'
  }
}

onMounted(redraw)
watch([() => props.data, questionTypeFilter, errorFilter, sortMode, linkModel], redraw)
</script>

<style scoped>
.link-breadcrumb { display: flex; align-items: center; gap: 6px; margin: 2px 0 12px; padding: 7px 12px; background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px; color: #1e3a8a; font-size: 12px; }
.link-breadcrumb b { color: #1d4ed8; }
.lb-arrow { font-weight: 700; color: #3b82f6; }
.lb-clear { margin-left: auto; border: 1px solid #bfdbfe; background: #fff; color: #1d4ed8; border-radius: 6px; padding: 3px 8px; font-size: 11px; cursor: pointer; }
.lb-clear:hover { background: #dbeafe; }
.filter-row { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin: 4px 0 12px; }
.filter-field { display: flex; align-items: center; gap: 6px; color: #475569; font-size: 12px; }
.filter-field select { height: 28px; border: 1px solid #dbe3ef; border-radius: 6px; background: #fff; color: #1e293b; padding: 0 8px; font-size: 12px; outline: none; }
.filter-field select:focus { border-color: #3b82f6; box-shadow: 0 0 0 2px rgba(59,130,246,0.12); }
.filter-field-lg { display: flex; align-items: center; gap: 8px; }
.filter-field-lg span { width: 5em; flex-shrink: 0; text-align: left; }
.filter-field-lg select { height: 28px; font-size: 12px; padding: 0 8px; flex: 1; width: 0; min-width: 140px; }
.em-left { flex: 0 0 480px; min-width: 0; display: flex; flex-direction: column; gap: 12px; }
.em-right { flex: 1; min-width: 0; }
.method-note { border: 1px solid #fde68a; background: #fffbeb; color: #92400e; border-radius: 8px; padding: 10px 12px; font-size: 12px; line-height: 1.6; margin-bottom: 12px; }
.sample-hero { border: 1px solid #e2e8f0; background: linear-gradient(180deg, #f8fafc, #fff); border-radius: 10px; padding: 16px 18px; margin-bottom: 12px; }
.sample-meta { display: flex; gap: 8px; flex-wrap: wrap; color: #64748b; font-size: 12px; font-weight: 600; margin-bottom: 8px; }
.sample-meta span { background: #e0f2fe; color: #0369a1; border-radius: 999px; padding: 3px 8px; }
.sample-question { color: #0f172a; font-size: 20px; line-height: 1.35; font-weight: 700; margin-bottom: 10px; }
.sample-answer { color: #334155; font-size: 14px; }
.sample-answer b { color: #0f172a; font-size: 16px; margin-left: 4px; }
.diagnosis-box { border: 1px solid #dbeafe; background: #eff6ff; border-radius: 8px; padding: 12px 14px; margin-bottom: 16px; }
.diagnosis-title { color: #1d4ed8; font-size: 12px; font-weight: 700; margin-bottom: 6px; }
.diagnosis-text { color: #1e293b; font-size: 13px; line-height: 1.7; }
.img-label { font-size: 12px; font-weight: 600; margin-bottom: 4px; }
.img-pred { font-size: 11px; color: #888; margin-bottom: 4px; max-width: 132px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.attn-img { width: 132px; height: 132px; object-fit: cover; border-radius: 8px; display: block; }
.attn-placeholder { width: 132px; height: 132px; background: #f3f4f6; border-radius: 8px; display: none; align-items: center; justify-content: center; font-size: 11px; color: #bbb; border: 1px solid #e5e7eb; }
.hm-legend { display: flex; align-items: center; gap: 4px; justify-content: center; margin-top: 4px; }
.hm-legend-label { font-size: 10px; color: #9ca3af; }
.hm-legend-bar { width: 72px; height: 6px; border-radius: 3px; background: linear-gradient(to right, #fff5cc, #f59e0b 50%, #dc2626); flex-shrink: 0; }
</style>
