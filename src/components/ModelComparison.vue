<template>
  <div>
    <div class="card">
      <div class="section-title">模型准确率对比
        <span class="drill-hint">点击模型行或问题类型 → 跳转「样本诊断」并自动筛选</span>
      </div>
      <div class="summary-grid">
        <div class="summary-card summary-best">
          <div class="summary-label">最佳模型</div>
          <div class="summary-main model-tooltip" :data-full-name="modelFullName(bestModel?.model)">{{ bestModel?.model || '—' }}</div>
          <div class="summary-sub">Overall {{ pct(bestModel?.value) }}</div>
        </div>
        <div class="summary-card summary-worst">
          <div class="summary-label">最弱模型</div>
          <div class="summary-main model-tooltip" :data-full-name="modelFullName(worstModel?.model)">{{ worstModel?.model || '—' }}</div>
          <div class="summary-sub">Overall {{ pct(worstModel?.value) }}</div>
        </div>
        <div class="summary-card summary-insight">
          <div class="summary-label">关键差异</div>
          <div class="summary-main">视觉细粒度问题拉开差距</div>
          <div class="summary-sub">
            Qwen2-VL/LLaVA 在 what color、who 上稳定，CLIP 在开放式问题上明显较弱。
          </div>
        </div>
      </div>
      <div style="margin-bottom:28px">
        <table class="acc-table">
          <thead>
            <tr>
              <th data-col="0" :class="{ 'col-hover': hoverCol === 0 }">Model</th>
              <th data-col="1" :class="{ 'col-hover': hoverCol === 1 }">Overall</th>
              <th v-for="(qt, ci) in qtypes" :key="qt" class="qt-head"
                :class="{ emphasis: emphasisTypes.has(qt), 'col-hover': hoverCol === ci + 2 }"
                :data-col="ci + 2"
                @click="drill({ questionType: qt })"
                title="点击按该问题类型筛选样本诊断">
                {{ qt }}
                <span v-if="emphasisTypes.has(qt)" class="type-mark">重点</span>
              </th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="m in models" :key="m" class="model-row" @click="drill({ model: m })"
              title="点击在样本诊断中高亮该模型">
              <td class="model-name"
                @mouseenter="hoverCol = 0" @mouseleave="hoverCol = -1"><span class="model-tooltip" :data-full-name="modelFullName(m)">{{ m }}</span></td>
              <td :style="{background:accColor(stats[m]?.overall)}"
                @mouseenter="hoverCol = 1" @mouseleave="hoverCol = -1">{{ pct(stats[m]?.overall) }}</td>
              <td v-for="(qt, ci) in qtypes" :key="qt"
                class="acc-cell"
                :class="{ emphasis: emphasisTypes.has(qt) }"
                :style="{background:accColor(stats[m]?.byType[qt])}"
                @click.stop="drill({ model: m, questionType: qt })"
                @mouseenter="hoverCol = ci + 2" @mouseleave="hoverCol = -1"
                title="点击按该模型+问题类型下钻">
                {{ pct(stats[m]?.byType[qt]) }}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <div style="display:flex;gap:16px;margin-top:16px;align-items:stretch">
      <div class="card" style="flex:1;min-width:0;margin-top:0">
        <div class="section-title">总体准确率</div>
        <svg ref="barRef" style="width:100%"></svg>
      </div>
      <div class="card" style="flex:2;min-width:0;margin-top:0">
        <div class="section-title">按问题类型准确率</div>
        <div style="overflow-x:auto"><svg ref="groupRef"></svg></div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch, inject } from 'vue'
import * as d3 from 'd3'

const props = defineProps({ data: Array })
const barRef = ref(null)
const groupRef = ref(null)

const broadcastFields = inject('broadcastFields', () => {})
const navigate = inject('navigate', () => {})
const hoverCol = ref(-1)

// Drill down into the sample-diagnosis view, carrying the clicked dimension(s).
function drill(fields) {
  broadcastFields(fields, 'compare')
  navigate('matrix')
}

const models = computed(() => [...new Set(props.data.map(d => d.model))])
const qtypes = computed(() => [...new Set(props.data.map(d => d.question_type))].sort())
const emphasisTypes = new Set(['yes/no', 'how many', 'what color'])

const stats = computed(() => {
  const s = {}
  for (const m of models.value) {
    const rows = props.data.filter(d => d.model === m)
    const overall = rows.filter(d => d.correct).length / rows.length
    const byType = {}
    for (const qt of qtypes.value) {
      const sub = rows.filter(d => d.question_type === qt)
      byType[qt] = sub.length ? sub.filter(d => d.correct).length / sub.length : null
    }
    s[m] = { overall, byType }
  }
  return s
})

const rankedModels = computed(() =>
  models.value
    .map(model => ({ model, value: stats.value[model]?.overall ?? null }))
    .filter(d => d.value != null)
    .sort((a, b) => b.value - a.value)
)
const bestModel = computed(() => rankedModels.value[0] || null)
const worstModel = computed(() => rankedModels.value[rankedModels.value.length - 1] || null)

function modelFullName(model) {
  if (!model) return ''
  return props.data.find(d => d.model === model)?.model_full_name || model
}

function pct(v) { return v == null ? '—' : (v * 100).toFixed(1) + '%' }
function accColor(v) {
  if (v == null) return ''
  const t = Math.max(0, Math.min(1, v))
  return `rgba(${Math.round(239*(1-t)+34*t)},${Math.round(68*(1-t)+197*t)},${Math.round(68*(1-t)+94*t)},0.22)`
}

const colors = ['#3b82f6','#10b981','#f59e0b','#ef4444','#8b5cf6']

function drawBar() {
  const ms = models.value
  const W = Math.max(340, (barRef.value?.parentElement?.clientWidth || 340) - 4)
  const H = 220, ml = 50, mb = 28, mt = 8, mr = 16
  const svg = d3.select(barRef.value).attr('width', W).attr('height', H)
  svg.selectAll('*').remove()
  const g = svg.append('g').attr('transform', `translate(${ml},${mt})`)
  const w = W-ml-mr, h = H-mt-mb
  const x = d3.scaleBand().domain(ms).range([0,w]).padding(0.35)
  const y = d3.scaleLinear().domain([0,1]).range([h,0])
  g.append('g').attr('transform',`translate(0,${h})`).call(d3.axisBottom(x).tickSize(0))
    .selectAll('text').style('font-size','12px').style('fill','#555')
  g.append('g').call(d3.axisLeft(y).ticks(4).tickFormat(d3.format('.0%')).tickSize(-w))
    .selectAll('line').style('stroke','#f0f0f0')
  g.select('.domain').remove()
  const barTip = d3.select('body').selectAll('.acc-tip').data([0]).join('div').attr('class','acc-tip')
    .style('position','absolute').style('background','rgba(15,23,42,0.92)').style('color','#fff')
    .style('padding','7px 10px').style('border-radius','8px').style('font-size','12px')
    .style('pointer-events','none').style('opacity',0).style('line-height','1.5')
  ms.forEach((m,i) => {
    const v = stats.value[m]?.overall ?? 0
    g.append('rect').attr('x',x(m)).attr('y',y(v)).attr('width',x.bandwidth()).attr('height',h-y(v))
      .attr('fill',colors[i%colors.length]).attr('rx',4).style('cursor','pointer')
      .on('mouseover',function(e){ d3.select(this).attr('opacity',0.85)
        barTip.style('opacity',1).html(`<b>${m}</b> · Overall ${pct(v)}<br/><span style="color:#93c5fd">▶ 点击 → 样本诊断 · 高亮该模型</span>`)
          .style('left',`${e.pageX+12}px`).style('top',`${e.pageY-34}px`) })
      .on('mousemove',e=>barTip.style('left',`${e.pageX+12}px`).style('top',`${e.pageY-34}px`))
      .on('mouseout',function(){ d3.select(this).attr('opacity',1); barTip.style('opacity',0) })
      .on('click',()=>{ barTip.style('opacity',0); drill({ model: m }) })
    g.append('text').attr('x',x(m)+x.bandwidth()/2).attr('y',y(v)-5)
      .attr('text-anchor','middle').attr('font-size',11).attr('fill','#444').text((v*100).toFixed(1)+'%')
  })
}

function drawGroup() {
  const ms = models.value, qts = qtypes.value
  const W = Math.max(380, qts.length * ms.length * 20 + 80, (groupRef.value?.parentElement?.clientWidth || 400) - 4)
  const H = 220, ml = 44, mb = 56, mt = 8, mr = 12
  const svg = d3.select(groupRef.value).attr('width', W).attr('height', H)
  svg.selectAll('*').remove()
  const g = svg.append('g').attr('transform', `translate(${ml},${mt})`)
  const w = W-ml-mr, h = H-mt-mb
  const x0 = d3.scaleBand().domain(qts).range([0,w]).padding(0.25)
  const x1 = d3.scaleBand().domain(ms).range([0,x0.bandwidth()]).padding(0.08)
  const y = d3.scaleLinear().domain([0,1]).range([h,0])
  g.append('g').attr('transform',`translate(0,${h})`).call(d3.axisBottom(x0).tickSize(0))
    .selectAll('text').attr('transform','rotate(-30)').style('text-anchor','end').style('font-size','11px').style('fill','#666')
  g.append('g').call(d3.axisLeft(y).ticks(4).tickFormat(d3.format('.0%')).tickSize(-w))
    .selectAll('line').style('stroke','#f0f0f0')
  g.selectAll('.domain').remove()

  const tooltip = d3.select('body').selectAll('.acc-tip').data([0]).join('div')
    .attr('class', 'acc-tip')
    .style('position', 'absolute')
    .style('background', 'rgba(15, 23, 42, 0.92)')
    .style('color', '#fff')
    .style('padding', '8px 10px')
    .style('border-radius', '8px')
    .style('font-size', '12px')
    .style('line-height', '1.5')
    .style('box-shadow', '0 8px 20px rgba(15,23,42,0.18)')
    .style('pointer-events', 'none')
    .style('opacity', 0)

  qts.forEach(qt => ms.forEach((m,i) => {
    const v = stats.value[m]?.byType[qt]
    if (v == null) return
    g.append('rect').attr('x',x0(qt)+x1(m)).attr('y',y(v))
      .attr('width',x1.bandwidth()).attr('height',h-y(v))
      .attr('fill',colors[i%colors.length]).attr('rx',3)
      .style('cursor', 'pointer')
      .on('mouseover', function(e) {
        d3.select(this)
          .attr('stroke', '#0f172a')
          .attr('stroke-width', 1.4)
          .attr('opacity', 0.88)
        tooltip.style('opacity', 1)
          .html(`<b>${m}</b><br/><span style="color:#cbd5e1">${modelFullName(m)}</span><br/>${qt}: ${pct(v)}<br/><span style="color:#93c5fd">▶ 点击 → 样本诊断 · 筛选「${qt}」</span>`)
          .style('left', `${e.pageX + 12}px`)
          .style('top', `${e.pageY - 34}px`)
      })
      .on('mousemove', function(e) {
        tooltip
          .style('left', `${e.pageX + 12}px`)
          .style('top', `${e.pageY - 34}px`)
      })
      .on('mouseout', function() {
        d3.select(this)
          .attr('stroke', null)
          .attr('stroke-width', null)
          .attr('opacity', 1)
        tooltip.style('opacity', 0)
      })
      .on('click', () => { tooltip.style('opacity', 0); drill({ model: m, questionType: qt }) })
  }))
  // legend below
  ms.forEach((m,i) => {
    svg.append('rect').attr('x',ml+i*76).attr('y',H-14).attr('width',10).attr('height',10).attr('fill',colors[i%colors.length]).attr('rx',2)
    svg.append('text').attr('x',ml+i*76+14).attr('y',H-5).attr('font-size',10).attr('fill','#555').text(m)
  })
}

function draw() { drawBar(); drawGroup() }
onMounted(draw)
watch(() => props.data, draw)
</script>

<style scoped>
.acc-table { border-collapse: collapse; font-size: 14px; width: 100%; }
.acc-table th { padding: 10px 16px; background: #f8fafc; border: 1px solid #e5e7eb; font-weight: 600; color: #374151; text-align: center; }
.acc-table th:first-child { text-align: center; }
.acc-table td { padding: 10px 16px; border: 1px solid #e5e7eb; text-align: center; font-size: 14px; }
.model-name { font-weight: 600; color: #1e293b; text-align: center !important; }
.acc-table th.col-hover { background: #dbeafe; color: #1d4ed8; }
.summary-grid { display: grid; grid-template-columns: 1fr 1fr 1.5fr; gap: 12px; margin-bottom: 16px; }
.summary-card { border: 1px solid #e5e7eb; border-radius: 8px; padding: 14px 16px; background: #fff; }
.summary-best { border-left: 4px solid #10b981; background: linear-gradient(90deg, rgba(16,185,129,0.08), #fff 45%); }
.summary-worst { border-left: 4px solid #ef4444; background: linear-gradient(90deg, rgba(239,68,68,0.08), #fff 45%); }
.summary-insight { border-left: 4px solid #3b82f6; background: linear-gradient(90deg, rgba(59,130,246,0.08), #fff 45%); }
.summary-label { color: #64748b; font-size: 12px; margin-bottom: 6px; }
.summary-main { color: #0f172a; font-size: 18px; font-weight: 700; line-height: 1.25; }
.summary-sub { color: #475569; font-size: 12px; line-height: 1.45; margin-top: 4px; }
.acc-table th.emphasis { background: #eff6ff; color: #1d4ed8; }
.acc-table td.emphasis { font-weight: 700; box-shadow: inset 0 0 0 999px rgba(255,255,255,0.12); }
.type-mark { display: block; margin-top: 2px; color: #2563eb; font-size: 10px; font-weight: 500; }
.drill-hint { font-weight: 400; color: #94a3b8; font-size: 12px; margin-left: 10px; }
.qt-head { cursor: pointer; transition: background 0.15s ease; }
.qt-head:hover { background: #dbeafe; color: #1d4ed8; }
.model-row { cursor: pointer; transition: background 0.15s ease; }
.model-row:hover td { background: #f1f5f9; }
.acc-cell { cursor: pointer; }
.acc-cell:hover { outline: 2px solid #3b82f6; outline-offset: -2px; }
@media (max-width: 900px) {
  .summary-grid { grid-template-columns: 1fr; }
}
</style>
