<template>
  <div>
    <div class="card">
      <div class="section-title">模型准确率对比</div>
      <div style="overflow-x:auto;margin-bottom:28px">
        <table class="acc-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Overall</th>
              <th v-for="qt in qtypes" :key="qt">{{ qt }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="m in models" :key="m">
              <td class="model-name">{{ m }}</td>
              <td :style="{background:accColor(stats[m]?.overall)}">{{ pct(stats[m]?.overall) }}</td>
              <td v-for="qt in qtypes" :key="qt" :style="{background:accColor(stats[m]?.byType[qt])}">
                {{ pct(stats[m]?.byType[qt]) }}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px">
      <div class="card">
        <div class="section-title">Overall Accuracy</div>
        <svg ref="barRef"></svg>
      </div>
      <div class="card">
        <div class="section-title">按问题类型准确率</div>
        <div style="overflow-x:auto"><svg ref="groupRef"></svg></div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import * as d3 from 'd3'

const props = defineProps({ data: Array })
const barRef = ref(null)
const groupRef = ref(null)

const models = computed(() => [...new Set(props.data.map(d => d.model))])
const qtypes = computed(() => [...new Set(props.data.map(d => d.question_type))].sort())

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

function pct(v) { return v == null ? '—' : (v * 100).toFixed(1) + '%' }
function accColor(v) {
  if (v == null) return ''
  const t = Math.max(0, Math.min(1, v))
  return `rgba(${Math.round(239*(1-t)+34*t)},${Math.round(68*(1-t)+197*t)},${Math.round(68*(1-t)+94*t)},0.22)`
}

const colors = ['#3b82f6','#10b981','#f59e0b','#ef4444','#8b5cf6']

function drawBar() {
  const ms = models.value
  const W = 340, H = 180, ml = 50, mb = 28, mt = 8, mr = 16
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
  ms.forEach((m,i) => {
    const v = stats.value[m]?.overall ?? 0
    g.append('rect').attr('x',x(m)).attr('y',y(v)).attr('width',x.bandwidth()).attr('height',h-y(v))
      .attr('fill',colors[i%colors.length]).attr('rx',4)
    g.append('text').attr('x',x(m)+x.bandwidth()/2).attr('y',y(v)-5)
      .attr('text-anchor','middle').attr('font-size',11).attr('fill','#444').text((v*100).toFixed(1)+'%')
  })
}

function drawGroup() {
  const ms = models.value, qts = qtypes.value
  const W = Math.max(380, qts.length * ms.length * 20 + 80)
  const H = 200, ml = 44, mb = 56, mt = 8, mr = 12
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
  qts.forEach(qt => ms.forEach((m,i) => {
    const v = stats.value[m]?.byType[qt]
    if (v == null) return
    g.append('rect').attr('x',x0(qt)+x1(m)).attr('y',y(v))
      .attr('width',x1.bandwidth()).attr('height',h-y(v))
      .attr('fill',colors[i%colors.length]).attr('rx',3)
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
.acc-table { border-collapse: collapse; font-size: 13px; width: 100%; }
.acc-table th { padding: 8px 14px; background: #f8fafc; border: 1px solid #e5e7eb; font-weight: 600; color: #374151; text-align: center; }
.acc-table th:first-child { text-align: left; }
.acc-table td { padding: 8px 14px; border: 1px solid #e5e7eb; text-align: center; font-size: 13px; }
.model-name { font-weight: 600; color: #1e293b; text-align: left !important; }
</style>
