<template>
  <div style="padding:20px">
    <h3 style="margin:0 0 16px">Attention 统计分析（Entropy / Sparsity / Center Bias）</h3>

    <!-- metric selector -->
    <div style="margin-bottom:16px;display:flex;gap:8px">
      <button v-for="m in metrics" :key="m.key"
        @click="metric=m.key"
        :style="{padding:'5px 14px',border:'1px solid #ccc',borderRadius:'4px',cursor:'pointer',
          background:metric===m.key?'#1976d2':'#fff',color:metric===m.key?'#fff':'#333',fontSize:'13px'}">
        {{ m.label }}
      </button>
    </div>

    <!-- box plot per model -->
    <h4 style="margin:0 0 8px">各模型分布（Box Plot）</h4>
    <svg ref="boxRef"></svg>

    <!-- bar: mean by question type -->
    <h4 style="margin:24px 0 8px">按问题类型均值</h4>
    <svg ref="qtRef"></svg>

    <!-- summary table -->
    <h4 style="margin:24px 0 8px">模型均值汇总</h4>
    <table style="border-collapse:collapse;font-size:13px">
      <thead>
        <tr style="background:#f0f4ff">
          <th style="padding:7px 14px;border:1px solid #ddd;text-align:left">Model</th>
          <th v-for="m in metrics" :key="m.key" style="padding:7px 14px;border:1px solid #ddd">{{ m.label }}</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="model in models" :key="model">
          <td style="padding:7px 14px;border:1px solid #ddd;font-weight:bold">{{ model }}</td>
          <td v-for="m in metrics" :key="m.key" style="padding:7px 14px;border:1px solid #ddd;text-align:center">
            {{ mean(modelVals(model, m.key)).toFixed(3) }}
          </td>
        </tr>
      </tbody>
    </table>

    <div style="margin-top:12px;font-size:12px;color:#888">
      Entropy高 = 注意力分散 | Sparsity高 = 注意力集中在少数patch | Center Bias高 = 偏向图像中心
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted } from 'vue'
import * as d3 from 'd3'
import rawData from '../data/vl_attention_data.json'

const props = defineProps({ data: Array })

const attentionData = rawData.attention_data || {}
const metrics = [
  { key: 'entropy', label: 'Entropy' },
  { key: 'sparsity', label: 'Sparsity' },
  { key: 'center_bias', label: 'Center Bias' },
]
const metric = ref('entropy')
const boxRef = ref(null)
const qtRef = ref(null)

const models = computed(() => [...new Set(props.data.map(d => d.model))])
const qtypes = computed(() => [...new Set(props.data.map(d => d.question_type))].sort())
const colors = d3.schemeTableau10

function modelVals(model, key) {
  return props.data
    .filter(d => d.model === model)
    .map(d => attentionData[`${model}__${d.sample_id}`]?.[key])
    .filter(v => v != null)
}

function mean(arr) { return arr.length ? arr.reduce((a,b)=>a+b,0)/arr.length : 0 }

function boxStats(arr) {
  const s = [...arr].sort((a,b)=>a-b)
  const q1 = d3.quantile(s, 0.25), q3 = d3.quantile(s, 0.75)
  const iqr = q3 - q1
  return {
    min: Math.max(d3.min(s), q1 - 1.5*iqr),
    q1, median: d3.quantile(s, 0.5), q3,
    max: Math.min(d3.max(s), q3 + 1.5*iqr),
    mean: mean(arr)
  }
}

function drawBox() {
  const ms = models.value
  const W = 500, H = 220, ml = 60, mb = 40, mt = 10, mr = 20
  const svg = d3.select(boxRef.value).attr('width', W).attr('height', H)
  svg.selectAll('*').remove()
  const g = svg.append('g').attr('transform', `translate(${ml},${mt})`)
  const w = W - ml - mr, h = H - mt - mb

  const allVals = ms.flatMap(m => modelVals(m, metric.value))
  const x = d3.scaleBand().domain(ms).range([0, w]).padding(0.4)
  const y = d3.scaleLinear().domain([0, d3.max(allVals)*1.05]).range([h, 0]).nice()

  g.append('g').attr('transform',`translate(0,${h})`).call(d3.axisBottom(x))
  g.append('g').call(d3.axisLeft(y).ticks(5))

  ms.forEach((m, i) => {
    const vals = modelVals(m, metric.value)
    if (!vals.length) return
    const s = boxStats(vals)
    const cx = x(m) + x.bandwidth()/2
    const bw = x.bandwidth()
    const col = colors[i % colors.length]

    // whiskers
    g.append('line').attr('x1',cx).attr('x2',cx).attr('y1',y(s.min)).attr('y2',y(s.max))
      .attr('stroke',col).attr('stroke-width',1.5)
    // box
    g.append('rect').attr('x',x(m)).attr('y',y(s.q3)).attr('width',bw).attr('height',y(s.q1)-y(s.q3))
      .attr('fill',col).attr('opacity',0.4).attr('stroke',col).attr('stroke-width',1.5)
    // median
    g.append('line').attr('x1',x(m)).attr('x2',x(m)+bw).attr('y1',y(s.median)).attr('y2',y(s.median))
      .attr('stroke',col).attr('stroke-width',2.5)
    // mean dot
    g.append('circle').attr('cx',cx).attr('cy',y(s.mean)).attr('r',4)
      .attr('fill','white').attr('stroke',col).attr('stroke-width',2)
  })
}

function drawQT() {
  const ms = models.value, qts = qtypes.value
  const W = Math.max(600, qts.length * ms.length * 22 + 100)
  const H = 220, ml = 50, mb = 60, mt = 10, mr = 20
  const svg = d3.select(qtRef.value).attr('width', W).attr('height', H)
  svg.selectAll('*').remove()
  const g = svg.append('g').attr('transform', `translate(${ml},${mt})`)
  const w = W - ml - mr, h = H - mt - mb

  const allVals = ms.flatMap(m => modelVals(m, metric.value))
  const x0 = d3.scaleBand().domain(qts).range([0, w]).padding(0.2)
  const x1 = d3.scaleBand().domain(ms).range([0, x0.bandwidth()]).padding(0.05)
  const y = d3.scaleLinear().domain([0, d3.max(allVals)*1.05]).range([h, 0]).nice()

  g.append('g').attr('transform',`translate(0,${h})`)
    .call(d3.axisBottom(x0)).selectAll('text')
    .attr('transform','rotate(-30)').style('text-anchor','end').style('font-size','11px')
  g.append('g').call(d3.axisLeft(y).ticks(5))

  qts.forEach(qt => {
    ms.forEach((m, i) => {
      const vals = props.data
        .filter(d => d.model===m && d.question_type===qt)
        .map(d => attentionData[`${m}__${d.sample_id}`]?.[metric.value])
        .filter(v => v != null)
      if (!vals.length) return
      const v = mean(vals)
      g.append('rect')
        .attr('x', x0(qt)+x1(m)).attr('y', y(v))
        .attr('width', x1.bandwidth()).attr('height', h-y(v))
        .attr('fill', colors[i % colors.length]).attr('opacity', 0.85)
    })
  })

  // legend
  ms.forEach((m, i) => {
    svg.append('rect').attr('x', ml + i*80).attr('y', H-16).attr('width',12).attr('height',12).attr('fill',colors[i%colors.length])
    svg.append('text').attr('x', ml + i*80+16).attr('y', H-5).attr('font-size',11).text(m)
  })
}

function draw() { drawBox(); drawQT() }
onMounted(draw)
watch([metric, () => props.data], draw)
</script>
