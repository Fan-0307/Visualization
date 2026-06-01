<template>
  <div style="padding:16px">
    <div style="margin-bottom:10px;display:flex;gap:16px;align-items:center;flex-wrap:wrap">
      <label style="font-size:13px">
        <input type="checkbox" v-model="showOnlyErrors" @change="redraw" />
        只显示有模型答错的样本
      </label>
      <span style="font-size:12px;color:#999">{{ filteredSamples.length }} / {{ allSamples.length }} 个样本</span>
      <span style="font-size:12px;color:#555">点击格子 → 查看 Attention</span>
    </div>
    <div style="display:flex;gap:16px;margin-bottom:8px;font-size:12px">
      <span><span style="display:inline-block;width:12px;height:12px;background:#4caf50;border-radius:2px;vertical-align:middle"></span> 正确</span>
      <span><span style="display:inline-block;width:12px;height:12px;background:#f44336;border-radius:2px;vertical-align:middle"></span> 错误</span>
      <span><span style="display:inline-block;width:12px;height:12px;border:2px solid #ff9800;border-radius:2px;vertical-align:middle"></span> 所有模型都错</span>
    </div>
    <div style="overflow:auto;max-height:75vh"><svg ref="svgRef"></svg></div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import * as d3 from 'd3'

const props = defineProps({ data: Array })
const emit = defineEmits(['cell-click'])
const svgRef = ref(null)
const showOnlyErrors = ref(true)

const models = computed(() => [...new Set(props.data.map(d => d.model))])
const allSamples = computed(() => [...new Set(props.data.map(d => d.sample_id))])

const filteredSamples = computed(() => {
  if (!showOnlyErrors.value) return allSamples.value
  const errorSamples = new Set()
  props.data.forEach(d => { if (!d.correct) errorSamples.add(d.sample_id) })
  return allSamples.value.filter(s => errorSamples.has(s))
})

const allWrongSamples = computed(() => {
  const byS = {}
  props.data.forEach(d => {
    if (!byS[d.sample_id]) byS[d.sample_id] = { total: 0, wrong: 0 }
    byS[d.sample_id].total++
    if (!d.correct) byS[d.sample_id].wrong++
  })
  return new Set(Object.entries(byS).filter(([,v]) => v.wrong === v.total).map(([k]) => k))
})

function redraw() {
  const samples = filteredSamples.value
  const ms = models.value
  const cell = 20, ml = 70, mt = 30, mb = 50, mr = 20
  const W = samples.length * cell + ml + mr
  const H = ms.length * cell + mt + mb

  const svg = d3.select(svgRef.value).attr('width', W).attr('height', H)
  svg.selectAll('*').remove()
  const g = svg.append('g').attr('transform', `translate(${ml},${mt})`)

  const x = d3.scaleBand().domain(samples).range([0, samples.length * cell]).padding(0.05)
  const y = d3.scaleBand().domain(ms).range([0, ms.length * cell]).padding(0.05)

  g.append('g').attr('transform', `translate(0,${ms.length * cell})`)
    .call(d3.axisBottom(x).tickSize(0))
    .selectAll('text').attr('transform','rotate(-45)').style('text-anchor','end').style('font-size','7px')
  g.append('g').call(d3.axisLeft(y).tickSize(0))
    .selectAll('text').style('font-size','12px')

  const tooltip = d3.select('body').selectAll('.em-tip').data([0]).join('div')
    .attr('class','em-tip')
    .style('position','absolute').style('background','rgba(0,0,0,0.8)').style('color','#fff')
    .style('padding','6px 10px').style('border-radius','4px').style('font-size','12px')
    .style('pointer-events','none').style('opacity',0)

  const filtered = props.data.filter(d => samples.includes(d.sample_id))
  g.selectAll('rect').data(filtered).join('rect')
    .attr('x', d => x(d.sample_id))
    .attr('y', d => y(d.model))
    .attr('width', x.bandwidth()).attr('height', y.bandwidth())
    .attr('fill', d => d.correct ? '#4caf50' : '#f44336')
    .attr('opacity', d => 0.5 + d.confidence * 0.5)
    .attr('rx', 2)
    .attr('stroke', d => allWrongSamples.value.has(d.sample_id) ? '#ff9800' : 'none')
    .attr('stroke-width', 2)
    .style('cursor','pointer')
    .on('mouseover', function(e, d) {
      d3.select(this).attr('stroke','#333').attr('stroke-width',2)
      tooltip.style('opacity',1)
        .html(`<b>${d.model}</b> · ${d.sample_id}<br/>${d.correct?'✓ 正确':'✗ 错误'} · conf ${d.confidence?.toFixed(2)}`)
        .style('left',(e.pageX+10)+'px').style('top',(e.pageY-20)+'px')
    })
    .on('mouseout', function(e, d) {
      d3.select(this).attr('stroke', allWrongSamples.value.has(d.sample_id)?'#ff9800':'none').attr('stroke-width',2)
      tooltip.style('opacity',0)
    })
    .on('click', (_, d) => emit('cell-click', { model: d.model, sample_id: d.sample_id }))
}

onMounted(redraw)
watch(() => props.data, redraw)
watch(showOnlyErrors, redraw)
</script>
