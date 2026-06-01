<template>
  <div>
    <div class="card">
      <div class="section-title">正确 vs 错误时的注意力模式</div>
      <div style="display:flex;gap:8px;margin-bottom:20px">
        <button v-for="m in models" :key="m" @click="selectedModel=m"
          :class="['model-btn', selectedModel===m && 'model-btn-active']">{{ m }}</button>
      </div>

      <div v-if="selectedModel" style="display:flex;gap:32px;flex-wrap:wrap">
        <div>
          <div style="font-size:13px;font-weight:600;color:#16a34a;margin-bottom:10px">
            答对 <span style="font-weight:400;color:#888">{{ correctSamples.length }} 个</span>
          </div>
          <div class="thumb-grid">
            <img v-for="sid in correctSamples.slice(0,24)" :key="sid"
              :src="'/photos/' + modelKey(selectedModel) + '/' + sid + '.png'"
              class="thumb thumb-correct"
              @error="e => e.target.style.opacity='0'" />
          </div>
          <div v-if="correctSamples.length > 24" class="more-label">+{{ correctSamples.length - 24 }} 更多</div>
        </div>
        <div>
          <div style="font-size:13px;font-weight:600;color:#dc2626;margin-bottom:10px">
            答错 <span style="font-weight:400;color:#888">{{ wrongSamples.length }} 个</span>
          </div>
          <div class="thumb-grid">
            <img v-for="sid in wrongSamples.slice(0,24)" :key="sid"
              :src="'/photos/' + modelKey(selectedModel) + '/' + sid + '.png'"
              class="thumb thumb-wrong"
              @error="e => e.target.style.opacity='0'" />
          </div>
          <div v-if="wrongSamples.length > 24" class="more-label">+{{ wrongSamples.length - 24 }} 更多</div>
        </div>
      </div>
    </div>

    <div class="card" style="margin-top:16px">
      <div class="section-title">各问题类型答对/答错分布</div>
      <div style="overflow-x:auto"><svg ref="svgRef"></svg></div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted } from 'vue'
import * as d3 from 'd3'
import rawData from '../data/vl_attention_data.json'

const props = defineProps({ data: Array, models: Array })
const svgRef = ref(null)
const selectedModel = ref(null)
watch(() => props.models, ms => { if (ms?.length) selectedModel.value = ms[0] }, { immediate: true })

const labelToKey = Object.fromEntries(
  Object.entries(rawData.model_labels || {}).map(([k,v]) => [v, k])
)
const modelKey = (label) => labelToKey[label] || label.toLowerCase()

const correctSamples = computed(() =>
  props.data.filter(d => d.model === selectedModel.value && d.correct).map(d => d.sample_id)
)
const wrongSamples = computed(() =>
  props.data.filter(d => d.model === selectedModel.value && !d.correct).map(d => d.sample_id)
)
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
    // correct (bottom)
    if (correct > 0)
      g.append('rect').attr('x',x(qt)).attr('y',y(correct)).attr('width',bw).attr('height',h-y(correct))
        .attr('fill','#22c55e').attr('opacity',0.8).attr('rx',3)
    // wrong (stacked on top)
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
.thumb-grid { display: flex; flex-wrap: wrap; gap: 4px; max-width: 320px; }
.thumb { width: 48px; height: 48px; object-fit: cover; border-radius: 4px; }
.thumb-correct { border: 1.5px solid #22c55e; }
.thumb-wrong { border: 1.5px solid #ef4444; }
.more-label { font-size: 11px; color: #999; margin-top: 6px; }
</style>
