<template>
  <div>
    <!-- top: error matrix -->
    <div class="card">
      <div class="section-title">Error Matrix
        <span style="font-weight:400;color:#999;font-size:12px;margin-left:8px">点击格子查看 Attention</span>
        <label style="float:right;font-size:12px;font-weight:400;color:#666;cursor:pointer">
          <input type="checkbox" v-model="showOnlyErrors" style="margin-right:4px" />只显示有错误的样本
        </label>
      </div>
      <div style="display:flex;gap:12px;margin-bottom:10px;font-size:12px;color:#666">
        <span><i style="display:inline-block;width:10px;height:10px;background:#22c55e;border-radius:2px;vertical-align:middle;margin-right:3px"></i>正确</span>
        <span><i style="display:inline-block;width:10px;height:10px;background:#ef4444;border-radius:2px;vertical-align:middle;margin-right:3px"></i>错误</span>
        <span><i style="display:inline-block;width:10px;height:10px;border:2px solid #f97316;border-radius:2px;vertical-align:middle;margin-right:3px"></i>所有模型都错</span>
        <span style="margin-left:auto;color:#999">{{ filteredSamples.length }} / {{ allSamples.length }} 个样本</span>
      </div>
      <div style="overflow:auto;max-height:260px"><svg ref="svgRef"></svg></div>
    </div>

    <!-- bottom: attention detail for selected sample -->
    <div class="card" style="margin-top:16px">
      <div class="section-title">
        Attention 对比
        <span v-if="selected" style="font-weight:400;color:#666;font-size:12px;margin-left:8px">
          样本 {{ selected.sample_id }} · {{ selectedEntry?.question }}
          <span style="margin-left:8px;color:#888">答案：{{ selectedEntry?.ground_truth }}</span>
        </span>
      </div>
      <div v-if="!selected" style="color:#bbb;text-align:center;padding:40px;font-size:13px">
        点击上方矩阵中的格子查看 Attention 热力图
      </div>
      <template v-else>
        <div style="display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start">
          <!-- original image -->
          <div style="text-align:center">
            <div class="img-label">原图</div>
            <img :src="selectedEntry?.image_src"
              class="attn-img" style="border:2px solid #e5e7eb"
              @error="e => e.target.style.opacity='0.15'" />
          </div>
          <!-- per-model -->
          <div v-for="m in models" :key="m" style="text-align:center">
            <div class="img-label" :style="{color: modelCorrect(m) ? '#16a34a' : '#dc2626'}">
              {{ m }} {{ modelCorrect(m) ? '✓' : '✗' }}
            </div>
            <div class="img-pred">{{ modelPred(m) }}</div>
            <img :src="modelPhoto(m)"
              class="attn-img" :style="{border: selected.model===m ? '2px solid #3b82f6' : '2px solid #e5e7eb'}"
              @error="e => { e.target.style.display='none'; e.target.nextElementSibling.style.display='flex' }" />
            <div class="attn-placeholder">暂无热力图</div>
          </div>
        </div>
      </template>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import * as d3 from 'd3'
import { attentionData, getPhotoUrl } from '../data/processedData'

const props = defineProps({ data: Array, models: Array })
const svgRef = ref(null)
const showOnlyErrors = ref(true)
const selected = ref(null)

const modelKey = (label) =>
  props.data.find(d => d.model === label)?.model_key || label.toLowerCase()

const allSamples = computed(() => [...new Set(props.data.map(d => d.sample_id))])
const allWrongSamples = computed(() => {
  const byS = {}
  props.data.forEach(d => {
    if (!byS[d.sample_id]) byS[d.sample_id] = { total: 0, wrong: 0 }
    byS[d.sample_id].total++
    if (!d.correct) byS[d.sample_id].wrong++
  })
  return new Set(Object.entries(byS).filter(([,v]) => v.wrong === v.total).map(([k]) => k))
})

const filteredSamples = computed(() => {
  if (!showOnlyErrors.value) return allSamples.value
  const err = new Set(props.data.filter(d => !d.correct).map(d => d.sample_id))
  return allSamples.value.filter(s => err.has(s))
})

const selectedEntry = computed(() => selected.value
  ? attentionData[`${selected.value.model}__${selected.value.sample_id}`]
  : null)

const modelCorrect = (m) => props.data.find(d => d.model === m && d.sample_id === selected.value?.sample_id)?.correct
const modelPred = (m) => attentionData[`${m}__${selected.value?.sample_id}`]?.prediction || ''
const modelPhoto = (m) => getPhotoUrl(modelKey(m), selected.value?.sample_id)

function redraw() {
  const samples = filteredSamples.value
  const ms = props.models
  const cell = 18, ml = 80, mt = 24, mb = 40, mr = 16
  const W = samples.length * cell + ml + mr
  const H = ms.length * cell + mt + mb

  const svg = d3.select(svgRef.value).attr('width', W).attr('height', H)
  svg.selectAll('*').remove()
  const g = svg.append('g').attr('transform', `translate(${ml},${mt})`)

  const x = d3.scaleBand().domain(samples).range([0, samples.length * cell]).padding(0.06)
  const y = d3.scaleBand().domain(ms).range([0, ms.length * cell]).padding(0.06)

  g.append('g').attr('transform', `translate(0,${ms.length * cell})`)
    .call(d3.axisBottom(x).tickSize(0))
    .selectAll('text').attr('transform','rotate(-45)').style('text-anchor','end').style('font-size','7px').style('fill','#999')
  g.append('g').call(d3.axisLeft(y).tickSize(0))
    .selectAll('text').style('font-size','12px').style('fill','#444')
  g.selectAll('.domain').style('display','none')

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
    .attr('stroke', d => allWrongSamples.value.has(d.sample_id) ? '#f97316' : 'none')
    .attr('stroke-width', 1.5)
    .style('cursor','pointer')
    .on('mouseover', function(e, d) {
      d3.select(this).attr('opacity', 1).attr('stroke','#1d4ed8').attr('stroke-width', 2)
      tooltip.style('opacity',1)
        .html(`<b>${d.model}</b> · ${d.sample_id}<br/>${d.correct ? '✓ 正确' : '✗ 错误'}`)
        .style('left',(e.pageX+12)+'px').style('top',(e.pageY-28)+'px')
    })
    .on('mouseout', function(e, d) {
      d3.select(this).attr('opacity', 0.82)
        .attr('stroke', allWrongSamples.value.has(d.sample_id) ? '#f97316' : 'none')
        .attr('stroke-width', 1.5)
      tooltip.style('opacity',0)
    })
    .on('click', (_, d) => { selected.value = d })
}

onMounted(redraw)
watch([() => props.data, showOnlyErrors], redraw)
</script>

<style scoped>
.img-label { font-size: 12px; font-weight: 600; margin-bottom: 4px; }
.img-pred { font-size: 11px; color: #888; margin-bottom: 4px; max-width: 168px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.attn-img { width: 168px; height: 168px; object-fit: cover; border-radius: 8px; display: block; }
.attn-placeholder { width: 168px; height: 168px; background: #f3f4f6; border-radius: 8px; display: none; align-items: center; justify-content: center; font-size: 11px; color: #bbb; border: 1px solid #e5e7eb; }
</style>
