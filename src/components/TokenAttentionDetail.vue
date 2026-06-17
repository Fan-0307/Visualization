<template>
  <div>
    <div class="card">
      <div class="section-title">逐层 Token 注意力</div>
      <div class="controls">
        <label>样本 <select v-model="sid" @change="onSampleChange">
          <option v-for="s in samples" :key="s.file" :value="s.file">{{ s.file }}</option>
        </select></label>
        <label>层 <input type="range" v-model.number="layerIdx" :min="0" :max="maxLayer" /> {{ layerIdx }}</label>
        <span v-if="sampleData" class="correct-badge" :class="sampleData.correct ? 'ok' : 'err'">
          {{ sampleData.correct ? '正确' : '错误' }}
        </span>
      </div>
    </div>

    <div v-if="sampleData" class="card vis-layout" style="margin-top:16px">
      <div class="img-col">
        <img :src="`/img/${sampleData.image.img_id}.jpeg`" class="main-img"
          @error="e => e.target.style.display='none'" />
        <div class="img-meta">{{ sampleData.image.w }} × {{ sampleData.image.h }}</div>
      </div>

      <div class="token-col">
        <div class="token-section">
          <div class="section-title">问题 Token</div>
          <div v-for="(tok, key) in questionList" :key="'q'+key" class="token-row">
            <span class="tok-text">{{ tok.token }}</span>
            <svg :width="hmSize" :height="hmSize" class="hm-svg"
              :ref="el => { if(el) drawVis(el, tok.vis, gridW, gridH) }"></svg>
            <svg :width="barW" :height="barH" class="bar-svg"
              :ref="el => { if(el) drawBar(el, tok.que, qlen) }"></svg>
          </div>
        </div>
        <div class="token-section" style="margin-top:16px">
          <div class="section-title">答案 Token</div>
          <div v-for="(tok, key) in answerList" :key="'a'+key" class="token-row">
            <span class="tok-text">{{ tok.token }}</span>
            <svg v-if="tok.vis" :width="hmSize" :height="hmSize" class="hm-svg"
              :ref="el => { if(el) drawVis(el, tok.vis, gridW, gridH) }"></svg>
            <span v-else class="na">—</span>
            <svg :width="barW" :height="barH" class="bar-svg"
              :ref="el => { if(el) drawBar(el, tok.que, qlen) }"></svg>
          </div>
        </div>
      </div>
    </div>

    <div v-else class="card" style="margin-top:16px;text-align:center;padding:40px;color:#bbb">
      请先运行 data/blip/run.py 或 data/qwen/run.py 生成 per-token 注意力数据，<br />
      然后执行 python data/build_index.py --data-dir &lt;输出目录&gt; --out-dir src/data
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import * as d3 from 'd3'
import attnIndex from '../data/attn_index.json'

const hmSize = 100
const barW = 140
const barH = 16

const samples = attnIndex.samples || []
const sid = ref(samples[0]?.file || '')
const layerIdx = ref(0)
const maxLayer = ref(11)
const sampleData = ref(null)

const gridW = computed(() => sampleData.value?.image.grid.w || 1)
const gridH = computed(() => sampleData.value?.image.grid.h || 1)

const qlen = computed(() => Object.keys(sampleData.value?.question || {}).length)

function tokenList(obj) {
  return Object.entries(obj || {}).map(([k, v]) => {
    const lk = `layer_${layerIdx.value}`
    return {
      token: v.token,
      vis: v.vis_attn?.[lk] || null,
      que: v.que_attn?.[lk] || null,
    }
  })
}
const questionList = computed(() => tokenList(sampleData.value?.question))
const answerList = computed(() => tokenList(sampleData.value?.answer))

async function loadSample(file) {
  const resp = await fetch(`/data/attn/${file}`)
  const json = await resp.json()
  sampleData.value = json
  const nLayers = Math.max(
    ...Object.values(json.question || {}).map(t => Object.keys(t.vis_attn || {}).length),
    ...Object.values(json.answer || {}).map(t => Object.keys(t.que_attn || {}).length),
    1
  )
  maxLayer.value = nLayers - 1
  if (layerIdx.value > maxLayer.value) layerIdx.value = 0
}

function onSampleChange() {
  loadSample(sid.value)
}

if (samples.length) loadSample(sid.value)

function drawVis(el, vec, gW, gH) {
  const svg = d3.select(el)
  svg.selectAll('*').remove()
  if (!vec || vec.length < 2) return
  const patches = vec.slice(1)  // skip CLS
  const g = svg.append('g')
  const cw = hmSize / gW, ch = hmSize / gH
  const mx = d3.max(patches) || 1
  for (let r = 0; r < gH; r++) {
    for (let c = 0; c < gW; c++) {
      const idx = r * gW + c
      const v = idx < patches.length ? patches[idx] : 0
      g.append('rect')
        .attr('x', c * cw).attr('y', r * ch)
        .attr('width', cw + 0.5).attr('height', ch + 0.5)
        .attr('fill', d3.interpolateBlues(v / mx))
    }
  }
}

function drawBar(el, vec, len) {
  const svg = d3.select(el)
  svg.selectAll('*').remove()
  if (!vec) return
  const g = svg.append('g')
  const bw = barW / len
  const mx = d3.max(vec) || 1
  vec.forEach((v, i) => {
    g.append('rect')
      .attr('x', i * bw).attr('y', 0)
      .attr('width', bw).attr('height', barH)
      .attr('fill', d3.interpolateReds(v / mx))
  })
}
</script>

<style scoped>
.controls { display: flex; gap: 16px; align-items: center; font-size: 13px; }
.controls select { padding: 4px 8px; border: 1px solid #e5e7eb; border-radius: 4px; }
.controls input[type=range] { width: 120px; vertical-align: middle; }
.correct-badge { padding: 2px 10px; border-radius: 10px; font-size: 12px; font-weight: 600; }
.correct-badge.ok { background: #dcfce7; color: #16a34a; }
.correct-badge.err { background: #fee2e2; color: #dc2626; }
.vis-layout { display: flex; gap: 24px; }
.img-col { flex-shrink: 0; }
.main-img { width: 280px; border-radius: 8px; }
.img-meta { text-align: center; font-size: 11px; color: #999; margin-top: 4px; }
.token-col { flex: 1; min-width: 0; }
.token-row { display: flex; align-items: center; gap: 6px; margin: 3px 0; }
.tok-text { width: 50px; font-size: 11px; font-family: monospace; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex-shrink: 0; }
.hm-svg { flex-shrink: 0; border-radius: 4px; }
.bar-svg { flex-shrink: 0; border-radius: 2px; }
.na { color: #ccc; font-size: 11px; width: 100px; text-align: center; }
</style>
