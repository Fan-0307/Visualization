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
        <label class="smooth-label">
          <input type="checkbox" v-model="smooth" /> 平滑
        </label>
        <button class="play-btn" :class="{ playing }" @click="togglePlay">{{ playing ? '⏸' : '▶' }}</button>
      </div>
    </div>

    <div v-if="sampleData" class="card" style="margin-top:16px">
      <div class="vis-area">
        <div class="img-section">
          <div class="img-wrap">
            <img :src="`/img/${sampleData.image.img_id}.jpeg`"
              :width="imgDisplayW" :height="imgDisplayH"
              class="main-img" alt="input image"
              @error="e => e.target.style.display='none'" />
            <canvas ref="hmCanvas" class="hm-canvas"
              :width="imgDisplayW" :height="imgDisplayH"></canvas>
          </div>
        </div>
        <div class="tokens-section">
          <div class="tok-row-group">
            <div class="tok-row">
              <span v-for="(t, i) in questionList" :key="'q'+i"
                class="tok-box"
                :class="{ active: selType === 'q' && selIdx === i }"
                @click="selectToken('q', i)"
                :style="queBoxStyle(i)">
                {{ t.token }}
              </span>
            </div>
          </div>
          <div class="tok-row-group" style="margin-top:8px">
            <div class="tok-row">
              <span v-for="(t, i) in answerList" :key="'a'+i"
                class="tok-box"
                :class="{ active: selType === 'a' && selIdx === i }"
                @click="selectToken('a', i)">
                {{ t.token }}
              </span>
            </div>
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
import { ref, computed, watch, nextTick, onBeforeUnmount } from 'vue'
import * as d3 from 'd3'
import attnIndex from '../data/attn_index.json'

const samples = attnIndex.samples || []
const sid = ref(samples[0]?.file || '')
const layerIdx = ref(0)
const maxLayer = ref(11)
const sampleData = ref(null)
const selType = ref('q')
const selIdx = ref(0)
const hmCanvas = ref(null)
const smooth = ref(true)
const playing = ref(false)
const imgDisplayW = 360
const imgDisplayH = ref(270)
let playTimer = null

const gridW = computed(() => sampleData.value?.image.grid.w || 1)
const gridH = computed(() => sampleData.value?.image.grid.h || 1)

function tokenList(obj) {
  const lk = `layer_${layerIdx.value}`
  return Object.entries(obj || {}).map(([k, v]) => ({
    token: v.token,
    vis: v.vis_attn?.[lk] || null,
    que: v.que_attn?.[lk] || null,
  }))
}

const questionList = computed(() => tokenList(sampleData.value?.question))
const answerList = computed(() => tokenList(sampleData.value?.answer))

const selToken = computed(() => {
  const list = selType.value === 'q' ? questionList.value : answerList.value
  return list[selIdx.value] || null
})

function selectToken(type, idx) {
  selType.value = type
  selIdx.value = idx
}

function nextToken() {
  const ql = questionList.value
  const al = answerList.value
  const total = ql.length + al.length
  if (total === 0) return
  const cur = selType.value === 'q' ? selIdx.value : ql.length + selIdx.value
  const next = (cur + 1) % total
  if (next < ql.length) {
    selType.value = 'q'
    selIdx.value = next
  } else {
    selType.value = 'a'
    selIdx.value = next - ql.length
  }
}

function togglePlay() {
  playing.value = !playing.value
  if (playing.value) {
    playTimer = setInterval(nextToken, 2500)
  } else {
    clearInterval(playTimer)
    playTimer = null
  }
}

onBeforeUnmount(() => {
  clearInterval(playTimer)
})

const qLen = computed(() => questionList.value.length)

function queBoxStyle(i) {
  if (!selToken.value?.que) return { background: 'transparent' }
  const vec = selToken.value.que
  if (i >= vec.length) return { background: 'transparent' }
  const mx = d3.max(vec) || 1
  const c = d3.interpolateReds(vec[i] / mx)
  return { background: c }
}

function drawHeatmap() {
  const canvas = hmCanvas.value
  if (!canvas) return
  const ctx = canvas.getContext('2d')
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  if (!selToken.value?.vis) return

  const gW = gridW.value
  const gH = gridH.value
  const vec = selToken.value.vis
  const nPatches = gW * gH
  const patches = vec.length > nPatches ? vec.slice(vec.length - nPatches) : vec
  const mx = d3.max(patches) || 1

  const off = document.createElement('canvas')
  off.width = gW
  off.height = gH
  const offCtx = off.getContext('2d')
  const imgData = offCtx.createImageData(gW, gH)
  const data = imgData.data

  for (let i = 0; i < gW * gH && i < patches.length; i++) {
    const v = patches[i] / mx
    const c = d3.rgb(d3.interpolateTurbo(v))
    data[i * 4] = c.r
    data[i * 4 + 1] = c.g
    data[i * 4 + 2] = c.b
    data[i * 4 + 3] = 165
  }
  offCtx.putImageData(imgData, 0, 0)

  ctx.imageSmoothingEnabled = smooth.value
  ctx.imageSmoothingQuality = 'high'
  ctx.drawImage(off, 0, 0, canvas.width, canvas.height)
}

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
  selType.value = 'q'
  selIdx.value = 0

  const ratio = json.image.h / json.image.w
  imgDisplayH.value = Math.round(imgDisplayW * ratio)
}

function onSampleChange() {
  loadSample(sid.value)
}

watch([selType, selIdx, layerIdx, sampleData, smooth], () => {
  nextTick(drawHeatmap)
})

if (samples.length) loadSample(sid.value)
</script>

<style scoped>
.controls { display: flex; gap: 16px; align-items: center; font-size: 13px; flex-wrap: wrap; }
.controls select { padding: 4px 8px; border: 1px solid #e5e7eb; border-radius: 4px; }
.controls input[type=range] { width: 120px; vertical-align: middle; }
.correct-badge { padding: 2px 10px; border-radius: 10px; font-size: 12px; font-weight: 600; }
.correct-badge.ok { background: #dcfce7; color: #16a34a; }
.correct-badge.err { background: #fee2e2; color: #dc2626; }
.smooth-label { display: flex; align-items: center; gap: 4px; font-size: 12px; cursor: pointer; user-select: none; }
.smooth-label input { margin: 0; }
.play-btn {
  padding: 2px 12px; font-size: 14px; cursor: pointer; border: 1px solid #ccc;
  border-radius: 4px; background: #fff; line-height: 1.4;
}
.play-btn:hover { background: #f5f5f5; }
.play-btn.playing { background: #e8f4e8; border-color: #4caf50; }

.vis-area { display: flex; gap: 16px; align-items: flex-start; }
.img-section { flex-shrink: 0; }
.img-wrap { position: relative; display: inline-block; line-height: 0; }
.main-img { display: block; border: none; }
.hm-canvas { position: absolute; top: 0; left: 0; pointer-events: none; }
.tokens-section { flex: 1; min-width: 0; padding-top: 0; }
.tok-row { display: flex; flex-wrap: wrap; gap: 4px; }
.tok-box {
  display: inline-block; padding: 3px 8px; font-size: 12px; font-family: monospace;
  border: 1.5px solid #ccc; border-radius: 3px; cursor: pointer;
  transition: background 0.15s, border-color 0.15s; line-height: 1.4; color: #222;
}
.tok-box.active { border-color: #ff8c00; box-shadow: 0 0 0 1.5px #ff8c00; }
.tok-box:hover { border-color: #888; }
</style>
