<template>
  <div style="padding:16px">
    <div style="margin-bottom:12px;display:flex;gap:12px;align-items:center;flex-wrap:wrap">
      <span style="font-size:13px;font-weight:bold">样本：</span>
      <select v-model="localSampleId" style="font-size:12px;padding:2px 6px">
        <option v-for="s in sampleIds" :key="s" :value="s">{{ s }}</option>
      </select>
      <span style="font-size:13px">基准模型 A：</span>
      <select v-model="modelA" style="font-size:12px;padding:2px 6px">
        <option v-for="m in models" :key="m" :value="m">{{ m }}</option>
      </select>
    </div>
    <div v-if="!localSampleId" style="color:#999;padding:40px;text-align:center">请选择样本</div>
    <template v-else>
      <div style="margin-bottom:10px;font-size:13px;color:#555">
        <b>问题：</b>{{ question }}
      </div>
      <div style="display:flex;gap:12px;flex-wrap:wrap;align-items:flex-start">
        <div style="text-align:center">
          <div style="font-size:12px;font-weight:bold;margin-bottom:4px">
            {{ modelA }} (基准)
            <span :style="{color:isCorrect(modelA)?'#4caf50':'#f44336',marginLeft:'4px'}">{{ isCorrect(modelA)?'✓':'✗' }}</span>
          </div>
          <img :src="'/photos/' + modelA + '/' + localSampleId + '.png'"
            style="width:168px;height:168px;object-fit:cover;border-radius:4px;border:2px solid #1976d2"
            @error="e => e.target.style.opacity='0.1'" />
        </div>
        <div v-for="m in otherModels" :key="m" style="text-align:center">
          <div style="font-size:12px;font-weight:bold;margin-bottom:4px">
            {{ m }}
            <span :style="{color:isCorrect(m)?'#4caf50':'#f44336',marginLeft:'4px'}">{{ isCorrect(m)?'✓':'✗' }}</span>
          </div>
          <img :src="'/photos/' + m + '/' + localSampleId + '.png'"
            style="width:168px;height:168px;object-fit:cover;border-radius:4px;border:1px solid #eee"
            @error="e => e.target.style.opacity='0.1'" />
        </div>
      </div>
      <div style="margin-top:10px;font-size:12px;color:#888">
        注：差异图需要原始 attention 矩阵，当前展示各模型热力图供视觉对比。
      </div>
    </template>
  </div>
</template>

<script setup>
import { ref, computed, watch } from 'vue'
import rawData from '../data/vl_attention_data.json'

const props = defineProps({ sampleId: String, models: Array })
const matrixData = rawData.matrix_data || []
const attentionData = rawData.attention_data || {}
const sampleIds = computed(() => [...new Set(matrixData.map(d => d.sample_id))])

const localSampleId = ref(props.sampleId || sampleIds.value[0])
const modelA = ref(props.models?.[0])
watch(() => props.sampleId, v => { if (v) localSampleId.value = v })
watch(() => props.models, ms => { if (ms?.length) modelA.value = ms[0] })

const otherModels = computed(() => props.models?.filter(m => m !== modelA.value) || [])
const entry = (m) => attentionData[`${m}__${localSampleId.value}`]
const question = computed(() => entry(modelA.value)?.question || '')
const isCorrect = (m) => matrixData.find(d => d.model === m && d.sample_id === localSampleId.value)?.correct
</script>
