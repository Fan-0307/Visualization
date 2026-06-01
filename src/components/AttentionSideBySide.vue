<template>
  <div style="padding:16px">
    <div style="margin-bottom:12px;display:flex;gap:12px;align-items:center;flex-wrap:wrap">
      <span style="font-size:13px;font-weight:bold">样本：</span>
      <select v-model="localSampleId" style="font-size:12px;padding:2px 6px">
        <option v-for="s in sampleIds" :key="s" :value="s">{{ s }}</option>
      </select>
    </div>
    <div v-if="!localSampleId" style="color:#999;padding:40px;text-align:center">请选择样本</div>
    <template v-else>
      <div style="margin-bottom:10px;font-size:13px;color:#555;display:flex;gap:16px;flex-wrap:wrap">
        <span><b>问题：</b>{{ question }}</span>
        <span><b>答案：</b>{{ groundTruth }}</span>
      </div>
      <div style="display:flex;gap:12px;align-items:flex-start;flex-wrap:wrap">
        <!-- original image -->
        <div style="text-align:center">
          <div style="font-size:12px;margin-bottom:4px;color:#888">原图</div>
          <img :src="'/val2014/' + imagePath"
            style="width:168px;height:168px;object-fit:cover;border-radius:4px;border:1px solid #eee;display:block"
            @error="e => e.target.style.opacity='0.1'" />
        </div>
        <!-- per-model attention heatmap images -->
        <div v-for="m in models" :key="m" style="text-align:center">
          <div style="font-size:12px;margin-bottom:4px;font-weight:bold">
            {{ m }}
            <span :style="{color:isCorrect(m)?'#4caf50':'#f44336',marginLeft:'4px'}">
              {{ isCorrect(m)?'✓':'✗' }}
            </span>
          </div>
          <div style="font-size:11px;color:#888;margin-bottom:4px;max-width:168px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">
            {{ getPred(m) }}
          </div>
          <img :src="'/photos/' + m + '/' + localSampleId + '.png'"
            style="width:168px;height:168px;object-fit:cover;border-radius:4px;border:1px solid #eee;display:block"
            @error="e => { e.target.style.display='none'; e.target.nextElementSibling.style.display='flex' }" />
          <div style="width:168px;height:168px;background:#f5f5f5;border-radius:4px;display:none;align-items:center;justify-content:center;font-size:11px;color:#aaa;border:1px solid #eee">
            暂无热力图
          </div>
        </div>
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
watch(() => props.sampleId, v => { if (v) localSampleId.value = v })

const entry = (m) => attentionData[`${m}__${localSampleId.value}`]
const question = computed(() => entry(props.models?.[0])?.question || '')
const groundTruth = computed(() => entry(props.models?.[0])?.ground_truth || '')
const imagePath = computed(() => entry(props.models?.[0])?.image_path || '')
const isCorrect = (m) => matrixData.find(d => d.model === m && d.sample_id === localSampleId.value)?.correct
const getPred = (m) => entry(m)?.prediction || ''
</script>
