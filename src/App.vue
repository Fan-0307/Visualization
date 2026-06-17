<template>
  <div class="app">
    <nav class="nav">
      <span class="nav-brand">VLM Attention Analysis</span>
      <div class="nav-tabs">
        <button v-for="t in tabs" :key="t.id" @click="activeTab=t.id"
          :class="['tab', activeTab===t.id && 'tab-active']">{{ t.label }}</button>
      </div>
    </nav>
    <div class="content">
      <ModelComparison v-if="activeTab==='compare'" :data="matrixData" />
      <MatrixAttention v-if="activeTab==='matrix'" :data="matrixData" :models="models" />
      <CorrectErrorPattern v-if="activeTab==='pattern'" :data="matrixData" :models="models" />
      <TokenAttentionDetail v-if="activeTab==='token'" />
    </div>
  </div>
</template>

<script setup>
import { reactive, ref, computed } from 'vue'
import ModelComparison from './components/ModelComparison.vue'
import MatrixAttention from './components/MatrixAttention.vue'
import CorrectErrorPattern from './components/CorrectErrorPattern.vue'
import TokenAttentionDetail from './components/TokenAttentionDetail.vue'
import data from './data/vl_attention_data.json'

const matrixData = data.matrix_data || []
const models = computed(() => [...new Set(matrixData.map(d => d.model))])
const activeTab = ref('compare')

const tabs = [
  { id: 'compare', label: '模型对比' },
  { id: 'matrix',  label: 'Error Matrix & Attention' },
  { id: 'pattern', label: '正确 / 错误模式' },
  { id: 'token',   label: '逐层 Token 注意力' },
]
</script>

<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f4f6f9; color: #1a1a2e; }
.app { display: flex; flex-direction: column; height: 100vh; }
.nav { display: flex; align-items: center; gap: 24px; padding: 0 24px; background: #1a1a2e; height: 52px; flex-shrink: 0; }
.nav-brand { color: #fff; font-weight: 700; font-size: 15px; letter-spacing: 0.3px; white-space: nowrap; }
.nav-tabs { display: flex; gap: 4px; }
.tab { padding: 6px 16px; border: none; background: transparent; color: rgba(255,255,255,0.6); cursor: pointer; font-size: 13px; border-radius: 6px; transition: all 0.15s; }
.tab:hover { background: rgba(255,255,255,0.1); color: #fff; }
.tab-active { background: rgba(255,255,255,0.15); color: #fff; font-weight: 600; }
.content { flex: 1; overflow: auto; padding: 24px; }
.card { background: #fff; border-radius: 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); padding: 20px; }
.card + .card { margin-top: 16px; }
.section-title { font-size: 14px; font-weight: 600; color: #444; margin-bottom: 14px; }
</style>
