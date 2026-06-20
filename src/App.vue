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
      <AnswerAggCompare v-if="activeTab==='agg'" />
      <TokenAttentionDetail v-if="activeTab==='token'" />
      <LayerEvolution v-if="activeTab==='layer'" />
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import ModelComparison from './components/ModelComparison.vue'
import MatrixAttention from './components/MatrixAttention.vue'
import CorrectErrorPattern from './components/CorrectErrorPattern.vue'
import TokenAttentionDetail from './components/TokenAttentionDetail.vue'
import AnswerAggCompare from './components/AnswerAggCompare.vue'
import LayerEvolution from './components/LayerEvolution.vue'
import { matrixData, models } from './data/processedData'

const activeTab = ref('compare')

const tabs = [
  { id: 'compare', label: '模型对比' },
  { id: 'matrix',  label: 'Error Matrix & Evidence' },
  { id: 'pattern', label: '正确 / 错误模式' },
  { id: 'agg',     label: '聚合注意力对比' },
  { id: 'token',   label: '逐层 Token 注意力' },
  { id: 'layer',   label: '逐层演化 (V3)' },
]
</script>

<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f4f6f9; color: #1a1a2e; }
.app { display: flex; flex-direction: column; height: 100vh; }
.nav { display: flex; align-items: center; gap: 24px; padding: 0 24px; background: #1a1a2e; height: 52px; flex-shrink: 0; }
.nav-brand { color: #fff; font-weight: 700; font-size: 15px; letter-spacing: 0.3px; white-space: nowrap; }
.nav-tabs { display: flex; gap: 4px; }
.tab { padding: 6px 16px; border: none; background: transparent; color: rgba(255,255,255,0.6); cursor: pointer; font-size: 13px; border-radius: 6px; transition: all 0.2s ease; }
.tab:hover { background: rgba(255,255,255,0.1); color: #fff; }
.tab-active { background: rgba(255,255,255,0.15); color: #fff; font-weight: 600; }
.content { flex: 1; overflow: auto; padding: 24px; }
.card { background: #fff; border-radius: 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); padding: 20px; transition: box-shadow 0.2s ease; }
.card:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.12); }
.card + .card { margin-top: 16px; }
.section-title { font-size: 14px; font-weight: 600; color: #444; margin-bottom: 14px; }
.model-tooltip {
  position: relative;
  cursor: help;
}
.model-tooltip::after {
  content: attr(data-full-name);
  position: absolute;
  left: 50%;
  top: calc(100% + 8px);
  z-index: 60;
  transform: translate(-50%, -2px);
  opacity: 0;
  pointer-events: none;
  white-space: nowrap;
  padding: 7px 10px;
  border: 1px solid rgba(148, 163, 184, 0.28);
  border-radius: 8px;
  background: rgba(15, 23, 42, 0.94);
  color: #f8fafc;
  box-shadow: 0 10px 24px rgba(15, 23, 42, 0.18);
  font-size: 12px;
  font-weight: 600;
  line-height: 1.2;
  letter-spacing: 0;
  transition: opacity 0.14s ease, transform 0.14s ease;
}
.model-tooltip::before {
  content: '';
  position: absolute;
  left: 50%;
  top: calc(100% + 4px);
  z-index: 61;
  width: 8px;
  height: 8px;
  transform: translate(-50%, -2px) rotate(45deg);
  opacity: 0;
  pointer-events: none;
  background: rgba(15, 23, 42, 0.94);
  transition: opacity 0.14s ease, transform 0.14s ease;
}
.model-tooltip:hover::after,
.model-tooltip:hover::before {
  opacity: 1;
  transform: translate(-50%, 0) rotate(0);
}
.model-tooltip:hover::before {
  transform: translate(-50%, 0) rotate(45deg);
}
</style>
