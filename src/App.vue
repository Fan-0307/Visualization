<template>
  <div class="app">
    <nav class="nav">
      <span class="nav-brand">VLM Attention Analysis</span>
      <div class="nav-tabs">
        <button v-for="t in mainTabs" :key="t.id" @click="activeTab=t.id"
          :class="['tab', activeTab===t.id && 'tab-active']">{{ t.label }}</button>
        <!-- 更多: 数据线不同、无法与主流程联动的视图收纳于此 -->
        <div class="more-wrap">
          <button :class="['tab', moreActive && 'tab-active']"
            @click="moreOpen=!moreOpen">更多 ▾</button>
          <div v-if="moreOpen" class="more-menu">
            <button v-for="t in moreTabs" :key="t.id"
              @click="activeTab=t.id; moreOpen=false"
              :class="['more-item', activeTab===t.id && 'more-item-active']">{{ t.label }}</button>
          </div>
        </div>
        <!-- click-away backdrop so moving the mouse into the menu doesn't dismiss it -->
        <div v-if="moreOpen" class="more-backdrop" @click="moreOpen=false"></div>
      </div>
      <div v-if="linkBadge" class="link-badge" title="跨视图联动选择">
        <span class="link-dot"></span>
        <span class="link-text">联动：{{ linkBadge }}</span>
        <button class="link-clear" @click="clearLink" title="清除联动">✕</button>
      </div>
    </nav>
    <div class="content">
      <ModelComparison v-if="activeTab==='compare'" :data="matrixData" />
      <MatrixAttention v-if="activeTab==='matrix'" :data="matrixData" :models="models" />
      <CorrectErrorPattern v-if="activeTab==='pattern'" :data="matrixData" :models="models" />
      <DiffusionFingerprint v-if="activeTab==='diffusion'" />
      <AnswerAggCompare v-if="activeTab==='agg'" />
      <TokenAttentionDetail v-if="activeTab==='token'" />
      <LayerEvolution v-if="activeTab==='layer'" />
    </div>
  </div>
</template>

<script setup>
import { ref, provide, computed } from 'vue'
import ModelComparison from './components/ModelComparison.vue'
import MatrixAttention from './components/MatrixAttention.vue'
import CorrectErrorPattern from './components/CorrectErrorPattern.vue'
import DiffusionFingerprint from './components/DiffusionFingerprint.vue'
import TokenAttentionDetail from './components/TokenAttentionDetail.vue'
import AnswerAggCompare from './components/AnswerAggCompare.vue'
import LayerEvolution from './components/LayerEvolution.vue'
import { matrixData, models } from './data/processedData'
import { selection, broadcast, broadcastFields, consume, clearSelection } from './store/selection'

const activeTab = ref('compare')
// Lets any view drive a cross-view drill-down (e.g. overview → sample diagnosis).
function navigate(tabId) { if (tabId) activeTab.value = tabId }

provide('selection', selection)
provide('broadcast', broadcast)
provide('broadcastFields', broadcastFields)
provide('consume', consume)
provide('navigate', navigate)

// Main flow (share results.json / layer_evo and link across views)
const mainTabs = [
  { id: 'compare', label: '模型对比' },
  { id: 'matrix',  label: '样本诊断' },
  { id: 'pattern', label: '错误模式' },
  { id: 'diffusion', label: '注意力扩散' },
  { id: 'layer',   label: '逐层演化' },
]
// 更多: token-level attn_index data line — different sample set, no cross-view linking
const moreTabs = [
  { id: 'agg',     label: '聚合注意力' },
  { id: 'token',   label: 'Token 注意力' },
]
const moreOpen = ref(false)
const moreActive = computed(() => moreTabs.some(t => t.id === activeTab.value))

// Global linked-selection badge — makes brushing & linking visible across tabs.
const linkBadge = computed(() => {
  const parts = []
  if (selection.questionType) parts.push(selection.questionType)
  if (selection.model) parts.push(selection.model)
  if (selection.sampleId) parts.push('#' + selection.sampleId)
  return parts.length ? parts.join(' · ') : null
})
function clearLink() { clearSelection() }
</script>

<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f4f6f9; color: #1a1a2e; }
.app { display: flex; flex-direction: column; height: 100vh; }
.nav { display: flex; align-items: center; gap: 24px; padding: 0 24px; background: #1a1a2e; height: 52px; flex-shrink: 0; }
.link-badge { display: flex; align-items: center; gap: 8px; margin-left: auto; padding: 5px 10px 5px 12px; background: rgba(59,130,246,0.18); border: 1px solid rgba(96,165,250,0.5); border-radius: 999px; color: #dbeafe; font-size: 12px; font-weight: 600; white-space: nowrap; }
.link-dot { width: 7px; height: 7px; border-radius: 50%; background: #60a5fa; box-shadow: 0 0 0 3px rgba(96,165,250,0.25); flex-shrink: 0; }
.link-text { max-width: 320px; overflow: hidden; text-overflow: ellipsis; }
.link-clear { border: none; background: transparent; color: #bfdbfe; cursor: pointer; font-size: 12px; line-height: 1; padding: 2px 4px; border-radius: 4px; }
.link-clear:hover { background: rgba(255,255,255,0.15); color: #fff; }
.nav-brand { color: #fff; font-weight: 700; font-size: 15px; letter-spacing: 0.3px; white-space: nowrap; }
.nav-tabs { display: flex; gap: 4px; align-items: center; }
.more-wrap { position: relative; z-index: 81; }
.more-backdrop { position: fixed; inset: 0; z-index: 70; background: transparent; }
.more-menu { position: absolute; top: calc(100% + 4px); left: 0; z-index: 82; background: #20203a; border: 1px solid rgba(255,255,255,0.12); border-radius: 8px; padding: 4px; min-width: 140px; box-shadow: 0 12px 28px rgba(0,0,0,0.35); display: flex; flex-direction: column; gap: 2px; }
.more-item { padding: 7px 12px; border: none; background: transparent; color: rgba(255,255,255,0.7); cursor: pointer; font-size: 13px; border-radius: 6px; text-align: left; }
.more-item:hover { background: rgba(255,255,255,0.1); color: #fff; }
.more-item-active { background: rgba(255,255,255,0.16); color: #fff; font-weight: 600; }
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
