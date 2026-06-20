import { reactive } from 'vue'

export const selection = reactive({
  sampleId: null,
  model: null,
  questionType: null,
  layerIdx: null,
  source: null,
})

// Legacy signature kept for existing callers (matrix / agg / layer broadcast sampleId).
export function broadcast(sampleId, model, source) {
  if (sampleId != null) selection.sampleId = sampleId
  if (model != null) selection.model = model
  selection.source = source
}

// Generic field-based broadcast. Any subset of {sampleId, model, questionType} can be sent.
// Used by the overview (Tab1) to drive a drill-down into the sample-diagnosis view.
export function broadcastFields(fields, source) {
  if (fields.sampleId !== undefined) selection.sampleId = fields.sampleId
  if (fields.model !== undefined) selection.model = fields.model
  if (fields.questionType !== undefined) selection.questionType = fields.questionType
  selection.source = source
}

// Clear the cross-view linked selection (used by the "clear link" breadcrumb).
export function clearSelection() {
  selection.sampleId = null
  selection.model = null
  selection.questionType = null
  selection.source = null
}

// Called by a target view on mount. Returns the pending cross-view selection
// (or null if it originated in this same view, or nothing has been selected yet).
export function consume(source) {
  if (selection.source === source) return null
  if (selection.sampleId == null && selection.questionType == null && selection.model == null) return null
  return {
    sampleId: selection.sampleId,
    model: selection.model,
    questionType: selection.questionType,
  }
}
