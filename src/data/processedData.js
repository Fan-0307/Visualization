import results from './processed_data/results.json'

export const modelLabels = {
  qwen: 'Qwen2-VL',
  llava: 'LLaVA',
  salesforce: 'BLIP2',
  openai: 'CLIP',
}

export const modelFullNames = {
  qwen: 'Qwen2-VL-7B-Instruct',
  llava: 'llava-1.5-7b-hf',
  salesforce: 'blip2-opt-2.7b',
  openai: 'clip-vit-base-patch32',
}

export const modelKeys = results.models || Object.keys(modelLabels)
export const models = modelKeys.map(key => modelLabels[key] || key)

const photoModules = import.meta.glob('./processed_data/photos/*/*.png', {
  eager: true,
  query: '?url',
  import: 'default',
})

const photoUrls = Object.fromEntries(
  Object.entries(photoModules).map(([path, url]) => {
    const match = path.match(/photos\/([^/]+)\/([^/]+)\.png$/)
    return match ? [`${match[1]}__${match[2]}`, url] : [path, url]
  })
)

function basename(path) {
  return String(path || '').split('/').pop()
}

export function getPhotoUrl(modelKey, sampleId) {
  return photoUrls[`${modelKey}__${sampleId}`] || ''
}

export const sampleEntries = Object.entries(results.data || {}).map(([sampleId, entry]) => ({
  sample_id: sampleId,
  image_id: entry.image_id,
  image_path: entry.image_path,
  image_src: `/img/${basename(entry.image_path)}`,
  image_src_fallback: `/img/${entry.image_id}.jpeg`,
  question: entry.question,
  ground_truth: entry.ground_truth,
}))

export const sampleById = Object.fromEntries(sampleEntries.map(entry => [entry.sample_id, entry]))

export const matrixData = []
export const attentionData = {}

for (const [sampleId, entry] of Object.entries(results.data || {})) {
  for (const modelKey of modelKeys) {
    const result = Array.isArray(entry[modelKey]) ? entry[modelKey][0] : entry[modelKey]
    if (!result) continue

    const model = modelLabels[modelKey] || modelKey
    const row = {
      model,
      model_full_name: modelFullNames[modelKey] || model,
      model_key: modelKey,
      sample_id: sampleId,
      image_id: entry.image_id,
      image_path: entry.image_path,
      image_src: `/img/${basename(entry.image_path)}`,
      image_src_fallback: `/img/${entry.image_id}.jpeg`,
      question: entry.question,
      ground_truth: entry.ground_truth,
      prediction: result.answer,
      correct: !!result.correct,
      confidence: result.confidence ?? null,
      question_type: entry.question_type || inferQuestionType(entry.question),
    }

    matrixData.push(row)
    attentionData[`${model}__${sampleId}`] = row
  }
}

function inferQuestionType(question) {
  const text = String(question || '').trim().toLowerCase()
  if (!text) return 'other'
  if (text.startsWith('how many')) return 'how many'
  if (text.startsWith('what color') || text.includes(' what color') || text.includes(' is what color')) {
    return 'what color'
  }
  if (text.startsWith('what')) return 'what'
  if (text.startsWith('where')) return 'where'
  if (text.startsWith('which')) return 'which'
  if (text.startsWith('who') || text.startsWith('whose')) return 'who'
  if (/^(is|are|was|were|do|does|did|can|could|will|would|has|have|had)\b/.test(text)) {
    return 'yes/no'
  }
  return 'other'
}
