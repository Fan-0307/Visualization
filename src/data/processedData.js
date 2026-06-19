import results from './processed_data/results.json'

export const modelLabels = {
  qwen: 'Qwen3-VL',
  llava: 'LLaVA',
  salesforce: 'Salesforce BLIP',
  openai: 'OpenAI CLIP',
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
      model_key: modelKey,
      sample_id: sampleId,
      image_id: entry.image_id,
      image_path: entry.image_path,
      image_src: `/img/${basename(entry.image_path)}`,
      question: entry.question,
      ground_truth: entry.ground_truth,
      prediction: result.answer,
      correct: !!result.correct,
      confidence: result.confidence ?? null,
      question_type: 'All',
    }

    matrixData.push(row)
    attentionData[`${model}__${sampleId}`] = row
  }
}
