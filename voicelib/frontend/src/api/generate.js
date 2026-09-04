import client from './client'

export const generateApi = {
  generate: (voiceId, text, options = {}) =>
    client.post('/generate', { voice_id: voiceId, text, ...options }).then((r) => r.data),

  history: (params = {}) =>
    client.get('/generations', { params }).then((r) => r.data),

  similarity: (generationId) =>
    client.get(`/generations/${generationId}/similarity`).then((r) => r.data),

  getEval: (generationId) =>
    client.get(`/generations/${generationId}/eval`).then((r) => r.data),

  getColabStatus: () =>
    client.get('/colab-status').then((r) => r.data),

  // Model Switcher — live engine status
  engineStatus: () =>
    client.get('/engines/status').then((r) => r.data),

  // Pocket TTS Studio presets
  presets: () =>
    client.get('/engines/presets').then((r) => r.data),

  downloadAudio: async (generationId, format = 'wav', customFilename = null) => {
    const res = await client.get(`/generations/${generationId}/download`, {
      params: { format },
      responseType: 'blob',
    })
    const blob = new Blob([res.data], {
      type: format === 'mp3' ? 'audio/mpeg' : 'audio/wav',
    })
    const url = window.URL.createObjectURL(blob)
    const link = document.createElement('a')
    const filename = customFilename || `voicelib-${generationId.slice(0, 8)}.${format}`
    link.href = url
    link.download = filename
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    window.URL.revokeObjectURL(url)
    return true
  },
}

