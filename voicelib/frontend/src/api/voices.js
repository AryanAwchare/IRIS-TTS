import client from './client'

export const voicesApi = {
  list: () => client.get('/voices').then((r) => r.data),

  create: (formData) => client.post('/voices', formData).then((r) => r.data),

  remove: (voiceId) => client.delete(`/voices/${voiceId}`),

  updateSettings: (voiceId, settings) =>
    client.patch(`/voices/${voiceId}/settings`, settings).then((r) => r.data),
}
