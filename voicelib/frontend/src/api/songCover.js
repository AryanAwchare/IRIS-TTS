import client from './client'

export const songCoverApi = {
  create: (formData) =>
    client.post('/song-covers', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }).then((r) => r.data),

  getStatus: (coverId) =>
    client.get(`/song-covers/${coverId}/status`).then((r) => r.data),

  getDetail: (coverId) =>
    client.get(`/song-covers/${coverId}`).then((r) => r.data),

  list: (params = {}) =>
    client.get('/song-covers', { params }).then((r) => r.data),

  getCuratedCatalog: () =>
    client.get('/song-covers/curated').then((r) => r.data),

  getLibrarySongs: () =>
    client.get('/song-covers/library').then((r) => r.data),
}
