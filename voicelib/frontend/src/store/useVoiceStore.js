import { create } from 'zustand'
import { voicesApi } from '../api/voices'

export const useVoiceStore = create((set, get) => ({
  voices: [],
  isLoading: false,
  error: null,

  fetchVoices: async () => {
    set({ isLoading: true, error: null })
    try {
      const voices = await voicesApi.list()
      set({ voices, isLoading: false })
    } catch (err) {
      set({ error: err.message, isLoading: false })
    }
  },

  addVoice: (voice) =>
    set((s) => ({ voices: [voice, ...s.voices] })),

  updateVoice: (updatedVoice) =>
    set((s) => ({
      voices: s.voices.map((v) => (v.id === updatedVoice.id ? updatedVoice : v)),
    })),

  removeVoice: (id) =>
    set((s) => ({ voices: s.voices.filter((v) => v.id !== id) })),

  clearError: () => set({ error: null }),
}))
