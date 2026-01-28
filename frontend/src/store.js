import { create } from 'zustand'

export const useStore = create((set) => ({
  jobId: null,
  videoUrl: null,
  results: null,
  setJobId: (jobId) => set({ jobId }),
  setVideoUrl: (videoUrl) => set({ videoUrl }),
  setResults: (results) => set({ results }),
  reset: () => set({ jobId: null, videoUrl: null, results: null })
}))
