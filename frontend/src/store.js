import { create } from 'zustand'

export const useStore = create((set) => ({
  jobId: null,
  videoUrl: null,
  results: null,
  frameSamplingRate: 2, // Default: every 2nd frame (50%)
  setJobId: (jobId) => set({ jobId }),
  setVideoUrl: (videoUrl) => set({ videoUrl }),
  setResults: (results) => set({ results }),
  setFrameSamplingRate: (rate) => set({ frameSamplingRate: rate }),
  reset: () => set({ jobId: null, videoUrl: null, results: null, frameSamplingRate: 2 })
}))
