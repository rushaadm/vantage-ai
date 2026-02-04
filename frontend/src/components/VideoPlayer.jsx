import { useRef, useEffect, useState } from 'react'
import styled, { keyframes } from 'styled-components'
import axios from 'axios'
import { Download, RefreshCw, Loader2 } from 'lucide-react'
import { useStore } from '../store'

// API URL with fallback
const API_URL = import.meta.env.VITE_API_URL || 'https://vantage-ai-25ct.onrender.com'

const Container = styled.div`
  max-width: 1200px;
  margin: 0 auto;
`

const GlassCard = styled.div`
  ${props => props.theme.glass}
  padding: 2rem;
  margin-bottom: 2rem;
`

const VideoContainer = styled.div`
  position: relative;
  width: 100%;
  background-color: #000;
  min-height: 400px;
  border-radius: 12px;
  overflow: hidden;
`

const Video = styled.video`
  width: 100%;
  height: auto;
  display: block;
  background-color: #000;
`

const Canvas = styled.canvas`
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  z-index: 10;
  mix-blend-mode: screen;
  opacity: 0.7;
  display: none;
`

const Controls = styled.div`
  display: flex;
  gap: 1rem;
  margin-top: 1rem;
  flex-wrap: wrap;
`

const Button = styled.button`
  ${props => props.theme.glass}
  padding: 0.75rem 1.5rem;
  border: none;
  color: ${props => props.theme.colors.text};
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  transition: all 0.3s ease;
  
  &:hover {
    background-color: rgba(0, 242, 255, 0.2);
    border-color: ${props => props.theme.colors.cyan};
  }
  
  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`

const StatsCard = styled.div`
  ${props => props.theme.glass}
  padding: 1.5rem;
  margin-top: 1rem;
`

const StatRow = styled.div`
  display: flex;
  justify-content: space-between;
  margin-bottom: 0.5rem;
  
  &:last-child {
    margin-bottom: 0;
  }
`

const StatLabel = styled.span`
  color: ${props => props.theme.colors.textSecondary};
`

const StatValue = styled.span`
  color: ${props => props.theme.colors.cyan};
  font-weight: bold;
`

const spin = keyframes`
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
`

const LoadingSpinner = styled(Loader2)`
  animation: ${spin} 1s linear infinite;
  color: ${props => props.theme.colors.cyan};
  width: 48px;
  height: 48px;
`

const LoadingContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 2rem;
  text-align: center;
`

const LoadingText = styled.p`
  color: ${props => props.theme.colors.textSecondary};
  margin-top: 1rem;
`

function VideoPlayer() {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const { jobId, videoUrl, results, setResults } = useStore()
  const [loading, setLoading] = useState(true)
  const [heatmapWorker, setHeatmapWorker] = useState(null)

  // Fetch results
  useEffect(() => {
    if (!jobId) {
      setLoading(false)
      return
    }

    const fetchResults = async () => {
      try {
        const apiUrl = API_URL || import.meta.env.VITE_API_URL || 'https://vantage-ai-25ct.onrender.com'
        console.log('🔍 Fetching results from:', `${apiUrl}/results/${jobId}`)
        const response = await axios.get(`${apiUrl}/results/${jobId}`)
        
        console.log('📦 Response received:', {
          status: response.data.status,
          hasFrames: !!response.data.frames,
          framesCount: response.data.frames?.length || 0,
          keys: Object.keys(response.data)
        })
        
        if (response.data.status === 'error') {
          console.error('❌ Error status:', response.data)
          setLoading(false)
          alert(`Error: ${response.data.message || response.data.error}`)
          return
        }
        
        // Check if still processing (no frames yet or status is processing)
        if (response.data.status === 'processing' || !response.data.frames || response.data.frames.length === 0) {
          console.log('⏳ Still processing...', {
            status: response.data.status,
            message: response.data.message,
            hasFrames: !!response.data.frames
          })
          setTimeout(fetchResults, 2000)
          return
        }
        
        // Results ready!
        if (response.data.frames && response.data.frames.length > 0) {
          console.log('✅ Results loaded:', response.data.frames.length, 'frames')
          console.log('Results data:', {
            status: response.data.status,
            framesCount: response.data.frames.length,
            hasHeatmap: !!response.data.frames[0]?.saliency_heatmap,
            clarityScore: response.data.clarity_score,
            firstFrameTime: response.data.frames[0]?.time
          })
          setResults(response.data)
          setLoading(false)
        } else {
          console.log('⏳ Waiting for frames...', response.data)
          setTimeout(fetchResults, 2000)
        }
      } catch (error) {
        console.error('❌ Fetch error:', error)
        if (error.response?.status === 404) {
          console.log('⏳ Job not found yet, retrying...')
          setTimeout(fetchResults, 2000)
        } else {
          console.error('Error details:', error.response?.data || error.message)
          setLoading(false)
          alert(`Failed to fetch results: ${error.message}`)
        }
      }
    }

    fetchResults()
  }, [jobId, setResults])

  // Initialize Web Worker
  useEffect(() => {
    try {
      const worker = new Worker(new URL('../workers/heatmapWorker.js', import.meta.url), { type: 'module' })
      setHeatmapWorker(worker)
      return () => worker.terminate()
    } catch (error) {
      console.warn('Worker not available')
      setHeatmapWorker(null)
    }
  }, [])

  // Render heatmap
  useEffect(() => {
    if (!videoRef.current || !canvasRef.current) {
      console.log('⚠️ VideoPlayer: Missing video or canvas ref')
      return
    }
    
    if (!results?.frames || results.frames.length === 0) {
      console.log('⚠️ VideoPlayer: No results or frames available', { hasResults: !!results, framesCount: results?.frames?.length })
      return
    }

    console.log('✅ VideoPlayer: Starting heatmap rendering', {
      framesCount: results.frames.length,
      hasHeatmap: !!results.frames[0]?.saliency_heatmap
    })

    const video = videoRef.current
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d', { alpha: true })

    const updateCanvas = () => {
      if (video.videoWidth === 0 || video.videoHeight === 0) return

      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth
        canvas.height = video.videoHeight
      }

      const currentTime = video.currentTime
      const fps = results.fps || 30
      
      // Dynamic frame interpolation for smooth, changing heatmaps
      let frame1 = null
      let frame2 = null
      let t = 0
      
      // Find frames before and after current time for interpolation
      for (let i = 0; i < results.frames.length; i++) {
        if (results.frames[i].time >= currentTime) {
          frame2 = results.frames[i]
          if (i > 0) {
            frame1 = results.frames[i - 1]
            const timeDiff = frame2.time - frame1.time
            if (timeDiff > 0) {
              t = (currentTime - frame1.time) / timeDiff
              t = Math.max(0, Math.min(1, t))
            }
          } else {
            frame1 = frame2
          }
          break
        }
      }
      
      // Use last frame if past all frames
      if (!frame1 && results.frames.length > 0) {
        frame1 = results.frames[results.frames.length - 1]
        frame2 = frame1
      }
      
      const frame = frame1

      if (!frame?.saliency_heatmap) {
        console.log('⚠️ No heatmap data for frame at time', currentTime)
        canvas.style.display = 'none'
        return
      }

      canvas.style.display = 'block'
      console.log('🎨 Rendering heatmap for time', currentTime.toFixed(2), 's')
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // Interpolate between frames for dynamic, smooth heatmap changes
      let saliencyMap = frame.saliency_heatmap
      if (frame2 && frame2.saliency_heatmap && t > 0 && t < 1) {
        // Interpolate between two frames for smooth transitions
        const map1 = frame.saliency_heatmap
        const map2 = frame2.saliency_heatmap
        saliencyMap = map1.map((row, y) => 
          row.map((val, x) => {
            const val2 = map2[y]?.[x] ?? val
            return val + (val2 - val) * t
          })
        )
      }
      
      const heatmapH = saliencyMap.length
      const heatmapW = saliencyMap[0]?.length || 0
      if (heatmapH === 0 || heatmapW === 0) return

      const scaleX = canvas.width / heatmapW
      const scaleY = canvas.height / heatmapH

      // Granular bilinear interpolation for smooth, pixel-level heatmap
      const imgData = ctx.createImageData(canvas.width, canvas.height)
      
      for (let y = 0; y < canvas.height; y++) {
        for (let x = 0; x < canvas.width; x++) {
          // Bilinear interpolation for granular, smooth heatmap
          const mapX = x / scaleX
          const mapY = y / scaleY
          const x1 = Math.floor(mapX)
          const y1 = Math.floor(mapY)
          const x2 = Math.min(x1 + 1, heatmapW - 1)
          const y2 = Math.min(y1 + 1, heatmapH - 1)
          
          const fx = mapX - x1
          const fy = mapY - y1
          
          const val11 = saliencyMap[y1]?.[x1] || 0
          const val21 = saliencyMap[y1]?.[x2] || 0
          const val12 = saliencyMap[y2]?.[x1] || 0
          const val22 = saliencyMap[y2]?.[x2] || 0
          
          // Bilinear interpolation
          const val = val11 * (1 - fx) * (1 - fy) +
                     val21 * fx * (1 - fy) +
                     val12 * (1 - fx) * fy +
                     val22 * fx * fy
          
          if (val > 0.2) {
            // Dynamic gradient: bright green for high attention, darker for lower
            const intensity = Math.min(1, (val - 0.2) / 0.8)
            const idx = (y * canvas.width + x) * 4
            
            // Green gradient: bright lime to forest green
            const green = Math.floor(50 + 205 * intensity)  // 50-255 range
            const alpha = Math.floor(120 + 135 * intensity)  // 120-255 opacity
            
            imgData.data[idx] = 0           // R
            imgData.data[idx + 1] = green  // G - dynamic green
            imgData.data[idx + 2] = Math.floor(50 * (1 - intensity))  // B - slight blue tint for lower values
            imgData.data[idx + 3] = alpha   // A - dynamic opacity
          }
        }
      }

      // Apply Gaussian blur for smoother appearance
      ctx.putImageData(imgData, 0, 0)
      
      // Additional smoothing pass
      const tempCanvas = document.createElement('canvas')
      tempCanvas.width = canvas.width
      tempCanvas.height = canvas.height
      const tempCtx = tempCanvas.getContext('2d')
      tempCtx.putImageData(imgData, 0, 0)
      
      // Draw blurred version with lower opacity for smoothness
      ctx.save()
      ctx.globalAlpha = 0.7
      ctx.filter = 'blur(2px)'
      ctx.drawImage(tempCanvas, 0, 0)
      ctx.restore()
      
      // Draw sharp version on top
      ctx.putImageData(imgData, 0, 0)
    }

    video.addEventListener('timeupdate', updateCanvas)
    const rafId = requestAnimationFrame(function animate() {
      updateCanvas()
      requestAnimationFrame(animate)
    })

    return () => {
      video.removeEventListener('timeupdate', updateCanvas)
      cancelAnimationFrame(rafId)
    }
  }, [results])

  const handleDownloadPDF = async () => {
    try {
      const response = await axios.get(`${API_URL}/download-pdf/${jobId}`, {
        responseType: 'blob'
      })
      const url = window.URL.createObjectURL(new Blob([response.data]))
      const link = document.createElement('a')
      link.href = url
      link.setAttribute('download', `vantage_report_${jobId}.pdf`)
      document.body.appendChild(link)
      link.click()
      link.remove()
    } catch (error) {
      alert('Failed to download PDF')
    }
  }

  const handleReset = () => {
    useStore.getState().reset()
    if (videoUrl) {
      URL.revokeObjectURL(videoUrl)
    }
  }

  // SIMPLE RENDERING - NO COMPLEX LOGIC
  if (!videoUrl) {
    return null
  }

  console.log('VideoPlayer render:', {
    videoUrl: !!videoUrl,
    jobId,
    loading,
    hasResults: !!results,
    resultsFrames: results?.frames?.length || 0,
    resultsStatus: results?.status
  })

  return (
    <Container>
      {loading && (
        <GlassCard>
          <LoadingContainer>
            <LoadingSpinner />
            <LoadingText>Processing video...</LoadingText>
          </LoadingContainer>
        </GlassCard>
      )}

      <GlassCard>
        {/* Debug info */}
        <div style={{ marginBottom: '1rem', padding: '0.5rem', background: 'rgba(0, 242, 255, 0.1)', borderRadius: '8px', fontSize: '0.875rem', color: '#00F2FF' }}>
          Debug: loading={loading ? 'yes' : 'no'}, results={results ? `✅ (${results.frames?.length || 0} frames)` : '❌'}, jobId={jobId || 'none'}
        </div>

        <VideoContainer>
          <Video 
            ref={videoRef}
            src={videoUrl}
            controls
            style={{ display: 'block', width: '100%' }}
          />
          <Canvas ref={canvasRef} style={{ display: results?.frames?.length > 0 ? 'block' : 'none' }} />
        </VideoContainer>

        <Controls>
          <Button onClick={handleDownloadPDF} disabled={!results || !results.frames || results.frames.length === 0}>
            <Download size={20} />
            Download PDF Report {results?.frames?.length > 0 ? `(${results.frames.length} frames)` : '(no data)'}
          </Button>
          <Button onClick={handleReset}>
            <RefreshCw size={20} />
            Upload New Video
          </Button>
        </Controls>

        {results && results.frames && results.frames.length > 0 && (
          <StatsCard>
            <StatRow>
              <StatLabel>Clarity Score</StatLabel>
              <StatValue>{Math.round(results.clarity_score || 0)}/100</StatValue>
            </StatRow>
            <StatRow>
              <StatLabel>Processed Frames</StatLabel>
              <StatValue>{results.processed_frames || results.frames?.length || 0}</StatValue>
            </StatRow>
            <StatRow>
              <StatLabel>Processing Time</StatLabel>
              <StatValue>{results.processing_time?.toFixed(2) || '0'}s</StatValue>
            </StatRow>
            <StatRow>
              <StatLabel>Heatmap Status</StatLabel>
              <StatValue>{results.frames[0]?.saliency_heatmap ? '✅ Ready' : '❌ No data'}</StatValue>
            </StatRow>
          </StatsCard>
        )}
      </GlassCard>
    </Container>
  )
}

export default VideoPlayer
