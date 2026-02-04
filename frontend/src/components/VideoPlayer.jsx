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

  // Stream results using Server-Sent Events - NO POLLING!
  useEffect(() => {
    if (!jobId) {
      setLoading(false)
      return
    }

    const apiUrl = API_URL || import.meta.env.VITE_API_URL || 'https://vantage-ai-25ct.onrender.com'
    console.log('🔍 Starting SSE stream for job:', jobId)
    
    // Use Server-Sent Events instead of polling
    const eventSource = new EventSource(`${apiUrl}/results-stream/${jobId}`)
    
    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        console.log('📦 SSE data received:', {
          status: data.status,
          hasFrames: !!data.frames,
          framesCount: data.frames?.length || 0
        })
        
        if (data.status === 'error') {
          console.error('❌ Error status:', data)
          setLoading(false)
          eventSource.close()
          alert(`Error: ${data.message || data.error}`)
          return
        }
        
        if (data.status === 'processing') {
          console.log('⏳ Processing...', data.message)
          // Keep loading state - SSE will push updates automatically
          return
        }
        
        // Results ready!
        if (data.frames && data.frames.length > 0) {
          console.log('✅ Results loaded via SSE:', data.frames.length, 'frames')
          setResults(data)
          setLoading(false)
          eventSource.close() // Close connection once we have results
        }
      } catch (error) {
        console.error('❌ SSE parse error:', error)
        eventSource.close()
        setLoading(false)
      }
    }
    
    eventSource.onerror = (error) => {
      console.error('❌ SSE connection error:', error)
      eventSource.close()
      // Fallback to regular polling if SSE fails
      console.log('🔄 Falling back to polling...')
      const fallbackPoll = async () => {
        try {
          const response = await axios.get(`${apiUrl}/results/${jobId}`)
          if (response.data.status === 'complete' && response.data.frames?.length > 0) {
            setResults(response.data)
            setLoading(false)
          } else {
            setTimeout(fallbackPoll, 3000) // Poll every 3 seconds as fallback
          }
        } catch (err) {
          console.error('Fallback poll error:', err)
          setTimeout(fallbackPoll, 3000)
        }
      }
      fallbackPoll()
    }
    
    return () => {
      eventSource.close()
    }
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
      
      // Find exact frame matching current video time (all frames processed)
      let frame = null
      let closestFrame = null
      let minTimeDiff = Infinity
      
      // Find frame with time closest to currentTime
      for (let i = 0; i < results.frames.length; i++) {
        const timeDiff = Math.abs(results.frames[i].time - currentTime)
        if (timeDiff < minTimeDiff) {
          minTimeDiff = timeDiff
          closestFrame = results.frames[i]
        }
        // Exact match (within 0.05 seconds)
        if (timeDiff < 0.05) {
          frame = results.frames[i]
          break
        }
      }
      
      // Use closest frame if no exact match
      if (!frame && closestFrame) {
        frame = closestFrame
      }
      
      // Fallback to last frame
      if (!frame && results.frames.length > 0) {
        frame = results.frames[results.frames.length - 1]
      }

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
          
          // Brighter, more generalized - only show main focus areas
          if (val > 0.1) {  // Lower threshold for brighter display
            // Bright green gradient - more intense
            const intensity = Math.min(1, val * 1.5)  // Amplify brightness
            const idx = (y * canvas.width + x) * 4
            
            // Bright lime green to bright yellow-green
            const green = Math.floor(100 + 155 * intensity)  // 100-255 (brighter)
            const red = Math.floor(50 * intensity)  // Slight red tint for warmth
            const alpha = Math.floor(150 + 105 * intensity)  // 150-255 opacity (more visible)
            
            imgData.data[idx] = red           // R - slight red
            imgData.data[idx + 1] = green    // G - bright green
            imgData.data[idx + 2] = 0        // B
            imgData.data[idx + 3] = alpha     // A - brighter
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
      
      // Draw fixation points (bright stars)
      if (frame.fixation_points && frame.fixation_points.length > 0) {
        ctx.save()
        frame.fixation_points.forEach((point, idx) => {
          const x = (point.x / frame.original_size[0]) * canvas.width
          const y = (point.y / frame.original_size[1]) * canvas.height
          const size = 8 + (point.intensity * 8)  // 8-16px based on intensity
          
          // Bright orange/red star
          ctx.fillStyle = `rgba(255, ${200 - point.intensity * 100}, 0, 0.9)`
          ctx.strokeStyle = '#FFFFFF'
          ctx.lineWidth = 2
          
          // Draw star shape
          ctx.beginPath()
          for (let i = 0; i < 5; i++) {
            const angle = (i * 4 * Math.PI) / 5 - Math.PI / 2
            const px = x + Math.cos(angle) * size
            const py = y + Math.sin(angle) * size
            if (i === 0) ctx.moveTo(px, py)
            else ctx.lineTo(px, py)
          }
          ctx.closePath()
          ctx.fill()
          ctx.stroke()
          
          // Time annotation
          ctx.fillStyle = '#FFFFFF'
          ctx.font = 'bold 10px Arial'
          ctx.textAlign = 'center'
          ctx.fillText(frame.time.toFixed(2) + 's', x, y + size + 12)
        })
        ctx.restore()
      }
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
