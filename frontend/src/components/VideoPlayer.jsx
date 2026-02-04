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

  // Stream results progressively - updates as frames are processed
  useEffect(() => {
    if (!jobId) {
      setLoading(false)
      return
    }

    const apiUrl = API_URL || import.meta.env.VITE_API_URL || 'https://vantage-ai-25ct.onrender.com'
    console.log('🔍 Starting progressive SSE stream for job:', jobId)
    
    const eventSource = new EventSource(`${apiUrl}/results-stream/${jobId}`)
    let lastFrameCount = 0
    
    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        const currentFrameCount = data.frames?.length || 0
        
        console.log('📦 Progressive update:', {
          status: data.status,
          framesCount: currentFrameCount,
          newFrames: currentFrameCount - lastFrameCount
        })
        
        if (data.status === 'error') {
          console.error('❌ Error status:', data)
          setLoading(false)
          eventSource.close()
          alert(`Error: ${data.message || data.error}`)
          return
        }
        
        // Update results progressively - show heatmap as frames come in
        if (data.frames && data.frames.length > 0) {
          setResults(data)
          lastFrameCount = currentFrameCount
          
          // Show loading until complete
          if (data.status === 'complete') {
            setLoading(false)
            eventSource.close()
            console.log('✅ Processing complete:', currentFrameCount, 'frames')
          } else {
            // Still processing - keep loading but show partial results
            setLoading(true)
            console.log('⏳ Processing...', currentFrameCount, 'frames so far')
          }
        } else if (data.status === 'processing') {
          // No frames yet, keep loading
          setLoading(true)
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
      // Fallback to polling
      const fallbackPoll = async () => {
        try {
          const response = await axios.get(`${apiUrl}/results/${jobId}`)
          const data = response.data
          if (data.frames && data.frames.length > 0) {
            setResults(data)
            if (data.status === 'complete') {
              setLoading(false)
            } else {
              setTimeout(fallbackPoll, 2000) // Poll every 2 seconds
            }
          } else {
            setTimeout(fallbackPoll, 2000)
          }
        } catch (err) {
          console.error('Fallback poll error:', err)
          setTimeout(fallbackPoll, 2000)
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
      
      // Find closest frame for smooth playback
      let frame = null
      let nextFrame = null
      let minTimeDiff = Infinity
      
      for (let i = 0; i < results.frames.length; i++) {
        const timeDiff = results.frames[i].time - currentTime
        
        if (timeDiff >= 0 && timeDiff < minTimeDiff) {
          minTimeDiff = timeDiff
          frame = results.frames[i]
          nextFrame = results.frames[i + 1] || frame
        }
        
        if (timeDiff < 0 && Math.abs(timeDiff) < minTimeDiff) {
          frame = results.frames[i]
        }
      }
      
      if (!frame && results.frames.length > 0) {
        frame = results.frames[results.frames.length - 1]
        nextFrame = frame
      }

      if (!frame?.saliency_heatmap) {
        canvas.style.display = 'none'
        return
      }

      canvas.style.display = 'block'
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // Smooth interpolation between frames
      let saliencyMap = frame.saliency_heatmap
      if (nextFrame && nextFrame.saliency_heatmap && nextFrame !== frame) {
        const timeDiff = nextFrame.time - frame.time
        const t = timeDiff > 0 ? (currentTime - frame.time) / timeDiff : 0
        const t_clamped = Math.max(0, Math.min(1, t))
        
        if (t_clamped > 0 && t_clamped < 1) {
          saliencyMap = frame.saliency_heatmap.map((row, y) => 
            row.map((val, x) => {
              const val2 = nextFrame.saliency_heatmap[y]?.[x] ?? val
              return val + (val2 - val) * t_clamped
            })
          )
        }
      }
      
      const heatmapH = saliencyMap.length
      const heatmapW = saliencyMap[0]?.length || 0
      if (heatmapH === 0 || heatmapW === 0) return

      const scaleX = canvas.width / heatmapW
      const scaleY = canvas.height / heatmapH

      // Smooth, rounded heatmap - circular/radial gradients for eye-tracking visualization
      const imgData = ctx.createImageData(canvas.width, canvas.height)
      
      // Create radial gradients around fixation points for rounded appearance
      const centerX = canvas.width / 2
      const centerY = canvas.height / 2
      
      for (let y = 0; y < canvas.height; y++) {
        for (let x = 0; x < canvas.width; x++) {
          const mapX = x / scaleX
          const mapY = y / scaleY
          const x1 = Math.floor(mapX)
          const y1 = Math.floor(mapY)
          const x2 = Math.min(x1 + 1, heatmapW - 1)
          const y2 = Math.min(y1 + 1, heatmapH - 1)
          
          const fx = mapX - x1
          const fy = mapY - y1
          
          let val = (saliencyMap[y1]?.[x1] || 0) * (1 - fx) * (1 - fy) +
                     (saliencyMap[y1]?.[x2] || 0) * fx * (1 - fy) +
                     (saliencyMap[y2]?.[x1] || 0) * (1 - fx) * fy +
                     (saliencyMap[y2]?.[x2] || 0) * fx * fy
          
          // Apply circular/radial smoothing around center
          const dx = x - centerX
          const dy = y - centerY
          const distFromCenter = Math.sqrt(dx * dx + dy * dy)
          const maxDist = Math.sqrt(centerX * centerX + centerY * centerY)
          const radialFactor = 1 - (distFromCenter / maxDist) * 0.3  // Slight radial falloff
          val = val * radialFactor
          
          if (val > 0.05) {
            const intensity = Math.min(1, val)
            const idx = (y * canvas.width + x) * 4
            
            // Smooth gradient: Blue → Teal → Green → Yellow → Orange → Red
            let r, g, b, alpha
            
            if (intensity < 0.2) {
              const t = intensity / 0.2
              r = Math.floor(30 + 20 * t)
              g = Math.floor(100 + 80 * t)
              b = Math.floor(180 - 50 * t)
              alpha = Math.floor(80 + 40 * t)
            } else if (intensity < 0.4) {
              const t = (intensity - 0.2) / 0.2
              r = Math.floor(50 - 20 * t)
              g = Math.floor(180 + 50 * t)
              b = Math.floor(130 - 100 * t)
              alpha = Math.floor(120 + 50 * t)
            } else if (intensity < 0.6) {
              const t = (intensity - 0.4) / 0.2
              r = Math.floor(30 + 180 * t)
              g = Math.floor(230 - 30 * t)
              b = Math.floor(30 - 30 * t)
              alpha = Math.floor(170 + 50 * t)
            } else if (intensity < 0.8) {
              const t = (intensity - 0.6) / 0.2
              r = Math.floor(210 + 45 * t)
              g = Math.floor(200 - 50 * t)
              b = Math.floor(0)
              alpha = Math.floor(220 + 35 * t)
            } else {
              const t = (intensity - 0.8) / 0.2
              r = Math.floor(255)
              g = Math.floor(150 - 150 * t)
              b = Math.floor(0)
              alpha = Math.floor(255)
            }
            
            imgData.data[idx] = r
            imgData.data[idx + 1] = g
            imgData.data[idx + 2] = b
            imgData.data[idx + 3] = alpha
          }
        }
      }
      
      // Apply heavy blur for smooth, rounded appearance
      ctx.putImageData(imgData, 0, 0)
      ctx.save()
      ctx.globalAlpha = 0.5
      ctx.filter = 'blur(5px)'  // Heavier blur for roundness
      ctx.drawImage(canvas, 0, 0)
      ctx.restore()
      
      // Draw with reduced opacity for smooth blending
      ctx.globalAlpha = 0.75
      ctx.putImageData(imgData, 0, 0)
      ctx.globalAlpha = 1.0
      
      ctx.putImageData(imgData, 0, 0)
      
      // Draw fixation points with clean annotations
      if (frame.fixation_points && frame.fixation_points.length > 0) {
        ctx.save()
        frame.fixation_points.forEach((point) => {
          const x = (point.x / frame.original_size[0]) * canvas.width
          const y = (point.y / frame.original_size[1]) * canvas.height
          const size = 10 + (point.intensity * 6)
          
          // Bright star
          ctx.fillStyle = `rgba(255, 180, 0, 0.95)`
          ctx.strokeStyle = '#FFFFFF'
          ctx.lineWidth = 2.5
          
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
          
          // Clean annotation
          ctx.fillStyle = '#FFFFFF'
          ctx.strokeStyle = '#000000'
          ctx.lineWidth = 3
          ctx.font = 'bold 11px Arial'
          ctx.textAlign = 'center'
          ctx.textBaseline = 'top'
          
          const label = `${frame.time.toFixed(1)}s`
          ctx.strokeText(label, x, y + size + 8)
          ctx.fillText(label, x, y + size + 8)
        })
        ctx.restore()
      }
      
      // Draw attention percentage annotation (top right)
      if (attentionPixels > 0) {
        const attentionPercent = ((totalAttention / attentionPixels) * 100).toFixed(1)
        ctx.save()
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)'
        ctx.fillRect(canvas.width - 120, 10, 110, 35)
        
        ctx.fillStyle = '#00FF88'
        ctx.font = 'bold 12px Arial'
        ctx.textAlign = 'left'
        ctx.fillText(`Attention: ${attentionPercent}%`, canvas.width - 115, 20)
        
        ctx.fillStyle = '#FFFFFF'
        ctx.font = '10px Arial'
        ctx.fillText(`Time: ${currentTime.toFixed(1)}s`, canvas.width - 115, 35)
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

        {results && results.frames && results.frames.length > 0 && results.stats && (
          <StatsCard>
            <StatRow>
              <StatLabel>Clarity Score</StatLabel>
              <StatValue>{Math.round(results.stats.clarity_score || 0)}/100</StatValue>
            </StatRow>
            <StatRow>
              <StatLabel>Engagement Score</StatLabel>
              <StatValue>{Math.round(results.stats.engagement_score || 0)}/100</StatValue>
            </StatRow>
            <StatRow>
              <StatLabel>Fixation Rate</StatLabel>
              <StatValue>{results.stats.fixation_rate?.toFixed(2) || '0'}/frame</StatValue>
            </StatRow>
            <StatRow>
              <StatLabel>Attention Stability</StatLabel>
              <StatValue>{Math.round(results.stats.attention_stability || 0)}/100</StatValue>
            </StatRow>
            <StatRow>
              <StatLabel>Total Fixations</StatLabel>
              <StatValue>{results.stats.total_fixations || 0}</StatValue>
            </StatRow>
            <StatRow>
              <StatLabel>Avg Saliency</StatLabel>
              <StatValue>{results.stats.avg_saliency?.toFixed(3) || '0'}</StatValue>
            </StatRow>
            <StatRow>
              <StatLabel>Processed Frames</StatLabel>
              <StatValue>{results.processed_frames || results.frames?.length || 0}</StatValue>
            </StatRow>
            <StatRow>
              <StatLabel>Processing Time</StatLabel>
              <StatValue>{results.processing_time?.toFixed(2) || '0'}s</StatValue>
            </StatRow>
          </StatsCard>
        )}
      </GlassCard>
    </Container>
  )
}

export default VideoPlayer
