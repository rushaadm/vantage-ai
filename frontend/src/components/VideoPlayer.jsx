import { useRef, useEffect, useState } from 'react'
import styled from 'styled-components'
import axios from 'axios'
import { Download, RefreshCw } from 'lucide-react'
import { useStore } from '../store'

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
  max-width: 100%;
  margin-bottom: 1rem;
`

const Video = styled.video`
  width: 100%;
  height: auto;
  border-radius: 12px;
  display: block;
`

const Canvas = styled.canvas`
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  border-radius: 12px;
  z-index: 10;
  mix-blend-mode: normal;
  opacity: 0.8;
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

const AnnotationLayer = styled.div`
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  z-index: 11;
  border-radius: 12px;
`

const AnnotationText = styled.div`
  position: absolute;
  background: rgba(0, 0, 0, 0.8);
  color: #00FF00;
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
  font-size: 0.75rem;
  font-weight: bold;
  border: 1px solid #00FF00;
`

function VideoPlayer() {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const annotationRef = useRef(null)
  const { jobId, videoUrl, results, setResults } = useStore()
  const [loading, setLoading] = useState(true)
  const [currentFrame, setCurrentFrame] = useState(null)
  const [attentionZones, setAttentionZones] = useState([])

  useEffect(() => {
    if (!jobId) return

    const fetchResults = async () => {
      try {
        const response = await axios.get(`http://localhost:8000/results/${jobId}`)
        console.log('Results response:', response.data)
        
        if (response.data.status === 'processing' || response.data.message) {
          setTimeout(fetchResults, 2000)
          return
        }
        
        if (response.data.frames && response.data.frames.length > 0) {
          console.log('Results ready with', response.data.frames.length, 'frames')
          setResults(response.data)
          setLoading(false)
        } else {
          console.warn('Results ready but no frames:', response.data)
          setLoading(false)
        }
      } catch (error) {
        if (error.response?.status === 404 || error.response?.status === 500) {
          setTimeout(fetchResults, 2000)
        } else {
          console.error('Error fetching results:', error)
          setLoading(false)
        }
      }
    }

    fetchResults()
  }, [jobId, setResults])

  useEffect(() => {
    if (!videoRef.current || !canvasRef.current || !results) return

    const video = videoRef.current
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    
    if (results.frames && results.frames.length > 0) {
      const firstFrame = results.frames[0]
      console.log('VideoPlayer: Results loaded', {
        frameCount: results.frames.length,
        fps: results.fps,
        firstFrame: {
          time: firstFrame.time,
          hasHeatmap: !!firstFrame.saliency_heatmap,
          heatmapType: Array.isArray(firstFrame.saliency_heatmap) ? 'array' : typeof firstFrame.saliency_heatmap,
          heatmapLength: firstFrame.saliency_heatmap?.length
        }
      })
    }
    
    const handleLoadedMetadata = () => {
      if (video.videoWidth > 0 && video.videoHeight > 0) {
        canvas.width = video.videoWidth
        canvas.height = video.videoHeight
        console.log('Canvas initialized:', canvas.width, 'x', canvas.height)
      }
    }
    
    video.addEventListener('loadedmetadata', handleLoadedMetadata)
    if (video.readyState >= 1) {
      handleLoadedMetadata()
    }

    const updateCanvas = () => {
      const currentTime = video.currentTime
      const fps = results.fps || 30
      
      if (!results.frames || results.frames.length === 0) return

      // Find closest frame by time
      let closestIdx = 0
      let minTimeDiff = Infinity
      
      for (let i = 0; i < results.frames.length; i++) {
        const timeDiff = Math.abs(results.frames[i].time - currentTime)
        if (timeDiff < minTimeDiff) {
          minTimeDiff = timeDiff
          closestIdx = i
        }
      }
      
      let frame1 = results.frames[closestIdx]
      let frame2 = null
      let t = 0
      
      if (closestIdx < results.frames.length - 1 && results.frames[closestIdx + 1].time > currentTime) {
        frame2 = results.frames[closestIdx + 1]
        const timeDiff = frame2.time - frame1.time
        if (timeDiff > 0) {
          t = (currentTime - frame1.time) / timeDiff
          t = Math.max(0, Math.min(1, t))
        }
      } else if (closestIdx > 0 && results.frames[closestIdx - 1].time < currentTime) {
        frame2 = frame1
        frame1 = results.frames[closestIdx - 1]
        const timeDiff = frame2.time - frame1.time
        if (timeDiff > 0) {
          t = (currentTime - frame1.time) / timeDiff
          t = Math.max(0, Math.min(1, t))
        }
      } else {
        frame2 = frame1
      }
      
      if (!frame1 || !frame1.saliency_heatmap || !Array.isArray(frame1.saliency_heatmap)) {
        ctx.clearRect(0, 0, canvas.width, canvas.height)
        return
      }
      
      const currentFrame = {
        entropy: frame1.entropy + (frame2 ? (frame2.entropy - frame1.entropy) * t : 0),
        conflict: frame1.conflict + (frame2 ? (frame2.conflict - frame1.conflict) * t : 0)
      }
      setCurrentFrame(currentFrame)

      // Ensure video dimensions are available
      if (video.videoWidth === 0 || video.videoHeight === 0) {
        return
      }
      
      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth
        canvas.height = video.videoHeight
      }

      if (frame1.saliency_heatmap) {
        // Get heatmap data (interpolate if needed)
        let saliencyMap = frame1.saliency_heatmap
        if (frame2 && frame2.saliency_heatmap && t > 0 && t < 1) {
          const map1 = frame1.saliency_heatmap
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
        
        if (heatmapH === 0 || heatmapW === 0) {
          ctx.clearRect(0, 0, canvas.width, canvas.height)
          return
        }
        
        const scaleX = canvas.width / heatmapW
        const scaleY = canvas.height / heatmapH
        
        // Find min/max for contrast
        let minVal = Infinity
        let maxVal = -Infinity
        for (let y = 0; y < heatmapH; y++) {
          for (let x = 0; x < heatmapW; x++) {
            const val = saliencyMap[y]?.[x] || 0
            if (val < minVal) minVal = val
            if (val > maxVal) maxVal = val
          }
        }
        const range = maxVal - minVal || 1
        
        // Simplified detection - lighter on CPU
        const fixationPoints = []
        const zones = []
        const threshold = minVal + range * 0.7
        const zoneGridSize = 4 // Smaller grid (4x4 instead of 8x8) for performance
        
        // Simplified attention zones
        const zoneWidth = Math.floor(heatmapW / zoneGridSize)
        const zoneHeight = Math.floor(heatmapH / zoneGridSize)
        let totalAttention = 0
        
        // Sample zones (every other one) to reduce computation
        for (let zoneY = 0; zoneY < zoneGridSize; zoneY += 1) {
          for (let zoneX = 0; zoneX < zoneGridSize; zoneX += 1) {
            let zoneAttention = 0
            const startY = zoneY * zoneHeight
            const endY = Math.min(startY + zoneHeight, heatmapH)
            const startX = zoneX * zoneWidth
            const endX = Math.min(startX + zoneWidth, heatmapW)
            
            // Sample pixels instead of checking all
            for (let y = startY; y < endY; y += 2) {
              for (let x = startX; x < endX; x += 2) {
                const val = saliencyMap[y]?.[x] || 0
                zoneAttention += val
                totalAttention += val
              }
            }
            
            zones.push({
              x: zoneX,
              y: zoneY,
              attention: zoneAttention * 4, // Approximate for sampled pixels
              screenX: (startX + endX) / 2 * scaleX,
              screenY: (startY + endY) / 2 * scaleY
            })
          }
        }
        
        zones.forEach(zone => {
          zone.percentage = totalAttention > 0 ? (zone.attention / (totalAttention * 4)) * 100 : 0
        })
        
        const significantZones = zones.filter(z => z.percentage > 2 && z.screenX && z.screenY) // Higher threshold + safety check
        setAttentionZones(significantZones)
        
        // Simplified fixation detection - sample fewer points
        for (let y = 2; y < heatmapH - 2; y += 2) {
          for (let x = 2; x < heatmapW - 2; x += 2) {
            const val = saliencyMap[y]?.[x] || 0
            if (val < threshold) continue
            
            // Simple check - just compare with immediate neighbors
            let isMax = true
            for (let dy = -1; dy <= 1 && isMax; dy++) {
              for (let dx = -1; dx <= 1 && isMax; dx++) {
                if (dx === 0 && dy === 0) continue
                const neighborVal = saliencyMap[y + dy]?.[x + dx] || 0
                if (neighborVal > val) isMax = false
              }
            }
            
            if (isMax) {
              fixationPoints.push({
                x: x * scaleX,
                y: y * scaleY,
                intensity: val
              })
            }
          }
        }
        
        // Draw bright green heatmap
        const imgData = ctx.createImageData(canvas.width, canvas.height)

        for (let y = 0; y < canvas.height; y++) {
          for (let x = 0; x < canvas.width; x++) {
            const idx = (y * canvas.width + x) * 4
            const mapY = Math.min(Math.floor(y / scaleY), heatmapH - 1)
            const mapX = Math.min(Math.floor(x / scaleX), heatmapW - 1)
            let saliencyValue = saliencyMap[mapY]?.[mapX] || 0
            
            saliencyValue = (saliencyValue - minVal) / range
            saliencyValue = Math.pow(saliencyValue, 0.8)
            
            // Bright green circles/areas - highly visible
            const intensity = Math.max(150, Math.min(255, saliencyValue * 255))
            const greenIntensity = Math.floor(100 + saliencyValue * 155) // 100-255 bright green
            imgData.data[idx] = 0
            imgData.data[idx + 1] = greenIntensity
            imgData.data[idx + 2] = Math.floor(greenIntensity * 0.3)
            imgData.data[idx + 3] = intensity
          }
        }

        ctx.putImageData(imgData, 0, 0)
        
        // Draw bigger red stars for fixation points
        ctx.save()
        fixationPoints.forEach(point => {
          const size = 16 + (point.intensity - minVal) / range * 8 // 16-24px stars (bigger)
          
          ctx.fillStyle = '#FF0000'
          ctx.strokeStyle = '#FFFFFF'
          ctx.lineWidth = 2.5
          
          ctx.beginPath()
          const spikes = 5
          const outerRadius = size / 2
          const innerRadius = outerRadius * 0.4
          
          for (let i = 0; i < spikes * 2; i++) {
            const radius = i % 2 === 0 ? outerRadius : innerRadius
            const angle = (i * Math.PI) / spikes - Math.PI / 2
            const x = point.x + radius * Math.cos(angle)
            const y = point.y + radius * Math.sin(angle)
            
            if (i === 0) {
              ctx.moveTo(x, y)
            } else {
              ctx.lineTo(x, y)
            }
          }
          ctx.closePath()
          ctx.fill()
          ctx.stroke()
        })
        ctx.restore()
      } else {
        ctx.clearRect(0, 0, canvas.width, canvas.height)
      }
    }

    video.addEventListener('timeupdate', updateCanvas)
    const rafId = requestAnimationFrame(function animate() {
      updateCanvas()
      requestAnimationFrame(animate)
    })

    return () => {
      video.removeEventListener('timeupdate', updateCanvas)
      video.removeEventListener('loadedmetadata', handleLoadedMetadata)
      cancelAnimationFrame(rafId)
    }
  }, [results])

  const handleDownloadPDF = async () => {
    try {
      const response = await axios.get(`http://localhost:8000/download-pdf/${jobId}`, {
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
      console.error('Error downloading PDF:', error)
      alert('Failed to download PDF report')
    }
  }

  const handleReset = () => {
    useStore.getState().reset()
    if (videoUrl) {
      URL.revokeObjectURL(videoUrl)
    }
  }

  if (loading) {
    return (
      <GlassCard>
        <p>Processing video... This may take a few minutes.</p>
      </GlassCard>
    )
  }

  return (
    <Container>
      <GlassCard>
        <VideoContainer>
          <Video ref={videoRef} src={videoUrl} controls />
          <Canvas ref={canvasRef} />
          <AnnotationLayer ref={annotationRef}>
            {attentionZones.map((zone, idx) => {
              const videoWidth = videoRef.current?.videoWidth || 1
              const videoHeight = videoRef.current?.videoHeight || 1
              if (videoWidth === 0 || videoHeight === 0) return null
              
              return (
                <AnnotationText
                  key={idx}
                  style={{
                    left: `${(zone.screenX / videoWidth) * 100}%`,
                    top: `${(zone.screenY / videoHeight) * 100}%`,
                    transform: 'translate(-50%, -50%)'
                  }}
                >
                  {zone.percentage?.toFixed(1) || '0.0'}%
                </AnnotationText>
              )
            })}
          </AnnotationLayer>
        </VideoContainer>

        <Controls>
          <Button onClick={handleDownloadPDF} disabled={!results}>
            <Download size={20} />
            Download PDF Report
          </Button>
          <Button onClick={handleReset}>
            <RefreshCw size={20} />
            Upload New Video
          </Button>
        </Controls>

        {results && (
          <StatsCard>
            <StatRow>
              <StatLabel>Clarity Score</StatLabel>
              <StatValue>{Math.round(results.clarity_score)}/100</StatValue>
            </StatRow>
            {currentFrame && (
              <>
                <StatRow>
                  <StatLabel>Current Entropy</StatLabel>
                  <StatValue>{currentFrame.entropy.toFixed(2)}</StatValue>
                </StatRow>
                <StatRow>
                  <StatLabel>Current Conflict</StatLabel>
                  <StatValue>{currentFrame.conflict.toFixed(2)}</StatValue>
                </StatRow>
              </>
            )}
            {results.ai_narrative && (
              <div style={{ marginTop: '1rem', paddingTop: '1rem', borderTop: '1px solid rgba(255, 255, 255, 0.1)' }}>
                <StatLabel style={{ display: 'block', marginBottom: '0.5rem' }}>AI Analysis:</StatLabel>
                <p style={{ color: '#FFFFFF', lineHeight: '1.6' }}>{results.ai_narrative}</p>
              </div>
            )}
          </StatsCard>
        )}
      </GlassCard>
    </Container>
  )
}

export default VideoPlayer
