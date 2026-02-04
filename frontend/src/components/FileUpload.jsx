import { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import styled from 'styled-components'
import axios from 'axios'
import { Upload, Video } from 'lucide-react'
import { useStore } from '../store'

// API URL with fallback
const API_URL = import.meta.env.VITE_API_URL || 'https://vantage-ai-25ct.onrender.com'

const GlassCard = styled.div`
  ${props => props.theme.glass}
  padding: 3rem;
  max-width: 600px;
  margin: 0 auto;
  text-align: center;
`

const Dropzone = styled.div`
  border: 2px dashed ${props => props.isDragActive ? props.theme.colors.cyan : props.theme.colors.border};
  border-radius: 12px;
  padding: 4rem 2rem;
  cursor: pointer;
  transition: all 0.3s ease;
  background: ${props => props.isDragActive ? 'rgba(0, 242, 255, 0.1)' : 'transparent'};
  
  &:hover {
    border-color: ${props => props.theme.colors.cyan};
    background: rgba(0, 242, 255, 0.05);
  }
`

const UploadIcon = styled.div`
  display: flex;
  justify-content: center;
  margin-bottom: 1rem;
  color: ${props => props.theme.colors.cyan};
`

const Text = styled.p`
  color: ${props => props.theme.colors.text};
  font-size: 1.1rem;
  margin-bottom: 0.5rem;
`

const Subtext = styled.p`
  color: ${props => props.theme.colors.textSecondary};
  font-size: 0.9rem;
`

const Status = styled.div`
  margin-top: 1rem;
  padding: 1rem;
  border-radius: 8px;
  background: rgba(0, 242, 255, 0.1);
  color: ${props => props.theme.colors.cyan};
`

const SliderContainer = styled.div`
  margin-top: 1rem;
  padding: 1rem;
  background: rgba(0, 242, 255, 0.1);
  border-radius: 8px;
`

const Slider = styled.input`
  width: 100%;
  margin: 0.5rem 0;
  height: 6px;
  border-radius: 3px;
  background: rgba(255, 255, 255, 0.1);
  outline: none;
  -webkit-appearance: none;
  
  &::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: ${props => props.theme.colors.cyan};
    cursor: pointer;
  }
  
  &::-moz-range-thumb {
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: ${props => props.theme.colors.cyan};
    cursor: pointer;
    border: none;
  }
`

const SliderLabel = styled.label`
  color: ${props => props.theme.colors.text};
  display: block;
  margin-bottom: 0.5rem;
  font-size: 0.9rem;
`

function FileUpload() {
  const [uploading, setUploading] = useState(false)
  const [status, setStatus] = useState('')
  const { setJobId, setVideoUrl, frameSamplingRate, setFrameSamplingRate } = useStore()

  const onDrop = useCallback(async (acceptedFiles) => {
    const file = acceptedFiles[0]
    if (!file) return

    if (!file.type.startsWith('video/')) {
      setStatus('Please upload a video file')
      return
    }

    setUploading(true)
    setStatus('Preparing upload...')
    
    console.log('=== UPLOAD START ===')
    console.log('File:', file.name)
    console.log('File size:', (file.size / 1024 / 1024).toFixed(2), 'MB')
    console.log('File type:', file.type)

    try {
      const formData = new FormData()
      formData.append('file', file)
      const sampleRate = useStore.getState().frameSamplingRate || 2
      formData.append('sample_rate', sampleRate.toString())
      
      console.log('FormData created, sample_rate:', sampleRate)

      // Force Render.com URL for production
      const apiUrl = 'https://vantage-ai-25ct.onrender.com'
      console.log('API URL:', apiUrl)
      console.log('Full upload URL:', `${apiUrl}/upload`)
      console.log('⚠️ Uploading to Render.com cloud server')
      
      setStatus('Connecting to server...')
      
      // Add upload progress tracking
      const uploadStartTime = Date.now()
      
      console.log('Sending POST request...')
      setStatus('Uploading video...')
      
      const response = await axios.post(`${apiUrl}/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 300000, // 300 second (5 minute) timeout for large files
        onUploadProgress: (progressEvent) => {
          if (progressEvent.total) {
            const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total)
            const elapsed = (Date.now() - uploadStartTime) / 1000
            const statusMsg = `Uploading video... ${percentCompleted}% (${(progressEvent.loaded / 1024 / 1024).toFixed(2)} MB / ${(progressEvent.total / 1024 / 1024).toFixed(2)} MB)`
            setStatus(statusMsg)
            console.log(`Upload progress: ${percentCompleted}% in ${elapsed.toFixed(1)}s`)
          } else {
            console.log('Upload progress: loaded', progressEvent.loaded, 'bytes')
            setStatus(`Uploading video... ${(progressEvent.loaded / 1024 / 1024).toFixed(2)} MB`)
          }
        },
      })

      console.log('=== UPLOAD SUCCESS ===')
      console.log('Response status:', response.status)
      console.log('Response data:', response.data)

      // Check for error response
      if (response.data.error) {
        throw new Error(response.data.error)
      }

      const jobId = response.data.job_id
      if (!jobId) {
        console.error('Response data:', response.data)
        throw new Error('No job ID returned from server. Response: ' + JSON.stringify(response.data))
      }

      // Create object URL for video preview FIRST (before setting jobId)
      const videoUrl = URL.createObjectURL(file)
      setVideoUrl(videoUrl)
      
      // Then set jobId (this triggers VideoPlayer to show)
      setJobId(jobId)
      setStatus('Video uploaded! Processing...')

      // Poll for results
      pollResults(jobId)
    } catch (error) {
      console.error('=== UPLOAD ERROR ===')
      console.error('Error object:', error)
      console.error('Error message:', error.message)
      console.error('Error code:', error.code)
      console.error('Error response:', error.response)
      console.error('Error response data:', error.response?.data)
      console.error('Error response status:', error.response?.status)
      console.error('Error stack:', error.stack)
      
      let errorMessage = 'Upload failed. Please try again.'
      
      if (error.code === 'ECONNABORTED' || error.message?.includes('timeout')) {
        errorMessage = 'Upload timed out. The file may be too large or the server is slow. Please try again.'
      } else if (error.code === 'ECONNREFUSED' || error.message?.includes('Network Error') || error.message?.includes('ERR_CONNECTION_REFUSED')) {
        errorMessage = 'Cannot connect to server. Please check your internet connection or try again later.'
      } else if (error.message?.includes('Network Error') || error.message?.includes('ERR_FAILED')) {
        errorMessage = 'Network error. Please check your internet connection and try again.'
      } else if (error.response?.data?.error) {
        errorMessage = error.response.data.error
      } else if (error.response?.status === 413) {
        errorMessage = 'File too large. Please upload a smaller video file.'
      } else if (error.message) {
        errorMessage = error.message
      }
      
      setStatus(`Upload failed: ${errorMessage}`)
      setUploading(false)
    }
  }, [])

  const pollResults = async (jobId) => {
    const maxAttempts = 60
    let attempts = 0

    const interval = setInterval(async () => {
      attempts++
      try {
        const response = await axios.get(`${API_URL}/results/${jobId}`)
        
        if (response.data.status !== 'processing') {
          clearInterval(interval)
          setUploading(false)
          setStatus('Processing complete!')
          useStore.getState().setResults(response.data)
        } else if (attempts >= maxAttempts) {
          clearInterval(interval)
          setUploading(false)
          setStatus('Processing is taking longer than expected...')
        }
      } catch (error) {
        if (error.response?.status === 404) {
          // Still processing
          return
        }
        clearInterval(interval)
        setUploading(false)
        setStatus('Error checking results')
      }
    }, 2000) // Poll every 2 seconds
  }

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/*': ['.mp4', '.mov', '.avi', '.mkv']
    },
    disabled: uploading
  })

  return (
    <GlassCard>
      <Dropzone {...getRootProps()} isDragActive={isDragActive}>
        <input {...getInputProps()} />
        <UploadIcon>
          {isDragActive ? <Video size={64} /> : <Upload size={64} />}
        </UploadIcon>
        <Text>
          {isDragActive ? 'Drop your video here' : 'Drag & drop your video here'}
        </Text>
        <Subtext>or click to browse</Subtext>
        <Subtext style={{ marginTop: '0.5rem', fontSize: '0.8rem' }}>
          Supports MP4, MOV, AVI, MKV
        </Subtext>
      </Dropzone>
      
      {status && (
        <Status>{status}</Status>
      )}
      
      {/* Frame sampling slider - show before upload */}
      {!uploading && (
        <SliderContainer>
          <SliderLabel>
            Frame Sampling Rate: Process every {frameSamplingRate === 1 ? 'frame (100%)' : frameSamplingRate === 2 ? '2nd frame (50%)' : frameSamplingRate === 3 ? '3rd frame (33%)' : `${frameSamplingRate}th frame (${Math.round(100/frameSamplingRate)}%)`}
          </SliderLabel>
          <Slider
            type="range"
            min="1"
            max="10"
            value={frameSamplingRate}
            onChange={(e) => setFrameSamplingRate(parseInt(e.target.value))}
          />
          <Subtext style={{ fontSize: '0.8rem', marginTop: '0.5rem' }}>
            Lower = more frames analyzed (slower but more detailed). Higher = fewer frames (faster but less granular).
          </Subtext>
        </SliderContainer>
      )}
    </GlassCard>
  )
}

export default FileUpload
