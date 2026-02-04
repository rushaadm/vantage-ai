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

function FileUpload() {
  const [uploading, setUploading] = useState(false)
  const [status, setStatus] = useState('')
  const { setJobId, setVideoUrl } = useStore()

  const onDrop = useCallback(async (acceptedFiles) => {
    const file = acceptedFiles[0]
    if (!file) return

    if (!file.type.startsWith('video/')) {
      setStatus('Please upload a video file')
      return
    }

    setUploading(true)
    setStatus('Uploading video...')

    try {
      const formData = new FormData()
      formData.append('file', file)

      const apiUrl = API_URL || import.meta.env.VITE_API_URL || 'https://vantage-ai-25ct.onrender.com'
      const response = await axios.post(`${apiUrl}/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })

      // Check for error response
      if (response.data.error) {
        throw new Error(response.data.error)
      }

      const jobId = response.data.job_id
      if (!jobId) {
        throw new Error('No job ID returned from server')
      }

      setJobId(jobId)
      setStatus('Video uploaded! Processing...')
      
      // Create object URL for video preview
      const videoUrl = URL.createObjectURL(file)
      setVideoUrl(videoUrl)

      // Poll for results
      pollResults(jobId)
    } catch (error) {
      console.error('Upload error:', error)
      let errorMessage = 'Upload failed. Please try again.'
      
      if (error.code === 'ECONNREFUSED' || error.message?.includes('Network Error') || error.message?.includes('ERR_CONNECTION_REFUSED')) {
        errorMessage = 'Cannot connect to server. Please make sure the backend is running on http://localhost:8000'
      } else if (error.response?.data?.error) {
        errorMessage = error.response.data.error
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
    </GlassCard>
  )
}

export default FileUpload
