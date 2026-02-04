import { useState } from 'react'
import styled, { ThemeProvider } from 'styled-components'
import { theme } from './theme'
import FileUpload from './components/FileUpload'
import VideoPlayer from './components/VideoPlayer'
import { useStore } from './store'
import { Info } from 'lucide-react'

const Container = styled.div`
  min-height: 100vh;
  background: ${props => props.theme.colors.background};
  padding: 2rem;
`

const Header = styled.div`
  text-align: center;
  margin-bottom: 3rem;
`

const Title = styled.h1`
  font-size: 3rem;
  background: linear-gradient(135deg, ${props => props.theme.colors.cyan}, ${props => props.theme.colors.pink});
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 0.5rem;
`

const Subtitle = styled.p`
  color: ${props => props.theme.colors.textSecondary};
  font-size: 1.2rem;
`

const MethodButton = styled.button`
  ${props => props.theme.glass}
  padding: 0.5rem 1rem;
  border: none;
  color: ${props => props.theme.colors.text};
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin: 1rem auto;
  transition: all 0.3s ease;
  
  &:hover {
    background-color: rgba(0, 242, 255, 0.2);
    border-color: ${props => props.theme.colors.cyan};
  }
`

const MethodModal = styled.div`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.8);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  padding: 2rem;
`

const MethodContent = styled.div`
  ${props => props.theme.glass}
  max-width: 800px;
  max-height: 90vh;
  overflow-y: auto;
  padding: 2rem;
  border-radius: 16px;
`

const MethodTitle = styled.h2`
  color: ${props => props.theme.colors.cyan};
  margin-bottom: 1rem;
`

const MethodText = styled.p`
  color: ${props => props.theme.colors.text};
  line-height: 1.6;
  margin-bottom: 1rem;
`

const Citation = styled.div`
  background: rgba(0, 242, 255, 0.1);
  padding: 1rem;
  border-radius: 8px;
  margin: 1rem 0;
  font-size: 0.9rem;
  color: ${props => props.theme.colors.textSecondary};
`

function App() {
  const { jobId, videoUrl } = useStore()
  const [showMethodology, setShowMethodology] = useState(false)

  return (
    <ThemeProvider theme={theme}>
      <Container>
        <Header>
          <Title>VANTAGE AI</Title>
          <Subtitle>Visual Attention Analysis Engine</Subtitle>
          <MethodButton onClick={() => setShowMethodology(true)}>
            <Info size={18} />
            Methodology & Citations
          </MethodButton>
        </Header>
        
        {showMethodology && (
          <MethodModal onClick={() => setShowMethodology(false)}>
            <MethodContent onClick={(e) => e.stopPropagation()}>
              <MethodTitle>Methodology & Research Citations</MethodTitle>
              
              <MethodText>
                <strong>Vantage AI</strong> employs a dual-stream visual attention model based on established eye-tracking research to analyze visual content.
              </MethodText>
              
              <MethodTitle style={{ fontSize: '1.2rem', marginTop: '1.5rem' }}>1. Static Saliency Detection</MethodTitle>
              <MethodText>
                Uses color contrast and edge detection algorithms to predict attention based on visual features. This approach is grounded in computational models of visual attention.
              </MethodText>
              
              <MethodTitle style={{ fontSize: '1.2rem', marginTop: '1.5rem' }}>2. Dynamic Motion Analysis</MethodTitle>
              <MethodText>
                Analyzes frame-to-frame differences to detect motion patterns that capture viewer attention.
              </MethodText>
              
              <MethodTitle style={{ fontSize: '1.2rem', marginTop: '1.5rem' }}>3. Score Calculations</MethodTitle>
              
              <MethodText><strong>Clarity Score:</strong> Measures visual hierarchy clarity based on motion-saliency conflict. Calculated as 100 - (avg_conflict × 20), normalized to 0-100 scale. Lower conflict indicates clearer visual structure.</MethodText>
              
              <MethodText><strong>Attention Stability:</strong> Measures consistency of attention patterns across frames. Calculated as 100 - (std_entropy × 30). Lower entropy variance indicates more stable, predictable attention patterns.</MethodText>
              
              <MethodText><strong>Engagement Score:</strong> Combines saliency intensity and fixation rate. Calculated as (avg_saliency × 60) + (fixation_rate × 8). Higher values indicate stronger visual interest and engagement.</MethodText>
              
              <MethodTitle style={{ fontSize: '1.2rem', marginTop: '1.5rem' }}>Research Citations</MethodTitle>
              
              <Citation>
                <strong>Itti, L., & Koch, C. (2001).</strong> Computational modelling of visual attention. <em>Nature Reviews Neuroscience</em>, 2(3), 194-203.
                <br /><br />
                This foundational work established computational models for predicting visual attention based on bottom-up saliency features including color, intensity, and orientation contrasts.
              </Citation>
              
              <Citation>
                <strong>Yarbus, A. L. (1967).</strong> <em>Eye movements and vision.</em> Plenum Press.
                <br /><br />
                Classic research demonstrating that eye movements and fixations reflect cognitive processes and visual interest, forming the basis for fixation-based engagement metrics.
              </Citation>
              
              <Citation>
                <strong>Tatler, B. W., Hayhoe, M. M., Land, M. F., & Ballard, D. H. (2011).</strong> Eye guidance in natural vision: Reinterpreting saliency. <em>Journal of Vision</em>, 11(5), 5.
                <br /><br />
                Research on attention stability and consistency in natural viewing, informing our entropy-based stability measurements.
              </Citation>
              
              <MethodButton onClick={() => setShowMethodology(false)} style={{ marginTop: '1.5rem' }}>
                Close
              </MethodButton>
            </MethodContent>
          </MethodModal>
        )}
        
        {!videoUrl ? (
          <FileUpload />
        ) : (
          <VideoPlayer />
        )}
      </Container>
    </ThemeProvider>
  )
}

export default App
