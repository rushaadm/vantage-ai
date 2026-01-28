import { useState } from 'react'
import styled, { ThemeProvider } from 'styled-components'
import { theme } from './theme'
import FileUpload from './components/FileUpload'
import VideoPlayer from './components/VideoPlayer'
import { useStore } from './store'

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

function App() {
  const { jobId, videoUrl } = useStore()

  return (
    <ThemeProvider theme={theme}>
      <Container>
        <Header>
          <Title>VANTAGE AI</Title>
          <Subtitle>Visual Attention Analysis Engine</Subtitle>
        </Header>
        
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
