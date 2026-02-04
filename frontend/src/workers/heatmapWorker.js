// Web Worker for heatmap rendering - offloads CPU-intensive work from main thread

self.onmessage = function(e) {
  const { saliencyMap, heatmapH, heatmapW, canvasWidth, canvasHeight, minVal, maxVal, currentTime, frameTime } = e.data
  
  // Calculate scaling
  const scaleX = canvasWidth / heatmapW
  const scaleY = canvasHeight / heatmapH
  const range = maxVal - minVal || 1
  
  // Create smooth, granular heatmap with green gradient
  const smoothMap = new Array(canvasHeight)
  for (let y = 0; y < canvasHeight; y++) {
    smoothMap[y] = new Array(canvasWidth)
    for (let x = 0; x < canvasWidth; x++) {
      // Bilinear interpolation for smoothness
      const mapY = y / scaleY
      const mapX = x / scaleX
      const y1 = Math.floor(mapY)
      const y2 = Math.min(y1 + 1, heatmapH - 1)
      const x1 = Math.floor(mapX)
      const x2 = Math.min(x1 + 1, heatmapW - 1)
      
      const fx = mapX - x1
      const fy = mapY - y1
      
      // Bilinear interpolation
      const v11 = saliencyMap[y1]?.[x1] || 0
      const v21 = saliencyMap[y1]?.[x2] || 0
      const v12 = saliencyMap[y2]?.[x1] || 0
      const v22 = saliencyMap[y2]?.[x2] || 0
      
      const v1 = v11 * (1 - fx) + v21 * fx
      const v2 = v12 * (1 - fx) + v22 * fx
      let saliencyValue = v1 * (1 - fy) + v2 * fy
      
      // Normalize
      saliencyValue = (saliencyValue - minVal) / range
      
      // Higher threshold - only show what MOST people focus on
      const threshold = 0.3
      if (saliencyValue < threshold) {
        saliencyValue = 0
      } else {
        saliencyValue = (saliencyValue - threshold) / (1 - threshold)
      }
      
      saliencyValue = Math.pow(saliencyValue, 0.6)
      smoothMap[y][x] = saliencyValue
    }
  }
  
  // Apply Gaussian blur for extra smoothness
  const blurredMap = []
  const blurRadius = 3
  for (let y = 0; y < canvasHeight; y++) {
    blurredMap[y] = []
    for (let x = 0; x < canvasWidth; x++) {
      let sum = 0
      let count = 0
      for (let dy = -blurRadius; dy <= blurRadius; dy++) {
        for (let dx = -blurRadius; dx <= blurRadius; dx++) {
          const ny = Math.max(0, Math.min(canvasHeight - 1, y + dy))
          const nx = Math.max(0, Math.min(canvasWidth - 1, x + dx))
          const weight = 1 / (1 + dx * dx + dy * dy)
          sum += smoothMap[ny][nx] * weight
          count += weight
        }
      }
      blurredMap[y][x] = sum / count
    }
  }
  
  // Create image data with green gradient
  const imgData = new ImageData(canvasWidth, canvasHeight)
  for (let y = 0; y < canvasHeight; y++) {
    for (let x = 0; x < canvasWidth; x++) {
      const idx = (y * canvasWidth + x) * 4
      let saliencyValue = blurredMap[y][x]
      
      const intensity = Math.max(0, Math.min(255, saliencyValue * 220))
      
      if (saliencyValue > 0.6) {
        const t = (saliencyValue - 0.6) / 0.4
        imgData.data[idx] = 0
        imgData.data[idx + 1] = Math.floor(150 + t * 105)
        imgData.data[idx + 2] = Math.floor(50 + t * 50)
        imgData.data[idx + 3] = intensity
      } else if (saliencyValue > 0.3) {
        const t = (saliencyValue - 0.3) / 0.3
        imgData.data[idx] = 0
        imgData.data[idx + 1] = Math.floor(100 + t * 50)
        imgData.data[idx + 2] = Math.floor(30 + t * 20)
        imgData.data[idx + 3] = intensity
      } else if (saliencyValue > 0) {
        const t = saliencyValue / 0.3
        imgData.data[idx] = 0
        imgData.data[idx + 1] = Math.floor(50 + t * 50)
        imgData.data[idx + 2] = Math.floor(20 + t * 10)
        imgData.data[idx + 3] = intensity * 0.7
      } else {
        imgData.data[idx] = 0
        imgData.data[idx + 1] = 0
        imgData.data[idx + 2] = 0
        imgData.data[idx + 3] = 0
      }
    }
  }
  
  // Detect fixation points
  const fixationPoints = []
  const threshold = minVal + range * 0.5
  
  for (let y = 2; y < heatmapH - 2; y++) {
    for (let x = 2; x < heatmapW - 2; x++) {
      const val = saliencyMap[y]?.[x] || 0
      if (val < threshold) continue
      
      let isMax = true
      for (let dy = -2; dy <= 2 && isMax; dy++) {
        for (let dx = -2; dx <= 2 && isMax; dx++) {
          if (dx === 0 && dy === 0) continue
          const neighborVal = saliencyMap[y + dy]?.[x + dx] || 0
          if (neighborVal > val) isMax = false
        }
      }
      
      if (isMax) {
        fixationPoints.push({
          x: x * scaleX,
          y: y * scaleY,
          intensity: val,
          time: frameTime
        })
      }
    }
  }
  
  // Send results back to main thread
  self.postMessage({
    imgData: imgData,
    fixationPoints: fixationPoints
  })
}
