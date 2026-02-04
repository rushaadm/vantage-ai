# Cloud Deployment Guide

## Offloading Processing to Cloud

To offload video processing and reduce local CPU load, you have several options:

### Option 1: Deploy Backend to Cloud (Recommended)

**Services:**
- **Render.com** (Free tier available)
- **Railway.app** (Free tier available)
- **Fly.io** (Free tier available)
- **Google Cloud Run** (Pay per use)
- **AWS Lambda** (Pay per use)

**Steps:**
1. Create account on chosen platform
2. Connect your GitHub repository
3. Set environment variables (GEMINI_API_KEY)
4. Deploy backend
5. Update frontend API URL to cloud endpoint

### Option 2: Use Cloud Video Processing Services

**Services:**
- **AWS MediaConvert**
- **Google Cloud Video Intelligence**
- **Azure Media Services**

### Option 3: Optimize Current Setup

The current backend is optimized for cloud deployment:
- Processes at 480px width (granular but efficient)
- Samples at 10 FPS (smooth heatmaps)
- Max 60 second processing time
- Can handle up to 300 samples per video

### Environment Variables

Set these in your cloud platform:
```
GEMINI_API_KEY=your_key_here
```

### Current Processing Specs

- **Resolution**: 480px width (granular pixel-level analysis)
- **Frame Rate**: 10 samples/second (smooth dynamic heatmaps)
- **Heatmap Resolution**: 1/2 of processing resolution (granular detail)
- **Max Processing Time**: 60 seconds
- **Max Samples**: 300 frames per video

### Performance Notes

- Processing is CPU-intensive but optimized
- Memory usage is minimized
- Can be scaled horizontally in cloud
- Consider using GPU instances for faster processing
