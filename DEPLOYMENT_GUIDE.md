# 🚀 Vantage AI - Cloud Deployment Guide (Render.com)

## Quick Start: Deploy Backend to Render.com

### Step 1: Prepare Your Code

1. **Push to GitHub** (if not already):
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/vantage-ai.git
   git push -u origin main
   ```

### Step 2: Deploy Backend Web Service

1. **Go to Render Dashboard**: https://dashboard.render.com
2. **Click "New +" → "Web Service"**
3. **Connect Repository**:
   - Select your GitHub account
   - Choose `vantage-ai` repository
   - Select branch: `main`
4. **Configure Service**:
   - **Name**: `vantage-ai-api`
   - **Root Directory**: `backend`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. **Environment Variables**:
   - Click "Advanced" → "Add Environment Variable"
   - Key: `GEMINI_API_KEY`
   - Value: `your_gemini_api_key_here`
6. **Add Disk** (for video storage):
   - Click "Advanced" → "Persistent Disk"
   - Name: `vantage-ai-disk`
   - Mount Path: `/opt/render/project/src`
   - Size: `10 GB`
7. **Click "Create Web Service"**

### Step 3: Update Frontend API URL

Once deployed, Render gives you a URL like: `https://vantage-ai-api.onrender.com`

Update your frontend:

**File: `frontend/src/components/FileUpload.jsx`**
```javascript
// Change this line:
const response = await axios.post('http://localhost:8000/upload', formData, {

// To:
const API_URL = process.env.VITE_API_URL || 'https://vantage-ai-api.onrender.com'
const response = await axios.post(`${API_URL}/upload`, formData, {
```

**File: `frontend/src/components/VideoPlayer.jsx`**
```javascript
// Change all instances of:
axios.get(`http://localhost:8000/results/${jobId}`)

// To:
const API_URL = process.env.VITE_API_URL || 'https://vantage-ai-api.onrender.com'
axios.get(`${API_URL}/results/${jobId}`)
```

### Step 4: Deploy Frontend (Optional - Static Site)

1. **Build Frontend**:
   ```bash
   cd frontend
   npm run build
   ```

2. **Deploy to Render Static Site**:
   - Go to Render Dashboard
   - Click "New +" → "Static Site"
   - Connect your GitHub repo
   - **Root Directory**: `frontend`
   - **Build Command**: `npm install && npm run build`
   - **Publish Directory**: `dist`
   - **Add Environment Variable**:
     - Key: `VITE_API_URL`
     - Value: `https://vantage-ai-api.onrender.com`
   - Click "Create Static Site"

## 🎯 Advanced: Background Worker for Heavy Processing

For even better performance, use a Background Worker:

### Step 1: Create Background Worker Service

1. **Create `backend/worker.py`**:
```python
import os
import time
import redis
from main import process_video

# Connect to Redis (Render Key Value)
redis_client = redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379'))

def worker():
    while True:
        # Get job from queue
        job_data = redis_client.blpop(['video_queue'], timeout=10)
        if job_data:
            job_id, video_path = job_data[1].decode().split(':')
            print(f"Processing job {job_id}")
            process_video(job_id, video_path)
            redis_client.set(f"job:{job_id}:status", "complete")

if __name__ == "__main__":
    worker()
```

2. **Update `backend/main.py`** to use Redis queue:
```python
import redis
import os

redis_client = redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379'))

@app.post("/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    # ... save file ...
    
    # Add to Redis queue instead of background task
    redis_client.lpush('video_queue', f"{job_id}:{video_path}")
    
    return {"job_id": job_id, "status": "queued"}
```

3. **Deploy Background Worker**:
   - Render Dashboard → "New +" → "Background Worker"
   - Connect repo, set root to `backend`
   - **Start Command**: `python worker.py`
   - **Add Environment Variables**:
     - `REDIS_URL` (from Key Value service)
     - `GEMINI_API_KEY`

### Step 2: Create Key Value Store (Redis)

1. **Render Dashboard → "New +" → "Key Value"**
2. **Name**: `vantage-ai-redis`
3. **Plan**: Free tier is fine
4. **Copy the `REDIS_URL`** and add it to your services

## 📊 Current Setup (Local → Cloud Migration)

### What Changes:
- ✅ Backend runs on Render (cloud CPU)
- ✅ Videos stored on Render disk
- ✅ Processing happens in cloud
- ✅ Frontend can be static site or local

### What Stays the Same:
- ✅ Same API endpoints
- ✅ Same processing logic
- ✅ Same heatmap rendering

## 🔧 Environment Variables Needed

**Backend (Web Service)**:
- `GEMINI_API_KEY` - Your Gemini API key
- `REDIS_URL` - (Optional) If using Background Worker

**Frontend (Static Site)**:
- `VITE_API_URL` - Your backend URL (e.g., `https://vantage-ai-api.onrender.com`)

## 💰 Cost Estimate (Render Free Tier)

- **Web Service**: Free (spins down after 15 min inactivity)
- **Background Worker**: Free (spins down after inactivity)
- **Key Value (Redis)**: Free (25 MB)
- **Persistent Disk**: Free (1 GB, $0.25/GB/month after)

**Total**: ~$0-5/month for light usage

## 🚨 Important Notes

1. **Free tier spins down** after inactivity - first request takes ~30s to wake up
2. **Disk storage** is persistent - videos persist between deployments
3. **Environment variables** are encrypted
4. **Logs** are available in Render dashboard

## 🎉 After Deployment

Your backend will be at: `https://vantage-ai-api.onrender.com`

Test it:
```bash
curl https://vantage-ai-api.onrender.com/docs
```

Update frontend to use cloud backend, and all processing happens in the cloud! 🚀
