# Vantage AI - Visual Attention Analysis Engine

A dual-stream visual attention analysis tool that combines static saliency (DeepGaze IIE) with dynamic motion detection (Farneback Optical Flow) to analyze video content.

## Features

- 🎥 **Video Upload & Processing**: Upload videos and get real-time attention analysis
- 🧠 **Dual-Stream AI**: Combines static saliency maps with motion detection
- 📊 **Real-time Heatmaps**: Visual overlay showing attention patterns synced to video playback
- 📄 **PDF Reports**: Generate comprehensive audit reports with AI-powered suggestions
- 🎨 **Modern UI**: Glass morphism design with smooth animations

## Project Structure

```
vantage-ai/
├── backend/
│   ├── engine/
│   │   ├── saliency.py      # VantageEngine with dual-stream processing
│   │   └── reporter.py      # PDF report generation with Gemini AI
│   ├── main.py              # FastAPI application
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── FileUpload.jsx
│   │   │   └── VideoPlayer.jsx
│   │   ├── theme.js         # Glass morphism theme
│   │   ├── store.js         # Zustand state management
│   │   └── App.jsx
│   └── package.json
└── README.md
```

## Setup

### Backend

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

5. Run the server:
```bash
uvicorn main:app --reload --port 8000
```

### Frontend

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Run the development server:
```bash
npm run dev
```

4. Open your browser to `http://localhost:5173`

## Usage

1. **Upload a Video**: Drag and drop or click to upload a video file (MP4, MOV, AVI, MKV)
2. **Wait for Processing**: The backend will process the video frame-by-frame (this may take a few minutes)
3. **View Results**: Once processing is complete, you'll see:
   - Real-time heatmap overlay on the video
   - Clarity score and metrics
   - AI-generated suggestions for improvement
4. **Download Report**: Click "Download PDF Report" to get a comprehensive audit document

## API Endpoints

- `POST /upload` - Upload a video file
- `GET /results/{job_id}` - Get processing results
- `GET /download-pdf/{job_id}` - Download PDF report

## Technologies

### Backend
- FastAPI - Modern Python web framework
- PySaliency - DeepGaze IIE saliency model
- OpenCV - Video processing and optical flow
- Google Gemini AI - AI-powered suggestions
- FPDF2 - PDF generation

### Frontend
- React 18 - UI framework
- Vite - Build tool
- Styled Components - CSS-in-JS styling
- Zustand - State management
- Axios - HTTP client
- React Dropzone - File uploads

## License

MIT
