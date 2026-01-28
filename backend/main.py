from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uuid
import os
import json
from pathlib import Path
import cv2
import numpy as np
import time
from engine.saliency import VantageEngine
from engine.reporter import AuditReport, get_ai_suggestions

app = FastAPI(title="Vantage AI API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure directories exist
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

engine = VantageEngine()
# Removed parallel processing to reduce CPU load

# Removed batch processing to reduce memory usage

def process_video(job_id: str, video_path: str):
    """Lightweight video processing - aggressive sampling to reduce CPU/memory load"""
    start_time = time.time()
    max_processing_time = 20  # 20 second limit
    
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        # Aggressive optimizations for lightweight processing
        PROCESS_WIDTH = 320  # Much smaller resolution
        SAMPLE_RATE = max(1, int(fps / 2))  # Sample every 0.5 seconds (2 FPS max)
        if SAMPLE_RATE > 15:
            SAMPLE_RATE = 15  # Cap at every 15th frame
        
        frames_data = []
        all_entropies = []
        all_conflicts = []
        
        frame_idx = 0
        prev_frame = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames based on sample rate
            if frame_idx % SAMPLE_RATE != 0:
                frame_idx += 1
                continue
            
            # Check time limit
            elapsed = time.time() - start_time
            if elapsed > max_processing_time:
                print(f"Time limit reached at frame {frame_idx}")
                break
            
            # Aggressive downscaling for speed
            h, w = frame.shape[:2]
            if w > PROCESS_WIDTH:
                scale = PROCESS_WIDTH / w
                new_h = int(h * scale)
                frame_small = cv2.resize(frame, (PROCESS_WIDTH, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                frame_small = frame
            
            # Get saliency and motion maps
            saliency_map = engine.get_saliency_map(frame_small)
            
            if prev_frame is not None:
                if prev_frame.shape[1] != frame_small.shape[1]:
                    prev_frame_small = cv2.resize(prev_frame, (frame_small.shape[1], frame_small.shape[0]))
                else:
                    prev_frame_small = prev_frame
                motion_map = engine.get_motion_map(prev_frame_small, frame_small)
            else:
                motion_map = np.zeros((frame_small.shape[0], frame_small.shape[1]), dtype=np.float32)
            
            # Calculate metrics
            metrics = engine.calculate_metrics(saliency_map, motion_map)
            all_entropies.append(metrics["entropy"])
            all_conflicts.append(metrics["conflict"])
            
            # Aggressive downsampling for storage (1/8 resolution to save memory)
            heatmap_h, heatmap_w = saliency_map.shape[:2]
            saliency_small = cv2.resize(saliency_map, (heatmap_w // 8, heatmap_h // 8), interpolation=cv2.INTER_AREA)
            motion_small = cv2.resize(motion_map, (heatmap_w // 8, heatmap_h // 8), interpolation=cv2.INTER_AREA)
            
            # Store frame data (sampled frames only)
            frames_data.append({
                "frame": frame_idx,
                "time": frame_idx / fps if fps > 0 else frame_idx / 30,
                "entropy": metrics["entropy"],
                "conflict": metrics["conflict"],
                "saliency_heatmap": saliency_small.tolist(),
                "motion_heatmap": motion_small.tolist(),
                "original_size": [h, w],
                "heatmap_size": [heatmap_h // 8, heatmap_w // 8]
            })
            
            prev_frame = frame.copy()
            frame_idx += 1
        
        cap.release()
        
        # Calculate statistics
        if all_entropies:
            avg_entropy = np.mean(all_entropies)
            max_entropy = np.max(all_entropies)
            min_entropy = np.min(all_entropies)
            std_entropy = np.std(all_entropies)
            
            avg_conflict = np.mean(all_conflicts)
            max_conflict = np.max(all_conflicts)
            min_conflict = np.min(all_conflicts)
            std_conflict = np.std(all_conflicts)
            
            clarity_score = max(0, min(100, 100 - (avg_conflict * 10)))
        else:
            avg_entropy = max_entropy = min_entropy = std_entropy = 0
            avg_conflict = max_conflict = min_conflict = std_conflict = 0
            clarity_score = 50
        
        # Get AI suggestions
        stats = {
            "avg_entropy": float(avg_entropy),
            "max_entropy": float(max_entropy),
            "min_entropy": float(min_entropy),
            "std_entropy": float(std_entropy),
            "avg_conflict": float(avg_conflict),
            "max_conflict": float(max_conflict),
            "min_conflict": float(min_conflict),
            "std_conflict": float(std_conflict),
            "clarity_score": clarity_score,
            "total_frames": frame_count,
            "processed_frames": len(frames_data),
            "duration": duration,
            "fps": fps
        }
        
        try:
            ai_narrative = get_ai_suggestions(stats)
        except Exception as e:
            print(f"AI suggestions failed: {e}")
            ai_narrative = "AI analysis unavailable. Focus on reducing motion conflict and increasing visual clarity."
        
        # Save results
        result_data = {
            "job_id": job_id,
            "fps": fps,
            "frame_count": frame_count,
            "processed_frames": len(frames_data),
            "clarity_score": clarity_score,
            "stats": stats,
            "ai_narrative": ai_narrative,
            "frames": frames_data,
            "processing_time": time.time() - start_time
        }
        
        result_path = RESULTS_DIR / f"{job_id}.json"
        with open(result_path, "w") as f:
            json.dump(result_data, f)
        
        print(f"Processing complete for job {job_id} in {time.time() - start_time:.2f}s ({len(frames_data)} frames)")
        
    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()
        error_path = RESULTS_DIR / f"{job_id}_error.json"
        with open(error_path, "w") as f:
            json.dump({"error": str(e)}, f)

@app.post("/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload video and start processing"""
    job_id = str(uuid.uuid4())
    file_extension = Path(file.filename).suffix
    video_path = UPLOAD_DIR / f"{job_id}{file_extension}"
    
    # Save uploaded file
    with open(video_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Start background processing
    background_tasks.add_task(process_video, job_id, str(video_path))
    
    return {"job_id": job_id, "status": "processing"}

@app.get("/results/{job_id}")
async def get_results(job_id: str):
    """Get processing results"""
    result_path = RESULTS_DIR / f"{job_id}.json"
    if not result_path.exists():
        return {"status": "processing", "message": "Video is still being processed"}
    
    with open(result_path, "r") as f:
        return json.load(f)

@app.get("/download-pdf/{job_id}")
async def download_pdf(job_id: str):
    """Generate and download PDF report"""
    result_path = RESULTS_DIR / f"{job_id}.json"
    if not result_path.exists():
        return {"error": "Results not found"}
    
    with open(result_path, "r") as f:
        data = json.load(f)
    
    # Create PDF
    pdf = AuditReport()
    pdf.add_comprehensive_report(data)
    
    pdf_path = RESULTS_DIR / f"{job_id}_report.pdf"
    pdf.output(str(pdf_path))
    
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"vantage_report_{job_id}.pdf"
    )

@app.get("/")
async def root():
    return {"message": "Vantage AI API", "version": "1.0.0"}
