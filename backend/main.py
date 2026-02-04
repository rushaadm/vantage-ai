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

# CORS middleware - allow all origins for cloud deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for cloud deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure directories exist
# On Render, persistent disk is mounted at /opt/render/project/src
# Use absolute path to ensure files are saved to persistent disk
BASE_DIR = Path("/opt/render/project/src") if Path("/opt/render/project/src").exists() else Path(".")
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
print(f"📁 Using directories: UPLOAD_DIR={UPLOAD_DIR}, RESULTS_DIR={RESULTS_DIR}")

engine = VantageEngine()
# Removed parallel processing to reduce CPU load

# Removed batch processing to reduce memory usage

def process_video(job_id: str, video_path: str):
    """Granular video processing - optimized for cloud deployment"""
    start_time = time.time()
    max_processing_time = 60  # Increased for cloud processing
    
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        # Granular processing - higher resolution and more frames
        PROCESS_WIDTH = 480  # Higher resolution for granularity
        # Process every 0.1 seconds (10 FPS analysis rate for smooth heatmaps)
        BASE_SAMPLE_RATE = max(1, int(fps / 10))  # 10 samples per second
        if BASE_SAMPLE_RATE < 1:
            BASE_SAMPLE_RATE = 1
        
        # For very long videos, cap at reasonable number but keep granularity
        max_samples = min(300, int(duration * 10))  # Max 300 samples or 10 per second
        if frame_count > max_samples * BASE_SAMPLE_RATE:
            BASE_SAMPLE_RATE = max(1, int(frame_count / max_samples))
        
        # Motion thresholds for adaptive sampling
        motion_threshold_high = 0.1  # High motion threshold
        motion_threshold_low = 0.02   # Low motion threshold
        
        frames_data = []
        all_entropies = []
        all_conflicts = []
        
        frame_idx = 0
        prev_frame = None
        processed_frames_list = []  # For progressive streaming
        adaptive_sample_rate = BASE_SAMPLE_RATE  # Initialize adaptive rate
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Adaptive sampling: adjust rate based on motion
            if prev_frame is not None:
                # Quick motion check
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_gray.shape == curr_gray.shape:
                    diff = cv2.absdiff(prev_gray, curr_gray)
                    motion_level = np.mean(diff) / 255.0
                    
                    # Adjust sample rate based on motion
                    if motion_level > motion_threshold_high:
                        adaptive_sample_rate = max(1, BASE_SAMPLE_RATE // 2)  # More frames (2x)
                    elif motion_level < motion_threshold_low:
                        adaptive_sample_rate = BASE_SAMPLE_RATE * 2  # Fewer frames (0.5x)
                    else:
                        adaptive_sample_rate = BASE_SAMPLE_RATE  # Normal rate
            
            # Skip frames based on adaptive sample rate
            if frame_idx % adaptive_sample_rate != 0:
                frame_idx += 1
                prev_frame = frame  # Update prev_frame even when skipping
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
            
            # Get saliency and motion maps (skip motion if no prev frame to save time)
            saliency_map = engine.get_saliency_map(frame_small)
            
            if prev_frame is not None:
                # Resize prev_frame if needed
                if prev_frame.shape[1] != frame_small.shape[1] or prev_frame.shape[0] != frame_small.shape[0]:
                    prev_frame_small = cv2.resize(prev_frame, (frame_small.shape[1], frame_small.shape[0]), interpolation=cv2.INTER_LINEAR)
                else:
                    prev_frame_small = prev_frame
                motion_map = engine.get_motion_map(prev_frame_small, frame_small)
            else:
                # Skip motion calculation for first frame
                motion_map = np.zeros((frame_small.shape[0], frame_small.shape[1]), dtype=np.float32)
            
            # Calculate metrics
            metrics = engine.calculate_metrics(saliency_map, motion_map)
            all_entropies.append(metrics["entropy"])
            all_conflicts.append(metrics["conflict"])
            
            # Optimized heatmap - balance granularity with file size
            heatmap_h, heatmap_w = saliency_map.shape[:2]
            # Store at 1/3 resolution to reduce JSON size (still granular but manageable)
            target_w = max(48, heatmap_w // 3)  # Reduced from 1/2 to 1/3 for smaller files
            target_h = max(48, heatmap_h // 3)  # Reduced from 1/2 to 1/3
            # Apply Gaussian blur for smoothness before downsampling
            saliency_blurred = cv2.GaussianBlur(saliency_map, (7, 7), 1.5)
            motion_blurred = cv2.GaussianBlur(motion_map, (7, 7), 1.5)
            # Use cubic interpolation for smooth, granular heatmaps
            saliency_small = cv2.resize(saliency_blurred, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            motion_small = cv2.resize(motion_blurred, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            
            # Quantize to reduce precision and file size (round to 2 decimal places)
            saliency_small = np.round(saliency_small * 100) / 100
            motion_small = np.round(motion_small * 100) / 100
            
            # Store frame data (sampled frames only)
            frames_data.append({
                "frame": frame_idx,
                "time": frame_idx / fps if fps > 0 else frame_idx / 30,
                "entropy": metrics["entropy"],
                "conflict": metrics["conflict"],
                "saliency_heatmap": saliency_small.tolist(),
                "motion_heatmap": motion_small.tolist(),
                "original_size": [h, w],
                "heatmap_size": [target_h, target_w]
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
            "status": "complete",  # Explicitly set status to complete
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
        print(f"Saving results to {result_path}")
        print(f"Results: {len(frames_data)} frames, status: complete")
        
        # Save with error handling and file size check
        try:
            with open(result_path, "w") as f:
                json.dump(result_data, f, separators=(',', ':'))  # Compact JSON
            file_size = result_path.stat().st_size
            print(f"✅ Results saved successfully: {file_size / 1024 / 1024:.2f} MB")
            
            # Verify JSON is valid by reading it back
            with open(result_path, "r") as f:
                test_data = json.load(f)
            print(f"✅ JSON validation passed: {len(test_data.get('frames', []))} frames")
        except Exception as e:
            print(f"❌ ERROR saving results: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        print(f"Processing complete for job {job_id} in {time.time() - start_time:.2f}s ({len(frames_data)} frames)")
        
    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()
        # Save error result so frontend stops polling
        error_path = RESULTS_DIR / f"{job_id}.json"
        with open(error_path, "w") as f:
            json.dump({
                "job_id": job_id,
                "status": "error",
                "error": str(e),
                "message": "Video processing failed. Please try again with a different video."
            }, f)

@app.post("/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload video and start processing"""
    try:
        job_id = str(uuid.uuid4())
        file_extension = Path(file.filename).suffix
        
        # Validate file extension
        if file_extension.lower() not in ['.mp4', '.mov', '.avi', '.mkv', '.webm']:
            return {"error": "Unsupported file format. Please upload MP4, MOV, AVI, MKV, or WEBM."}, 400
        
        video_path = UPLOAD_DIR / f"{job_id}{file_extension}"
        
        # Save uploaded file
        with open(video_path, "wb") as f:
            content = await file.read()
            if len(content) == 0:
                return {"error": "Empty file uploaded"}, 400
            f.write(content)
        
        # Start background processing
        background_tasks.add_task(process_video, job_id, str(video_path))
        
        return {"job_id": job_id, "status": "processing"}
    except Exception as e:
        print(f"Upload error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Upload failed: {str(e)}"}, 500

@app.get("/results/{job_id}")
async def get_results(job_id: str):
    """Get processing results (supports progressive/partial results)"""
    result_path = RESULTS_DIR / f"{job_id}.json"
    if not result_path.exists():
        return {"status": "processing", "message": "Video is still being processed"}
    
    try:
        with open(result_path, "r") as f:
            data = json.load(f)
            # If partial results, mark as still processing
            if data.get("partial", False):
                data["status"] = "processing"
            return data
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error for {job_id}: {e}")
        # Return error status instead of crashing
        return {
            "job_id": job_id,
            "status": "error",
            "error": "Results file corrupted",
            "message": "Processing completed but results file is corrupted. Please try uploading again."
        }
    except Exception as e:
        print(f"❌ Error reading results for {job_id}: {e}")
        return {
            "job_id": job_id,
            "status": "error",
            "error": str(e),
            "message": "Error reading results file"
        }

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
