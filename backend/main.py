from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
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
    """Process ALL frames - complete analysis"""
    start_time = time.time()
    # No time limit - process all frames (user wants complete analysis)
    
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        # Process ALL frames - lightweight but complete
        PROCESS_WIDTH = 320  # Moderate resolution for speed
        # Process EVERY frame (no sampling)
        
        frames_data = []
        all_entropies = []
        all_conflicts = []
        
        frame_idx = 0
        prev_frame = None
        
        print(f"Processing ALL {frame_count} frames at {fps} FPS...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every frame - no skipping
            
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
            
            # Generalized, brighter heatmap - focus on main areas
            heatmap_h, heatmap_w = saliency_map.shape[:2]
            # Store at 1/6 resolution - more generalized, less granular
            target_w = max(32, heatmap_w // 6)  # More generalized
            target_h = max(32, heatmap_h // 6)  # More generalized
            
            # Heavy blur for generalization (focus on main areas, not details)
            saliency_blurred = cv2.GaussianBlur(saliency_map, (15, 15), 3.0)  # Larger blur
            motion_blurred = cv2.GaussianBlur(motion_map, (15, 15), 3.0)
            
            # Resize with linear interpolation
            saliency_small = cv2.resize(saliency_blurred, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            motion_small = cv2.resize(motion_blurred, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            
            # Apply threshold to focus on main areas only (brighter = more focused)
            threshold = 0.3  # Only show top 30% of attention
            saliency_small = np.maximum(0, saliency_small - threshold) / (1 - threshold)  # Normalize after threshold
            saliency_small = np.clip(saliency_small, 0, 1)
            
            # Quantize to 1 decimal place for file size
            saliency_small = np.round(saliency_small * 10) / 10
            motion_small = np.round(motion_small * 10) / 10
            
            # Convert to list
            saliency_list = saliency_small.tolist()
            motion_list = motion_small.tolist()
            
            # Detect fixation points (local maxima in saliency)
            fixation_points = []
            for y in range(1, heatmap_h - 1):
                for x in range(1, heatmap_w - 1):
                    val = saliency_map[y, x]
                    if val > 0.6:  # High attention threshold
                        # Check if local maximum
                        is_max = True
                        for dy in [-1, 0, 1]:
                            for dx in [-1, 0, 1]:
                                if dx == 0 and dy == 0:
                                    continue
                                if saliency_map[y + dy, x + dx] > val:
                                    is_max = False
                                    break
                            if not is_max:
                                break
                        if is_max:
                            # Store in original video coordinates
                            fixation_points.append({
                                "x": int(x * w / heatmap_w),
                                "y": int(y * h / heatmap_h),
                                "intensity": float(val)
                            })
            
            # Store frame data - ALL frames with exact timing
            frame_time = frame_idx / fps if fps > 0 else frame_idx / 30
            frames_data.append({
                "frame": frame_idx,
                "time": round(frame_time, 3),  # More precise timing (3 decimals)
                "entropy": round(metrics["entropy"], 2),
                "conflict": round(metrics["conflict"], 2),
                "saliency_heatmap": saliency_list,
                "motion_heatmap": motion_list,
                "fixation_points": fixation_points[:10],  # Top 10 fixation points per frame
                "original_size": [h, w],
                "heatmap_size": [target_h, target_w]
            })
            
            prev_frame = frame.copy()
            frame_idx += 1
            
            # Progress update every 100 frames
            if frame_idx % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Processed {frame_idx}/{frame_count} frames ({frame_idx/frame_count*100:.1f}%) in {elapsed:.1f}s")
        
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
            # Write to temp file first, then rename (atomic write)
            temp_path = result_path.with_suffix('.json.tmp')
            
            # Use ensure_ascii=False and compact format
            with open(temp_path, "w", encoding='utf-8') as f:
                json.dump(result_data, f, separators=(',', ':'), ensure_ascii=False)
            
            # Verify temp file is valid JSON before renaming
            with open(temp_path, "r", encoding='utf-8') as f:
                test_data = json.load(f)
            
            # Atomic rename (prevents corruption)
            temp_path.replace(result_path)
            
            file_size = result_path.stat().st_size
            print(f"✅ Results saved successfully: {file_size / 1024 / 1024:.2f} MB")
            print(f"✅ JSON validation passed: {len(test_data.get('frames', []))} frames")
        except json.JSONEncodeError as e:
            print(f"❌ JSON encode error: {e}")
            # Try to save without heatmaps as fallback
            result_data_no_heatmaps = result_data.copy()
            for frame in result_data_no_heatmaps.get('frames', []):
                frame.pop('saliency_heatmap', None)
                frame.pop('motion_heatmap', None)
            with open(result_path, "w", encoding='utf-8') as f:
                json.dump(result_data_no_heatmaps, f, separators=(',', ':'))
            print(f"⚠️ Saved without heatmaps due to size issue")
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

@app.get("/results-stream/{job_id}")
async def stream_results(job_id: str):
    """Stream results using Server-Sent Events - no polling needed!"""
    async def event_generator():
        import asyncio
        max_wait = 300  # Max 5 minutes
        check_interval = 1  # Check every second
        elapsed = 0
        
        while elapsed < max_wait:
            result_path = RESULTS_DIR / f"{job_id}.json"
            
            if result_path.exists():
                try:
                    with open(result_path, "r") as f:
                        data = json.load(f)
                    
                    # Send complete results
                    yield f"data: {json.dumps(data)}\n\n"
                    return
                except json.JSONDecodeError:
                    yield f"data: {json.dumps({'status': 'error', 'error': 'Results file corrupted'})}\n\n"
                    return
                except Exception as e:
                    yield f"data: {json.dumps({'status': 'error', 'error': str(e)})}\n\n"
                    return
            else:
                # Send progress update
                yield f"data: {json.dumps({'status': 'processing', 'message': f'Processing... ({elapsed}s elapsed)'})}\n\n"
            
            await asyncio.sleep(check_interval)
            elapsed += check_interval
        
        # Timeout
        yield f"data: {json.dumps({'status': 'error', 'error': 'Processing timeout'})}\n\n"
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")

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
