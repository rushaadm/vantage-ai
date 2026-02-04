from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
import uuid
import json
from pathlib import Path
import cv2
import numpy as np
import time
import gc
from engine.saliency import VantageEngine
from engine.reporter import AuditReport, get_ai_suggestions

app = FastAPI(title="Vantage AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path("/opt/render/project/src") if Path("/opt/render/project/src").exists() else Path(".")
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

engine = VantageEngine()

def _process_single_frame(frame_idx, frame, prev_frame, fps):
    """Process a single frame - memory efficient"""
    h, w = frame.shape[:2]
    
    if w > 240:
        scale = 240 / w
        new_h = int(h * scale)
        frame_small = cv2.resize(frame, (240, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        frame_small = frame
    
    saliency_map = engine.get_saliency_map(frame_small)
    
    if prev_frame is not None:
        if prev_frame.shape[1] != frame_small.shape[1] or prev_frame.shape[0] != frame_small.shape[0]:
            prev_frame_small = cv2.resize(prev_frame, (frame_small.shape[1], frame_small.shape[0]), interpolation=cv2.INTER_LINEAR)
        else:
            prev_frame_small = prev_frame
        motion_map = engine.get_motion_map(prev_frame_small, frame_small)
    else:
        motion_map = np.zeros((frame_small.shape[0], frame_small.shape[1]), dtype=np.float32)
    
    metrics = engine.calculate_metrics(saliency_map, motion_map)
    
    # Generate heatmap
    heatmap_h, heatmap_w = saliency_map.shape[:2]
    target_w = max(50, heatmap_w // 3)
    target_h = max(50, heatmap_h // 3)
    
    saliency_blurred = cv2.GaussianBlur(saliency_map, (15, 15), 3.0)
    motion_blurred = cv2.GaussianBlur(motion_map, (15, 15), 3.0)
    
    saliency_small = cv2.resize(saliency_blurred, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    motion_small = cv2.resize(motion_blurred, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    
    saliency_small = np.clip(saliency_small, 0, 1)
    saliency_small = np.round(saliency_small * 10) / 10
    motion_small = np.round(motion_small * 10) / 10
    
    # Detect fixation points
    fixation_points = []
    for y in range(2, heatmap_h - 2, 3):
        for x in range(2, heatmap_w - 2, 3):
            val = saliency_map[y, x]
            if val > 0.65:
                is_max = True
                for dy in [-2, -1, 0, 1, 2]:
                    for dx in [-2, -1, 0, 1, 2]:
                        if dx == 0 and dy == 0:
                            continue
                        if y + dy < 0 or y + dy >= heatmap_h or x + dx < 0 or x + dx >= heatmap_w:
                            continue
                        if saliency_map[y + dy, x + dx] > val:
                            is_max = False
                            break
                    if not is_max:
                        break
                if is_max:
                    fixation_points.append({
                        "x": int(x * w / heatmap_w),
                        "y": int(y * h / heatmap_h),
                        "intensity": float(val)
                    })
                    if len(fixation_points) >= 8:
                        break
        if len(fixation_points) >= 8:
            break
    
    frame_time = frame_idx / fps if fps > 0 else frame_idx / 30
    
    return {
        "frame": frame_idx,
        "time": round(frame_time, 2),
        "entropy": round(metrics["entropy"], 2),
        "conflict": round(metrics["conflict"], 2),
        "saliency_heatmap": saliency_small.tolist(),
        "motion_heatmap": motion_small.tolist(),
        "fixation_points": fixation_points[:8],
        "original_size": [h, w],
        "heatmap_size": [target_h, target_w]
    }, metrics, frame_small.copy()

def _process_batch(batch_frames, job_id, fps, prev_frame, all_entropies, all_conflicts):
    """Process a batch and append to results file"""
    result_path = RESULTS_DIR / f"{job_id}.json"
    
    # Load current results
    try:
        with open(result_path, "r") as f:
            data = json.load(f)
    except:
        data = {"frames": []}
    
    batch_results = []
    last_frame = prev_frame
    
    for frame_idx, frame in batch_frames:
        result, metrics, processed_frame = _process_single_frame(frame_idx, frame, last_frame, fps)
        batch_results.append(result)
        all_entropies.append(metrics["entropy"])
        all_conflicts.append(metrics["conflict"])
        last_frame = processed_frame
        
        # Free memory
        del frame
    
    # Append to results file
    data["frames"].extend(batch_results)
    data["processed_frames"] = len(data["frames"])
    data["status"] = "processing"
    
    # Atomic write
    temp_path = RESULTS_DIR / f"{job_id}.json.tmp"
    with open(temp_path, "w") as f:
        json.dump(data, f, separators=(',', ':'))
    temp_path.replace(result_path)
    
    print(f"✅ Processed batch: {len(batch_results)} frames, total: {data['processed_frames']}")
    
    return last_frame

def _finalize_results(job_id, all_entropies, all_conflicts, start_time):
    """Finalize results with stats"""
    result_path = RESULTS_DIR / f"{job_id}.json"
    
    try:
        with open(result_path, "r") as f:
            data = json.load(f)
    except:
        return
    
    if all_entropies:
        avg_entropy = float(np.mean(all_entropies))
        avg_conflict = float(np.mean(all_conflicts))
        clarity_score = max(0, min(100, 100 - (avg_conflict * 10)))
    else:
        avg_entropy = 0
        avg_conflict = 0
        clarity_score = 50
    
    try:
        ai_narrative = get_ai_suggestions({
            "avg_entropy": avg_entropy,
            "avg_conflict": avg_conflict,
            "clarity_score": clarity_score
        })
    except Exception as e:
        print(f"AI suggestions failed: {e}")
        ai_narrative = "Focus on reducing motion conflict and increasing visual clarity for better engagement."
    
    data["status"] = "complete"
    data["stats"] = {
        "avg_entropy": avg_entropy,
        "avg_conflict": avg_conflict,
        "clarity_score": clarity_score
    }
    data["ai_suggestions"] = ai_narrative
    data["processing_time"] = round(time.time() - start_time, 2)
    
    temp_path = RESULTS_DIR / f"{job_id}.json.tmp"
    with open(temp_path, "w") as f:
        json.dump(data, f, separators=(',', ':'))
    temp_path.replace(result_path)
    
    elapsed = time.time() - start_time
    print(f"✅ Processing complete: {data['processed_frames']} frames in {elapsed:.1f}s")

def process_video(job_id: str, video_path: str):
    """Progressive processing - stream results in chunks"""
    start_time = time.time()
    
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        result_path = RESULTS_DIR / f"{job_id}.json"
        
        # Initialize result file
        initial_data = {
            "job_id": job_id,
            "status": "processing",
            "fps": float(fps),
            "frame_count": frame_count,
            "duration": round(duration, 2),
            "frames": [],
            "processed_frames": 0
        }
        
        with open(result_path, "w") as f:
            json.dump(initial_data, f, separators=(',', ':'))
        
        print(f"📹 Processing {frame_count} frames at {fps} FPS (progressive batches)")
        
        # Process in small batches
        batch_size = 25  # Small batches to stay under memory limit
        all_entropies = []
        all_conflicts = []
        
        frame_idx = 0
        prev_frame = None
        batch_frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                # Process remaining batch
                if batch_frames:
                    prev_frame = _process_batch(batch_frames, job_id, fps, prev_frame, all_entropies, all_conflicts)
                break
            
            batch_frames.append((frame_idx, frame.copy()))
            del frame  # Free immediately
            
            # Process batch when full
            if len(batch_frames) >= batch_size:
                prev_frame = _process_batch(batch_frames, job_id, fps, prev_frame, all_entropies, all_conflicts)
                batch_frames = []
                gc.collect()  # Force garbage collection
            
            frame_idx += 1
        
        cap.release()
        
        # Finalize
        _finalize_results(job_id, all_entropies, all_conflicts, start_time)
        
    except Exception as e:
        print(f"❌ Processing error: {e}")
        import traceback
        traceback.print_exc()
        
        result_path = RESULTS_DIR / f"{job_id}.json"
        with open(result_path, "w") as f:
            json.dump({
                "job_id": job_id,
                "status": "error",
                "error": str(e),
                "message": "Video processing failed. Please try again."
            }, f)

@app.post("/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload video and start processing"""
    try:
        job_id = str(uuid.uuid4())
        file_extension = Path(file.filename).suffix
        
        if file_extension.lower() not in ['.mp4', '.mov', '.avi', '.mkv', '.webm']:
            return {"error": "Unsupported file format"}, 400
        
        video_path = UPLOAD_DIR / f"{job_id}{file_extension}"
        
        with open(video_path, "wb") as f:
            content = await file.read()
            if len(content) == 0:
                return {"error": "Empty file"}, 400
            f.write(content)
        
        background_tasks.add_task(process_video, job_id, str(video_path))
        
        return {"job_id": job_id, "status": "processing"}
    except Exception as e:
        print(f"Upload error: {e}")
        return {"error": f"Upload failed: {str(e)}"}, 500

@app.get("/results/{job_id}")
async def get_results(job_id: str):
    """Get processing results (supports partial results)"""
    result_path = RESULTS_DIR / f"{job_id}.json"
    if not result_path.exists():
        return {"status": "processing", "message": "Video is still being processed"}
    
    try:
        with open(result_path, "r") as f:
            data = json.load(f)
        
        # Return partial results if still processing
        if data.get("status") == "processing":
            data["partial"] = True
        
        return data
    except json.JSONDecodeError as e:
        return {
            "job_id": job_id,
            "status": "error",
            "error": "Results file corrupted"
        }
    except Exception as e:
        return {
            "job_id": job_id,
            "status": "error",
            "error": str(e)
        }

@app.get("/results-stream/{job_id}")
async def stream_results(job_id: str):
    """Stream results using Server-Sent Events - progressive updates"""
    async def event_generator():
        import asyncio
        max_wait = 600  # 10 minutes max
        check_interval = 0.5  # Check every 0.5 seconds for faster updates
        elapsed = 0
        last_frame_count = 0
        
        while elapsed < max_wait:
            result_path = RESULTS_DIR / f"{job_id}.json"
            
            if result_path.exists():
                try:
                    with open(result_path, "r") as f:
                        data = json.load(f)
                    
                    # Send update if new frames available
                    current_frame_count = len(data.get("frames", []))
                    if current_frame_count > last_frame_count or data.get("status") == "complete":
                        yield f"data: {json.dumps(data)}\n\n"
                        last_frame_count = current_frame_count
                        
                        if data.get("status") == "complete":
                            return
                    else:
                        # Send progress update
                        yield f"data: {json.dumps({'status': 'processing', 'processed_frames': current_frame_count, 'message': f'Processing... ({current_frame_count} frames)'})}\n\n"
                except:
                    yield f"data: {json.dumps({'status': 'error', 'error': 'Results file corrupted'})}\n\n"
                    return
            else:
                yield f"data: {json.dumps({'status': 'processing', 'message': f'Starting... ({elapsed}s)'})}\n\n"
            
            await asyncio.sleep(check_interval)
            elapsed += check_interval
        
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
    return {"message": "Vantage AI API", "version": "2.0.0"}
