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
import os
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
    """Process a single frame - ultra memory efficient"""
    h, w = frame.shape[:2]
    
    # Resize for processing
    if w > 200:
        scale = 200 / w
        new_h = int(h * scale)
        frame_small = cv2.resize(frame, (200, new_h), interpolation=cv2.INTER_LINEAR)
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
    
    # Generate smooth, rounded heatmap
    heatmap_h, heatmap_w = saliency_map.shape[:2]
    target_w = max(40, heatmap_w // 4)  # Smaller for memory
    target_h = max(40, heatmap_h // 4)
    
    # Heavy blur for smooth, rounded appearance
    saliency_blurred = cv2.GaussianBlur(saliency_map, (21, 21), 4.0)  # Larger blur for roundness
    motion_blurred = cv2.GaussianBlur(motion_map, (21, 21), 4.0)
    
    saliency_small = cv2.resize(saliency_blurred, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    motion_small = cv2.resize(motion_blurred, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    
    saliency_small = np.clip(saliency_small, 0, 1)
    saliency_small = np.round(saliency_small * 10) / 10
    motion_small = np.round(motion_small * 10) / 10
    
    # Detect fixation points (eye-tracking data)
    fixation_points = []
    for y in range(3, heatmap_h - 3, 4):
        for x in range(3, heatmap_w - 3, 4):
            val = saliency_map[y, x]
            if val > 0.6:
                is_max = True
                for dy in [-3, -2, -1, 0, 1, 2, 3]:
                    for dx in [-3, -2, -1, 0, 1, 2, 3]:
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
                        "intensity": float(val),
                        "duration": 0.1  # Estimated fixation duration
                    })
                    if len(fixation_points) >= 10:
                        break
        if len(fixation_points) >= 10:
            break
    
    frame_time = frame_idx / fps if fps > 0 else frame_idx / 30
    
    # Calculate eye-tracking metrics
    avg_saliency = float(np.mean(saliency_map))
    max_saliency = float(np.max(saliency_map))
    attention_spread = float(np.std(saliency_map))
    
    return {
        "frame": frame_idx,
        "time": round(frame_time, 2),
        "entropy": round(metrics["entropy"], 2),
        "conflict": round(metrics["conflict"], 2),
        "saliency_heatmap": saliency_small.tolist(),
        "motion_heatmap": motion_small.tolist(),
        "fixation_points": fixation_points[:10],
        "original_size": [h, w],
        "heatmap_size": [target_h, target_w],
        # Eye-tracking psychology stats
        "avg_saliency": round(avg_saliency, 3),
        "max_saliency": round(max_saliency, 3),
        "attention_spread": round(attention_spread, 3),
        "fixation_count": len(fixation_points)
    }, metrics, frame_small.copy() if frame_idx % 20 == 19 else None  # Only keep every 20th for motion

def process_video(job_id: str, video_path: str):
    """NUCLEAR OPTION: Process in batches, save incrementally, delete frames immediately"""
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
        
        print(f"📹 Processing {frame_count} frames (NUCLEAR MODE: batches of 20)")
        
        batch_size = 20
        all_entropies = []
        all_conflicts = []
        all_saliencies = []
        all_attention_spreads = []
        total_fixations = 0
        
        frame_idx = 0
        prev_frame = None
        batch_results = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                # Process remaining batch
                if batch_results:
                    _save_batch(job_id, batch_results, all_entropies, all_conflicts)
                    batch_results = []
                break
            
            # Process frame
            result, metrics, keep_frame = _process_single_frame(frame_idx, frame, prev_frame, fps)
            batch_results.append(result)
            
            all_entropies.append(metrics["entropy"])
            all_conflicts.append(metrics["conflict"])
            all_saliencies.append(result["avg_saliency"])
            all_attention_spreads.append(result["attention_spread"])
            total_fixations += result["fixation_count"]
            
            # Only keep frame for motion if it's the last of a batch
            if keep_frame is not None:
                prev_frame = keep_frame
            else:
                prev_frame = None
            
            # DELETE frame immediately
            del frame
            
            # Process batch when full
            if len(batch_results) >= batch_size:
                _save_batch(job_id, batch_results, all_entropies, all_conflicts)
                batch_results = []
                
                # AGGRESSIVE garbage collection
                gc.collect()
                gc.collect()  # Twice for good measure
            
            frame_idx += 1
        
        cap.release()
        
        # Delete video file to free storage
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
                print(f"🗑️ Deleted video file: {video_path}")
        except:
            pass
        
        # Finalize with psychology stats
        _finalize_results(job_id, all_entropies, all_conflicts, all_saliencies, all_attention_spreads, total_fixations, start_time)
        
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
                "message": "Video processing failed."
            }, f)

def _save_batch(job_id, batch_results, all_entropies, all_conflicts):
    """Save batch to JSON and immediately delete from memory"""
    result_path = RESULTS_DIR / f"{job_id}.json"
    
    try:
        with open(result_path, "r") as f:
            data = json.load(f)
    except:
        data = {"frames": []}
    
    # Append batch
    data["frames"].extend(batch_results)
    data["processed_frames"] = len(data["frames"])
    data["status"] = "processing"
    
    # Atomic write
    temp_path = RESULTS_DIR / f"{job_id}.json.tmp"
    with open(temp_path, "w") as f:
        json.dump(data, f, separators=(',', ':'))
    temp_path.replace(result_path)
    
    print(f"💾 Saved batch: {len(batch_results)} frames, total: {data['processed_frames']}")
    
    # DELETE batch from memory
    del batch_results
    gc.collect()

def _finalize_results(job_id, all_entropies, all_conflicts, all_saliencies, all_attention_spreads, total_fixations, start_time):
    """Finalize with psychology/eye-tracking stats"""
    result_path = RESULTS_DIR / f"{job_id}.json"
    
    try:
        with open(result_path, "r") as f:
            data = json.load(f)
    except:
        return
    
    # Calculate psychology stats
    if all_entropies:
        avg_entropy = float(np.mean(all_entropies))
        max_entropy = float(np.max(all_entropies))
        min_entropy = float(np.min(all_entropies))
        std_entropy = float(np.std(all_entropies))
        
        avg_conflict = float(np.mean(all_conflicts))
        max_conflict = float(np.max(all_conflicts))
        
        avg_saliency = float(np.mean(all_saliencies))
        max_saliency = float(np.max(all_saliencies))
        
        avg_attention_spread = float(np.mean(all_attention_spreads))
        
        clarity_score = max(0, min(100, 100 - (avg_conflict * 10)))
        
        # Eye-tracking psychology metrics
        fixation_rate = total_fixations / len(data["frames"]) if data["frames"] else 0
        attention_stability = 100 - (std_entropy * 10)  # Lower entropy = more stable
        engagement_score = (avg_saliency * 50) + (attention_stability * 0.5)
    else:
        avg_entropy = max_entropy = min_entropy = std_entropy = 0
        avg_conflict = max_conflict = 0
        avg_saliency = max_saliency = 0
        avg_attention_spread = 0
        clarity_score = 50
        fixation_rate = 0
        attention_stability = 50
        engagement_score = 50
    
    try:
        ai_narrative = get_ai_suggestions({
            "avg_entropy": avg_entropy,
            "avg_conflict": avg_conflict,
            "clarity_score": clarity_score,
            "fixation_rate": fixation_rate,
            "engagement_score": engagement_score
        })
    except Exception as e:
        print(f"AI suggestions failed: {e}")
        ai_narrative = "Focus on reducing motion conflict and increasing visual clarity for better engagement."
    
    data["status"] = "complete"
    data["stats"] = {
        "avg_entropy": round(avg_entropy, 2),
        "max_entropy": round(max_entropy, 2),
        "min_entropy": round(min_entropy, 2),
        "std_entropy": round(std_entropy, 2),
        "avg_conflict": round(avg_conflict, 2),
        "max_conflict": round(max_conflict, 2),
        "clarity_score": round(clarity_score, 2),
        # Eye-tracking psychology stats
        "avg_saliency": round(avg_saliency, 3),
        "max_saliency": round(max_saliency, 3),
        "avg_attention_spread": round(avg_attention_spread, 3),
        "total_fixations": total_fixations,
        "fixation_rate": round(fixation_rate, 2),
        "attention_stability": round(attention_stability, 2),
        "engagement_score": round(engagement_score, 2)
    }
    data["ai_suggestions"] = ai_narrative
    data["processing_time"] = round(time.time() - start_time, 2)
    
    temp_path = RESULTS_DIR / f"{job_id}.json.tmp"
    with open(temp_path, "w") as f:
        json.dump(data, f, separators=(',', ':'))
    temp_path.replace(result_path)
    
    elapsed = time.time() - start_time
    print(f"✅ Complete: {data['processed_frames']} frames in {elapsed:.1f}s")

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
    """Get results - ONLY return if complete"""
    result_path = RESULTS_DIR / f"{job_id}.json"
    if not result_path.exists():
        return {"status": "processing", "message": "Video is still being processed"}
    
    try:
        with open(result_path, "r") as f:
            data = json.load(f)
        
        # ONLY return if complete - no partial results
        if data.get("status") != "complete":
            return {"status": "processing", "message": f"Processing... ({data.get('processed_frames', 0)} frames done)"}
        
        return data
    except json.JSONDecodeError:
        return {"job_id": job_id, "status": "error", "error": "Results file corrupted"}
    except Exception as e:
        return {"job_id": job_id, "status": "error", "error": str(e)}

@app.get("/results-stream/{job_id}")
async def stream_results(job_id: str):
    """Stream results - only send when complete"""
    async def event_generator():
        import asyncio
        max_wait = 600
        check_interval = 1.0
        elapsed = 0
        
        while elapsed < max_wait:
            result_path = RESULTS_DIR / f"{job_id}.json"
            
            if result_path.exists():
                try:
                    with open(result_path, "r") as f:
                        data = json.load(f)
                    
                    # Only send when COMPLETE
                    if data.get("status") == "complete":
                        yield f"data: {json.dumps(data)}\n\n"
                        return
                    else:
                        # Just send progress, no partial data
                        processed_count = data.get('processed_frames', 0)
                        progress_msg = f'Processing... ({processed_count} frames)'
                        yield f"data: {json.dumps({'status': 'processing', 'processed_frames': processed_count, 'message': progress_msg})}\n\n"
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
