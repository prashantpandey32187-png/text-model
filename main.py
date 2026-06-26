import os
import sys
import time
import uuid
import torch
import trimesh
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from multiprocessing import Process, Queue, Manager

# Initialize FastAPI app
app = FastAPI(title="Native Multi-GPU 3D Load-Balanced API")

# Add current directory to path to allow absolute imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from generation.mesh_utils import convert_obj_to_stl

# Globals initialization
task_queue = Queue()          # Incoming jobs queue
task_status_dict = None       # Will be initialized safely in startup

# Ensure output directory exists
os.makedirs("outputs", exist_ok=True)

class TextRequest(BaseModel):
    prompt: str

# --- THE WORKER PROCESS CODE (Runs on each GPU) ---
def gpu_worker_process(gpu_id: int, input_queue: Queue, status_dict: dict):
    """
    This process lives forever, listens to the queue, and runs on its dedicated GPU.
    """
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Worker started successfully on {device} (PID: {os.getpid()})")
    
    while True:
        try:
            # Wait for a job indefinitely
            task_id, prompt = input_queue.get()
            
            status_dict[task_id] = {"status": "Processing", "gpu_used": gpu_id}
            print(f"⚙️ [GPU {gpu_id}] Processing prompt: '{prompt}'")
            
            temp_obj_path = os.path.join("outputs", f"{task_id}.obj")
            final_stl_path = os.path.join("outputs", f"{task_id}.stl")
            
            # --- SIMULATE AI GENERATION ---
            time.sleep(5) # Simulating GPU work time
            
            # Creating dummy object mesh
            dummy_mesh = trimesh.creation.cone(radius=1, height=2)
            dummy_mesh.export(temp_obj_path)
            # -------------------------------
            
            # Convert to STL
            convert_obj_to_stl(temp_obj_path, final_stl_path)
            
            if os.path.exists(temp_obj_path):
                os.remove(temp_obj_path)
                
            # Mark as completed
            status_dict[task_id] = {
                "status": "Completed",
                "file_path": final_stl_path,
                "gpu_used": gpu_id
            }
            print(f"✅ [GPU {gpu_id}] Finished task: {task_id}")
            
        except Exception as e:
            if 'task_id' in locals():
                status_dict[task_id] = {"status": "Failed", "error": str(e)}
            print(f"❌ Error in Worker GPU {gpu_id}: {e}")

# --- FASTAPI LIFECYCLE MANAGEMENT ---
@app.on_event("startup")
def startup_event():
    """
    Automatically spans 4 worker processes safely on API start.
    """
    global task_status_dict
    num_gpus = 4 # 4-GPU Rig Setup
    
    print(f"--- Spawning {num_gpus} Native GPU Workers Safely ---")
    
    # Safely initialize Manager only when the main app process starts
    mp_manager = Manager()
    task_status_dict = mp_manager.dict()
    
    for gpu_id in range(num_gpus):
        p = Process(
            target=gpu_worker_process, 
            args=(gpu_id, task_queue, task_status_dict),
            daemon=True
        )
        p.start()

# --- API ENDPOINTS ---
@app.post("/api/v1/generate")
async def create_generation_job(request: TextRequest):
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
        
    job_id = str(uuid.uuid4())
    
    # Initialize state in our shared dict
    task_status_dict[job_id] = {"status": "In Queue"}
    task_queue.put((job_id, request.prompt))
    
    return {
        "message": "Job queued successfully",
        "task_id": job_id,
        "status_endpoint": f"/api/v1/status/{job_id}"
    }

@app.get("/api/v1/status/{task_id}")
async def check_job_status(task_id: str):
    if task_status_dict is None or task_id not in task_status_dict:
        raise HTTPException(status_code=404, detail="Job not found")
        
    job_info = task_status_dict[task_id]
    
    if job_info["status"] == "In Queue":
        return {"status": "Processing / In Queue"}
    elif job_info["status"] == "Processing":
        return {"status": "Processing", "gpu_used": job_info.get("gpu_used")}
    elif job_info["status"] == "Completed":
        return {
            "status": "Completed",
            "gpu_used": job_info.get("gpu_used"),
            "download_url": f"/api/v1/download/{task_id}"
        }
    else:
        return {"status": "Failed", "error": job_info.get("error")}

@app.get("/api/v1/download/{task_id}")
async def download_stl_file(task_id: str):
    if task_status_dict is None or task_id not in task_status_dict:
        raise HTTPException(status_code=404, detail="Job not found")
        
    job_info = task_status_dict[task_id]
    if job_info["status"] == "Completed":
        file_path = job_info.get("file_path")
        if os.path.exists(file_path):
            return FileResponse(
                path=file_path, 
                filename=f"model_{task_id}.stl", 
                media_type='application/octet-stream'
            )
    raise HTTPException(status_code=404, detail="STL File not found or task not completed yet")

# 🌟 Windows Security Shield: This block blocks infinite sub-process loops
if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)