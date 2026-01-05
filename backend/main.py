import os
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from redis import Redis
from rq import Queue

# Import task
from worker.tasks import analyze_parquet_task

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup Paths
BASE_DIR = os.path.abspath("/app")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULTS_DIR = os.path.join(BASE_DIR, "backend", "static", "results")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "backend", "static")), name="static")

# Redis
redis_conn = Redis(host='redis', port=6379)
q = Queue(connection=redis_conn)

@app.get("/")
def read_root():
    return {"message": "API is up and running", "docs_url": "/docs"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    job = q.enqueue(analyze_parquet_task, file_path, job_timeout='20m') # Increased timeout for building graphs
    return {"job_id": job.get_id()}

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """
    Check RQ job status including 'stage' and 'progress'.
    """
    from rq.job import Job
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except Exception:
        return {"state": "unknown", "error": "Job not found"}

    job.refresh()
    
    # Get custom metadata
    progress = job.meta.get('progress', 0)
    total_samples = job.meta.get('total_samples', 0)
    stage = job.meta.get('stage', 'Queued') # New Field

    response = {
        "state": job.get_status(),
        "progress": progress,
        "total_samples": total_samples,
        "stage": stage
    }

    if job.is_finished:
        response["state"] = "finished"
        response["result"] = job.result
    elif job.is_failed:
        response["state"] = "failed"
        response["error"] = str(job.exc_info)
    
    return response

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, filename=filename)
    return {"error": "File not found"}