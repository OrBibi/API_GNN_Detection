from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from redis import Redis
from rq import Queue
import os
import sys

# Ensure root is in path
sys.path.append('/app')

# Import the task function
from worker.tasks import process_request

app = FastAPI(title="API Intrusion Detection System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect to Redis
redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
redis_conn = Redis.from_url(redis_url)
task_queue = Queue("default", connection=redis_conn)

@app.get("/")
async def root():
    return {"message": "API Security Engine is running"}

@app.post("/analyze", status_code=202)
async def analyze_request(full_log: dict):
    """
    Receives a full API log (JSON) and enqueues it for the worker.
    """
    # Enqueue using the imported function object for reliability
    job = task_queue.enqueue(process_request, full_log)
    
    return {
        "job_id": job.get_id(),
        "status": "Queued",
        "message": "Full log received and queued for analysis"
    }

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    job = task_queue.fetch_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.is_finished:
        return {
            "job_id": job_id,
            "status": "Completed",
            "result": job.result 
        }
    
    return {
        "job_id": job_id,
        "status": job.get_status()
    }