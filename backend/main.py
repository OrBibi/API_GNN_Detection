from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from redis import Redis
from rq import Queue
import os

# Initialize FastAPI app
app = FastAPI(
    title="API Intrusion Detection System",
    description="Backend API for detecting malicious HTTP requests using GNN and Ensemble models.",
    version="1.0.0"
)

# CORS Middleware Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow requests from any origin
    allow_credentials=True,
    allow_methods=["*"],  # allow all HTTP methods
    allow_headers=["*"],
)

# Connect to Redis
redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
redis_conn = Redis.from_url(redis_url)
task_queue = Queue("default", connection=redis_conn)

# Define the request schema for OpenAPI validation
class APIRequest(BaseModel):
    method: str
    path: str
    headers: dict
    body: str = ""

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "API Security Engine is running"}

@app.post("/analyze", status_code=202)
async def analyze_request(request: APIRequest):
    """
    Receives an API request, pushes it to the worker queue for analysis.
    Returns a Job ID for status tracking.
    """
    # English comment: Using full module path for the worker
    job = task_queue.enqueue("worker.tasks.process_request", request.dict())
    
    return {
        "job_id": job.get_id(),
        "status": "Queued",
        "message": "Analysis started in background"
    }

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """
    Check the status and result of a specific analysis job.
    """
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