"""
Backend API service for Omics UI.
Integrates Agent_Tools and LLM directories without modifying them.
"""
import os
import sys
import json
import asyncio
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

# Add parent directories to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "Agent_Tools"))
sys.path.insert(0, str(project_root / "llm"))

app = FastAPI(title="Omics UI Backend API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import LLM and Agent_Tools modules
LLM_AVAILABLE = False
AGENT_TOOLS_AVAILABLE = False

try:
    # Import LLM modules
    sys.path.insert(0, str(project_root / "llm"))
    from input import process_user_query, extract_perturbation_info
    from output import interpret_results, collect_results_from_pipeline
    LLM_AVAILABLE = True
    print("✓ LLM modules loaded successfully")
except ImportError as e:
    print(f"⚠ Warning: LLM modules not available: {e}")

try:
    # Import Agent_Tools modules
    sys.path.insert(0, str(project_root / "Agent_Tools"))
    import perturbation_pipeline
    AGENT_TOOLS_AVAILABLE = True
    print("✓ Agent_Tools modules loaded successfully")
except ImportError as e:
    print(f"⚠ Warning: Agent_Tools not available: {e}")


# Request/Response models
class QueryRequest(BaseModel):
    query: str


class PerturbationRequest(BaseModel):
    perturbation_info: Dict[str, Any]
    condition: str  # "Control", "IFNγ", "Co-Culture"
    perturbation_type: str  # "gene" or "drug"


class WorkflowStatus(BaseModel):
    status: str  # "pending", "running", "completed", "error"
    current_step: str
    progress: float  # 0.0 to 1.0
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    logs: Optional[List[Dict[str, Any]]] = None  # Real-time logs from pipeline


# In-memory storage for workflow status (in production, use Redis or database)
workflow_status: Dict[str, WorkflowStatus] = {}
# Store logs for each workflow
workflow_logs: Dict[str, List[Dict[str, Any]]] = {}


@app.get("/")
async def root():
    return {"message": "Omics UI Backend API", "llm_available": LLM_AVAILABLE, "agent_tools_available": AGENT_TOOLS_AVAILABLE}


@app.post("/api/query/process")
async def process_query(request: QueryRequest):
    """
    Process user query using Gemini API from LLM directory.
    Extracts perturbation information.
    """
    if not LLM_AVAILABLE:
        raise HTTPException(status_code=503, detail="LLM modules not available")
    
    try:
        result = process_user_query(request.query)
        return {
            "success": True,
            "perturbation_info": result,
            "original_query": request.query
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/api/workflow/start")
async def start_workflow(request: PerturbationRequest, background_tasks: BackgroundTasks):
    """
    Start perturbation workflow.
    Returns workflow ID immediately, runs workflow in background.
    """
    if not AGENT_TOOLS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Agent_Tools not available")
    
    # Generate workflow ID
    import uuid
    workflow_id = str(uuid.uuid4())
    
    # Initialize status and logs
    workflow_status[workflow_id] = WorkflowStatus(
        status="pending",
        current_step="Initializing",
        progress=0.0,
        logs=[]
    )
    workflow_logs[workflow_id] = []
    
    # Start workflow in background
    background_tasks.add_task(
        run_workflow_background,
        workflow_id,
        request.perturbation_info,
        request.condition,
        request.perturbation_type
    )
    
    return {
        "workflow_id": workflow_id,
        "status": "started"
    }


@app.get("/api/workflow/{workflow_id}/status")
async def get_workflow_status(workflow_id: str):
    """Get current workflow status and results."""
    if workflow_id not in workflow_status:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    status = workflow_status[workflow_id]
    return status.dict()


@app.get("/api/workflow/{workflow_id}/stream")
async def stream_workflow_results(workflow_id: str):
    """
    Stream workflow results as they become available.
    Uses Server-Sent Events (SSE).
    """
    if workflow_id not in workflow_status:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    async def event_generator():
        while True:
            status = workflow_status.get(workflow_id)
            if not status:
                break
            
            # Send current status
            # Get latest logs
            logs = workflow_logs.get(workflow_id, [])
            
            data = {
                "status": status.status,
                "current_step": status.current_step,
                "progress": status.progress,
                "results": status.results,
                "error": status.error,
                "logs": logs[-50:]  # Send last 50 log entries
            }
            yield f"data: {json.dumps(data)}\n\n"
            
            # If completed or error, break
            if status.status in ["completed", "error"]:
                break
            
            # Wait before next update
            await asyncio.sleep(1)
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")


def parse_log_line(line: str) -> Optional[Dict[str, Any]]:
    """
    Parse a log line from Agent_Tools/LLM and categorize it.
    Returns a log entry dict with type and message.
    """
    if not line or not line.strip():
        return None
    
    line = line.strip()
    
    # Categorize log messages based on content
    log_type = "INFO"
    message = line
    
    # Detect step types from Agent_Tools pipeline
    if "PREPARING" in line.upper() or "PREPARING GENE" in line.upper() or "PREPARING DRUG" in line.upper():
        log_type = "INIT"
    elif "RUNNING" in line.upper() and ("INFERENCE" in line.upper() or "STATE" in line.upper() or "SCTRANSLATOR" in line.upper()):
        log_type = "MODEL"
    elif "EVALUATING" in line.upper() or "EVALUATION" in line.upper():
        log_type = "VALIDATE"
    elif "PATHWAY" in line.upper() and ("ANALYSIS" in line.upper() or "ENRICHMENT" in line.upper() or "GSEA" in line.upper()):
        log_type = "COMPUTE"
    elif "COMPLETED" in line.upper() or "FINISHED" in line.upper() or "SUCCESS" in line.upper():
        log_type = "RESULT"
    elif "ERROR" in line.upper() or "FAILED" in line.upper() or "WARNING" in line.upper() or "⚠️" in line:
        log_type = "ERROR"
    elif "=" * 70 in line or "=" * 50 in line:
        # Section headers
        log_type = "PLAN"
    elif "STEP" in line.upper() or "PHASE" in line.upper():
        log_type = "PLAN"
    elif "R2" in line or "Pearson" in line or "RMSE" in line or "MAE" in line:
        log_type = "VALIDATE"
    elif "✓" in line or "✔" in line:
        log_type = "INFO"
    elif "✗" in line or "✖" in line:
        log_type = "ERROR"
    
    import time
    return {
        "type": log_type,
        "message": message,
        "timestamp": time.time()
    }


async def run_workflow_background(
    workflow_id: str,
    perturbation_info: Dict[str, Any],
    condition: str,
    perturbation_type: str
):
    """
    Run workflow in background and update status.
    Calls Agent_Tools perturbation_pipeline.py directly.
    Captures and streams real-time logs.
    """
    try:
        status = workflow_status[workflow_id]
        logs = workflow_logs[workflow_id]
        
        def add_log(log_entry: Dict[str, Any]):
            """Add log entry and update status"""
            if log_entry:
                logs.append(log_entry)
                # Keep only last 1000 logs
                if len(logs) > 1000:
                    logs.pop(0)
                # Update current step from log if it's a PLAN or INIT type
                if log_entry.get("type") in ["PLAN", "INIT", "MODEL", "COMPUTE"]:
                    msg = log_entry.get("message", status.current_step)
                    # Truncate very long messages
                    if len(msg) > 100:
                        msg = msg[:97] + "..."
                    status.current_step = msg
                # Update status logs field
                status.logs = logs[-50:]  # Keep last 50 for status updates
        
        # Add initial log
        add_log({"type": "INIT", "message": f"Initializing Multi-Omics Hypothesis Engine for {perturbation_type} perturbation"})
        
        status.status = "running"
        status.current_step = "Preparing perturbation data"
        status.progress = 0.1
        
        # Extract target from perturbation_info
        target = perturbation_info.get("target", "")
        if not target:
            raise ValueError("No target specified in perturbation_info")
        
        add_log({"type": "INIT", "message": f"Target: {target}, Condition: {condition}, Type: {perturbation_type}"})
        
        # Prepare output directory
        output_dir = project_root / "Agent_Tools" / "temp_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build command to run perturbation_pipeline.py
        status.current_step = "Running perturbation pipeline"
        status.progress = 0.2
        
        pipeline_script = project_root / "Agent_Tools" / "perturbation_pipeline.py"
        
        # Map condition to choice number
        condition_map = {"IFNγ": "1", "Control": "2", "Co-Culture": "3"}
        condition_choice = condition_map.get(condition, "2")
        
        # Build command
        cmd = ["python", str(pipeline_script)]
        
        if perturbation_type == "gene":
            cmd.extend(["--target-gene", target])
        else:
            cmd.extend(["--drug", target])
            cmd.extend(["--perturbation-type", "drug"])
        
        cmd.extend(["--output-dir", str(output_dir)])
        
        add_log({"type": "PLAN", "message": f"Starting Agent_Tools pipeline: {' '.join(cmd)}"})
        
        # Prepare stdin input for interactive prompts
        stdin_input = ""
        if perturbation_type == "gene":
            stdin_input = f"1\n{condition_choice}\n"
        else:
            stdin_input = f"2\n{condition_choice}\n"
        
        # Run pipeline with real-time output capture
        status.current_step = "Executing perturbation → RNA → Protein pipeline"
        status.progress = 0.3
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,  # Combine stderr into stdout
            cwd=str(project_root / "Agent_Tools")
        )
        
        # Send stdin input
        if process.stdin:
            process.stdin.write(stdin_input.encode())
            process.stdin.close()
        
        # Read output line by line in real-time
        if process.stdout:
            buffer = ""
            while True:
                chunk = await process.stdout.read(1024)
                if not chunk:
                    break
                
                buffer += chunk.decode('utf-8', errors='ignore')
                
                # Process complete lines
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line.strip():
                        log_entry = parse_log_line(line)
                        if log_entry:
                            add_log(log_entry)
                            
                            # Update progress based on log content
                            if "RNA" in line.upper() and "PREDICT" in line.upper():
                                status.progress = min(0.5, status.progress + 0.1)
                            elif "PROTEIN" in line.upper() and "PREDICT" in line.upper():
                                status.progress = min(0.7, status.progress + 0.1)
                            elif "PATHWAY" in line.upper():
                                status.progress = min(0.85, status.progress + 0.05)
            
            # Process remaining buffer
            if buffer.strip():
                log_entry = parse_log_line(buffer)
                if log_entry:
                    add_log(log_entry)
        
        # Wait for process to complete
        return_code = await process.wait()
        
        if return_code != 0:
            error_msg = f"Pipeline failed with return code {return_code}"
            add_log({"type": "ERROR", "message": error_msg})
            raise RuntimeError(error_msg)
        
        status.current_step = "Processing results"
        status.progress = 0.85
        add_log({"type": "COMPUTE", "message": "Collecting results from pipeline output"})
        
        # Collect results from output directory
        if LLM_AVAILABLE:
            results = collect_results_from_pipeline(
                str(output_dir),
                target
            )
            add_log({"type": "COMPUTE", "message": "Results collected successfully"})
        else:
            # Fallback: create basic results structure
            results = {
                "target_gene": target,
                "rna_metrics": {},
                "protein_metrics": {},
                "pathway_analysis": {}
            }
        
        status.current_step = "Completed"
        status.progress = 1.0
        status.status = "completed"
        status.results = results
        add_log({"type": "RESULT", "message": "Hypothesis generation complete. All analyses finished successfully."})
            
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        status = workflow_status[workflow_id]
        logs = workflow_logs.get(workflow_id, [])
        logs.append({"type": "ERROR", "message": f"Workflow failed: {error_msg}"})
        status.status = "error"
        status.error = error_msg
        status.progress = 0.0


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

