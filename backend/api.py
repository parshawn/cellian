"""
Backend API service for Cellian Multi-Omics Hypothesis Engine.
Integrates Agent_Tools and LLM directories for full pipeline execution.
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

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Try loading .env from backend directory first
    backend_dir = Path(__file__).parent
    backend_env = backend_dir / '.env'
    if backend_env.exists():
        load_dotenv(backend_env)
        print(f"✓ Loaded .env from {backend_env}")
    # Also try loading from backend/llm directory
    llm_env = backend_dir / 'llm' / '.env'
    if llm_env.exists():
        load_dotenv(llm_env, override=False)  # Don't override if already loaded
        print(f"✓ Loaded .env from {llm_env}")
    # Also try loading from project root
    project_root = backend_dir.parent
    root_env = project_root / '.env'
    if root_env.exists():
        load_dotenv(root_env, override=False)
        print(f"✓ Loaded .env from {root_env}")
except ImportError:
    print("⚠ python-dotenv not installed. Install with: pip install python-dotenv")
    print("  Environment variables must be set manually or via system environment.")

# Get project root (cellian directory)
project_root = Path(__file__).parent.parent
backend_dir = Path(__file__).parent

# Add backend subdirectories to path
sys.path.insert(0, str(backend_dir / "Agent_Tools"))
sys.path.insert(0, str(backend_dir / "llm"))

app = FastAPI(title="Cellian Backend API")

# CORS middleware - allow frontend origin
# Note: Can't use allow_origins=["*"] with allow_credentials=True
# So we explicitly list all possible frontend origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:8080",
        "http://localhost:8081",
        "http://localhost:8082",
        "http://localhost:8083",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8080",
        "http://127.0.0.1:8081",
        "http://127.0.0.1:8082",
        "http://127.0.0.1:8083",
        "http://192.168.0.65:8080",
        "http://192.168.0.65:8081",
        "http://172.17.0.1:8080",
        "http://172.17.0.1:8081",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Import LLM and Agent_Tools modules
LLM_AVAILABLE = False
AGENT_TOOLS_AVAILABLE = False

try:
    # Import LLM modules
    from input import process_user_query, extract_perturbation_info
    from output import interpret_results, collect_results_from_pipeline
    LLM_AVAILABLE = True
    print("✓ LLM modules loaded successfully")
except ImportError as e:
    print(f"⚠ Warning: LLM modules not available: {e}")

try:
    # Import Agent_Tools modules
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
    logs: Optional[List[Dict[str, Any]]] = None
    pipeline_stage: Optional[str] = None  # "perturbation", "rna", "protein", "analysis"


# In-memory storage for workflow status
workflow_status: Dict[str, WorkflowStatus] = {}
workflow_logs: Dict[str, List[Dict[str, Any]]] = {}


@app.get("/")
async def root():
    return {
        "message": "Cellian Backend API",
        "llm_available": LLM_AVAILABLE,
        "agent_tools_available": AGENT_TOOLS_AVAILABLE
    }


@app.post("/api/query/process")
async def process_query(request: QueryRequest):
    """
    Process user query using LLM to extract perturbation information.
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
        logs=[],
        pipeline_stage="perturbation"
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
    Stream workflow results as they become available using Server-Sent Events.
    """
    if workflow_id not in workflow_status:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    async def event_generator():
        while True:
            status = workflow_status.get(workflow_id)
            if not status:
                break
            
            # Get latest logs
            logs = workflow_logs.get(workflow_id, [])
            
            # Convert results to JSON-serializable format (handle DataFrames)
            serializable_results = _make_json_serializable(status.results) if status.results else None
            
            data = {
                "status": status.status,
                "current_step": status.current_step,
                "progress": status.progress,
                "results": serializable_results,
                "error": status.error,
                "logs": _filter_important_logs(logs[-100:]),  # Filter to important logs only
                "pipeline_stage": status.pipeline_stage
            }
            yield f"data: {json.dumps(data, default=str)}\n\n"
            
            # If completed or error, break
            if status.status in ["completed", "error"]:
                break
            
            # Wait before next update
            await asyncio.sleep(1)
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")


def _make_json_serializable(obj: Any) -> Any:
    """Recursively convert objects to JSON-serializable format."""
    import pandas as pd
    import numpy as np
    
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        # Convert DataFrame to list of dicts (records)
        try:
            # Replace NaN/None with null for JSON
            df_clean = obj.fillna('').replace([np.inf, -np.inf], '')
            return df_clean.to_dict('records')
        except Exception as e:
            # If conversion fails, return string representation
            return f"DataFrame({obj.shape[0]} rows, {obj.shape[1]} cols)"
    elif isinstance(obj, pd.Series):
        # Convert Series to dict
        try:
            series_clean = obj.fillna('').replace([np.inf, -np.inf], '')
            return series_clean.to_dict()
        except Exception as e:
            return str(obj)
    elif isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    elif hasattr(obj, 'to_dict'):
        try:
            result = obj.to_dict('records') if hasattr(obj, 'to_dict') else str(obj)
            return _make_json_serializable(result)
        except Exception as e:
            return str(obj)
    elif hasattr(obj, '__dict__'):
        return str(obj)
    else:
        return str(obj)


def _filter_important_logs(logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter logs to only show important ones."""
    important_keywords = [
        "INIT", "MODEL", "COMPUTE", "RESULT", "ERROR",
        "RNA PREDICT", "PROTEIN PREDICT", "PATHWAY",
        "STATE", "SCTRANSLATOR", "COMPLETED", "STARTED",
        "Loading", "Running", "Finished", "Error", "Warning"
    ]
    
    filtered = []
    for log in logs:
        msg = log.get("message", "").upper()
        log_type = log.get("type", "")
        
        # Keep if it's an important type or contains important keywords
        if log_type in ["INIT", "MODEL", "COMPUTE", "RESULT", "ERROR"]:
            filtered.append(log)
        elif any(keyword in msg for keyword in important_keywords):
            filtered.append(log)
        # Skip verbose/debug logs
        elif any(skip in msg for skip in ["DEBUG", "TRACE", "VERBOSE", "INFO:"]):
            continue
        else:
            # Keep first and last few logs for context
            if len(filtered) < 3 or len(logs) - len(filtered) < 3:
                filtered.append(log)
    
    return filtered[:30]  # Limit to 30 most important logs


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
                status.logs = logs[-50:]
        
        # Add initial log
        add_log({"type": "INIT", "message": f"Initializing Multi-Omics Hypothesis Engine for {perturbation_type} perturbation"})
        
        status.status = "running"
        status.current_step = "Preparing perturbation data"
        status.progress = 0.1
        status.pipeline_stage = "perturbation"
        
        # Extract target from perturbation_info (from LLM)
        target = perturbation_info.get("target", "")
        target2 = perturbation_info.get("target2", "")
        has_both = perturbation_info.get("has_both", False)
        pert_type = perturbation_info.get("type", "")
        pert_type2 = perturbation_info.get("type2", "")
        is_comparison = perturbation_info.get("is_comparison", False)
        confidence = perturbation_info.get("confidence", 0.0)
        
        if not target:
            raise ValueError("No target specified in perturbation_info")
        
        # Prepare output directory
        output_dir = backend_dir / "Agent_Tools" / "temp_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file for this workflow
        log_dir = backend_dir.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"workflow_{workflow_id}.log"
        
        # Open log file for writing (keep it open until workflow completes)
        log_fp = open(log_file, 'w', encoding='utf-8')
        import datetime
        log_fp.write(f"=== Workflow {workflow_id} ===\n")
        log_fp.write(f"Started at: {datetime.datetime.now().isoformat()}\n")
        log_fp.write("=" * 80 + "\n\n")
        
        # Log LLM extraction results first
        log_fp.write(f"{'='*80}\n")
        log_fp.write("LLM EXTRACTION RESULTS\n")
        log_fp.write(f"{'='*80}\n")
        log_fp.write(f"Primary Target: {target}\n")
        log_fp.write(f"Primary Type: {pert_type}\n")
        if target2:
            log_fp.write(f"Secondary Target: {target2}\n")
            log_fp.write(f"Secondary Type: {pert_type2}\n")
        log_fp.write(f"Has Both (LLM): {has_both}\n")
        log_fp.write(f"Is Comparison: {is_comparison}\n")
        log_fp.write(f"Confidence: {confidence:.2f}\n")
        log_fp.write(f"{'='*80}\n\n")
        log_fp.flush()
        
        # Always check for both perturbations: if target2 exists, validate if it's gene+drug
        auto_detected = False
        if target2:
            # Check if one looks like a gene (uppercase, short) and one looks like a drug (lowercase, longer, or multi-word)
            is_gene_like = target and len(target) <= 10 and target[0].isupper() and target.isupper()
            is_drug_like = target2 and (target2[0].islower() or len(target2.split()) > 1 or " " in target2)
            if is_gene_like and is_drug_like:
                has_both = True
                auto_detected = True
                log_fp.write(f"  [AUTO-DETECT] Both perturbations detected: {target} (gene-like) + {target2} (drug-like)\n")
                log_fp.write(f"  [AUTO-DETECT] Setting has_both=True (overriding LLM value: {perturbation_info.get('has_both', False)})\n")
                log_fp.flush()
            elif has_both:
                # LLM said has_both=True, validate it
                log_fp.write(f"  [VALIDATE] LLM set has_both=True, target2 exists: {target2}\n")
                log_fp.write(f"  [VALIDATE] Keeping has_both=True from LLM\n")
                log_fp.flush()
            else:
                # target2 exists but doesn't look like gene+drug, or both are same type
                log_fp.write(f"  [INFO] target2 exists ({target2}) but not detected as gene+drug combination\n")
                log_fp.write(f"  [INFO] Keeping has_both=False\n")
                log_fp.flush()
        elif has_both:
            # LLM said has_both=True but no target2 - invalid
            log_fp.write(f"  [WARNING] LLM set has_both=True but target2 is missing. Setting has_both=False.\n")
            has_both = False
            log_fp.flush()
        
        # Enhanced logging to show final perturbation detection
        log_fp.write(f"{'='*80}\n")
        log_fp.write("PERTURBATION DETECTION SUMMARY (FINAL)\n")
        log_fp.write(f"{'='*80}\n")
        log_fp.write(f"Primary Target: {target}\n")
        log_fp.write(f"Primary Type: {pert_type}\n")
        if target2:
            log_fp.write(f"Secondary Target: {target2}\n")
            log_fp.write(f"Secondary Type: {pert_type2}\n")
        log_fp.write(f"Has Both (Final): {has_both}\n")
        if auto_detected:
            log_fp.write(f"  → Auto-detection set has_both=True (gene+drug pattern detected)\n")
        elif has_both:
            log_fp.write(f"  → Using LLM result (has_both=True validated)\n")
        else:
            log_fp.write(f"  → Single perturbation (no target2 or not gene+drug pattern)\n")
        log_fp.write(f"Condition: {condition}\n")
        log_fp.write(f"Perturbation Type (pipeline): {perturbation_type}\n")
        log_fp.write(f"{'='*80}\n\n")
        log_fp.flush()
        
        add_log({"type": "INIT", "message": f"Full logs saved to: {log_file}"})
        
        # Log both perturbations if has_both is true
        if has_both and target2:
            source = "LLM" if not auto_detected else "auto-detection"
            add_log({"type": "INIT", "message": f"✅ BOTH perturbations detected ({source}): Gene={target}, Drug={target2}, Condition={condition}"})
        else:
            # Don't show type if it's "unknown"
            type_str = f" ({pert_type})" if pert_type and pert_type != "unknown" else ""
            add_log({"type": "INIT", "message": f"✅ Single perturbation detected: Target={target}{type_str}, Condition={condition}"})
        
        # Build command to run perturbation_pipeline.py
        status.current_step = "Running perturbation pipeline"
        status.progress = 0.2
        
        pipeline_script = backend_dir / "Agent_Tools" / "perturbation_pipeline.py"
        
        # Map condition to choice number
        condition_map = {"IFNγ": "1", "Control": "2", "Co-Culture": "3", "Co-culture": "3"}
        condition_choice = condition_map.get(condition, "2")
        
        # Check if we need to run both perturbations
        if has_both and target2:
            # Run gene perturbation first
            add_log({"type": "PLAN", "message": f"🚀 Running BOTH perturbations sequentially: Gene={target} ({pert_type}), Drug={target2} ({pert_type2})"})
            
            # First: Gene perturbation
            add_log({"type": "INIT", "message": f"Step 1/2: Running gene perturbation for {target} ({pert_type})"})
            cmd_gene = ["python", str(pipeline_script), "--target-gene", target, "--output-dir", str(output_dir)]
            stdin_input_gene = f"1\n{condition_choice}\n"
            
            # Run gene perturbation
            process_gene = await asyncio.create_subprocess_exec(
                *cmd_gene,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(backend_dir / "Agent_Tools")
            )
            
            if process_gene.stdin:
                process_gene.stdin.write(stdin_input_gene.encode())
                process_gene.stdin.close()
            
            # Stream gene perturbation output
            if process_gene.stdout:
                buffer = ""
                while True:
                    chunk = await process_gene.stdout.read(1024)
                    if not chunk:
                        break
                    buffer += chunk.decode('utf-8', errors='ignore')
                    try:
                        log_fp.write(chunk.decode('utf-8', errors='ignore'))
                        log_fp.flush()
                    except:
                        pass
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        if line.strip():
                            log_entry = parse_log_line(line)
                            if log_entry:
                                add_log(log_entry)
            
            await process_gene.wait()
            if process_gene.returncode != 0:
                raise RuntimeError(f"Gene perturbation failed with return code {process_gene.returncode}")
            
            add_log({"type": "COMPUTE", "message": f"✓ Gene perturbation completed. Starting drug perturbation..."})
            
            log_fp.write(f"\n{'='*80}\n")
            log_fp.write("Step 2/2: Drug Perturbation\n")
            log_fp.write(f"  Target: {target2}\n")
            log_fp.write(f"  Type: {pert_type2}\n")
            log_fp.write(f"  Condition: {condition}\n")
            log_fp.write(f"{'='*80}\n\n")
            log_fp.flush()
            
            # Second: Drug perturbation
            add_log({"type": "INIT", "message": f"Step 2/2: Running drug perturbation for {target2} ({pert_type2})"})
            cmd_drug = ["python", str(pipeline_script), "--drug", target2, "--perturbation-type", "drug", "--output-dir", str(output_dir)]
            stdin_input_drug = f"2\n{condition_choice}\n"
            
            # Run drug perturbation
            process_drug = await asyncio.create_subprocess_exec(
                *cmd_drug,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(backend_dir / "Agent_Tools")
            )
            
            if process_drug.stdin:
                process_drug.stdin.write(stdin_input_drug.encode())
                process_drug.stdin.close()
            
            # Stream drug perturbation output
            if process_drug.stdout:
                buffer = ""
                while True:
                    chunk = await process_drug.stdout.read(1024)
                    if not chunk:
                        break
                    buffer += chunk.decode('utf-8', errors='ignore')
                    try:
                        log_fp.write(chunk.decode('utf-8', errors='ignore'))
                        log_fp.flush()
                    except:
                        pass
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        if line.strip():
                            log_entry = parse_log_line(line)
                            if log_entry:
                                add_log(log_entry)
            
            await process_drug.wait()
            if process_drug.returncode != 0:
                raise RuntimeError(f"Drug perturbation failed with return code {process_drug.returncode}")
            
            add_log({"type": "COMPUTE", "message": "✓ Both perturbations completed successfully!"})
            
            # Collect results for both perturbations
            status.current_step = "Collecting results from both perturbations"
            status.progress = 0.85
            status.pipeline_stage = "analysis"
            
            if LLM_AVAILABLE:
                try:
                    log_fp.write(f"\n{'='*80}\n")
                    log_fp.write("Collecting results from pipeline output (both perturbations)...\n")
                    log_fp.write(f"Output directory: {output_dir}\n")
                    log_fp.write(f"Target 1 (Gene): {target}\n")
                    log_fp.write(f"Target 2 (Drug): {target2}\n")
                    log_fp.flush()
                    
                    # Collect results - will include both if files exist
                    results = collect_results_from_pipeline(
                        str(output_dir),
                        target  # Primary target
                    )
                    
                    # Also try to collect drug results if separate files exist
                    try:
                        drug_results = collect_results_from_pipeline(
                            str(output_dir),
                            target2  # Drug target
                        )
                        # Merge drug results if available
                        if drug_results and "pathway_analysis" in drug_results:
                            if "pathway_analysis" not in results:
                                results["pathway_analysis"] = {}
                            # Add drug-specific keys
                            for key in drug_results["pathway_analysis"]:
                                if key not in results["pathway_analysis"]:
                                    results["pathway_analysis"][f"{key}_drug"] = drug_results["pathway_analysis"][key]
                    except:
                        pass  # Ignore if drug results not found separately
                    
                    log_fp.write(f"Results collected successfully.\n")
                    log_fp.write(f"Results keys: {list(results.keys())}\n")
                    if "pathway_analysis" in results:
                        log_fp.write(f"Pathway analysis keys: {list(results['pathway_analysis'].keys())}\n")
                    log_fp.flush()
                    
                    add_log({"type": "COMPUTE", "message": "Results collected successfully"})
                except Exception as e:
                    import traceback
                    error_trace = traceback.format_exc()
                    log_fp.write(f"\n{'!'*80}\n")
                    log_fp.write(f"ERROR collecting results: {str(e)}\n")
                    log_fp.write(f"{error_trace}\n")
                    log_fp.flush()
                    results = {}
            
            # Set status to completed
            status.status = "completed"
            status.progress = 1.0
            status.current_step = "Both perturbations completed successfully"
            
            # Close log file
            try:
                log_fp.write(f"\n{'='*80}\n")
                log_fp.write(f"Workflow completed at: {datetime.datetime.now().isoformat()}\n")
                log_fp.write(f"Status: completed\n")
                log_fp.write(f"Results collected: {'Yes' if results else 'No'}\n")
                log_fp.close()
            except:
                pass
            
            # Skip the single perturbation code below
            process = None
            stdin_input = ""
            return_code = 0  # Both processes completed successfully
        else:
            # Single perturbation (original code)
            log_fp.write(f"\n{'='*80}\n")
            log_fp.write("EXECUTING SINGLE PERTURBATION\n")
            log_fp.write(f"{'='*80}\n")
            log_fp.write(f"Target: {target}\n")
            log_fp.write(f"Type: {pert_type}\n")
            log_fp.write(f"Perturbation Type (pipeline): {perturbation_type}\n")
            log_fp.write(f"Condition: {condition}\n")
            log_fp.write(f"{'='*80}\n\n")
            log_fp.flush()
            
            cmd = ["python", str(pipeline_script)]
            
            if perturbation_type == "gene":
                cmd.extend(["--target-gene", target])
            else:
                cmd.extend(["--drug", target])
                cmd.extend(["--perturbation-type", "drug"])
            
            cmd.extend(["--output-dir", str(output_dir)])
            
            add_log({"type": "PLAN", "message": f"🚀 Starting {perturbation_type} perturbation pipeline for {target} ({pert_type})"})
            
            # Prepare stdin input for interactive prompts
            stdin_input = ""
            if perturbation_type == "gene":
                stdin_input = f"1\n{condition_choice}\n"
            else:
                stdin_input = f"2\n{condition_choice}\n"
        
        # Update pipeline stage based on progress
        status.pipeline_stage = "perturbation"
        status.progress = 0.3
        
        # Run pipeline with real-time output capture
        status.current_step = "Executing perturbation → RNA → Protein pipeline"
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(backend_dir / "Agent_Tools")
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
                
                # Write raw output to log file
                try:
                    log_fp.write(chunk.decode('utf-8', errors='ignore'))
                    log_fp.flush()
                except:
                    pass
                
                # Process complete lines
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line.strip():
                        log_entry = parse_log_line(line)
                        if log_entry:
                            add_log(log_entry)
                            
                            # Update progress and pipeline stage based on log content
                            line_upper = line.upper()
                            
                            # Stage 1: Perturbation → RNA
                            if ("RNA" in line_upper and "PREDICT" in line_upper) or ("STATE" in line_upper and "INFERENCE" in line_upper):
                                status.pipeline_stage = "rna"
                                status.progress = min(0.5, status.progress + 0.1)
                                add_log({"type": "MODEL", "message": "RNA prediction in progress..."})
                            
                            # Stage 2: RNA → Protein
                            elif ("PROTEIN" in line_upper and "PREDICT" in line_upper) or ("SCTRANSLATOR" in line_upper and "INFERENCE" in line_upper):
                                status.pipeline_stage = "protein"
                                status.progress = min(0.7, status.progress + 0.1)
                                add_log({"type": "MODEL", "message": "Protein prediction in progress..."})
                            
                            # Stage 3: Pathway Analysis
                            elif "PATHWAY" in line_upper or "DIFFERENTIAL" in line_upper or "GSEA" in line_upper:
                                status.pipeline_stage = "analysis"
                                status.progress = min(0.85, status.progress + 0.05)
                                add_log({"type": "COMPUTE", "message": "Pathway analysis in progress..."})
                                # Don't collect results here - wait until end to avoid repeated calls
            
            # Process remaining buffer
            if buffer.strip():
                try:
                    log_fp.write(buffer)
                    log_fp.flush()
                except:
                    pass
                log_entry = parse_log_line(buffer)
                if log_entry:
                    add_log(log_entry)
            
            # Process remaining buffer
            if buffer.strip():
                try:
                    log_fp.write(buffer)
                    log_fp.flush()
                except:
                    pass
        
        # Wait for process to complete
        return_code = await process.wait()
        
        # Log completion status
        try:
            log_fp.write(f"\n{'='*80}\n")
            log_fp.write(f"Process completed with return code: {return_code}\n")
            log_fp.flush()
        except:
            pass
        
        if return_code != 0:
            error_msg = f"Pipeline failed with return code {return_code}"
            add_log({"type": "ERROR", "message": error_msg})
            raise RuntimeError(error_msg)
        
        status.current_step = "Processing results"
        status.progress = 0.85
        status.pipeline_stage = "analysis"
        add_log({"type": "COMPUTE", "message": "Collecting results from pipeline output"})
        
        # Collect results from output directory
        if LLM_AVAILABLE:
            try:
                log_fp.write(f"\n{'='*80}\n")
                log_fp.write("Collecting results from pipeline output...\n")
                log_fp.write(f"Output directory: {output_dir}\n")
                log_fp.write(f"Target: {target}\n")
                log_fp.flush()
                
                results = collect_results_from_pipeline(
                    str(output_dir),
                    target
                )
                
                log_fp.write(f"Results collected successfully.\n")
                log_fp.write(f"Results keys: {list(results.keys())}\n")
                if "pathway_analysis" in results:
                    log_fp.write(f"Pathway analysis keys: {list(results['pathway_analysis'].keys())}\n")
                log_fp.flush()
                
                add_log({"type": "COMPUTE", "message": "Results collected successfully"})
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                log_fp.write(f"\n{'!'*80}\n")
                log_fp.write(f"ERROR collecting results: {str(e)}\n")
                log_fp.write(f"{error_trace}\n")
                log_fp.write(f"{'!'*80}\n")
                log_fp.flush()
                
                print(f"Warning: Could not collect results using LLM output module: {e}")
                add_log({"type": "ERROR", "message": f"Could not collect results: {str(e)}"})
                # Fallback: create basic results structure
                results = {
                    "target_gene": target,
                    "rna_metrics": {},
                    "protein_metrics": {},
                    "pathway_analysis": {}
                }
        else:
            log_fp.write("LLM module not available, using fallback results structure\n")
            log_fp.flush()
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
        status.pipeline_stage = "completed"
        add_log({"type": "RESULT", "message": "Hypothesis generation complete. All analyses finished successfully."})
        
        # Close log file
        try:
            import datetime
            log_fp.write(f"\n{'='*80}\n")
            log_fp.write(f"Workflow completed at: {datetime.datetime.now().isoformat()}\n")
            log_fp.write(f"Status: {status.status}\n")
            log_fp.write(f"Results collected: {'Yes' if results else 'No'}\n")
            log_fp.close()
        except Exception as e:
            print(f"Error closing log file: {e}")
            
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        status = workflow_status[workflow_id]
        logs = workflow_logs.get(workflow_id, [])
        logs.append({"type": "ERROR", "message": f"Workflow failed: {error_msg}"})
        status.status = "error"
        status.error = error_msg
        status.progress = 0.0
        status.pipeline_stage = "error"
        
        # Log error to file if log file exists
        try:
            log_file = backend_dir.parent / "logs" / f"workflow_{workflow_id}.log"
            if log_file.exists():
                with open(log_file, 'a', encoding='utf-8') as log_fp:
                    import datetime
                    log_fp.write(f"\n{'!'*80}\n")
                    log_fp.write(f"WORKFLOW ERROR at {datetime.datetime.now().isoformat()}\n")
                    log_fp.write(f"{error_msg}\n")
                    log_fp.write(f"{'!'*80}\n")
        except:
            pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

