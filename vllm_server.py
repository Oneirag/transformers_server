import asyncio
import logging
import os
import signal
import subprocess
import sys
import time
import json
import threading
from datetime import datetime
from typing import Optional, List, Dict, Any

import httpx
import psutil
import uvicorn
from fastapi import FastAPI, Request, Response, Depends, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from dotenv import load_dotenv
from utils.models_endpoints import add_models_endpoints
from utils.unload_services import unload_ollama_models, unload_trasformers_models, unload_vllm_models

# Load environment variables
load_dotenv()

# Configuration
VLLM_PORT = int(os.environ.get("VLLM_PORT", 8889))
PROXY_PORT = int(os.environ.get("PROXY_PORT", 8888))
INACTIVITY_TIMEOUT = int(os.environ.get("INACTIVITY_TIMEOUT", 300))  # 5 minutes
VLLM_CONFIG_FILE = os.environ.get("VLLM_CONFIG_FILE", "vllm_config.json")
ALLOWED_API_KEYS = os.environ.get("VLLM_PROXY_ALLOWED_API_KEYS", "").split(",")

# Security
api_key_header = APIKeyHeader(name="api_key", auto_error=False)

async def verify_api_key(api_key: str = Depends(api_key_header)):
    if not ALLOWED_API_KEYS or (api_key in ALLOWED_API_KEYS and api_key) or ALLOWED_API_KEYS == ['']:
        return api_key
    raise HTTPException(status_code=403, detail="Invalid or missing API Key")

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("vllm_proxy.log")]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="vLLM Lazy Proxy")
add_models_endpoints(app)


class VLLMProxy:

    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.last_activity = datetime.now()
        self.current_model: Optional[str] = None
        self.shutdown_task: Optional[asyncio.Task] = None
        self.is_starting = False
        self.startup_error: Optional[str] = None
        self.log_subscribers: List[asyncio.Queue] = []
        self.startup_log: List[str] = []
        
    def load_defaults(self) -> Dict[str, Any]:
        """Load default arguments from a JSON configuration file."""
        if os.path.exists(VLLM_CONFIG_FILE):
            try:
                with open(VLLM_CONFIG_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading config file {VLLM_CONFIG_FILE}: {e}")
        return {}

    async def start_vllm(self, model: str, extra_args: dict):
        """Launch vLLM subprocess."""
        if self.process or self.is_starting:
            return

        self.is_starting = True
        try:
            unload_ollama_models()
            unload_trasformers_models()
            
            # Construct command
            vllm_path = os.path.join(os.path.dirname(sys.executable), "vllm")
            if not os.path.exists(vllm_path):
                # Fallback to just "vllm" if not found in venv
                vllm_path = "vllm"
                
            cmd = [vllm_path, "serve", model, "--port", str(VLLM_PORT)]

            # Merge default args with extra_args from query string
            # extra_args overrides defaults
            defaults = self.load_defaults()
            vllm_args = {**defaults}
            
            # Apply extra_args and handle the removal logic for False values
            for key, val in extra_args.items():
                # If a parameter is received as "false" (string or bool) and it was in defaults, we might want to skip it
                # The user requirement: "si un parametro se recibe a false entonces si en la configuración estaba en true se debe borrar"
                if str(val).lower() == "false":
                    if key in vllm_args:
                        del vllm_args[key]
                        logger.info(f"Param {key} set to false, removed from vLLM args as requested.")
                else:
                    vllm_args[key] = val

            for key, val in vllm_args.items():
                # Handle flags (values that are empty or True-ish)
                if val is None or val == "" or str(val).lower() == "true":
                    cmd.append(f"--{key.replace('_', '-')}")
                else:
                    cmd.append(f"--{key.replace('_', '-')}")
                    cmd.append(str(val))

            logger.info(f"Starting vLLM: {' '.join(cmd)}")
            self.startup_error = None
            self.startup_log = []
            self.process = subprocess.Popen(
                cmd, 
                start_new_session=True, 
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            self.current_model = model
            
            loop = asyncio.get_running_loop()
            def relay_logs():
                if not self.process or not self.process.stdout:
                    return
                for line in iter(self.process.stdout.readline, ''):
                    if not line:
                        break
                    logger.info(f"[vLLM process] {line.strip()}")
                    self.startup_log.append(line)
                    if len(self.startup_log) > 100:
                        self.startup_log.pop(0)
                    for q in list(self.log_subscribers):
                        try:
                            loop.call_soon_threadsafe(q.put_nowait, line)
                        except asyncio.QueueFull:
                            pass
                        except Exception:
                            pass
            
            threading.Thread(target=relay_logs, daemon=True).start()
            
            # Wait for health check
            logger.info("Waiting for vLLM health check...")
            start_time = time.time()
            async with httpx.AsyncClient() as client:
                while time.time() - start_time < 300:  # 5 min timeout
                    # Check if process died prematurely
                    if self.process.poll() is not None:
                        # Process exited, read captured stdout/stderr
                        error_output = "".join(self.startup_log)
                        self.startup_error = error_output
                        logger.error(f"vLLM failed to start with exit code {self.process.returncode}:\n{error_output}")
                        self.stop_vllm()
                        raise Exception(f"vLLM failed to start: {error_output.strip().splitlines()[-1] if error_output else 'Unknown error'}")

                    try:
                        resp = await client.get(f"http://localhost:{VLLM_PORT}/health")
                        if resp.status_code == 200:
                            logger.info("vLLM is healthy!")
                            break
                    except:
                        pass
                    await asyncio.sleep(2)
                else:
                    logger.error("vLLM health check timed out.")
                    self.stop_vllm()
                    raise Exception("vLLM health check timed out.")
            
            self.last_activity = datetime.now()
            if not self.shutdown_task:
                self.shutdown_task = asyncio.create_task(self.monitor_inactivity())
        finally:
            self.is_starting = False

    def stop_vllm(self) -> bool:
        """Kill vLLM and its child processes."""
        if self.process:
            logger.info(f"Stopping vLLM (PID: {self.process.pid})...")
            try:
                parent = psutil.Process(self.process.pid)
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
            except psutil.NoSuchProcess:
                pass
            self.process = None
            self.current_model = None
            logger.info("vLLM stopped.")
            return True
        
        if self.shutdown_task:
            self.shutdown_task.cancel()
            self.shutdown_task = None
        return False

    async def monitor_inactivity(self):
        """Background task to stop vLLM after inactivity timeout."""
        while True:
            await asyncio.sleep(10)
            if not self.process:
                break
            elapsed = (datetime.now() - self.last_activity).total_seconds()
            if elapsed > INACTIVITY_TIMEOUT:
                logger.info(f"Inactivity timeout ({INACTIVITY_TIMEOUT}s). Shutting down vLLM.")
                self.stop_vllm()
                break
            else:
                logger.info(f"vLLM to be unloaded in {INACTIVITY_TIMEOUT - elapsed:.2f}s.")

proxy = VLLMProxy()

@app.get("/vllm_proxy/status", dependencies=[Depends(verify_api_key)])
async def status():
    elapsed = (datetime.now() - proxy.last_activity).total_seconds()
    remaining = max(0, INACTIVITY_TIMEOUT - elapsed) if proxy.process else 0
    return {
        "status": "running" if proxy.process else "stopped",
        "model": proxy.current_model,
        "seconds_until_shutdown": int(remaining),
        "inactivity_timeout": INACTIVITY_TIMEOUT
    }

@app.post("/vllm_proxy/kill", dependencies=[Depends(verify_api_key)])
async def kill_vllm():
    stopped = proxy.stop_vllm()
    return {"message": "vLLM stopped" if stopped else "No vLLM model to unload"}

@app.get("/vllm_proxy/nvidia-smi", dependencies=[Depends(verify_api_key)])
async def nvidia_smi():
    try:
        res = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        return {"output": res.stdout}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/vllm_proxy/logs", dependencies=[Depends(verify_api_key)])
async def stream_logs():
    if not proxy.process:
        return JSONResponse({"error": "vLLM is not running"}, status_code=400)

    async def log_generator():
        q = asyncio.Queue(maxsize=500)
        proxy.log_subscribers.append(q)
        try:
            while proxy.process and proxy.process.poll() is None:
                try:
                    line = await asyncio.wait_for(q.get(), timeout=1.0)
                    yield line
                except asyncio.TimeoutError:
                    continue
        finally:
            if q in proxy.log_subscribers:
                proxy.log_subscribers.remove(q)

    return StreamingResponse(log_generator(), media_type="text/plain")

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "HEAD"])
async def proxy_all(request: Request, path: str):
    # Determine model and args from request
    query_params = dict(request.query_params)
    
    # Try to get model name from body or query
    model_name = None
    try:
        body = await request.json()
        model_name = body.get("model")
    except:
        body = {}
    
    if not model_name:
        model_name = query_params.get("model", proxy.current_model or "facebook/opt-125m") # Defaulting or using current

    # If vLLM is not running, start it
    try:
        if not proxy.process:
            # Filter query params to pass to vLLM (excluding 'model' if it's there as a helper)
            vllm_args = {k: v for k, v in query_params.items() if k != "model"}
            await proxy.start_vllm(model_name, vllm_args)
        elif model_name != proxy.current_model and model_name is not None and path != "health":
            # Restart if different model requested (and it's not a generic request like health)
            logger.info(f"Switching model from {proxy.current_model} to {model_name}")
            proxy.stop_vllm()
            vllm_args = {k: v for k, v in query_params.items()}
            await proxy.start_vllm(model_name, vllm_args)
    except Exception as e:
        logger.error(f"Failed to start vLLM: {e}")
        return JSONResponse(
            {"error": "vLLM startup failed", "details": str(e), "traceback": proxy.startup_error},
            status_code=500
        )

    proxy.last_activity = datetime.now()

    # Proxy variables
    target_url = f"http://localhost:{VLLM_PORT}/{path}"
    headers = dict(request.headers)
    headers.pop("host", None) # Let httpx handle host

    # Prepare request
    async with httpx.AsyncClient(timeout=600) as client:
        try:
            req = client.build_request(
                method=request.method,
                url=target_url,
                headers=headers,
                content=await request.body(),
                params=query_params
            )
            response = await client.send(req, stream=True)
            
            return StreamingResponse(
                response.aiter_bytes(),
                status_code=response.status_code,
                headers=dict(response.headers)
            )
        except Exception as e:
            logger.error(f"Proxy error: {e}")
            return Response(content=f"Proxy error: {str(e)}", status_code=500)

def signal_handler(sig, frame):
    proxy.stop_vllm()
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    uvicorn.run(app, host="0.0.0.0", port=PROXY_PORT)
