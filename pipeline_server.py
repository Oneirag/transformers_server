from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import torch
from transformers import pipeline
from time import time, sleep
import threading
from contextlib import asynccontextmanager
import subprocess
import json
import os
import sys
import asyncio
import httpx
import gc
from threading import Thread
from dotenv import load_dotenv

# Load environment variables from .env file
# An HF_TOKEN is recommended to download models from Hugging Face
load_dotenv()

if not os.environ.get("HF_TOKEN"):
    raise ValueError("HF_TOKEN environment variable is not set")    

# Default model
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

OLLAMA_HOST = httpx.URL(os.environ.get("OLLAMA_HOST", "http://localhost:11434"))

# Timeout for unloading models (300s by default)
TIMEOUT = 300

class ModelCache:
    def __init__(self):
        self.current_model = None
        self.current_pipeline = None
        self.last_used = None

model_cache = ModelCache()

# Lock to serialize requests
request_lock = asyncio.Lock()

class Message(BaseModel):
    role: str
    content: Any  # Can be str or list of dicts

class ChatCompletionRequest(BaseModel):
    model: str = DEFAULT_MODEL
    messages: List[Message]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: Optional[bool] = False
    # Other optional parameters...

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


def unload_model_cache():
    if model_cache.current_pipeline is not None:
        print(f"Unloading model: {model_cache.current_model}")
        
        # Explicitamente limpiar del y vaciar
        if hasattr(model_cache.current_pipeline, "model"):
            del model_cache.current_pipeline.model
        if hasattr(model_cache.current_pipeline, "tokenizer"):
            del model_cache.current_pipeline.tokenizer
            
        del model_cache.current_pipeline
        model_cache.current_pipeline = None
        model_cache.current_model = None
        model_cache.last_used = None
        
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print("Model unloaded and GPU memory cleared.")
        print("Note: ~300-600MB of GPU memory may remain allocated by the PyTorch CUDA context.")
        print("To completely release 100% of GPU memory, the Python process must be terminated.")
        
def cleanup():
    while True:
        sleep(10)  # Check every 10 seconds
        if model_cache.current_pipeline is not None:
            remaining = TIMEOUT - (time() - model_cache.last_used)
            if remaining > 0:
                print(f"Model {model_cache.current_model} will be unloaded in {remaining:.0f} seconds")
            else:
                unload_model_cache()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    threading.Thread(target=cleanup, daemon=True).start()
    yield
    # Shutdown (nothing needed for daemon thread)

app = FastAPI(lifespan=lifespan)

def unload_ollama_models():
    try:
        models = httpx.get(f"{OLLAMA_HOST}/api/ps").json()
        for model in models['models']:
            model_name = model['name']
            httpx.post(f"{OLLAMA_HOST}/api/generate", json={"model": model_name, "keep_alive": 0})
            print(f"Ollama model {model_name} unloaded")
    except httpx.ConnectError:
        print("Ollama server not running or unreachable.")

def get_pipeline(model_name: str):
    current_time = time()
    if model_cache.current_model != model_name:
        if model_cache.current_pipeline is not None:
            unload_model_cache()
        print(f"Loading pipeline for model: {model_name}")
        tic = current_time
        model_cache.current_pipeline = pipeline("image-text-to-text", 
                                                model=model_name, 
                                                trust_remote_code=True, 
                                                device_map="auto", 
                                                dtype=torch.bfloat16)
        model_cache.current_model = model_name
        model_cache.last_used = current_time
        print(f"Pipeline loaded in {time() - tic:.2f}s")
    else:
        # Update last_used
        model_cache.last_used = current_time
    return model_cache.current_pipeline

@app.post("/v1/chat/completions")
@app.get("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    async with request_lock:
        unload_ollama_models()  # Ensure Ollama models are unloaded before processing new request   
        try:
            model_name = request.model

            models = await get_models()
            names = [m['repo_id'] for m in models['models']]
            if model_name not in names:
                raise HTTPException(status_code=400, detail=f"Model '{model_name}' is not downloaded. Available models are: {', '.join(names)}")

            messages = request.messages
            # Adaptar messages al formato esperado por el pipeline
            for user_message in messages:
                if isinstance(user_message.content, str):
                    # Si es string, convertir a list con text
                    user_message.content = [{"type": "text", "text": user_message.content}]
                else:
                    # Mapear image_url a url
                    for item in user_message.content:
                        if item.get("type") == "image_url":
                            item["type"] = "image"
                            item["url"] = item["image_url"]["url"] if isinstance(item["image_url"], dict) else item["image_url"]
            # Convert to dicts
            adapted_messages = [m.model_dump() for m in messages]    

            # Get pipeline
            pipe = get_pipeline(model_name)

            # Call the pipeline
            tic = time()
            # Use generation parameters to make the output longer
            generate_kwargs = {
                "max_new_tokens": request.max_tokens or 2048,
                "temperature": request.temperature or 0.01, # low temperature by default for more deterministic output
                "do_sample": True if (request.temperature or 0.7) > 0 else False,
            }

            if request.stream:
                from transformers import TextIteratorStreamer
                from fastapi.responses import StreamingResponse
                
                # Setup streamer
                streamer = TextIteratorStreamer(pipe.tokenizer, skip_prompt=True, skip_special_tokens=True)
                
                # Start pipeline in a separate thread because pipeline() is blocking
                def run_pipeline():
                    # Monkey-patch generate to inject the streamer because pipeline strips it out
                    original_generate = pipe.model.generate
                    def patched_generate(*args, **kwargs):
                        kwargs["streamer"] = streamer
                        # Ensure generation parameters are applied
                        kwargs.update(generate_kwargs)
                        return original_generate(*args, **kwargs)
                    
                    try:
                        pipe.model.generate = patched_generate
                        with torch.no_grad():
                            pipe(text=adapted_messages, **generate_kwargs)
                    finally:
                        pipe.model.generate = original_generate

                thread = Thread(target=run_pipeline)
                thread.start()
                
                async def event_generator():
                    # Stream the output
                    for new_text in streamer:
                        chunk = {
                            "id": "chatcmpl-" + str(int(time())),
                            "object": "chat.completion.chunk",
                            "created": int(time()),
                            "model": model_name,
                            "choices": [{"delta": {"content": new_text}, "index": 0, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                        await asyncio.sleep(0.001)  # small sleep to yield control
                        
                    # Final chunk indicating stop
                    final_chunk = {
                        "id": "chatcmpl-" + str(int(time())),
                        "object": "chat.completion.chunk",
                        "created": int(time()),
                        "model": model_name,
                        "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                    
                return StreamingResponse(event_generator(), media_type="text/event-stream")
            else:
                with torch.no_grad():
                    res = pipe(text=adapted_messages, **generate_kwargs)
                elapsed = time() - tic

                output_text = res[0]['generated_text'][-1]['content'] if res[0]['generated_text'] else ""
                print(f"Model: {model_name}, Output: {output_text}, Time taken: {elapsed:.2f}s")

                # Create response in OpenAI format
                response = ChatCompletionResponse(
                    id="chatcmpl-" + str(int(time())),
                    created=int(time()),
                    model=model_name,
                    choices=[
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": output_text
                            },
                            "finish_reason": "stop"
                        }
                    ],
                    usage={
                        "prompt_tokens": 0,  # Placeholder
                        "completion_tokens": 0,  # Placeholder
                        "total_tokens": 0  # Placeholder
                    }
                )
                return response

        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status():
    if model_cache.current_model is not None:
        remaining = TIMEOUT - (time() - model_cache.last_used)
        if remaining > 0:
            return {
                "loaded_model": model_cache.current_model,
                "remaining_time_seconds": remaining,
                "status": "loaded"
            }
        else:
            return {
                "loaded_model": model_cache.current_model,
                "remaining_time_seconds": 0,
                "status": "expiring_soon"
            }
    else:
        return {
            "loaded_model": None,
            "remaining_time_seconds": None,
            "status": "no_model_loaded"
        }

@app.post("/unload")
async def unload_model():
    if model_cache.current_pipeline is not None:
        model_name = model_cache.current_model
        print(f"Manually unloading model: {model_name}")
        unload_model_cache()
        return {
            "message": f"Model {model_name} unloaded successfully. Note: Some GPU memory (around 356MB) may still be in use by CUDA context.",
            "status": "unloaded"
        }
    else:
        return {
            "message": "No model is currently loaded.",
            "status": "no_action"
        }

@app.get("/models")
async def get_models():
    try:
        # Use full path to hf from the virtual environment
        venv_bin = os.path.dirname(sys.executable)
        hf_path = os.path.join(venv_bin, "hf")
        result = subprocess.run([hf_path, "cache", "ls", "--format", "json"], capture_output=True, text=True, check=True)
        cache_data = json.loads(result.stdout)
        return {"models": cache_data}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Error running hf cache ls: {e}")
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Error parsing JSON: {e}")
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="hf command not found. Ensure the virtual environment is activated.")

if __name__ == "__main__":
    import uvicorn
    import traceback
    uvicorn.run(app, host="0.0.0.0", port=8880)
