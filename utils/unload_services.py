import os
import httpx
from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
VLLM_HOST = os.environ.get("VLLM_HOST", "http://localhost:8000")
TRANSFORMERS_HOST = os.environ.get("TRANSFORMERS_HOST", "http://localhost:8001")
API_KEY = os.environ.get("API_KEY", "")

def unload_ollama_models():
    """
    Gracefully sends a request to a local Ollama instance (if any) to unload its active models.
    This prevents memory clashes between Ollama and this server.
    """
    try:
        models = httpx.get(f"{OLLAMA_HOST}/api/ps").json()
        for model in models['models']:
            model_name = model['name']
            httpx.post(f"{OLLAMA_HOST}/api/generate", json={"model": model_name, "keep_alive": 0})
            print(f"Ollama model {model_name} unloaded")
        else:
            print("No Ollama models to unload.")
    except (httpx.ConnectError, httpx.HTTPStatusError) as e:
        print(f"Ollama server not running or unreachable. {e}")


def unload_vllm_models():
    """
    Gracefully sends a request to a local vLLM instance (if any) to unload its active models.
    This prevents memory clashes between vLLM and this server.
    """
    try:
        res = httpx.post(f"{VLLM_HOST}/vllm_proxy/kill", headers={"api_key": API_KEY})
        res.raise_for_status()
        print(res.json()['message'])
    except (httpx.ConnectError, httpx.HTTPStatusError) as e:
        print(f"vLLM server not running or unreachable. {e}")


def unload_trasformers_models():
    """
    Gracefully sends a request to a local Transformers instance (if any) to unload its active models.
    This prevents memory clashes between Transformers and this server.
    """
    try:
        res = httpx.post(f"{TRANSFORMERS_HOST}/unload")
        res.raise_for_status()
        print(res.json()['message'])
    except (httpx.ConnectError , httpx.HTTPStatusError) as e:
        print(f"Transformers server not running or unreachable. {e}")

if __name__ == "__main__":
    unload_ollama_models()
    unload_vllm_models()
    unload_trasformers_models()
