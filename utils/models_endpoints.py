import subprocess
import json
from fastapi import HTTPException
from time import time
import os


def get_hf_command():
    """
    Returns the path to the `hf` command.
    Command should be in ~/.local/bin/hf.
    If command is not located there, execute curl -LsSf https://hf.co/cli/install.sh | bash
    to install it.
    """
    hf_path = os.path.join(os.path.expanduser("~"), ".local", "bin", "hf")
    if not os.path.exists(hf_path):
        raise HTTPException(status_code=500, detail="hf command not found. Install with `curl -LsSf https://hf.co/cli/install.sh | bash`.")
    return hf_path


async def get_models():
    """
    Queries the underlying Hugging Face cache using the `hf cache ls` CLI tool.
    Returns raw metadata about locally downloaded models and datasets.
    """
    try:
        hf_path = get_hf_command()
        result = subprocess.run([hf_path, "cache", "ls", "--format", "json"], capture_output=True, text=True, check=True)
        cache_data = json.loads(result.stdout)
        return {"models": cache_data}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Error running hf cache ls: {e}")
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Error parsing JSON: {e}")

def add_models_endpoints(app):
    """Adds the /models and /v1/models endpoints to the FastAPI app."""
    @app.get("/models")
    async def get_models_endpoint():
        """
        Queries the underlying Hugging Face cache using the `hf cache ls` CLI tool.
        Returns raw metadata about locally downloaded models and datasets.
        """
        return await get_models()

    @app.get("/v1/models")
    async def v1_get_models():
        """
        Simulates the OpenAI /v1/models endpoint.
        Retrieves the list of locally cached models matching huggingface cache and structures
        it in the format expected by OpenAI API clients.
        """
        models_response = await get_models()
        models_data = models_response.get("models", [])
        
        data = []
        # Iterate through cached Hugging Face models and datasets
        for m in models_data:
            # Filter for only models (ignoring datasets or other repo types)
            if m['repo_type'] == "model":
                data.append({
                    "id": m.get("repo_id", "unknown-model"), # Map repo_id to model id
                    "object": m.get("repo_type", "model"), # Object is 'model' in OpenAI format
                    "created": int(m.get("last_modified", time())), # Unix timestamp of last modification
                    "owned_by": "huggingface", # Set owner static text
            })
        
        return {
            "object": "list",
            "data": data
        }
            
if __name__ == "__main__":
    print(get_hf_command())
