# Transformers Server - OpenAI-compatible API

This FastAPI server simulates the OpenAI API for chat completions, using transformers pipelines for vision-language models. It supports loading only one model in memory at a time, with automatic unloading due to inactivity.

## Features

- **OpenAI-compatible API**: `/v1/chat/completions` endpoint for generating text from images and text.
- **Single model cache**: Only one model in memory; automatic unloading after 300 seconds of inactivity.
- **Supported models**: Hugging Face models like `Qwen/Qwen2.5-VL-7B-Instruct`, `microsoft/Fara-7B`, etc.
- **Memory management**: Automatic GPU memory release.
- **Additional endpoints**: Status, manual unload, and list of cached models.

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure you have GPU access (optional, but recommended for large models).

## Running the Server

You can configure the server using environment variables or a `.env` file:
- `PIPELINE_SERVER_HOST`: The host interface to bind to (default: `0.0.0.0`).
- `PIPELINE_SERVER_PORT`: The port to listen on (default: `8880`).
- `PIPELINE_SERVER_TIMEOUT`: The idle timeout in seconds before unloading the model (default: `300`).

```bash
PIPELINE_SERVER_PORT=8000 python pipeline_server.py
```

The server will be available at `http://localhost:8880` by default.

## Endpoints

### POST /v1/chat/completions

Simulates the OpenAI chat completions endpoint. Supports messages with text and images.

**Example request:**

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "Qwen/Qwen2.5-VL-7B-Instruct",
       "messages": [
         {
           "role": "user",
           "content": [
             {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
             {"type": "text", "text": "Describe the image."}
           ]
         }
       ],
       "max_tokens": 512,
       "temperature": 0.7
     }'
```

**Parameters:**
- `model`: Hugging Face model name (default: `Qwen/Qwen2.5-VL-7B-Instruct`).
- `messages`: List of messages in OpenAI format, with support for `image_url`.
- `max_tokens`: Maximum number of tokens to generate (default: 512).
- `temperature`: Temperature for generation (default: 0.7).

### GET /status

Returns the status of the loaded model and the remaining time until automatic unloading.

**Example:**
```bash
curl http://localhost:8000/status
```

**Response:**
```json
{
  "loaded_model": "Qwen/Qwen2.5-VL-7B-Instruct",
  "remaining_time_seconds": 250,
  "status": "loaded"
}
```

### POST /unload

Manually unloads the loaded model and frees GPU memory.

**Example:**
```bash
curl -X POST http://localhost:8000/unload
```

**Response:**
```json
{
  "message": "Model Qwen/Qwen2.5-VL-7B-Instruct unloaded successfully. Note: Some GPU memory (around 356MB) may still be in use by CUDA context.",
  "status": "unloaded"
}
```

### GET /models

Lists locally cached models and datasets using `hf cache ls`.

**Example:**
```bash
curl http://localhost:8000/models
```

**Response:**
```json
{
  "models": [
    {
      "repo_id": "Qwen/Qwen2.5-VL-7B-Instruct",
      "repo_type": "model",
      "size_on_disk": 123456789
    }
  ]
}
```

### GET /v1/models

Lists locally cached models in a format compatible with the OpenAI API.

**Example:**
```bash
curl http://localhost:8000/v1/models
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen/Qwen2.5-VL-7B-Instruct",
      "object": "model",
      "created": 1709230000,
      "owned_by": "huggingface"
    }
  ]
}
```

## Notes

- **Model caching**: Models are loaded on first use and kept in memory. If a different model is requested, the current one is unloaded automatically.
- **GPU memory**: After unloading a model, ~356MB of memory may still be in use by the CUDA context.
- **Configuration**: The unload timeout is 300 seconds; you can change `TIMEOUT` in the code.
- **Vision models**: For images, use public URLs or base64 (not implemented yet, but extensible).
- **Security**: This is a local server; do not expose to the internet without authentication.

## Development

- Based on FastAPI and transformers.
- Background thread for automatic cleanup.
- GPU-compatible (device_map="auto").

For more details, check the code in `pipeline_server.py`.
