from openai import OpenAI

host = "http://localhost:8880"

client = OpenAI(base_url=f"{host}/v1", api_key="<random_string>")

for stream in False, True:
    completion = client.chat.completions.create(
        #model="Qwen/Qwen2.5-VL-7B-Instruct",
        model="Qwen/Qwen3-VL-8B-Instruct",
        
        messages=[
            {
                "role": "user",
                "content": "What is the Transformers library known for?"
            }
        ],
        stream=stream
    )
    if stream:
        for chunk in completion:
            token = chunk.choices[0].delta.content
            if token:
                print(token, end='')
    else:
        print(completion.choices[0].message.content)
        
# Force model unload to test cleanup
import httpx
httpx.post(f"{host}/unload")