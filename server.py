from fastapi import FastAPI
from pydantic import BaseModel
import requests

app = FastAPI()

OPENROUTER_KEY = "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"  # <-- paste your key here safely

class Query(BaseModel):
    message: str
    model: str = "meta-llama/llama-3.1-8b-instruct"

@app.post("/generate")
def generate_text(data: Query):
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": data.model,
            "messages": [
                {"role": "system", "content": "You give farm advice."},
                {"role": "user", "content": data.message}
            ]
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )

        response.raise_for_status()
        result = response.json()

        return {
            "response": result["choices"][0]["message"]["content"]
        }

    except Exception as e:
        return {"error": str(e)}
