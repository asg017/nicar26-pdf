import base64
import json
import os

import httpx
from pydantic import BaseModel


def parse_page(
    prompt: str, image: bytes, schema=BaseModel, model="google/gemini-3-flash-preview"
) -> BaseModel:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Set OPENROUTER_API_KEY before calling parse_schedule_a_page")

    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64.b64encode(image).decode()}"
                        },
                    },
                ],
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "strict": True,
                "schema": schema.model_json_schema(),
            },
        },
    }

    response = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=60.0,
    )
    response.raise_for_status()

    data = response.json()
    content = json.loads(data["choices"][0]["message"]["content"])
    return schema.model_validate(content)