import base64
import os

import httpx
from pydantic import BaseModel


def parse_page(
    prompt: str, image: bytes, schema=BaseModel, model="google/gemini-3-flash-preview"
) -> dict:
    """Send a page image to a vision LLM and return structured data.

    Uses the OpenRouter API to send a prompt and a page image to a
    vision-capable model, requesting a response that conforms to the
    provided Pydantic schema.

    More info about the OpenRouter API here: https://openrouter.ai/docs/guides/features/structured-outputs

    Args:
        prompt: Instructions telling the model what to extract from the image.
        image: Raw PNG bytes of the page to analyze.
        schema: A Pydantic BaseModel subclass defining the expected response
            structure. The model is constrained to return JSON matching this schema.
        model: OpenRouter model identifier to use for the request.

    Returns:
        A dict matching the fields defined in ``schema``.

    Raises:
        ValueError: If the OPENROUTER_API_KEY environment variable is not set.
        httpx.HTTPStatusError: If the API returns a non-2xx response.
    """
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
    content = data["choices"][0]["message"]["content"]
    return schema.model_validate_json(content).model_dump()