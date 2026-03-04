from pydantic import BaseModel
from typing import List, Optional
import httpx
import base64
import json

PROMPT = """

  Parse the given CA Form 460 Schedule into JSON.

  For any dates, provide the date in the format YYYY-MM-DD.
  If a field is not applicable, use `null` for the value.
"""


class Form460ScheduleALineItem(BaseModel):
    date_received: str
    full_name: str
    city: str
    state: str
    zipcode: str
    contributor_code: str
    occupation: str
    employer: str
    amount_this_period: float
    amount_cumulative_calendar_year: float
    amount_per_election_code: Optional[str]
    amount_per_election: Optional[float]


class Form460ScheduleA(BaseModel):
    line_items: List[Form460ScheduleALineItem]

OPENROUTER_API_KEY = ""

def parse_schedule_a_page(prompt: str, image: bytes, model = "google/gemini-3-flash-preview") -> Form460ScheduleA:
  response = httpx.post(
      "https://openrouter.ai/api/v1/chat/completions",
      headers={
          "Authorization": f"Bearer {OPENROUTER_API_KEY}",
          "Content-Type": "application/json",
      },
      json={
          "model": model,
          "messages": [
              {
                  "role": "user",
                  "content": [
                      {"type": "text", "text": PROMPT},
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
                  "name": "Form460ScheduleA",
                  "strict": True,
                  "schema": Form460ScheduleA.model_json_schema(),
              },
          },
      },
      timeout=60.0,
  )

  data = response.json()
  content = json.loads(data["choices"][0]["message"]["content"])
  return Form460ScheduleA.model_validate(content)

image = open("tmp.png", "rb").read()
parsed = parse_schedule_a_page(PROMPT, image)
print(parsed.model_dump_json(indent=2))
