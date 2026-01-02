from pydantic import BaseModel
from typing import Dict

class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    label: str
    confidence: float
    scores: Dict[str, float]
