from pydantic import BaseModel, Field
from typing import Dict


class SentimentRequest(BaseModel):
    text: str = Field(..., example="I love this product!")


class SentimentResponse(BaseModel):
    label: str = Field(..., example="positive")
    confidence: float = Field(..., ge=0.0, le=1.0, example=0.85)
    scores: Dict[str, float] = Field(
        ..., example={"positive": 0.85, "neutral": 0.10, "negative": 0.05}
    )
