from typing import Dict, Literal
from pydantic import BaseModel, Field, field_validator, confloat

LABELS = ("negative", "neutral", "positive")


class SentimentRequest(BaseModel):
    """
    Request model for sentiment prediction.
    Contains the input text to be analyzed for sentiment.
    """

    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,  # Limit input size to prevent DoS
        example="I love this product",
        description="Text to analyze for sentiment (max 10000 characters)"
    )


class SentimentResponse(BaseModel):
    """
    Response model for sentiment prediction results.
    Includes the predicted label, its confidence score, and a normalized
    probability distribution over all sentiment classes.
    """

    label: Literal["negative", "neutral", "positive"]
    confidence: confloat(ge=0, le=1) = Field(..., example=0.98)
    scores: Dict[str, float]

    @field_validator("scores")
    def validate_scores(cls, v):
        """
        Validates that sentiment scores contain all expected labels and sum to 1.
        Ensures the score dictionary has exactly the required keys and represents
        a normalized probability distribution within a defined tolerance.
        """
        expected = set(LABELS)
        if set(v.keys()) != expected:
            raise ValueError(f"'scores' must contain exactly these keys: {expected}")
        s = sum(v.values())
        if abs(s - 1.0) > 1e-3:
            raise ValueError("'scores' must sum to 1.0 (within tolerance 1e-3)")
        return v

    model_config = {
        "json_schema_extra": {
            "examples": {
                "normal": {
                    "label": "positive",
                    "confidence": 0.9,
                    "scores": {"negative": 0.0, "neutral": 0.1, "positive": 0.9},
                }
            }
        }
    }


__all__ = ["SentimentRequest", "SentimentResponse"]
