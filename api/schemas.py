from pydantic import BaseModel, Field
from typing import Optional


class PredictionResponse(BaseModel):
    filename:        str
    prediction:      str          = Field(..., examples=["HEALTHY", "UNHEALTHY"])
    confidence:      float        = Field(..., ge=0.0, le=1.0)
    is_tomato:       bool
    tomato_similarity: float      = Field(..., description="Cosine sim vs. training centroid")
    model_used:      str
    message:         Optional[str] = None


class RejectionResponse(BaseModel):
    filename:        str
    rejected:        bool = True
    reason:          str
    tomato_similarity: float
    threshold_used:  float


class HealthResponse(BaseModel):
    status:      str
    model_loaded: bool
    model_type:   Optional[str] = None
    version:      str = "1.0.0"


class ClassesResponse(BaseModel):
    classes: list[str]
    label_map: dict[str, int]