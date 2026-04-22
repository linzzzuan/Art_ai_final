"""Dataset response schemas."""

from pydantic import BaseModel


class ClassInfo(BaseModel):
    name: str
    count: int
    percentage: float


class DatasetInfoResponse(BaseModel):
    name: str
    path: str | None
    total_samples: int
    image_size: str
    num_classes: int
    classes: list[str]


class ClassDistributionResponse(BaseModel):
    classes: list[ClassInfo]
