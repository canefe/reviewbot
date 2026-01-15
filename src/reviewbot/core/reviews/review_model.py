from pydantic import BaseModel


class ReviewSummary(BaseModel):
    summary: str
