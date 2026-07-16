from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class ToolDecision(BaseModel):
    action: Literal["user_db_only", "vector_db_only", "user_db_then_vector_db", "plain_answer"]
    reason: str = Field(default="")
    user_db_fields: List[str] = Field(default_factory=list)
    search_query_hint: str = Field(default="")


class RouterDecision(BaseModel):
    intent: Literal["single", "uncertain"] = Field(...)
    targets: List[str] = Field(default_factory=list)
    scope: str = Field(default="")
    clarifying_question: Optional[str] = Field(default=None)