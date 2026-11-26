"""The golden-set row format. One JSON object per line in data/golden.jsonl."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

QuestionType = Literal["factoid", "multi_hop", "numeric", "unanswerable"]


class GoldenItem(BaseModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_.\-]{1,64}$")
    question: str = Field(min_length=5, max_length=2000)
    type: QuestionType
    expected_doc_ids: list[str] = Field(default_factory=list)  # empty for unanswerable questions
    evidence_quote: str | None = None  # short exact phrase from an expected document
    reference_answer: str | None = None  # needed by Ragas context recall
    source: str = "manual"  # manual | synthetic-reviewed | financebench

    @property
    def answerable(self) -> bool:
        return self.type != "unanswerable"

    @model_validator(mode="after")
    def _consistent(self) -> "GoldenItem":
        if self.answerable:
            if not self.expected_doc_ids:
                raise ValueError(f"{self.id}: answerable questions need expected_doc_ids")
            if not self.reference_answer:
                raise ValueError(f"{self.id}: answerable questions need reference_answer")
        elif self.expected_doc_ids:
            raise ValueError(f"{self.id}: unanswerable questions must have empty expected_doc_ids")
        return self
