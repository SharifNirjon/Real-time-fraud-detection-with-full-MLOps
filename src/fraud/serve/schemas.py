"""Request / response contracts. Optional raw fields are generated from configs/features.yaml."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from fraud.config import load_feature_config
from fraud.features.pipeline import raw_input_columns


class TransactionCore(BaseModel):
    """Fields every transaction must carry. Unknown extra fields are ignored."""

    model_config = ConfigDict(extra="ignore")

    TransactionID: int = Field(ge=0)
    TransactionDT: int = Field(ge=0, description="seconds since the dataset reference time")
    TransactionAmt: float = Field(gt=0, le=1_000_000)
    ProductCD: Literal["W", "C", "R", "H", "S"]
    card1: int = Field(ge=0)
    addr1: float | None = None
    D1: float | None = Field(default=None, ge=0)


def _build_transaction_model() -> type[BaseModel]:
    numeric, categorical = raw_input_columns(load_feature_config())
    core = set(TransactionCore.model_fields)
    fields: dict[str, Any] = {}
    for c in numeric:
        if c not in core:
            fields[c] = (float | None, None)
    for c in categorical:
        if c not in core:
            fields[c] = (str | None, Field(default=None, max_length=200))
    return create_model("Transaction", __base__=TransactionCore, **fields)


Transaction = _build_transaction_model()


class BatchRequest(BaseModel):
    transactions: list[Transaction] = Field(min_length=1, max_length=1000)  # type: ignore[valid-type]


class ReasonCode(BaseModel):
    feature: str
    value: float | None
    contribution: float


class ScoreResponse(BaseModel):
    transaction_id: int
    fraud_probability: float
    decision: Literal["approve", "review", "block"]
    reason_codes: list[ReasonCode]
    model_version: str
    latency_ms: float


class BatchResponse(BaseModel):
    results: list[ScoreResponse]
    latency_ms: float


class Label(BaseModel):
    TransactionID: int = Field(ge=0)
    isFraud: int = Field(ge=0, le=1)
    TransactionDT: int | None = None


class LabelBatch(BaseModel):
    labels: list[Label] = Field(min_length=1, max_length=10_000)
