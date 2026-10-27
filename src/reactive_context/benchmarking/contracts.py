"""Public inference input is separate from evaluator-only labels by construction."""

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


class Evidence(BaseModel):
    id: str
    text: str
    source_uri: str
    source_revision: str
    source_span: tuple[int, int] | None = None
    relevance: float = 0


class InferenceCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str
    benchmark: str
    question: str
    question_date: str = ""
    documents: list[Evidence]
    repository: str = ""
    base_commit: str = ""
    language: str = ""


@dataclass(frozen=True)
class BenchmarkCase:
    inference: InferenceCase
    evaluator_labels: dict[str, Any] = field(repr=False)
    subgroup: str = ""
    cluster: str = ""


class TraceRecord(BaseModel):
    benchmark: str
    attempt: int = 1
    harness_revision: str = ""
    benchmark_mode: Literal["official", "extension", "protocol"]
    dataset_revision: str
    case_id: str
    strategy: str
    model_id: str
    agent_runtime_version: str
    random_seed: int
    input_digest: str = ""
    context_ir_digest: str = ""
    source_revisions: dict[str, str] = Field(default_factory=dict)
    official_score: float | None = None
    research_score: float | None = None
    score_origin: str = ""
    scorer_revision: str = ""
    scorer_source_hash: str = ""
    subgroup: str = ""
    cluster: str = ""
    prediction: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    estimated_cost_usd_upper_bound: float | None = None
    refresh_ms: float | None = None
    compile_ms: float | None = None
    end_to_end_ms: float | None = None
    failure_kind: str | None = None
    error_type: str | None = None
    retrieved: list[dict[str, Any]] = Field(default_factory=list)
    included: list[dict[str, Any]] = Field(default_factory=list)
    used: list[dict[str, Any]] | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    request_count: int = 0
    judge_prediction: str | None = None
    noncomparability_reasons: list[str] = Field(default_factory=list)


class DatasetAdapter(Protocol):
    def iter_cases(self, split: str): ...
    def get_official_scorer(self): ...


class ContextStrategy(Protocol):
    async def prepare_request(self, invocation: Any) -> Any: ...
