"""Secret-safe run configuration; the request budget is shared by all strategies and judges."""

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dotenv import dotenv_values


@dataclass(frozen=True)
class BenchmarkConfig:
    model_id: str = "deepseek-flash"
    base_url: str = "https://api.deepseek.com"
    api_key: str = field(default="", repr=False)
    judge_model: str = "deepseek-flash"
    judge_base_url: str = "https://api.deepseek.com"
    judge_key: str = field(default="", repr=False)
    seed: int = 17
    pilot_cases: int = 30
    max_cost_usd: float = 5.0
    max_requests: int = 400
    concurrency: int = 2
    timeout_seconds: int = 120
    max_output_tokens: int = 1024
    context_tokens_proxy: int = 128000
    rag_sessions: int = 10
    code_context_tokens_proxy: int = 12000
    candidate_code_spans: int = 80
    retrieval_chunk_lines: int = 80
    input_miss_usd_per_million: float = 0.3
    input_hit_usd_per_million: float = 0.006
    output_usd_per_million: float = 1.2
    pricing_basis: str = "DeepSeek Flash peak prices, verified 2026-10-23; conservative upper bound"
    provider_announced_version: str = "DeepSeek-V4.1-Flash-1017"
    history_context_ratio: float = 0.8

    @classmethod
    def load(cls, path: str | Path = ".env"):
        values = {**dotenv_values(path), **os.environ}

        def number(key, default):
            return int(values.get(key) or default)

        key = values.get("DEEPSEEK_API_KEY") or ""
        judge_key_name = values.get("BENCH_JUDGE_API_KEY_ENV") or "DEEPSEEK_API_KEY"
        return cls(
            model_id=values.get("DEEPSEEK_MODEL") or "deepseek-flash",
            base_url=values.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com",
            api_key=key,
            judge_model=values.get("BENCH_JUDGE_MODEL") or "deepseek-flash",
            judge_base_url=values.get("BENCH_JUDGE_BASE_URL") or "https://api.deepseek.com",
            judge_key=values.get(judge_key_name) or "",
            pilot_cases=number("BENCH_PILOT_CASES", 30),
            max_cost_usd=float(values.get("BENCH_MAX_COST_USD") or 5),
            max_requests=number("BENCH_MAX_REQUESTS", 400),
            concurrency=number("BENCH_MAX_CONCURRENCY", 2),
            timeout_seconds=number("BENCH_REQUEST_TIMEOUT_SECONDS", 120),
            history_context_ratio=float(values.get("BENCH_HISTORY_CONTEXT_RATIO") or 0.8),
        )

    def public(self):
        return {
            key: value for key, value in asdict(self).items() if key not in {"api_key", "judge_key"}
        }

    def __post_init__(self):
        if min(self.pilot_cases, self.max_requests, self.concurrency, self.timeout_seconds) <= 0:
            raise ValueError("benchmark limits must be positive")
        if self.max_cost_usd <= 0:
            raise ValueError("cost limit must be positive")
        if not 0 < self.history_context_ratio <= 1:
            raise ValueError("history context ratio must be in (0, 1]")
