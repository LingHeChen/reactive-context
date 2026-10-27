"""One Pydantic AI runtime and shared budget across answer, compression and scoring calls."""

import asyncio
import json
from dataclasses import dataclass, field, replace
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from openai import AsyncOpenAI
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.capabilities import AbstractCapability, Caching
from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelRequest
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.deepseek import DeepSeekProvider

from ..errors import ContextError


class BudgetExceeded(ContextError):
    code = "benchmark_budget_exceeded"


def request_projection(messages, parameters):
    def project_content(value):
        if isinstance(value, BinaryContent):
            return {
                "binary_sha256": sha256(value.data).hexdigest(),
                "bytes": len(value.data),
                "media_type": value.media_type,
            }
        if isinstance(value, (list, tuple)):
            return [project_content(item) for item in value]
        if isinstance(value, dict):
            return {key: project_content(item) for key, item in value.items()}
        return value

    result = []
    for message in messages:
        parts = []
        for part in message.parts:
            item = {"kind": part.part_kind}
            for name in ("content", "tool_name", "tool_call_id", "args"):
                if hasattr(part, name):
                    item[name] = project_content(getattr(part, name))
            parts.append(item)
        if parts:
            result.append({"kind": message.kind, "parts": parts})
    return {
        "messages": result,
        "instructions": [part.content for part in parameters.instruction_parts or []],
        "tools": [
            {
                "name": tool.name,
                "schema": tool.parameters_json_schema,
                "description": tool.description,
            }
            for tool in parameters.function_tools
        ],
        "output_tools": [
            {
                "name": tool.name,
                "schema": tool.parameters_json_schema,
                "description": tool.description,
            }
            for tool in parameters.output_tools
        ],
    }


@dataclass
class CallRecord:
    phase: str
    input_digest: str
    input_bytes: int
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_upper_bound: float = 0
    duration_ms: float = 0
    response_model: str = ""
    response_id: str | None = None
    error_type: str | None = None
    request_file: str | None = None
    response_file: str | None = None
    input_image_count: int = 0


@dataclass
class CallTrace:
    calls: list[CallRecord] = field(default_factory=list)
    capture_directory: Path | None = None

    def totals(self):
        return {
            "input_tokens": sum(call.input_tokens for call in self.calls),
            "output_tokens": sum(call.output_tokens for call in self.calls),
            "cache_read_tokens": sum(call.cache_read_tokens for call in self.calls),
            "cache_write_tokens": sum(call.cache_write_tokens for call in self.calls),
            "estimated_cost_usd_upper_bound": sum(call.cost_upper_bound for call in self.calls),
            "request_count": len(self.calls),
        }


class BudgetLedger:
    def __init__(self, config, state_path=None):
        self.config = config
        self.lock = asyncio.Lock()
        self.spent = self.reserved = 0.0
        self.requests = 0
        self.state_path = state_path
        if state_path and state_path.exists():
            data = json.loads(state_path.read_text())
            self.spent = data["cost_upper_bound"] + data.get("outstanding_reservation", 0)
            self.requests = data["requests"]

    async def reserve(self, input_bytes, max_output):
        config = self.config
        estimate = (
            (input_bytes + 256) * config.input_miss_usd_per_million
            + max_output * config.output_usd_per_million
        ) / 1e6
        async with self.lock:
            if self.requests >= config.max_requests:
                raise BudgetExceeded("request limit reached")
            if self.spent + self.reserved + estimate > config.max_cost_usd:
                raise BudgetExceeded("cost reservation limit reached")
            self.requests += 1
            self.reserved += estimate
            self._save()
        return estimate

    async def settle(self, reservation, usage):
        config = self.config
        if usage is None:
            cost = reservation  # Unknown billed usage consumes the entire conservative reservation.
        else:
            reads = min(usage.cache_read_tokens, usage.input_tokens)
            cost = (
                (usage.input_tokens - reads) * config.input_miss_usd_per_million
                + reads * config.input_hit_usd_per_million
                + usage.output_tokens * config.output_usd_per_million
            ) / 1e6
        async with self.lock:
            self.reserved -= reservation
            self.spent += cost
            self._save()
        return cost

    def _save(self):
        if self.state_path:
            self.state_path.write_text(
                json.dumps(
                    {
                        "requests": self.requests,
                        "cost_upper_bound": self.spent,
                        "outstanding_reservation": self.reserved,
                    },
                    indent=2,
                )
            )


class AuditedModel(WrapperModel):
    def __init__(self, wrapped, ledger, trace, phase):
        super().__init__(wrapped)
        self.ledger, self.trace, self.phase = ledger, trace, phase

    async def request(self, messages, model_settings, model_request_parameters):
        projection = request_projection(messages, model_request_parameters)
        projection["model_settings"] = model_settings or {}
        serialized = json.dumps(
            projection, sort_keys=True, ensure_ascii=False, default=str
        ).encode()
        max_output = (model_settings or {}).get("max_tokens", 1024)

        # DeepSeek documents an upper bound of 1024 tokens per image. Avoid
        # charging the UTF-8/base64 transport bytes as if they were text tokens.
        def images(value):
            if isinstance(value, dict):
                return int(
                    str(value.get("media_type", "")).startswith("image/")
                    and "binary_sha256" in value
                ) + sum(images(v) for v in value.values())
            if isinstance(value, list):
                return sum(images(v) for v in value)
            return 0

        image_count = images(projection)
        reservation = await self.ledger.reserve(len(serialized) + 1024 * image_count, max_output)
        record = CallRecord(self.phase, sha256(serialized).hexdigest(), len(serialized))
        record.input_image_count = image_count
        self.trace.calls.append(record)
        start = perf_counter()
        capture_id = uuid4().hex
        settled = False
        try:
            if self.trace.capture_directory is not None:
                self.trace.capture_directory.mkdir(parents=True, exist_ok=True)
                path = self.trace.capture_directory / (
                    f"{len(self.trace.calls):03d}-{self.phase}-"
                    f"{record.input_digest}-{capture_id}.json"
                )
                path.write_bytes(serialized)
                record.request_file = str(path)
            async with asyncio.timeout(self.ledger.config.timeout_seconds):
                response = await self.wrapped.request(
                    messages, model_settings, model_request_parameters
                )
            usage = response.usage
            record.input_tokens, record.output_tokens = usage.input_tokens, usage.output_tokens
            record.cache_read_tokens, record.cache_write_tokens = (
                usage.cache_read_tokens,
                usage.cache_write_tokens,
            )
            record.response_model, record.response_id = (
                response.model_name or "",
                response.provider_response_id,
            )
            record.cost_upper_bound = await self.ledger.settle(reservation, usage)
            settled = True
            if self.trace.capture_directory is not None:
                response_path = (
                    self.trace.capture_directory
                    / f"{len(self.trace.calls):03d}-{self.phase}-{capture_id}-response.json"
                )
                response_path.write_bytes(ModelMessagesTypeAdapter.dump_json([response]))
                record.response_file = str(response_path)
            return response
        except BaseException as error:
            record.error_type = type(error).__name__
            if not settled:
                record.cost_upper_bound = await self.ledger.settle(reservation, None)
            raise
        finally:
            record.duration_ms = (perf_counter() - start) * 1000


class SourceBootstrap(AbstractCapability):
    """Remove the zero-content framework bootstrap from the request-only view.

    Pydantic AI rejects runs with no user input before before_model_request hooks fire. An empty
    user part passes that validation; this hook withdraws only that empty placeholder, preserving
    raw history. The following Reactive capability supplies the complete benchmark prompt.
    """

    async def before_model_request(self, ctx, request_context):
        messages = []
        for message in request_context.messages:
            if isinstance(message, ModelRequest):
                parts = [
                    part
                    for part in message.parts
                    if not (part.part_kind == "user-prompt" and part.content == "")
                ]
                if parts:
                    messages.append(replace(message, parts=parts))
            else:
                messages.append(message)
        return replace(request_context, messages=messages)


def make_model(config, *, judge=False):
    key = config.judge_key if judge else config.api_key
    base = config.judge_base_url if judge else config.base_url
    if not key:
        raise RuntimeError("model API credential is not configured")
    client = AsyncOpenAI(api_key=key, base_url=base, max_retries=0, timeout=config.timeout_seconds)
    return OpenAIChatModel(
        config.judge_model if judge else config.model_id,
        provider=DeepSeekProvider(openai_client=client),
        profile=OpenAIModelProfile(openai_chat_supports_max_completion_tokens=False),
    )


async def generate(
    model, ledger, trace, phase, prompt=None, *, capabilities=(), max_tokens=None, instructions=None
):
    audited = AuditedModel(model, ledger, trace, phase)
    agent = Agent(
        audited,
        name="public_benchmark_" + phase,
        retries=0,
        **({"instructions": instructions} if instructions is not None else {}),
        capabilities=[
            Caching(messages=False),
            *([SourceBootstrap()] if prompt is None else []),
            *capabilities,
        ],
        model_settings={
            "temperature": 0,
            "max_tokens": max_tokens or ledger.config.max_output_tokens,
            "extra_body": {"thinking": {"type": "disabled"}},
        },
    )
    result = await agent.run(prompt if prompt is not None else "")
    return result.output
