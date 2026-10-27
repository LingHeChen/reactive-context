"""OpenRouter Decisions adapter for Jev; no chat protocol or generated objective text.

Wire contract: https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-request
"""

import json
import math
import os
from dataclasses import asdict
from typing import Annotated, Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ..core.snapshot import thaw
from ..errors import ContextError


class JevError(ContextError):
    code = "jev_request_failed"


class JevProtocolError(ContextError):
    code = "jev_protocol_invalid"


class JevQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    type: Literal["noul", "choice", "score"]
    instructions: str = Field(min_length=1, max_length=16000)
    criteria: dict[str, str] | list[str] | None = None

    @model_validator(mode="after")
    def valid_criteria(self):
        if self.type == "choice" and not (
            isinstance(self.criteria, dict) and 1 <= len(self.criteria) <= 255
        ):
            raise ValueError("choice requires named options")
        if self.type == "score" and not (
            isinstance(self.criteria, list) and 2 <= len(self.criteria) <= 10
        ):
            raise ValueError("score requires 2 to 10 ordered criteria")
        if (
            self.type == "noul"
            and self.criteria is not None
            and not (isinstance(self.criteria, dict) and set(self.criteria) <= {"true", "false"})
        ):
            raise ValueError("noul criteria use true/false keys")
        return self


Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False, strict=True)]


class NoulAnswer(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: Literal["noul"]
    noul: Probability


class ChoiceAnswer(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: Literal["choice"]
    choice: str
    probabilities: dict[str, Probability] | None = None
    confidence: Probability | None = None


class ScoreAnswer(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: Literal["score"]
    score: float = Field(ge=0, allow_inf_nan=False, strict=True)
    probabilities: dict[str, Probability] | None = None
    confidence: Probability | None = None


Answer = Annotated[NoulAnswer | ChoiceAnswer | ScoreAnswer, Field(discriminator="type")]


class JevResult(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str = Field(min_length=1)
    answers: dict[str, Answer]
    usage: dict[str, Any] = Field(default_factory=dict)
    id: str | None = None


class JevAdapter:
    def __init__(
        self,
        *,
        api_key=None,
        model="typesafe/jev-1.14",
        base_url="https://openrouter.ai/api",
        client=None,
        timeout=30,
        decision_threshold=0.8,
        max_request_bytes=2 * 1024 * 1024,
        max_response_bytes=1024 * 1024,
        on_usage=None,
    ):
        self._key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not self._key:
            raise ValueError("configure OPENROUTER_API_KEY for Jev")
        if not 0 <= decision_threshold <= 1 or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("invalid Jev threshold or timeout")
        if max_request_bytes < 1 or max_response_bytes < 1:
            raise ValueError("invalid Jev payload limits")
        self.model, self.base_url, self.timeout = model, base_url.rstrip("/"), timeout
        self.threshold = decision_threshold
        self.max_request_bytes, self.max_response_bytes = max_request_bytes, max_response_bytes
        self._owned_client = client is None
        self.client = client or httpx.AsyncClient(timeout=timeout)
        self.on_usage = on_usage
        self.last_result = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def close(self):
        if self._owned_client:
            await self.client.aclose()

    async def decide(self, state, questions):
        if not isinstance(state, (str, dict, list)) or not questions or len(questions) > 256:
            raise ValueError("provide JSON state and 1 to 256 named questions")
        parsed = {key: JevQuestion.model_validate(q) for key, q in questions.items()}
        if any(not isinstance(key, str) or not key for key in parsed):
            raise ValueError("question IDs must be nonempty strings")
        body = {
            "model": self.model,
            "state": state,
            "questions": {k: q.model_dump(exclude_none=True) for k, q in parsed.items()},
        }
        serialized = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
        if len(serialized) > self.max_request_bytes:
            raise JevError(detail="decision input exceeds configured byte budget")
        try:
            async with self.client.stream(
                "POST",
                self.base_url + "/alpha/decisions",
                content=serialized,
                headers={
                    "Authorization": "Bearer " + self._key,
                    "Content-Type": "application/json",
                },
                timeout=self.timeout,
            ) as response:
                if response.status_code >= 400:
                    raise JevError(detail=f"HTTP {response.status_code}")
                chunks, length = [], 0
                async for chunk in response.aiter_bytes():
                    length += len(chunk)
                    if length > self.max_response_bytes:
                        raise JevProtocolError(detail="decision response exceeds byte budget")
                    chunks.append(chunk)
            data = json.loads(b"".join(chunks))
            result = JevResult.model_validate(data)
            if set(result.answers) != set(parsed):
                raise JevProtocolError(detail="decision question IDs do not match")
            for key, answer in result.answers.items():
                question = parsed[key]
                if answer.type != question.type:
                    raise JevProtocolError(detail="decision type does not match question")
                if isinstance(answer, ChoiceAnswer):
                    if not isinstance(question.criteria, dict):
                        raise JevProtocolError(detail="invalid choice criteria")
                    if answer.choice not in question.criteria:
                        raise JevProtocolError(detail="unknown decision option")
                    if answer.probabilities is not None and set(answer.probabilities) != set(
                        question.criteria
                    ):
                        raise JevProtocolError(detail="decision option probabilities do not match")
                if isinstance(answer, ScoreAnswer):
                    if (
                        not isinstance(question.criteria, list)
                        or answer.score > len(question.criteria) - 1
                    ):
                        raise JevProtocolError(detail="decision score exceeds rubric")
        except httpx.HTTPError:
            raise JevError(detail="decision transport failed") from None
        except (ValidationError, ValueError, TypeError, KeyError):
            raise JevProtocolError(detail="malformed decision response") from None
        self.last_result = result
        if self.on_usage is not None:
            from ..core.scheduler import invoke

            await invoke(self.on_usage, dict(result.usage))
        return result

    async def decide_context(self, manager, scope, questions):
        """Send only the authorized current working set; never pull the entire archive."""
        compiled = manager.compile(await manager.next_context(scope))
        return await self.decide(
            {
                "context": [asdict(m) for m in compiled.provider_messages],
                "revision_vector": dict(compiled.revision_vector),
                "tools": thaw(compiled.provider_tools),
            },
            questions,
        )

    async def should_update_objective(self, request):
        result = await self.decide(
            {
                "current_objective": request.current,
                "recent_events": list(request.events),
                "reason": request.reason,
                "update_conditions": list(getattr(request, "conditions", ())),
            },
            {
                "update_needed": {
                    "type": "noul",
                    "instructions": "Do recent events materially change the current goal, "
                    "next steps or completion status? Do not request an update for "
                    "unrelated chatter or unchanged state.",
                }
            },
        )
        answer = result.answers["update_needed"]
        assert isinstance(answer, NoulAnswer)
        return answer.noul >= self.threshold

    async def validate_summary(self, events, draft):
        result = await self.decide(
            {"sources": list(events), "summary": draft.model_dump()},
            {
                "supported": {
                    "type": "noul",
                    "instructions": "Are the summary's factual claims supported by the sources, "
                    "without reversing updates or dropping explicit constraints?",
                },
            },
        )
        answer = result.answers["supported"]
        assert isinstance(answer, NoulAnswer)
        return answer.noul >= self.threshold
