"""Bounded finish within the original run budget; preserve partial observations on error."""

import re
from copy import deepcopy
from dataclasses import replace
from typing import Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    InstructionPart,
    ModelRequest,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from ..evidence.session import ResearchStopped


def contains_unexecuted_tool_markup(text):
    """Known tool protocol markers emitted as text, not actual SDK tool calls."""
    return bool(
        re.search(r"<[｜|]*DSML[｜|]*\s*(?:calls|invoke)\b|<tool_call\b|<function_call\b", text)
    )


class ResearchRecoveryCapability(AbstractCapability[Any]):
    def __init__(self, session=None, *, max_model_requests=30, finish_requests=1):
        if not 0 < finish_requests < max_model_requests:
            raise ValueError("reserve finish requests within the existing request limit")
        self.session = session
        if session is not None:
            session.finish_handler_attached = True
        self.max_model_requests = max_model_requests
        self.finish_requests = finish_requests
        self.requests = self.finishes = 0
        self.partial_messages = []

    def current_messages(self):
        starts = [
            i
            for i, message in enumerate(self.partial_messages)
            if isinstance(message, ModelRequest)
            and any(isinstance(p, UserPromptPart) for p in message.parts)
            and not any(isinstance(p, ToolReturnPart) for p in message.parts)
        ]
        return deepcopy(self.partial_messages[starts[-1] :]) if starts else []

    async def on_run_error(self, ctx, *, error):
        if len(ctx.messages) >= len(self.partial_messages):
            self.partial_messages = deepcopy(list(ctx.messages))
        raise error

    async def before_model_request(self, ctx, request_context):
        self.partial_messages = deepcopy(list(request_context.messages))
        if self.session is None:
            return request_context
        if self.requests >= self.max_model_requests - self.finish_requests:
            self.session.begin_finish("model_request_reserve")
        if self.session.finalizing:
            if self.finishes >= self.finish_requests:
                raise ResearchStopped("finish_attempt_limit")
            self.finishes += 1
            parameters = request_context.model_request_parameters
            instructions = [
                *(parameters.instruction_parts or []),
                InstructionPart(
                    content="Optional retrieval has ended. Produce your final response now using "
                    "available evidence. Explicitly mark unresolved dependencies as unknown; "
                    "do not "
                    "invent prior decisions or claim an incomplete task succeeded. No further "
                    "search/read/business tools are available in this finishing request."
                ),
            ]
            request_context = replace(
                request_context,
                model_request_parameters=replace(
                    parameters,
                    function_tools=[],
                    instruction_parts=instructions,
                ),
            )
        self.requests += 1
        return request_context

    async def after_model_request(self, ctx, *, request_context, response):
        if self.session is not None and self.session.finalizing:
            self.partial_messages = deepcopy([*request_context.messages, response])
            if any(
                isinstance(p, TextPart) and contains_unexecuted_tool_markup(p.content)
                for p in response.parts
            ):
                raise ResearchStopped("unexecuted_tool_markup")
            allowed = {t.name for t in request_context.model_request_parameters.output_tools}
            if any(
                isinstance(p, ToolCallPart) and p.tool_name not in allowed for p in response.parts
            ):
                # A model ignoring the tool-free finish must not execute hidden tools.
                raise ResearchStopped("tool_call_during_finish")
        return response
