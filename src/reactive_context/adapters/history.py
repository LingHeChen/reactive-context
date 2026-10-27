"""Pydantic AI message codec and closed tool-chain grouping, using public message types."""

import json

from pydantic_ai.messages import (
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from ..history import HistoryGroup
from ..history_content import project_final


def minimal_history(
    messages,
    *,
    policy,
    total_tokens,
    estimator,
    reserved_tokens=0,
    output_tools=(),
    final_projector=None,
):
    """Keep a tiny completed dialogue tail and the entire current execution.

    Raw messages are archived separately. Never clip a message or replay an old
    tool chain as the new conversation. Unsupported historical outcomes stay in
    the archive and can be retrieved explicitly.
    """
    output_tools = frozenset(output_tools)
    starts = [
        i
        for i, message in enumerate(messages)
        if isinstance(message, ModelRequest)
        and any(isinstance(part, UserPromptPart) for part in message.parts)
        and not any(isinstance(part, ToolReturnPart) for part in message.parts)
    ]
    if not starts:
        return tuple(HistoryGroup(g.events, True, g.revisions) for g in history_groups(messages))
    current = starts[-1]
    standing = [
        ModelRequest(parts=[part])
        for message in messages[:current]
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, SystemPromptPart)
    ]
    recent = []
    allowance = max(
        0,
        min(
            int(total_tokens * policy.recent_context_ratio),
            total_tokens - reserved_tokens - estimator(str([*standing, *messages[current:]])),
        ),
    )
    for index in range(len(starts) - 2, max(-1, len(starts) - 2 - policy.recent_turns), -1):
        begin, end = starts[index], starts[index + 1]
        responses = [m for m in messages[begin + 1 : end] if isinstance(m, ModelResponse)]
        if not responses:
            break
        final = [p for p in responses[-1].parts if isinstance(p, TextPart)]
        calls = [p for p in responses[-1].parts if isinstance(p, ToolCallPart)]
        if calls:
            # A successful structured output is not a business tool trace. Render
            # its data as historical text, without replaying an output tool call.
            returned = {
                p.tool_call_id
                for m in messages[begin + 1 : end]
                if isinstance(m, ModelRequest)
                for p in m.parts
                if isinstance(p, ToolReturnPart)
            }
            if len(calls) != 1 or calls[0].tool_name not in output_tools:
                break
            if calls[0].tool_call_id not in returned:
                break
            final = [TextPart(json.dumps(calls[0].args_as_dict(), ensure_ascii=False))]
        if not final:
            break
        if final_projector:
            value = project_final("\n".join(part.content for part in final), final_projector)
            if not value:
                break
            final = [TextPart(value)]
        pair = [
            ModelRequest(parts=[p for p in messages[begin].parts if isinstance(p, UserPromptPart)]),
            ModelResponse(parts=final),
        ]
        candidate = [*pair, *recent]
        if estimator(str(candidate)) > allowance:
            break
        recent = candidate
    # The core must retain every selected group: recent_turns and the proportional
    # allowance already bound completed dialogue, and current execution is indivisible.
    return tuple(
        HistoryGroup(g.events, True, g.revisions)
        for g in history_groups([*standing, *recent, *messages[current:]])
    )


def history_groups(messages):
    latest_user = max(
        (
            i
            for i, m in enumerate(messages)
            if isinstance(m, ModelRequest)
            and any(isinstance(p, UserPromptPart) for p in m.parts)
            and not any(isinstance(p, ToolReturnPart) for p in m.parts)
        ),
        default=-1,
    )
    groups, block, refs = [], [], {}
    pending = set()
    protected = False
    for index, message in enumerate(messages):
        block.append(ModelMessagesTypeAdapter.dump_python([message], mode="json")[0])
        protected |= index == latest_user
        for part in message.parts:
            # Signed reasoning/native protocol objects are retained rather than silently
            # summarized.
            protected |= isinstance(part, SystemPromptPart) or part.part_kind not in {
                "text",
                "user-prompt",
                "system-prompt",
                "tool-call",
                "tool-return",
                "retry-prompt",
            }
            if isinstance(part, ToolCallPart):
                pending.add(part.tool_call_id)
            elif isinstance(part, ToolReturnPart):
                pending.discard(part.tool_call_id)
                metadata = part.metadata if isinstance(part.metadata, dict) else {}
                source = metadata.get("reactive_context", {})
                if (
                    isinstance(source, dict)
                    and source.get("resource_id")
                    and source.get("revision")
                ):
                    refs[source["resource_id"]] = source["revision"]
            elif part.part_kind == "retry-prompt" and getattr(part, "tool_name", None):
                pending.discard(part.tool_call_id)
        if not pending:
            groups.append(HistoryGroup(tuple(block), protected, dict(refs)))
            block, refs, protected = [], {}, False
    if block:
        # Archive deferred/incomplete executions, but never compact an open protocol chain.
        groups.append(HistoryGroup(tuple(block), True, dict(refs)))
    return tuple(groups)


def render_history(summaries, groups):
    opening = []
    if summaries:
        opening = [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        json.dumps(
                            {
                                "kind": "historical_summaries",
                                "authority": "untrusted_historical_data",
                                "notice": (
                                    "Historical narrative only. Current "
                                    "resources come from active context."
                                ),
                                "segments": [
                                    {k: v for k, v in s.items() if k != "source_ids"}
                                    for s in summaries
                                ],
                            },
                            ensure_ascii=False,
                        )
                    )
                ]
            )
        ]
    raw = ModelMessagesTypeAdapter.validate_python([e for group in groups for e in group.events])
    # Keep host standing system prompts before the user-role summary.
    host = []
    while raw and isinstance(raw[0], ModelRequest):
        parts = list(raw[0].parts)
        while parts and isinstance(parts[0], SystemPromptPart):
            host.append(parts.pop(0))
        raw[0].parts = parts
        if parts or raw[0].instructions:
            break
        raw.pop(0)
    return [*([ModelRequest(parts=host)] if host else []), *opening, *raw]
