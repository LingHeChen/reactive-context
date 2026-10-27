"""Content-only archive indexing; raw events and their source references remain immutable."""

import inspect
import json
import re
from bisect import bisect_left, bisect_right
from collections.abc import Mapping


def text_value(value):
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value) if value is not None else ""


def project_final(text, projector):
    value = projector(text) if projector is not None else text
    if value is not None and not isinstance(value, str):
        if inspect.iscoroutine(value):
            value.close()
        raise TypeError("final_projector must return text or None synchronously")
    return value


def event_text(event, final_projector=None, *, final=False):
    """Project supported message bodies, excluding instructions, IDs and control chatter."""
    if not isinstance(event, Mapping):
        return ""
    parts = event.get("parts")
    if (
        event.get("kind") in {"request", "response"}
        and isinstance(parts, list)
        and all(isinstance(part, Mapping) for part in parts)
    ):
        calls = [part for part in parts if part.get("part_kind") == "tool-call"]
        if calls:
            if not final:
                return (
                    ""  # successful output calls are marked by the runtime, never guessed by name
                )
            text = "\n".join(text_value(part.get("args")) for part in calls)
            return project_final(text, final_projector) or ""
        result = []
        for part in parts:
            kind = part.get("part_kind")
            if kind == "user-prompt":
                content = part.get("content", "")
                if isinstance(content, str):
                    result.append(content)
                elif isinstance(content, list):
                    result.extend(value for value in content if isinstance(value, str))
            elif kind == "text":
                content = text_value(part.get("content"))
                content = project_final(content, final_projector)
                if content:
                    result.append(content)
            elif kind == "tool-return" and not part.get("tool_name", "").startswith("context_"):
                result.append(part.get("tool_name", "") + ": " + text_value(part.get("content")))
        return "\n".join(result)
    # Generic host-owned events (including historical benchmark states).
    return text_value(dict(event))


def matching_preview(text, words, limit=400):
    """Show the query neighborhood, rather than the beginning of an event envelope."""
    if limit < 1:
        raise ValueError("preview limit must be positive")
    text = text[:65536]
    by_word = {
        word: [m.start() for m in re.finditer(re.escape(word), text, re.I)]
        for word in dict.fromkeys(words)
    }
    candidates = {position for positions in by_word.values() for position in positions[:8]}
    if not candidates:
        return text[:limit]

    def relevance(position):
        score = sum(
            1 / (1 + len(positions))
            for positions in by_word.values()
            if bisect_right(positions, position + limit // 2)
            > bisect_left(positions, position - limit // 2)
        )
        return score, -position

    center = max(candidates, key=relevance)
    start = max(0, center - limit // 4)
    return ("…" if start else "") + text[start : start + limit - int(bool(start))]
