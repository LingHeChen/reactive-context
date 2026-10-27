"""Provider-neutral cache configuration and conservative, metadata-only prefix diagnostics."""

import json
from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha256
from typing import Literal


@dataclass(frozen=True)
class PromptCachePolicy:
    retention: bool | Literal["5m", "30m", "1h"] = True
    messages: bool = True
    prefix_breakpoint: bool = True

    def __post_init__(self):
        if type(self.retention) is not bool and self.retention not in {"5m", "30m", "1h"}:
            raise ValueError("unsupported prompt cache retention")


def stable_json(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=repr
    )


class PrefixMonitor:
    """Compare complete logical prefix blocks, never claim a provider cache hit.

    Only hashes and lengths are retained. This deliberately undercounts reuse inside changed
    blocks, and cannot model a provider's tokenizer, serialization, routing or retention.
    """

    def __init__(self, max_scopes=1000):
        self._previous = OrderedDict()
        self.max_scopes = max_scopes

    def observe(self, key, blocks):
        current = tuple(
            (name, sha256(text.encode()).hexdigest(), len(text.encode())) for name, text in blocks
        )
        previous = self._previous.get(key, ())
        reused_bytes = reused_blocks = 0
        for before, after in zip(previous, current, strict=False):
            if before != after:
                break
            reused_bytes += after[2]
            reused_blocks += 1
        changed = (
            current[reused_blocks][0]
            if reused_blocks < len(current)
            else previous[reused_blocks][0]
            if reused_blocks < len(previous)
            else ""
        )
        self._previous[key] = current
        self._previous.move_to_end(key)
        if len(self._previous) > self.max_scopes:
            self._previous.popitem(last=False)
        return {
            "prefix_hash": sha256(stable_json(current).encode()).hexdigest(),
            "prefix_bytes": sum(item[2] for item in current),
            "prefix_blocks": len(current),
            "reused_prefix_bytes": reused_bytes,
            "reused_prefix_blocks": reused_blocks,
            "reason": "first_prefix"
            if not previous
            else f"changed:{changed}"
            if changed
            else "prefix_unchanged",
        }


DEFAULT_PROMPT_CACHE_POLICY = PromptCachePolicy()
