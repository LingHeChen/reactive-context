"""Offline interface isolation, never a substitute for end-to-end memory evaluation."""

import json
import random
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Any

from ..compiler.compiler import Compiler
from ..core.snapshot import digest, freeze, thaw


@dataclass(frozen=True)
class ReplayFixture:
    id: str
    resource_order: tuple[str, ...]
    resources: Any
    dataset_revision: str
    model_config: Any
    scorer_revision: str
    tokenizer_revision: str

    def __post_init__(self):
        object.__setattr__(self, "resource_order", tuple(self.resource_order))
        object.__setattr__(self, "resources", freeze(self.resources))
        object.__setattr__(self, "model_config", freeze(self.model_config))
        if len(set(self.resource_order)) != len(self.resource_order):
            raise ValueError("duplicate logical resource")
        if set(self.resources) != set(self.resource_order):
            raise ValueError("fixture resource/order mismatch")

    @property
    def hash(self):
        return digest(freeze(self))


def canonical_replay(snapshot, fixture, policy, *, release="none"):
    """Normalize one variable at a time at the IR boundary, using the SAME compiler.

    Payload mismatches fail, never replace actual payloads with expected answers.
    Exact messages are for interface isolation only. Production wrappers remain intact.
    """
    if release not in {"none", "resource", "revision", "order"}:
        raise ValueError("unknown intervention")
    if {s.part_id for s in snapshot.segments} != set(fixture.resource_order):
        raise ValueError("replay topology mismatch")
    segments = []
    for segment in snapshot.segments:
        expected = fixture.resources[segment.part_id]
        if digest(segment.payload) != expected["payload_hash"]:
            raise ValueError("replay payload mismatch")
        segments.append(
            replace(
                segment,
                id=f"replay:{segment.part_id}",
                resource_id=segment.resource_id
                if release == "resource"
                else expected["resource_id"],
                source_revision=segment.source_revision
                if release == "revision"
                else expected["revision"],
            )
        )
    pinned_policy = replace(
        policy, part_order=(() if release == "order" else fixture.resource_order)
    )
    normalized = replace(
        snapshot, segments=tuple(segments), compiler_config_revision=pinned_policy.fingerprint
    )
    return Compiler(pinned_policy).compile(normalized)


def capture(compiled, *, tokenizer=None):
    messages = [{"role": m.role, "content": m.content} for m in compiled.messages]
    body = json.dumps(
        {"messages": messages, "tools": thaw(compiled.provider_tools)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "canonical_messages": messages,
        "tools": thaw(compiled.provider_tools),
        "serialized_utf8": body,
        "bytes_hash": sha256(body.encode()).hexdigest(),
        "token_ids": list(tokenizer(body)) if tokenizer else None,
        "tokenization_status": "measured" if tokenizer else "NOT_MEASURED",
        "manifest": thaw(compiled.manifest),
    }


def write_replay(directory, fixture, variants, *, tokenizer=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    reports = {name: capture(value, tokenizer=tokenizer) for name, value in variants.items()}
    for name, report in reports.items():
        (directory / f"{name}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    summary = {
        "fixture": thaw(freeze(fixture)),
        "fixture_hash": fixture.hash,
        "message_hashes": {name: report["bytes_hash"] for name, report in reports.items()},
        "identical": len({r["bytes_hash"] for r in reports.values()}) == 1,
        "model_calls": 0,
        "task_quality": "NOT_MEASURED",
        "verdict": "inconclusive",
    }
    (directory / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def paired_bootstrap(baseline, candidate, *, seed=83, resamples=2000):
    """Inputs are independent episode aggregates, NOT stages within one episode."""
    if not baseline or len(baseline) != len(candidate) or resamples < 100:
        raise ValueError("paired episode measurements required")
    delta = [b - a for a, b in zip(baseline, candidate, strict=True)]
    rng = random.Random(seed)
    samples = sorted(sum(rng.choices(delta, k=len(delta))) / len(delta) for _ in range(resamples))
    return {
        "episodes": len(delta),
        "difference": sum(delta) / len(delta),
        "ci95": [samples[int(0.025 * resamples)], samples[int(0.975 * resamples)]],
        "verdict": "inconclusive",
        "unit": "independent_episode",
    }
